#!/usr/bin/env python3
# src/simple_or_agent/react_agent.py
# Thinker→Operator→Validator ReACT loop with simple tool handling.
# This exists to execute tools in steps and converge to a final answer.
# RELEVANT FILES: src/simple_or_agent/simple_agent.py,src/simple_or_agent/next_agent.py,src/simple_or_agent/orchestrator.py

from __future__ import annotations
from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Optional, List, Tuple, Callable

from .openrouter_client import OpenRouterClient, OpenRouterError
from .simple_agent import SimpleAgent, ToolSpec, AskResult

# ---- Core data structures ----

@dataclass
class ActionSpec:
    type: str  # "tool" | "finish"
    name: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    say: Optional[str] = None


@dataclass
class ReActStep:
    thought: str
    action: ActionSpec
    observation: str


# ---- Thinker, Operator, Validator ----

class ThinkerAgent:
    """LLM that produces the next Thought and a plain-text Action (no execution)."""

    DEFAULT_SYSTEM = (
        "You are Thinker. Reason step-by-step and choose the next action.\n"
        "Output EXACTLY this format (no extra text):\n"
        "Thought: <what you will do next and why>\n"
        "Action: <a short, specific instruction for the operator>\n"
        "Rules:\n"
        "- Never include Observation.\n"
        "- If the task is solved, do not plan an action. Output 'Final Answer: <answer>'.\n"
        "- The Action is natural language, e.g.: 'Search for \"clojure in game development\"', 'Open https://example.com', 'Summarize findings so far'.\n"
    )

    def __init__(
        self,
        client: OpenRouterClient,
        model: Optional[str],
        temperature: float,
        reasoning_effort: Optional[str],
        keep_history: bool = True,
    ) -> None:
        self.agent = SimpleAgent(
            client,
            model=model,
            system_prompt=self.DEFAULT_SYSTEM,
            keep_history=False,
            temperature=temperature,
            max_rounds=4,
            max_tool_iters=0,
            response_format=None,
            reasoning_effort=reasoning_effort,
            parallel_tool_calls=False,
            tool_choice="none",
            inline_tools=False,
        )

    def propose(self, goal: str, tools_catalog: str, history_text: str) -> Tuple[str, ActionSpec, AskResult]:
        prompt = (
            f"Goal:\n{goal}\n\n"
            f"Tools (name, description, JSON params):\n{tools_catalog or '(none)'}\n\n"
            f"History so far (Thought/Action/Observation per step):\n{history_text or '(none)'}\n\n"
            "Propose only the next Thought and Action per the required format."
        )
        res = self.agent.ask(prompt)
        thought, action = self._parse_thought_and_action(res.content)
        return thought, action, res

    @staticmethod
    def _parse_thought_and_action(text: str) -> Tuple[str, ActionSpec]:
        # Final Answer short-circuit
        mfa = re.search(r"^\s*Final\s+Answer:\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
        if mfa:
            ans = (mfa.group(1) or "").strip()
            return "", ActionSpec(type="finish", name=None, args=None, say=ans)

        # Extract plain Thought and Action lines
        thought = ""
        action_txt = ""
        mt = re.search(r"Thought:\s*(.*?)(?:\n\s*Action:|$)", text, re.DOTALL | re.IGNORECASE)
        if mt:
            thought = (mt.group(1) or "").strip()
        ma = re.search(r"Action:\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
        if ma:
            action_txt = (ma.group(1) or "").strip()

        # Treat action text as operator instruction; operator will decide tool or not
        return thought, ActionSpec(type="plan", name=None, args=None, say=action_txt)


class OperatorAgent:
    """Executes actions using registered ToolSpec handlers and returns Observation text."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def add_tool(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def remove_tool(self, name: str) -> None:
        self._tools.pop(name, None)

    def tool_catalog(self) -> str:
        if not self._tools:
            return ""
        parts: List[str] = []
        for t in self._tools.values():
            try:
                params = json.dumps(t.parameters, ensure_ascii=False)
            except Exception:
                params = "{}"
            parts.append(f"- {t.name}: {t.description}\n  params: {params}")
        return "\n".join(parts)

    def _infer_tool_from_text(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Infer a tool call from a natural-language Action.

        Simple and predictable:
        - URL present -> fetch_page {url}
        - Contains 'search'/'look up'/'find' -> web_search {query}
        """
        s = (text or "").strip()
        if not s:
            return None
        # URL → fetch_page
        m = re.search(r"https?://\S+", s)
        if m and "fetch_page" in self._tools:
            return "fetch_page", {"url": m.group(0)}
        # search query in quotes → web_search
        mq = re.search(r"(?:search|look\s*up|find)\s+(?:for\s+)?\"([^\"]+)\"", s, re.IGNORECASE)
        if mq and "web_search" in self._tools:
            return "web_search", {"query": mq.group(1)}
        # search ... for <tail>
        mt = re.search(r"(?:search|look\s*up|find)\s+(?:for\s+)?(.+)$", s, re.IGNORECASE)
        if mt and "web_search" in self._tools:
            return "web_search", {"query": mt.group(1).strip()}
        return None

    def execute(self, action: ActionSpec, progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None, step_idx: Optional[int] = None) -> str:
        # Finish handling
        if (action.type or "").lower() == "finish":
            return action.say or "finish"

        # Tool with explicit name/args
        if (action.name or None) is not None:
            tool = self._tools.get(action.name)
            if not tool:
                names = ", ".join(sorted(self._tools.keys()))
                return f"error: unknown_tool {action.name}; valid: {names}"
            try:
                # Emit tool progress before calling the handler
                if progress_cb:
                    try:
                        progress_cb("tool", {"step": step_idx, "name": action.name, "args": action.args or {}})
                    except Exception:
                        pass
                out = tool.handler(action.args or {})
                return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            except Exception as e:
                return f"error: {e}"

        # Natural-language action: infer tool or provide a helpful observation
        inferred = self._infer_tool_from_text(action.say or "")
        if inferred:
            name, args = inferred
            try:
                if progress_cb:
                    try:
                        progress_cb("tool", {"step": step_idx, "name": name, "args": args or {}})
                    except Exception:
                        pass
                out = self._tools[name].handler(args)
                return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            except Exception as e:
                return f"error: tool_failed {name}: {e}"

        # No tool chosen; return a concise note so the Thinker can adjust
        names = ", ".join(sorted(self._tools.keys()))
        return f"note: no_tool_selected; available: {names}; action='{(action.say or '').strip()}'"


class ValidatorAgent:
    """LLM that decides whether to continue or provide the Final Answer."""

    DEFAULT_SYSTEM = (
        "You are Validator. Given the goal and the latest step (Thought/Action/Observation), decide if the task is solved.\n"
        "Respond with exactly one of:\n"
        "- 'Decision: continue' (optionally followed by one short feedback line), or\n"
        "- 'Final Answer: <complete, concise final answer>'\n"
        "Do not invent new observations or call tools."
    )

    def __init__(
        self,
        client: OpenRouterClient,
        model: Optional[str],
        temperature: float,
        reasoning_effort: Optional[str],
        keep_history: bool = True,
    ) -> None:
        self.agent = SimpleAgent(
            client,
            model=model,
            system_prompt=self.DEFAULT_SYSTEM,
            keep_history=False,
            temperature=temperature,
            max_rounds=4,
            max_tool_iters=0,
            response_format=None,
            reasoning_effort=reasoning_effort,
            parallel_tool_calls=False,
            tool_choice="none",
            inline_tools=False,
        )

    def judge(self, goal: str, last_step_text: str, transcript: str) -> Tuple[bool, str, AskResult]:
        prompt = (
            f"Goal:\n{goal}\n\n"
            f"Transcript so far:\n{transcript}\n\n"
            f"Latest step:\n{last_step_text}\n\n"
            "Return either 'Decision: continue' or 'Final Answer: ...'"
        )
        res = self.agent.ask(prompt)
        text = res.content.strip()
        # Parse decision
        m_final = re.search(r"^\s*Final\s+Answer:\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
        if m_final:
            return True, m_final.group(1).strip(), res
        m_cont = re.search(r"^\s*Decision:\s*continue\b", text, re.IGNORECASE)
        if m_cont:
            return False, "", res
        # Fallback: if neither matched, assume continue with feedback
        return False, "", res

    def finalize(self, goal: str, transcript: str) -> Tuple[str, AskResult]:
        """Ask the model to produce a Final Answer explicitly.

        This is used when max steps are reached to still return a best-effort answer.
        """
        prompt = (
            f"Goal:\n{goal}\n\n"
            f"Transcript so far:\n{transcript}\n\n"
            "Provide the Final Answer now. Respond with exactly one line starting with 'Final Answer:'."
        )
        res = self.agent.ask(prompt)
        text = (res.content or "").strip()
        m_final = re.search(r"^\s*Final\s+Answer:\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
        if m_final:
            return m_final.group(1).strip(), res
        # Fallback: return raw content if format not respected
        return text, res


# ---- Orchestrator ----

class ReActAgent:
    """Implements a simple ReACT loop using Thinker, Operator, and Validator agents."""

    def __init__(
        self,
        client: OpenRouterClient,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,  # ignored for ReACT; kept for API compatibility
        keep_history: bool = True,
        temperature: float = 0.1,
        max_rounds: int = 6,
        max_tool_iters: int = 3,  # ignored here; kept for API compatibility
        response_format: Optional[Dict[str, Any]] = None,  # unused here
        reasoning_effort: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,  # unused here
        tool_choice: Optional[Any] = "auto",  # unused here
        planner_system: Optional[str] = None,  # unused here
    ) -> None:
        # Core sub-agents
        self.thinker = ThinkerAgent(client, model=model, temperature=temperature, reasoning_effort=reasoning_effort, keep_history=keep_history)
        self.operator = OperatorAgent()
        self.validator = ValidatorAgent(client, model=model, temperature=temperature, reasoning_effort=reasoning_effort, keep_history=keep_history)

        # Controls
        self.max_steps = max(1, int(max_rounds))

    def add_tool(self, tool: ToolSpec) -> None:
        self.operator.add_tool(tool)

    def remove_tool(self, name: str) -> None:
        self.operator.remove_tool(name)

    def reset(self) -> None:
        # Reset Thinker and Validator memories
        self.thinker.agent.reset()
        self.validator.agent.reset()

    def ask(self, prompt: str, mode: Optional[str] = None, progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> AskResult:
        if not prompt:
            raise OpenRouterError("Empty prompt")

        tools_catalog = self.operator.tool_catalog()
        steps: List[ReActStep] = []
        # transcript_lines feeds the model context; do not alter with display-only tweaks
        transcript_lines: List[str] = []
        # display_transcript_lines is for user-visible output; safe to include draft substitutes
        display_transcript_lines: List[str] = []
        messages_accum: List[Dict[str, Any]] = []
        latest_reasoning: Optional[str] = None
        usage: Optional[Dict[str, Any]] = None

        if progress_cb:
            try:
                progress_cb("start", {"goal": prompt})
            except Exception:
                pass

        for step_idx in range(1, self.max_steps + 1):
            history_text = "\n\n".join(transcript_lines)
            thought, action, thinker_res = self.thinker.propose(prompt, tools_catalog, history_text)
            if progress_cb:
                try:
                    progress_cb("think_raw", {"step": step_idx, "raw": thinker_res.content})
                except Exception:
                    pass
            latest_reasoning = thinker_res.reasoning or latest_reasoning
            usage = thinker_res.usage or usage
            messages_accum.extend(thinker_res.messages or [])

            action_desc = action.say or (action.name or action.type)
            # Model transcript: use raw thought
            transcript_lines.append(f"Thought: {thought}")
            transcript_lines.append(f"Action: {action_desc}")
            # Display transcript: if Thought is empty and finish.say exists, show draft as Thought
            disp_thought = (thought or "").strip()
            if not disp_thought and (action.type or "").strip().lower() == "finish" and (action.say or "").strip():
                disp_thought = (action.say or "").strip()
            display_transcript_lines.append(f"Thought: {disp_thought}")
            display_transcript_lines.append(f"Action: {action_desc}")

            if progress_cb:
                try:
                    # If model omitted Thought but provided a finish draft in say,
                    # surface that draft as the displayed thought for readability.
                    disp_thought = thought.strip()
                    if not disp_thought and (action.type or "").lower() == "finish" and (action.say or "").strip():
                        disp_thought = (action.say or "").strip()
                    progress_cb(
                        "think",
                        {
                            "step": step_idx,
                            "thought": disp_thought,
                            "action": {"type": action.type, "name": action.name, "say": action.say},
                        },
                    )
                except Exception:
                    pass

            # Execute (Operator decides whether to call a tool)
            observation = self.operator.execute(action, progress_cb=progress_cb, step_idx=step_idx)
            if progress_cb:
                try:
                    progress_cb("operator_raw", {"step": step_idx, "raw": observation})
                except Exception:
                    pass
            transcript_lines.append(f"Observation: {observation}")
            display_transcript_lines.append(f"Observation: {observation}")
            steps.append(ReActStep(thought=thought, action=action, observation=observation))

            if progress_cb:
                try:
                    progress_cb(
                        "observe",
                        {
                            "step": step_idx,
                            "observation": observation,
                            "action": {"type": action.type, "name": action.name, "say": action.say},
                        },
                    )
                except Exception:
                    pass

            last_step_text = "\n".join(transcript_lines[-3:])
            done, final_answer, val_res = self.validator.judge(prompt, last_step_text, "\n".join(transcript_lines))
            messages_accum.extend(val_res.messages or [])
            usage = val_res.usage or usage
            if progress_cb:
                try:
                    progress_cb("judge", {"step": step_idx, "decision": "final" if done else "continue", "text": (val_res.content or "").strip(), "final": final_answer if done else ""})
                except Exception:
                    pass
            if progress_cb:
                try:
                    progress_cb("judge_raw", {"step": step_idx, "raw": (val_res.content or "")})
                except Exception:
                    pass
            if done:
                content = "\n".join(display_transcript_lines + ["", f"Final Answer: {final_answer}"])
                return AskResult(content=content, reasoning=latest_reasoning, usage=usage, messages=messages_accum)

        # Max steps reached: ask Validator to finalize anyway and attach a note.
        final_txt, fin_res = self.validator.finalize(prompt, "\n".join(transcript_lines))
        messages_accum.extend(fin_res.messages or [])
        usage = fin_res.usage or usage
        if progress_cb:
            try:
                progress_cb("judge", {"step": self.max_steps, "decision": "final", "text": (fin_res.content or "").strip(), "final": final_txt, "limited_by_max_steps": True})
            except Exception:
                pass
        content = "\n".join(display_transcript_lines + ["", f"Final Answer: {final_txt}", "", "Note: max steps reached; answer may be incomplete."])
        return AskResult(content=content, reasoning=latest_reasoning, usage=usage, messages=messages_accum)


__all__ = [
    "ToolSpec",
    "AskResult",
    "ActionSpec",
    "ReActStep",
    "ThinkerAgent",
    "OperatorAgent",
    "ValidatorAgent",
    "ReActAgent",
]
