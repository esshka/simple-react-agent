from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Optional, List, Tuple

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
    """LLM that produces the next Thought and an Action plan (no execution)."""

    DEFAULT_SYSTEM = (
        "You are Thinker. Plan one next action using tools when helpful.\n"
        "Follow this exact format:\n"
        "Thought: <your step-by-step reasoning for what to do next>\n"
        "Action: ```json\n{\n  \"type\": \"tool|finish\",\n  \"name\": \"<tool_name_if_type_tool>\",\n  \"args\": { /* JSON args if tool */ },\n  \"say\": \"very short description of the action\"\n}\n```\n"
        "- Never include Observation.\n- Never execute tools.\n- If you believe the task is solved, use type=finish and put the draft answer in say."
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
            keep_history=keep_history,
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
        thought = ""
        # Thought: capture until Action:
        m = re.search(r"Thought:\s*(.*?)(?:\n\s*Action:|$)", text, re.DOTALL | re.IGNORECASE)
        if m:
            thought = m.group(1).strip()
        # Action JSON block
        action_obj: Dict[str, Any] = {}
        jm = re.search(r"Action:\s*```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if jm:
            raw = jm.group(1)
            try:
                action_obj = json.loads(raw)
            except Exception:
                action_obj = {}
        else:
            # Fallback: try to find a JSON object anywhere
            jm2 = re.search(r"\{[\s\S]*\}", text)
            if jm2:
                try:
                    action_obj = json.loads(jm2.group(0))
                except Exception:
                    action_obj = {}
            else:
                # Very weak fallback: try to read a simple 'finish' directive
                if re.search(r"\bfinish\b", text, re.IGNORECASE):
                    action_obj = {"type": "finish"}
        a_type = str(action_obj.get("type", "")).strip().lower() if isinstance(action_obj, dict) else ""
        name = action_obj.get("name") if isinstance(action_obj, dict) else None
        args = action_obj.get("args") if isinstance(action_obj, dict) else None
        say = action_obj.get("say") if isinstance(action_obj, dict) else None
        if a_type not in ("tool", "finish"):
            # Default to tool if looks like a tool name is present
            a_type = "tool" if name else "finish"
        if not isinstance(args, dict):
            args = {}
        return thought, ActionSpec(type=a_type, name=name, args=args, say=say)


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

    def execute(self, action: ActionSpec) -> str:
        if action.type == "finish":
            return action.say or "finish"
        if action.type != "tool":
            return "error: unknown_action_type"
        if not action.name:
            return "error: missing_tool_name"
        tool = self._tools.get(action.name)
        if not tool:
            return f"error: unknown_tool {action.name}"
        try:
            out = tool.handler(action.args or {})
            if isinstance(out, str):
                return out
            return json.dumps(out, ensure_ascii=False)
        except Exception as e:
            return f"error: {e}"


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
            keep_history=keep_history,
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

    def ask(self, prompt: str, mode: Optional[str] = None) -> AskResult:
        if not prompt:
            raise OpenRouterError("Empty prompt")

        tools_catalog = self.operator.tool_catalog()
        steps: List[ReActStep] = []
        transcript_lines: List[str] = []
        messages_accum: List[Dict[str, Any]] = []
        latest_reasoning: Optional[str] = None
        usage: Optional[Dict[str, Any]] = None

        for _ in range(self.max_steps):
            history_text = "\n\n".join(transcript_lines)
            thought, action, thinker_res = self.thinker.propose(prompt, tools_catalog, history_text)
            latest_reasoning = thinker_res.reasoning or latest_reasoning
            usage = thinker_res.usage or usage
            messages_accum.extend(thinker_res.messages or [])

            action_desc = action.say or (action.name or action.type)
            transcript_lines.append(f"Thought: {thought}")
            transcript_lines.append(f"Action: {action_desc}")

            # Execute
            observation = self.operator.execute(action)
            transcript_lines.append(f"Observation: {observation}")
            steps.append(ReActStep(thought=thought, action=action, observation=observation))

            last_step_text = "\n".join(transcript_lines[-3:])
            done, final_answer, val_res = self.validator.judge(prompt, last_step_text, "\n".join(transcript_lines))
            messages_accum.extend(val_res.messages or [])
            usage = val_res.usage or usage
            if done:
                content = "\n".join(transcript_lines + ["", f"Final Answer: {final_answer}"])
                return AskResult(content=content, reasoning=latest_reasoning, usage=usage, messages=messages_accum)

        # Max steps reached
        content = "\n".join(transcript_lines + ["", "Note: max steps reached without a final answer."])
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
