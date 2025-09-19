# src/simple_or_agent/next_agent.py
# Planner + ReACT agent orchestrated with scoped memory.
# This exists to keep plans focused and execution context clean.
# RELEVANT FILES: src/simple_or_agent/orchestrator.py,src/simple_or_agent/memory.py,src/simple_or_agent/react_agent.py

#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple, Callable

from .openrouter_client import OpenRouterClient, OpenRouterError
from .simple_agent import SimpleAgent, ToolSpec, AskResult
from .react_agent import ReActAgent as ExecReActAgent
from .orchestrator import Orchestrator
from .memory import ConversationMemory, MemoryPolicy


class NextAgent:
    """Planner + ReACT execution coordinated by an Orchestrator.

    - Orchestrator decides what context to pass each subagent.
    - ConversationMemory keeps short summaries and recent steps.
    """

    PLANNER_SYSTEM = (
        "You are a Planner for extremely hard tasks. Design a focused, minimal plan that maximizes signal and reduces risk.\n"
        "Rules:\n"
        "- Output 5–9 numbered steps (short, actionable, dependency-aware).\n"
        "- Include Assumptions (bullet list) and Risks/Mitigations (bullet list).\n"
        "- Include up to 3 Clarifying Questions if critical.\n"
        "- Do NOT solve; do NOT compute results; no conclusions or numbers."
    )

    def __init__(
        self,
        client: OpenRouterClient,
        model: Optional[str] = None,
        planner_system: Optional[str] = None,
        keep_history: bool = True,
        temperature: float = 0.1,
        max_rounds: int = 8,
        reasoning_effort: Optional[str] = None,
        memory_policy: Optional[MemoryPolicy] = None,
    ) -> None:
        self.planner = SimpleAgent(
            client,
            model=model,
            system_prompt=planner_system or self.PLANNER_SYSTEM,
            keep_history=keep_history,
            temperature=temperature,
            max_rounds=max_rounds,
            max_tool_iters=0,
            response_format=None,
            reasoning_effort=reasoning_effort,
            parallel_tool_calls=False,
            tool_choice="none",
            inline_tools=False,
        )
        self.executor = ExecReActAgent(
            client,
            model=model,
            system_prompt=None,
            keep_history=keep_history,
            temperature=temperature,
            max_rounds=max_rounds,
            max_tool_iters=3,
            response_format=None,
            reasoning_effort=reasoning_effort,
            parallel_tool_calls=False,
        )

        # Orchestrator + memory. No model summarizer by default; use naive.
        self.memory = ConversationMemory(policy=memory_policy or MemoryPolicy())
        self.orch = Orchestrator(self.memory)

    def add_tool(self, tool: ToolSpec) -> None:
        self.executor.add_tool(tool)

    def remove_tool(self, name: str) -> None:
        self.executor.remove_tool(name)

    def reset(self) -> None:
        self.planner.reset()
        self.executor.reset()
        self.memory.clear_all()

    # --- prompt builders: keep consistent phrasing across subagents ---
    @staticmethod
    def _build_planner_prompt(task: str, prior: str) -> str:
        extra = f"\n\nPrior context (summary):\n{prior}" if prior else ""
        return (
            f"Task:\n{task}{extra}\n\n"
            "Produce ONLY: Steps, Assumptions, Risks/Mitigations, and optional Clarifying Questions.\n"
            "Do not solve or compute anything."
        )

    @staticmethod
    def _build_exec_prompt(task: str, plan_text: str, notes: str) -> str:
        ctx = f"\n\nNotes (summary):\n{notes}" if notes else ""
        return (
            f"Task:\n{task}\n\n"
            f"Plan (follow step-by-step; adapt if needed):\n{plan_text}{ctx}\n\n"
            "Use a Thought → Action → Observation cycle. Prefer tools when helpful.\n"
            "Stop when the task is solved and provide the Final Answer."
        )

    # --- public API ---
    def plan(self, prompt: str) -> AskResult:
        return self.orch.plan(self.planner, prompt, self._build_planner_prompt)

    def execute_with_plan(self, task: str, plan_text: str, progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> AskResult:
        return self.orch.execute(self.executor, task, plan_text, self._build_exec_prompt, progress_cb)

    def ask(self, prompt: str, mode: Optional[str] = None, progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> AskResult:
        if mode == "planner":
            return self.plan(prompt)
        if mode == "react":
            # Treat prompt as a standalone goal (no pre-plan)
            return self.executor.ask(prompt, progress_cb=progress_cb)

        plan_res, exec_res = self.orch.ask(self.planner, self.executor, prompt, self._build_planner_prompt, self._build_exec_prompt, progress_cb)
        content = "Plan\n" + (plan_res.content or "").strip() + "\n\n" + (exec_res.content or "").strip()
        messages: List[Dict[str, Any]] = (plan_res.messages or []) + (exec_res.messages or [])
        return AskResult(content=content, reasoning=exec_res.reasoning, usage=exec_res.usage, messages=messages)

    # Utilities
    @staticmethod
    def split_transcript_and_final(text: str) -> Tuple[str, str]:
        """Return (transcript, final_answer_text) using the last 'Final Answer:' marker.

        - If not found, returns (text, "").
        - Trims surrounding whitespace from both parts.
        """
        if not isinstance(text, str) or not text:
            return "", ""
        i = text.rfind("Final Answer:")
        if i < 0:
            return text, ""
        transcript = text[:i].rstrip()
        final = text[i + len("Final Answer:"):].strip()
        return transcript, final


__all__ = ["ToolSpec", "AskResult", "NextAgent"]
