from __future__ import annotations

from typing import Any, Dict, Optional, List

from .openrouter_client import OpenRouterClient, OpenRouterError
from .simple_agent import SimpleAgent, ToolSpec, AskResult
from .react_agent import ReActAgent as ExecReActAgent


class NextAgent:
    """Planner + ReACT execution for hard tasks.

    - Planner: creates a concise, high‑leverage plan and clarifying questions.
    - ReACT: executes via Thought → Action → Observation until Final Answer.
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

    def add_tool(self, tool: ToolSpec) -> None:
        self.executor.add_tool(tool)

    def remove_tool(self, name: str) -> None:
        self.executor.remove_tool(name)

    def reset(self) -> None:
        self.planner.reset()
        self.executor.reset()

    def plan(self, prompt: str) -> AskResult:
        if not prompt:
            raise OpenRouterError("Empty prompt")
        plan_prompt = (
            f"Task:\n{prompt}\n\n"
            "Produce ONLY: Steps, Assumptions, Risks/Mitigations, and optional Clarifying Questions.\n"
            "Do not solve or compute anything."
        )
        return self.planner.ask(plan_prompt)

    def execute_with_plan(self, task: str, plan_text: str) -> AskResult:
        if not task:
            raise OpenRouterError("Empty task")
        goal = (
            f"Task:\n{task}\n\n"
            f"Plan (follow step-by-step; adapt if needed):\n{plan_text}\n\n"
            "Use a Thought → Action → Observation cycle. Prefer tools when helpful.\n"
            "Stop when the task is solved and provide the Final Answer."
        )
        return self.executor.ask(goal)

    def ask(self, prompt: str, mode: Optional[str] = None) -> AskResult:
        if mode == "planner":
            return self.plan(prompt)
        if mode == "react":
            # Treat prompt as a standalone goal (no pre-plan)
            return self.executor.ask(prompt)

        plan_res = self.plan(prompt)
        exec_res = self.execute_with_plan(prompt, plan_res.content)
        content = (
            "Plan\n" + plan_res.content.strip() + "\n\n" + exec_res.content.strip()
        )
        messages: List[Dict[str, Any]] = (plan_res.messages or []) + (exec_res.messages or [])
        return AskResult(content=content, reasoning=exec_res.reasoning, usage=exec_res.usage, messages=messages)


__all__ = ["ToolSpec", "AskResult", "NextAgent"]

