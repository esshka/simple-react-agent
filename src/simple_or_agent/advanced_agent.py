from __future__ import annotations

from typing import Any, Dict, Optional, List

from .openrouter_client import OpenRouterClient
from .simple_agent import SimpleAgent, ToolSpec, AskResult


class ReActAgent:
    """Composition of a planner and a worker SimpleAgent."""

    WORKER_SYSTEM = (
        "You are a Worker. Execute the given plan step-by-step, using tools to gather facts, and produce concise, actionable outputs. "
        "Ask for missing info when needed. Provide the Final Answer last."
    )
    PLANNER_SYSTEM = (
        "You are a Planner. Your job is to design a short, actionable plan (3–7 numbered steps) and optional clarifying questions. "
        "Do NOT solve or compute the final answer in any channel. Do not include results, numbers, or conclusions. "
        "Avoid calling tools. Keep reasoning focused on prioritization, dependencies, and information gaps, not on solving the task."
    )
    DEFAULT_SYSTEM = WORKER_SYSTEM

    def __init__(
        self,
        client: OpenRouterClient,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        keep_history: bool = True,
        temperature: float = 0.1,
        max_rounds: int = 6,
        max_tool_iters: int = 3,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Optional[Any] = "auto",
        planner_system: Optional[str] = None,
    ) -> None:
        worker_sys = system_prompt or self.WORKER_SYSTEM
        planner_sys = planner_system or self.PLANNER_SYSTEM
        self.planner = SimpleAgent(
            client,
            model=model,
            system_prompt=planner_sys,
            keep_history=keep_history,
            temperature=temperature,
            max_rounds=max_rounds,
            max_tool_iters=max_tool_iters,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            parallel_tool_calls=False,
            tool_choice="none",
            inline_tools=False,
        )
        self.worker = SimpleAgent(
            client,
            model=model,
            system_prompt=worker_sys,
            keep_history=keep_history,
            temperature=temperature,
            max_rounds=max_rounds,
            max_tool_iters=max_tool_iters,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            inline_tools=True,
        )

    def add_tool(self, tool: ToolSpec) -> None:
        self.worker.add_tool(tool)

    def remove_tool(self, name: str) -> None:
        self.worker.remove_tool(name)

    def reset(self) -> None:
        self.planner.reset(); self.worker.reset()

    def plan(self, prompt: str) -> AskResult:
        return self.planner.ask(prompt)

    def work(self, prompt: str) -> AskResult:
        return self.worker.ask(prompt)

    def ask(self, prompt: str, mode: Optional[str] = None) -> AskResult:
        if mode == "planner":
            return self.plan(prompt)
        if mode == "worker":
            return self.work(prompt)
        plan_res = self.plan(prompt)
        exec_input = (
            f"Task:\n{prompt}\n\nPlan:\n{plan_res.content}\n\n"
            f"Execute the plan carefully and provide the Final Answer."
        )
        work_res = self.work(exec_input)
        content = f"Plan\n{plan_res.content}\n\nFinal Answer\n{work_res.content}"
        messages: List[Dict[str, Any]] = (plan_res.messages or []) + (work_res.messages or [])
        return AskResult(content=content, reasoning=work_res.reasoning, usage=work_res.usage, messages=messages)


__all__ = ["ToolSpec", "AskResult", "ReActAgent", "SimpleAgent"]

