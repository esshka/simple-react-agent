# src/simple_or_agent/instructor_based/reasoning_agent.py
# Implements a structured reasoning agent with optional tool calls.
# Exists to provide a reusable Instructor loop that mirrors the ReAct agent features.
# RELEVANT FILES: agent.py, openrouter_client.py, tools.py
from __future__ import annotations

import os
import instructor
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_src = Path(__file__).resolve().parent.parent.parent
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

import json
from typing import Any, Dict, List, Optional, Type

from instructor import Mode
from instructor.core.exceptions import InstructorRetryException

from pydantic import BaseModel

from simple_or_agent.instructor_based.reasoning_models import (
    MaybeToolCall,
    NextAction,
    ReasoningStep,
    ReasoningSteps,
)
from simple_or_agent.instructor_based.reasoning_prompts import get_system_prompt
from simple_or_agent.instructor_based.agent import ObservationResponse
from simple_or_agent.instructor_based.tools import ToolRegistry, ToolSpec

api_key = os.getenv("OPENROUTER_API_KEY")

class ReasoningAgent:
    def __init__(
        self,
        *,
        min_steps: int = 1,
        max_steps: int = 10,
        temperature: float = 0.01,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for ReasoningAgent")
        self.client = self._build_client()
        self.temperature = temperature
        self.min_steps = max(1, int(min_steps))
        self.max_steps = max(self.min_steps, int(max_steps))
        self._tools = ToolRegistry()
        self._prompt_template = get_system_prompt(self.min_steps, self.max_steps)
        self._system_prompt = self._compose_system_prompt()
        self._steps: List[ReasoningStep] = []

    def _build_client(self) -> Any:
        return instructor.from_provider("openrouter/qwen/qwen3-next-80b-a3b-instruct", api_key=api_key, mode=Mode.TOOLS)

    def _compose_system_prompt(self) -> str:
        base = self._prompt_template.strip()
        catalog = self._tools.tool_names_and_descriptions()
        if catalog:
            base = (
                f"{base}\n\nAvailable tools:\n{catalog}\n\n"
                "Set the `tool` field only to one of these names."
            )
        return base

    def _history_text(self) -> str:
        if not self._steps:
            return "No steps recorded yet."
        blocks: List[str] = []
        for index, step in enumerate(self._steps, start=1):
            parts = [f"Step {index}: {step.title or 'Untitled step'}"]
            if step.action:
                parts.append(f"Action: {step.action}")
            if step.reasoning:
                parts.append(f"Reasoning: {step.reasoning}")
            if step.tool:
                parts.append(f"Tool: {step.tool}")
            if step.result:
                parts.append(f"Result: {step.result}")
            if step.next_action:
                if isinstance(step.next_action, NextAction):
                    choice = step.next_action.value
                else:
                    choice = str(step.next_action)
                parts.append(f"Next Action: {choice}")
            if step.confidence is not None:
                parts.append(f"Confidence: {step.confidence}")
            blocks.append("\n".join(parts))
        return "\n\n".join(blocks)

    def _user_message(self, prompt: str, history: str, *blocks: str) -> str:
        lines = [prompt, "", "Reasoning history:", history]
        for block in blocks:
            if block:
                lines.extend(["", block])
        return "\n".join(lines)

    def _chat(self, user_content: str, response_model: Type[BaseModel]) -> BaseModel:
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_model": response_model,
            "temperature": self.temperature,
            "extra_body": {"provider": {"require_parameters": True}},
        }
        try:
            return self.client.chat.completions.create(**payload)
        except InstructorRetryException as exc:
            raise RuntimeError(str(exc)) from exc

    def _stringify(self, value: Any) -> str:
        if isinstance(value, BaseModel):
            return json.dumps(value.model_dump(), indent=2, sort_keys=True)
        if isinstance(value, dict):
            return json.dumps(value, indent=2, sort_keys=True)
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value), indent=2, sort_keys=True)
        return str(value)

    def add_tool(self, tool: ToolSpec) -> None:
        self._tools.add(tool)
        self._system_prompt = self._compose_system_prompt()

    def remove_tool(self, name: str) -> None:
        self._tools.remove(name)
        self._system_prompt = self._compose_system_prompt()

    def run(self, prompt: str) -> ReasoningSteps:
        if not prompt:
            raise ValueError("Prompt must not be empty.")
        self._system_prompt = self._compose_system_prompt()
        self._steps = []
        for _ in range(self.max_steps * 2):
            history = self._history_text()
            directive = (
                "Return the next ReasoningStep JSON object. Provide exactly one step. "
                "If you need to run a tool set `tool` to its exact name and leave "
                "`result` empty until the tool is observed."
            )
            step = self._chat(
                self._user_message(prompt, history, directive),
                ReasoningStep,
            )
            self._steps.append(step)
            if step.next_action == NextAction.RESET:
                self._steps.clear()
                continue
            if len(self._steps) > self.max_steps:
                raise RuntimeError("Exceeded configured max_steps before final answer.")
            if step.tool:  # Ask the model for structured tool args and run the handler.
                name = step.tool.strip()
                if not name:
                    raise ValueError("Tool field is present but empty.")
                if not self._tools.has_tools():
                    raise RuntimeError("A tool was requested but no tools are registered.")
                tools_map = self._tools.as_mapping()
                if name not in tools_map:
                    known = ", ".join(self._tools.tool_names()) or "no tools"
                    raise ValueError(f"Unknown tool '{name}'. Known tools: {known}")
                spec = tools_map[name]
                maybe = self._chat(
                    self._user_message(
                        prompt,
                        self._history_text(),
                        (
                            f"Return a MaybeToolCall JSON object for the `{name}` tool. "
                            "Populate `result` with the exact arguments when the call is valid. "
                            "If the call cannot proceed, set `error` to true and explain why "
                            "in `message`."
                        ),
                    ),
                    MaybeToolCall,
                )
                step.tool = name
                if maybe.error or maybe.result is None:
                    if maybe.message:
                        failure_note = self._stringify(maybe.message)
                    else:
                        failure_note = "Tool call failed without an explanation."
                    partial_payload = (
                        self._stringify(maybe.result) if maybe.result else "None available"
                    )
                    failure_prompt = self._user_message(
                        prompt,
                        self._history_text(),
                        f"Tool `{name}` reported an error.",
                        f"Error message: {failure_note}",
                        f"Partial payload: {partial_payload}",
                        "Explain how this affects the plan and suggest a next step.",
                    )
                    observation = self._chat(
                        failure_prompt,
                        ObservationResponse,
                    )
                    step.result = observation.observation
                    continue
                validated_args = spec.model_class()(**(maybe.result or {}))
                tool_output = spec.handler(validated_args.model_dump())
                step.result = self._chat(
                    self._user_message(
                        prompt,
                        self._history_text(),
                        f"You called `{name}`.",
                        f"Tool args:\n{self._stringify(validated_args.model_dump())}",
                        f"Tool output:\n{self._stringify(tool_output)}",
                        "Summarize what this output means and how it affects the plan.",
                    ),
                    ObservationResponse,
                ).observation

            if step.next_action == NextAction.FINAL_ANSWER and len(self._steps) >= self.min_steps:
                break
        if not self._steps:
            raise RuntimeError("No reasoning steps were produced.")
        if self._steps[-1].next_action != NextAction.FINAL_ANSWER:
            raise RuntimeError("Reasoning agent stopped without a final answer.")
        return ReasoningSteps(reasoning_steps=self._steps)

def run_reasoning_agent(
    prompt: str,
    *,
    tools: Optional[List[ToolSpec]] = None,
    **agent_kwargs: Any,
) -> ReasoningSteps:
    agent = ReasoningAgent(**agent_kwargs)
    for tool in tools or []:
        agent.add_tool(tool)
    return agent.run(prompt)

if __name__ == "__main__":
    from simple_or_agent.instructor_based.calculator_tool import build_calculator_tool

    demo_agent = ReasoningAgent()
    demo_agent.add_tool(build_calculator_tool())
    demo_steps = demo_agent.run("Calculate the square root of the base-10 log of 1234234.")
    print("Final answer:", demo_steps.reasoning_steps[-1].result)
