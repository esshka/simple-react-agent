# src/simple_or_agent/instructor_based/reasoning_agent.py
# Implements a structured reasoning agent with optional tool calls.
# Exists to provide a reusable Instructor loop that mirrors the ReAct agent features.
# RELEVANT FILES: src/simple_or_agent/instructor_based/agent.py, src/simple_or_agent/instructor_based/openrouter_client.py, src/simple_or_agent/instructor_based/tools.py

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_src = Path(__file__).resolve().parent.parent.parent
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

import json
from enum import Enum
from typing import Any, Dict, List, Optional

from instructor import Mode
from pydantic import BaseModel, Field

from simple_or_agent.instructor_based import openrouter_client
from simple_or_agent.instructor_based.agent import ObservationResponse
from simple_or_agent.instructor_based.provider_profiles import resolve_profile
from simple_or_agent.instructor_based.tools import ToolRegistry, ToolSpec
from simple_or_agent.instructor_based.calculator_tool import build_calculator_tool


class NextAction(str, Enum):
    CONTINUE = "continue"
    VALIDATE = "validate"
    FINAL_ANSWER = "final_answer"
    RESET = "reset"


class ReasoningStep(BaseModel):
    title: Optional[str] = Field(None, description="Short step title.")
    action: Optional[str] = Field(None, description="Planned action written in first person.")
    result: Optional[str] = Field(None, description="Outcome summary for the step.")
    reasoning: Optional[str] = Field(None, description="Why this step matters.")
    tool: Optional[str] = Field(None, description="Tool name when you need external help.")
    next_action: Optional[NextAction] = Field(None, description="continue, validate, final_answer, or reset.")
    confidence: Optional[float] = Field(None, description="Confidence score between 0.0 and 1.0.")


class ReasoningSteps(BaseModel):
    reasoning_steps: List[ReasoningStep] = Field(..., description="Ordered reasoning steps.")

def get_system_prompt(min_steps: int = 1, max_steps: int = 10) -> str:
    return f"""\
    You are a meticulous, thoughtful, and logical Reasoning Agent who solves complex problems through clear, structured, step-by-step analysis.\n

    Step 1 - Problem Analysis:
        - Restate the user's task clearly in your own words to ensure full comprehension.
        - Identify explicitly what information is required and what tools or resources might be necessary.

        Step 2 - Decompose and Strategize:
        - Break down the problem into clearly defined subtasks.
        - Develop at least two distinct strategies or approaches to solving the problem to ensure thoroughness.

        Step 3 - Intent Clarification and Planning:
        - Clearly articulate the user's intent behind their request.
        - Select the most suitable strategy from Step 2, clearly justifying your choice based on alignment with the user's intent and task constraints.
        - Formulate a detailed step-by-step action plan outlining the sequence of actions needed to solve the problem.

        Step 4 - Execute the Action Plan:
        For each planned step, document:
        1. **Title**: Concise title summarizing the step.
        2. **Action**: Explicitly state your next action in the first person ('I will...').
        3. **Result**: Execute your action using necessary tools and provide a concise summary of the outcome.
        4. **Reasoning**: Clearly explain your rationale, covering:
            - Necessity: Why this action is required.
            - Considerations: Highlight key considerations, potential challenges, and mitigation strategies.
            - Progression: How this step logically follows from or builds upon previous actions.
            - Assumptions: Explicitly state any assumptions made and justify their validity.
        5. **Tool**: When you need external data, set the `tool` field to the tool name listed in the prompt and leave `result` empty until the tool completes.
        6. **Next Action**: Clearly select your next step from:
            - **continue**: If further steps are needed.
            - **validate**: When you reach a potential answer, signaling it's ready for validation.
            - **final_answer**: Only if you have confidently validated the solution.
            - **reset**: Immediately restart analysis if a critical error or incorrect result is identified.
        7. **Confidence Score**: Provide a numeric confidence score (0.0–1.0) indicating your certainty in the step's correctness and its outcome.

        Step 5 - Validation (mandatory before finalizing an answer):
        - Explicitly validate your solution by:
            - Cross-verifying with alternative approaches (developed in Step 2).
            - Using additional available tools or methods to independently confirm accuracy.
        - Clearly document validation results and reasoning behind the validation method chosen.
        - If validation fails or discrepancies arise, explicitly identify errors, reset your analysis, and revise your plan accordingly.

        Step 6 - Provide the Final Answer:
        - Once thoroughly validated and confident, deliver your solution clearly and succinctly.
        - Restate briefly how your answer addresses the user's original intent and resolves the stated task.

        General Operational Guidelines:
        - Ensure your analysis remains:
            - **Complete**: Address all elements of the task.
            - **Comprehensive**: Explore diverse perspectives and anticipate potential outcomes.
            - **Logical**: Maintain coherence between all steps.
            - **Actionable**: Present clearly implementable steps and actions.
            - **Insightful**: Offer innovative and unique perspectives where applicable.
        - Always explicitly handle errors and mistakes by resetting or revising steps immediately.
        - Adhere strictly to a minimum of {min_steps} and maximum of {max_steps} steps to ensure effective task resolution.
        - Execute necessary tools proactively and without hesitation, clearly documenting tool usage.
        - Only create a single instance of ReasoningSteps for your response.\
    """


class ReasoningAgent:
    def __init__(
        self,
        *,
        min_steps: int = 1,
        max_steps: int = 10,
        temperature: float = 0.1,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> None:
        profile = resolve_profile()
        if not openrouter_client.is_openrouter(profile.provider_id):
            profile = resolve_profile("openrouter")
        provider = (
            provider_id
            or profile.provider_id
            or openrouter_client.DEFAULT_OPENROUTER_PROVIDER
        )
        if not openrouter_client.is_openrouter(provider):
            provider = profile.provider_id or openrouter_client.DEFAULT_OPENROUTER_PROVIDER
        key = api_key or openrouter_client.resolve_api_key_from_env() or profile.default_api_key
        if not key:
            raise ValueError("api_key is required for ReasoningAgent")
        if model:
            self.model_id = model
        elif profile.model_id:
            self.model_id = profile.model_id
        elif openrouter_client.is_openrouter(provider):
            self.model_id = openrouter_client.provider_model_hint(provider)
        elif provider.startswith("openai/"):
            self.model_id = provider.split("/", 1)[1]
        else:
            self.model_id = provider
        mode = openrouter_client.normalize_mode(profile.mode or Mode.TOOLS) if openrouter_client.is_openrouter(provider) else profile.mode or Mode.TOOLS
        self.client = openrouter_client.build_client(api_key=key, provider_id=provider, mode=mode)
        self.temperature = temperature
        self.min_steps = max(1, int(min_steps))
        self.max_steps = max(self.min_steps, int(max_steps))
        self._tools = ToolRegistry()
        self._prompt_template = get_system_prompt(self.min_steps, self.max_steps)
        self._system_prompt = self._compose_system_prompt()
        self._steps: List[ReasoningStep] = []

    def _compose_system_prompt(self) -> str:
        base = self._prompt_template.strip()
        catalog = self._tools.tool_names_and_descriptions()
        if catalog:
            base = f"{base}\n\nAvailable tools:\n{catalog}\n\nSet the `tool` field only to one of these names."
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
                choice = step.next_action.value if isinstance(step.next_action, NextAction) else str(step.next_action)
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
                "If you need to run a tool set `tool` to its exact name and leave `result` empty until the tool is observed."
            )
            step = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": self._user_message(prompt, history, directive)},
                ],
                response_model=ReasoningStep,
                temperature=self.temperature,
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
                if name not in self._tools.as_mapping():
                    known = ", ".join(self._tools.tool_names()) or "no tools"
                    raise ValueError(f"Unknown tool '{name}'. Known tools: {known}")
                union_model = self._tools.response_union()
                payload = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": self._user_message(
                            prompt,
                            self._history_text(),
                            f"Return only the arguments for the `{name}` tool as a JSON object.",
                        )},
                    ],
                    response_model=union_model,
                    temperature=self.temperature,
                )
                resolved_name, spec = self._tools.resolve(payload)
                if resolved_name != name:
                    raise RuntimeError(f"Expected tool '{name}' but model returned '{resolved_name}'.")
                tool_args = payload.model_dump()
                tool_result = spec.handler(tool_args)
                observation = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": self._user_message(
                            prompt,
                            self._history_text(),
                            f"You called `{resolved_name}`.",
                            f"Tool args:\n{self._stringify(tool_args)}",
                            f"Tool output:\n{self._stringify(tool_result)}",
                            "Summarize what this output means and how it affects the plan.",
                        )},
                    ],
                    response_model=ObservationResponse,
                    temperature=self.temperature,
                )
                step.tool = resolved_name
                step.result = observation.observation  # Record a short observation so the next step can build on it.
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
    response = run_reasoning_agent("Find the exact value of log(1234234) and then calculate the square root of the result", tools=[build_calculator_tool()])
    print("Final Answer: ", response.reasoning_steps[-1].result)
    print("Reasoning Steps: ", response.reasoning_steps)
    print("\n")