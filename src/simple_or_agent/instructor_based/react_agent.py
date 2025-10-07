# src/simple_or_agent/instructor_based/react_agent.py
# Implements a minimal Instructor-driven ReAct agent with typed actions.
# Exists to offer a simple template for building union-based tool loops.
# RELEVANT FILES: src/simple_or_agent/instructor_based/reasoning_agent.py, src/simple_or_agent/instructor_based/calculator_tool.py, src/simple_or_agent/instructor_based/tools.py

from __future__ import annotations
from typing import Any, Callable, Dict, List, Union, Optional

import instructor
from pydantic import BaseModel, Field

from enum import Enum


from reasoning_prompts import get_system_prompt

# Use the modern provider shortcut so the client matches current Instructor docs.
CLIENT = instructor.from_provider("openrouter/qwen/qwen3-next-80b-a3b-instruct")



# TODO add planning and think tools 


def search_web(query: str) -> str:
    """Return canned web search answers for demo purposes."""
    lowered = query.lower()
    if "capital of france" in lowered:
        return "The capital of France is Paris."
    if "paris" in lowered and "known for" in lowered:
        return "Paris is known for the Eiffel Tower, the Louvre, and its café culture."
    return f"No results for: {query}"


def get_weather(city: str) -> str:
    """Return canned weather data for demo purposes."""
    if city.lower() == "paris":
        return "Paris weather: 20°C and sunny."
    return f"No weather data for {city}."


class SearchTool(BaseModel):
    """Inputs required to perform a faux web search."""

    query: str = Field(..., description="Search query text")


class WeatherTool(BaseModel):
    """Inputs required to fetch faux weather."""

    city: str = Field(..., description="City name")


class FinalAnswerTool(BaseModel):
    """Deliver the final answer when reasoning is complete."""

    answer: str


class AgentAction(BaseModel):
    """Structured Thought → Action payload enforced by Instructor."""

    thought_topic: str = Field(..., description="Topic of the thought. What do I need to think about?")
    action: Union[SearchTool, WeatherTool, FinalAnswerTool]


_TOOL_MAP: Dict[type, Callable[..., str]] = {
    SearchTool: search_web,
    WeatherTool: get_weather,
}


def _summarize_action(action_name: str, action_args: Dict[str, Any]) -> str:
    """Return a compact string that mirrors the exact tool call."""
    if not action_args:
        return f"{action_name}()"
    formatted_args = ", ".join(f"{key}={value}" for key, value in action_args.items())
    return f"{action_name}({formatted_args})"


def _format_reasoning_outline(entries: List[Dict[str, str]]) -> str:
    """Build a readable outline so we can inspect each reasoning hop quickly."""
    if not entries:
        return "Reasoning outline is empty."

    lines: List[str] = ["Reasoning Outline"]
    lines.append("------------------")
    for entry in entries:
        lines.append(f"Step {entry['step']}: {entry['topic']}")
        lines.append(f"  Thought: {entry['thought']}")
        lines.append(f"  Action: {entry['action']}")
        lines.append(f"  Observation: {entry['observation']}")
        lines.append("")
    return "\n".join(lines).rstrip()


class NextAction(str, Enum):
    """Allowed directives emitted by the LLM."""

    CONTINUE = "continue"
    VALIDATE = "validate"
    FINAL_ANSWER = "final_answer"
    RESET = "reset"


class ReasoningStep(BaseModel):
    """Single reasoning step returned by the LLM."""

    title: Optional[str] = Field(None, description="Short step title.")
    action: Optional[str] = Field(None, description="Planned action written in first person.")
    result: Optional[str] = Field(None, description="Outcome summary for the step.")
    reasoning: Optional[str] = Field(None, description="Why this step matters.")
    next_action: Optional[NextAction] = Field(None, description="continue, validate, final_answer, or reset.")
    confidence: Optional[float] = Field(None, description="Confidence score between 0.0 and 1.0.")


class ReasoningSteps(BaseModel):
    """Ordered reasoning steps returned by the agent."""

    reasoning_steps: List[ReasoningStep] = Field(..., description="Ordered reasoning steps.")


def _format_thought_steps(steps: List[ReasoningStep]) -> str:
    """Build a readable log that shows every step returned by the thought model."""
    if not steps:
        return "Thought agent returned no steps."

    lines: List[str] = ["Thought Agent Steps"]
    lines.append("-------------------")
    for index, step in enumerate(steps, start=1):
        title = step.title or f"Step {index}"
        lines.append(f"{index}. {title}")
        if step.reasoning:
            lines.append(f"   Reasoning: {step.reasoning}")
        if step.action:
            lines.append(f"   Action: {step.action}")
        if step.next_action:
            lines.append(f"   Next Action: {step.next_action.value}")
        if step.confidence is not None:
            lines.append(f"   Confidence: {step.confidence}")
        lines.append("")
    return "\n".join(lines).rstrip()


def generate_tought(
    *,
    topic: str,
    query: str,
    history: List[str],
    action_name: str,
    action_args: Dict[str, Any],
) -> str:
    """Create the text that fills the Thought stage using the latest context."""

    # Feed the thought agent grounded context so it mirrors the main loop faithfully.
    history_text = "\n".join(history) if history else "No previous steps."
    args_text = ", ".join(f"{key}={value}" for key, value in action_args.items()) if action_args else "no arguments"
    user_payload = (
        f"User query: {query}\n"
        f"Thought topic: {topic}\n"
        f"Planned action: {action_name}({args_text})\n"
        f"History:\n{history_text}\n"
        "Write the next Thought now."
    )

    messages = [
        {"role": "system", "content": get_system_prompt(min_steps=1, max_steps=1, mode="thought")},
        {"role": "user", "content": user_payload},
    ]
   

    thought_reply = CLIENT.chat.completions.create(
            messages=messages,
            response_model=ReasoningSteps,
            extra_body={"provider": {"require_parameters": True}},
        )

    steps = thought_reply.reasoning_steps
    print(_format_thought_steps(steps))
    print("")
    if not steps:
        return "No structured reasoning returned."

    # Fall back to the richest step field so the loop always records a thought.
    final_step = steps[-1]
    return final_step.reasoning or final_step.action or ""


def run_react_loop(query: str, max_steps: int = 10) -> str:
    """Run the ReAct loop until the agent returns a final answer."""
    history: List[str] = []
    reasoning_outline: List[Dict[str, str]] = []

    for _ in range(max_steps):
        # Keep the prompt simple: include the system framing plus the rolling history.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an agent that uses a Thought → Action → Observation loop.\n"
                    "Available tools: SearchTool, WeatherTool.\n"
                    "Call FinalAnswerTool when you can answer the user.\n\n"
                ),
            },
            {"role": "user", "content": f"User query: {query}"},
            {"role": "system", "content": f"History:\n{'\n'.join(history)}"},
        ]
        

        # Ask Instructor to produce the next step, enforcing the AgentAction schema.
        step = CLIENT.chat.completions.create(
                messages=messages,
                response_model=AgentAction,
                extra_body={"provider": {"require_parameters": True}},
            )

        thought_topic = step.thought_topic
        act = step.action

        if isinstance(act, FinalAnswerTool):
            # Final hop: return the model's answer.
            return act.answer

        payload = act.model_dump()
        action_summary = _summarize_action(type(act).__name__, payload)
        # Ask the auxiliary agent to expand the topic into the actual Thought text.
        thought = generate_tought(
            topic=thought_topic,
            query=query,
            history=history,
            action_name=type(act).__name__,
            action_args=payload,
        )

        # Look up the correct tool handler by model type.
        handler = _TOOL_MAP.get(type(act))
        if handler is None:
            observation = f"Unknown tool: {type(act).__name__}"
        else:
            observation = handler(**payload)

        # Record action + observation for the next turn so the model keeps context.
        history.append(f"Thought: {thought}")
        history.append(f"Action: {action_summary}")
        history.append(f"Observation: {observation}")

        # Keep a separate outline printout so we can read the complete reasoning easily.
        reasoning_outline.append(
            {
                "step": str(len(reasoning_outline) + 1),
                "topic": thought_topic,
                "thought": thought,
                "action": action_summary,
                "observation": observation,
            }
        )
        print(_format_reasoning_outline(reasoning_outline))
        print("")

    return "Max steps reached before final answer."


if __name__ == "__main__":
    ANSWER = run_react_loop("What is the weather in the capital of France, and what is that city known for?")
    print("Final Answer:", ANSWER)
