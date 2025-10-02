# src/simple_or_agent/instructor_based/react_agent.py
# Implements a minimal Instructor-driven ReAct agent with typed actions.
# Exists to offer a simple template for building union-based tool loops.
# RELEVANT FILES: src/simple_or_agent/instructor_based/reasoning_agent.py, src/simple_or_agent/instructor_based/calculator_tool.py, src/simple_or_agent/instructor_based/tools.py

from __future__ import annotations

from typing import Callable, Dict, List, Union

import instructor
from pydantic import BaseModel, Field

# Use the modern provider shortcut so the client matches current Instructor docs.
CLIENT = instructor.from_provider("openrouter/qwen/qwen3-next-80b-a3b-instruct")


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

    thought: str = Field(..., description="Reasoning that motivates the next step")
    action: Union[SearchTool, WeatherTool, FinalAnswerTool]


_TOOL_MAP: Dict[type, Callable[..., str]] = {
    SearchTool: search_web,
    WeatherTool: get_weather,
}


def run_react_loop(query: str, max_steps: int = 10) -> str:
    """Run the ReAct loop until the agent returns a final answer."""
    history: List[str] = []

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
        step: AgentAction = CLIENT.chat.completions.create(
            messages=messages,
            response_model=AgentAction,
        )

        thought = step.thought
        act = step.action

        if isinstance(act, FinalAnswerTool):
            # Final hop: return the model's answer.
            return act.answer

        # Look up the correct tool handler by model type.
        handler = _TOOL_MAP.get(type(act))
        if handler is None:
            observation = f"Unknown tool: {type(act).__name__}"
        else:
            payload = act.model_dump()
            observation = handler(**payload)

        # Record action + observation for the next turn so the model keeps context.
        history.append(f"Thought: {thought}")
        history.append(f"Action: {type(act).__name__}({act.model_dump()})")
        history.append(f"Observation: {observation}")

        print(f"History: {history}")

    return "Max steps reached before final answer."


if __name__ == "__main__":
    ANSWER = run_react_loop("What is the weather in the capital of France, and what is that city known for?")
    print("Final Answer:", ANSWER)
