# src/simple_or_agent/instructor_based/react_agent.py
# Implements a minimal Instructor-driven ReAct agent with typed actions.
# Exists to offer a simple template for building union-based tool loops.
# RELEVANT FILES: src/simple_or_agent/instructor_based/reasoning_agent.py, src/simple_or_agent/instructor_based/calculator_tool.py, src/simple_or_agent/instructor_based/tools.py

from __future__ import annotations
from typing import Any, Callable, Dict, List, Union, cast

import instructor
from pydantic import BaseModel, Field

from reasoning_prompts import get_system_prompt
from openai.types.chat import ChatCompletionMessageParam

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


class ThoughtResponse(BaseModel):
    """Short reflection produced by the auxiliary thought agent."""

    thought: str = Field(..., description="First-person reasoning text for the current step")


_TOOL_MAP: Dict[type, Callable[..., str]] = {
    SearchTool: search_web,
    WeatherTool: get_weather,
}

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

    raw_messages = [
        {"role": "system", "content": get_system_prompt(mode="thought")},
        {"role": "user", "content": user_payload},
    ]
    messages = cast(List[ChatCompletionMessageParam], raw_messages)

    thought_reply = cast(
        ThoughtResponse,
        CLIENT.chat.completions.create(
            messages=messages,
            response_model=ThoughtResponse,
            extra_body={"provider": {"require_parameters": True}},
        ),
    )

    return thought_reply.thought.strip()


def run_react_loop(query: str, max_steps: int = 10) -> str:
    """Run the ReAct loop until the agent returns a final answer."""
    history: List[str] = []

    for _ in range(max_steps):
        # Keep the prompt simple: include the system framing plus the rolling history.
        raw_messages = [
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
        messages = cast(List[ChatCompletionMessageParam], raw_messages)

        # Ask Instructor to produce the next step, enforcing the AgentAction schema.
        step = cast(
            AgentAction,
            CLIENT.chat.completions.create(
                messages=messages,
                response_model=AgentAction,
                extra_body={"provider": {"require_parameters": True}},
            ),
        )

        thought_topic = step.thought_topic
        act = step.action

        if isinstance(act, FinalAnswerTool):
            # Final hop: return the model's answer.
            return act.answer

        payload = act.model_dump()
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
        history.append(f"Action: {type(act).__name__}({payload})")
        history.append(f"Observation: {observation}")

        print(f"History: {history}")

    return "Max steps reached before final answer."


if __name__ == "__main__":
    ANSWER = run_react_loop("What is the weather in the capital of France, and what is that city known for?")
    print("Final Answer:", ANSWER)
