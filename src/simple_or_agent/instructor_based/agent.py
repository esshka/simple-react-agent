# src/simple_or_agent/instructor_based/agent.py
# Implements an Instructor-powered ReAct agent loop with tool support.
# Exists to offer a minimal OpenRouter-friendly orchestrator in this codebase.
# RELEVANT FILES: src/simple_or_agent/instructor_based/openrouter_client.py, src/simple_or_agent/instructor_based/provider_profiles.py, src/simple_or_agent/instructor_based/calculator_tool.py

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ''}:
    project_src = Path(__file__).resolve().parent.parent.parent
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Literal, Union

from instructor import Mode
from pydantic import BaseModel

from simple_or_agent.instructor_based.prompt_manager import (
    DEFAULT_REACT_SYSTEM_PROMPT_TEMPLATE,
    render_system_prompt,
)
from simple_or_agent.instructor_based import openrouter_client
from simple_or_agent.instructor_based.calculator_tool import build_calculator_tool
from simple_or_agent.instructor_based.provider_profiles import resolve_profile, resolved_model_from_env
from simple_or_agent.instructor_based.tools import ToolRegistry, ToolSpec

MODE_ENV = "INSTRUCTOR_MODE"


def _resolve_api_key() -> Optional[str]:
    """Read the preferred API key from the environment."""
    return openrouter_client.resolve_api_key_from_env()


def _resolve_provider() -> Optional[str]:
    """Return the explicit provider override when set."""
    return openrouter_client.resolve_provider_from_env()


def _resolve_mode() -> Optional[Mode]:
    """Map INSTRUCTOR_MODE to an Instructor Mode enum value."""
    raw = os.getenv(MODE_ENV)
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    try:
        return Mode[candidate.upper()]
    except KeyError:
        names = ", ".join(member.name for member in Mode)
        print(f"Unknown INSTRUCTOR_MODE '{raw}'. Valid options: {names}")
        return None


def _derive_model_id(model: Optional[str], provider_id: Optional[str]) -> str:
    """Return the completion model id based on the hints provided."""
    if model:
        return model
    if provider_id:
        if openrouter_client.is_openrouter(provider_id):
            return openrouter_client.provider_model_hint(provider_id)
        if provider_id.startswith("openai/"):
            return provider_id.split("/", 1)[1]
        return provider_id
    return "qwen/qwen3-next-80b-a3b-instruct"


def _build_client(
    api_key: str,
    provider_id: str,
    mode: Optional[Mode],
) -> Any:
    """Create an Instructor client using the OpenRouter settings."""
    return openrouter_client.build_client(
        api_key=api_key,
        provider_id=provider_id,
        mode=mode,
    )


class FinalAnswer(BaseModel):
    """Final answer response"""
    type: Literal["final"] = "final"
    answer: str


class ThinkResponse(BaseModel):
    """Thought response"""
    type: Literal["think"] = "think"
    thoughts: str


class ObservationResponse(BaseModel):
    """Observation response"""
    type: Literal["observation"] = "observation"
    observation: str


def _stringify_message_content(value: Any) -> str:
    """Convert structured results into plain text for chat message transport."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _maybe_extract_final_answer(text: str) -> Optional[str]:
    """Detect a final answer embedded inside a thought response."""
    lowered = text.lower()
    markers = ["answer:", "answer is", "final answer:", "final answer is", "final answer", "final result"]
    for marker in markers:
        index = lowered.find(marker)
        if index == -1:
            continue
        remainder = text[index + len(marker):].strip(" .:")
        if remainder:
            return remainder
    return None


class ReActAgent:
    """Minimal ReAct loop that works."""
    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: str = DEFAULT_REACT_SYSTEM_PROMPT_TEMPLATE,
        temperature: float = 0.1,
        max_steps: int = 6,
        api_key: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> None:
        base_profile = resolve_profile()  # Load provider defaults from providers.ini.
        profile = base_profile if openrouter_client.is_openrouter(base_profile.provider_id) else resolve_profile("openrouter")
        # Always pivot to OpenRouter defaults so this agent talks to the expected service.
        env_provider = _resolve_provider()  # Let env override the provider.
        env_mode = _resolve_mode()  # Let env override the mode.
        env_model = resolved_model_from_env()  # Allow env to override the model id.
        using_profile_defaults = (
            provider_id is None
            and env_provider is None
            and env_mode is None
            and env_model is None
        )  # Only rely on the profile when nothing else is set.
        resolved_provider = (
            provider_id
            or env_provider
            or profile.provider_id
            or openrouter_client.DEFAULT_OPENROUTER_PROVIDER
        )
        if not openrouter_client.is_openrouter(resolved_provider):
            # Enforce the OpenRouter contract even when a non-OpenRouter id slips in.
            resolved_provider = profile.provider_id or openrouter_client.DEFAULT_OPENROUTER_PROVIDER

        resolved_api_key = (
            api_key
            or _resolve_api_key()
            or (profile.default_api_key if using_profile_defaults else None)
        )
        if not resolved_api_key:
            raise ValueError("api_key is required for ReActAgent")

        explicit_model = model or env_model or (profile.model_id if using_profile_defaults else None)
        if explicit_model:
            resolved_model = explicit_model
        else:
            resolved_model = _derive_model_id(None, resolved_provider)

        if using_profile_defaults and profile.mode:
            resolved_mode = profile.mode
        elif env_mode is not None:
            resolved_mode = env_mode
        else:
            resolved_mode = None

        if resolved_mode and openrouter_client.is_openrouter(resolved_provider):
            # OpenRouter rejects JSON_SCHEMA, so we align with its JSON expectation.
            resolved_mode = openrouter_client.normalize_mode(resolved_mode)

        if resolved_mode is None and openrouter_client.is_openrouter(resolved_provider):
            fallback_mode = profile.mode or Mode.TOOLS
            resolved_mode = openrouter_client.normalize_mode(fallback_mode)

        mode_label = resolved_mode.name if resolved_mode else "provider default"
        print(f"ReActAgent provider: {resolved_provider}")
        print(f"ReActAgent model: {resolved_model}")
        print(f"ReActAgent mode: {mode_label}")

        self.client = _build_client(
            api_key=resolved_api_key,
            provider_id=resolved_provider,
            mode=resolved_mode,
        )
        self.model_id = resolved_model
        self.temperature = temperature
        self.max_steps = max(1, int(max_steps))
        self._tools = ToolRegistry()
        self._system_prompt_template = system_prompt
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._render_system_prompt()}
        ]

    def add_tool(self, tool: ToolSpec) -> None:
        self._tools.add(tool)
        self._refresh_system_prompt()

    def remove_tool(self, name: str) -> None:
        self._tools.remove(name)
        self._refresh_system_prompt()

    def _refresh_system_prompt(self) -> None:
        """Update the stored system prompt so the model sees the latest tool list."""
        prompt = self._render_system_prompt()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = prompt
            return
        # Insert a fresh system message if the log somehow lost the original.
        self.messages.insert(0, {"role": "system", "content": prompt})

    def _render_system_prompt(self) -> str:
        """Render the prompt template with the current tool block."""
        return render_system_prompt(self._system_prompt_template, self._tools.as_mapping())

    def think(self) -> Union[ThinkResponse, FinalAnswer]:
        self.messages.append({"role": "user", "content": "Think about the current question or observation and decide what to do next. Can we answer the question with the information we have? If so, respond with FinalAnswer. If not, respond with ThinkResponse."})
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=self.messages,
            response_model=Union[ThinkResponse, FinalAnswer],
        )
        
    def action(self) -> Tuple[str, ToolSpec, BaseModel]:
        if not self._tools.has_tools():
            raise RuntimeError("No tools registered for this agent")

        self.messages.append({
            "role": "user", 
            "content": "Choose and call an available tool based on the latest thoughts"
        })

        available_tool_response_models = self._tools.response_union()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.messages,
            response_model=available_tool_response_models,
        )

        # Identify which tool the language model implied by checking the response type.
        tool_name, spec = self._tools.resolve(response)
        return tool_name, spec, response
       

    def observation(self) -> Dict[str, Any]:
        self.messages.append({"role": "user", "content": "Review the outcome of your recent Action. Use it to inform your next Thought."})
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=self.messages,
            response_model=ObservationResponse,
        )

    def run(self, prompt: str) -> str:
        """Run the ReAct loop until we reach a final answer or max steps."""
        if not prompt:
            raise ValueError("Empty prompt")
        if self.client is None:
            raise RuntimeError("Instructor client is not configured")

        self.messages.append({"role": "user", "content": prompt})

        for _ in range(self.max_steps):
            # ReAct step order: Thought -> Action -> Observation.
            think_response = self.think()

            if think_response.type == "final":
                final_answer = think_response.answer.strip()
                self.messages.append({"role": "assistant", "content": f"Answer: {final_answer}"})
                return final_answer

            if think_response.type != "think":
                raise ValueError("Invalid think response")

            extracted = _maybe_extract_final_answer(think_response.thoughts)
            if extracted:
                final_answer = extracted.strip()
                self.messages.append({"role": "assistant", "content": f"Answer: {final_answer}"})
                return final_answer

            self.messages.append({"role": "assistant", "content": think_response.thoughts})

            if not self._tools.has_tools():
                raise RuntimeError("No tools registered for this agent")

            tool_name, tool_spec, action_payload = self.action()
            payload_dict = action_payload.model_dump()
            self.messages.append({"role": "assistant", "content": f"Action: {tool_name} -> {payload_dict}"})

            handler = tool_spec.handler
            tool_call_result = handler(payload_dict)
            tool_message = _stringify_message_content(tool_call_result)
            self.messages.append({"role": "tool", "content": tool_message})

            observation_response = self.observation()
            self.messages.append({"role": "assistant", "content": observation_response.observation})

        raise RuntimeError("Reached max steps without a final answer")


if __name__ == "__main__":
    # Set the OpenRouter API key in the environment before running this quick demo.
    agent = ReActAgent()
    agent.add_tool(build_calculator_tool())
    agent.run("Find the value of 42 + 3")
    print(agent.messages)
