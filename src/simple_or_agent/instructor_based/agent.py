# src/simple_or_agent/instructor_based/agent.py
# Implements an Instructor-powered ReAct agent loop with tool support.
# Exists to offer a minimal OpenRouter-friendly orchestrator in this codebase.
# RELEVANT FILES: src/simple_or_agent/instructor_based/openrouter_client.py, src/simple_or_agent/instructor_based/provider_profiles.py, src/simple_or_agent/instructor_based/calculator_tool.py

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ''}:
    # Ensure direct execution can resolve the src/ package namespace.
    project_src = Path(__file__).resolve().parent.parent.parent
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

from simple_or_agent.instructor_based import simple_client

from typing import Any, Dict, List, Optional, Tuple

from instructor import Mode
from pydantic import BaseModel

from simple_or_agent.instructor_based.prompt_manager import (
    DEFAULT_REACT_SYSTEM_PROMPT_TEMPLATE,
    render_system_prompt,
)
from simple_or_agent.instructor_based import openrouter_client
from simple_or_agent.instructor_based.calculator_tool import build_calculator_tool
from simple_or_agent.instructor_based.provider_profiles import resolve_profile
from simple_or_agent.instructor_based.tools import ToolRegistry, ToolSpec

def _resolve_api_key() -> Optional[str]:
    """Read the preferred API key from the environment."""
    return openrouter_client.resolve_api_key_from_env()

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


def _build_simple_client(
    api_key: str,
) -> Any:
    return simple_client.build_chat_client(
        api_key=api_key,
    )

class ThinkResponse(BaseModel):
    """Thought response"""
    is_final: bool
    thoughts: str

class ObservationResponse(BaseModel):
    """Observation response"""
    observation: str

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
        resolved_provider = (
            provider_id
            or profile.provider_id
            or openrouter_client.DEFAULT_OPENROUTER_PROVIDER
        )
        if not openrouter_client.is_openrouter(resolved_provider):
            # Enforce the OpenRouter contract even when a non-OpenRouter id slips in.
            resolved_provider = profile.provider_id or openrouter_client.DEFAULT_OPENROUTER_PROVIDER

        resolved_api_key = (
            api_key
            or _resolve_api_key()
            or profile.default_api_key
        )
        if not resolved_api_key:
            raise ValueError("api_key is required for ReActAgent")

        explicit_model = model or profile.model_id
        if explicit_model:
            resolved_model = explicit_model
        else:
            resolved_model = _derive_model_id(None, resolved_provider)

        if openrouter_client.is_openrouter(resolved_provider):
            fallback_mode = profile.mode or Mode.TOOLS
            resolved_mode = openrouter_client.normalize_mode(fallback_mode)
        else:
            resolved_mode = profile.mode

        mode_label = resolved_mode.name if resolved_mode else "provider default"
        print(f"ReActAgent provider: {resolved_provider}")
        print(f"ReActAgent model: {resolved_model}")
        print(f"ReActAgent mode: {mode_label}")

        self.client = _build_client(
            api_key=resolved_api_key,
            provider_id=resolved_provider,
            mode=resolved_mode,
        )

        self.simple_client = _build_simple_client(
            api_key=resolved_api_key,
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
        print(f"Refreshing system prompt: {prompt}")
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = prompt
            return
        # Insert a fresh system message if the log somehow lost the original.
        self.messages.insert(0, {"role": "system", "content": prompt})

    def _render_system_prompt(self) -> str:
        """Render the prompt template with the current tool block."""
        print(f"Rendering system prompt: {self._system_prompt_template}")
        tool_names = self._tools.tool_names()
        print(f"Tool names: {tool_names}")
        return render_system_prompt(self._system_prompt_template, tool_names)

    def think(self) -> ThinkResponse:

        print(f"Thinking about the current user question or observation.")
        print(f"Messages: {self.messages}")
        self.messages.append({
            "role": "user",
            "content": (
                "Think about current user question or last observation and plan next steps. We have following tools available: " + ", ".join(self._tools.tool_names()) + ". Do you need to call any tool on next step or do you have the answer already? Respond using ThinkResponse. You are allowed to call ThinkResponse only once. You will be allowed to call available tools on next step."
            ),
        })
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=self.messages,
            response_model=ThinkResponse,
        )
        
    def action(self) -> Tuple[str, ToolSpec, BaseModel]:
        if not self._tools.has_tools():
            raise RuntimeError("No tools registered for this agent")

        print(f"Actioning the current user question or observation.")
        print(f"Messages: {self.messages}")

        self.messages.append({
            "role": "user",
            "content": (
                "Respond with the tool you want to call. You are allowed to call only one tool on this step."
            ),
        })

        available_tool_response_models = self._tools.response_union()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.messages,
            response_model=available_tool_response_models,
        )

        print(f"Action response: {response}")

        # Identify which tool the language model implied by checking the response type.
        tool_name, spec = self._tools.resolve(response)
        return tool_name, spec, response
       

    def observation(self) -> ObservationResponse:

        print(f"Observing the current user question or observation.")
        print(f"Messages: {self.messages}")

        self.messages.append({
            "role": "user",
            "content": (
                "Review the latest tool result and explain what it means."
            ),
        })
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

            print(f"Think response: {think_response}")

            if think_response.is_final:
                final_answer = think_response.thoughts
                self.messages.append({"role": "assistant", "content": "Final answer: " + final_answer})
                return final_answer

            self.messages.append({"role": "assistant", "content": "Thought: " + think_response.thoughts})

            if not self._tools.has_tools():
                raise RuntimeError("No tools registered for this agent")

            tool_name, tool_spec, action_payload = self.action()
            payload_dict = action_payload.model_dump()
            self.messages.append({"role": "assistant", "content": f"Action: {tool_name} -> {payload_dict}"})

            observation_response = self.observation()
            self.messages.append({"role": "assistant", "content": "Observation: " + observation_response.observation})

        raise RuntimeError("Reached max steps without a final answer")


if __name__ == "__main__":
    # Set the OpenRouter API key in the environment before running this quick demo.
    agent = ReActAgent()
    agent.add_tool(build_calculator_tool())
    agent.run("Find the exact value of log(1234234)")
    print(agent.messages)   
