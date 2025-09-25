
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


def _build_client(api_key: str, provider_id: str, mode: Optional[Mode]) -> Any:
    """Create an Instructor client using the OpenRouter settings."""
    return openrouter_client.build_client(api_key=api_key, provider_id=provider_id, mode=mode)

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
        base_profile = resolve_profile()
        profile = base_profile if openrouter_client.is_openrouter(base_profile.provider_id) else resolve_profile("openrouter")
        resolved_provider = (
            provider_id
            or profile.provider_id
            or openrouter_client.DEFAULT_OPENROUTER_PROVIDER
        )
        if not openrouter_client.is_openrouter(resolved_provider):
            resolved_provider = profile.provider_id or openrouter_client.DEFAULT_OPENROUTER_PROVIDER

        resolved_api_key = (
            api_key
            or _resolve_api_key()
            or profile.default_api_key
        )
        if not resolved_api_key:
            raise ValueError("api_key is required for ReActAgent")

        explicit_model = model or profile.model_id
        resolved_model = explicit_model if explicit_model else _derive_model_id(None, resolved_provider)

        if openrouter_client.is_openrouter(resolved_provider):
            fallback_mode = profile.mode or Mode.TOOLS
            resolved_mode = openrouter_client.normalize_mode(fallback_mode)
        else:
            resolved_mode = profile.mode

        mode_label = resolved_mode.name if resolved_mode else "provider default"
        print(f"ReActAgent provider: {resolved_provider}")
        print(f"ReActAgent model: {resolved_model}")
        print(f"ReActAgent mode: {mode_label}")

        self.client = _build_client(api_key=resolved_api_key, provider_id=resolved_provider, mode=resolved_mode)
        self.model_id = resolved_model
        self.temperature = temperature
        self.max_steps = max(1, int(max_steps))
        self._tools = ToolRegistry()
        self._system_prompt_template = system_prompt
        self._system_prompt = self._render_system_prompt()
        # Track the evolving plan scratchpad between LLM calls.
        self._scratchpad: Dict[str, Any] = {"request": "", "steps": [], "final_answer": ""}

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
        self._system_prompt = prompt

    def _render_system_prompt(self) -> str:
        """Render the prompt template with the current tool block."""
        print(f"Rendering system prompt: {self._system_prompt_template}")
        tool_name_and_descriptions = self._tools.tool_names_and_descriptions()
        print(f"Tool name and descriptions: {tool_name_and_descriptions}")
        return render_system_prompt(self._system_prompt_template, tool_name_and_descriptions)

    def _append_plan(self, thoughts: str) -> None:
        """Add a fresh plan entry for the current step."""
        step_number = len(self._scratchpad["steps"]) + 1
        self._scratchpad["steps"].append({
            "step": step_number,
            "plan": thoughts,
            "tool": "",
            "tool_result": "",
            "observation": "",
        })

    def _update_last_step(
        self,
        *,
        tool: Optional[str] = None,
        tool_result: Optional[str] = None,
        observation: Optional[str] = None,
    ) -> None:
        """Store extra data on the most recent scratchpad step."""
        if not self._scratchpad["steps"]:
            raise RuntimeError("No scratchpad step available")
        entry = self._scratchpad["steps"][-1]
        if tool is not None:
            entry["tool"] = tool
        if tool_result is not None:
            entry["tool_result"] = tool_result
        if observation is not None:
            entry["observation"] = observation

    def _render_scratchpad(self) -> str:
        """Create a simple text view of the current scratchpad."""
        request = self._scratchpad.get("request", "")
        blocks: List[str] = []
        if request:
            blocks.append(f"Request: {request}")
        for entry in self._scratchpad.get("steps", []):
            step_lines = [f"Step {entry['step']}"]
            if entry.get("plan"):
                step_lines.append(f"Plan: {entry['plan']}")
            if entry.get("tool"):
                step_lines.append(f"Tool: {entry['tool']}")
            if entry.get("tool_result"):
                step_lines.append(f"Tool Result: {entry['tool_result']}")
            if entry.get("observation"):
                step_lines.append(f"Observation: {entry['observation']}")
            blocks.append("\n".join(step_lines))
        final_answer = self._scratchpad.get("final_answer", "")
        if final_answer:
            blocks.append(f"Final Answer: {final_answer}")
        return "\n\n".join(blocks) if blocks else "Scratchpad is empty."

    def _build_messages(self, instruction: str) -> List[Dict[str, str]]:
        """Return a fresh message list that includes the scratchpad."""
        scratchpad_text = self._render_scratchpad()
        tool_overview = self._tools.tool_names_and_descriptions()
        user_chunks = [instruction, "", "Scratchpad:", scratchpad_text]
        if tool_overview:
            user_chunks.extend(["", "Tools:", tool_overview])
        user_content = "\n".join(user_chunks)
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def think(self) -> ThinkResponse:
        """Plan the next move using the scratchpad for context."""
        messages = self._build_messages(
            "Think about the request or the last observation and plan the next step."
        )
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            response_model=ThinkResponse, 
        )
        self._append_plan(response.thoughts)
        return response

    def action(self) -> Tuple[str, ToolSpec, BaseModel]:
        """Select the next tool call."""
        if not self._tools.has_tools():
            raise RuntimeError("No tools registered for this agent")
        messages = self._build_messages(
            "Respond with the single tool you want to call for the next step."
        )
        available_tool_response_models = self._tools.response_union()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            response_model=available_tool_response_models,
        )
        tool_name, spec = self._tools.resolve(response)
        payload_dict = response.model_dump()
        args_text = ", ".join(f"{key}={value}" for key, value in payload_dict.items()) or "no arguments"
        self._update_last_step(tool=f"{tool_name}({args_text})")
        return tool_name, spec, response

    def observation(self, tool_result: Dict[str, Any]) -> ObservationResponse:
        """Interpret the tool result and update the scratchpad."""
        self._update_last_step(tool_result=str(tool_result))
        messages = self._build_messages(
            "Review the latest tool result and explain what it means for the plan."
        )
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            response_model=ObservationResponse,
        )
        self._update_last_step(observation=response.observation)
        return response

    def _log_final_scratchpad(self) -> None:
        """Print the final scratchpad snapshot once the run ends."""
        final_snapshot = self._render_scratchpad()
        print("Final scratchpad state:\n")
        print(final_snapshot)

    def run(self, prompt: str) -> str:
        """Run the ReAct loop until we reach a final answer or max steps."""
        if not prompt:
            raise ValueError("Empty prompt")
        if self.client is None:
            raise RuntimeError("Instructor client is not configured")

        # Start a fresh scratchpad for this run.
        self._scratchpad = {"request": prompt, "steps": [], "final_answer": ""}

        for _ in range(self.max_steps):
            think_response = self.think()
            print(f"Think response: {think_response}")
            if think_response.is_final:
                final_answer = think_response.thoughts
                self._scratchpad["final_answer"] = final_answer
                # Share a readable copy of the scratchpad before returning.
                self._log_final_scratchpad()
                return final_answer
            if not self._tools.has_tools():
                self._log_final_scratchpad()
                raise RuntimeError("No tools registered for this agent")
            tool_name, tool_spec, action_payload = self.action()
            payload_dict = action_payload.model_dump()
            tool_output = tool_spec.handler(payload_dict)
            observation_response = self.observation(tool_output)
            print(f"Observation response: {observation_response}")

        self._log_final_scratchpad()
        raise RuntimeError("Reached max steps without a final answer")


if __name__ == "__main__":
    # Set the OpenRouter API key in the environment before running this quick demo.
    agent = ReActAgent()
    agent.add_tool(build_calculator_tool())
    agent.run("Find the exact value of log(1234234) and then calculate the square root of the result")
    print(agent._render_scratchpad())
