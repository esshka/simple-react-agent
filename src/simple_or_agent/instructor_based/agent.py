# src/simple_or_agent/instructor_based/agent.py
# Implements an Instructor-powered ReAct agent loop with tool support.
# Exists to offer a minimal LMStudio-friendly orchestrator in this codebase.
# RELEVANT FILES: src/simple_or_agent/instructor_based/instructor_client.py, src/simple_or_agent/instructor_based/prompt_manager.py, src/simple_or_agent/instructor_based/tools.py

from __future__ import annotations

import ast
import json
import operator as op
from typing import Any, Dict, List, Optional, Tuple, Literal, Union

from pydantic import BaseModel

from simple_or_agent.instructor_based.prompt_manager import (
    DEFAULT_REACT_SYSTEM_PROMPT_TEMPLATE,
    render_system_prompt,
)
from simple_or_agent.instructor_based import instructor_client as instructor_helpers
from simple_or_agent.instructor_based.instructor_client import build_instructor_client
from simple_or_agent.instructor_based.provider_profiles import resolve_profile
from simple_or_agent.instructor_based.tools import ToolRegistry, ToolSpec


def _derive_model_id(model: Optional[str], provider_id: Optional[str]) -> str:
    """Return the completion model id based on the hints provided."""
    if model:
        return model
    if provider_id:
        if provider_id.startswith("openrouter/"):
            return provider_id.split("/", 1)[1]
        if provider_id.startswith("openai/"):
            return provider_id.split("/", 1)[1]
        return provider_id
    return "qwen/qwen3-next-80b-a3b-instruct"


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
    markers = [
        "answer:",
        "answer is",
        "final answer:",
        "final answer is",
        "final answer",
        "final result",
    ]
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
        base_url: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> None:
        profile = resolve_profile()  # Load provider defaults from providers.ini.
        env_provider = instructor_helpers._resolve_provider()  # Let env override the provider.
        env_base_url = instructor_helpers._resolve_base_url()  # Let env override the base URL.
        env_mode = instructor_helpers._resolve_mode()  # Let env override the mode.
        using_profile_defaults = (
            provider_id is None and base_url is None and env_provider is None and env_base_url is None
        )  # Only rely on the profile when nothing else is set.
        resolved_provider = provider_id or env_provider or profile.provider_id
        if base_url is not None:
            resolved_base_url = base_url
        elif env_base_url is not None:
            resolved_base_url = env_base_url
        elif using_profile_defaults:
            resolved_base_url = profile.base_url
        else:
            resolved_base_url = None

        resolved_api_key = (
            api_key
            or instructor_helpers._resolve_api_key()
            or (profile.default_api_key if using_profile_defaults else None)
        )
        if not resolved_api_key:
            raise ValueError("api_key is required for ReActAgent")

        fallback_model = _derive_model_id(model, resolved_provider)
        if using_profile_defaults and not model and resolved_base_url:
            resolved_model = instructor_helpers._discover_model_id(
                resolved_base_url,
                resolved_api_key,
                fallback_model,
            )
        else:
            resolved_model = fallback_model

        build_kwargs: Dict[str, Any] = {"api_key": resolved_api_key, "provider_id": resolved_provider, "base_url": resolved_base_url}
        if using_profile_defaults and profile.mode:
            build_kwargs["mode"] = profile.mode
        elif env_mode is not None:
            build_kwargs["mode"] = env_mode

        self.client = build_instructor_client(**build_kwargs)
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


class CalcArgs(BaseModel):
    expr: str

OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

def _eval_expression(node: ast.AST) -> float:
    """Evaluate a safe arithmetic AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_expression(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left = _eval_expression(node.left)
        right = _eval_expression(node.right)
        return OPS[type(node.op)](left, right)
    raise ValueError("Unsupported expression")


def calculate(raw_args: Dict[str, Any]) -> Dict[str, Any]:
    args = CalcArgs(**raw_args)
    parsed = ast.parse(args.expr, mode="eval")
    value = _eval_expression(parsed.body)
    return {"expr": args.expr, "value": value}


if __name__ == "__main__":
    # The default constructor now targets LMStudio, so no explicit key is required here.
    agent = ReActAgent()
    calculate_tool = ToolSpec(name="calculate", description="Evaluate a mathematical expression.", response_model=CalcArgs, handler=calculate)
    agent.add_tool(calculate_tool)
    agent.run("Find the value of 42 + 3")
    print(agent.messages)
