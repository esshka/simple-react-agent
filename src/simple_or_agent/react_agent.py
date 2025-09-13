"""REAct-style agent abstraction on top of OpenRouterClient (<=300 LOC)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import os
import json
import logging

from .openrouter_client import (
    OpenRouterClient,
    OpenRouterError,
    OpenRouterAPIError,
)

logger = logging.getLogger(__name__)
@dataclass
class ToolSpec:
    """Python-executed tool with JSON schema for LLM tool-calls."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]

    def to_openrouter(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

@dataclass
class AskResult:
    """Outcome of a single ask() REAct run."""

    content: str
    reasoning: Optional[str]
    usage: Optional[Dict[str, Any]]
    messages: List[Dict[str, Any]]


class ReActAgent:
    """Plan-then-Act loop with optional tool-calling."""

    DEFAULT_SYSTEM = (
        "You are an expert assistant that uses a Plan-then-Act approach. "
        "Start with a brief Plan as bullet points. When needed, call tools to gather facts. "
        "Reflect briefly if a step changes your plan. Provide the Final Answer last."
    )

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
    ) -> None:
        self.client = client
        # Default model from env if not provided
        self.model = model or os.getenv("MODEL_ID", "qwen/qwen3-next-80b-a3b-thinking")
        self.temperature = temperature
        self.keep_history = keep_history
        self.max_rounds = max(1, int(max_rounds))
        self.max_tool_iters = max(0, int(max_tool_iters))
        self.response_format = response_format
        self.reasoning_effort = reasoning_effort
        self.parallel_tool_calls = parallel_tool_calls
        self.tool_choice = tool_choice

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self._messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            self._messages.append({"role": "system", "content": self.system_prompt})

        self._tools: Dict[str, ToolSpec] = {}

    # Memory management
    def reset(self) -> None:
        """Clear conversation history while keeping the system prompt."""
        self._messages = []
        if self.system_prompt:
            self._messages.append({"role": "system", "content": self.system_prompt})

    # Tools
    def add_tool(self, tool: ToolSpec) -> None:
        if not tool.name or not callable(tool.handler):
            raise OpenRouterError("Tool must have a name and a callable handler")
        self._tools[tool.name] = tool

    def remove_tool(self, name: str) -> None:
        self._tools.pop(name, None)

    def _tool_list_for_api(self) -> Optional[List[Dict[str, Any]]]:
        if not self._tools:
            return None
        return [t.to_openrouter() for t in self._tools.values()]

    # Core loop
    def ask(self, prompt: str) -> AskResult:
        """Run a REAct loop for a single user prompt.

        Returns the final assistant content and metadata. Conversation
        history is retained unless keep_history=False.
        """
        if not prompt:
            raise OpenRouterError("Empty prompt")

        # Build initial messages
        if not self.keep_history:
            self.reset()
        messages = list(self._messages)
        messages.append({"role": "user", "content": prompt})

        tools = self._tool_list_for_api()
        reasoning = {"effort": self.reasoning_effort} if self.reasoning_effort else None

        last_resp: Optional[Dict[str, Any]] = None
        rounds = 0

        while True:
            rounds += 1
            if rounds > self.max_rounds:
                raise OpenRouterError(
                    f"Exceeded max_rounds={self.max_rounds} without final answer"
                )

            # Ask the model
            try:
                resp = self.client.complete_chat(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    response_format=self.response_format,
                    reasoning=reasoning,
                    tools=tools,
                    tool_choice=self.tool_choice if tools is not None else None,
                    parallel_tool_calls=self.parallel_tool_calls,
                )
            except (OpenRouterAPIError, OpenRouterError) as e:
                raise e

            last_resp = resp
            # Tool-calling phase (bounded iterations)
            tool_iters = 0
            calls = self.client.get_tool_calls(resp)
            if calls:
                try:
                    m = (resp.get("choices") or [{}])[0].get("message")
                    if isinstance(m, dict):
                        messages.append(m)
                except Exception:
                    pass
            if not calls:
                # Fallback: some models emit inline <tools>{...}</tools> blocks in content
                try:
                    content_for_tools = self.client.extract_content(resp)
                except Exception:
                    content_for_tools = None
                if content_for_tools:
                    calls = self._parse_inline_tool_calls(content_for_tools)
            while calls and tool_iters < self.max_tool_iters:
                for call in calls:
                    tool_iters += 1
                    func_info = call.get("function", {})
                    name = func_info.get("name")
                    args_raw = func_info.get("arguments")
                    try:
                        args: Dict[str, Any]
                        if isinstance(args_raw, str):
                            args = json.loads(args_raw) if args_raw else {}
                        elif isinstance(args_raw, dict):
                            args = args_raw
                        else:
                            args = {}
                    except Exception as e:  # invalid JSON
                        args = {"_parse_error": str(e)}

                    result: str
                    if name in self._tools:
                        try:
                            out = self._tools[name].handler(args)
                            if isinstance(out, str):
                                result = out
                            else:
                                result = json.dumps(out, ensure_ascii=False)
                        except Exception as e:
                            result = f"error: {e}"
                    else:
                        result = "error: unknown tool"

                    messages.append(
                        self.client.make_tool_result(call.get("id", ""), result)
                    )

                # Ask again; include tools and reasoning to allow multi-round calls
                try:
                    resp = self.client.complete_chat(
                        messages=messages,
                        model=self.model,
                        temperature=self.temperature,
                        response_format=self.response_format,
                        reasoning=reasoning,
                        tools=tools,
                        tool_choice=self.tool_choice if tools is not None else None,
                        parallel_tool_calls=self.parallel_tool_calls,
                    )
                except (OpenRouterAPIError, OpenRouterError) as e:
                    raise e
                last_resp = resp
                calls = self.client.get_tool_calls(resp)
                if calls:
                    try:
                        m = (resp.get("choices") or [{}])[0].get("message")
                        if isinstance(m, dict):
                            messages.append(m)
                    except Exception:
                        pass
                if not calls:
                    try:
                        content_for_tools = self.client.extract_content(resp)
                    except Exception:
                        content_for_tools = None
                    if content_for_tools:
                        calls = self._parse_inline_tool_calls(content_for_tools)

            # If no tools were used or tool loop ended, extract final content
            try:
                content = self.client.extract_content(last_resp)
            except OpenRouterError:
                # If the model insists on more tools beyond limits
                if tool_iters >= self.max_tool_iters and last_resp is not None:
                    content = json.dumps(
                        {
                            "note": "max_tool_iters reached",
                            "assistant_message": last_resp,
                        },
                        ensure_ascii=False,
                    )
                else:
                    raise

            # Update memory with the final assistant reply
            messages.append({"role": "assistant", "content": content})
            if self.keep_history:
                self._messages = messages
            else:
                # retain only system message for stateless mode
                self.reset()

            reasoning_txt = None
            try:
                if last_resp is not None:
                    reasoning_txt = self.client.extract_reasoning(last_resp)
            except Exception:
                reasoning_txt = None

            usage = last_resp.get("usage") if isinstance(last_resp, dict) else None
            return AskResult(content=content, reasoning=reasoning_txt, usage=usage, messages=messages)

    def _parse_inline_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse <tools>/<tool_call>/<tool> JSON tags and code-fenced JSON."""
        import re
        calls: List[Dict[str, Any]] = []
        def _add(obj: Dict[str, Any]) -> None:
            fn = obj.get("name") or obj.get("tool") or (obj.get("function") or {}).get("name")
            if not fn:
                return
            ar = obj.get("arguments") or obj.get("args") or (obj.get("function") or {}).get("arguments") or {}
            if not isinstance(ar, (dict, str)):
                ar = {}
            ar_s = ar if isinstance(ar, str) else json.dumps(ar, ensure_ascii=False)
            calls.append({"id": f"inline-{len(calls)}-{fn}", "type": "function", "function": {"name": fn, "arguments": ar_s}})
        # Tag forms
        for m in re.finditer(r"<(tools|tool_call|tool)>\s*(\{.*?\})\s*</\1>", content, re.DOTALL):
            try:
                _add(json.loads(m.group(2)))
            except Exception:
                pass
        # Code-fenced JSON blocks
        for m in re.finditer(r"```[a-zA-Z0-9_-]*\s*(\{.*?\})\s*```", content, re.DOTALL):
            try:
                _add(json.loads(m.group(1)))
            except Exception:
                pass
        return calls


__all__ = ["ToolSpec", "AskResult", "ReActAgent"]
