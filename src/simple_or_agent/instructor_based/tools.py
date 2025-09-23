# src/simple_or_agent/instructor_based/tools.py
# Defines tool specifications and registry for Instructor-powered agents.
# Exists to help agents declare and resolve tool calls consistently.
# RELEVANT FILES: src/simple_or_agent/instructor_based/agent.py, src/simple_or_agent/instructor_based/prompt_manager.py, src/simple_or_agent/instructor_based/lmstudio_client.py

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type, Union
from typing_extensions import List

from pydantic import BaseModel


class ToolSpec(BaseModel):
    """Describe a single tool the agent may call."""

    name: str
    description: str
    response_model: Optional[Type[BaseModel]] = None
    handler: Callable[[Dict[str, Any]], Any]
    args_model: Optional[Type[BaseModel]] = None
    parameters: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    def model_class(self) -> Type[BaseModel]:
        """Return whichever model definition the tool exposes."""
        model = self.response_model or self.args_model
        if model is None:
            raise ValueError("ToolSpec requires response_model or args_model")
        return model


class ToolRegistry:
    """Store tools and help the agent resolve tool calls."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def add(self, tool: ToolSpec) -> None:
        """Register or replace a tool."""
        self._tools[tool.name] = tool

    def remove(self, name: str) -> None:
        """Remove a tool without raising if it is missing."""
        self._tools.pop(name, None)

    def as_mapping(self) -> Mapping[str, ToolSpec]:
        """Expose a read-only view for prompt rendering."""
        return MappingProxyType(self._tools)

    def has_tools(self) -> bool:
        """Return True when at least one tool is registered."""
        return bool(self._tools)

    def response_union(self) -> Type[BaseModel]:
        """Return the correct Pydantic response union for the current tool set."""
        if not self._tools:
            raise RuntimeError("No tools registered")
        models = [spec.model_class() for spec in self._tools.values()]
        return models[0] if len(models) == 1 else Union[*models]

    def tool_names(self) -> List[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    def resolve(self, payload: BaseModel) -> Tuple[str, ToolSpec]:
        """Identify which tool produced the payload."""
        for name, spec in self._tools.items():
            if isinstance(payload, spec.model_class()):
                return name, spec
        raise ValueError("Received tool payload that matches no registered tool")


__all__ = ["ToolRegistry", "ToolSpec"]
