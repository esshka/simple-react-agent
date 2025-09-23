# src/simple_or_agent/instructor_based/prompt_manager.py
# Stores template helpers for the Instructor-based ReAct system prompt.
# Exists so agents can render up-to-date tool listings for the LLM.
# RELEVANT FILES: src/simple_or_agent/instructor_based/agent.py, src/simple_or_agent/instructor_based/tools.py, src/simple_or_agent/instructor_based/provider_profiles.py

from __future__ import annotations

from typing import List

DEFAULT_REACT_SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant.
"""

# Backwards-compatible alias for older imports.
DEFAULT_REACT_SYSTEM_PROMPT = DEFAULT_REACT_SYSTEM_PROMPT_TEMPLATE


def append_tool_names(template: str, tool_names: List[str]) -> str:
    tools_prompt = "You have the following tools available: " + ", ".join(tool_names)
    return template + "\n\n" + tools_prompt


def render_system_prompt(template: str, tool_names: List[str]) -> str:
    # print(f"Rendering system prompt: {template}")
    # print(f"Tool names: {tool_names}")
    # template = append_tool_names(template, tool_names)
    return template
