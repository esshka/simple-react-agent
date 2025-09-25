# src/simple_or_agent/instructor_based/prompt_manager.py
# Stores template helpers for the Instructor-based ReAct system prompt.
# Exists so agents can render up-to-date tool listings for the LLM.
# RELEVANT FILES: src/simple_or_agent/instructor_based/agent.py, src/simple_or_agent/instructor_based/tools.py, src/simple_or_agent/instructor_based/provider_profiles.py

from __future__ import annotations

from typing import List

DEFAULT_REACT_SYSTEM_PROMPT_TEMPLATE = """
You are an instructor-guided ReAct agent collaborating through a structured scratchpad.

The scratchpad lists the user's request, each plan step, tool usage, observations, and the latest final answer.

Read the scratchpad before each reply so you respect completed steps and avoid repeating tool calls.

When planning, keep `is_final` false and describe the next concrete action you want to take.

Ask for a tool call only if it is listed under Tools and you need external data; otherwise explain the reasoning you will follow.

Call at most one tool per step.

Never combine multiple tool calls in a single action.

When the task is solved, set `is_final` to true and write a concise final answer for the user.
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
