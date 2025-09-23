# src/simple_or_agent/instructor_based/prompt_manager.py
# Stores template helpers for the Instructor-based ReAct system prompt.
# Exists so agents can render up-to-date tool listings for the LLM.
# RELEVANT FILES: src/simple_or_agent/instructor_based/agent.py, src/simple_or_agent/instructor_based/tools.py, src/simple_or_agent/instructor_based/provider_profiles.py

from __future__ import annotations

from typing import Any, List, Mapping

DEFAULT_REACT_SYSTEM_PROMPT_TEMPLATE = """
You work in a simple loop with Thought, Action, and Observation.

When asked to think, call ThinkResponse with a 'thoughts' field. Do not call any task tools during that step.

When you already know the solution, call FinalAnswer with an 'answer' field. Never send plain text.

When asked to take an action, call exactly one tool from the list. Supply every required field in the JSON you return.

After a tool runs, describe the result by calling ObservationResponse with an 'observation' field.

When the task is complete, finish with FinalAnswer.

Available tools:
<tools>
{tool_block}
</tools>

Example flow:
1. ThinkResponse(thoughts="I will calculate the expression.")
2. calculate(expr="42 + 3")
3. ObservationResponse(observation="The calculation returned 45.")
4. FinalAnswer(answer="45")

Start from the given question and follow this structure.
"""

# Backwards-compatible alias for older imports.
DEFAULT_REACT_SYSTEM_PROMPT = DEFAULT_REACT_SYSTEM_PROMPT_TEMPLATE


def format_tool_block(tools: Mapping[str, Any]) -> str:
    """Return a printable block that lists the currently registered tools."""
    if not tools:
        return ""

    rendered: List[str] = []
    for spec in tools.values():
        params = getattr(spec, "parameters", None)
        param_names = ", ".join(params.keys()) if params else "no parameters"
        rendered.append(
            f"- {getattr(spec, 'name', '')}: {getattr(spec, 'description', '')} (params: {param_names})"
        )
    return "\n".join(rendered)


def render_system_prompt(template: str, tools: Mapping[str, Any]) -> str:
    """Render the prompt template with an optional tool block."""
    tool_block = format_tool_block(tools)

    if "{tool_block}" in template:
        return template.format(tool_block=tool_block)

    if tool_block:
        if "<tools>" in template and "</tools>" in template:
            prefix, remainder = template.split("<tools>", 1)
            _, suffix = remainder.split("</tools>", 1)
            return f"{prefix}<tools>\n{tool_block}\n</tools>{suffix}"
        return f"{template.rstrip()}\n\n<tools>\n{tool_block}\n</tools>"

    if "<tools>" in template and "</tools>" in template:
        prefix, remainder = template.split("<tools>", 1)
        _, suffix = remainder.split("</tools>", 1)
        return f"{prefix}{suffix}"

    return template
