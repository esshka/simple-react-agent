# src/simple_or_agent/instructor_based/prompt_manager.py
# Stores template helpers for the Instructor-based ReAct system prompt.
# Exists so agents can render up-to-date tool listings for the LLM.
# RELEVANT FILES: src/simple_or_agent/instructor_based/agent.py, src/simple_or_agent/instructor_based/tools.py, src/simple_or_agent/instructor_based/provider_profiles.py

from __future__ import annotations

from typing import Any, List, Mapping

DEFAULT_REACT_SYSTEM_PROMPT_TEMPLATE = """
You are an agent that iterates in the following loop:
1. Thought: Reflect on the current question or observation and decide what to do next.
2. Action: Choose and execute an available tool (from the tool list below) in this format: Action: <tool_name>: <input>
   - After specifying an Action, pause and wait for the Observation.
3. Observation: Review the outcome of your recent Action. Use it to inform your next Thought.

At any time, if you have fully answered the user's question, output in the format: Answer: <your_answer_here>


Available tools:
<tools>
{tool_block}
</tools>

Example interaction:
Question: What is the capital of France multiplied by the number of hours in a day?
Thought: I need to find the capital of France.
Action: lookup: capital of France
PAUSE

Observation: The capital of France is Paris.
Thought: Now I need to multiply something. That doesn't make sense—capitals are strings.
Answer: The capital of France is Paris.

(If a calculation makes sense, Action: calculate: <expression> is used and integrated similarly.)

Start the loop using the question provided. Always follow this structure.
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
