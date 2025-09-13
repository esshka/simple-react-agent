# OpenRouter LLM Client

Minimal, transport-focused client for OpenRouter Chat Completions with clean helpers and optional utilities.

- File: `bo_ag/llm/openrouter_client.py`
- Parsers: `bo_ag/llm/response_parsers.py`
- Demo CLI: `scripts/chat_loop.py`
- REAct Agent: `bo_ag/llm/react_agent.py`

## Features

- Robust HTTP client with instance-configured retries and timeouts
- Optional app headers (`HTTP-Referer`, `X-Title`)
- Reasoning support (`reasoning` payload + extraction helper)
- Tool calling support (function tools, tool results, follow-up call helper)
- Minimal helpers to extract content, reasoning, and tool calls
- Strict validation of inputs and consistent exceptions

## Installation & Env

- Requires `OPENROUTER_API_KEY` exported or present in `.env`
- Uses `requests` and `tenacity` (already in project)
- The demo `scripts/chat_loop.py` will load `.env` automatically (via `python-dotenv` if available; otherwise a silent no-op)

```bash
export OPENROUTER_API_KEY="sk-or-..."
# or put it into .env at the project root
```

## Client Usage

```python
from bo_ag.llm.openrouter_client import OpenRouterClient

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2 + 2?"},
]

client = OpenRouterClient(app_name="bo-ag", app_url="https://github.com/bo-ag/trading-system")
resp = client.complete_chat(
    messages=messages,
    model="openai/gpt-4o-mini",
    temperature=0.2,
)
text = client.extract_content(resp)
print(text)
client.close()
```

### Reasoning

Request reasoning and display it if the model supports it (e.g., `openai/o3-mini`).

```python
resp = client.complete_chat(
    messages=messages,
    model="openai/o3-mini",
    reasoning={"effort": "high"},
)
reasoning = client.extract_reasoning(resp)  # str | None
print(reasoning)
```

### Tool Calling (Functions)

Provide JSON schema tools and handle a tool call round, then ask for the final answer.

```python
# 1) Provide tools
calc_tool = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate basic arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}

# 2) Ask the model
resp = client.complete_chat(
    messages=[{"role": "user", "content": "Compute (2+3)*4"}],
    model="openai/o3-mini",
    tools=[calc_tool],
)

# 3) Execute tool calls locally
messages = [{"role": "user", "content": "Compute (2+3)*4"}]
for call in client.get_tool_calls(resp):
    args = call.get("function", {}).get("arguments", "{}")
    # Evaluate args and run your tool here...
    result_str = "20"  # example result
    # 4) Append tool result message
    messages.append(client.make_tool_result(call.get("id", ""), result_str))

# 5) Ask again for the final assistant reply
resp = client.complete_chat(messages=messages, model="openai/o3-mini")
print(client.extract_content(resp))
```

## Helpers

- `extract_content(response) -> str`: returns first choice text content or raises `OpenRouterError`
- `extract_reasoning(response) -> Optional[str]`: best-effort extraction of `choices[0].message.reasoning`
- `get_tool_calls(response) -> list[dict]`: returns assistant `tool_calls` list or `[]`
- `make_tool_result(tool_call_id, content) -> dict`: build a `role="tool"` message for follow-up

## Exceptions

- `OpenRouterError`: base client error, used for validation and non-retryable 4xx
- `OpenRouterAPIError`: retryable API/transport errors, may include `status_code` and `response_data`

## Retries & Timeouts

- Configure on client creation:
  - `timeout_s` (default 30)
  - `max_retries` (default 3)
  - `retry_base_wait` (default 1)
  - `retry_max_wait` (default 60)

## JSON Parsing Utility (optional)

`bo_ag/llm/response_parsers.py` contains `parse_json_response(content, strict=False)` with multiple extraction strategies (direct, fenced blocks, balanced braces, and safe fallbacks). This is intentionally separate from the transport client.

```python
from bo_ag.llm.response_parsers import parse_json_response

content = client.extract_content(resp)
parsed = parse_json_response(content)
```

## Demo: Chat Loop

`scripts/chat_loop.py` provides an interactive CLI for quick testing.

Key flags:
- `--model` to select the model
- `--reasoning-effort {low,medium,high}` and `--show-reasoning`
- `--with-calculator` to expose the `calculate` tool (single tool-call round)
- `--parse-json` to parse assistant content via `parse_json_response`
- `--verbose` to print full raw API responses
- `--no-color` to disable ANSI colors

Examples:
```bash
python3 scripts/chat_loop.py --model openai/o3-mini --reasoning-effort high --show-reasoning
python3 scripts/chat_loop.py --model openai/o3-mini --with-calculator --verbose
```

## REAct Agent

`bo_ag/llm/react_agent.py` provides a small, reliable REAct loop abstraction on top of the transport client. It keeps memory (optional), supports tool-calling, and encourages a plan-first style via the system prompt.

```python
from bo_ag.llm.openrouter_client import OpenRouterClient
from bo_ag.llm.react_agent import ReActAgent, ToolSpec

client = OpenRouterClient(app_name="bo-ag")
agent = ReActAgent(
    client,
    model="openai/o3-mini",            # or any tool-capable model
    reasoning_effort="high",            # optional
    max_rounds=6,
    max_tool_iters=3,
)

# Safe calculator tool
def safe_calc(args):
    import ast, operator as op
    expr = str(args.get("expression", ""))
    allowed = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg}
    def _eval(node):
        if isinstance(node, ast.Num): return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub,): return allowed[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed: return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.Expression): return _eval(node.body)
        raise ValueError("Unsupported expression")
    return str(_eval(ast.parse(expr, mode="eval")))

agent.add_tool(ToolSpec(
    name="calculate",
    description="Evaluate arithmetic expression and return a number",
    parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    handler=safe_calc,
))

result = agent.ask("What is (2+3)*4? Use tools if helpful.")
print(result.content)
print(result.reasoning)
```

Notes:
- `ReActAgent.ask()` performs a bounded tool-calling loop and returns the final assistant content plus optional reasoning and usage.
- Register tools via `ToolSpec`; handlers receive a dict of parsed arguments and must return a string or JSON-serializable value.
- Use `keep_history=False` or call `agent.reset()` for stateless turns.

## Notes

- This client does not implement streaming; can be added if needed.
- Keep business/domain parsing outside the client (use the parsers module or your own logic).
- File sizes are kept under 300 lines per project guideline.
