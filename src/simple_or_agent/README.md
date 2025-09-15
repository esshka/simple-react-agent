<!--
src/simple_or_agent/README.md
Package README for the OpenRouter client and lightweight agents in this package.
Explains how to use the client and agents, and why this module exists.
RELEVANT FILES: src/simple_or_agent/openrouter_client.py,src/simple_or_agent/simple_agent.py,src/simple_or_agent/react_agent.py,src/simple_or_agent/next_agent.py
-->

# simple_or_agent — Client and Agents

Minimal, transport-focused OpenRouter client with small, readable agents on top.  Clean helpers, simple retries, reasoning support, and tool calling.  Examples included.


## What’s here

- OpenRouter client: requests + tenacity, strict validation, small helpers.  Easy to drop into any project.
- SimpleAgent: single chat loop with optional function tools.  Keeps history if you want.
- ReActAgent: Thinker → Operator → Validator orchestration.  Produces a transcript and a final answer.
- NextAgent: Plan first, then execute via ReACT.  Great for harder tasks and research flows.


## Install and setup

- Python 3.12+
- `OPENROUTER_API_KEY` in your env or `.env` at repo root
- Dependencies via Poetry

```bash
poetry install
export OPENROUTER_API_KEY="sk-or-..."   # or put into .env
```


## OpenRouterClient

File: `src/simple_or_agent/openrouter_client.py`.

Features
- Retries and backoff for transient failures.  Timeouts per request.
- Optional attribution headers: `X-Title` (app name) and `HTTP-Referer` (app URL).
- Tolerant content parsing: handles plain strings, list-based parts, and `parsed` fields.
- Reasoning support through the `reasoning` payload.  Helper to extract if present.
- Tool calling with helpers to list tool calls and append tool results.

Basic usage
```python
from src.simple_or_agent.openrouter_client import OpenRouterClient

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2 + 2?"},
]

client = OpenRouterClient(app_name="simple-or-agent")
resp = client.complete_chat(messages=messages, model="qwen/qwen3-next-80b-a3b-thinking", temperature=0.2)
print(client.extract_content(resp))
client.close()
```

Reasoning
```python
resp = client.complete_chat(
    messages=messages,
    model="qwen/qwen3-next-80b-a3b-thinking",
    reasoning={"effort": "high"},
)
print(client.extract_reasoning(resp))  # may be None
```

One-round tool calls
```python
calc_tool = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate arithmetic expression and return a number",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
}

messages = [{"role": "user", "content": "Compute (2+3)*4"}]
resp = client.complete_chat(messages=messages, model="qwen/qwen3-next-80b-a3b-thinking", tools=[calc_tool])
for call in client.get_tool_calls(resp):
    args = call.get("function", {}).get("arguments", "{}")
    result = "20"  # run your tool
    messages.append(client.make_tool_result(call.get("id", ""), result))
resp = client.complete_chat(messages=messages, model="qwen/qwen3-next-80b-a3b-thinking")
print(client.extract_content(resp))
```

Exceptions
- `OpenRouterError`: validation and non-retryable 4xx.
- `OpenRouterAPIError`: retryable transport / 5xx / 429.

Tuning
- `timeout_s` (default 30), `max_retries` (default 3), `retry_base_wait` (1), `retry_max_wait` (60).


## SimpleAgent

File: `src/simple_or_agent/simple_agent.py`.

Small chat agent with history and optional tools.  Uses `OpenRouterClient` under the hood.

```python
from src.simple_or_agent.openrouter_client import OpenRouterClient
from src.simple_or_agent.simple_agent import SimpleAgent, ToolSpec

client = OpenRouterClient(app_name="simple-agent")
agent = SimpleAgent(client, temperature=0.1, reasoning_effort="high")

def echo(args):
    return {"echo": args}

agent.add_tool(ToolSpec(
    name="echo", description="Echo input as JSON", parameters={"type": "object", "properties": {}}, handler=echo
))

res = agent.ask("Say hello using the echo tool if helpful.")
print(res.content)
```


## ReActAgent

File: `src/simple_or_agent/react_agent.py`.

Orchestrates Thinker → Operator → Validator.  Emits a transcript and a final answer.  Register tools with `ToolSpec`.

```python
from src.simple_or_agent.openrouter_client import OpenRouterClient
from src.simple_or_agent.react_agent import ReActAgent, ToolSpec

client = OpenRouterClient(app_name="react-agent")
agent = ReActAgent(client, model="qwen/qwen3-next-80b-a3b-thinking", reasoning_effort="high")

# Minimal safe calculator using AST (supports +, -, *, /, **, parentheses)
import ast, operator as op
ALLOWED = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}
def _eval(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub,):
        return ALLOWED[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED:
        return ALLOWED[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("unsupported expression")
def calc(args):
    expr = str(args.get("expression", "")).strip()
    val = _eval(ast.parse(expr, mode="eval").body)
    return {"value": val}

agent.add_tool(ToolSpec(
    name="calc",
    description="Evaluate a basic arithmetic expression",
    parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    handler=calc,
))

res = agent.ask("What is (2+3)*4? Use tools if helpful.")
print(res.content)
```


## NextAgent (plan → act)

File: `src/simple_or_agent/next_agent.py`.

Plans first, then executes via ReACT.  Useful for deeper tasks and research flows.

```python
from src.simple_or_agent.openrouter_client import OpenRouterClient
from src.simple_or_agent.next_agent import NextAgent, ToolSpec

client = OpenRouterClient(app_name="next-agent")
agent = NextAgent(client, model="qwen/qwen3-next-80b-a3b-thinking", reasoning_effort="high")

# Add tools as in the ReAct example

plan = agent.plan("Research top 3 approaches to agent reliability.")
result = agent.execute_with_plan("Research top 3 approaches to agent reliability.", plan.content)
print(result.content)
```


## Examples (CLI)

- `examples/chat_loop.py`: interactive chat loop with optional calculator tool and reasoning display.
- `examples/react_chat.py`: interactive NextAgent chat with optional web tools.
- `examples/next_demo.py`: one-shot deep research demo with plan + ReACT + citations.
- `examples/react_minimal.py`: minimal ReACT calculator demo.

Run with Poetry
```bash
poetry run python examples/chat_loop.py --show-reasoning
poetry run python examples/react_chat.py --with-web
poetry run python examples/next_demo.py "your research topic here" --time m --max-results 8
poetry run python examples/react_minimal.py "Solve (2+3)*4" --show-transcript
```


## Utilities

- `format_inline_citations(text)` in `src/simple_or_agent/__init__.py` converts bare URLs into `[n]` markers and appends a Sources section.  Used by demos.


## Troubleshooting

- Missing `OPENROUTER_API_KEY`: set it in the environment or `.env`.
- Rate limits (429) or server errors (5xx): the client retries automatically.  Reduce request rate if needed.
- Import errors for optional deps like `ddgs` or `beautifulsoup4`: run `poetry install` to ensure extras are present.


## Project intent

Keep things small, simple, and readable.  The client focuses on transport and parsing.  Agents compose behavior without heavy abstractions.


## Notes

- This client does not implement streaming; can be added if needed.
- Keep business/domain parsing outside the client (use the parsers module or your own logic).
- File sizes are kept under 300 lines per project guideline.
