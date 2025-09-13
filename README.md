# simple-or-agent

A small, focused ReAct-style agent built on top of OpenRouter. It exposes a minimal API, a tiny research demo with web tooling, and an interactive chat loop — all designed to stay readable and modular.

Key points
- Minimal: small modules, simple defaults, under-300‑LOC guideline per file.
- Reliable: automatic retries/backoff, robust response parsing, error surfacing.
- Tools: ddgs web search and HTTP page fetch with basic HTML→text extraction.
- ReAct: plan-then-act loop with tool calling, optional multi-round reasoning.
- LLM-friendly: clear system prompts and tool schemas for better tool use.


Requirements
- Python 3.12+
- Poetry (for dependency management)
- OpenRouter API key in `OPENROUTER_API_KEY`


Quick start
1) Install dependencies
- `poetry install`

2) Configure API key
- Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY=...`
- Optional: set a default model via `MODEL_ID` (defaults to `qwen/qwen3-next-80b-a3b-thinking`).

3) Run the interactive chat loop
- `poetry run python examples/chat_loop.py --show-reasoning`
- Flags: `--model`, `--temperature`, `--with-calculator`, `--system`, `--once`, `--prompt`.

4) Run the ReAct chat (with optional web tools)
- `poetry run python examples/react_chat.py`
- One-shot: `poetry run python examples/react_chat.py --once --prompt "hello"`
- Enable web tools: `poetry run python examples/react_chat.py --with-web --show-reasoning`
- Flags: `--model`, `--temperature`, `--system`, `--reasoning-effort`, `--show-reasoning`, `--no-history`, `--once`, `--prompt`, `--verbose`,
  and if `--with-web`: `--region`, `--time` (`d`,`w`,`m`,`y`), `--max-results`, `--fetch-chars`.

5) Run the deep research demo
- `poetry run python examples/react_demo.py "what is BTC trend today?" --time d --max-results 8`
- The agent uses ddgs search and fetches a few pages, synthesizing a short report with inline citations.


How it works
- OpenRouter client (`src/simple_or_agent/openrouter_client.py`):
  - Thin wrapper over `chat/completions` with retries (tenacity) and helpers:
    - `complete_chat`, `extract_content`, `extract_reasoning`, `get_tool_calls`, `make_tool_result`.
- ReAct agent (`src/simple_or_agent/react_agent.py`):
  - Maintains messages, calls the model, executes declared tools, and loops until a final answer or limits.
  - Supports tool-calls (function calling), optional parallel calls, and bounded tool iterations.
- Tools in the demo (`examples/react_demo.py`):
  - `web_search` via `ddgs` (DuckDuckGo). Returns title/url/snippet tuples.
  - `fetch_page` via `requests` + `beautifulsoup4`. Extracts readable text (caps length).
  - SERP safety: search engine result pages (Google/Bing/DDG/Yahoo) are filtered or blocked from fetches.
- Utility (`src/simple_or_agent/__init__.py`):
  - `format_inline_citations(text)` adds simple `[n]` markers for bare URLs and appends a Sources section (used by the demo output).


Programmatic use
```python
from src.simple_or_agent.openrouter_client import OpenRouterClient
from src.simple_or_agent.react_agent import ReActAgent, ToolSpec

# Minimal client
client = OpenRouterClient(app_name="simple-or-agent")

# Agent
agent = ReActAgent(client, keep_history=True, temperature=0.1, reasoning_effort="high")

# Optional: add tools (see examples/react_demo.py for full versions)
from ddgs import DDGS

def web_search_handler(args):
    query = args.get("query", "")
    out = []
    with DDGS() as ddg:
        for r in ddg.text(query, region="us-en", max_results=5):
            out.append({"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")})
    return {"query": query, "results": out}

web_search = ToolSpec(
    name="web_search",
    description="Search the web via ddgs and return top results",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    handler=web_search_handler,
)

agent.add_tool(web_search)

res = agent.ask("Find recent coverage of AI safety news")
print(res.content)
```


CLI references
- Chat loop: `examples/chat_loop.py`
  - Interactive, supports a lightweight calculator tool (`--with-calculator`).
  - Optional response schema, system prompts, reasoning, and one-shot mode.
- ReAct chat: `examples/react_chat.py`
  - Interactive REAct agent; optionally exposes `web_search` + `fetch_page` via `--with-web`.
  - Supports multi-round tool use, reasoning display, and history toggling.
- Research demo: `examples/react_demo.py`
  - ddgs search + HTML fetch, SERP filtering, simple inline citations.
  - Tunables: `--max-results`, `--time` (`d`, `w`, `m`, `y`), `--region`, `--fetch-chars`.


Configuration & env
- `OPENROUTER_API_KEY`: required (can be set in `.env`).
- `MODEL_ID`: default model identifier (overridden by `--model`).
- Optional headers: `app_name`, `app_url` passed to `OpenRouterClient` for OpenRouter dashboard attribution.


Notes & limitations
- ddgs may rate-limit or return sparse results for certain regions/time windows; the agent will attempt multiple queries but stays conservative.
- `fetch_page` limits text length (default 5000 chars) and strips scripts/styles; some dynamic sites may yield little text.
- SERP pages are intentionally not fetched to avoid noisy content; the model is nudged to select real articles.
- This is not a browser/JS runtime; it’s an HTTP fetch + parse flow intended for concise research.

Recent improvements
- Structured content handling: `extract_content` now tolerates list-based and nested content parts and the `parsed` field from providers.
- Tool-call flow: the agent now preserves assistant tool-call messages in history, improving multi-round tool reliability.


Troubleshooting
- `ddgs_import_failed: No module named 'ddgs'`
  - Ensure dependencies are installed: `poetry update ddgs` or `poetry install`.
  - Remove old package to avoid confusion: `poetry remove duckduckgo_search`.
- OpenRouter errors (401/429/5xx)
  - Verify `OPENROUTER_API_KEY`, wait and retry on 429/5xx, or lower request rate.


Project layout
- `src/simple_or_agent/openrouter_client.py` — OpenRouter wrapper + helpers
- `src/simple_or_agent/react_agent.py` — ReAct loop and tool execution
- `src/simple_or_agent/__init__.py` — small utilities (inline citations)
- `examples/chat_loop.py` — interactive chat CLI
- `examples/react_demo.py` — web research demo (ddgs + fetch)
