#!/usr/bin/env python3
"""Deep research demo using ddgs (DuckDuckGo) search and simple page fetch.

- Exposes two tools: `web_search` (DuckDuckGo) and `fetch_page` (HTTP + HTML->text).
- The agent plans, searches, fetches content, and synthesizes findings with citations.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse
import os

# Load .env if available so OPENROUTER_API_KEY is picked up
try:  # pragma: no cover - convenience
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Ensure project root import when running from scripts/
import pathlib
root = pathlib.Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.simple_or_agent.react_agent import ReActAgent, ToolSpec  # noqa: E402
from src.simple_or_agent.openrouter_client import OpenRouterClient  # noqa: E402
from src.simple_or_agent import format_inline_citations  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deep research via DuckDuckGo + fetch")
    p.add_argument("topic", type=str, help="Research topic or question")
    p.add_argument("--model", default=os.getenv("MODEL_ID", "qwen/qwen3-next-80b-a3b-thinking"), help="Model ID")
    p.add_argument("--max-results", type=int, default=8, help="Max search results per query (3-15)")
    p.add_argument("--fetch-chars", type=int, default=5000, help="Max characters to return from fetched page (1000-15000)")
    p.add_argument("--region", default="us-en", help="DuckDuckGo region (e.g., us-en, uk-en)")
    p.add_argument("--time", default=None, help="DuckDuckGo time limit (d,w,m,y). Omit for any time.")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high", help="Reasoning effort")
    p.add_argument("--verbose", action="store_true", help="Print raw agent usage JSON")
    p.add_argument("--show-reasoning", action="store_true", help="Print model reasoning if present")
    return p


def make_web_search_tool(default_region: str, default_time: str | None, default_max: int) -> ToolSpec:
    """DuckDuckGo search (via ddgs). Returns compact results."""
    SERP_HOSTS = {
        "bing.com", "www.bing.com",
        "google.com", "www.google.com",
        "duckduckgo.com", "www.duckduckgo.com",
        "search.yahoo.com", "yahoo.com", "www.yahoo.com",
        "startpage.com", "www.startpage.com",
        "yandex.com", "www.yandex.com", "yandex.ru", "www.yandex.ru",
        "baidu.com", "www.baidu.com",
    }

    def _is_serp(url: str) -> bool:
        try:
            p = urlparse(url)
            host = (p.hostname or "").lower()
            if not host:
                return False
            if host in SERP_HOSTS:
                return True
            # Common SERP path patterns
            if any(seg in (p.path or "").lower() for seg in ("/search", "/html", "/lite")) and (
                host.endswith("google.com") or host.endswith("bing.com") or host.endswith("duckduckgo.com") or host.endswith("yahoo.com")
            ):
                return True
            return False
        except Exception:
            return False
    def handler(args: Dict[str, Any]) -> Any:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "empty_query"}

        # Use the new ddgs package exclusively
        try:
            from ddgs import DDGS  # type: ignore
        except Exception as e:
            return {"error": f"ddgs_import_failed: {e}"}

        region = str(args.get("region", default_region) or default_region)
        timelimit = args.get("time", default_time)
        max_results = int(args.get("max_results", default_max) or default_max)
        max_results = max(3, min(max_results, 15))

        out: List[Dict[str, Any]] = []
        try:
            with DDGS() as ddg:
                for r in ddg.text(query, region=region, safesearch="moderate", timelimit=timelimit, max_results=max_results):
                    url = r.get("href")
                    if not url or _is_serp(url):
                        # Skip search engine result pages; they are not sources
                        continue
                    out.append({
                        "title": r.get("title"),
                        "url": url,
                        "snippet": r.get("body"),
                    })
        except Exception as e:
            return {"error": f"ddgs_search_failed: {e}", "query": query}

        # If filtering removed everything, still return empty list (let model refine query)
        return {"query": query, "results": out[:max_results]}

    params: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "minimum": 3, "maximum": 15, "default": default_max},
            "region": {"type": "string", "default": default_region},
            "time": {"type": ["string", "null"], "enum": [None, "d", "w", "m", "y"], "default": default_time},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    return ToolSpec(name="web_search", description="Search the web via ddgs (DuckDuckGo) and return top results (title, url, snippet)", parameters=params, handler=handler)


def make_fetch_page_tool(default_max_chars: int) -> ToolSpec:
    """HTTP GET a URL and extract readable text using BeautifulSoup."""
    UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

    SERP_HOSTS = {
        "bing.com", "www.bing.com",
        "google.com", "www.google.com",
        "duckduckgo.com", "www.duckduckgo.com",
        "search.yahoo.com", "yahoo.com", "www.yahoo.com",
        "startpage.com", "www.startpage.com",
        "yandex.com", "www.yandex.com", "yandex.ru", "www.yandex.ru",
        "baidu.com", "www.baidu.com",
    }

    def _is_serp(url: str) -> bool:
        try:
            p = urlparse(url)
            host = (p.hostname or "").lower()
            if not host:
                return False
            if host in SERP_HOSTS:
                return True
            if any(seg in (p.path or "").lower() for seg in ("/search", "/html", "/lite")) and (
                host.endswith("google.com") or host.endswith("bing.com") or host.endswith("duckduckgo.com") or host.endswith("yahoo.com")
            ):
                return True
            return False
        except Exception:
            return False

    def _extract_text(html: str) -> str:
        try:
            from bs4 import BeautifulSoup  # local import
        except Exception:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        parts: List[str] = []
        # Prefer headings and paragraphs to keep it compact
        for sel in ["h1", "h2", "h3", "p", "li"]:
            for el in soup.select(sel):
                text = (el.get_text(" ", strip=True) or "").strip()
                if text:
                    parts.append(text)
        text = "\n".join(parts)
        # Collapse whitespace
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    def handler(args: Dict[str, Any]) -> Any:
        import requests
        url = str(args.get("url", "")).strip()
        if not url:
            return {"error": "empty_url"}
        # Block fetching search engine result pages to avoid garbage content
        if _is_serp(url):
            return {"error": "blocked_serp_url", "url": url}
        max_chars = int(args.get("max_chars", default_max_chars) or default_max_chars)
        max_chars = max(1000, min(max_chars, 15000))
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": UA})
            resp.raise_for_status()
        except Exception as e:
            return {"error": f"fetch_failed: {e}", "url": url}

        title = None
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(resp.text, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except Exception:
            title = None

        text = _extract_text(resp.text)
        if len(text) > max_chars:
            text = text[:max_chars]
        return {"url": url, "title": title, "text": text, "length": len(text)}

    params: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Page URL to fetch"},
            "max_chars": {"type": "integer", "minimum": 1000, "maximum": 15000, "default": default_max_chars},
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    return ToolSpec(name="fetch_page", description="Fetch a web page and return cleaned text (capped by max_chars)", parameters=params, handler=handler)


def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Agent & client
    try:
        client = OpenRouterClient(app_name="deep-research-demo")
    except Exception as e:
        print(f"Client init failed: {e}", file=sys.stderr)
        print("Ensure OPENROUTER_API_KEY is set (optionally via .env).", file=sys.stderr)
        return 2

    agent = ReActAgent(
        client,
        model=args.model,
        system_prompt=(
            "You are a meticulous research assistant. Use web_search to discover diverse, high-quality sources; "
            "then use fetch_page to extract key passages. Never fetch search engine results pages (SERPs) like Bing/Google/" \
            "DuckDuckGo — only fetch actual content URLs from web_search results. Plan briefly as bullet points, avoid " \
            "hallucinations, and cross-check claims. Provide a structured Final Answer with: Key Findings, Evidence with " \
            "inline citations [1], [2], Counterpoints, Gaps/Limitations, and a Sources list with titles and URLs."
        ),
        keep_history=True,
        temperature=0.1,
        max_rounds=8,
        max_tool_iters=8,
        reasoning_effort=args.reasoning_effort,
        parallel_tool_calls=True,
    )

    # Add tools
    agent.add_tool(make_web_search_tool(default_region=args.region, default_time=args.time, default_max=args.max_results))
    agent.add_tool(make_fetch_page_tool(default_max_chars=args.fetch_chars))

    user_prompt = (
        "Research this topic in depth: \"{topic}\". "
        "Search broadly, fetch a handful of the most relevant pages, extract critical facts and quotes, "
        "compare viewpoints, and synthesize. Always cite sources inline like [n] and provide the list at the end."
    ).format(topic=args.topic)

    # Ask the agent (multi-round tool calls allowed)
    res = agent.ask(user_prompt)

    # Output (with minimal inline-citation formatting)
    print("assistant>")
    print(format_inline_citations(res.content))
    if args.show_reasoning and res.reasoning:
        print("reasoning>")
        print(res.reasoning)
    if args.verbose:
        print("usage>")
        print(json.dumps(res.usage or {}, indent=2))

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
