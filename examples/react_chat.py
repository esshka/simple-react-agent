#!/usr/bin/env python3
"""Interactive chat using ReActAgent with optional web tools (<=300 LOC).

- Similar to examples/chat_loop.py but leverages ReActAgent for tool use.
- Optional DuckDuckGo search + page fetch tools via ddgs + BeautifulSoup.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse

# Load .env so OPENROUTER_API_KEY is picked up
try:  # pragma: no cover
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Ensure project root import when running from examples/
import pathlib
root = pathlib.Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.simple_or_agent.openrouter_client import OpenRouterClient, OpenRouterError  # noqa: E402
from src.simple_or_agent.react_agent import ReActAgent, ToolSpec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive chat with ReActAgent")
    p.add_argument("--model", default=os.getenv("MODEL_ID", "qwen/qwen3-next-80b-a3b-thinking"), help="Model ID")
    p.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    p.add_argument("--system", type=str, default=None, help="Optional system message override")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None, help="Enable and set reasoning effort")
    p.add_argument("--show-reasoning", action="store_true", help="Print model reasoning if available")
    p.add_argument("--no-history", action="store_true", help="Do not keep conversation history between turns")
    p.add_argument("--once", action="store_true", help="Run a single turn and exit (requires --prompt)")
    p.add_argument("--prompt", type=str, default=None, help="One-shot user input for --once mode")
    p.add_argument("--verbose", action="store_true", help="Log raw responses and usage")
    # Optional web tools
    p.add_argument("--with-web", action="store_true", help="Enable web_search and fetch_page tools")
    p.add_argument("--region", default="us-en", help="DuckDuckGo region (us-en, uk-en, ...) for web_search")
    p.add_argument("--time", default=None, help="Time filter: d,w,m,y (or omit)")
    p.add_argument("--max-results", type=int, default=8, help="Max search results per query (3-15)")
    p.add_argument("--fetch-chars", type=int, default=5000, help="Max characters to return from fetched page (1000-15000)")
    return p


def make_web_search_tool(default_region: str, default_time: str | None, default_max: int) -> ToolSpec:
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
    def handler(args: Dict[str, Any]) -> Any:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "empty_query"}
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
                        continue
                    out.append({"title": r.get("title"), "url": url, "snippet": r.get("body")})
        except Exception as e:
            return {"error": f"ddgs_search_failed: {e}", "query": query}
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
            from bs4 import BeautifulSoup  # type: ignore
        except Exception:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        parts: List[str] = []
        for sel in ["h1", "h2", "h3", "p", "li"]:
            for el in soup.select(sel):
                t = (el.get_text(" ", strip=True) or "").strip()
                if t:
                    parts.append(t)
        return "\n".join(line.strip() for line in "\n".join(parts).splitlines() if line.strip())
    def handler(args: Dict[str, Any]) -> Any:
        import requests
        url = str(args.get("url", "")).strip()
        if not url:
            return {"error": "empty_url"}
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

    def supports_color() -> bool:
        if os.environ.get("NO_COLOR"):
            return False
        try:
            return sys.stdout.isatty()
        except Exception:
            return False
    USE_COLOR = supports_color()
    def color(s: str, code: str) -> str:
        return f"\033[{code}m{s}\033[0m" if USE_COLOR else s
    c_header = lambda s: color(s, "95;1")  # noqa: E731
    c_reason = lambda s: color(s, "33")    # noqa: E731

    try:
        client = OpenRouterClient(app_name="react-chat")
    except OpenRouterError as e:
        print(f"Client error: {e}", file=sys.stderr)
        print("Ensure OPENROUTER_API_KEY is set (optionally via .env).", file=sys.stderr)
        return 2

    agent = ReActAgent(
        client,
        model=args.model,
        system_prompt=(args.system or ReActAgent.DEFAULT_SYSTEM),
        keep_history=not args.no_history,
        temperature=args.temperature,
        max_rounds=8,
        max_tool_iters=8,
        reasoning_effort=args.reasoning_effort,
        parallel_tool_calls=True,
    )
    if args.with_web:
        agent.add_tool(make_web_search_tool(args.region, args.time, args.max_results))
        agent.add_tool(make_fetch_page_tool(args.fetch_chars))

    def run_turn(user_text: str) -> None:
        res = agent.ask(user_text)
        print(c_header("assistant>"))
        print(res.content)
        if args.show_reasoning and res.reasoning:
            print(c_header("reasoning>"))
            print(c_reason(res.reasoning))
        if args.verbose and res.usage:
            print(c_header("usage>"))
            print(json.dumps(res.usage, indent=2))

    # One-shot or interactive
    if args.once:
        if not args.prompt:
            print("--once requires --prompt", file=sys.stderr)
            return 2
        try:
            run_turn(args.prompt)
        finally:
            client.close()
        return 0

    print("ReAct chat. Type /exit to quit, /reset to clear history.")
    if args.system:
        print("[system message set]")
    try:
        while True:
            try:
                user_text = input("you> ")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                break
            cmd = user_text.strip().lower()
            if cmd in {"/exit", ":q", ":wq", "/quit"}:
                break
            if cmd == "/reset":
                agent.reset()
                print("[history cleared]")
                continue
            if not user_text.strip():
                continue
            try:
                run_turn(user_text)
            except KeyboardInterrupt:
                print()
                break
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

