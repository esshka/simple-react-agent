# examples/next_demo.py
# Demo for NextAgent orchestrated planning and execution.
# This file exists to showcase scoped memory and focused context passing.
# RELEVANT FILES: src/simple_or_agent/next_agent.py,src/simple_or_agent/orchestrator.py,src/simple_or_agent/memory.py

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse

# Optional .env loader
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Ensure project root import when running from examples/
import pathlib
root = pathlib.Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.simple_or_agent.openrouter_client import OpenRouterClient  # noqa: E402
from src.simple_or_agent.lmstudio_client import LMStudioClient  # noqa: E402
from src.simple_or_agent.next_agent import NextAgent, ToolSpec  # noqa: E402
from src.simple_or_agent import format_inline_citations  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Planner + ReACT deep-research demo")
    p.add_argument("topic", type=str, help="Research topic or question")
    p.add_argument("--client", choices=["openrouter", "lmstudio"], default="openrouter", help="Backend client to use")
    p.add_argument("--model", default=os.getenv("MODEL_ID", "qwen/qwen3-next-80b-a3b-thinking"), help="Model ID")
    p.add_argument("--max-results", type=int, default=8, help="Max search results per query (3-15)")
    p.add_argument("--fetch-chars", type=int, default=5000, help="Max characters to return from fetched page (1000-15000)")
    p.add_argument("--region", default="us-en", help="DuckDuckGo region (e.g., us-en, uk-en)")
    p.add_argument("--time", default=None, help="DuckDuckGo time limit (d,w,m,y). Omit for any time.")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high", help="Reasoning effort")
    p.add_argument("--show-plan", action="store_true", help="Print the generated plan")
    p.add_argument("--show-transcript", action="store_true", help="Print the Thought/Action/Observation transcript")
    p.add_argument("--show-reasoning", action="store_true", help="Print model reasoning if present")
    p.add_argument("--verbose", action="store_true", help="Print raw usage JSON")
    return p


SERP_HOSTS = {"bing.com", "www.bing.com", "google.com", "www.google.com", "duckduckgo.com", "www.duckduckgo.com", "search.yahoo.com", "yahoo.com", "www.yahoo.com", "startpage.com", "www.startpage.com", "yandex.com", "www.yandex.com", "yandex.ru", "www.yandex.ru", "baidu.com", "www.baidu.com"}


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


def make_web_search_tool(default_region: str, default_time: str | None, default_max: int) -> ToolSpec:
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
    return ToolSpec(name="web_search", description="Search the web via ddgs and return top results (title, url, snippet)", parameters=params, handler=handler)


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
            text = (el.get_text(" ", strip=True) or "").strip()
            if text:
                parts.append(text)
    text = "\n".join(parts)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def make_fetch_page_tool(default_max_chars: int) -> ToolSpec:
    UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
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


def _split_transcript_and_final(text: str) -> tuple[str, str]:
    if not isinstance(text, str) or not text:
        return "", ""
    i = text.rfind("Final Answer:")
    if i < 0:
        return text, ""
    return text[:i].rstrip(), text[i + len("Final Answer:"):].strip()


def _extract_plan(content: str) -> str:
    if not content.startswith("Plan\n"):
        return ""
    # Split first double newline after Plan section
    sep = content.find("\n\n")
    if sep == -1:
        return content[len("Plan\n"):].strip()
    return content[len("Plan\n"):sep].strip()


def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)
    # Keep logging minimal for a clean demo output

    try:
        if args.client == "openrouter":
            client = OpenRouterClient(app_name="next-demo")
        else:
            # Allow raising the LM Studio timeout for slow remote models.
            # Reads LMSTUDIO_TIMEOUT_S if present, else uses 180s default.
            lm_timeout = int(os.getenv("LMSTUDIO_TIMEOUT_S", "180"))
            client = LMStudioClient(timeout_s=lm_timeout)
    except Exception as e:
        print(f"Client init failed: {e}", file=sys.stderr)
        if args.client == "openrouter":
            print("Ensure OPENROUTER_API_KEY is set (optionally via .env).", file=sys.stderr)
        else:
            print("Check LMSTUDIO_BASE_URL and that LM Studio is reachable.", file=sys.stderr)
        return 2

    agent = NextAgent(
        client,
        model=args.model,
        planner_system=None,
        keep_history=True,
        temperature=0.1,
        max_rounds=16,
        reasoning_effort=args.reasoning_effort,
    )
    agent.add_tool(make_web_search_tool(default_region=args.region, default_time=args.time, default_max=args.max_results))
    agent.add_tool(make_fetch_page_tool(default_max_chars=args.fetch_chars))

    user_prompt = (
        "Research this topic in depth: \"{topic}\". "
        "Search broadly, fetch key sources, extract critical facts and quotes, compare viewpoints, and synthesize. "
        "Always cite sources inline like [n] and provide the list at the end."
    ).format(topic=args.topic)

    # Live, compact progress printer
    step_counter = {"n": 0}
    def progress_cb(event: str, data: Dict[str, Any]) -> None:
        if event == "plan" and args.show_plan:
            print("plan>")
            print((data.get("text") or "").strip(), flush=True)
            return
        if event == "think":
            step_counter["n"] += 1
            t = (data.get("thought") or "").strip()
            an = (data.get("action") or {}).get("name") or (data.get("action") or {}).get("type")
            print(f"step {step_counter['n']} > think>")
            if t:
                print(t, flush=True)
            if an:
                print(f"action: {an}", flush=True)
            return
        if event == "tool":
            name = data.get("name") or "(missing)"
            args_d = data.get("args") or {}
            try:
                import json as _json
                args_s = _json.dumps(args_d, ensure_ascii=False)
            except Exception:
                args_s = str(args_d)
            if len(args_s) > 600:
                args_s = args_s[:600] + "..."
            print(f"step {step_counter['n']} > tool>")
            print(f"{name} args: {args_s}", flush=True)
            return
        if event == "observe":
            act = data.get("action") or {}
            if (act.get("type") or "").strip().lower() == "finish":
                return
            print(f"step {step_counter['n']} > observation>")
            print(str(data.get("observation") or "").strip(), flush=True)
            return
        if event == "judge":
            if data.get("decision") == "final":
                print(f"step {step_counter['n']} > final>")
                txt = (data.get("final") or "").strip()
                if txt:
                    print(txt, flush=True)
            else:
                print(f"step {step_counter['n']} > continue>", flush=True)
            return

    res = agent.ask(user_prompt, progress_cb=progress_cb)

    transcript, final = _split_transcript_and_final(res.content)
    plan = _extract_plan(res.content)

    # Plan already printed during progress if --show-plan
    print("assistant>")
    print(format_inline_citations(final or res.content))
    if args.show_transcript and transcript:
        print("transcript>")
        print(transcript)
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
