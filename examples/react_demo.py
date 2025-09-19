#!/usr/bin/env python3
"""Deep research via a ReACT agent using ddgs.
Exposes `web_search` and `fetch_page`; runs Thought→Action→Observation until Final Answer.
Prints Final Answer (formatted). Use --show-transcript to print the steps."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List
import os

from simple_or_agent.tools.web_search import make_fetch_page_tool, make_web_search_tool

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


def _split_transcript_and_final(text: str) -> tuple[str, str]:
    """Return (transcript, final_answer_text) using the last 'Final Answer:' marker."""
    if not isinstance(text, str) or not text:
        return "", ""
    i = text.rfind("Final Answer:")
    if i < 0:
        return text, ""
    return text[:i].rstrip(), text[i + len("Final Answer:"):].strip()


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
    p.add_argument("--show-transcript", action="store_true", help="Print Thought/Action/Observation transcript")
    return p


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
        system_prompt=None,
        keep_history=True,
        temperature=0.1,
        max_rounds=8,
        max_tool_iters=8,
        reasoning_effort=args.reasoning_effort,
        parallel_tool_calls=False,
    )

    # Add tools
    agent.add_tool(make_web_search_tool(default_region=args.region, default_time=args.time, default_max=args.max_results))
    agent.add_tool(make_fetch_page_tool(default_max_chars=args.fetch_chars))

    user_prompt = (
        "Research this topic in depth: \"{topic}\". "
        "Search broadly, fetch a handful of the most relevant pages, extract critical facts and quotes, "
        "compare viewpoints, and synthesize. Always cite sources inline like [n] and provide the list at the end."
    ).format(topic=args.topic)

    res = agent.ask(user_prompt)

    transcript, final = _split_transcript_and_final(res.content)
    print("assistant>")
    if final:
        print(format_inline_citations(final))
    else:
        # Fallback if no explicit Final Answer was produced
        print(format_inline_citations(res.content))
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
