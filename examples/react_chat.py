#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List

from simple_or_agent.tools.web_search import make_fetch_page_tool, make_web_search_tool

# Load .env if available (optional)
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

from src.simple_or_agent.openrouter_client import OpenRouterClient, OpenRouterError  # noqa: E402
from src.simple_or_agent.next_agent import NextAgent, ToolSpec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive chat with NextAgent (Planner + ReACT)")
    p.add_argument("--model", default=os.getenv("MODEL_ID", "qwen/qwen3-next-80b-a3b-thinking"), help="Model ID")
    p.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    p.add_argument("--planner-system", type=str, default=None, help="Optional planner system override")
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

    agent = NextAgent(
        client,
        model=args.model,
        planner_system=args.planner_system,
        keep_history=not args.no_history,
        temperature=args.temperature,
        max_rounds=8,
        reasoning_effort=args.reasoning_effort,
    )
    if args.with_web:
        agent.add_tool(make_web_search_tool(args.region, args.time, args.max_results))
        agent.add_tool(make_fetch_page_tool(args.fetch_chars))

    def run_turn(user_text: str) -> None:
        def show(label: str, res) -> None:
            print(c_header(f"{label}>"))
            rt = None
            try:
                rt = (res.usage or {}).get("reasoning_tokens")
            except Exception:
                rt = None
            if rt is not None:
                print(f"[reasoning_tokens: {rt}]")
            if args.show_reasoning and res.reasoning:
                print(c_reason(res.reasoning))
            print(res.content)
            if args.verbose and res.usage:
                print(c_header("usage>"))
                print(json.dumps(res.usage, indent=2))

        # 1) Planner emits a plan (no tools)
        plan_res = agent.plan(user_text)
        show("planner", plan_res)
        # 2) ReACT executor runs tools and solves the task
        react_res = agent.execute_with_plan(user_text, plan_res.content)
        show("react", react_res)

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

    print("NextAgent chat. Type /exit to quit, /reset to clear history.")
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
