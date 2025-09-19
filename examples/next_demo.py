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

from simple_or_agent.tools.web_search import make_fetch_page_tool, make_web_search_tool

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
