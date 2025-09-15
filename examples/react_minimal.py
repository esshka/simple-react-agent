#!/usr/bin/env python3
"""Minimal ReACT example.

- Uses the new ReActAgent orchestrator (Thinker → Operator → Validator)
- Exposes a single tool: `calc` to evaluate basic arithmetic expressions
  (safe AST-based evaluation; supports +, -, *, /, **, parentheses).
"""

from __future__ import annotations

import argparse
import sys
import os
import ast
import operator as op

# Load .env if available (so OPENROUTER_API_KEY is picked up)
try:  # pragma: no cover - convenience
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
from src.simple_or_agent.react_agent import ReActAgent, ToolSpec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal ReACT demo with a calculator tool")
    p.add_argument("prompt", type=str, help="User prompt/task")
    p.add_argument("--model", default=os.getenv("MODEL_ID", "qwen/qwen3-next-80b-a3b-thinking"), help="Model ID")
    p.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None, help="Enable and set reasoning effort")
    p.add_argument("--show-transcript", action="store_true", help="Print ReACT Thought/Action/Observation transcript")
    return p


def make_calc_tool() -> ToolSpec:
    """Create a safe calculator tool using a tiny AST evaluator."""
    allowed = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            return allowed[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("unsupported expression")

    def handler(args):
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return {"error": "empty_expression"}
        try:
            tree = ast.parse(expr, mode="eval")
            val = _eval(tree.body)  # type: ignore[arg-type]
            return {"expression": expr, "value": val}
        except Exception as e:
            return {"error": str(e)}

    params = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Arithmetic expression"}},
        "required": ["expression"],
        "additionalProperties": False,
    }
    return ToolSpec(name="calc", description="Evaluate basic arithmetic expression and return a JSON result", parameters=params, handler=handler)


def _split_transcript_and_final(text: str) -> tuple[str, str]:
    if not isinstance(text, str) or not text:
        return "", ""
    i = text.rfind("Final Answer:")
    if i < 0:
        return text, ""
    return text[:i].rstrip(), text[i + len("Final Answer:"):].strip()


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    try:
        client = OpenRouterClient(app_name="react-minimal")
    except Exception as e:
        print(f"Client init failed: {e}", file=sys.stderr)
        print("Ensure OPENROUTER_API_KEY is set (optionally via .env).", file=sys.stderr)
        return 2

    agent = ReActAgent(
        client,
        model=args.model,
        system_prompt=None,
        keep_history=True,
        temperature=args.temperature,
        max_rounds=6,
        max_tool_iters=3,
        reasoning_effort=args.reasoning_effort,
        parallel_tool_calls=False,
    )
    agent.add_tool(make_calc_tool())

    res = agent.ask(args.prompt)
    transcript, final = _split_transcript_and_final(res.content)

    print("assistant>")
    print(final or res.content)
    if args.show_transcript and transcript:
        print("transcript>")
        print(transcript)

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

