#!/usr/bin/env python3
"""Interactive chat loop for testing OpenRouter client."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List
import os

# Load .env if available so OPENROUTER_API_KEY is picked up
try:  # pragma: no cover - convenience
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Ensure project root is importable when running from scripts/
try:
    from src.simple_or_agent.openrouter_client import (
        OpenRouterClient,
        OpenRouterError,
        OpenRouterAPIError,
    )
except ModuleNotFoundError:
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.simple_or_agent.openrouter_client import (
        OpenRouterClient,
        OpenRouterError,
        OpenRouterAPIError,
    )

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenRouter chat loop")
    p.add_argument("--model", default=os.getenv("MODEL_ID", "qwen/qwen3-next-80b-a3b-thinking"), help="Model ID")
    p.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    p.add_argument("--max-tokens", type=int, default=None, help="Max tokens in response")
    p.add_argument("--system", type=str, default=None, help="Optional system message")
    p.add_argument("--app-name", type=str, default=None, help="Optional app name header")
    p.add_argument("--app-url", type=str, default=None, help="Optional app URL header")
    p.add_argument("--schema-file", type=str, default=None, help="Path to JSON schema file for response_format")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None, help="Enable and set reasoning effort")
    p.add_argument("--show-reasoning", action="store_true", help="Display model reasoning if provided by the model")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors in output")
    p.add_argument("--with-calculator", action="store_true", help="Expose a simple calculator tool to the model")
    p.add_argument("--prompt", type=str, default=None, help="One-shot user message (non-interactive if provided with --once)")
    p.add_argument("--once", action="store_true", help="Run a single turn and exit")
    p.add_argument("--no-history", action="store_true", help="Do not keep conversation history between turns")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p


def load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ANSI color helpers
    def _supports_color() -> bool:
        if args.no_color:
            return False
        if os.environ.get("NO_COLOR"):
            return False
        try:
            return sys.stdout.isatty()
        except Exception:
            return False

    USE_COLOR = _supports_color()

    def color(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

    def c_header(s: str) -> str:
        return color(s, "95;1")  # bright magenta bold

    def c_reasoning(s: str) -> str:
        return color(s, "33")  # yellow


    # Prepare optional response_format from schema file
    response_format = None
    if args.schema_file:
        try:
            schema = load_schema(args.schema_file)
        except Exception as e:
            print(f"Failed to load schema file: {e}", file=sys.stderr)
            return 2
        # Build response_format directly to avoid requiring an API key early
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "LLMOutput", "schema": schema},
        }

    tools = None
    if args.with_calculator:
        tools = [{"type": "function", "function": {"name": "calculate", "description": "Evaluate arithmetic expression with +, -, *, /, **, %, parentheses. Return a number.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]

    def _calc(expr: str) -> str:
        import ast, operator as op
        allowed = {
            ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
            ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos,
        }
        def _eval(node):
            if isinstance(node, ast.Num):
                return node.n
            if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
                return allowed[type(node.op)](_eval(node.operand))
            if isinstance(node, ast.BinOp) and type(node.op) in allowed:
                return allowed[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            raise ValueError("Unsupported expression")
        try:
            tree = ast.parse(expr, mode="eval")
            return str(_eval(tree))
        except Exception as e:
            return f"error: {e}"

    try:
        client = OpenRouterClient(app_name=args.app_name, app_url=args.app_url)
    except OpenRouterError as e:
        print(f"Client error: {e}", file=sys.stderr)
        print("Ensure OPENROUTER_API_KEY is set.", file=sys.stderr)
        return 2

    messages: List[Dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    def run_turn(user_text: str) -> None:
        nonlocal messages
        # Optionally drop history
        if args.no_history:
            messages = messages[:1] if messages and messages[0]["role"] == "system" else []

        messages.append({"role": "user", "content": user_text})

        try:
            resp = client.complete_chat(
                messages=messages,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                response_format=response_format,
                reasoning={"effort": args.reasoning_effort} if args.reasoning_effort else None,
                tools=tools,
            )
        except (OpenRouterAPIError, OpenRouterError) as e:
            print(f"API error: {e}", file=sys.stderr)
            return

        if args.verbose:
            print(c_header("response>"))
            print(json.dumps(resp, indent=2, ensure_ascii=False))

        # Handle tool calls (single round)
        try:
            calls = client.get_tool_calls(resp)
        except Exception:
            calls = []
        if calls:
            # Execute each tool call and append tool results
            for c in calls:
                try:
                    func = c.get("function", {})
                    name = func.get("name")
                    args_s = func.get("arguments") or "{}"
                    args_j = json.loads(args_s) if isinstance(args_s, str) else args_s
                except Exception:
                    name, args_j = None, {}
                result = "unsupported tool"
                if name == "calculate":
                    expr = str(args_j.get("expression", ""))
                    result = _calc(expr)
                messages.append(client.make_tool_result(c.get("id", ""), result))
            # Ask again for final assistant message
            try:
                resp = client.complete_chat(
                    messages=messages,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    response_format=response_format,
                )
            except (OpenRouterAPIError, OpenRouterError) as e:
                print(f"API error (after tools): {e}", file=sys.stderr)
                return
            if args.verbose:
                print(c_header("response (after tools)>"))
                print(json.dumps(resp, indent=2, ensure_ascii=False))

        try:
            content = client.extract_content(resp)
        except OpenRouterError as e:
            print(f"Response parse error: {e}", file=sys.stderr)
            return

        # Record assistant reply in history
        messages.append({"role": "assistant", "content": content})

        # Print assistant reply
        print(c_header("assistant>"))
        print(content)

        # Optional: print reasoning
        if args.show_reasoning:
            try:
                reasoning_txt = client.extract_reasoning(resp)
            except Exception:
                reasoning_txt = None
            if reasoning_txt:
                print(c_header("reasoning>"))
                print(c_reasoning(reasoning_txt))

        # Token usage
        usage = resp.get("usage")
        if usage:
            pt, ct, tt = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
            rt = usage.get("reasoning_tokens") or usage.get("reasoning")
            extra = f" reasoning={c_reasoning(str(rt))}" if rt is not None else ""
            print(f"[tokens] prompt={pt} completion={ct} total={tt}{extra}")

    # One-shot or interactive loop
    if args.once:
        try:
            if args.prompt is None:
                try:
                    user_text = input("you> ")
                except EOFError:
                    return 0
                except KeyboardInterrupt:
                    print()
                    return 0
            else:
                user_text = args.prompt
            run_turn(user_text)
        except KeyboardInterrupt:
            print()
        finally:
            client.close()
        return 0

    # Interactive loop
    print("OpenRouter chat loop. Type /exit to quit, /reset to clear history.")
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
                messages = []
                if args.system:
                    messages.append({"role": "system", "content": args.system})
                print("[history cleared]")
                continue
            if not user_text.strip():
                continue

            try:
                run_turn(user_text)
            except KeyboardInterrupt:
                print()
                break
    except KeyboardInterrupt:
        print()
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
