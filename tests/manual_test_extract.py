from __future__ import annotations

import pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.simple_or_agent.openrouter_client import OpenRouterClient


def main() -> None:
    c = OpenRouterClient.__new__(OpenRouterClient)  # bypass __init__

    def ext(resp):
        print("->", c.extract_content(resp))

    # 1) plain string
    ext({"choices": [{"message": {"content": "Hello"}}]})
    # 2) structured list types
    ext({"choices": [{"message": {"content": [
        {"type": "reasoning", "text": "think"},
        {"type": "output_text", "text": "Final answer"}
    ]}}]})
    # 3) nested text list
    ext({"choices": [{"message": {"content": [
        {"text": [{"text": "part1"}, {"text": "part2"}]}
    ]}}]})
    # 4) parsed present
    ext({"choices": [{"message": {"content": [], "parsed": {"a": 1}}}]})
    # 5) tool_calls summary
    ext({"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "abc", "function": {"name": "web_search"}}
    ]}}]})


if __name__ == "__main__":
    main()

