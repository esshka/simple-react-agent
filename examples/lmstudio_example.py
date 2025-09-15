# examples/lmstudio_example.py
# Small example using LMStudioClient against a local LM Studio server.
# This exists to show basic, minimal usage with safe defaults.
# RELEVANT FILES: src/simple_or_agent/lmstudio_client.py,src/simple_or_agent/__init__.py,README.md

"""Run a minimal chat completion against a local LM Studio server.

Usage is simple.

Adjust the base URL or model via environment variables if needed.
"""

from __future__ import annotations

import os
from simple_or_agent import LMStudioClient


def main() -> None:
    # Base URL defaults to the LAN address and will add /v1 automatically.
    # Override with LMSTUDIO_BASE_URL if your server is elsewhere.
    base_url = os.getenv("LMSTUDIO_BASE_URL")  # e.g., "http://127.0.0.1:1234"

    # Choose a local model that exists in your LM Studio server.
    # You can set LMSTUDIO_MODEL to avoid editing the file.
    model = os.getenv("LMSTUDIO_MODEL", "local-model")

    # Simple single-turn user message.
    messages = [{"role": "user", "content": "Say hello in one short line."}]

    with LMStudioClient(base_url=base_url) as client:
        resp = client.chat_completions(model=model, messages=messages, temperature=0.7)
        print(client.extract_content(resp))


if __name__ == "__main__":
    main()

