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
from dotenv import load_dotenv
import requests
import sys


def main() -> None:
    # Load variables from .env so LMSTUDIO_BASE_URL is respected when running directly.
    load_dotenv()

    # Prefer localhost by default to avoid LAN routing issues.
    # Override with LMSTUDIO_BASE_URL if your server is on another host.
    base_url = os.getenv("LMSTUDIO_BASE_URL") or "http://127.0.0.1:1234"

    # Choose a local model that exists in your LM Studio server.
    # You can set LMSTUDIO_MODEL to avoid editing the file.
    model = os.getenv("LMSTUDIO_MODEL", "local-model")

    # Simple single-turn user message.
    messages = [{"role": "user", "content": "Say hello in one short line."}]

    print(f"Using LM Studio base URL: {base_url}")

    # Tiny preflight: GET /v1/models to verify reachability.
    # If base_url already ends with /v1, use /models instead.
    root = base_url.rstrip("/")
    models_url = f"{root}/models" if root.endswith("/v1") else f"{root}/v1/models"
    try:
        r = requests.get(models_url, timeout=5)
        if not r.ok:
            print(f"Preflight failed: GET {models_url} -> HTTP {r.status_code}")
            print("Check that LM Studio is running, listening on the network, and firewall allows the port.\n")
            sys.exit(2)
        print("Preflight OK: /models reachable.\n")
    except requests.exceptions.RequestException as e:
        print(f"Preflight error: cannot reach {models_url}: {e}")
        print("If LM Studio runs remotely, bind to 0.0.0.0 and open port 1234.\n")
        sys.exit(2)

    with LMStudioClient(base_url=base_url) as client:
        resp = client.chat_completions(model=model, messages=messages, temperature=0.7)
        print(client.extract_content(resp))


if __name__ == "__main__":
    main()
