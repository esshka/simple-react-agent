# src/simple_or_agent/instructor_based/lmstudio_client.py
# Builds Instructor clients that talk to a local LMStudio server.
# Exists to isolate LMStudio-specific logic and CLI helpers.
# RELEVANT FILES: src/simple_or_agent/instructor_based/openrouter_client.py, src/simple_or_agent/instructor_based/agent.py, src/simple_or_agent/instructor_based/provider_profiles.py

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ''}:
    project_src = Path(__file__).resolve().parent.parent.parent
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

import os
from typing import Any, Optional

import instructor
from instructor import Mode
from openai import OpenAI
from pydantic import BaseModel

from simple_or_agent.instructor_based.provider_profiles import resolve_profile, resolved_model_from_env

LMSTUDIO_PROVIDER_PREFIX = "openai/lmstudio"
LMSTUDIO_DEFAULT_BASE_URL = "http://100.66.248.94:1234/v1"
LMSTUDIO_DEFAULT_MODE = Mode.JSON_SCHEMA
LMSTUDIO_DEFAULT_API_KEY = "lm-studio"

API_KEY_ENV = "INSTRUCTOR_API_KEY"
FALLBACK_KEY_ENV = "OPENROUTER_API_KEY"
BASE_URL_ENV = "INSTRUCTOR_BASE_URL"
MODE_ENV = "INSTRUCTOR_MODE"


def normalize_base_url(value: Optional[str]) -> str:
    """Return a base URL that ends with /v1."""
    raw = (value or LMSTUDIO_DEFAULT_BASE_URL).strip()
    if not raw:
        raw = LMSTUDIO_DEFAULT_BASE_URL
    trimmed = raw.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed
    return f"{trimmed}/v1"


def build_client(api_key: str, base_url: Optional[str] = None, mode: Optional[Mode] = None) -> Any:
    """Create an Instructor client pointed at LMStudio."""
    resolved_base_url = normalize_base_url(base_url)
    resolved_mode = mode or LMSTUDIO_DEFAULT_MODE
    openai_client = OpenAI(api_key=api_key, base_url=resolved_base_url)
    return instructor.from_openai(openai_client, mode=resolved_mode)


def discover_model_id(base_url: Optional[str], api_key: str, fallback: str) -> str:
    """Ask LMStudio for its first model so we have a sensible default."""
    resolved_base_url = normalize_base_url(base_url)
    try:
        client = OpenAI(api_key=api_key, base_url=resolved_base_url)
        models = client.models.list()
        data = getattr(models, "data", None) or []
        if data:
            first = data[0]
            model_id = getattr(first, "id", None)
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()
    except Exception as exc:
        print(f"Model discovery failed: {exc}")
    return fallback


def probe_connection(base_url: Optional[str], api_key: str) -> bool:
    """Return True when LMStudio responds to a simple models.list call."""
    resolved_base_url = normalize_base_url(base_url)
    try:
        client = OpenAI(api_key=api_key, base_url=resolved_base_url)
        client.models.list()
        return True
    except Exception as exc:
        print(f"LMStudio health check failed: {exc}")
        return False


def is_lmstudio_provider(provider_id: str) -> bool:
    """Return True when the provider string refers to LMStudio."""
    return provider_id.startswith(LMSTUDIO_PROVIDER_PREFIX)


def _resolve_api_key() -> Optional[str]:
    """Read the preferred API key from the environment."""
    value = os.getenv(API_KEY_ENV) or os.getenv(FALLBACK_KEY_ENV)
    if not value:
        return None
    trimmed = value.strip()
    return trimmed or None


def _resolve_base_url() -> Optional[str]:
    """Return the explicit LMStudio base URL override when set."""
    value = os.getenv(BASE_URL_ENV)
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _resolve_mode() -> Optional[Mode]:
    """Map INSTRUCTOR_MODE to an Instructor Mode enum value."""
    raw = os.getenv(MODE_ENV)
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    try:
        return Mode[candidate.upper()]
    except KeyError:
        names = ", ".join(member.name for member in Mode)
        print(f"Unknown INSTRUCTOR_MODE '{raw}'. Valid options: {names}")
        return None


def main() -> int:
    """Allow quick manual checks against a running LMStudio server."""
    profile = resolve_profile()
    env_base_url = _resolve_base_url()
    base_url = env_base_url if env_base_url is not None else (profile.base_url or LMSTUDIO_DEFAULT_BASE_URL)
    resolved_base_url = normalize_base_url(base_url)

    api_key = _resolve_api_key() or profile.default_api_key or LMSTUDIO_DEFAULT_API_KEY
    if not api_key:
        print("Client build failed: Missing API key. Set INSTRUCTOR_API_KEY or OPENROUTER_API_KEY.")
        return 1

    mode = _resolve_mode() or profile.mode or LMSTUDIO_DEFAULT_MODE

    if not probe_connection(resolved_base_url, api_key):
        print(f"Could not reach LMStudio. Confirm the server is running at {resolved_base_url}.")
        return 2

    try:
        client = build_client(api_key=api_key, base_url=resolved_base_url, mode=mode)
    except Exception as exc:
        print(f"Client build failed: {exc}")
        return 3

    model_override = resolved_model_from_env()
    if model_override:
        model_id = model_override
    else:
        fallback_model = profile.model_id or "lmstudio"
        model_id = discover_model_id(resolved_base_url, api_key, fallback_model)

    class Person(BaseModel):
        name: str
        age: int

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Ana is 34 years old."}],
            response_model=Person,
        )
        print(response)
    except Exception as exc:
        print(f"Sample request failed: {exc}")
        return 4

    print(f"Client ready for provider: {profile.provider_id}")
    print(f"Base URL: {resolved_base_url}")
    print(f"Mode: {mode.name}")
    print(f"Model: {model_id}")
    return 0


__all__ = [
    "LMSTUDIO_PROVIDER_PREFIX",
    "LMSTUDIO_DEFAULT_BASE_URL",
    "LMSTUDIO_DEFAULT_MODE",
    "LMSTUDIO_DEFAULT_API_KEY",
    "build_client",
    "discover_model_id",
    "is_lmstudio_provider",
    "probe_connection",
    "normalize_base_url",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
