# src/simple_or_agent/instructor_based/instructor_client.py
# Builds Instructor clients for OpenRouter or LMStudio connections.
# Provides shared client helpers so agents can talk to their providers.
# RELEVANT FILES: src/simple_or_agent/instructor_based/agent.py, src/simple_or_agent/instructor_based/prompt_manager.py, src/simple_or_agent/instructor_based/tools.py

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import instructor
from instructor import Mode
from pydantic import BaseModel
from openai import OpenAI

DEFAULT_PROVIDER_ID = "openrouter/qwen/qwen3-next-80b-a3b-instruct"

PROVIDER_ENV = "INSTRUCTOR_PROVIDER_ID"
BASE_URL_ENV = "INSTRUCTOR_BASE_URL"
API_KEY_ENV = "INSTRUCTOR_API_KEY"
MODE_ENV = "INSTRUCTOR_MODE"
FALLBACK_KEY_ENV = "OPENROUTER_API_KEY"

LMSTUDIO_HOST_ENV = "LMSTUDIO_BASE_URL"
LMSTUDIO_MODEL_ENV = "LMSTUDIO_MODEL_ID"
LMSTUDIO_DEFAULT_HOST = "http://100.66.248.94:1234/v1"
LMSTUDIO_DEFAULT_PROVIDER = "openai/lmstudio"
LMSTUDIO_DEFAULT_API_KEY = "lm-studio"
LMSTUDIO_DEFAULT_MODE = Mode.JSON_SCHEMA


def build_instructor_client(
    api_key: Optional[str],
    provider_id: Optional[str] = None,
    base_url: Optional[str] = None,
    mode: Optional[Mode] = None,
) -> Any:
    """Return a ready-to-use Instructor client for the desired provider."""
    resolved_provider = provider_id or DEFAULT_PROVIDER_ID
    resolved_key = api_key or _resolve_api_key()
    if not resolved_key:
        raise ValueError(
            "api_key is required. Set OPENROUTER_API_KEY or INSTRUCTOR_API_KEY before building the client."
        )

    resolved_mode = mode
    print(f"Building instructor client for provider: {resolved_provider}")
    if base_url:
        print(f"Using base_url: {base_url}")
    if resolved_mode:
        print(f"Using mode: {resolved_mode.name}")

    if base_url:
        # Work around instructor bug by patching an explicit OpenAI client.
        openai_client = OpenAI(api_key=resolved_key, base_url=base_url)
        if resolved_mode:
            return instructor.from_openai(openai_client, mode=resolved_mode)
        return instructor.from_openai(openai_client)

    extra: Dict[str, Any] = {}
    if resolved_mode:
        extra["mode"] = resolved_mode
    return instructor.from_provider(resolved_provider, api_key=resolved_key, **extra)


__all__ = ["build_instructor_client", "DEFAULT_PROVIDER_ID"]


def _resolve_api_key() -> Optional[str]:
    """Read the API key from the supported environment variables."""
    value = os.getenv(API_KEY_ENV) or os.getenv(FALLBACK_KEY_ENV)
    if not value:
        return None
    return value.strip() or None


def _resolve_provider() -> Optional[str]:
    """Return the provider override when specified."""
    value = os.getenv(PROVIDER_ENV)
    if not value:
        return None
    return value.strip() or None


def _resolve_base_url() -> Optional[str]:
    """Return the base URL override when specified."""
    value = os.getenv(BASE_URL_ENV)
    if not value:
        return None
    return value.strip() or None


def _resolve_mode() -> Optional[Mode]:
    """Map INSTRUCTOR_MODE to an Instructor Mode enum value."""
    raw = os.getenv(MODE_ENV)
    if not raw:
        return None

    candidate = raw.strip().upper()
    try:
        return Mode[candidate]
    except KeyError:
        names = ", ".join(member.name for member in Mode)
        print(f"Unknown INSTRUCTOR_MODE '{raw}'. Valid options: {names}")
        return None


def _default_lmstudio_host() -> str:
    """Return the LMStudio host URL without the /v1 suffix."""
    value = os.getenv(LMSTUDIO_HOST_ENV)
    if value:
        return value.strip() or LMSTUDIO_DEFAULT_HOST
    return LMSTUDIO_DEFAULT_HOST


def _default_lmstudio_base_url() -> str:
    """Return the LMStudio base URL that Instructor expects."""
    host = _default_lmstudio_host().rstrip("/")
    if host.endswith("/v1"):
        return host
    return f"{host}/v1"


def _default_lmstudio_provider() -> str:
    """Return the provider id for LMStudio."""
    model_id = os.getenv(LMSTUDIO_MODEL_ENV)
    if model_id:
        trimmed = model_id.strip()
        if trimmed:
            if trimmed.startswith("openai/"):
                return trimmed
            return f"openai/{trimmed}"
    return LMSTUDIO_DEFAULT_PROVIDER


def _discover_model_id(base_url: Optional[str], api_key: str, fallback: str) -> str:
    """Ask the server for its models so we can pick a sensible default."""
    if not base_url:
        return fallback

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
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


def _probe_connection(base_url: Optional[str], api_key: str) -> bool:
    """Return True if we can reach the configured LMStudio endpoint."""
    if not base_url:
        return True
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        client.models.list()
        return True
    except Exception as exc:
        print(f"LMStudio health check failed: {exc}")
        return False


def main() -> int:
    """Allow quick manual client checks from the command line."""
    provider = _resolve_provider() or _default_lmstudio_provider()
    base_url = _resolve_base_url() or _default_lmstudio_base_url()
    mode = _resolve_mode() or LMSTUDIO_DEFAULT_MODE
    api_key = _resolve_api_key() or LMSTUDIO_DEFAULT_API_KEY

    if not _probe_connection(base_url, api_key):
        print(
            "Could not reach LMStudio. Confirm the server is running and accessible at "
            f"{base_url}."
        )
        return 2

    try:
        client = build_instructor_client(
            api_key=api_key,
            provider_id=provider,
            base_url=base_url,
            mode=mode,
        )
    except Exception as exc:
        print(f"Client build failed: {exc}")
        return 3

    class Person(BaseModel):
        name: str
        age: int

    try:
        # Ask LMStudio to map a plain sentence into structured data.
        response = client.chat.completions.create(
            model=_discover_model_id(base_url, api_key, provider.split("/", 1)[1] if "/" in provider else provider),
            messages=[{"role": "user", "content": "Ana is 34 years old."}],
            response_model=Person,
        )
        print(response)
    except Exception as exc:
        print(f"Sample request failed: {exc}")
        return 4

    print(f"Client ready for provider: {provider}")
    print(f"Base URL: {base_url}")
    print(f"Mode: {mode.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
