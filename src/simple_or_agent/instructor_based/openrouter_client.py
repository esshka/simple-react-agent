# src/simple_or_agent/instructor_based/openrouter_client.py
# Builds Instructor clients for the OpenRouter API.
# Exists to isolate OpenRouter-specific helpers and CLI tools.
# RELEVANT FILES: src/simple_or_agent/instructor_based/lmstudio_client.py, src/simple_or_agent/instructor_based/agent.py, src/simple_or_agent/instructor_based/provider_profiles.py

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ''}:
    project_src = Path(__file__).resolve().parent.parent.parent
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

import os
from typing import Any, List, Optional

import instructor
from instructor import Mode
from pydantic import BaseModel

from simple_or_agent.instructor_based.provider_profiles import resolve_profile, resolved_model_from_env

DEFAULT_OPENROUTER_PROVIDER = "openrouter/openai/gpt-oss-20b"
PROVIDER_ENV = "INSTRUCTOR_PROVIDER_ID"
API_KEY_ENV = "INSTRUCTOR_API_KEY"
MODE_ENV = "INSTRUCTOR_MODE"
FALLBACK_KEY_ENV = "OPENROUTER_API_KEY"


def build_client(api_key: str, provider_id: Optional[str] = None, mode: Optional[Mode] = None) -> Any:
    """Create an Instructor client that talks to OpenRouter."""
    resolved_provider = provider_id or DEFAULT_OPENROUTER_PROVIDER
    extra = {}
    normalized_mode = _normalize_mode(mode)
    if normalized_mode:
        extra["mode"] = normalized_mode
    return instructor.from_provider(resolved_provider, api_key=api_key, **extra)


def provider_model_hint(provider_id: str) -> str:
    """Return the model name implied by an OpenRouter provider id."""
    if "/" in provider_id:
        return provider_id.split("/", 1)[1]
    return provider_id


def is_openrouter(provider_id: str) -> bool:
    """Return True when the provider string targets OpenRouter."""
    return provider_id.startswith("openrouter/")


def run_example(
    api_key: str,
    provider_id: Optional[str] = None,
    mode: Optional[Mode] = None,
    model_id: Optional[str] = None,
) -> int:
    if mode is None:
        mode = Mode.TOOLS
    """Build a client and run the sample OpenRouter request used in manuals."""
    resolved_provider = provider_id or DEFAULT_OPENROUTER_PROVIDER
    try:
        client = build_client(api_key=api_key, provider_id=resolved_provider, mode=mode)
    except Exception as exc:
        print(f"Client build failed: {exc}")
        return 3

    resolved_model = model_id or provider_model_hint(resolved_provider)

    class Person(BaseModel):
        name: str
        age: int

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": "Ana is 34 years old."}],
            response_model=Person,
        )
    except Exception as exc:
        print(f"Sample request failed: {exc}")
        return 4

    persons = response if isinstance(response, list) else [response]
    if not persons:
        print("Sample request failed: OpenRouter returned no items.")
        return 5

    for index, person in enumerate(persons, start=1):
        print(f"Person #{index}: name={person.name!r} age={person.age}")

    print(f"Client ready for provider: {resolved_provider}")
    print(f"Mode: {mode.name if mode else 'provider default'}")
    print(f"Model: {resolved_model}")
    return 0


def _resolve_api_key() -> Optional[str]:
    value = os.getenv(API_KEY_ENV) or os.getenv(FALLBACK_KEY_ENV)
    if not value:
        return None
    return value.strip() or None


def _resolve_provider() -> Optional[str]:
    value = os.getenv(PROVIDER_ENV)
    if not value:
        return None
    trimmed = value.strip()
    return trimmed or None


def _normalize_mode(value: Optional[Mode]) -> Optional[Mode]:
    """Coerce unsupported OpenRouter modes into the closest valid value."""
    if value == Mode.JSON_SCHEMA:
        return Mode.JSON
    return value


def _resolve_mode() -> Optional[Mode]:
    raw = os.getenv(MODE_ENV)
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    print(f"Candidate: {candidate}")
    try:
        return _normalize_mode(Mode[candidate.upper()])
    except KeyError:
        names = ", ".join(member.name for member in Mode)
        print(f"Unknown INSTRUCTOR_MODE '{raw}'. Valid options: {names}")
        return None


def resolve_api_key_from_env() -> Optional[str]:
    """Return the OpenRouter API key configured via environment variables."""
    return _resolve_api_key()


def resolve_provider_from_env() -> Optional[str]:
    """Return the OpenRouter provider override configured via environment variables."""
    return _resolve_provider()


def resolve_mode_from_env() -> Optional[Mode]:
    """Return the OpenRouter mode override configured via environment variables."""
    return _resolve_mode()


def normalize_mode(value: Optional[Mode]) -> Optional[Mode]:
    """Expose the OpenRouter mode normalization logic for reuse."""
    return _normalize_mode(value)


def main() -> int:
    profile = resolve_profile()
    if not is_openrouter(profile.provider_id):
        profile = resolve_profile("openrouter")

    provider = _resolve_provider() or profile.provider_id or DEFAULT_OPENROUTER_PROVIDER
    api_key = _resolve_api_key() or profile.default_api_key
    if not api_key:
        print("Client build failed: Missing API key. Set INSTRUCTOR_API_KEY or OPENROUTER_API_KEY.")
        return 1

    print(f"Profile mode: {profile.mode}")
    print(f"Resolve mode: {_resolve_mode()}")
    mode = _normalize_mode(_resolve_mode() or profile.mode)
    print(f"Mode: {mode}")
    model_override = resolved_model_from_env() or profile.model_id
    print(f"Model override: {model_override}")
    return run_example(
        api_key=api_key,
        provider_id=provider,
        mode=mode,
        model_id=model_override,
    )


__all__ = [
    "DEFAULT_OPENROUTER_PROVIDER",
    "build_client",
    "is_openrouter",
    "provider_model_hint",
    "run_example",
    "main",
    "resolve_api_key_from_env",
    "resolve_provider_from_env",
    "resolve_mode_from_env",
    "normalize_mode",
]


if __name__ == "__main__":
    raise SystemExit(main())
