# src/simple_or_agent/instructor_based/openrouter_client.py
# Builds Instructor clients for the OpenRouter API.
# Exists to isolate OpenRouter-specific helpers and CLI tools.
# RELEVANT FILES: src/simple_or_agent/instructor_based/lmstudio_client.py, src/simple_or_agent/instructor_based/agent.py, src/simple_or_agent/instructor_based/provider_profiles.py

from __future__ import annotations

import logging
from pathlib import Path
import sys

if __package__ in {None, ''}:
    project_src = Path(__file__).resolve().parent.parent.parent
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

import json
import os
from typing import Any, Dict, List, Optional

import instructor
from instructor import Mode
from pydantic import BaseModel

from simple_or_agent.instructor_based.provider_profiles import resolve_profile

# Enable Instructor debug logs so we can inspect OpenRouter traffic.
logging.basicConfig(level=logging.DEBUG)

DEFAULT_OPENROUTER_PROVIDER = "openrouter/openai/gpt-oss-20b"
API_KEY_ENV = "INSTRUCTOR_API_KEY"
FALLBACK_KEY_ENV = "OPENROUTER_API_KEY"


def _describe(value: Any) -> Any:
    """Convert complex objects into JSON-friendly data."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if isinstance(value, dict):
        return {key: _describe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe(item) for item in value]
    return value


def build_client(api_key: str, provider_id: Optional[str] = None, mode: Optional[Mode] = None) -> Any:
    """Create an Instructor client that talks to OpenRouter."""
    resolved_provider = provider_id or DEFAULT_OPENROUTER_PROVIDER
    extra: Dict[str, Any] = {}
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

    captured_kwargs: List[Dict[str, Any]] = []
    captured_responses: List[Any] = []

    def capture_kwargs(*args: Any, **kwargs: Any) -> None:
        payload: Dict[str, Any] = {}
        if args:
            payload["args"] = _describe(list(args))
        if kwargs:
            payload["kwargs"] = _describe(kwargs)
        captured_kwargs.append(payload)

    def capture_response(response: Any) -> None:
        captured_responses.append(_describe(response))

    client.on("completion:kwargs", capture_kwargs)
    client.on("completion:response", capture_response)

    resolved_model = model_id or provider_model_hint(resolved_provider)

    class Person(BaseModel):
        name: str
        age: int

    request_messages: List[Dict[str, Any]] = [
        {"role": "user", "content": "Ana is 34 years old."}
    ]

    request_payload: Dict[str, Any] = {
        "model": resolved_model,
        "messages": request_messages,
    }
    if mode is not None:
        request_payload["mode"] = mode.name

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-30b-a3b",
            messages=request_messages,
            response_model=Person,
            extra_body={"provider": {"require_parameters": True}}
        )
    except Exception as exc:
        print(f"Sample request failed: {exc}")
        return 4

    persons = response if isinstance(response, list) else [response]
    history: List[Dict[str, Any]] = request_messages.copy()
    person_payloads: List[Dict[str, Any]] = []

    if not persons:
        print("Sample request failed: OpenRouter returned no items.")
        return 5

    for index, person in enumerate(persons, start=1):
        print(f"Person #{index}: name={person.name!r} age={person.age}")
        dump = person.model_dump()
        history.append(
            {
                "role": "assistant",
                "content": dump,
            }
        )
        person_payloads.append(dump)

    print(f"Client ready for provider: {resolved_provider}")
    print(f"Mode: {mode.name if mode else 'provider default'}")
    print(f"Model: {resolved_model}")
    print("Raw request payload:")
    print(json.dumps(request_payload, indent=2))
    print("Parsed response payload:")
    print(
        json.dumps(
            person_payloads if len(person_payloads) > 1 else person_payloads[0],
            indent=2,
        )
    )
    print("Message history:")
    for idx, message in enumerate(history, start=1):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        print(f"{idx}. {role}: {content}")
    print("Raw message history:")
    print(json.dumps(history, indent=2, default=str))
    if captured_kwargs:
        print("Captured completion kwargs:")
        print(json.dumps(captured_kwargs, indent=2, default=str))
    if captured_responses:
        print("Captured completion responses:")
        print(json.dumps(captured_responses, indent=2, default=str))
    return 0


def _resolve_api_key() -> Optional[str]:
    value = os.getenv(API_KEY_ENV) or os.getenv(FALLBACK_KEY_ENV)
    if not value:
        return None
    return value.strip() or None


def _normalize_mode(value: Optional[Mode]) -> Optional[Mode]:
    """Coerce unsupported OpenRouter modes into the closest valid value."""
    if value == Mode.JSON_SCHEMA:
        return Mode.JSON
    return value


def resolve_api_key_from_env() -> Optional[str]:
    """Return the OpenRouter API key configured via environment variables."""
    return _resolve_api_key()


def normalize_mode(value: Optional[Mode]) -> Optional[Mode]:
    """Expose the OpenRouter mode normalization logic for reuse."""
    return _normalize_mode(value)


def main() -> int:
    profile = resolve_profile()
    if not is_openrouter(profile.provider_id):
        profile = resolve_profile("openrouter")

    provider = profile.provider_id or DEFAULT_OPENROUTER_PROVIDER
    api_key = _resolve_api_key() or profile.default_api_key
    if not api_key:
        print("Client build failed: Missing API key. Set INSTRUCTOR_API_KEY or OPENROUTER_API_KEY.")
        return 1

    print(f"Profile mode: {profile.mode}")
    mode = _normalize_mode(profile.mode)
    print(f"Mode: {mode}")
    model_override = profile.model_id
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
    "normalize_mode",
]


if __name__ == "__main__":
    raise SystemExit(main())
