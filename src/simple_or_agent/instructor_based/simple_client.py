# src/simple_or_agent/instructor_based/simple_client.py
# Builds a lightweight OpenAI client for raw OpenRouter chat completions.
# Exists so callers can fetch plain text without Instructor structured output.
# RELEVANT FILES: src/simple_or_agent/instructor_based/openrouter_client.py, src/simple_or_agent/instructor_based/provider_profiles.py, src/simple_or_agent/instructor_based/agent.py

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ''}:
    project_src = Path(__file__).resolve().parent.parent.parent
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from openai import OpenAI

from simple_or_agent.instructor_based import openrouter_client
from simple_or_agent.instructor_based.provider_profiles import resolve_profile


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
REFERER_ENV = "OPENROUTER_HTTP_REFERER"
TITLE_ENV = "OPENROUTER_SITE_TITLE"


@dataclass(frozen=True)
class RawClientSettings:
    """Resolved inputs needed to reach OpenRouter for raw chat output."""

    api_key: str
    provider_id: str
    model_id: str
    base_url: str
    referer: Optional[str]
    site_title: Optional[str]


def _strip_or_none(value: Optional[str]) -> Optional[str]:
    """Return the string stripped of whitespace or None when empty."""
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _resolve_openrouter_profile():
    """Always return the OpenRouter profile so defaults stay consistent."""
    profile = resolve_profile()
    if openrouter_client.is_openrouter(profile.provider_id):
        return profile
    return resolve_profile("openrouter")


def _resolve_settings(
    api_key: Optional[str],
    provider_id: Optional[str],
    model_id: Optional[str],
    base_url: Optional[str],
    referer: Optional[str],
    site_title: Optional[str],
) -> RawClientSettings:
    """Resolve all optional inputs into a complete configuration."""
    profile = _resolve_openrouter_profile()

    resolved_api_key = (
        _strip_or_none(api_key)
        or openrouter_client.resolve_api_key_from_env()
        or _strip_or_none(profile.default_api_key)
    )
    if not resolved_api_key:
        raise ValueError(
            "Missing OpenRouter API key. Set INSTRUCTOR_API_KEY or OPENROUTER_API_KEY."
        )

    resolved_provider = (
        _strip_or_none(provider_id)
        or profile.provider_id
        or openrouter_client.DEFAULT_OPENROUTER_PROVIDER
    )
    if not openrouter_client.is_openrouter(resolved_provider):
        resolved_provider = openrouter_client.DEFAULT_OPENROUTER_PROVIDER

    resolved_model = _strip_or_none(model_id) or _strip_or_none(profile.model_id)
    if not resolved_model:
        resolved_model = openrouter_client.provider_model_hint(resolved_provider)

    resolved_base_url = _strip_or_none(base_url) or profile.base_url or DEFAULT_BASE_URL

    resolved_referer = _strip_or_none(referer) or _strip_or_none(os.getenv(REFERER_ENV))
    resolved_title = _strip_or_none(site_title) or _strip_or_none(os.getenv(TITLE_ENV))

    return RawClientSettings(
        api_key=resolved_api_key,
        provider_id=resolved_provider,
        model_id=resolved_model,
        base_url=resolved_base_url,
        referer=resolved_referer,
        site_title=resolved_title,
    )


def _build_openai_client(settings: RawClientSettings) -> OpenAI:
    """Create the OpenAI client with optional OpenRouter ranking headers."""
    headers: Dict[str, str] = {}
    if settings.referer:
        headers["HTTP-Referer"] = settings.referer
    if settings.site_title:
        headers["X-Title"] = settings.site_title

    client_kwargs: Dict[str, Any] = {
        "api_key": settings.api_key,
        "base_url": settings.base_url,
    }
    if headers:
        client_kwargs["default_headers"] = headers
    return OpenAI(**client_kwargs)


@dataclass
class RawTextChatClient:
    """Thin wrapper around OpenAI chat completions that returns plain text."""

    _client: OpenAI
    model_id: str
    provider_id: str

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.1,
        **completion_kwargs: Any,
    ) -> str:
        """Send messages to OpenRouter and return the first response content."""
        if not messages:
            raise ValueError("messages must not be empty")

        payload: Dict[str, Any] = dict(completion_kwargs)
        payload["model"] = self.model_id
        payload["messages"] = list(messages)
        payload.setdefault("temperature", temperature)

        response = self._client.chat.completions.create(**payload)
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("OpenRouter returned no choices")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message:
            content = getattr(message, "content", None)
        else:
            content = getattr(first_choice, "text", None)

        if content is None:
            raise RuntimeError("OpenRouter response did not include text content")

        return content


def build_chat_client(
    api_key: Optional[str] = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
    base_url: Optional[str] = None,
    referer: Optional[str] = None,
    site_title: Optional[str] = None,
) -> RawTextChatClient:
    """Return a ready-to-use RawTextChatClient using the provided overrides."""
    settings = _resolve_settings(
        api_key=api_key,
        provider_id=provider_id,
        model_id=model_id,
        base_url=base_url,
        referer=referer,
        site_title=site_title,
    )
    client = _build_openai_client(settings)
    return RawTextChatClient(client, settings.model_id, settings.provider_id)


def run_example(prompt: str = "What is the meaning of life?") -> str:
    """Quick manual test that prints the model answer for the given prompt."""
    chat_client = build_chat_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}]
    response = chat_client.chat(messages)
    print(response)
    return response


__all__ = [
    "DEFAULT_BASE_URL",
    "RawTextChatClient",
    "build_chat_client",
    "run_example",
]

if __name__ == "__main__":
    run_example()
