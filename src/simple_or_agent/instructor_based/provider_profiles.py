# src/simple_or_agent/instructor_based/provider_profiles.py
# Loads Instructor provider profiles and helps apply them to the environment.
# Exists to keep provider defaults simple and let us switch via one-liners.
# RELEVANT FILES: src/simple_or_agent/instructor_based/providers.ini, src/simple_or_agent/instructor_based/agent.py, src/simple_or_agent/instructor_based/lmstudio_client.py

from __future__ import annotations

import configparser
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from instructor import Mode

PROFILE_ENV = "INSTRUCTOR_PROFILE"
MODEL_ENV = "INSTRUCTOR_MODEL_ID"
DEFAULT_PROFILE = "lmstudio"
CONFIG_PATH = Path(__file__).with_name("providers.ini")


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    provider_id: str
    base_url: Optional[str]
    mode: Optional[Mode]
    model_id: Optional[str]
    default_api_key: Optional[str]

    def env_overrides(self) -> Dict[str, Optional[str]]:
        """Return the environment values this profile controls."""
        return {
            PROFILE_ENV: self.name,
            "INSTRUCTOR_PROVIDER_ID": self.provider_id,
            "INSTRUCTOR_BASE_URL": self.base_url,
            "INSTRUCTOR_MODE": self.mode.name if self.mode else None,
            MODEL_ENV: self.model_id,
            "INSTRUCTOR_API_KEY": self.default_api_key,
        }


_PROFILES: Optional[Dict[str, ProviderProfile]] = None


def _load_profiles() -> Dict[str, ProviderProfile]:
    """Parse providers.ini into ProviderProfile objects."""
    global _PROFILES
    if _PROFILES is not None:
        return _PROFILES

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing provider config at {CONFIG_PATH}")

    parser = configparser.ConfigParser()
    parser.read(CONFIG_PATH, encoding="utf-8")

    profiles: Dict[str, ProviderProfile] = {}
    for section in parser.sections():
        normalized = section.strip().lower()
        raw_provider = parser.get(section, "provider_id", fallback="").strip()
        if not raw_provider:
            raise ValueError(f"Provider id missing for profile '{section}' in providers.ini")
        base_url = parser.get(section, "base_url", fallback="").strip() or None
        mode_name = parser.get(section, "mode", fallback="").strip().upper() or None
        if normalized == 'openrouter' and not mode_name:
            mode_name = 'JSON'
        if normalized == 'openrouter' and mode_name == 'JSON_SCHEMA':
            mode_name = 'JSON'
        model_id = parser.get(section, "model", fallback="").strip() or None
        default_key = parser.get(section, "default_api_key", fallback="").strip() or None
        mode = Mode[mode_name] if mode_name else None
        profiles[normalized] = ProviderProfile(
            name=normalized,
            provider_id=raw_provider,
            base_url=base_url,
            mode=mode,
            model_id=model_id,
            default_api_key=default_key,
        )

    if DEFAULT_PROFILE not in profiles:
        raise ValueError(f"Default profile '{DEFAULT_PROFILE}' is not defined in providers.ini")

    _PROFILES = profiles
    return profiles


def available_profiles() -> Iterable[str]:
    """Return the list of configured profile names."""
    return sorted(_load_profiles().keys())


def resolve_profile(name: Optional[str] = None) -> ProviderProfile:
    """Return the profile requested by name or the active environment profile."""
    candidates = _load_profiles()
    requested = name or os.getenv(PROFILE_ENV) or DEFAULT_PROFILE
    key = requested.strip().lower()
    profile = candidates.get(key)
    if profile is None:
        raise ValueError(f"Unknown profile '{requested}'. Try one of: {', '.join(available_profiles())}")
    return profile


def resolved_model_from_env() -> Optional[str]:
    """Return the explicit model override if present."""
    raw = os.getenv(MODEL_ENV)
    if not raw:
        return None
    trimmed = raw.strip()
    return trimmed or None


def format_shell_exports(profile: ProviderProfile) -> str:
    """Return shell commands that apply the profile env in one go."""
    lines = [f"# Active profile: {profile.name}"]
    for key, value in profile.env_overrides().items():
        if value:
            lines.append(f"export {key}={value}")
        else:
            lines.append(f"unset {key}")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Simple CLI for listing or switching profiles."""
    args = list(argv or sys.argv[1:])
    if not args:
        profile = resolve_profile()
        print(f"Active profile: {profile.name}")
        return 0

    command = args[0]
    if command == "list":
        print("Available profiles:")
        for name in available_profiles():
            print(f"- {name}")
        return 0

    if command == "use":
        if len(args) < 2:
            print("usage: provider_profiles.py use <profile>")
            return 1
        profile = resolve_profile(args[1])
        print(format_shell_exports(profile))
        return 0

    print("Supported commands: list, use <profile>")
    return 1


__all__ = [
    "PROFILE_ENV",
    "MODEL_ENV",
    "DEFAULT_PROFILE",
    "available_profiles",
    "format_shell_exports",
    "resolve_profile",
    "resolved_model_from_env",
]


if __name__ == "__main__":
    raise SystemExit(main())
