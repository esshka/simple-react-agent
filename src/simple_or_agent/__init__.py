# src/simple_or_agent/__init__.py
# Package exports and small utilities.
# This file centralizes simple helpers and re-exports public clients.
# RELEVANT FILES: src/simple_or_agent/lmstudio_client.py,src/simple_or_agent/openrouter_client.py,src/simple_or_agent/simple_agent.py

"""Package exports and small utilities."""

from __future__ import annotations

import re
from typing import List

from .lmstudio_client import LMStudioClient


def format_inline_citations(text: str) -> str:
    """Add minimal inline [n] citations from bare URLs and append Sources.

    - If the content already contains [n] citations or a Sources/References
      section, return it unchanged.
    - Otherwise, replace bare http(s) URLs with [n] markers in order of
      first appearance and append a Sources list mapping [n] -> URL.
    """
    if not isinstance(text, str) or not text.strip():
        return text or ""

    # If citations or sources already present, do not modify
    if re.search(r"\[[0-9]+\]", text):
        return text
    if re.search(r"(?mi)^\s*(sources|references)\b", text):
        return text

    url_re = re.compile(r"https?://[^\s\]\)\>\}\"']+")
    seen: List[str] = []

    def _repl(m: re.Match) -> str:
        url = m.group(0)
        trimmed = url.rstrip(".,);:'\"]")
        suffix = url[len(trimmed):]
        if trimmed not in seen:
            seen.append(trimmed)
        idx = seen.index(trimmed) + 1
        return f"[{idx}]" + suffix

    new_text = url_re.sub(_repl, text)
    if not seen:
        return text

    sources_lines = [f"[{i}] {u}" for i, u in enumerate(seen, 1)]
    return new_text.rstrip() + "\n\nSources\n" + "\n".join(sources_lines) + "\n"


__all__ = [
    "format_inline_citations",
    "LMStudioClient",
]
