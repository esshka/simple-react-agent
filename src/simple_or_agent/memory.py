# src/simple_or_agent/memory.py
# Conversation memory with scoped context and simple summarization.
# This exists to keep agent context focused and compact across loops.
# RELEVANT FILES: src/simple_or_agent/orchestrator.py,src/simple_or_agent/next_agent.py,src/simple_or_agent/simple_agent.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Set


@dataclass
class MemoryPolicy:
    """Config for retention and context shaping.

    keep_last: how many recent items to keep verbatim per selection.
    max_chars: soft budget for composed context passed to an agent.
    summarize_beyond: if True, older items are summarized when over budget.
    """

    keep_last: int = 6
    max_chars: int = 4000
    summarize_beyond: bool = True


@dataclass
class MemoryItem:
    role: str
    content: str
    tags: Set[str] = field(default_factory=set)
    scope: str = "session"  # "session" | "app" | "ephemeral"


class ConversationMemory:
    """Small, explicit memory store with tagging and optional summarization.

    Simplicity first. We keep a list of items with tags and return filtered,
    sized context strings. Older content can be summarized when over budget.
    """

    def __init__(
        self,
        policy: Optional[MemoryPolicy] = None,
        summarizer: Optional[Callable[[str, int], str]] = None,
    ) -> None:
        self.policy = policy or MemoryPolicy()
        self._items: List[MemoryItem] = []
        self._summarizer = summarizer

    # --- mutation ---
    def add(self, role: str, content: str, tags: Optional[Iterable[str]] = None, scope: str = "session") -> None:
        if not content:
            return
        self._items.append(MemoryItem(role=role, content=content, tags=set(tags or []), scope=scope))

    def clear_ephemeral(self) -> None:
        self._items = [it for it in self._items if it.scope != "ephemeral"]

    def clear_all(self) -> None:
        self._items.clear()

    # --- selection ---
    def _select_items(self, include_tags: Optional[Sequence[str]] = None) -> List[MemoryItem]:
        if not include_tags:
            return list(self._items)
        want = set(include_tags)
        return [it for it in self._items if it.tags & want]

    def _join(self, items: Sequence[MemoryItem]) -> str:
        # Prefix with role for light provenance while staying compact.
        lines: List[str] = []
        for it in items:
            lines.append(f"{it.role}: {it.content.strip()}")
        return "\n".join(lines)

    def build_context(self, include_tags: Optional[Sequence[str]] = None, max_chars: Optional[int] = None) -> str:
        """Return a compact, ordered context string filtered by tags.

        - Keeps last N items verbatim, summarizes older ones if needed.
        - Respects max_chars budget (defaults to policy.max_chars).
        """
        items = self._select_items(include_tags)
        if not items:
            return ""
        keep_last = max(1, int(self.policy.keep_last))
        max_chars = int(max_chars or self.policy.max_chars)

        # Fast path: short enough
        raw = self._join(items)
        if len(raw) <= max_chars:
            return raw

        # Split into old vs recent
        head = items[:-keep_last]
        tail = items[-keep_last:]

        # Summarize head if allowed, else trim
        head_txt = self._join(head)
        tail_txt = self._join(tail)

        if not head_txt:
            return tail_txt[-max_chars:]

        budget_for_head = max(0, max_chars - len(tail_txt) - 50)  # keep small gap
        if budget_for_head <= 0:
            return tail_txt[-max_chars:]

        if self.policy.summarize_beyond and self._summarizer:
            try:
                head_sum = self._summarizer(head_txt, budget_for_head)
            except Exception:
                head_sum = self._naive_summarize(head_txt, budget_for_head)
        else:
            head_sum = self._naive_summarize(head_txt, budget_for_head)

        parts = []
        if head_sum:
            parts.append(f"summary: {head_sum.strip()}")
        if tail_txt:
            parts.append(tail_txt)
        out = "\n".join(parts)
        return out[-max_chars:]

    # --- helpers ---
    @staticmethod
    def _naive_summarize(text: str, max_chars: int) -> str:
        if max_chars <= 0 or not text:
            return ""
        # Keep start and end snippets to preserve context and recency.
        if len(text) <= max_chars:
            return text
        take = max_chars // 2
        return text[:take].rstrip() + " … " + text[-take:].lstrip()


__all__ = ["MemoryPolicy", "MemoryItem", "ConversationMemory"]

