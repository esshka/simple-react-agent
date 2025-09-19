# src/simple_or_agent/orchestrator.py
# Orchestrates multi-agent flows with scoped memory and context shaping.
# This exists to keep each agent's working context focused and efficient.
# RELEVANT FILES: src/simple_or_agent/memory.py,src/simple_or_agent/next_agent.py,src/simple_or_agent/react_agent.py

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from .openrouter_client import OpenRouterError
from .simple_agent import AskResult
from .memory import ConversationMemory


class Orchestrator:
    """Coordinates subagents and decides what context to pass to each.

    Simplicity: pass only what is needed. Keep short, relevant context.
    Memory stores task, plan, and transcript summaries.
    """

    def __init__(self, memory: ConversationMemory) -> None:
        self.memory = memory

    # --- planning ---
    def plan(self, planner, task: str, planner_prompt_builder: Callable[[str, str], str]) -> AskResult:
        if not task:
            raise OpenRouterError("Empty task")

        # Record user task and prior notes; keep tags for filtering.
        self.memory.add("user", task, tags={"task"})

        # Provide compact prior context to the planner if any summary exists.
        prior = self.memory.build_context(include_tags=["summary", "notes"], max_chars=1200)
        prompt = planner_prompt_builder(task, prior)

        res = planner.ask(prompt)
        plan_text = (res.content or "").strip()
        if plan_text:
            self.memory.add("assistant", plan_text, tags={"plan"})
        return res

    # --- execution ---
    def execute(self, executor, task: str, plan_text: str, exec_prompt_builder: Callable[[str, str, str], str], progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> AskResult:
        if not task:
            raise OpenRouterError("Empty task")
        self.memory.add("assistant", plan_text, tags={"plan"})

        # Offer a short notes/summary context to the executor.
        notes = self.memory.build_context(include_tags=["summary", "notes"], max_chars=1200)
        goal = exec_prompt_builder(task, plan_text, notes)

        # Wrap progress callback to record useful context each iteration.
        def _wrap_cb(event: str, data: Dict[str, Any]) -> None:
            try:
                if event == "think":
                    thought = (data.get("thought") or "").strip()
                    action = data.get("action") or {}
                    if thought:
                        self.memory.add("assistant", thought, tags={"think", "notes"})
                    # Record brief action hint for traceability
                    an = (action.get("name") or action.get("type") or "").strip()
                    say = (action.get("say") or "").strip()
                    if an or say:
                        self.memory.add("assistant", f"action: {an or 'nl'} {say}", tags={"action", "notes"})
                elif event == "tool":
                    name = (data.get("name") or "").strip()
                    args = data.get("args") or {}
                    try:
                        import json as _json
                        args_s = _json.dumps(args, ensure_ascii=False)
                    except Exception:
                        args_s = str(args)
                    self.memory.add("tool", f"{name} {args_s}", tags={"action"})
                elif event == "observe":
                    obs = str(data.get("observation") or "").strip()
                    if obs:
                        # Keep observation concise; ConversationMemory will compress when needed.
                        self.memory.add("system", obs[:1500], tags={"observation", "facts"})
                elif event == "judge":
                    dec = data.get("decision")
                    if dec == "final":
                        final = (data.get("final") or "").strip()
                        if final:
                            self.memory.add("assistant", final, tags={"final", "summary"})
            finally:
                if progress_cb:
                    try:
                        progress_cb(event, data)
                    except Exception:
                        pass

        res = executor.ask(goal, progress_cb=_wrap_cb)
        content = (res.content or "").strip()
        if content:
            # Store a transcript+final as a single item under transcript.
            self.memory.add("assistant", content, tags={"transcript"})
            # Add a compact run summary for future iterations.
            summary = self._naive_clip(content, 1200)
            if summary:
                self.memory.add("assistant", summary, tags={"summary"})
        return res

    # --- full run ---
    def ask(self, planner, executor, task: str, planner_builder: Callable[[str, str], str], exec_builder: Callable[[str, str, str], str], progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> Tuple[AskResult, AskResult]:
        plan_res = self.plan(planner, task, planner_builder)
        if progress_cb:
            try:
                progress_cb("plan", {"text": (plan_res.content or "").strip()})
            except Exception:
                pass
        exec_res = self.execute(executor, task, plan_res.content, exec_builder, progress_cb)
        return plan_res, exec_res

    # --- helpers ---
    @staticmethod
    def _naive_clip(text: str, max_chars: int) -> str:
        if not text or max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        take = max_chars // 2
        return text[:take].rstrip() + " … " + text[-take:].lstrip()


__all__ = ["Orchestrator"]
