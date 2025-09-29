# src/simple_or_agent/instructor_based/reasoning_models.py
# Defines Pydantic models that describe reasoning agent steps.
# Exists to isolate schema helpers from the core reasoning agent loop.
# RELEVANT FILES: src/simple_or_agent/instructor_based/reasoning_agent.py, src/simple_or_agent/instructor_based/tools.py

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class NextAction(str, Enum):
    """Allowed directives emitted by the LLM."""

    CONTINUE = "continue"
    VALIDATE = "validate"
    FINAL_ANSWER = "final_answer"
    RESET = "reset"


class ReasoningStep(BaseModel):
    """Single reasoning step returned by the LLM."""

    title: Optional[str] = Field(None, description="Short step title.")
    action: Optional[str] = Field(None, description="Planned action written in first person.")
    result: Optional[str] = Field(None, description="Outcome summary for the step.")
    reasoning: Optional[str] = Field(None, description="Why this step matters.")
    tool: Optional[str] = Field(None, description="Tool name when you need external help.")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="Arguments for the selected tool.")
    next_action: Optional[NextAction] = Field(None, description="continue, validate, final_answer, or reset.")
    confidence: Optional[float] = Field(None, description="Confidence score between 0.0 and 1.0.")

    @staticmethod
    def _flatten_text(value: Any) -> Optional[str]:
        """Handle providers that wrap string fields in arrays."""
        if isinstance(value, list):
            items = [item for item in value if item not in (None, "")]
            return "\n".join(str(item) for item in items) if items else None
        return value

    @field_validator("title", "action", "result", "reasoning", "tool", mode="before")
    def _coerce_text_fields(cls, value: Any) -> Optional[str]:
        return cls._flatten_text(value)

    @field_validator("next_action", mode="before")
    def _coerce_next_action(cls, value: Any) -> Optional[NextAction]:
        if isinstance(value, list):
            value = next((item for item in value if item is not None), None)
        if isinstance(value, str):
            try:
                return NextAction(value)
            except ValueError:
                return value
        return value

    @field_validator("confidence", mode="before")
    def _coerce_confidence(cls, value: Any) -> Optional[float]:
        if isinstance(value, list):
            value = next((item for item in value if item is not None), None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


class ReasoningSteps(BaseModel):
    """Ordered reasoning steps returned by the agent."""

    reasoning_steps: List[ReasoningStep] = Field(..., description="Ordered reasoning steps.")


__all__ = [
    "NextAction",
    "ReasoningStep",
    "ReasoningSteps",
]
