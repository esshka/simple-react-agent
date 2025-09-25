# src/simple_or_agent/instructor_based/calculator_tool.py
# Provides the calculator tool and helpers for Instructor agents.
# Exists to keep arithmetic evaluation logic separate from the main agent loop.
# RELEVANT FILES: src/simple_or_agent/instructor_based/agent.py, src/simple_or_agent/instructor_based/tools.py, src/simple_or_agent/instructor_based/openrouter_client.py

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ''}:
    project_src = Path(__file__).resolve().parent.parent.parent
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

import ast
import math
import operator as op
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, ConfigDict

from simple_or_agent.instructor_based.tools import ToolSpec


class Calculate(BaseModel):
    """Inputs for the calculator tool."""
    expr: str

OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

ALLOWED_FUNCS: Dict[str, Callable[..., float]] = {
    # Keep the whitelist small so we only expose safe helpers.
    "log": lambda value, base=10: math.log(value, base),
    "log10": math.log10,
    "ln": math.log,
    "sqrt": math.sqrt,
}


def _eval_expression(node: ast.AST) -> float:
    """Evaluate a safe arithmetic AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_expression(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left = _eval_expression(node.left)
        right = _eval_expression(node.right)
        return OPS[type(node.op)](left, right)
    if isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("Unsupported expression")
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported expression")
        name = node.func.id
        func = ALLOWED_FUNCS.get(name)
        if func is None:
            raise ValueError("Unsupported expression")
        args: List[float] = [_eval_expression(arg) for arg in node.args]
        try:
            return float(func(*args))
        except TypeError as exc:
            raise ValueError("Unsupported expression") from exc
    raise ValueError("Unsupported expression")


def calculate(raw_args: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and evaluate an arithmetic expression."""
    args = Calculate(**raw_args)
    parsed = ast.parse(args.expr, mode="eval")
    value = _eval_expression(parsed.body)
    return {"expr": args.expr, "value": value}


def build_calculator_tool() -> ToolSpec:
    """Return the ToolSpec wired to the calculator handler."""
    return ToolSpec(
        name="calculate",
        description="Evaluate a mathematical expression.",
        args_model=Calculate,
        handler=calculate,
        parameters={"expr": "string expression to evaluate"},
    )


__all__ = [
    "Calculate",
    "OPS",
    "build_calculator_tool",
    "calculate",
]
