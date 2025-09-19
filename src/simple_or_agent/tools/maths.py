from sympy import sympify, solve, simplify, diff, integrate, Matrix, expand, factor
from typing import Dict, Any
import operator as op
import ast

from simple_or_agent.simple_agent import ToolSpec

# --- Basic Calculator Tool ---

def make_calc_tool() -> ToolSpec:
    """Create a safe calculator tool using a tiny AST evaluator."""
    allowed = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            return allowed[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("unsupported expression")

    def handler(args):
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return {"error": "empty_expression"}
        try:
            tree = ast.parse(expr, mode="eval")
            val = _eval(tree.body)  # type: ignore[arg-type]
            return {"expression": expr, "value": val}
        except Exception as e:
            return {"error": str(e)}

    params = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Arithmetic expression"}},
        "required": ["expression"],
        "additionalProperties": False,
    }
    return ToolSpec(name="calc", description="Evaluate basic arithmetic expression and return a JSON result", parameters=params, handler=handler)

# --- Core Algebra and Expression Tools ---

def make_sympy_solve_equation_tool() -> ToolSpec:
    """Solves algebraic equations."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            equations = [sympify(eq) for eq in args["equations"]]
            variables = [sympify(var) for var in args["variables"]]
            solution = solve(equations, variables)
            return {"solution": str(solution)}
        except Exception as e:
            return {"error": f"SymPy solve failed: {e}"}

    return ToolSpec(
        name="sympy_solve_equation",
        description="Solve a single or a system of algebraic equations for a set of variables.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "equations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of equations (as strings) to be solved. E.g., ['x**2 - 4 = 0']",
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of variables (as strings) to solve for. E.g., ['x']",
                },
            },
            "required": ["equations", "variables"],
        },
    )

def make_sympy_simplify_expression_tool() -> ToolSpec:
    """Simplifies a mathematical expression."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            expression = sympify(args["expression"])
            simplified_expr = simplify(expression)
            return {"result": str(simplified_expr)}
        except Exception as e:
            return {"error": f"SymPy simplify failed: {e}"}

    return ToolSpec(
        name="sympy_simplify_expression",
        description="Simplify a mathematical expression into its most readable and compact form.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to simplify, e.g., 'sin(x)**2 + cos(x)**2'.",
                }
            },
            "required": ["expression"],
        },
    )

def make_sympy_expand_expression_tool() -> ToolSpec:
    """Expands a mathematical expression."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            expression = sympify(args["expression"])
            expanded_expr = expand(expression)
            return {"result": str(expanded_expr)}
        except Exception as e:
            return {"error": f"SymPy expand failed: {e}"}

    return ToolSpec(
        name="sympy_expand_expression",
        description="Expand a mathematical expression by carrying out products and powers.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to expand, e.g., '(x + 1)**2'.",
                }
            },
            "required": ["expression"],
        },
    )

def make_sympy_factor_expression_tool() -> ToolSpec:
    """Factors a mathematical expression."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            expression = sympify(args["expression"])
            factored_expr = factor(expression)
            return {"result": str(factored_expr)}
        except Exception as e:
            return {"error": f"SymPy factor failed: {e}"}

    return ToolSpec(
        name="sympy_factor_expression",
        description="Factor a polynomial into irreducible factors.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to factor, e.g., 'x**2 + 2*x + 1'.",
                }
            },
            "required": ["expression"],
        },
    )

# --- Calculus Tools ---

def make_sympy_differentiate_tool() -> ToolSpec:
    """Differentiates an expression."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            expression = sympify(args["expression"])
            variable = sympify(args["variable"])
            derivative = diff(expression, variable)
            return {"derivative": str(derivative)}
        except Exception as e:
            return {"error": f"SymPy differentiate failed: {e}"}

    return ToolSpec(
        name="sympy_differentiate",
        description="Compute the derivative of an expression with respect to a variable.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to differentiate, e.g., 'sin(x)*exp(x)'.",
                },
                "variable": {
                    "type": "string",
                    "description": "The variable to differentiate with respect to, e.g., 'x'.",
                },
            },
            "required": ["expression", "variable"],
        },
    )

def make_sympy_integrate_tool() -> ToolSpec:
    """Integrates an expression."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            expression = sympify(args["expression"])
            variable = sympify(args["variable"])
            
            if "bounds" in args and args["bounds"]:
                lower_bound, upper_bound = args["bounds"]
                integral = integrate(expression, (variable, sympify(lower_bound), sympify(upper_bound)))
            else:
                integral = integrate(expression, variable)
            
            return {"integral": str(integral)}
        except Exception as e:
            return {"error": f"SymPy integrate failed: {e}"}

    return ToolSpec(
        name="sympy_integrate",
        description="Compute the indefinite or definite integral of an expression.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to integrate, e.g., 'cos(x)'.",
                },
                "variable": {
                    "type": "string",
                    "description": "The variable of integration, e.g., 'x'.",
                },
                "bounds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional. A list of two elements for definite integration [lower_bound, upper_bound].",
                },
            },
            "required": ["expression", "variable"],
        },
    )

# --- Linear Algebra Tool ---

def make_sympy_matrix_operation_tool() -> ToolSpec:
    """Performs various matrix operations."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            matrix_data = args["matrix"]
            operation = args["operation"]
            
            M = Matrix(matrix_data)
            
            if operation == "det":
                result = M.det()
            elif operation == "inv":
                result = M.inv()
            elif operation == "eigenvals":
                result = M.eigenvals()
            elif operation == "rref":
                result, pivots = M.rref()
                return {"rref_form": str(result), "pivots": str(pivots)}
            else:
                return {"error": f"Unknown matrix operation: {operation}"}
            
            return {"result": str(result)}
        except Exception as e:
            return {"error": f"SymPy matrix operation failed: {e}"}

    return ToolSpec(
        name="sympy_matrix_operation",
        description="Perform a linear algebra operation on a matrix.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "The matrix as a list of lists, e.g., [[1, 2], [3, 4]].",
                },
                "operation": {
                    "type": "string",
                    "enum": ["det", "inv", "eigenvals", "rref"],
                    "description": "The operation to perform: 'det' (determinant), 'inv' (inverse), 'eigenvals' (eigenvalues), 'rref' (reduced row echelon form).",
                },
            },
            "required": ["matrix", "operation"],
        },
    )

__all__ = [
    "make_calc_tool",
    "make_sympy_solve_equation_tool",
    "make_sympy_simplify_expression_tool",
    "make_sympy_expand_expression_tool",
    "make_sympy_factor_expression_tool",
    "make_sympy_differentiate_tool",
    "make_sympy_integrate_tool",
    "make_sympy_matrix_operation_tool",
]
