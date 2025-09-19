import unittest
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from simple_or_agent.tools.maths import (
    make_calc_tool,
    make_sympy_solve_equation_tool,
    make_sympy_simplify_expression_tool,
    make_sympy_expand_expression_tool,
    make_sympy_factor_expression_tool,
    make_sympy_differentiate_tool,
    make_sympy_integrate_tool,
    make_sympy_matrix_operation_tool
)


class TestMathsTools(unittest.TestCase):
    """Test suite for mathematics tools."""

    def test_calc_tool_basic_operations(self):
        """Test basic arithmetic operations."""
        tool = make_calc_tool()

        # Test addition
        result = tool.handler({"expression": "2 + 3"})
        self.assertEqual(result["value"], 5)
        self.assertEqual(result["expression"], "2 + 3")

        # Test subtraction
        result = tool.handler({"expression": "10 - 4"})
        self.assertEqual(result["value"], 6)

        # Test multiplication
        result = tool.handler({"expression": "3 * 7"})
        self.assertEqual(result["value"], 21)

        # Test division
        result = tool.handler({"expression": "15 / 3"})
        self.assertEqual(result["value"], 5)

        # Test power
        result = tool.handler({"expression": "2 ** 3"})
        self.assertEqual(result["value"], 8)

        # Test modulo
        result = tool.handler({"expression": "10 % 3"})
        self.assertEqual(result["value"], 1)

    def test_calc_tool_complex_expressions(self):
        """Test complex arithmetic expressions."""
        tool = make_calc_tool()

        # Test order of operations
        result = tool.handler({"expression": "2 + 3 * 4"})
        self.assertEqual(result["value"], 14)

        # Test parentheses
        result = tool.handler({"expression": "(2 + 3) * 4"})
        self.assertEqual(result["value"], 20)

        # Test unary operations
        result = tool.handler({"expression": "-5"})
        self.assertEqual(result["value"], -5)

        result = tool.handler({"expression": "+5"})
        self.assertEqual(result["value"], 5)

        # Test complex expression
        result = tool.handler({"expression": "2 * (3 + 4) - 10 / 2"})
        self.assertEqual(result["value"], 9)

    def test_calc_tool_error_handling(self):
        """Test error handling for calculator tool."""
        tool = make_calc_tool()

        # Test empty expression
        result = tool.handler({"expression": ""})
        self.assertIn("error", result)
        self.assertEqual(result["error"], "empty_expression")

        # Test invalid characters
        result = tool.handler({"expression": "2 + abc"})
        self.assertIn("error", result)

        # Test unsupported operations
        result = tool.handler({"expression": "2 & 3"})
        self.assertIn("error", result)

    def test_sympy_solve_equation_tool_simple(self):
        """Test solving simple equations."""
        tool = make_sympy_solve_equation_tool()

        # Test linear equation (using Eq format)
        args = {
            "equations": ["Eq(x + 2, 5)"],
            "variables": ["x"]
        }
        result = tool.handler(args)
        self.assertIn("solution", result)
        self.assertIn("3", result["solution"])

        # Test quadratic equation (expression = 0 format)
        args = {
            "equations": ["x**2 - 4"],
            "variables": ["x"]
        }
        result = tool.handler(args)
        self.assertIn("solution", result)
        # Should contain both -2 and 2 as solutions

    def test_sympy_solve_equation_tool_system(self):
        """Test solving systems of equations."""
        tool = make_sympy_solve_equation_tool()

        # Test system of 2 equations (using Eq format)
        args = {
            "equations": ["Eq(x + y, 10)", "Eq(x - y, 2)"],
            "variables": ["x", "y"]
        }
        result = tool.handler(args)
        self.assertIn("solution", result)
        # Should give x=6, y=4

    def test_sympy_solve_equation_tool_error_handling(self):
        """Test error handling for equation solving."""
        tool = make_sympy_solve_equation_tool()

        # Test invalid equation format
        args = {
            "equations": ["x + * 2"],
            "variables": ["x"]
        }
        result = tool.handler(args)
        self.assertIn("error", result)

    def test_sympy_simplify_expression_tool(self):
        """Test expression simplification."""
        tool = make_sympy_simplify_expression_tool()

        # Test trigonometric identity
        args = {"expression": "sin(x)**2 + cos(x)**2"}
        result = tool.handler(args)
        self.assertIn("result", result)
        # Should simplify to 1

        # Test algebraic simplification
        args = {"expression": "x**2 + 2*x + 1"}
        result = tool.handler(args)
        self.assertIn("result", result)

    def test_sympy_expand_expression_tool(self):
        """Test expression expansion."""
        tool = make_sympy_expand_expression_tool()

        # Test binomial expansion
        args = {"expression": "(x + 1)**2"}
        result = tool.handler(args)
        self.assertIn("result", result)
        # Should expand to x**2 + 2*x + 1

        # Test polynomial expansion
        args = {"expression": "(x + 1)*(x - 1)"}
        result = tool.handler(args)
        self.assertIn("result", result)
        # Should expand to x**2 - 1

    def test_sympy_factor_expression_tool(self):
        """Test expression factoring."""
        tool = make_sympy_factor_expression_tool()

        # Test polynomial factoring
        args = {"expression": "x**2 - 4"}
        result = tool.handler(args)
        self.assertIn("result", result)
        # Should factor to (x - 2)*(x + 2)

        # Test quadratic factoring
        args = {"expression": "x**2 + 2*x + 1"}
        result = tool.handler(args)
        self.assertIn("result", result)
        # Should factor to (x + 1)**2

    def test_sympy_differentiate_tool(self):
        """Test differentiation."""
        tool = make_sympy_differentiate_tool()

        # Test polynomial differentiation
        args = {
            "expression": "x**3 + 2*x**2 + x",
            "variable": "x"
        }
        result = tool.handler(args)
        self.assertIn("derivative", result)
        # Should give 3*x**2 + 4*x + 1

        # Test trigonometric differentiation
        args = {
            "expression": "sin(x)",
            "variable": "x"
        }
        result = tool.handler(args)
        self.assertIn("derivative", result)
        # Should give cos(x)

        # Test exponential differentiation
        args = {
            "expression": "exp(x)",
            "variable": "x"
        }
        result = tool.handler(args)
        self.assertIn("derivative", result)
        # Should give exp(x)

    def test_sympy_integrate_tool_indefinite(self):
        """Test indefinite integration."""
        tool = make_sympy_integrate_tool()

        # Test polynomial integration
        args = {
            "expression": "x**2",
            "variable": "x"
        }
        result = tool.handler(args)
        self.assertIn("integral", result)
        # Should give x**3/3

        # Test trigonometric integration
        args = {
            "expression": "sin(x)",
            "variable": "x"
        }
        result = tool.handler(args)
        self.assertIn("integral", result)
        # Should give -cos(x)

    def test_sympy_integrate_tool_definite(self):
        """Test definite integration."""
        tool = make_sympy_integrate_tool()

        # Test definite integral
        args = {
            "expression": "x**2",
            "variable": "x",
            "bounds": ["0", "1"]
        }
        result = tool.handler(args)
        self.assertIn("integral", result)
        # Should give 1/3

        # Test definite integral with bounds
        args = {
            "expression": "sin(x)",
            "variable": "x",
            "bounds": ["0", "pi"]
        }
        result = tool.handler(args)
        self.assertIn("integral", result)
        # Should give 2

    def test_sympy_matrix_operation_tool_determinant(self):
        """Test matrix determinant calculation."""
        tool = make_sympy_matrix_operation_tool()

        # Test 2x2 matrix
        args = {
            "matrix": [[1, 2], [3, 4]],
            "operation": "det"
        }
        result = tool.handler(args)
        self.assertIn("result", result)
        # Should give -2

        # Test 3x3 matrix
        args = {
            "matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "operation": "det"
        }
        result = tool.handler(args)
        self.assertIn("result", result)
        # Should give 0 (singular matrix)

    def test_sympy_matrix_operation_tool_inverse(self):
        """Test matrix inverse calculation."""
        tool = make_sympy_matrix_operation_tool()

        # Test 2x2 matrix inverse
        args = {
            "matrix": [[1, 2], [3, 4]],
            "operation": "inv"
        }
        result = tool.handler(args)
        self.assertIn("result", result)

        # Test singular matrix (should give error)
        args = {
            "matrix": [[1, 1], [1, 1]],
            "operation": "inv"
        }
        result = tool.handler(args)
        self.assertIn("error", result)

    def test_sympy_matrix_operation_tool_eigenvalues(self):
        """Test matrix eigenvalue calculation."""
        tool = make_sympy_matrix_operation_tool()

        # Test 2x2 matrix eigenvalues
        args = {
            "matrix": [[2, 1], [1, 2]],
            "operation": "eigenvals"
        }
        result = tool.handler(args)
        self.assertIn("result", result)

    def test_sympy_matrix_operation_tool_rref(self):
        """Test matrix reduced row echelon form."""
        tool = make_sympy_matrix_operation_tool()

        # Test 2x2 matrix RREF
        args = {
            "matrix": [[1, 2], [3, 4]],
            "operation": "rref"
        }
        result = tool.handler(args)
        self.assertIn("rref_form", result)
        self.assertIn("pivots", result)

    def test_sympy_matrix_operation_tool_error_handling(self):
        """Test error handling for matrix operations."""
        tool = make_sympy_matrix_operation_tool()

        # Test invalid operation
        args = {
            "matrix": [[1, 2], [3, 4]],
            "operation": "invalid_operation"
        }
        result = tool.handler(args)
        self.assertIn("error", result)
        self.assertIn("Unknown matrix operation", result["error"])

        # Test invalid matrix dimensions
        args = {
            "matrix": [[1, 2], [3]],
            "operation": "det"
        }
        result = tool.handler(args)
        self.assertIn("error", result)

    def test_math_tools_error_handling_general(self):
        """Test general error handling across all math tools."""
        # Test missing required parameters
        tools_with_required_params = [
            (make_sympy_solve_equation_tool, ["equations", "variables"]),
            (make_sympy_simplify_expression_tool, ["expression"]),
            (make_sympy_expand_expression_tool, ["expression"]),
            (make_sympy_factor_expression_tool, ["expression"]),
            (make_sympy_differentiate_tool, ["expression", "variable"]),
            (make_sympy_integrate_tool, ["expression", "variable"]),
            (make_sympy_matrix_operation_tool, ["matrix", "operation"]),
        ]

        for tool_func, required_params in tools_with_required_params:
            with self.subTest(tool=tool_func.__name__):
                tool = tool_func()
                # Test with empty args
                result = tool.handler({})
                self.assertIn("error", result)

    def test_calc_tool_precision(self):
        """Test precision handling in calculator."""
        tool = make_calc_tool()

        # Test floating point division
        result = tool.handler({"expression": "1 / 3"})
        self.assertIsInstance(result["value"], float)

        # Test large numbers
        result = tool.handler({"expression": "2 ** 10"})
        self.assertEqual(result["value"], 1024)

        # Test negative results
        result = tool.handler({"expression": "5 - 10"})
        self.assertEqual(result["value"], -5)

    def test_sympy_expression_tools_edge_cases(self):
        """Test edge cases for SymPy expression tools."""
        # Test empty expression
        tool = make_sympy_simplify_expression_tool()
        result = tool.handler({"expression": ""})
        self.assertIn("error", result)

        # Test complex expressions
        tool = make_sympy_expand_expression_tool()
        args = {"expression": "(x + y + z)**2"}
        result = tool.handler(args)
        self.assertIn("result", result)

        # Test already simplified expression
        tool = make_sympy_simplify_expression_tool()
        args = {"expression": "x"}
        result = tool.handler(args)
        self.assertIn("result", result)
        self.assertIn("x", result["result"])


if __name__ == "__main__":
    # Run the tests
    unittest.main()