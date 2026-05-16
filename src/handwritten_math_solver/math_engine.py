"""Turn recognized math text into an answer.

SymPy is used when installed so the app can solve equations, integrals, sigma
summations, and expressions involving constants such as pi and e.  A restricted
standard-library fallback handles basic arithmetic for machines without SymPy.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Solution:
    recognized: str
    normalized: str
    answer: str
    engine: str


def normalize_expression(expression: str) -> str:
    normalized = expression.replace(" ", "")
    replacements = {
        "×": "*",
        "÷": "/",
        "−": "-",
        "π": "pi",
        "Π": "pi",
        "Σ": "sigma",
        "∑": "sigma",
        "∫": "integral",
        "^": "**",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"(?<=\d)(?=[a-zA-Z(])", "*", normalized)
    normalized = re.sub(r"(?<=[a-zA-Z)])(?=\d)", "*", normalized)
    return normalized


def solve_expression(expression: str) -> Solution:
    normalized = normalize_expression(expression)
    try:
        answer = _solve_with_sympy(normalized)
        return Solution(expression, normalized, answer, "sympy")
    except ModuleNotFoundError:
        answer = _safe_eval(normalized)
        return Solution(expression, normalized, str(answer), "safe-eval")


def _solve_with_sympy(normalized: str) -> str:
    import sympy as sp

    symbols = {name: sp.symbols(name) for name in ["x", "y", "z", "k", "n"]}
    namespace = {
        **symbols,
        "pi": sp.pi,
        "e": sp.E,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "integral": lambda expr, var: sp.integrate(expr, var),
        "sigma": lambda expr, var, start, end: sp.summation(expr, (var, start, end)),
    }
    if "=" in normalized:
        left, right = normalized.split("=", 1)
        equation = sp.Eq(sp.sympify(left, locals=namespace), sp.sympify(right, locals=namespace))
        variables = sorted(equation.free_symbols, key=lambda symbol: symbol.name)
        if not variables:
            return str(bool(equation))
        return str(sp.solve(equation, variables[0]))
    return str(sp.simplify(sp.sympify(normalized, locals=namespace)))


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}


def _safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        return float(_ALLOWED_NAMES[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError("This expression needs SymPy. Install it with: pip install sympy")
