"""Beginner-friendly handwritten math recognition and solving demo."""

from .math_engine import Solution, solve_expression
from .model import SyntheticGlyphKNN
from .ocr import ExpressionRecognizer

__all__ = ["ExpressionRecognizer", "Solution", "SyntheticGlyphKNN", "solve_expression"]
