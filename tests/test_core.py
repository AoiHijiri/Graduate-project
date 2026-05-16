import unittest

from handwritten_math_solver.math_engine import normalize_expression, solve_expression
from handwritten_math_solver.model import SyntheticGlyphKNN, template_to_bitmap
from handwritten_math_solver.symbols import GLYPHS


class CoreTests(unittest.TestCase):
    def test_template_classifier_recognizes_digits_and_symbols(self):
        model = SyntheticGlyphKNN().train(samples_per_class=2, seed=1)
        for label in ["2", "+", "x", "S", "I"]:
            with self.subTest(label=label):
                self.assertEqual(model.predict(template_to_bitmap(GLYPHS[label])), label)

    def test_normalize_expression(self):
        self.assertEqual(normalize_expression("2x + π^2"), "2*x+pi**2")

    def test_basic_solver_fallback_path(self):
        solution = solve_expression("2+3*4")
        self.assertEqual(solution.answer, "14.0" if solution.engine == "safe-eval" else "14")


if __name__ == "__main__":
    unittest.main()
