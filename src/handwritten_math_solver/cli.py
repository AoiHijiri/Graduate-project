"""Command-line interface for the handwritten math solver demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from .math_engine import solve_expression
from .ocr import ExpressionRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recognize and solve a handwritten math expression demo.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Already-recognized expression, e.g. '2*x+3=7'.")
    group.add_argument("--ascii-file", type=Path, help="ASCII-art image made from #/1 foreground pixels.")
    group.add_argument("--pgm", type=Path, help="One-line handwritten expression in P2/P5 PGM format.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recognizer = ExpressionRecognizer()
    if args.text is not None:
        expression = args.text
    elif args.ascii_file is not None:
        expression = recognizer.recognize_ascii_art(args.ascii_file.read_text(encoding="utf-8"))
    else:
        expression = recognizer.recognize_pgm(args.pgm)

    solution = solve_expression(expression)
    print(f"recognized: {solution.recognized}")
    print(f"normalized: {solution.normalized}")
    print(f"answer: {solution.answer}")
    print(f"engine: {solution.engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
