"""A dependency-free toy machine-learning classifier for math glyphs.

It uses k-nearest neighbours over synthetic examples generated from 5x7 glyph
prototypes.  The goal is educational: the code is intentionally small enough for
new ML students to read, modify, and run without a GPU or large dataset.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import random
from typing import Iterable

from .symbols import GLYPHS

Bitmap = list[list[int]]
Vector = list[float]


def template_to_bitmap(template: list[str]) -> Bitmap:
    return [[1 if ch == "1" else 0 for ch in row] for row in template]


def flatten(bitmap: Bitmap) -> Vector:
    return [float(pixel) for row in bitmap for pixel in row]


def resize_bitmap(bitmap: Bitmap, height: int = 7, width: int = 5) -> Bitmap:
    """Resize a binary bitmap using nearest-neighbour sampling."""
    if not bitmap or not bitmap[0]:
        return [[0] * width for _ in range(height)]
    src_h, src_w = len(bitmap), len(bitmap[0])
    resized: Bitmap = []
    for y in range(height):
        src_y = min(src_h - 1, round(y * (src_h - 1) / max(1, height - 1)))
        row = []
        for x in range(width):
            src_x = min(src_w - 1, round(x * (src_w - 1) / max(1, width - 1)))
            row.append(1 if bitmap[src_y][src_x] else 0)
        resized.append(row)
    return resized


def augment_bitmap(bitmap: Bitmap, rng: random.Random, noise: float = 0.04) -> Bitmap:
    """Create a slightly noisy/shifted synthetic handwritten sample."""
    height, width = len(bitmap), len(bitmap[0])
    dy = rng.choice([-1, 0, 0, 1])
    dx = rng.choice([-1, 0, 0, 1])
    result: Bitmap = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            src_y, src_x = y - dy, x - dx
            value = bitmap[src_y][src_x] if 0 <= src_y < height and 0 <= src_x < width else 0
            if rng.random() < noise:
                value = 1 - value
            result[y][x] = value
    return result


@dataclass
class SyntheticGlyphKNN:
    """Small k-nearest-neighbour classifier for one-character math symbols."""

    k: int = 3
    samples: list[tuple[str, Vector]] = field(default_factory=list)

    def train(self, samples_per_class: int = 40, seed: int = 7) -> "SyntheticGlyphKNN":
        rng = random.Random(seed)
        self.samples.clear()
        for label, template in GLYPHS.items():
            base = template_to_bitmap(template)
            self.samples.append((label, flatten(base)))
            for _ in range(samples_per_class):
                self.samples.append((label, flatten(augment_bitmap(base, rng))))
        return self

    def predict(self, bitmap: Bitmap) -> str:
        if not self.samples:
            raise RuntimeError("The classifier has not been trained yet. Call train() first.")
        vector = flatten(resize_bitmap(bitmap))
        distances = sorted(
            ((self._distance(vector, sample), label) for label, sample in self.samples),
            key=lambda item: item[0],
        )
        if distances[0][0] == 0:
            return distances[0][1]
        neighbours = distances[: self.k]
        return Counter(label for _, label in neighbours).most_common(1)[0][0]

    def score(self, labelled_bitmaps: Iterable[tuple[str, Bitmap]]) -> float:
        total = correct = 0
        for expected, bitmap in labelled_bitmaps:
            total += 1
            correct += int(self.predict(bitmap) == expected)
        return correct / total if total else math.nan

    @staticmethod
    def _distance(left: Vector, right: Vector) -> float:
        return sum((a - b) ** 2 for a, b in zip(left, right))
