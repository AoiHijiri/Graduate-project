"""Image and ASCII-art recognition helpers for the toy math OCR model."""

from __future__ import annotations

from pathlib import Path

from .model import Bitmap, SyntheticGlyphKNN, resize_bitmap
from .symbols import ALIASES


def read_ascii_art(text: str) -> Bitmap:
    """Convert text made of '#', '1', or 'X' foreground pixels into a bitmap."""
    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    width = max(len(line) for line in lines)
    return [[1 if ch in "#1X@" else 0 for ch in line.ljust(width)] for line in lines]


def read_pgm(path: str | Path) -> Bitmap:
    """Read a plain PGM (P2) or binary PGM (P5) image and binarize it.

    PGM keeps this project dependency-free.  You can convert PNG/JPG files to
    PGM with common tools such as ImageMagick (`magick input.png output.pgm`).
    """
    data = Path(path).read_bytes()
    tokens: list[bytes] = []
    i = 0
    while len(tokens) < 4 and i < len(data):
        if data[i : i + 1] == b"#":
            while i < len(data) and data[i : i + 1] not in b"\r\n":
                i += 1
        elif data[i : i + 1].isspace():
            i += 1
        else:
            start = i
            while i < len(data) and not data[i : i + 1].isspace():
                i += 1
            tokens.append(data[start:i])
    magic, width_b, height_b, max_b = tokens
    width, height, max_value = int(width_b), int(height_b), int(max_b)
    while i < len(data) and data[i : i + 1].isspace():
        i += 1
    if magic == b"P2":
        pixels = [int(token) for token in data[i:].split()]
    elif magic == b"P5":
        pixels = list(data[i : i + width * height])
    else:
        raise ValueError("Only P2/P5 PGM images are supported.")
    threshold = max_value * 0.55
    return [
        [1 if pixels[y * width + x] < threshold else 0 for x in range(width)]
        for y in range(height)
    ]


def segment_characters(bitmap: Bitmap) -> list[Bitmap]:
    """Split a single-line expression into character boxes by blank columns."""
    if not bitmap:
        return []
    height, width = len(bitmap), len(bitmap[0])
    has_ink = [any(bitmap[y][x] for y in range(height)) for x in range(width)]
    boxes: list[tuple[int, int]] = []
    start: int | None = None
    for x, ink in enumerate(has_ink + [False]):
        if ink and start is None:
            start = x
        elif not ink and start is not None:
            boxes.append((start, x))
            start = None

    chars: list[Bitmap] = []
    for left, right in boxes:
        rows = [row[left:right] for row in bitmap]
        non_empty_rows = [i for i, row in enumerate(rows) if any(row)]
        if not non_empty_rows:
            continue
        top, bottom = non_empty_rows[0], non_empty_rows[-1] + 1
        chars.append(resize_bitmap(rows[top:bottom]))
    return chars


class ExpressionRecognizer:
    """Recognize a one-line math expression from segmented glyphs."""

    def __init__(self, classifier: SyntheticGlyphKNN | None = None) -> None:
        self.classifier = classifier or SyntheticGlyphKNN().train()

    def recognize_bitmap(self, bitmap: Bitmap) -> str:
        labels = [self.classifier.predict(char) for char in segment_characters(bitmap)]
        return "".join(ALIASES.get(label, label) for label in labels)

    def recognize_ascii_art(self, text: str) -> str:
        return self.recognize_bitmap(read_ascii_art(text))

    def recognize_pgm(self, path: str | Path) -> str:
        return self.recognize_bitmap(read_pgm(path))
