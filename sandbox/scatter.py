"""Random, deliberately unaligned positions for extra desktop icons.

Pure Python (no pygame) so it can be unit-tested. Positions come from a seeded RNG, so a
given seed always gives the same layout (handy for repeatable demos and benchmarks).
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

Rect = tuple[float, float, float, float]          # x, y, w, h


def _overlap(a: Rect, b: Rect) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def scatter_icon_positions(count: int, area: Rect, icon_size: tuple[float, float],
                           avoid: Sequence[Rect] = (), seed: Optional[int] = 7,
                           min_gap: float = 40.0, min_misalign: float = 40.0,
                           tries: int = 2000) -> list[tuple[float, float]]:
    """Return ``count`` top-left positions inside ``area`` for icons of ``icon_size``.

    No icon overlaps an ``avoid`` rect or another icon (``min_gap`` apart), and no two icons
    line up: their x values differ by at least ``min_misalign``, and so do their y values.
    """
    ax, ay, aw, ah = area
    w, h = icon_size
    rng = random.Random(seed)
    placed: list[tuple[float, float]] = []
    for _ in range(tries):
        if len(placed) == count:
            break
        x = rng.uniform(ax, ax + aw - w)
        y = rng.uniform(ay, ay + ah - h)
        box = (x - min_gap / 2, y - min_gap / 2, w + min_gap, h + min_gap)
        if any(_overlap(box, r) for r in avoid):
            continue
        if any(_overlap(box, (px, py, w, h)) or abs(x - px) < min_misalign or abs(y - py) < min_misalign
               for px, py in placed):
            continue
        placed.append((x, y))
    if len(placed) < count:
        raise ValueError(f"could only place {len(placed)} of {count} icons; enlarge the area or loosen the gaps")
    return placed

