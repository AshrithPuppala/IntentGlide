"""Rank candidate targets by proximity and movement-direction alignment."""
from __future__ import annotations

import math
from typing import Iterable

from shared.contracts import Target


def rank_targets(
    x: float, y: float, dx: float, dy: float, targets: Iterable[Target]
) -> list[tuple[Target, float]]:
    speed = math.hypot(dx, dy)
    ranked = []
    for target in targets:
        vx = target.center_x - x
        vy = target.center_y - y
        distance = math.hypot(vx, vy)
        scale = max(1.0, math.hypot(target.width, target.height))
        proximity = math.exp(-distance / (scale * 2.5))
        if speed > 1e-6 and distance > 1e-6:
            alignment = max(0.0, (dx * vx + dy * vy) / (speed * distance))
        else:
            alignment = 0.5
        ranked.append((target, float(0.7 * proximity + 0.3 * alignment)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)
