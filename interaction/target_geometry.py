"""Screen-space geometry shared by targeting, dwell, and visual diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
import math

from shared.contracts import Target


@dataclass(frozen=True)
class Rect:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


def actual_rect(target: Target) -> Rect:
    return Rect(target.center_x-target.width/2, target.center_y-target.height/2,
                target.center_x+target.width/2, target.center_y+target.height/2)


def expanded_rect(target: Target, margin_px: float) -> Rect:
    if margin_px < 0 or not math.isfinite(margin_px):
        raise ValueError("margin_px must be a finite non-negative value")
    rect = actual_rect(target)
    return Rect(rect.left-margin_px, rect.top-margin_px,
                rect.right+margin_px, rect.bottom+margin_px)


def contains(rect: Rect, x: float, y: float) -> bool:
    return rect.left <= x <= rect.right and rect.top <= y <= rect.bottom


def point_rect_distance(x: float, y: float, rect: Rect) -> float:
    dx = max(rect.left-x, 0.0, x-rect.right)
    dy = max(rect.top-y, 0.0, y-rect.bottom)
    return math.hypot(dx, dy)


def distance_to_actual(target: Target, x: float, y: float) -> float:
    return point_rect_distance(x, y, actual_rect(target))


def inside_actual(target: Target, x: float, y: float) -> bool:
    return contains(actual_rect(target), x, y)


def inside_virtual(target: Target, x: float, y: float, margin_px: float) -> bool:
    return contains(expanded_rect(target, margin_px), x, y)


def regions_overlap(a: Target, a_margin: float, b: Target, b_margin: float) -> bool:
    ra, rb = expanded_rect(a, a_margin), expanded_rect(b, b_margin)
    return ra.left <= rb.right and rb.left <= ra.right and ra.top <= rb.bottom and rb.top <= ra.bottom


def isolated_small_icon_margin(target: Target, targets, requested_px: float = 16.0,
                               minimum_clearance_px: float = 22.0,
                               safety_gap_px: float = 6.0) -> float:
    """Return safe extra dwell acquisition margin for an isolated close icon.

    The candidate must be a compact X/close control. Nearby clickable targets
    reduce the enlargement, and a close neighbor disables it entirely.
    """
    if not target.id.endswith(":close") or target.width > 40 or target.height > 32:
        return 0.0
    others = [other for other in targets if other.id != target.id]
    if not others:
        return float(requested_px)
    a = actual_rect(target)
    clearances = []
    for other in others:
        b = actual_rect(other)
        dx = max(b.left - a.right, a.left - b.right, 0.0)
        dy = max(b.top - a.bottom, a.top - b.bottom, 0.0)
        clearances.append(math.hypot(dx, dy))
    clearance = min(clearances)
    if clearance < minimum_clearance_px:
        return 0.0
    safe_margin = max(0.0, clearance - safety_gap_px)
    return min(float(requested_px), safe_margin)
