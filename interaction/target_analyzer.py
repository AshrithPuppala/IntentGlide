"""Causal, history-aware target ranking for assisted pointing.

The direction estimate is a velocity vector fitted to recent filtered cursor
positions. It deliberately does not use (raw_position - filtered_position),
which is a filtering residual rather than cursor travel direction.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from shared.contracts import Target
from interaction.target_geometry import (distance_to_actual, expanded_rect,
                                         point_rect_distance, actual_rect, contains)

TARGET_FEATURE_NAMES = (
    "distance_over_target_diagonal",
    "velocity_x_norm",
    "velocity_y_norm",
    "speed_norm",
    "direction_cosine",
    "toward_speed_norm",
    "history_alignment",
    "approach_rate_norm",
    "deceleration_norm",
    "inside_target",
    "distance_to_actual_rect_norm",
    "distance_to_virtual_rect_norm",
    "inside_virtual_target",
    "target_width_norm",
    "target_height_norm",
    "profile_variability_norm",
    "profile_overshoot_norm",
    "profile_speed_norm",
    "profile_acquisition_margin_norm",
)


@dataclass(frozen=True)
class MovementEstimate:
    velocity_x: float
    velocity_y: float
    speed: float
    previous_speed: float
    history_alignment: float
    approach_rate: float


class MovementHistory:
    """Short, timestamped history of filtered cursor positions."""

    def __init__(self, max_seconds: float = 0.65, max_points: int = 96):
        self.max_seconds = float(max_seconds)
        self._points: deque[tuple[float, float, float]] = deque(maxlen=max_points)

    def reset(self) -> None:
        self._points.clear()

    def add(self, timestamp: float, x: float, y: float) -> None:
        t, px, py = float(timestamp), float(x), float(y)
        if not all(math.isfinite(v) for v in (t, px, py)):
            return
        if self._points and t <= self._points[-1][0]:
            # A backwards/duplicate clock must not poison the velocity estimate.
            return
        self._points.append((t, px, py))
        cutoff = t - self.max_seconds
        while len(self._points) > 2 and self._points[1][0] < cutoff:
            self._points.popleft()

    @property
    def points(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(self._points)

    @property
    def duration(self) -> float:
        if len(self._points) < 2:
            return 0.0
        return max(0.0, self._points[-1][0] - self._points[0][0])

    def _slope(self, points: Sequence[tuple[float, float, float]]) -> tuple[float, float]:
        if len(points) < 2:
            return 0.0, 0.0
        arr = np.asarray(points, dtype=float)
        t = arr[:, 0] - arr[-1, 0]
        denom = float(np.sum((t - t.mean()) ** 2))
        if denom <= 1e-12:
            return 0.0, 0.0
        vx = float(np.sum((t - t.mean()) * (arr[:, 1] - arr[:, 1].mean())) / denom)
        vy = float(np.sum((t - t.mean()) * (arr[:, 2] - arr[:, 2].mean())) / denom)
        return vx, vy

    def estimate(self, target: Target | None = None) -> MovementEstimate:
        pts = list(self._points)
        if len(pts) < 3:
            return MovementEstimate(0.0, 0.0, 0.0, 0.0, 0.5, 0.0)
        end_t = pts[-1][0]
        recent = [p for p in pts if p[0] >= end_t - 0.25]
        vx, vy = self._slope(recent if len(recent) >= 3 else pts)
        speed = math.hypot(vx, vy)
        earlier = [p for p in pts if end_t - 0.55 <= p[0] < end_t - 0.25]
        if len(earlier) >= 3:
            pvx, pvy = self._slope(earlier)
            previous_speed = math.hypot(pvx, pvy)
        else:
            previous_speed = speed

        if target is None:
            return MovementEstimate(vx, vy, speed, previous_speed, 0.5, 0.0)

        cx, cy = pts[-1][1], pts[-1][2]
        tx, ty = target.center_x - cx, target.center_y - cy
        distance = math.hypot(tx, ty)
        direction_cosine = 0.0
        if speed > 1.0 and distance > 1e-6:
            direction_cosine = (vx * tx + vy * ty) / (speed * distance)

        # Average alignment of recent actual movement segments toward this target.
        alignments: list[float] = []
        recent_points = pts[-8:]
        for a, b in zip(recent_points[:-1], recent_points[1:]):
            dx, dy = b[1] - a[1], b[2] - a[2]
            step = math.hypot(dx, dy)
            rx, ry = target.center_x - a[1], target.center_y - a[2]
            rnorm = math.hypot(rx, ry)
            if step >= 0.5 and rnorm >= 1.0:
                cosine = (dx * rx + dy * ry) / (step * rnorm)
                alignments.append(float(np.clip((cosine + 1.0) * 0.5, 0.0, 1.0)))
        history_alignment = float(np.mean(alignments)) if alignments else 0.5
        approach_rate = speed * direction_cosine if speed > 1.0 else 0.0
        return MovementEstimate(vx, vy, speed, previous_speed,
                                history_alignment, approach_rate)


def target_feature_dict(
    history: Sequence[tuple[float, float, float]],
    x: float,
    y: float,
    target: Target,
    acquisition_margin_px: float = 30.0,
    profile_features: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build target-specific, causal features from position history."""
    hist = MovementHistory()
    for item in history:
        hist.add(*item)
    estimate = hist.estimate(target)
    dx, dy = target.center_x - float(x), target.center_y - float(y)
    distance = math.hypot(dx, dy)
    diagonal = max(1.0, math.hypot(target.width, target.height))
    direction_cosine = 0.0
    if estimate.speed > 1.0 and distance > 1e-6:
        direction_cosine = (estimate.velocity_x * dx + estimate.velocity_y * dy) / (estimate.speed * distance)
    deceleration = 0.0
    if estimate.previous_speed > 20.0:
        deceleration = float(np.clip(
            (estimate.previous_speed - estimate.speed) / estimate.previous_speed, 0.0, 1.0
        ))
    inside = (abs(dx) <= target.width / 2.0 and abs(dy) <= target.height / 2.0)
    virtual = expanded_rect(target, max(0.0, acquisition_margin_px))
    profile = profile_features or {}
    return {
        "distance_over_target_diagonal": float(distance / diagonal),
        "velocity_x_norm": float(np.clip(estimate.velocity_x / 1200.0, -1.5, 1.5)),
        "velocity_y_norm": float(np.clip(estimate.velocity_y / 1200.0, -1.5, 1.5)),
        "speed_norm": float(np.clip(estimate.speed / 1200.0, 0.0, 1.5)),
        "direction_cosine": float(np.clip(direction_cosine, -1.0, 1.0)),
        "toward_speed_norm": float(np.clip(estimate.approach_rate / 1200.0, -1.0, 1.0)),
        "history_alignment": float(np.clip(estimate.history_alignment, 0.0, 1.0)),
        "approach_rate_norm": float(np.clip(estimate.approach_rate / 1200.0, -1.0, 1.0)),
        "deceleration_norm": deceleration,
        "inside_target": float(inside),
        "distance_to_actual_rect_norm": float(distance_to_actual(target, x, y) / diagonal),
        "distance_to_virtual_rect_norm": float(point_rect_distance(x, y, virtual) / diagonal),
        "inside_virtual_target": float(contains(virtual, x, y)),
        "target_width_norm": float(np.clip(target.width / 500.0, 0.0, 2.0)),
        "target_height_norm": float(np.clip(target.height / 500.0, 0.0, 2.0)),
        "profile_variability_norm": float(np.clip(profile.get("motion_variability_rms", 0.0) / 50.0, 0.0, 2.0)),
        "profile_overshoot_norm": float(np.clip(profile.get("median_overshoot_px", 0.0) / 100.0, 0.0, 2.0)),
        "profile_speed_norm": float(np.clip(profile.get("median_movement_speed", 0.0) / 2000.0, 0.0, 2.0)),
        "profile_acquisition_margin_norm": float(np.clip(acquisition_margin_px / 50.0, 0.0, 1.0)),
    }


def heuristic_target_score(features: dict[str, float]) -> float:
    """Interpretable fallback score; it is a ranking score, not a probability."""
    centre_proximity = math.exp(-max(0.0, features["distance_over_target_diagonal"]) / 3.0)
    virtual_proximity = math.exp(-max(0.0, features["distance_to_virtual_rect_norm"]) / 1.5)
    proximity = max(centre_proximity, 0.75 * virtual_proximity)
    speed = features["speed_norm"]
    alignment = max(0.0, features["direction_cosine"]) if speed > 0.025 else 0.35
    history = features["history_alignment"] if speed > 0.025 else 0.5
    approach = max(0.0, features["approach_rate_norm"])
    deceleration = features["deceleration_norm"]
    inside = features["inside_target"]
    virtual_inside = features["inside_virtual_target"]
    score = (0.18 * proximity + 0.20 * alignment + 0.29 * history
             + 0.14 * approach + 0.05 * deceleration + 0.10 * inside
             + 0.04 * virtual_inside)
    return float(np.clip(score, 0.0, 1.0))


def rank_targets(
    x: float,
    y: float,
    dx: float,
    dy: float,
    targets: Iterable[Target],
    history: Sequence[tuple[float, float, float]] | None = None,
    policy=None,
    acquisition_margin_px: float = 30.0,
    profile_features: dict[str, float] | None = None,
) -> list[tuple[Target, float]]:
    """Rank targets using movement history and a velocity vector.

    ``dx``/``dy`` are now interpreted as estimated velocity components (px/s),
    not the raw-minus-filtered position residual used by the earlier prototype.
    ``policy`` may be an ML TargetIntentPolicy; without it, an explainable score
    is used. Existing call sites remain compatible.
    """
    points = list(history or [])
    if len(points) < 2:
        # Compatibility path for callers without history: seed a tiny synthetic
        # two-point history so direction can still be assessed.
        points = [(0.0, x - dx * 0.01, y - dy * 0.01), (0.01, x, y)]
    ranked = []
    for target in targets:
        features = target_feature_dict(points, x, y, target, acquisition_margin_px,
                                       profile_features)
        score = policy.score_features(features) if policy is not None else heuristic_target_score(features)
        ranked.append((target, float(np.clip(score, 0.0, 1.0))))
    return sorted(ranked, key=lambda item: item[1], reverse=True)
