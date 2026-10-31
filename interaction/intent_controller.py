"""Confidence-gated controller with explicit ambiguity abstention.

Two scoring modes:
  "posterior" (default)  Estimates the cursor's heading from the FILTERED path over a short
      window and asks, for each target, "how plausible is it that the user is heading here?".
      Plausibility = how close the heading ray passes to the target relative to how precisely
      people can aim at that distance. Plausibilities are normalised into probabilities
      together with a "none of them" hypothesis, so being near one key is not enough: the
      user must be heading to it AND no neighbour may be equally plausible.
  "legacy"  The original proximity/alignment score (kept for comparison).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Optional

from interaction.target_analyzer import rank_targets
from shared.contracts import FilterOutput, IntentDecision, MotionFeatures, Target


class IntentController:
    def __init__(
        self,
        confidence_threshold: float = 0.60,
        ambiguity_margin: float = 0.12,
        max_assist_px: float = 5.0,
        scoring: str = "posterior",
        velocity_window_s: float = 0.25,
        min_move_px: float = 10.0,
        aim_k: float = 0.12,
        null_weight: float = 0.05,
        distance_prior_px: float = 400.0,
        gate_ratio: Optional[float] = None,
        hover_k: float = 1.2,
    ):
        if scoring not in ("posterior", "legacy"):
            raise ValueError("scoring must be 'posterior' or 'legacy'")
        self.confidence_threshold = confidence_threshold
        self.ambiguity_margin = ambiguity_margin
        self.max_assist_px = max_assist_px
        self.scoring = scoring
        self.velocity_window_s = velocity_window_s
        self.min_move_px = min_move_px
        self.aim_k = aim_k
        self.null_weight = null_weight
        self.distance_prior_px = distance_prior_px
        # Optional deceleration gate (off by default: it did not improve precision in testing).
        # If set, abstain while moving faster than gate_ratio x the movement's peak speed.
        self.gate_ratio = gate_ratio
        self.hover_k = hover_k
        self._peak_speed = 0.0
        self._still_since: Optional[float] = None
        self._history: deque = deque()          # (timestamp, x, y) of the filtered path
        self.last_heading: Optional[tuple[float, float]] = None   # unit vector, None = not moving

    def reset(self):
        self._history.clear()
        self.last_heading = None
        self._peak_speed = 0.0
        self._still_since = None

    # ------------------------------------------------------------------ heading
    def _update_heading(self, timestamp: float, x: float, y: float) -> tuple[float, float, float, float]:
        """Return (dx, dy, displacement, speed) of the filtered path over the recent window."""
        self._history.append((timestamp, x, y))
        while self._history and timestamp - self._history[0][0] > self.velocity_window_s:
            self._history.popleft()
        t0, x0, y0 = self._history[0]
        if timestamp - t0 < 0.5 * self.velocity_window_s:       # not enough history yet
            return 0.0, 0.0, 0.0, 0.0
        dx, dy = x - x0, y - y0
        disp = math.hypot(dx, dy)
        speed = disp / (timestamp - t0)
        if disp >= self.min_move_px:
            self._still_since = None
            self._peak_speed = max(self._peak_speed, speed)
        else:
            if self._still_since is None:
                self._still_since = timestamp
            elif timestamp - self._still_since > 0.25:       # a new movement will start fresh
                self._peak_speed = 0.0
        return dx, dy, disp, speed

    # ------------------------------------------------------------------ scoring
    def _posterior(self, x, y, dx, dy, disp, targets) -> tuple[list[tuple[Target, float]], float]:
        """Return ([(target, probability)] best first, total plausibility excluding 'none')."""
        moving = disp >= self.min_move_px
        ux, uy = (dx / disp, dy / disp) if moving else (0.0, 0.0)
        self.last_heading = (ux, uy) if moving else None
        weights = []
        for t in targets:
            vx, vy = t.center_x - x, t.center_y - y
            d = math.hypot(vx, vy)
            half = 0.5 * min(t.width, t.height)
            prior = 1.0 / (1.0 + d / self.distance_prior_px)
            if d <= half:                                    # cursor is over the target
                lik = 1.0
            elif moving:
                cos = (ux * vx + uy * vy) / d
                if cos <= 0.0:
                    lik = 0.0                                # heading away from it
                else:
                    miss = d * math.sqrt(max(0.0, 1.0 - cos * cos))   # ray's miss distance
                    sigma = half + self.aim_k * d
                    lik = math.exp(-0.5 * (miss / sigma) ** 2)
            else:                                            # hovering: nearest target wins
                lik = math.exp(-0.5 * (d / (self.hover_k * half)) ** 2)
            weights.append(prior * lik)
        total = sum(weights)
        norm = total + self.null_weight
        ranked = sorted(((t, w / norm) for t, w in zip(targets, weights)), key=lambda i: i[1], reverse=True)
        return ranked, total

    # ------------------------------------------------------------------ decide
    def decide(
        self,
        raw_x: float,
        raw_y: float,
        filtered: FilterOutput,
        features: MotionFeatures,
        targets: Iterable[Target],
        override: bool = False,
        timestamp: Optional[float] = None,
    ) -> IntentDecision:
        if override:
            return IntentDecision("override", None, 0.0, 0.0, 0.0)

        target_list = list(targets)
        idle_state = "oscillation_detected" if features.oscillation_score > 0.5 else "stable"
        if not target_list:
            return IntentDecision(idle_state, None, 0.0, 0.0, 0.0)

        if self.scoring == "legacy" or timestamp is None:
            ranked = rank_targets(filtered.x, filtered.y, raw_x - filtered.x, raw_y - filtered.y, target_list)
            best_target, best_score = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            abstain = best_score < self.confidence_threshold or (
                len(ranked) > 1 and best_score - second < self.ambiguity_margin)
            plausible = True
        else:
            dx, dy, disp, speed = self._update_heading(timestamp, filtered.x, filtered.y)
            ranked, total = self._posterior(filtered.x, filtered.y, dx, dy, disp, target_list)
            best_target, best_score = ranked[0]
            still_fast = (self.gate_ratio is not None and disp >= self.min_move_px
                          and speed > self.gate_ratio * self._peak_speed)
            abstain = best_score < self.confidence_threshold or still_fast
            plausible = total > 0.05                    # is ANY target a credible destination?

        confidence = float(max(0.0, min(1.0, best_score)))
        if abstain:
            if plausible:
                return IntentDecision("target_ambiguous", None, confidence, 0.0, 0.0)
            return IntentDecision(idle_state, None, confidence, 0.0, 0.0)

        dx_t = best_target.center_x - filtered.x
        dy_t = best_target.center_y - filtered.y
        norm = max(1e-9, math.hypot(dx_t, dy_t))
        magnitude = min(self.max_assist_px, norm * 0.04)
        return IntentDecision("target_confident", best_target.id, confidence,
                              float(dx_t / norm * magnitude), float(dy_t / norm * magnitude))
