"""Adaptive exponential smoother with bounded, rate-limited alpha."""
from __future__ import annotations

from shared.contracts import FilterOutput, InputSample, MotionFeatures


class AdaptiveExponentialFilter:
    def __init__(
        self,
        min_alpha: float = 0.12,
        max_alpha: float = 0.85,
        initial_alpha: float = 0.65,
        max_alpha_step: float = 0.05,
    ):
        if not 0 < min_alpha <= max_alpha <= 1:
            raise ValueError("Require 0 < min_alpha <= max_alpha <= 1")
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.alpha = min(max(initial_alpha, min_alpha), max_alpha)
        self.max_alpha_step = max_alpha_step
        self._position = None

    def reset(self):
        self._position = None
        self.alpha = min(max(self.alpha, self.min_alpha), self.max_alpha)

    def update(
        self,
        sample: InputSample,
        features: MotionFeatures,
        target_context: bool = False,
    ) -> FilterOutput:
        desired = self.max_alpha - (
            features.oscillation_score * (self.max_alpha - self.min_alpha)
        )
        if target_context and features.directional_consistency > 0.7:
            desired = min(self.max_alpha, desired + 0.08)

        delta = max(-self.max_alpha_step,
                    min(self.max_alpha_step, desired - self.alpha))
        self.alpha = min(self.max_alpha,
                         max(self.min_alpha, self.alpha + delta))

        if self._position is None:
            self._position = (sample.x, sample.y)
        else:
            px, py = self._position
            self._position = (
                px + self.alpha * (sample.x - px),
                py + self.alpha * (sample.y - py),
            )

        return FilterOutput(
            x=float(self._position[0]),
            y=float(self._position[1]),
            alpha=float(self.alpha),
            mode="adaptive",
        )
