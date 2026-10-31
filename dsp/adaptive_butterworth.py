"""ML-controlled, time-smoothed second-order Butterworth low-pass filter.

The filter accepts a target cutoff in Hz from the ML policy. If no valid target
is supplied it falls back to a conservative feature-based policy. Coefficients
are updated smoothly while direct-form-I input/output history is retained, so
cutoff changes do not reset the filter state.
"""
from __future__ import annotations

import math
import time
from typing import Optional

from shared.contracts import FilterOutput, InputSample, MotionFeatures


class _BiquadAxis:
    """Second-order Butterworth low-pass using direct form I history."""

    def __init__(self, cutoff_hz: float, sample_rate: float):
        self.sample_rate = float(sample_rate)
        self.initialized = False
        self.x1 = self.x2 = 0.0
        self.y1 = self.y2 = 0.0
        self.set_cutoff(cutoff_hz)

    def set_cutoff(self, cutoff_hz: float) -> None:
        fc = min(max(float(cutoff_hz), 0.05), 0.45 * self.sample_rate)
        k = math.tan(math.pi * fc / self.sample_rate)
        k2 = k * k
        root2 = math.sqrt(2.0)
        norm = 1.0 / (1.0 + root2 * k + k2)
        self.b0 = k2 * norm
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0
        self.a1 = 2.0 * (k2 - 1.0) * norm
        self.a2 = (1.0 - root2 * k + k2) * norm

    def reset(self) -> None:
        self.initialized = False
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def step(self, value: float) -> float:
        x = float(value)
        if not self.initialized:
            # Constant-position steady-state initialisation: no artificial jump.
            self.x1 = self.x2 = x
            self.y1 = self.y2 = x
            self.initialized = True
            return x
        y = (
            self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1 - self.a2 * self.y2
        )
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return float(y)


class AdaptiveButterworthFilter:
    """Adaptive 2nd-order Butterworth filter controlled by cutoff frequency.

    Args:
        min_cutoff_hz: lower permitted cutoff; lower means stronger smoothing.
        max_cutoff_hz: upper permitted cutoff; higher means more responsiveness.
        nominal_fs: expected sample rate used for coefficient design bounds.
        warmup_cutoff_hz: neutral cutoff until MotionAnalyzer has a valid window.
        smoothing_tau: exponential time constant for cutoff changes (seconds).
        max_octaves_per_sec: hard log-frequency slew limit.
    """

    def __init__(
        self,
        min_cutoff_hz: float = 1.5,
        max_cutoff_hz: float = 18.0,
        nominal_fs: float = 120.0,
        warmup_cutoff_hz: float = 8.0,
        smoothing_tau: float = 0.10,
        max_octaves_per_sec: float = 5.0,
    ):
        if nominal_fs <= 0:
            raise ValueError("nominal_fs must be positive")
        if not (0.05 < min_cutoff_hz < max_cutoff_hz < nominal_fs * 0.45):
            raise ValueError("Require 0.05 < min_cutoff_hz < max_cutoff_hz < 0.45*nominal_fs")
        if not (min_cutoff_hz <= warmup_cutoff_hz <= max_cutoff_hz):
            raise ValueError("warmup_cutoff_hz must be inside the cutoff bounds")
        if smoothing_tau <= 0 or max_octaves_per_sec <= 0:
            raise ValueError("smoothing_tau and max_octaves_per_sec must be positive")

        self.min_cutoff_hz = float(min_cutoff_hz)
        self.max_cutoff_hz = float(max_cutoff_hz)
        self.nominal_fs = float(nominal_fs)
        self.warmup_cutoff_hz = float(warmup_cutoff_hz)
        self.tau = float(smoothing_tau)
        self.max_octaves_per_sec = float(max_octaves_per_sec)
        self.fx = _BiquadAxis(self.warmup_cutoff_hz, self.nominal_fs)
        self.fy = _BiquadAxis(self.warmup_cutoff_hz, self.nominal_fs)
        self.last_source = "warmup"
        self.last_processing_time_ms = 0.0
        self._last_t: Optional[float] = None
        self._last_input_source: Optional[str] = None
        self._cutoff_hz: Optional[float] = None
        self._last_out: Optional[FilterOutput] = None

    @property
    def cutoff_hz(self) -> Optional[float]:
        return self._cutoff_hz

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()
        self._last_t = None
        self._last_input_source = None
        self._cutoff_hz = None
        self._last_out = None
        self.last_source = "warmup"
        self.last_processing_time_ms = 0.0

    def _heuristic_cutoff(self, features: Optional[MotionFeatures]) -> float:
        if features is None or not getattr(features, "valid", True):
            return self.warmup_cutoff_hz
        score = min(max(float(features.oscillation_score), 0.0), 1.0)
        # Monotonic conservative fallback: stronger detected oscillation -> lower cutoff.
        return self.max_cutoff_hz * (self.min_cutoff_hz / self.max_cutoff_hz) ** score

    def update(
        self,
        sample: InputSample,
        features: Optional[MotionFeatures] = None,
        cutoff_target_hz: Optional[float] = None,
    ) -> FilterOutput:
        t0 = time.perf_counter()
        if not all(math.isfinite(v) for v in (sample.x, sample.y, sample.timestamp)):
            return self._last_out or FilterOutput(0.0, 0.0, None, "adaptive")

        if self._last_input_source is not None and sample.source != self._last_input_source:
            self.reset()
        if self._last_t is not None and sample.timestamp <= self._last_t:
            return self._last_out or FilterOutput(sample.x, sample.y, None, "adaptive")

        first = self._last_t is None
        dt = (1.0 / self.nominal_fs) if first else min(max(sample.timestamp - self._last_t, 1e-4), 0.1)
        self._last_t = float(sample.timestamp)
        self._last_input_source = sample.source

        valid_ml_target = cutoff_target_hz is not None and math.isfinite(float(cutoff_target_hz))
        if valid_ml_target:
            target = float(cutoff_target_hz)
            self.last_source = "ml"
        else:
            target = self._heuristic_cutoff(features)
            if cutoff_target_hz is not None:
                self.last_source = "heuristic_fallback"
            elif features is None or not getattr(features, "valid", True):
                self.last_source = "warmup"
            else:
                self.last_source = "heuristic"
        target = min(max(target, self.min_cutoff_hz), self.max_cutoff_hz)

        if self._cutoff_hz is None:
            applied_cutoff = target
        else:
            error_octaves = math.log2(target / self._cutoff_hz)
            smooth_step = error_octaves * (1.0 - math.exp(-dt / self.tau))
            max_step = self.max_octaves_per_sec * dt
            smooth_step = max(-max_step, min(max_step, smooth_step))
            applied_cutoff = self._cutoff_hz * (2.0 ** smooth_step)
        self._cutoff_hz = min(max(applied_cutoff, self.min_cutoff_hz), self.max_cutoff_hz)

        # Updating coefficients preserves the input/output histories in direct form I.
        self.fx.set_cutoff(self._cutoff_hz)
        self.fy.set_cutoff(self._cutoff_hz)
        out_x = self.fx.step(sample.x)
        out_y = self.fy.step(sample.y)

        self.last_processing_time_ms = (time.perf_counter() - t0) * 1000.0
        self._last_out = FilterOutput(
            x=out_x,
            y=out_y,
            alpha=None,
            mode="adaptive",
            processing_time_ms=self.last_processing_time_ms,
        )
        return self._last_out
