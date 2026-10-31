"""
Fixed 2nd-order Butterworth low-pass filter (Person 2).
Serves as the constant-cutoff baseline benchmark against AdaptiveFilter.
Fully compatible with Person 1's InputSample streams (MouseProvider, SyntheticInput, SyntheticReach).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import time
from typing import Optional
from shared.contracts import InputSample, FilterOutput


class _BiquadLowPass:
    """Single-axis 2nd-order Butterworth low-pass filter using Direct Form II Transposed."""

    def __init__(self, cutoff_hz: float, fs: float):
        self.cutoff_hz = cutoff_hz
        self.fs = fs
        self.initialized = False
        self.z1 = 0.0
        self.z2 = 0.0
        self._compute_coefficients(cutoff_hz, fs)

    def _compute_coefficients(self, fc: float, fs: float) -> None:
        nyq = 0.5 * fs
        fc_clamped = min(max(fc, 0.1), nyq * 0.95)
        k = math.tan(math.pi * fc_clamped / fs)
        k2 = k * k
        sqrt2 = math.sqrt(2.0)
        norm = 1.0 / (1.0 + sqrt2 * k + k2)

        self.b0 = k2 * norm
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0
        self.a1 = 2.0 * (k2 - 1.0) * norm
        self.a2 = (1.0 - sqrt2 * k + k2) * norm

    def reset(self) -> None:
        self.initialized = False
        self.z1 = 0.0
        self.z2 = 0.0

    def step(self, x: float) -> float:
        if not self.initialized:
            # Initialize internal state to steady-state for step input x
            self.z1 = x * (1.0 - self.b0)
            self.z2 = x * (self.b2 - self.a2)
            self.initialized = True
            return float(x)

        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return float(y)


class FixedFilter:
    def __init__(self, cutoff_hz: float = 3.0, nominal_fs: float = 120.0):
        self.cutoff_hz = cutoff_hz
        self.nominal_fs = nominal_fs
        self.fx = _BiquadLowPass(cutoff_hz, nominal_fs)
        self.fy = _BiquadLowPass(cutoff_hz, nominal_fs)
        self.last_processing_time_ms: float = 0.0
        self._last_t: Optional[float] = None
        self._last_source: Optional[str] = None
        self._last_out: Optional[FilterOutput] = None

    def reset(self) -> None:
        """Clear filter state for a new trial or input source switch."""
        self.fx.reset()
        self.fy.reset()
        self._last_t = None
        self._last_source = None
        self._last_out = None

    def update(self, s: InputSample) -> FilterOutput:
        t0 = time.perf_counter()

        if not (math.isfinite(s.x) and math.isfinite(s.y) and math.isfinite(s.timestamp)):
            return self._last_out or FilterOutput(x=0.0, y=0.0, alpha=None, mode="fixed")

        # Auto-reset if InputManager switches source (e.g. "mouse" <-> "synthetic")
        if self._last_source is not None and s.source != self._last_source:
            self.reset()

        if self._last_t is not None and s.timestamp <= self._last_t:
            return self._last_out

        self._last_t = s.timestamp
        self._last_source = s.source

        out_x = self.fx.step(s.x)
        out_y = self.fy.step(s.y)

        self.last_processing_time_ms = (time.perf_counter() - t0) * 1000.0
        self._last_out = FilterOutput(
            x=out_x,
            y=out_y,
            alpha=None,
            mode="fixed",
            processing_time_ms=self.last_processing_time_ms,
        )
        return self._last_out


if __name__ == "__main__":
    from input.synthetic_generator import SyntheticInputGenerator

    gen = SyntheticInputGenerator(dt=1.0 / 120.0, tremor_freq=8.0, tremor_amp=15.0)
    filt = FixedFilter(cutoff_hz=3.0, nominal_fs=120.0)
    samples = gen.generate(240)
    outs = [filt.update(s) for s in samples]
    print(f"FixedFilter verified with Person 1 SyntheticInputGenerator: "
          f"last=({outs[-1].x:.2f}, {outs[-1].y:.2f}), {filt.last_processing_time_ms:.4f} ms")