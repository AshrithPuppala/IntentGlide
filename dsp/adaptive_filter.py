
import math
import time
from typing import Optional

from shared.contracts import InputSample, MotionFeatures, FilterOutput
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import time
from typing import Optional

from shared.contracts import InputSample, MotionFeatures, FilterOutput
_TWO_PI = 2.0 * math.pi


def alpha_to_fc(alpha: float, dt: float) -> float:
    """Equivalent first-order cutoff (Hz) of an EMA with this alpha at sample interval dt."""
    alpha = min(max(alpha, 1e-6), 1.0 - 1e-6)
    return alpha / ((1.0 - alpha) * _TWO_PI * dt)


def fc_to_alpha(fc: float, dt: float) -> float:
    """EMA alpha giving cutoff fc (Hz) at sample interval dt."""
    k = _TWO_PI * fc * dt
    return k / (1.0 + k)


class AdaptiveFilter:
    def __init__(
        self,
        alpha_min: float = 0.12,
        alpha_max: float = 0.85,
        nominal_fs: float = 120.0,
        heuristic_fc_max: float = 15.0,
        flick_speed: float = 800.0,
        smoothing_tau: float = 0.08,
        max_octaves_per_sec: float = 6.0,
    ):
        self.nominal_dt = 1.0 / nominal_fs
        # README alpha bounds, expressed once as rate-independent cutoff bounds.
        self.fc_lo = alpha_to_fc(alpha_min, self.nominal_dt)
        self.fc_hi = alpha_to_fc(alpha_max, self.nominal_dt)
        self.heuristic_fc_max = min(heuristic_fc_max, self.fc_hi)
        self.flick_speed = flick_speed
        self.tau = smoothing_tau                  # parameter smoothing time constant (s)
        self.max_oct_per_s = max_octaves_per_sec  # hard rate limit on cutoff changes

        self.last_source: str = "heuristic"
        self.last_processing_time_ms: float = 0.0
        self.reset()

    @property
    def cutoff_hz(self) -> Optional[float]:
        """Currently applied smoothing cutoff (Hz); the rate-independent view of alpha."""
        return self._fc

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        """New trial / source switch: forget position, timing and applied cutoff."""
        self._x = 0.0
        self._y = 0.0
        self._fc: Optional[float] = None
        self._last_t: Optional[float] = None
        self._last_out: Optional[FilterOutput] = None

    def update(
        self,
        s: InputSample,
        feats: Optional[MotionFeatures] = None,
        alpha_ml: Optional[float] = None,
    ) -> FilterOutput:
        """
        s         current raw sample
        feats     MotionFeatures from MotionAnalyzer (None is treated as warm-up)
        alpha_ml  Person 3's predicted alpha (defined at the 120 Hz nominal rate), or None
        """
        t0 = time.perf_counter()

        # Reject bad samples: keep the last output rather than corrupting state.
        if not (math.isfinite(s.x) and math.isfinite(s.y) and math.isfinite(s.timestamp)):
            return self._last_out or FilterOutput(0.0, 0.0, None, "adaptive")
        if self._last_t is not None and s.timestamp <= self._last_t:
            return self._last_out

        first = self._last_t is None
        dt = self.nominal_dt if first else min(max(s.timestamp - self._last_t, 1e-3), 0.1)
        self._last_t = s.timestamp

        # 1. Target cutoff (Hz), from ML if usable, else from the heuristic.
        if alpha_ml is not None and math.isfinite(alpha_ml):
            fc_target = alpha_to_fc(min(max(alpha_ml, 0.0), 1.0), self.nominal_dt)
            self.last_source = "ml"
        else:
            fc_target = self._heuristic_fc(feats)
            self.last_source = "heuristic" if alpha_ml is None else "heuristic_fallback"
        fc_target = min(max(fc_target, self.fc_lo), self.fc_hi)  # README alpha bounds

        # 2. Smooth + rate-limit the applied cutoff (log domain, time-based).
        if self._fc is None:
            self._fc = fc_target
        else:
            err = math.log2(fc_target / self._fc)
            step = err * (1.0 - math.exp(-dt / self.tau))
            limit = self.max_oct_per_s * dt
            step = max(-limit, min(step, limit))
            self._fc *= 2.0 ** step

        alpha = min(fc_to_alpha(self._fc, dt), 1.0)

        # 3. Exponential smoothing (stateful, per axis).
        if first:
            self._x, self._y = s.x, s.y
        else:
            self._x += alpha * (s.x - self._x)
            self._y += alpha * (s.y - self._y)

        self.last_processing_time_ms = (time.perf_counter() - t0) * 1000.0
        self._last_out = FilterOutput(
            x=self._x, y=self._y, alpha=alpha, mode="adaptive",
            processing_time_ms=self.last_processing_time_ms,
        )
        return self._last_out

    # ------------------------------------------------------------ heuristic
    def _heuristic_fc(self, feats: Optional[MotionFeatures]) -> float:
        lo, hi = self.fc_lo, self.heuristic_fc_max
        if feats is None or not feats.valid:
            return math.sqrt(lo * hi)            # neutral during warm-up

        # More oscillation -> lower cutoff (geometric interpolation: smoothing
        # strength is perceived/attenuated on a log-frequency scale).
        osc = min(max(feats.oscillation_score, 0.0), 1.0)
        fc = hi * (lo / hi) ** osc

        # Fast + straight movement is intentional: open the filter up (less lag).
        v_norm = min(max(feats.velocity / self.flick_speed, 0.0), 1.0)
        if feats.directional_consistency > 0.5:
            boost = (feats.directional_consistency - 0.5) * 2.0 * v_norm
            fc += boost * (hi - fc)
        return fc


if __name__ == "__main__":
    # Self-test (run from repo root):  python -m dsp.adaptive_filter
    from dsp.motion_analyzer import MotionAnalyzer

    an, filt = MotionAnalyzer(), AdaptiveFilter()
    fs = 120.0
    for i in range(int(2.0 * fs)):
        t = i / fs
        s = InputSample(t, 300 + 100 * t + 15 * math.sin(2 * math.pi * 8 * t), 200, "synthetic")
        f = an.update(s)
        out = filt.update(s, f)
    print(f"osc={f.oscillation_score:.2f}  alpha={out.alpha:.3f}  source={filt.last_source}  "
          f"{filt.last_processing_time_ms:.3f} ms")