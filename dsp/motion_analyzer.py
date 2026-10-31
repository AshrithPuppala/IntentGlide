"""
Real-time kinematic and spectral feature extractor (Person 2).
Resamples jittery mouse/synthetic timestamps onto a uniform grid and extracts
4-12 Hz tremor power, sub-bin dominant frequency, spectral entropy, and kinematics.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from collections import deque
from typing import Deque, Optional
import numpy as np

from shared.contracts import InputSample, MotionFeatures


class MotionAnalyzer:
    def __init__(
        self,
        window_sec: float = 1.0,
        min_window_sec: float = 0.5,
        resample_n: int = 128,
        tremor_band: tuple[float, float] = (4.0, 12.0),
        analysis_band: tuple[float, float] = (2.0, 20.0),
        amp_gate_px: float = 1.5,
        speed_window_sec: float = 0.05,
    ):
        self.window_sec = window_sec
        self.min_window_sec = min_window_sec
        self.resample_n = resample_n
        self.tremor_lo, self.tremor_hi = tremor_band
        self.ana_lo, self.ana_hi = analysis_band
        self.amp_gate_px = amp_gate_px
        self.speed_window_sec = speed_window_sec

        self._hann = np.hanning(resample_n)
        self._t: Deque[float] = deque()
        self._x: Deque[float] = deque()
        self._y: Deque[float] = deque()
        self._last_source: Optional[str] = None
        self._last: MotionFeatures = self._default_features()

    def reset(self) -> None:
        """Clear sample buffer for a new trial or source switch."""
        self._t.clear()
        self._x.clear()
        self._y.clear()
        self._last_source = None
        self._last = self._default_features()

    def _default_features(self) -> MotionFeatures:
        """Neutral output during warm-up. All numeric fields initialized to finite values."""
        return MotionFeatures(
            oscillation_score=0.0,
            dominant_frequency=None,
            periodicity_confidence=0.0,
            directional_consistency=0.0,
            velocity=self._current_speed() if getattr(self, "_t", None) else 0.0,
            acceleration=0.0,
            jerk=0.0,
            spectral_entropy=0.0,
            est_amplitude=0.0,
            x_band_power=0.0,
            y_band_power=0.0,
            valid=False,
        )

    def _current_speed(self) -> float:
        """Instantaneous speed over a short recent window (~50 ms) for fast flick response."""
        if len(self._t) < 2:
            return 0.0
        t_now = self._t[-1]
        idx = len(self._t) - 2
        while idx > 0 and (t_now - self._t[idx]) < self.speed_window_sec:
            idx -= 1
        dt = t_now - self._t[idx]
        if dt <= 1e-6:
            return 0.0
        dx = self._x[-1] - self._x[idx]
        dy = self._y[-1] - self._y[idx]
        return float(math.hypot(dx, dy) / dt)

    def update(self, s: InputSample) -> MotionFeatures:
        # Guard against NaN / Inf inputs
        if not (math.isfinite(s.x) and math.isfinite(s.y) and math.isfinite(s.timestamp)):
            return self._last

        # Auto-reset if InputManager switches source (e.g. "mouse" <-> "synthetic")
        if self._last_source is not None and s.source != self._last_source:
            self.reset()

        # Ignore out-of-order or duplicate timestamps
        if self._t and s.timestamp <= self._t[-1]:
            return self._last

        self._last_source = s.source
        self._t.append(s.timestamp)
        self._x.append(s.x)
        self._y.append(s.y)

        # Evict samples older than window_sec
        cutoff_t = s.timestamp - self.window_sec
        while len(self._t) > 2 and self._t[0] < cutoff_t:
            self._t.popleft()
            self._x.popleft()
            self._y.popleft()

        span = self._t[-1] - self._t[0]
        if span < self.min_window_sec or len(self._t) < 16:
            self._last = self._default_features()
            return self._last

        self._last = self._extract_features(span)
        return self._last

    def _extract_features(self, span: float) -> MotionFeatures:
        N = self.resample_n
        t_raw = np.fromiter(self._t, dtype=np.float64, count=len(self._t))
        x_raw = np.fromiter(self._x, dtype=np.float64, count=len(self._x))
        y_raw = np.fromiter(self._y, dtype=np.float64, count=len(self._y))

        # 1. Uniform time-grid interpolation
        t_uni = np.linspace(t_raw[0], t_raw[-1], N)
        dt = (t_raw[-1] - t_raw[0]) / (N - 1)
        if dt <= 0:
            return self._default_features()

        x_uni = np.interp(t_uni, t_raw, x_raw)
        y_uni = np.interp(t_uni, t_raw, y_raw)

        # 2. Kinematics via finite differences
        vx = np.gradient(x_uni, dt)
        vy = np.gradient(y_uni, dt)
        speeds = np.hypot(vx, vy)

        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)
        accels = np.hypot(ax, ay)

        jx = np.gradient(ax, dt)
        jy = np.gradient(ay, dt)
        jerks = np.hypot(jx, jy)

        tail = max(4, N // 8)
        acceleration = float(np.mean(accels[-tail:]))
        jerk = float(np.mean(jerks[-tail:]))

        # Directional consistency over recent quarter-window
        q = max(8, N // 4)
        path_len = float(np.sum(speeds[-q:]) * dt)
        net_disp = float(math.hypot(x_uni[-1] - x_uni[-q], y_uni[-1] - y_uni[-q]))
        dir_consistency = float(np.clip(net_disp / (path_len + 1e-6), 0.0, 1.0)) if path_len > 0.5 else 0.0

        # 3. Polynomial detrending (degree 2)
        # Removes linear ramps and smooth minimum-jerk curves without attenuating 4-12 Hz tremor oscillations
        # 3. Polynomial detrending (degree 3)
        # Removes linear ramps and smooth minimum-jerk curves without attenuating 4-12 Hz tremor oscillations
        t_norm = np.linspace(-1.0, 1.0, N)
        poly_x = np.poly1d(np.polyfit(t_norm, x_uni, 3))
        poly_y = np.poly1d(np.polyfit(t_norm, y_uni, 3))
        x_det = (x_uni - poly_x(t_norm)) * self._hann
        y_det = (y_uni - poly_y(t_norm)) * self._hann

        # 4. Sinusoid-calibrated rFFT power spectrum
        # Coherent power scale factor for Hann window: (2.0 / sum(w))^2 = 16.0 / (N * N)
        freqs = np.fft.rfftfreq(N, d=dt)
        scale = 16.0 / (N * N)

        X = np.fft.rfft(x_det)
        Y = np.fft.rfft(y_det)
        px = (np.abs(X) ** 2) * scale
        py = (np.abs(Y) ** 2) * scale
        P = px + py

        band_mask = (freqs >= self.tremor_lo) & (freqs <= self.tremor_hi)
        ana_mask = (freqs >= self.ana_lo) & (freqs <= self.ana_hi)

        if band_mask.sum() < 2 or ana_mask.sum() < 2:
            return self._default_features()

        band_indices = np.where(band_mask)[0]
        pb = P[band_indices]
        peak_rel = int(np.argmax(pb))
        peak = int(band_indices[peak_rel])

        # Sum 3-bin main lobe around peak; for Hann window, the 3-bin sum contains 1.5x of peak A^2
        lobe_lo = max(band_indices[0], peak - 1)
        lobe_hi = min(band_indices[-1] + 1, peak + 2)

        x_band = float(np.sum(px[lobe_lo:lobe_hi]) / 1.5)
        y_band = float(np.sum(py[lobe_lo:lobe_hi]) / 1.5)
        band_p = x_band + y_band
        est_amp = float(math.sqrt(max(band_p, 0.0)))

        # Total power in the analysis band (2 - 20 Hz)
        total_ana_p = float(np.sum(P[ana_mask]) / 1.5) + 1e-9

        if est_amp < 1e-3 or band_p < 1e-6:
            return MotionFeatures(
                oscillation_score=0.0,
                dominant_frequency=None,
                periodicity_confidence=0.0,
                directional_consistency=dir_consistency,
                velocity=self._current_speed(),
                acceleration=acceleration,
                jerk=jerk,
                spectral_entropy=0.0,
                est_amplitude=0.0,
                x_band_power=0.0,
                y_band_power=0.0,
                valid=True,
            )

        # Parabolic log-power peak interpolation for sub-bin dominant frequency
        df = freqs[1] - freqs[0]
        if 0 < peak < len(P) - 1 and P[peak - 1] > 0 and P[peak] > 0 and P[peak + 1] > 0:
            a = math.log(P[peak - 1] + 1e-12)
            b = math.log(P[peak] + 1e-12)
            c = math.log(P[peak + 1] + 1e-12)
            denom = a - 2.0 * b + c
            delta = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
            delta = max(-0.5, min(0.5, delta))
            dominant = float(freqs[peak] + delta * df)
        else:
            dominant = float(freqs[peak])

        total_in_band = float(np.sum(pb) / 1.5) + 1e-9
        periodicity = float(np.clip(band_p / total_in_band, 0.0, 1.0))

        # Spectral entropy across 2-20 Hz analysis band
        pa = P[ana_mask]
        pa_sum = float(np.sum(pa))
        if pa_sum > 1e-6 and len(pa) > 1:
            p_norm = pa / pa_sum
            p_pos = p_norm[p_norm > 0]
            entropy = float(-(p_pos * np.log2(p_pos)).sum() / math.log2(len(pa)))
            entropy = float(np.clip(entropy, 0.0, 1.0))
        else:
            entropy = 0.0

        # Oscillation score: band power ratio gated by physical amplitude
        raw_ratio = float(np.clip(band_p / max(total_ana_p, band_p), 0.0, 1.0))
        amp_gate = float(np.clip((est_amp - 0.5) / max(self.amp_gate_px, 0.5), 0.0, 1.0))
        osc_score = float(np.clip(raw_ratio * amp_gate, 0.0, 1.0))

        return MotionFeatures(
            oscillation_score=osc_score,
            dominant_frequency=dominant,
            periodicity_confidence=periodicity,
            directional_consistency=dir_consistency,
            velocity=self._current_speed(),
            acceleration=acceleration,
            jerk=jerk,
            spectral_entropy=entropy,
            est_amplitude=est_amp,
            x_band_power=x_band,
            y_band_power=y_band,
            valid=True,
        )