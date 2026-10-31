
import math
from collections import deque
from typing import Optional, Tuple

import numpy as np
from scipy.signal import detrend

from shared.contracts import InputSample, MotionFeatures

_EPS = 1e-9
_EDGE_TRIM = 3  # samples dropped at each end of finite-difference estimates


class MotionAnalyzer:
    def __init__(
        self,
        window_sec: float = 1.0,
        grid_points: int = 120,
        min_span_sec: float = 0.5,
        tremor_band: Tuple[float, float] = (4.0, 12.0),
        analysis_band: Tuple[float, float] = (2.0, 20.0),
        amp_gate_px: float = 2.0,
        min_freq_amp_px: float = 0.1,
        velocity_horizon_sec: float = 0.05,
    ):
        self.window_sec = window_sec
        self.grid_points = grid_points
        self.min_span_sec = min_span_sec
        self.band = tremor_band
        self.analysis_band = analysis_band
        self.amp_gate_px = amp_gate_px
        self.min_freq_amp_px = min_freq_amp_px
        self.velocity_horizon_sec = velocity_horizon_sec

        self._t: deque = deque()
        self._x: deque = deque()
        self._y: deque = deque()
        self._last: MotionFeatures = self._default_features()

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        """Clear all state (new trial, or input source switched)."""
        self._t.clear()
        self._x.clear()
        self._y.clear()
        self._last = self._default_features()

    def update(self, s: InputSample) -> MotionFeatures:
        t, x, y = float(s.timestamp), float(s.x), float(s.y)

        # Ignore bad samples: NaN/inf, duplicate or backwards timestamps.
        if not (math.isfinite(t) and math.isfinite(x) and math.isfinite(y)):
            return self._last
        if self._t and t <= self._t[-1]:
            return self._last

        self._t.append(t)
        self._x.append(x)
        self._y.append(y)

        # Keep one sample at/before the window start so we can interpolate to it.
        cutoff = t - self.window_sec
        while len(self._t) >= 2 and self._t[1] <= cutoff:
            self._t.popleft()
            self._x.popleft()
            self._y.popleft()

        span = self._t[-1] - self._t[0]
        if len(self._t) < 4 or span < self.min_span_sec:
            self._last = self._default_features()
        else:
            self._last = self._extract_features()
        return self._last

    def analyze_window(self, xy, sample_rate: float = 120.0) -> MotionFeatures:
        """Analyze a uniformly sampled XY window using the live feature calculation.

        This batch entry point gives offline training the same terminal-window
        calculation as ``update`` without recomputing an FFT for every sample.
        """
        points = np.asarray(xy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
            raise ValueError("xy must have shape (n>=4, 2)")
        if not np.isfinite(points).all():
            raise ValueError("xy contains NaN or infinite values")
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("sample_rate must be positive and finite")

        self.reset()
        for index, (x, y) in enumerate(points):
            timestamp = index / sample_rate
            self._t.append(timestamp)
            self._x.append(float(x))
            self._y.append(float(y))

        cutoff = self._t[-1] - self.window_sec
        while len(self._t) >= 2 and self._t[1] <= cutoff:
            self._t.popleft()
            self._x.popleft()
            self._y.popleft()

        span = self._t[-1] - self._t[0]
        self._last = self._extract_features() if span >= self.min_span_sec else self._default_features()
        return self._last

    # ------------------------------------------------------------ internals
    def _current_speed(self) -> float:
        """Speed over the last ~velocity_horizon_sec of raw samples (px/s)."""
        n = len(self._t)
        if n < 2:
            return 0.0
        t = np.fromiter(self._t, float, n)
        i = int(np.searchsorted(t, t[-1] - self.velocity_horizon_sec, side="right")) - 1
        i = min(max(i, 0), n - 2)
        dt = t[-1] - t[i]
        if dt <= 0:
            return 0.0
        return float(math.hypot(self._x[-1] - self._x[i], self._y[-1] - self._y[i]) / dt)

    def _extract_features(self) -> MotionFeatures:
        n_raw = len(self._t)
        t_raw = np.fromiter(self._t, float, n_raw)
        x_raw = np.fromiter(self._x, float, n_raw)
        y_raw = np.fromiter(self._y, float, n_raw)

        # 1. Resample onto a uniform grid over the (up to) last window_sec.
        N = self.grid_points
        t_end = t_raw[-1]
        t_start = max(t_raw[0], t_end - self.window_sec)
        dt = (t_end - t_start) / (N - 1)
        if dt <= 0:
            return self._default_features()
        t_uni = np.linspace(t_start, t_end, N)
        x_uni = np.interp(t_uni, t_raw, x_raw)
        y_uni = np.interp(t_uni, t_raw, y_raw)

        # 2. Kinematics from finite differences on the uniform grid.
        vx, vy = np.gradient(x_uni, dt), np.gradient(y_uni, dt)
        ax, ay = np.gradient(vx, dt), np.gradient(vy, dt)
        jx, jy = np.gradient(ax, dt), np.gradient(ay, dt)
        v_mag = np.hypot(vx, vy)
        sl = slice(_EDGE_TRIM, -_EDGE_TRIM)
        acceleration = float(np.mean(np.hypot(ax, ay)[sl]))
        jerk = float(np.mean(np.hypot(jx, jy)[sl]))

        # Speed-weighted directional consistency: |sum v| / sum |v|  (0..1).
        # Straight movement -> ~1; back-and-forth oscillation -> ~0; still -> 0.
        dir_consistency = float(
            np.clip(math.hypot(vx.sum(), vy.sum()) / (v_mag.sum() + _EPS), 0.0, 1.0)
        )

        # 3. Spectrum: detrend (removes drift/ramp), Hann window, rFFT.
        win = np.hanning(N)
        X = np.fft.rfft(detrend(x_uni, type="linear") * win)
        Y = np.fft.rfft(detrend(y_uni, type="linear") * win)
        freqs = np.fft.rfftfreq(N, d=dt)
        nyq = 0.5 / dt

        # |FFT|^2 -> sinusoid-equivalent px^2 (Hann: sum(w^2)=3N/8, one-sided).
        scale = 32.0 / (3.0 * N * N)
        px = (np.abs(X) ** 2) * scale
        py = (np.abs(Y) ** 2) * scale
        P = px + py

        lo, hi = self.band
        a_lo, a_hi = self.analysis_band
        band_mask = (freqs >= lo) & (freqs <= min(hi, nyq))
        ana_mask = (freqs >= a_lo) & (freqs <= min(a_hi, nyq))
        if band_mask.sum() < 2 or ana_mask.sum() < 2:
            return self._default_features()

        x_band = float(px[band_mask].sum())
        y_band = float(py[band_mask].sum())
        band_p = x_band + y_band
        ana_p = float(P[ana_mask].sum())
        est_amp = math.sqrt(band_p)

        # 4. Oscillation score = in-band share of analysis-band power * amplitude gate.
        ratio = band_p / ana_p if ana_p > 1e-6 else 0.0
        gate = min(max(est_amp / self.amp_gate_px, 0.0), 1.0)
        osc_score = float(np.clip(ratio * gate, 0.0, 1.0))

        # 5. Dominant frequency (parabolic peak interpolation) + periodicity.
        dominant: Optional[float] = None
        periodicity = 0.0
        if est_amp >= self.min_freq_amp_px and band_p > _EPS:
            idx = np.flatnonzero(band_mask)
            peak = int(idx[np.argmax(P[idx])])
            df = freqs[1] - freqs[0]
            delta = 0.0
            if 0 < peak < len(P) - 1:
                a, b, c = (math.log(P[k] + _EPS) for k in (peak - 1, peak, peak + 1))
                denom = a - 2.0 * b + c
                if denom < 0:
                    delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
            dominant = float(freqs[peak] + delta * df)
            # Hann spreads a sine over ~3 bins, so count peak +/- 1 bin.
            periodicity = float(
                np.clip(P[max(peak - 1, 0): peak + 2].sum() / band_p, 0.0, 1.0)
            )

        # 6. Spectral entropy over the analysis band (0 = one tone, 1 = flat).
        pa = P[ana_mask]
        entropy = 0.0
        if pa.sum() > 1e-6 and len(pa) > 1:
            p = pa / pa.sum()
            p = p[p > 0]
            entropy = float(-(p * np.log2(p)).sum() / math.log2(len(pa)))
            entropy = float(np.clip(entropy, 0.0, 1.0))

        return MotionFeatures(
    oscillation_score=float(np.clip(osc_score, 0.0, 1.0)),
    dominant_frequency=dominant,
    periodicity_confidence=float(np.clip(periodicity, 0.0, 1.0)),
    directional_consistency=float(np.clip(dir_consistency, 0.0, 1.0)),
    velocity=self._current_speed(),
    acceleration=acceleration,
    jerk=jerk,
    spectral_entropy=float(np.clip(entropy, 0.0, 1.0)),
    est_amplitude=float(est_amp),
    x_band_power=x_band,
    y_band_power=y_band,
    valid=True,
)

    def _default_features(self) -> MotionFeatures:
        """Neutral output during warm-up. valid=False; dir. consistency 0 (not 'smooth')."""
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


if __name__ == "__main__":
    # Self-test (run from repo root):  python -m dsp.motion_analyzer
    import time

    fs, amp, f_t = 120.0, 15.0, 8.0
    analyzer = MotionAnalyzer()
    feats = None
    t0 = time.perf_counter()
    n = int(1.5 * fs)
    for i in range(n):
        t = i / fs
        feats = analyzer.update(
            InputSample(t, 300 + 100 * t + amp * math.sin(2 * math.pi * f_t * t), 200, "synthetic")
        )
    per_update_ms = (time.perf_counter() - t0) / n * 1000
    print(f"{per_update_ms:.3f} ms/update")
    for k, v in feats.__dict__.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
