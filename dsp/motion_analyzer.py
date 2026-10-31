"""Basic timestamp-aware motion feature estimation."""
from __future__ import annotations

from collections import deque
import numpy as np

from shared.contracts import InputSample, MotionFeatures


class MotionAnalyzer:
    def __init__(self, window_size: int = 64, analysis_rate: float = 120.0):
        if window_size < 8:
            raise ValueError("window_size must be at least 8")
        self.samples = deque(maxlen=window_size)
        self.analysis_rate = float(analysis_rate)

    def update(self, sample: InputSample) -> MotionFeatures:
        self.samples.append(sample)
        if len(self.samples) < 3:
            return MotionFeatures(0.0, None, 0.0, 1.0, 0.0)

        arr = list(self.samples)
        times = np.array([s.timestamp for s in arr], dtype=float)
        xs = np.array([s.x for s in arr], dtype=float)
        ys = np.array([s.y for s in arr], dtype=float)
        dt = np.diff(times)
        valid = dt > 0
        if not np.any(valid):
            return MotionFeatures(0.0, None, 0.0, 0.0, 0.0)

        dx = np.diff(xs)[valid]
        dy = np.diff(ys)[valid]
        dt_valid = dt[valid]
        velocity = np.hypot(dx, dy) / dt_valid
        speed = float(np.mean(velocity)) if len(velocity) else 0.0

        # Directional consistency is the magnitude of the mean unit direction.
        directions = np.column_stack((dx, dy))
        norms = np.linalg.norm(directions, axis=1)
        directions = directions[norms > 1e-9]
        if len(directions):
            directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
            directional = float(np.linalg.norm(np.mean(directions, axis=0)))
        else:
            directional = 0.0

        # Resample irregular event timestamps before an FFT estimate.
        dominant = None
        periodicity = 0.0
        oscillation = 0.0
        if times[-1] > times[0] and len(times) >= 8:
            uniform_t = np.arange(times[0], times[-1], 1.0 / self.analysis_rate)
            if len(uniform_t) >= 8:
                x_uniform = np.interp(uniform_t, times, xs)
                trend = np.linspace(x_uniform[0], x_uniform[-1], len(x_uniform))
                detrended = (x_uniform - trend) * np.hanning(len(x_uniform))
                spectrum = np.abs(np.fft.rfft(detrended)) ** 2
                freqs = np.fft.rfftfreq(len(detrended), d=1.0 / self.analysis_rate)
                band = (freqs >= 4.0) & (freqs <= 12.0)
                total = float(spectrum[1:].sum())
                if np.any(band) and total > 1e-12:
                    band_indices = np.where(band)[0]
                    peak_idx = band_indices[int(np.argmax(spectrum[band]))]
                    dominant = float(freqs[peak_idx])
                    periodicity = float(spectrum[peak_idx] / (spectrum[band].sum() + 1e-12))
                    oscillation = float(np.clip(spectrum[band].sum() / total * 2.0, 0.0, 1.0))

        return MotionFeatures(
            oscillation_score=oscillation,
            dominant_frequency=dominant,
            periodicity_confidence=float(np.clip(periodicity, 0.0, 1.0)),
            directional_consistency=float(np.clip(directional, 0.0, 1.0)),
            velocity=speed,
        )
