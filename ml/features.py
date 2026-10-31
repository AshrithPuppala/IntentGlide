"""Prototype causal cursor-window features for IntentGlide's ML policy.

These features are an initial P3 implementation. Freeze/align their names and
calculations with Person 2 before integrating this module into the main app.
Only observed cursor coordinates are used as model inputs; ground-truth labels
must never be passed to this feature extractor.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "oscillation_score",
    "dominant_frequency",
    "periodicity_confidence",
    "velocity_mean",
    "velocity_std",
    "acceleration_rms",
    "jerk_rms",
    "directional_consistency",
    "x_band_power",
    "y_band_power",
    "band_power_ratio",
    "oscillation_amplitude",
    "spectral_entropy",
]


def _detrend(values: np.ndarray) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, len(values))
    coeff = np.polyfit(t, values, 1)
    return values - np.polyval(coeff, t)


def _spectrum(values: np.ndarray, sample_rate: float):
    residual = _detrend(values)
    windowed = residual * np.hanning(len(residual))
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sample_rate)
    band = (frequencies >= 4.0) & (frequencies <= 12.0)
    positive = frequencies > 0
    band_power = float(power[band].sum())
    total_power = float(power[positive].sum()) + 1e-12
    if not np.any(band) or band_power <= 1e-12:
        dominant = 0.0
        periodicity = 0.0
        amplitude = 0.0
    else:
        band_idx = np.flatnonzero(band)
        peak_idx = band_idx[int(np.argmax(power[band]))]
        dominant = float(frequencies[peak_idx])
        periodicity = float(power[peak_idx] / (band_power + 1e-12))
        band_spectrum = np.zeros_like(spectrum)
        band_spectrum[band] = spectrum[band]
        band_signal = np.fft.irfft(band_spectrum, n=len(values))
        amplitude = float(np.sqrt(2.0) * np.sqrt(np.mean(band_signal ** 2)))
    probs = power[positive] / total_power
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(max(2, len(probs))))
    return band_power, total_power, dominant, periodicity, amplitude, entropy


def extract_features(window_xy, sample_rate: float = 120.0) -> dict[str, float]:
    """Extract a stable initial feature dictionary from an observed XY window."""
    xy = np.asarray(window_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 32:
        raise ValueError("window_xy must have shape (n>=32, 2)")
    if not np.isfinite(xy).all():
        raise ValueError("window_xy contains NaN or infinite values")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    vel = np.diff(xy, axis=0) * sample_rate
    acc = np.diff(vel, axis=0) * sample_rate
    jerk = np.diff(acc, axis=0) * sample_rate
    speed = np.linalg.norm(vel, axis=1)
    nonzero = speed > 1e-9
    if np.any(nonzero):
        unit = vel[nonzero] / speed[nonzero, None]
        directional = float(np.linalg.norm(np.mean(unit, axis=0)))
    else:
        directional = 0.0

    xp, xt, xf, xperiod, xamp, xentropy = _spectrum(xy[:, 0], sample_rate)
    yp, yt, yf, yperiod, yamp, yentropy = _spectrum(xy[:, 1], sample_rate)
    combined = xp + yp
    total = xt + yt
    ratio = float(np.clip(combined / (total + 1e-12), 0.0, 1.0))
    dominant = xf if xp >= yp else yf
    periodicity = max(xperiod, yperiod)
    return {
        "oscillation_score": ratio,
        "dominant_frequency": float(dominant),
        "periodicity_confidence": float(np.clip(periodicity, 0.0, 1.0)),
        "velocity_mean": float(np.mean(speed)) if len(speed) else 0.0,
        "velocity_std": float(np.std(speed)) if len(speed) else 0.0,
        "acceleration_rms": float(np.sqrt(np.mean(acc ** 2))) if acc.size else 0.0,
        "jerk_rms": float(np.sqrt(np.mean(jerk ** 2))) if jerk.size else 0.0,
        "directional_consistency": float(np.clip(directional, 0.0, 1.0)),
        "x_band_power": float(xp),
        "y_band_power": float(yp),
        "band_power_ratio": ratio,
        "oscillation_amplitude": float(np.hypot(xamp, yamp)),
        "spectral_entropy": float((xentropy + yentropy) / 2.0),
    }
