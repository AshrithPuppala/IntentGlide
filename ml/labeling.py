"""Offline optimal-alpha labels for a causal exponential smoother."""
from __future__ import annotations

import numpy as np


def _high_band_rms(signal: np.ndarray, sample_rate: float,
                   low_hz: float = 4.0, high_hz: float = 12.0) -> float:
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 16:
        return 0.0
    centered = signal - np.mean(signal)
    windowed = centered * np.hanning(len(centered))
    spec = np.fft.rfft(windowed)
    freq = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
    mask = (freq >= low_hz) & (freq <= high_hz)
    band = np.zeros_like(spec)
    band[mask] = spec[mask]
    component = np.fft.irfft(band, n=len(centered))
    return float(np.sqrt(np.mean(component ** 2)))


def _estimate_lag_pixels(filtered: np.ndarray, truth: np.ndarray,
                          sample_rate: float, max_lag: int = 8) -> float:
    direction = truth[-1] - truth[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return 0.0
    direction /= norm
    f = filtered @ direction
    g = truth @ direction
    speed = float(np.mean(np.abs(np.diff(g))) * sample_rate)
    best_lag = 0
    best_error = float("inf")
    max_lag = min(max_lag, len(f) // 4)
    for lag in range(max_lag + 1):
        if lag == 0:
            a, b = f, g
        else:
            # Positive lag tests whether the filtered output resembles an earlier truth position.
            a, b = f[lag:], g[:-lag]
        if len(a) < 8:
            continue
        err = float(np.mean((a - b) ** 2))
        if err < best_error:
            best_error, best_lag = err, lag
    return speed * best_lag / sample_rate


def score_alpha(observed_future: np.ndarray, truth_future: np.ndarray,
                initial_position: np.ndarray, alpha: float,
                sample_rate: float = 120.0) -> dict[str, float]:
    """Replay one candidate alpha over a common future horizon."""
    obs = np.asarray(observed_future, dtype=float)
    truth = np.asarray(truth_future, dtype=float)
    state = np.asarray(initial_position, dtype=float).copy()
    output = np.empty_like(obs)
    for i, point in enumerate(obs):
        state = state + alpha * (point - state)
        output[i] = state
    error = output - truth
    tracking = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
    jitter_x = _high_band_rms(error[:, 0], sample_rate)
    jitter_y = _high_band_rms(error[:, 1], sample_rate)
    jitter = float(np.hypot(jitter_x, jitter_y))
    lag_px = _estimate_lag_pixels(output, truth, sample_rate)
    # All terms are in pixels. This is an explicit initial engineering objective,
    # not a clinically validated or universally optimal weighting.
    cost = tracking + 0.50 * jitter + 0.50 * lag_px
    return {
        "tracking_rmse": tracking,
        "residual_jitter_rms": jitter,
        "lag_penalty_px": lag_px,
        "label_cost": cost,
    }


def select_optimal_alpha(observed_future: np.ndarray, truth_future: np.ndarray,
                         initial_position: np.ndarray,
                         candidates=None, sample_rate: float = 120.0):
    if candidates is None:
        candidates = np.linspace(0.12, 0.85, 16)
    results = []
    for alpha in candidates:
        metrics = score_alpha(observed_future, truth_future, initial_position,
                              float(alpha), sample_rate)
        results.append((float(metrics["label_cost"]), float(alpha), metrics))
    _, alpha, metrics = min(results, key=lambda x: x[0])
    return alpha, metrics
