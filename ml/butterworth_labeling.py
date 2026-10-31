"""Offline optimal-cutoff labels for a second-order Butterworth low-pass."""
from __future__ import annotations

import math
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def _high_band_rms(signal: np.ndarray, sample_rate: float,
                   low_hz: float = 4.0, high_hz: float = 12.0) -> float:
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 16:
        return 0.0
    centered = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered * np.hanning(len(centered)))
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    band = np.zeros_like(spectrum)
    band[mask] = spectrum[mask]
    component = np.fft.irfft(band, n=len(centered))
    return float(np.sqrt(np.mean(component ** 2)))


def _estimate_lag_pixels(filtered: np.ndarray, truth: np.ndarray,
                          sample_rate: float, max_lag: int = 8) -> float:
    direction = np.asarray(truth[-1] - truth[0], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return 0.0
    direction /= norm
    f = filtered @ direction
    g = truth @ direction
    speed = float(np.mean(np.abs(np.diff(g))) * sample_rate)
    best_lag, best_error = 0, float("inf")
    for lag in range(min(max_lag, len(f) // 4) + 1):
        a, b = (f, g) if lag == 0 else (f[lag:], g[:-lag])
        if len(a) < 8:
            continue
        error = float(np.mean((a - b) ** 2))
        if error < best_error:
            best_error, best_lag = error, lag
    return speed * best_lag / sample_rate


def filter_future_butterworth(history: np.ndarray, observed_future: np.ndarray,
                              cutoff_hz: float, sample_rate: float) -> np.ndarray:
    """Filter future X/Y samples after warming SOS state on past observed samples."""
    history = np.asarray(history, dtype=float)
    observed = np.asarray(observed_future, dtype=float)
    if history.ndim != 2 or history.shape[1] != 2 or len(history) < 2:
        raise ValueError("history must have shape (n>=2, 2)")
    if observed.ndim != 2 or observed.shape[1] != 2 or len(observed) < 1:
        raise ValueError("observed_future must have shape (n>=1, 2)")
    if not np.isfinite(history).all() or not np.isfinite(observed).all():
        raise ValueError("Filter input contains NaN or infinity")
    if not (0.05 < cutoff_hz < 0.45 * sample_rate):
        raise ValueError("cutoff_hz must be between 0.05 Hz and 0.45*sample_rate")

    sos = butter(2, float(cutoff_hz), btype="lowpass", fs=float(sample_rate), output="sos")
    output = np.empty_like(observed)
    for axis in range(2):
        zi = sosfilt_zi(sos) * float(history[0, axis])
        _, zi_final = sosfilt(sos, history[:, axis], zi=zi)
        output[:, axis], _ = sosfilt(sos, observed[:, axis], zi=zi_final)
    return output


def score_cutoff(observed_future: np.ndarray, truth_future: np.ndarray,
                 history: np.ndarray, cutoff_hz: float,
                 sample_rate: float = 120.0,
                 jitter_weight: float = 0.5,
                 lag_weight: float = 0.5) -> dict[str, float]:
    observed = np.asarray(observed_future, dtype=float)
    truth = np.asarray(truth_future, dtype=float)
    filtered = filter_future_butterworth(history, observed, cutoff_hz, sample_rate)
    error = filtered - truth
    tracking = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
    jitter = float(np.hypot(_high_band_rms(error[:, 0], sample_rate),
                            _high_band_rms(error[:, 1], sample_rate)))
    lag = _estimate_lag_pixels(filtered, truth, sample_rate)
    cost = tracking + float(jitter_weight) * jitter + float(lag_weight) * lag
    return {
        "tracking_rmse": tracking,
        "residual_jitter_rms": jitter,
        "lag_penalty_px": lag,
        "label_cost": cost,
    }


def default_cutoff_candidates(min_cutoff_hz: float = 1.5,
                              max_cutoff_hz: float = 18.0,
                              num_candidates: int = 32) -> np.ndarray:
    if not (0.05 < min_cutoff_hz < max_cutoff_hz):
        raise ValueError("Invalid cutoff candidate bounds")
    if num_candidates < 4:
        raise ValueError("num_candidates must be at least 4")
    grid = np.geomspace(min_cutoff_hz, max_cutoff_hz, int(num_candidates))
    anchors = np.array([3.0, 4.0, 6.0, 8.0, 12.0])
    candidates = np.unique(np.concatenate((grid, anchors[(anchors >= min_cutoff_hz) & (anchors <= max_cutoff_hz)])))
    return candidates


def select_optimal_cutoff(observed_future: np.ndarray, truth_future: np.ndarray,
                          history: np.ndarray, candidates=None,
                          sample_rate: float = 120.0,
                          jitter_weight: float = 0.5,
                          lag_weight: float = 0.5):
    if candidates is None:
        candidates = default_cutoff_candidates()
    results = []
    for cutoff in candidates:
        metrics = score_cutoff(
            observed_future, truth_future, history, float(cutoff), sample_rate,
            jitter_weight=jitter_weight, lag_weight=lag_weight,
        )
        results.append((float(metrics["label_cost"]), float(cutoff), metrics))
    _, best_cutoff, best_metrics = min(results, key=lambda item: item[0])
    return best_cutoff, best_metrics
