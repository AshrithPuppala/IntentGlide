"""DSP tests (Person 2). Run from the repo root:  python -m pytest tests/test_dsp.py -q"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.signal

# Allow `pytest` to be run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.contracts import FEATURE_NAMES, InputSample, to_vector
from dsp.fixed_filter import FixedFilter
from dsp.motion_analyzer import MotionAnalyzer


# =========================================================== fixed filter
def test_fixed_filter_initialization_and_reset():
    """Verify the filter initializes correctly and resets its state."""
    filt = FixedFilter(cutoff_hz=3.0, nominal_fs=120.0)
    assert not filt.fx.initialized

    s1 = InputSample(timestamp=0.0, x=100.0, y=200.0, source="synthetic")
    out1 = filt.update(s1)

    assert filt.fx.initialized
    assert out1.x == pytest.approx(100.0)  # steady-state init locks to first value
    assert out1.mode == "fixed"

    filt.reset()
    assert not filt.fx.initialized
    assert filt.fx.z1 == 0.0


def test_fixed_filter_8hz_attenuation():
    """8 Hz through a 3 Hz 2nd-order Butterworth -> amplitude ratio ~0.139."""
    fs, fc, f_test, duration = 120.0, 3.0, 8.0, 2.0
    t = np.arange(0, duration, 1.0 / fs)
    amplitude_in = 50.0
    x_in = amplitude_in * np.sin(2 * np.pi * f_test * t)

    filt = FixedFilter(cutoff_hz=fc, nominal_fs=fs)
    x_out = [filt.update(InputSample(t[i], v, 0.0, "synthetic")).x for i, v in enumerate(x_in)]

    settle_idx = int(0.5 * fs)
    ratio = np.max(np.abs(x_out[settle_idx:])) / amplitude_in
    assert 0.13 <= ratio <= 0.16, f"Expected ~0.14, got {ratio:.3f}"


def test_fixed_filter_matches_scipy():
    """Custom biquad must match scipy.signal.butter / lfilter."""
    fs, fc = 120.0, 3.0
    rng = np.random.default_rng(42)
    t = np.arange(0, 1.0, 1.0 / fs)
    x_in = rng.standard_normal(len(t)) * 20.0 + 100.0

    filt = FixedFilter(cutoff_hz=fc, nominal_fs=fs)
    custom_out = [filt.update(InputSample(t[i], v, 0.0, "synthetic")).x for i, v in enumerate(x_in)]

    b, a = scipy.signal.butter(2, fc, fs=fs, btype="low")
    zi = scipy.signal.lfilter_zi(b, a) * x_in[0]
    scipy_out, _ = scipy.signal.lfilter(b, a, x_in, zi=zi)

    max_error = np.max(np.abs(np.array(custom_out) - scipy_out))
    assert max_error < 1e-5, f"Custom filter deviates from SciPy by {max_error}"


# ============================================================ helpers
def run_stream(x_fn, y_fn=lambda t: 200.0, duration=1.5, fs=120.0,
               jitter=0.0, seed=0, analyzer=None):
    """Feed a synthetic stream through MotionAnalyzer; return (analyzer, all_features).

    jitter: fractional timestamp jitter (0.2 = +/-20% of dt). The signal is
    sampled at the jittered times, as a real pointer would be.
    """
    rng = np.random.default_rng(seed)
    analyzer = analyzer or MotionAnalyzer()
    dt = 1.0 / fs
    out = []
    for i in range(int(duration * fs)):
        t = i * dt + (rng.uniform(-jitter, jitter) * dt if jitter else 0.0) + 5.0  # offset: non-zero start
        out.append(analyzer.update(InputSample(t, float(x_fn(t)), float(y_fn(t)), "synthetic")))
    return analyzer, out


def sine(amp, f, base=300.0):
    return lambda t: base + amp * math.sin(2 * math.pi * f * t)


def min_jerk(t, t0, dur, dist):
    """Smooth intentional reach: minimum-jerk profile, `dist` px over `dur` s."""
    s = min(max((t - t0) / dur, 0.0), 1.0)
    return dist * (10 * s**3 - 15 * s**4 + 6 * s**5)


# ============================================================ analyzer
def test_warmup_is_invalid_and_not_smooth():
    _, feats = run_stream(sine(15, 8), duration=0.3)
    assert not feats[-1].valid
    assert feats[-1].oscillation_score == 0.0
    assert feats[-1].dominant_frequency is None
    assert feats[-1].directional_consistency == 0.0  # warm-up must not read as "perfectly smooth"


def test_detects_8hz_oscillation():
    _, feats = run_stream(sine(15, 8.0), duration=1.5)
    f = feats[-1]
    assert f.valid
    assert abs(f.dominant_frequency - 8.0) < 1.0
    assert f.oscillation_score > 0.7
    assert f.periodicity_confidence > 0.6
    assert f.spectral_entropy < 0.5


@pytest.mark.parametrize("freq", [8.0, 8.3, 8.7])
def test_amplitude_estimate_is_accurate_off_bin(freq):
    """15 px sine, any frequency between FFT bins -> est_amplitude within ~8%."""
    _, feats = run_stream(sine(15.0, freq), duration=1.5)
    assert feats[-1].est_amplitude == pytest.approx(15.0, rel=0.08)


@pytest.mark.parametrize("f_true", [4.5, 6.0, 10.0, 11.5])
def test_frequency_estimate_across_band(f_true):
    _, feats = run_stream(sine(10.0, f_true), duration=1.5)
    assert abs(feats[-1].dominant_frequency - f_true) < 0.7


def test_axis_power_is_separated():
    _, feats = run_stream(sine(12.0, 8.0), y_fn=lambda t: 200.0, duration=1.5)
    f = feats[-1]
    assert f.x_band_power > 50 * f.y_band_power + 1e-6
    assert f.x_band_power == pytest.approx(12.0**2, rel=0.15)


def test_oscillation_on_both_axes_adds_up():
    _, feats = run_stream(sine(10.0, 8.0), y_fn=sine(10.0, 8.0, base=200.0), duration=1.5)
    assert feats[-1].est_amplitude == pytest.approx(math.hypot(10.0, 10.0), rel=0.08)


def test_slow_ramp_is_not_oscillation():
    _, feats = run_stream(lambda t: 100.0 + 150.0 * t, duration=1.5)
    assert feats[-1].oscillation_score < 0.2
    assert feats[-1].directional_consistency > 0.95
    assert feats[-1].velocity == pytest.approx(150.0, rel=0.05)


@pytest.mark.parametrize("reach_start", [5.2, 5.5, 5.8])
def test_smooth_intentional_reach_is_not_oscillation(reach_start):
    """A large, fast minimum-jerk reach (300 px in 0.6 s) must not look like tremor."""
    _, feats = run_stream(lambda t: 100.0 + min_jerk(t, reach_start, 0.6, 300.0), duration=1.5)
    assert max(f.oscillation_score for f in feats if f.valid) < 0.25


def test_tremor_on_top_of_ramp_is_still_detected():
    x_fn = lambda t: 100.0 + 150.0 * t + 15.0 * math.sin(2 * math.pi * 8.0 * t)
    _, feats = run_stream(x_fn, duration=1.5)
    f = feats[-1]
    assert f.oscillation_score > 0.5
    assert abs(f.dominant_frequency - 8.0) < 1.0


def test_tremor_on_top_of_fast_reach_is_still_detected():
    x_fn = lambda t: 100.0 + min_jerk(t, 5.3, 0.6, 300.0) + 12.0 * math.sin(2 * math.pi * 8.0 * t)
    _, feats = run_stream(x_fn, duration=1.5)
    assert feats[-1].oscillation_score > 0.4
    assert abs(feats[-1].dominant_frequency - 8.0) < 1.0


def test_sub_pixel_noise_scores_near_zero():
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.2, size=(2000, 2))
    state = {"i": 0}

    def x_fn(t):
        state["i"] += 1
        return 300.0 + noise[state["i"], 0]

    def y_fn(t):
        return 200.0 + noise[state["i"], 1]

    _, feats = run_stream(x_fn, y_fn, duration=1.5)
    assert feats[-1].oscillation_score < 0.15


def test_stationary_cursor():
    _, feats = run_stream(lambda t: 300.0, duration=1.5)
    f = feats[-1]
    assert f.oscillation_score == 0.0
    assert f.dominant_frequency is None
    assert f.velocity == pytest.approx(0.0, abs=1e-6)
    assert all(math.isfinite(v) for v in to_vector(f))


def test_timestamp_jitter_gives_similar_result():
    _, clean = run_stream(sine(15.0, 8.0), duration=1.5)
    _, noisy = run_stream(sine(15.0, 8.0), duration=1.5, jitter=0.2, seed=3)
    a, b = clean[-1], noisy[-1]
    assert abs(a.dominant_frequency - b.dominant_frequency) < 0.5
    assert b.est_amplitude == pytest.approx(a.est_amplitude, rel=0.15)
    assert b.oscillation_score > 0.6


@pytest.mark.parametrize("fs", [60.0, 120.0, 250.0, 500.0])
def test_sample_rate_independence(fs):
    _, feats = run_stream(sine(15.0, 8.0), duration=1.5, fs=fs)
    f = feats[-1]
    assert abs(f.dominant_frequency - 8.0) < 1.0
    assert f.est_amplitude == pytest.approx(15.0, rel=0.12)
    assert f.oscillation_score > 0.7


def test_direction_consistency_low_for_pure_oscillation():
    _, feats = run_stream(sine(15.0, 8.0), duration=1.5)
    assert feats[-1].directional_consistency < 0.3


def test_velocity_tracks_a_sudden_flick_quickly():
    """Current speed must react within ~50 ms, not average over the whole window."""
    x_fn = lambda t: 300.0 if t < 6.4 else 300.0 + 1000.0 * (t - 6.4)
    _, feats = run_stream(x_fn, duration=1.5)  # t runs 5.0 -> 6.5
    assert feats[-1].velocity > 800.0


def test_feature_vector_is_finite_and_ordered():
    _, feats = run_stream(sine(15.0, 8.0), duration=1.5)
    vec = to_vector(feats[-1])
    assert len(vec) == len(FEATURE_NAMES)
    assert all(math.isfinite(v) for v in vec)
    # warm-up vector (dominant_frequency=None) must also be finite
    _, warm = run_stream(sine(15.0, 8.0), duration=0.2)
    assert all(math.isfinite(v) for v in to_vector(warm[-1]))


def test_outputs_stay_in_documented_ranges():
    rng = np.random.default_rng(7)
    _, feats = run_stream(lambda t: 300 + 40 * rng.standard_normal(),
                          lambda t: 200 + 40 * rng.standard_normal(), duration=2.0)
    for f in feats:
        assert 0.0 <= f.oscillation_score <= 1.0
        assert 0.0 <= f.periodicity_confidence <= 1.0
        assert 0.0 <= f.directional_consistency <= 1.0
        assert 0.0 <= f.spectral_entropy <= 1.0


def test_bad_timestamps_and_values_are_ignored():
    an = MotionAnalyzer()
    for i in range(130):
        an.update(InputSample(5.0 + i / 120.0, 300.0 + 15 * math.sin(2 * math.pi * 8 * i / 120), 200.0, "synthetic"))
    before = an._last
    assert an.update(InputSample(5.5, 999.0, 200.0, "synthetic")) is before       # time went backwards
    assert an.update(InputSample(9.0, float("nan"), 200.0, "synthetic")) is before  # NaN
    assert an.update(InputSample(5.0 + 129 / 120.0, 1.0, 1.0, "synthetic")) is before  # duplicate time


def test_reset_clears_state():
    an, feats = run_stream(sine(15.0, 8.0), duration=1.5)
    assert feats[-1].valid
    an.reset()
    out = an.update(InputSample(100.0, 0.0, 0.0, "synthetic"))
    assert not out.valid and out.oscillation_score == 0.0


def test_processing_time_is_low():
    an = MotionAnalyzer()
    t0 = time.perf_counter()
    n = 600
    for i in range(n):
        t = i / 120.0
        an.update(InputSample(t, 300 + 15 * math.sin(2 * math.pi * 8 * t), 200.0, "synthetic"))
    per_update_ms = (time.perf_counter() - t0) / n * 1000
    assert per_update_ms < 3.0, f"{per_update_ms:.2f} ms/update is too slow for a 60+ FPS UI"