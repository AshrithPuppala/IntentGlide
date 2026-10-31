import math
import numpy as np
import pytest

from dsp.adaptive_butterworth import AdaptiveButterworthFilter
from shared.contracts import InputSample, MotionFeatures


def _tone_amplitude(values, times, frequency):
    design = np.column_stack([
        np.sin(2 * np.pi * frequency * times),
        np.cos(2 * np.pi * frequency * times),
        np.ones_like(times),
    ])
    coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(math.hypot(coeff[0], coeff[1]))


def test_lower_cutoff_suppresses_8hz_more_than_higher_cutoff():
    fs, frequency, duration = 120.0, 8.0, 2.5
    results = {}
    for cutoff in (3.0, 12.0):
        filt = AdaptiveButterworthFilter(nominal_fs=fs)
        output, times = [], []
        for i in range(int(fs * duration)):
            t = i / fs
            sample = InputSample(t, 12 * math.sin(2 * math.pi * frequency * t), 0.0, "synthetic")
            out = filt.update(sample, cutoff_target_hz=cutoff)
            output.append(out.x)
            times.append(t)
        output, times = np.asarray(output), np.asarray(times)
        mask = times > 1.0
        results[cutoff] = _tone_amplitude(output[mask], times[mask], frequency)
    assert results[3.0] < 0.30 * 12.0
    assert results[3.0] < results[12.0]


def test_cutoff_is_bounded_and_slew_limited():
    filt = AdaptiveButterworthFilter(min_cutoff_hz=1.5, max_cutoff_hz=18.0, nominal_fs=120.0)
    previous = None
    values = []
    for i in range(120):
        target = 1.5 if i < 60 else 18.0
        out = filt.update(InputSample(i / 120, float(i), 20.0, "synthetic"), cutoff_target_hz=target)
        values.append(filt.cutoff_hz)
        assert np.isfinite(out.x) and np.isfinite(out.y)
        assert 1.5 <= filt.cutoff_hz <= 18.0
        if previous is not None:
            # Cutoff is continuous and cannot jump directly to the new target.
            assert abs(math.log2(filt.cutoff_hz / previous)) < 0.10
        previous = filt.cutoff_hz
    assert values[0] == pytest.approx(1.5)
    assert values[-1] > values[59]


def test_warmup_uses_neutral_cutoff_and_reports_warmup():
    filt = AdaptiveButterworthFilter(warmup_cutoff_hz=8.0)
    out = filt.update(InputSample(0.0, 10.0, 20.0, "synthetic"),
                      MotionFeatures(0, None, 0, 0, 0, valid=False))
    assert filt.last_source == "warmup"
    assert filt.cutoff_hz == pytest.approx(8.0)
    assert (out.x, out.y) == (10.0, 20.0)


def test_invalid_and_duplicate_samples_do_not_corrupt_state():
    filt = AdaptiveButterworthFilter()
    first = filt.update(InputSample(1.0, 10.0, 20.0, "synthetic"), cutoff_target_hz=4.0)
    assert filt.update(InputSample(1.0, 999.0, 999.0, "synthetic"), cutoff_target_hz=10.0) is first
    assert filt.update(InputSample(0.5, 999.0, 999.0, "synthetic"), cutoff_target_hz=10.0) is first
    assert filt.update(InputSample(1.1, float("nan"), 2.0, "synthetic"), cutoff_target_hz=5.0) is first


def test_source_change_resets_filter_history():
    filt = AdaptiveButterworthFilter()
    filt.update(InputSample(5.0, 100.0, 100.0, "synthetic"), cutoff_target_hz=3.0)
    out = filt.update(InputSample(0.0, 25.0, 30.0, "mouse"), cutoff_target_hz=3.0)
    assert (out.x, out.y) == (25.0, 30.0)
