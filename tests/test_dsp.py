from dsp.adaptive_filter import AdaptiveExponentialFilter
from dsp.fixed_filter import FixedButterworthFilter
from shared.contracts import InputSample, MotionFeatures


def test_filters_return_finite_coordinates():
    fixed = FixedButterworthFilter()
    adaptive = AdaptiveExponentialFilter()
    features = MotionFeatures(0.6, 8.0, 0.8, 0.7, 100.0)

    for i in range(20):
        sample = InputSample(i / 120, 100 + i, 150 + i * 0.5, "synthetic")
        f = fixed.update(sample)
        a = adaptive.update(sample, features)
        assert f.x == f.x and f.y == f.y
        assert a.x == a.x and a.y == a.y
        assert 0.12 <= a.alpha <= 0.85


def test_adaptive_filter_preserves_state():
    filt = AdaptiveExponentialFilter()
    features = MotionFeatures(0.9, 8.0, 0.8, 0.2, 10.0)
    first = filt.update(InputSample(0.0, 0, 0, "synthetic"), features)
    second = filt.update(InputSample(1/120, 100, 0, "synthetic"), features)
    assert second.x < 100
    assert second.x > first.x
