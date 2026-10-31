import math
import numpy as np
import pytest
from shared.contracts import InputSample
from dsp.motion_analyzer import MotionAnalyzer
from dsp.adaptive_filter import AdaptiveFilter, alpha_to_fc, fc_to_alpha
from dsp.fixed_filter import FixedFilter

# -------------------------------------------------------------------------
# 1. MOTION ANALYZER: FFT, Windowing, and Detrending Math
# -------------------------------------------------------------------------
def test_fft_amplitude_and_subbin_frequency():
    """Proves the Hann scaling and parabolic interpolation yield exact physical values."""
    an = MotionAnalyzer()
    
    fs = 120.0
    dt = 1.0 / fs
    true_amp = 15.0
    true_freq = 8.3  # Off-bin frequency
    
    # Feed 1.5 seconds so the 1.0-second sliding buffer is fully saturated
    for i in range(int(1.5 * fs)):
        t = i * dt
        x_val = true_amp * math.sin(2.0 * math.pi * true_freq * t)
        s = InputSample(timestamp=t, x=x_val, y=0.0, source="synthetic")
        feats = an.update(s)

    assert feats.valid is True
    # Amplitude within 10% tolerance for off-bin scalloping
    assert feats.est_amplitude == pytest.approx(true_amp, rel=0.10)
    # Frequency within 0.2 Hz despite 1 Hz FFT bin spacing
    assert feats.dominant_frequency == pytest.approx(true_freq, abs=0.20)
    assert feats.oscillation_score > 0.85


def test_cubic_detrending_removes_reach_but_keeps_tremor():
    """Proves degree-3 polyfit removes minimum-jerk curves without killing 8Hz sine."""
    an = MotionAnalyzer()
    
    fs = 120.0
    dt = 1.0 / fs
    true_amp = 10.0
    true_freq = 8.0
    
    for i in range(int(1.5 * fs)):
        t = i * dt
        tau = t / 1.5
        # 5th-order minimum jerk curve
        s_curve = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        reach_pos = 100.0 + s_curve * 400.0
        
        tremor_x = true_amp * math.sin(2.0 * math.pi * true_freq * t)
        tremor_y = true_amp * math.sin(2.0 * math.pi * true_freq * t)
        
        s = InputSample(timestamp=t, x=reach_pos + tremor_x, y=reach_pos + tremor_y, source="synthetic")
        feats = an.update(s)

    assert feats.valid is True
    expected_2d_amp = math.hypot(true_amp, true_amp)  # sqrt(10^2 + 10^2) ≈ 14.14
    assert feats.est_amplitude == pytest.approx(expected_2d_amp, rel=0.15)
    assert feats.dominant_frequency == pytest.approx(true_freq, abs=0.25)


# -------------------------------------------------------------------------
# 2. ADAPTIVE FILTER: Rate-Independent Cutoff Math
# -------------------------------------------------------------------------
def test_alpha_and_cutoff_inverses():
    """Proves the continuous-time to discrete-time conversions are exact inverses."""
    dt = 1.0 / 120.0
    
    for original_fc in [1.0, 3.0, 8.0, 15.0, 30.0]:
        alpha = fc_to_alpha(original_fc, dt)
        recovered_fc = alpha_to_fc(alpha, dt)
        assert recovered_fc == pytest.approx(original_fc, rel=1e-5)
        
    for original_alpha in [0.1, 0.5, 0.85, 0.99]:
        fc = alpha_to_fc(original_alpha, dt)
        recovered_alpha = fc_to_alpha(fc, dt)
        assert recovered_alpha == pytest.approx(original_alpha, rel=1e-5)


def test_adaptive_filter_exponential_convergence():
    """Proves the exponential smoother step-response converges correctly over time."""
    adapt = AdaptiveFilter(nominal_fs=120.0)
    
    s_step = InputSample(timestamp=0.0, x=100.0, y=100.0, source="synthetic")
    out = adapt.update(s_step, feats=None)
    assert out.x == 100.0
    
    class FakeFeats:
        valid = True
        oscillation_score = 1.0
        directional_consistency = 0.0
        velocity = 0.0
        
    s_step2 = InputSample(timestamp=0.01, x=200.0, y=200.0, source="synthetic")
    out2 = adapt.update(s_step2, feats=FakeFeats())
    
    expected_x = 100.0 + out2.alpha * (200.0 - 100.0)
    assert out2.x == pytest.approx(expected_x, rel=1e-5)


# -------------------------------------------------------------------------
# 3. FIXED FILTER: Biquad Gain Math
# -------------------------------------------------------------------------
def test_biquad_dc_gain_is_unity():
    """Proves the 2nd-order Butterworth has exactly 1.0 gain at DC (0 Hz)."""
    fixed = FixedFilter(cutoff_hz=3.0, nominal_fs=120.0)
    
    s0 = InputSample(timestamp=0.0, x=500.0, y=-250.0, source="synthetic")
    fixed.update(s0)
    
    for i in range(1, 10):
        out = fixed.update(InputSample(timestamp=i * 0.01, x=500.0, y=-250.0, source="synthetic"))
        
    assert out.x == pytest.approx(500.0, rel=1e-5)
    assert out.y == pytest.approx(-250.0, rel=1e-5)