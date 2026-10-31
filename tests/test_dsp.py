import pytest
import numpy as np
import scipy.signal

# Add the repository root to the path so tests can run easily
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.contracts import InputSample
from dsp.fixed_filter import FixedFilter

def test_fixed_filter_initialization_and_reset():
    """Verify the filter initializes correctly and resets its state."""
    filt = FixedFilter(cutoff_hz=3.0, nominal_fs=120.0)
    assert not filt.fx.initialized
    
    # Process one sample
    s1 = InputSample(timestamp=0.0, x=100.0, y=200.0, source="synthetic")
    out1 = filt.update(s1)
    
    assert filt.fx.initialized
    assert out1.x == 100.0  # Steady state initialization should lock to first value
    assert out1.mode == "fixed"
    
    filt.reset()
    assert not filt.fx.initialized
    assert filt.fx.z1 == 0.0

def test_fixed_filter_8hz_attenuation():
    """
    Workflow requirement: An 8 Hz sine wave fed into a 3 Hz 2nd-order
    Butterworth filter should be attenuated to roughly 0.14 to 0.15 amplitude.
    """
    fs = 120.0
    fc = 3.0
    f_test = 8.0
    duration = 2.0
    
    t = np.arange(0, duration, 1.0 / fs)
    amplitude_in = 50.0
    x_in = amplitude_in * np.sin(2 * np.pi * f_test * t)
    
    filt = FixedFilter(cutoff_hz=fc, nominal_fs=fs)
    
    x_out = []
    for i, val in enumerate(x_in):
        s = InputSample(timestamp=t[i], x=val, y=0.0, source="synthetic")
        x_out.append(filt.update(s).x)
        
    # Discard the first 0.5 seconds to let the filter settle into steady state
    settle_idx = int(0.5 * fs)
    amplitude_out = np.max(np.abs(x_out[settle_idx:]))
    
    ratio = amplitude_out / amplitude_in
    
    # Expected ratio: 1 / sqrt(1 + (8/3)^4) ≈ 0.139
    assert 0.13 <= ratio <= 0.16, f"Expected 8Hz attenuation ratio ~0.14, got {ratio:.3f}"

def test_fixed_filter_matches_scipy():
    """
    Workflow requirement: Test custom biquad against scipy.signal.butter.
    They should match closely to prove the math is correct.
    """
    fs = 120.0
    fc = 3.0
    
    # Generate a random noisy trajectory
    np.random.seed(42)
    t = np.arange(0, 1.0, 1.0 / fs)
    x_in = np.random.randn(len(t)) * 20.0 + 100.0 
    
    # 1. Process via custom FixedFilter
    filt = FixedFilter(cutoff_hz=fc, nominal_fs=fs)
    custom_out = []
    for i, val in enumerate(x_in):
        s = InputSample(timestamp=t[i], x=val, y=0.0, source="synthetic")
        custom_out.append(filt.update(s).x)
        
    # 2. Process via SciPy
    b, a = scipy.signal.butter(2, fc, fs=fs, btype='low')
    
    # Calculate initial conditions matching our steady-state logic
    zi = scipy.signal.lfilter_zi(b, a) * x_in[0]
    scipy_out, _ = scipy.signal.lfilter(b, a, x_in, zi=zi)
    
    # 3. Compare outputs (allow for tiny floating-point drift)
    max_error = np.max(np.abs(np.array(custom_out) - scipy_out))
    assert max_error < 1e-5, f"Custom filter deviates from SciPy by {max_error}"