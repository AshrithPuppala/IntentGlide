import math
import numpy as np
import scipy.signal
from collections import deque
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.contracts import InputSample, MotionFeatures

class MotionAnalyzer:
    """
    Extracts frequency-domain and kinematic features from a sliding window 
    of cursor movements for the adaptive DSP filter and ML filter policy.
    """
    def __init__(self, window_size_samples: int = 120, nominal_fs: float = 120.0):
        self.window_size = window_size_samples
        self.nominal_fs = nominal_fs
        
        # Sliding buffers for timestamp, X, and Y
        self.t_buf = deque(maxlen=self.window_size)
        self.x_buf = deque(maxlen=self.window_size)
        self.y_buf = deque(maxlen=self.window_size)
        
        # Tremor band definition (Hz)
        self.band_min = 4.0
        self.band_max = 12.0

    def update(self, s: InputSample) -> MotionFeatures:
        """Add a new sample and recalculate all motion features."""
        self.t_buf.append(s.timestamp)
        self.x_buf.append(s.x)
        self.y_buf.append(s.y)
        
        # Return empty/default features until the buffer is reasonably full (e.g., 50%)
        if len(self.t_buf) < self.window_size // 2:
            return self._default_features()
            
        return self._extract_features()

    def _extract_features(self) -> MotionFeatures:
        # Convert to numpy arrays
        t_raw = np.array(self.t_buf)
        x_raw = np.array(self.x_buf)
        y_raw = np.array(self.y_buf)
        
        # 1. Resample onto a uniform time grid to fix Pygame polling jitter
        N = len(t_raw)
        t_uniform = np.linspace(t_raw[0], t_raw[-1], N)
        dt = (t_raw[-1] - t_raw[0]) / max(1, N - 1)
        
        if dt <= 0:
            return self._default_features()
            
        x_uni = np.interp(t_uniform, t_raw, x_raw)
        y_uni = np.interp(t_uniform, t_raw, y_raw)
        fs_actual = 1.0 / dt

        # 2. Kinematics (Velocity, Accel, Jerk, Directional Consistency)
        # Using finite differences
        vx = np.gradient(x_uni, dt)
        vy = np.gradient(y_uni, dt)
        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)
        jx = np.gradient(ax, dt)
        jy = np.gradient(ay, dt)
        
        v_mag = np.sqrt(vx**2 + vy**2)
        a_mag = np.sqrt(ax**2 + ay**2)
        j_mag = np.sqrt(jx**2 + jy**2)
        
        # Directional consistency: magnitude of the mean unit velocity vector
        epsilon = 1e-6
        u_x = vx / (v_mag + epsilon)
        u_y = vy / (v_mag + epsilon)
        dir_consistency = float(np.sqrt(np.mean(u_x)**2 + np.mean(u_y)**2))
        
        # Average magnitudes over the window for the ML regressor
        velocity_mean = float(np.mean(v_mag))
        accel_mean = float(np.mean(a_mag))
        jerk_mean = float(np.mean(j_mag))

        # 3. Frequency-Domain Features (Dual-Axis)
        # Detrend to remove deliberate smooth motion (low frequency)
        x_detrend = scipy.signal.detrend(x_uni)
        y_detrend = scipy.signal.detrend(y_uni)
        
        # Apply Hann window to prevent spectral leakage
        hann_win = np.hanning(N)
        x_win = x_detrend * hann_win
        y_win = y_detrend * hann_win
        
        # FFT and Power Spectral Density
        X_fft = np.fft.rfft(x_win)
        Y_fft = np.fft.rfft(y_win)
        freqs = np.fft.rfftfreq(N, d=dt)
        
        # Combine power from both axes
        P_total = np.abs(X_fft)**2 + np.abs(Y_fft)**2
        
        # Ignore DC component (0 Hz) for spectral metrics
        P_non_dc = P_total[1:]
        total_energy = np.sum(P_non_dc)
        
        if total_energy < epsilon:
            return MotionFeatures(
                oscillation_score=0.0,
                dominant_frequency=None,
                periodicity_confidence=0.0,
                directional_consistency=dir_consistency,
                velocity=velocity_mean,
                acceleration=accel_mean,
                jerk=jerk_mean,
                spectral_entropy=0.0,
                est_amplitude=0.0
            )

        # 4. Extract Tremor Band Specifics (4-12 Hz)
        band_mask = (freqs >= self.band_min) & (freqs <= self.band_max)
        P_band = P_total[band_mask]
        freqs_band = freqs[band_mask]
        
        band_energy = np.sum(P_band)
        oscillation_score = float(np.clip(band_energy / total_energy, 0.0, 1.0))
        
        if band_energy > epsilon and len(P_band) > 0:
            peak_idx = np.argmax(P_band)
            dominant_freq = float(freqs_band[peak_idx])
            periodicity_conf = float(np.clip(P_band[peak_idx] / band_energy, 0.0, 1.0))
        else:
            dominant_freq = None
            periodicity_conf = 0.0

        # Estimated amplitude: derives physical pixels from FFT variance
        # Variance = Energy / N^2. Amp approx = 2 * sqrt(2 * variance)
        var_est = band_energy / (N**2)
        est_amp = float(2.0 * np.sqrt(2.0 * var_est))
        
        # 5. Spectral Entropy
        # Normalize non-DC power to sum to 1 to act as a probability distribution
        p_norm = P_non_dc / total_energy
        # -sum(p * log2(p)). Normalize by log2(N) so it bounds between 0 and 1
        entropy = -np.sum(p_norm * np.log2(p_norm + epsilon))
        max_entropy = np.log2(len(p_norm))
        spectral_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        return MotionFeatures(
            oscillation_score=oscillation_score,
            dominant_frequency=dominant_freq,
            periodicity_confidence=periodicity_conf,
            directional_consistency=dir_consistency,
            velocity=velocity_mean,
            acceleration=accel_mean,
            jerk=jerk_mean,
            spectral_entropy=spectral_entropy,
            est_amplitude=est_amp
        )

    def _default_features(self) -> MotionFeatures:
        """Returns baseline features when the buffer is empty or too short."""
        return MotionFeatures(
            oscillation_score=0.0,
            dominant_frequency=None,
            periodicity_confidence=0.0,
            directional_consistency=1.0,
            velocity=0.0,
            acceleration=0.0,
            jerk=0.0,
            spectral_entropy=0.0,
            est_amplitude=0.0
        )

if __name__ == "__main__":
    # Quick Self-Test
    from shared.mocks import generate_mock_trajectory
    
    analyzer = MotionAnalyzer(window_size_samples=120)
    samples = generate_mock_trajectory(duration_sec=1.0, tremor_hz=8.0, tremor_amp=15.0)
    
    print("Feeding samples to Motion Analyzer...")
    feats = None
    for s in samples:
        feats = analyzer.update(s)
        
    print("\nFinal Motion Features Output:")
    for k, v in feats.__dict__.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")