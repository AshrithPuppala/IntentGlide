"""Training feature extraction shared with the live DSP motion analyzer."""
from __future__ import annotations

import numpy as np

from dsp.motion_analyzer import MotionAnalyzer
from shared.contracts import FEATURE_NAMES as SHARED_FEATURE_NAMES
from shared.contracts import to_vector

FEATURE_NAMES = list(SHARED_FEATURE_NAMES)


def extract_features(window_xy, sample_rate: float = 120.0) -> dict[str, float]:
    """Run the live MotionAnalyzer over an observed, uniformly sampled XY window."""
    xy = np.asarray(window_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 32:
        raise ValueError("window_xy must have shape (n>=32, 2)")
    if not np.isfinite(xy).all():
        raise ValueError("window_xy contains NaN or infinite values")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive and finite")

    features = MotionAnalyzer().analyze_window(xy, sample_rate)
    vector = to_vector(features)
    return dict(zip(FEATURE_NAMES, vector))
