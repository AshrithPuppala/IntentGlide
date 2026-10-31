
"""Training features using the same streaming DSP implementation as runtime."""
from __future__ import annotations

import numpy as np

from dsp.motion_analyzer import MotionAnalyzer
from shared.contracts import (
    FEATURE_NAMES as SHARED_FEATURE_NAMES,
    InputSample,
    to_vector,
)

FEATURE_NAMES = list(SHARED_FEATURE_NAMES)


def extract_features(
    window_xy,
    sample_rate: float = 120.0
) -> dict[str, float]:
    """
    Replay an observed XY window through the live MotionAnalyzer.

    Only observed positions are used. Ground-truth positions and
    future samples are never passed to this feature extractor.
    """
    xy = np.asarray(window_xy, dtype=float)

    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 32:
        raise ValueError("window_xy must have shape (n>=32, 2)")

    if not np.isfinite(xy).all():
        raise ValueError("window_xy contains NaN or infinite values")

    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive and finite")

    analyzer = MotionAnalyzer()
    features = None

    for i, (x, y) in enumerate(xy):
        sample = InputSample(
            timestamp=float(i / sample_rate),
            x=float(x),
            y=float(y),
            source="synthetic",
        )
        features = analyzer.update(sample)

    if features is None or not features.valid:
        raise ValueError("Insufficient samples to calculate valid DSP features")

    vector = np.asarray(to_vector(features), dtype=float)

    if len(vector) != len(FEATURE_NAMES):
        raise ValueError("DSP feature vector does not match shared schema")

    if not np.isfinite(vector).all():
        raise ValueError("Extracted features contain NaN or infinity")

    return dict(zip(FEATURE_NAMES, vector.tolist()))
