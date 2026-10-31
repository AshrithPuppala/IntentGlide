from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple
import numpy as np

# Feature names in the exact order required for Person 3's ML regressor
FEATURE_NAMES: Tuple[str, ...] = (
    "oscillation_score",
    "dominant_frequency",
    "periodicity_confidence",
    "directional_consistency",
    "velocity",
    "acceleration",
    "jerk",
    "spectral_entropy",
    "est_amplitude",
    "x_band_power",
    "y_band_power",
)


@dataclass
class InputSample:
    timestamp: float                   # monotonic seconds
    x: float                           # screen pixels
    y: float                           # screen pixels
    source: Literal["synthetic", "mouse", "hardware"]
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None


@dataclass
class MotionFeatures:
    oscillation_score: float           # normalized 0.0–1.0
    dominant_frequency: Optional[float]
    periodicity_confidence: float      # normalized 0.0–1.0
    directional_consistency: float     # normalized 0.0–1.0
    velocity: float                    # pixels per second
    acceleration: Optional[float] = None
    jerk: Optional[float] = None
    spectral_entropy: Optional[float] = None
    est_amplitude: Optional[float] = None
    x_band_power: Optional[float] = None
    y_band_power: Optional[float] = None
    valid: bool = True


def to_vector(feats: MotionFeatures) -> List[float]:
    """
    Converts MotionFeatures to a fixed-order numeric list for ML inference/training.
    Fills None values with neutral defaults so the vector is always valid and finite.
    """
    dom_freq = feats.dominant_frequency if feats.dominant_frequency is not None else 0.0
    return [
        float(feats.oscillation_score),
        float(dom_freq),
        float(feats.periodicity_confidence),
        float(feats.directional_consistency),
        float(feats.velocity),
        float(feats.acceleration or 0.0),
        float(feats.jerk or 0.0),
        float(feats.spectral_entropy or 0.0),
        float(feats.est_amplitude or 0.0),
        float(feats.x_band_power or 0.0),
        float(feats.y_band_power or 0.0),
    ]


@dataclass
class FilterOutput:
    x: float
    y: float
    alpha: Optional[float] = None
    mode: Literal["raw", "fixed", "adaptive"] = "adaptive"
    processing_time_ms: Optional[float] = None


@dataclass
class Target:
    id: str
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass
class IntentDecision:
    state: Literal[
        "stable", "oscillation_detected", "target_confident",
        "target_ambiguous", "override"
    ]
    target_id: Optional[str]
    confidence: float                  # normalized 0.0–1.0
    assist_dx: float                   # bounded pixels
    assist_dy: float                   # bounded pixels
@dataclass
class TrialResult:
    trial_id: int
    intended_target_id: str
    clicked_target_id: Optional[str]
    outcome: Literal["success", "wrong_target", "miss", "timeout"]
    success: bool
    start_time: float
    end_time: float
    elapsed: float
    miss_count: int


@dataclass
class GroundTruth:
    timestamp: float
    x: float
    y: float
    progress: float
    arrived: bool


@dataclass
class GroundTruthLabel:
    trial_id: int
    timestamp: float
    intended_target_id: str
    true_state: Literal[
        "stable", "oscillation_detected", "target_confident",
        "target_ambiguous", "override"
    ]
    phase: Literal["onset", "transit", "approach", "dwell"]
    progress: float
    truth_x: float
    truth_y: float
    dist_to_target: float
    tremor_active: bool
    tremor_freq: float
    tremor_amp: float
    ambiguous: bool
    rival_target_id: Optional[str]
    override: bool