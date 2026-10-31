from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class InputSample:
    timestamp: float            # monotonic seconds
    x: float                    # screen pixels
    y: float                    # screen pixels
    source: Literal["synthetic", "mouse", "hardware"]
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None

@dataclass
class MotionFeatures:
    oscillation_score: float            # 0.0-1.0
    dominant_frequency: Optional[float]
    periodicity_confidence: float       # 0.0-1.0
    directional_consistency: float      # 0.0-1.0
    velocity: float                     # pixels per second

@dataclass
class FilterOutput:
    x: float
    y: float
    alpha: Optional[float] = None
    mode: Literal["raw", "fixed", "adaptive"] = "adaptive"

@dataclass
class Target:
    id: str
    center_x: float
    center_y: float
    width: float
    height: float

@dataclass
class IntentDecision:
    state: Literal["stable", "oscillation_detected", "target_confident",
                   "target_ambiguous", "override"]
    target_id: Optional[str]
    confidence: float           # 0.0-1.0
    assist_dx: float            # bounded offset, pixels
    assist_dy: float            # bounded offset, pixels

@dataclass
class TrialResult:
    trial_id: int
    intended_target_id: str
    clicked_target_id: Optional[str]
    outcome: Literal["success", "wrong_target", "miss", "timeout"]
    success: bool
    start_time: float
    end_time: float
    elapsed: float              # seconds
    miss_count: int             # empty-space clicks before the end

@dataclass
class GroundTruth:
    timestamp: float
    x: float                    # intended (tremor-free) position
    y: float
    progress: float             # 0.0-1.0 along the reach
    arrived: bool

@dataclass
class GroundTruthLabel:
    trial_id: int
    timestamp: float
    intended_target_id: str
    true_state: Literal["stable", "oscillation_detected", "target_confident",
                        "target_ambiguous", "override"]
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
