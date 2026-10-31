"""Shared data contracts; coordinate and timestamp conventions are in README.md."""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class InputSample:
    timestamp: float
    x: float
    y: float
    source: Literal["synthetic", "mouse", "hardware"]
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None


@dataclass
class MotionFeatures:
    oscillation_score: float
    dominant_frequency: Optional[float]
    periodicity_confidence: float
    directional_consistency: float
    velocity: float


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
    state: Literal[
        "stable",
        "oscillation_detected",
        "target_confident",
        "target_ambiguous",
        "override",
    ]
    target_id: Optional[str]
    confidence: float
    assist_dx: float
    assist_dy: float
