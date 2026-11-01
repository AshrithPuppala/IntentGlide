"""Validated, pseudonymous per-user settings and calibration summary."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import uuid


@dataclass
class UserProfile:
    profile_id: str = ""
    profile_version: int = 1
    dwell_seconds: float = 3.0
    dwell_tolerance_px: float = 32.0
    dwell_exit_grace_ms: float = 750.0
    magnetic_strength: str = "balanced"
    magnetic_strength_scale: float = 1.0
    magnetic_assistance_enabled: bool = True
    adaptive_target_enlargement_enabled: bool = True
    sticky_target_enabled: bool = True
    dwell_enabled: bool = False
    personalization_enabled: bool = True
    acquisition_margin_px: float = 30.0
    acquisition_base_margin_px: float = 30.0
    acquisition_radius_min_px: float = 15.0
    acquisition_radius_max_px: float = 50.0
    variability_sensitivity: float = 1.0
    overshoot_sensitivity: float = 0.5
    sticky_exit_margin_px: float = 20.0
    sticky_grace_ms: float = 400.0
    intent_confidence_threshold: float = 0.65
    minimum_confidence_margin: float = 0.10
    maximum_magnetic_speed_px_s: float = 145.0
    maximum_assistance_distance_px: float = 76.0
    maximum_dwell_speed_px_s: float = 75.0
    motion_variability_rms: float = 0.0
    median_overshoot_px: float = 0.0
    median_target_acquisition_time: float = 0.0
    median_movement_speed: float = 0.0
    calibration_trial_count: int = 0
    adaptation_source: str = "default"
    profile_schema_version: int = 1

    def __post_init__(self):
        if not self.profile_id:
            self.profile_id = "local-" + uuid.uuid4().hex[:12]
        self.validate()

    def validate(self):
        if self.profile_schema_version != 1 or self.profile_version < 1:
            raise ValueError("Unsupported user profile version")
        if self.magnetic_strength not in {"gentle", "balanced", "strong"}:
            raise ValueError("magnetic_strength must be gentle, balanced, or strong")
        if self.adaptation_source not in {"default", "calibration", "statistical"}:
            raise ValueError("Invalid adaptation source")
        for name in ("magnetic_assistance_enabled", "adaptive_target_enlargement_enabled",
                     "sticky_target_enabled", "dwell_enabled", "personalization_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        bounds = {
            "dwell_seconds": (2.0, 8.0), "dwell_tolerance_px": (10.0, 70.0),
            "dwell_exit_grace_ms": (200.0, 1500.0), "acquisition_margin_px": (15.0, 50.0),
            "acquisition_base_margin_px": (15.0, 50.0), "magnetic_strength_scale": (0.5, 1.5),
            "acquisition_radius_min_px": (10.0, 40.0), "acquisition_radius_max_px": (20.0, 80.0),
            "variability_sensitivity": (0.0, 3.0), "overshoot_sensitivity": (0.0, 2.0),
            "sticky_exit_margin_px": (15.0, 75.0), "sticky_grace_ms": (200.0, 1000.0),
            "intent_confidence_threshold": (0.5, 0.95), "minimum_confidence_margin": (0.02, 0.5),
            "maximum_magnetic_speed_px_s": (30.0, 250.0),
            "maximum_assistance_distance_px": (30.0, 150.0),
            "maximum_dwell_speed_px_s": (20.0, 180.0),
            "motion_variability_rms": (0.0, 100.0), "median_overshoot_px": (0.0, 300.0),
            "median_target_acquisition_time": (0.0, 120.0), "median_movement_speed": (0.0, 5000.0),
        }
        for name, (low, high) in bounds.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")
        if self.acquisition_radius_min_px >= self.acquisition_radius_max_px:
            raise ValueError("acquisition radius minimum must be below maximum")
        if self.calibration_trial_count < 0 or self.calibration_trial_count > 100000:
            raise ValueError("Invalid calibration_trial_count")
        if not self.profile_id.startswith("local-"):
            raise ValueError("profile_id must be a local pseudonymous identifier")
        return self

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "UserProfile":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in values.items() if k in allowed})
