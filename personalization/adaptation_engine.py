"""Explainable, bounded adaptation from calibration statistics and explicit feedback."""
from __future__ import annotations

from dataclasses import dataclass
import math

from personalization.user_profile import UserProfile


@dataclass(frozen=True)
class AdaptationResult:
    changed: bool
    source: str
    reason: str
    acquisition_margin_px: float
    magnetic_strength: str
    dwell_seconds: float


class AdaptationEngine:
    """Apply small reversible adjustments; inferred events never tune preferences."""
    STRENGTHS = ("gentle", "balanced", "strong")
    FEEDBACK = {"correct_accepted", "wrong_target_corrected", "magnetic_too_strong",
                "magnetic_too_weak", "dwell_too_short", "dwell_too_long",
                "sticky_too_short", "sticky_too_long"}

    def __init__(self, profile: UserProfile, learning_rate: float = 0.15):
        if not 0.01 <= learning_rate <= 0.5:
            raise ValueError("learning_rate must be in [0.01, 0.5]")
        self.profile = profile
        self.learning_rate = float(learning_rate)

    def acquisition_margin(self) -> float:
        p = self.profile
        value = (30.0 + p.variability_sensitivity * p.motion_variability_rms
                 + p.overshoot_sensitivity * p.median_overshoot_px)
        value += p.acquisition_base_margin_px - 30.0
        return min(p.acquisition_radius_max_px,
                   max(p.acquisition_radius_min_px, value))

    def apply_calibration(self, stats: dict) -> AdaptationResult:
        p = self.profile
        for field, key in (("motion_variability_rms", "motion_variability_rms"),
                           ("median_overshoot_px", "median_overshoot_px"),
                           ("median_target_acquisition_time", "median_acquisition_time_s"),
                           ("median_movement_speed", "median_movement_speed_px_s")):
            value = stats.get(key)
            if value is not None and math.isfinite(float(value)):
                old = getattr(p, field)
                setattr(p, field, old + self.learning_rate * (float(value) - old))
        p.calibration_trial_count = int(stats.get("trial_count", p.calibration_trial_count))
        corrections = max(0.0, float(stats.get("correction_frequency", 0.0)))
        overshoot = max(0.0, float(stats.get("median_overshoot_px", 0.0)))
        residual_rms = max(0.0, float(stats.get("motion_variability_rms", 0.0)))
        dwell_tolerance_target = min(60.0, max(18.0, 24.0 + 1.5 * residual_rms))
        p.dwell_tolerance_px += self.learning_rate * (dwell_tolerance_target - p.dwell_tolerance_px)
        p.sticky_grace_ms += self.learning_rate * (min(800.0, 300.0 + corrections*18.0) - p.sticky_grace_ms)
        p.maximum_magnetic_speed_px_s += self.learning_rate * (
            min(180.0, max(65.0, 150.0-overshoot*1.5)) - p.maximum_magnetic_speed_px_s)
        p.maximum_dwell_speed_px_s += self.learning_rate * (
            min(100.0, max(45.0, 90.0-corrections*1.2)) - p.maximum_dwell_speed_px_s)
        if float(stats.get("success_rate", 1.0)) < .70:
            p.intent_confidence_threshold = min(.90,
                p.intent_confidence_threshold + self.learning_rate*.02)
        p.acquisition_margin_px = self.acquisition_margin()
        p.adaptation_source = "calibration"
        p.profile_version += 1
        p.validate()
        return self._result(True, "Calibration statistics applied with bounded EWMA")

    def apply_feedback(self, feedback: str, *, explicit: bool = False) -> AdaptationResult:
        """Feedback must be an explicit user action; cancel/correction alone is inert."""
        if not explicit or feedback not in self.FEEDBACK:
            return self._result(False, "No explicit preference update")
        p = self.profile
        if feedback == "magnetic_too_strong":
            p.magnetic_strength_scale = max(.5, p.magnetic_strength_scale-.15*self.learning_rate)
            p.maximum_magnetic_speed_px_s = max(30.0, p.maximum_magnetic_speed_px_s-4.0)
        elif feedback == "magnetic_too_weak":
            p.magnetic_strength_scale = min(1.5, p.magnetic_strength_scale+.15*self.learning_rate)
            p.maximum_magnetic_speed_px_s = min(250.0, p.maximum_magnetic_speed_px_s+4.0)
        elif feedback == "dwell_too_short":
            p.dwell_seconds = min(8.0, p.dwell_seconds + 0.25)
        elif feedback == "dwell_too_long":
            p.dwell_seconds = max(2.0, p.dwell_seconds - 0.25)
        elif feedback == "wrong_target_corrected":
            p.intent_confidence_threshold = min(0.95,
                p.intent_confidence_threshold + .01*self.learning_rate)
        elif feedback == "sticky_too_short":
            p.sticky_grace_ms = min(1000.0, p.sticky_grace_ms+50.0*self.learning_rate)
        elif feedback == "sticky_too_long":
            p.sticky_grace_ms = max(200.0, p.sticky_grace_ms-50.0*self.learning_rate)
        # Correct acceptance is recorded, not used to increase assistance.
        p.adaptation_source = "statistical"
        p.profile_version += 1
        p.validate()
        return self._result(True, f"Explicit feedback: {feedback}")

    def observe_event(self, event: str) -> AdaptationResult:
        """Record-only hook for cancellation, inferred corrections, and outcomes."""
        return self._result(False, f"Observed event recorded only: {event}")

    def _result(self, changed: bool, reason: str) -> AdaptationResult:
        return AdaptationResult(changed, self.profile.adaptation_source, reason,
                                self.acquisition_margin(), self.profile.magnetic_strength,
                                self.profile.dwell_seconds)
