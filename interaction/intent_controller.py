"""History-aware confidence-gated intent controller for assisted pointing."""
from __future__ import annotations

from time import monotonic
from typing import Iterable

from shared.contracts import FilterOutput, IntentDecision, MotionFeatures, Target
from interaction.target_analyzer import MovementHistory, rank_targets
from ml.target_policy import TargetIntentPolicy
from interaction.sticky_target import StickyTargetController


class IntentController:
    def __init__(self, confidence_threshold: float = 0.50,
                 ambiguity_margin: float = 0.10, max_assist_px: float = 5.0,
                 persistence_seconds: float = 0.25, policy: TargetIntentPolicy | None = None,
                 max_assist_speed: float = 90.0,
                 sticky_retention_seconds: float = 0.35,
                 sticky_exit_tolerance: float = 20.0):
        self.confidence_threshold = float(confidence_threshold)
        self.ambiguity_margin = float(ambiguity_margin)
        self.max_assist_px = float(max_assist_px)
        self.max_assist_speed = float(max_assist_speed)
        self.persistence_seconds = float(persistence_seconds)
        self.sticky_retention_seconds = float(sticky_retention_seconds)
        self.sticky_exit_tolerance = float(sticky_exit_tolerance)
        self.sticky = StickyTargetController(exit_margin_px=sticky_exit_tolerance,
                                             grace_seconds=sticky_retention_seconds,
                                             switch_confirmation_seconds=0.25)
        self.acquisition_base_px = 30.0
        self.acquisition_min_px = 15.0
        self.acquisition_max_px = 50.0
        self.variability_sensitivity = 1.0
        self.overshoot_sensitivity = 0.5
        self.profile_features: dict[str, float] = {}
        self.adaptive_enlargement_enabled = True
        self.sticky_enabled = True
        self.history = MovementHistory()
        self.cursor_history = MovementHistory()
        self.policy = policy or TargetIntentPolicy()
        self._candidate_id: str | None = None
        self._candidate_since: float | None = None
        self._confirmed_target_id: str | None = None
        self._confirmed_score = 0.0
        self._confirmed_exit_since: float | None = None
        self.speed_px_s = 0.0
        self._last_update_time: float | None = None
        self.diagnostics = {
            "candidate_id": None, "score": 0.0, "second_score": 0.0,
            "margin": 0.0, "velocity_x": 0.0, "velocity_y": 0.0,
            "candidate_age": 0.0, "reason": "Waiting for movement history",
            "acquisition_margin_px": 30.0,
        }

    def reset(self) -> None:
        self.history.reset()
        self.cursor_history.reset()
        self.sticky.reset()
        self._candidate_id = None
        self._candidate_since = None
        self._confirmed_target_id = None
        self._confirmed_score = 0.0
        self._confirmed_exit_since = None
        self.speed_px_s = 0.0
        self._last_update_time = None
        self.diagnostics = {
            "candidate_id": None, "score": 0.0, "second_score": 0.0,
            "margin": 0.0, "velocity_x": 0.0, "velocity_y": 0.0,
            "candidate_age": 0.0, "reason": "Waiting for movement history",
            "acquisition_margin_px": 30.0,
        }

    def _retain_hovered_intent(self, x: float, y: float,
                               targets: list[Target], now: float):
        """Keep a confirmed target stable while the filtered cursor remains on it."""
        if not self.sticky_enabled:
            self.sticky.reset()
            self._confirmed_target_id = None
            return None
        if self.sticky.target_id is None and self._confirmed_target_id:
            self.sticky.acquire(self._confirmed_target_id)
        diag = self.diagnostics
        target = self.sticky.retain(
            x=x, y=y, now=now, targets=targets,
            velocity_x=diag.get("velocity_x", 0.0), velocity_y=diag.get("velocity_y", 0.0),
        )
        if target is None:
            self._confirmed_target_id = None
            return None
        self.diagnostics["candidate_id"] = target.id
        self.diagnostics["score"] = max(self._confirmed_score, self.confidence_threshold)
        self.diagnostics["candidate_age"] = max(
            0.0, now - (self._candidate_since if self._candidate_since is not None else now)
        )
        self.diagnostics["reason"] = "Sticky target retained"
        return IntentDecision("target_confident", target.id,
                              max(self._confirmed_score, self.confidence_threshold), 0.0, 0.0)

    def decide(self, raw_x: float, raw_y: float, filtered: FilterOutput,
               features: MotionFeatures, targets: Iterable[Target],
               override: bool = False, timestamp: float | None = None,
               retention_position: tuple[float, float] | None = None) -> IntentDecision:
        if override:
            self.reset()
            return IntentDecision("override", None, 0.0, 0.0, 0.0)

        now = float(timestamp if timestamp is not None else monotonic())
        dt = (1.0 / 120.0 if self._last_update_time is None else
              min(0.05, max(0.0, now - self._last_update_time)))
        self._last_update_time = now
        # Tremor amplitude is estimated from the motion spectrum. Expand only
        # the virtual acquisition frame; the desktop's real click geometry is
        # never changed. Cap expansion to keep adjacent controls distinct.
        sigma = max(self.profile_features.get("motion_variability_rms", 0.0),
                    max(0.0, features.est_amplitude or 0.0))
        overshoot = self.profile_features.get("median_overshoot_px", 0.0)
        acquisition_margin = min(self.acquisition_max_px, max(self.acquisition_min_px,
            self.acquisition_base_px + self.variability_sensitivity * sigma
            + self.overshoot_sensitivity * overshoot))
        if not self.adaptive_enlargement_enabled:
            acquisition_margin = 0.0
        self.diagnostics["acquisition_margin_px"] = acquisition_margin
        # Keep separate histories for observed input and filtered pointer motion.
        # Rank intent from the filtered trajectory, never from the raw-filter
        # residual, which is delay and tremor rather than travel direction.
        self.history.add(now, raw_x, raw_y)
        self.cursor_history.add(now, filtered.x, filtered.y)
        target_list = list(targets)
        if not target_list:
            self.reset()
            self.diagnostics["reason"] = "No reachable targets"
            state = "oscillation_detected" if features.oscillation_score > 0.5 else "stable"
            return IntentDecision(state, None, 0.0, 0.0, 0.0)

        points = self.cursor_history.points
        estimate = self.cursor_history.estimate()
        self.speed_px_s = estimate.speed
        self.diagnostics.update(velocity_x=estimate.velocity_x, velocity_y=estimate.velocity_y)
        retain_x, retain_y = retention_position or (filtered.x, filtered.y)
        if len(points) < 4 or self.cursor_history.duration < 0.08:
            retained = self._retain_hovered_intent(retain_x, retain_y, target_list, now)
            if retained is not None:
                return retained
            self._candidate_id = None
            self._candidate_since = None
            self.diagnostics.update(candidate_id=None, score=0.0, second_score=0.0,
                                    margin=0.0, candidate_age=0.0,
                                    reason="Collecting movement history")
            return IntentDecision("target_ambiguous", None, 0.0, 0.0, 0.0)

        ranked = rank_targets(filtered.x, filtered.y, estimate.velocity_x,
                              estimate.velocity_y, target_list,
                              history=points, policy=self.policy,
                              acquisition_margin_px=acquisition_margin,
                              profile_features=self.profile_features)
        if not ranked:
            self.diagnostics.update(candidate_id=None, reason="No target candidates")
            return IntentDecision("target_ambiguous", None, 0.0, 0.0, 0.0)
        best_target, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        self.diagnostics.update(candidate_id=best_target.id, score=best_score,
                                second_score=second_score, margin=margin)

        if best_score < self.confidence_threshold or (len(ranked) > 1 and margin < self.ambiguity_margin):
            retained = self._retain_hovered_intent(retain_x, retain_y, target_list, now)
            if retained is not None:
                return retained
            self._candidate_id = None
            self._candidate_since = None
            self.diagnostics.update(candidate_id=best_target.id, reason="Low score or ambiguous targets")
            return IntentDecision("target_ambiguous", None, best_score, 0.0, 0.0)

        if best_target.id != self._candidate_id:
            self._candidate_id = best_target.id
            self._candidate_since = now
            self.diagnostics.update(candidate_age=0.0, reason="Confirming candidate")
            return IntentDecision("target_ambiguous", None, best_score, 0.0, 0.0)
        if self._candidate_since is None or now - self._candidate_since < self.persistence_seconds:
            self.diagnostics.update(candidate_age=max(0.0, now - (self._candidate_since or now)),
                                    reason="Confirming candidate")
            return IntentDecision("target_ambiguous", None, best_score, 0.0, 0.0)

        dx = best_target.center_x - filtered.x
        dy = best_target.center_y - filtered.y
        distance = max(1e-9, (dx * dx + dy * dy) ** 0.5)
        # Bound attraction by both distance and elapsed time. A fixed pixel
        # offset per frame becomes an enormous pull at high frame rates.
        magnitude = min(self.max_assist_px, self.max_assist_speed * dt, distance * 0.04)
        self._confirmed_target_id = best_target.id
        self.sticky.acquire(best_target.id)
        self._confirmed_score = best_score
        self._confirmed_exit_since = None
        self.diagnostics.update(candidate_age=now - (self._candidate_since or now),
                                reason="Intent confirmed")
        return IntentDecision("target_confident", best_target.id, best_score,
                              dx / distance * magnitude, dy / distance * magnitude)

    def apply_profile(self, profile) -> None:
        """Apply validated calibration thresholds without coupling to filter ML."""
        self.confidence_threshold = float(profile.intent_confidence_threshold)
        self.ambiguity_margin = float(profile.minimum_confidence_margin)
        self.acquisition_base_px = float(profile.acquisition_base_margin_px)
        self.acquisition_min_px = float(profile.acquisition_radius_min_px)
        self.acquisition_max_px = float(profile.acquisition_radius_max_px)
        self.variability_sensitivity = float(profile.variability_sensitivity)
        self.overshoot_sensitivity = float(profile.overshoot_sensitivity)
        self.sticky_exit_tolerance = float(profile.acquisition_margin_px + profile.sticky_exit_margin_px)
        self.sticky_retention_seconds = float(profile.sticky_grace_ms) / 1000.0
        self.sticky.exit_margin_px = self.sticky_exit_tolerance
        self.sticky.grace_seconds = self.sticky_retention_seconds
        self.profile_features = {
            "motion_variability_rms": float(profile.motion_variability_rms),
            "median_overshoot_px": float(profile.median_overshoot_px),
            "median_movement_speed": float(profile.median_movement_speed),
        }
