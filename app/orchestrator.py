"""Connect input, DSP, intent and safety modules through shared contracts.

Filter selection and intent assistance are independent. Every filter mode can
use the same target-intent and assistance layer; ``full`` selects adaptive EMA
as its base filter for backward compatibility.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from dsp.adaptive_filter import AdaptiveFilter
from dsp.adaptive_butterworth import AdaptiveButterworthFilter
from dsp.fixed_filter import FixedFilter
from dsp.motion_analyzer import MotionAnalyzer
from interaction.intent_controller import IntentController
from interaction.magnetic_assistance import MagneticAssistance
from ml.filter_policy import FilterPolicy
from ml.butterworth_policy import ButterworthCutoffPolicy
from shared.contracts import FilterOutput, IntentDecision, InputSample


class IntentGlideOrchestrator:
    MODES = ("raw", "fixed", "adaptive", "full", "ml_butterworth")

    def __init__(self, analysis_rate: float = 120.0):
        self.analysis_rate = float(analysis_rate)
        self.motion = MotionAnalyzer()
        self.fixed = FixedFilter(cutoff_hz=3.0, nominal_fs=analysis_rate)
        self.adaptive = AdaptiveFilter(nominal_fs=analysis_rate)
        self.adaptive_butterworth = AdaptiveButterworthFilter(nominal_fs=analysis_rate)
        self.intent = IntentController()
        self.magnetic = MagneticAssistance()
        from personalization.user_profile import UserProfile
        self.user_profile = UserProfile()
        self.intent.apply_profile(self.user_profile)
        self.magnetic.apply_profile(self.user_profile)
        self.mode = "full"

        artifact_dir = Path(__file__).resolve().parents[1] / "ml" / "artifacts"
        self.alpha_policy = FilterPolicy(
            model_path=artifact_dir / "random_forest_alpha.joblib"
        )
        self.cutoff_policy = ButterworthCutoffPolicy(
            model_path=artifact_dir / "random_forest_cutoff.joblib"
        )

    def reset(self):
        """Clear filter/analyser state between trials, modes or input sources."""
        self.motion.reset()
        self.fixed.reset()
        self.adaptive.reset()
        self.adaptive_butterworth.reset()
        self.intent.reset()
        self.magnetic.reset()

    def set_mode(self, mode: str):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.reset()

    def apply_user_profile(self, profile):
        self.user_profile = profile
        self.intent.apply_profile(profile)
        self.magnetic.apply_profile(profile)

    def process(self, sample: InputSample, targets=(), override: bool = False,
                assist_enabled: bool = True):
        start = perf_counter()
        target_list = list(targets)
        features = self.motion.update(sample)
        raw_output = FilterOutput(sample.x, sample.y, mode="raw")
        fixed_output = adaptive_output = butterworth_output = None
        alpha_source = self.alpha_policy.source
        cutoff_policy_source = "inactive"
        if self.mode == "fixed":
            fixed_output = self.fixed.update(sample)
            base_output = fixed_output
        elif self.mode in {"adaptive", "full"}:
            alpha_ml = None
            if features.valid:
                predicted_alpha = self.alpha_policy.predict_alpha(features)
                alpha_source = self.alpha_policy.source
                if self.alpha_policy.source == "ml":
                    alpha_ml = predicted_alpha
            adaptive_output = self.adaptive.update(sample, features, alpha_ml=alpha_ml)
            base_output = adaptive_output
        elif self.mode == "ml_butterworth":
            cutoff_ml = None
            if features.valid:
                proposed_cutoff = self.cutoff_policy.predict_cutoff(features)
                cutoff_policy_source = self.cutoff_policy.source
                if self.cutoff_policy.source == "ml":
                    cutoff_ml = proposed_cutoff
            else:
                cutoff_policy_source = "warmup"
            butterworth_output = self.adaptive_butterworth.update(
                sample, features, cutoff_target_hz=cutoff_ml
            )
            base_output = butterworth_output
        else:
            base_output = raw_output
        decision = None
        if override:
            self.intent.reset()
            self.magnetic.reset()
            output = raw_output
            decision = IntentDecision("override", None, 0.0, 0.0, 0.0)
        else:
            # Target intent always runs on the selected filter's output. This
            # keeps filter experiments (including ML-BW) independent from the
            # assistance and dwell interaction layers.
            decision = self.intent.decide(
                raw_x=sample.x,
                raw_y=sample.y,
                filtered=base_output,
                features=features,
                targets=target_list,
                timestamp=sample.timestamp,
                retention_position=self.magnetic.last_output,
            )
            if assist_enabled:
                selected_target = next(
                    (target for target in target_list
                     if decision.state == "target_confident" and target.id == decision.target_id),
                    None,
                )
                diagnostics = self.intent.diagnostics
                output = self.magnetic.apply(
                    base_output, selected_target, timestamp=sample.timestamp,
                    velocity_x=diagnostics.get("velocity_x", 0.0),
                    velocity_y=diagnostics.get("velocity_y", 0.0),
                    tremor_amplitude=features.est_amplitude or 0.0,
                    confidence=decision.confidence,
                    intent_confirmed=decision.state == "target_confident",
                    manual_override=override,
                )
            else:
                self.magnetic.reset()
                output = base_output

        elapsed_ms = (perf_counter() - start) * 1000.0
        if self.mode == "ml_butterworth":
            active_filter_source = self.adaptive_butterworth.last_source
            active_cutoff_hz = self.adaptive_butterworth.cutoff_hz
        elif self.mode == "raw":
            active_filter_source = "raw"
            active_cutoff_hz = None
        elif self.mode == "fixed":
            active_filter_source = "fixed"
            active_cutoff_hz = self.fixed.fc
        else:
            active_filter_source = self.adaptive.last_source
            active_cutoff_hz = self.adaptive.cutoff_hz

        return {
            "raw": raw_output,
            "fixed": fixed_output,
            "adaptive": adaptive_output,
            "ml_butterworth": butterworth_output,
            "output": output,
            "filtered_output": base_output,
            "features": features,
            "decision": decision,
            "intent_source": self.intent.policy.get_status()["source"],
            "magnetic_profile": self.magnetic.profile,
            "target_lock": self.intent.sticky.target_id,
            "magnetic_displacement_px": (self.magnetic.offset_x ** 2 + self.magnetic.offset_y ** 2) ** 0.5,
            "personalization_source": self.user_profile.adaptation_source,
            "override": override,
            "filter_source": active_filter_source,
            "model_source": alpha_source,
            "cutoff_policy_source": cutoff_policy_source,
            "cutoff_hz": active_cutoff_hz,
            "ml_butterworth_cutoff_hz": self.adaptive_butterworth.cutoff_hz,
            "processing_time_ms": elapsed_ms,
        }
