"""Connect input, DSP, intent and safety modules through shared contracts.

Existing modes are preserved. Mode ``ml_butterworth`` (key 5) is the new
Architecture-C experiment: a Random Forest predicts a Butterworth cutoff Hz,
which is applied by AdaptiveButterworthFilter. The existing ML-EMA path remains
available as ``adaptive`` and the original intent-assisted ``full`` mode stays
unchanged for a fair comparison.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from dsp.adaptive_filter import AdaptiveFilter
from dsp.adaptive_butterworth import AdaptiveButterworthFilter
from dsp.fixed_filter import FixedFilter
from dsp.motion_analyzer import MotionAnalyzer
from interaction.assistance import apply_assistance
from interaction.intent_controller import IntentController
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

    def set_mode(self, mode: str):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.reset()

    def process(self, sample: InputSample, targets=(), override: bool = False):
        start = perf_counter()
        features = self.motion.update(sample)
        fixed_output = self.fixed.update(sample)

        # Architecture B: existing ML-predicted EMA alpha.
        alpha_ml = None
        if self.alpha_policy and features.valid:
            predicted_alpha = self.alpha_policy.predict_alpha(features)
            if self.alpha_policy.source == "ml":
                alpha_ml = predicted_alpha
        adaptive_output = self.adaptive.update(sample, features, alpha_ml=alpha_ml)

        # Architecture C: ML predicts a physical cutoff in Hz for Butterworth.
        cutoff_ml = None
        cutoff_policy_source = "warmup"
        if features.valid:
            proposed_cutoff = self.cutoff_policy.predict_cutoff(features)
            cutoff_policy_source = self.cutoff_policy.source
            if self.cutoff_policy.source == "ml":
                cutoff_ml = proposed_cutoff
        butterworth_output = self.adaptive_butterworth.update(
            sample, features, cutoff_target_hz=cutoff_ml
        )

        raw_output = FilterOutput(sample.x, sample.y, mode="raw")
        decision = None
        if override:
            output = raw_output
            decision = IntentDecision("override", None, 0.0, 0.0, 0.0)
        elif self.mode == "raw":
            output = raw_output
        elif self.mode == "fixed":
            output = fixed_output
        elif self.mode == "adaptive":
            output = adaptive_output
        elif self.mode == "ml_butterworth":
            output = butterworth_output
        else:
            # Keep current full-system behaviour unchanged for A/B comparison.
            decision = self.intent.decide(
                raw_x=sample.x,
                raw_y=sample.y,
                filtered=adaptive_output,
                features=features,
                targets=targets,
                override=False,
            )
            output = apply_assistance(adaptive_output, decision)

        elapsed_ms = (perf_counter() - start) * 1000.0
        if self.mode == "ml_butterworth":
            active_filter_source = self.adaptive_butterworth.last_source
            active_cutoff_hz = self.adaptive_butterworth.cutoff_hz
        else:
            active_filter_source = self.adaptive.last_source
            active_cutoff_hz = self.adaptive.cutoff_hz

        return {
            "raw": raw_output,
            "fixed": fixed_output,
            "adaptive": adaptive_output,
            "ml_butterworth": butterworth_output,
            "output": output,
            "features": features,
            "decision": decision,
            "override": override,
            "filter_source": active_filter_source,
            "model_source": self.alpha_policy.source,
            "cutoff_policy_source": cutoff_policy_source,
            "cutoff_hz": active_cutoff_hz,
            "ml_butterworth_cutoff_hz": self.adaptive_butterworth.cutoff_hz,
            "processing_time_ms": elapsed_ms,
        }
