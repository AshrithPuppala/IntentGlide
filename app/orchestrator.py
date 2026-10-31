"""Connect input, DSP, intent and safety modules through shared contracts."""
from __future__ import annotations

from time import perf_counter

from dsp.adaptive_filter import AdaptiveFilter
from dsp.fixed_filter import FixedFilter
from dsp.motion_analyzer import MotionAnalyzer
from interaction.assistance import apply_assistance
from interaction.intent_controller import IntentController
from shared.contracts import FilterOutput, InputSample


class IntentGlideOrchestrator:
    MODES = ("raw", "fixed", "adaptive", "full")

    def __init__(self, analysis_rate: float = 120.0):
        self.motion = MotionAnalyzer()
        self.fixed = FixedFilter(cutoff_hz=3.0, nominal_fs=analysis_rate)
        self.adaptive = AdaptiveFilter(nominal_fs=analysis_rate)
        self.intent = IntentController()
        self.mode = "full"

    def process(self, sample: InputSample, targets=(), override: bool = False):
        start = perf_counter()
        features = self.motion.update(sample)
        fixed_output = self.fixed.update(sample)
        adaptive_output = self.adaptive.update(sample, features)
        raw_output = FilterOutput(sample.x, sample.y, mode="raw")

        decision = None
        if self.mode == "raw":
            output = raw_output
        elif self.mode == "fixed":
            output = fixed_output
        elif self.mode == "adaptive":
            output = adaptive_output
        else:
            decision = self.intent.decide(
                raw_x=sample.x,
                raw_y=sample.y,
                filtered=adaptive_output,
                features=features,
                targets=targets,
                override=override,
            )
            output = apply_assistance(adaptive_output, decision)
            if override:
                output = raw_output

        elapsed_ms = (perf_counter() - start) * 1000.0
        return {
            "raw": raw_output,
            "fixed": fixed_output,
            "adaptive": adaptive_output,
            "output": output,
            "features": features,
            "decision": decision,
            "processing_time_ms": elapsed_ms,
        }
