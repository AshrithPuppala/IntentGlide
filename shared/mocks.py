"""Deterministic mock objects for building modules independently."""
from shared.contracts import FilterOutput, IntentDecision, InputSample, MotionFeatures


def sample_input_sequence(count: int = 20, start_time: float = 1.0):
    return [
        InputSample(
            timestamp=start_time + i / 120.0,
            x=100.0 + i * 1.5,
            y=120.0 + i * 0.8,
            source="synthetic",
        )
        for i in range(count)
    ]


def sample_motion_features() -> MotionFeatures:
    return MotionFeatures(0.65, 8.0, 0.82, 0.78, 180.0)


def sample_filter_outputs() -> tuple[FilterOutput, FilterOutput]:
    return (
        FilterOutput(120.0, 132.0, mode="fixed"),
        FilterOutput(123.0, 134.0, alpha=0.24, mode="adaptive"),
    )


def clear_target_decision() -> IntentDecision:
    return IntentDecision("target_confident", "target-1", 0.91, 2.0, -1.0)


def ambiguous_target_decision() -> IntentDecision:
    return IntentDecision("target_ambiguous", None, 0.48, 0.0, 0.0)
