from shared.contracts import (
    InputSample, MotionFeatures, FilterOutput, Target, IntentDecision
)

# Fixed base time so tests are deterministic. ~125 Hz spacing.
MOCK_INPUT_SEQUENCE = [
    InputSample(timestamp=i * 0.008, x=100.0 + i * 2.0, y=100.0 + i * 2.0,
                source="synthetic")
    for i in range(100)
]

MOCK_MOTION_FEATURES = MotionFeatures(
    oscillation_score=0.8, dominant_frequency=8.0,
    periodicity_confidence=0.85, directional_consistency=0.9, velocity=250.0,
)

MOCK_FILTER_OUTPUT = FilterOutput(x=105.0, y=105.0, alpha=0.2, mode="adaptive")

MOCK_TARGET = Target(id="btn_1", center_x=500.0, center_y=500.0,
                     width=50.0, height=50.0)

MOCK_INTENT_DECISION = IntentDecision(
    state="target_confident", target_id="btn_1", confidence=0.9,
    assist_dx=2.5, assist_dy=2.5,
)
