from interaction.intent_controller import IntentController
from shared.contracts import FilterOutput, MotionFeatures, Target


def make_features():
    return MotionFeatures(0.2, None, 0.1, 0.8, 100.0)


def test_manual_override_disables_assistance():
    controller = IntentController()
    decision = controller.decide(
        100, 100, FilterOutput(100, 100),
        make_features(), [Target("one", 200, 100, 60, 60)],
        override=True,
    )
    assert decision.state == "override"
    assert decision.assist_dx == 0
    assert decision.assist_dy == 0


def test_equal_targets_cause_abstention():
    controller = IntentController()
    targets = [
        Target("left", 200, 100, 60, 60),
        Target("right", 200, 100, 60, 60),
    ]
    decision = controller.decide(
        100, 100, FilterOutput(100, 100), make_features(), targets
    )
    assert decision.state == "target_ambiguous"
    assert decision.assist_dx == 0
    assert decision.assist_dy == 0
