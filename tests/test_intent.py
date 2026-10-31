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


# ---- heading-based (posterior) scoring ---------------------------------------------------
def _approach(controller, targets, start=(100.0, 100.0), end=(300.0, 100.0), frames=40):
    """Feed a steady left-to-right approach on the filtered path; return the last decision."""
    decision = None
    for i in range(frames):
        f = i / (frames - 1)
        x, y = start[0] + (end[0] - start[0]) * f, start[1] + (end[1] - start[1]) * f
        decision = controller.decide(x, y, FilterOutput(x, y), make_features(), targets,
                                     timestamp=1.0 + i / 120.0)
    return decision


def test_heading_at_one_clear_target_is_confident_and_pulls_toward_it():
    targets = [Target("goal", 330, 100, 60, 60), Target("far", 330, 400, 60, 60)]
    decision = _approach(IntentController(), targets)
    assert decision.state == "target_confident"
    assert decision.target_id == "goal"
    assert decision.assist_dx > 0 and abs(decision.assist_dy) < 1e-6
    assert abs(decision.assist_dx) <= IntentController().max_assist_px


def test_two_equally_plausible_targets_abstain():
    targets = [Target("a", 330, 100, 60, 60), Target("b", 330, 100, 60, 60)]
    decision = _approach(IntentController(), targets)
    assert decision.state == "target_ambiguous"
    assert (decision.assist_dx, decision.assist_dy) == (0.0, 0.0)


def test_moving_away_from_the_only_target_is_never_confident():
    targets = [Target("behind", 20, 100, 60, 60)]
    decision = _approach(IntentController(), targets)          # heading right, target on the left
    assert decision.state != "target_confident"


def test_reset_clears_heading_history():
    controller = IntentController()
    _approach(controller, [Target("goal", 330, 100, 60, 60)])
    assert controller._history
    controller.reset()
    assert not controller._history and controller.last_heading is None


def test_override_still_wins_with_timestamps():
    decision = IntentController().decide(
        100, 100, FilterOutput(100, 100), make_features(),
        [Target("one", 200, 100, 60, 60)], override=True, timestamp=1.0)
    assert decision.state == "override"
