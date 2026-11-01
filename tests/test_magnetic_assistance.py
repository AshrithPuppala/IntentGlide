from interaction.magnetic_assistance import (
    apply_magnetic_assistance, distance_to_target_rect,
)
from shared.contracts import FilterOutput, Target


def test_assistance_uses_distance_to_edge_not_target_center():
    target = Target("settings", 100, 100, 80, 60)
    assert distance_to_target_rect(145, 100, target) == 5


def test_magnetic_pull_is_bounded_and_moves_toward_nearest_edge():
    target = Target("settings", 100, 100, 80, 60)
    out = apply_magnetic_assistance(
        FilterOutput(150, 100), target, dt=1 / 120,
        intent_confirmed=True, confidence=0.9,
    )
    assert out.x < 150
    assert out.y == 100
    assert 0 < 150 - out.x <= 130 / 120 + 1e-9


def test_no_pull_without_confirmed_intent_or_when_overridden():
    target = Target("settings", 100, 100, 80, 60)
    start = FilterOutput(150, 100)
    no_intent = apply_magnetic_assistance(
        start, target, dt=1/120, intent_confirmed=False, confidence=0.99
    )
    override = apply_magnetic_assistance(
        start, target, dt=1/120, intent_confirmed=True,
        confidence=0.99, manual_override=True,
    )
    assert (no_intent.x, no_intent.y) == (start.x, start.y)
    assert (override.x, override.y) == (start.x, start.y)


def test_no_pull_outside_acquisition_radius_or_inside_button():
    target = Target("settings", 100, 100, 80, 60)
    outside = apply_magnetic_assistance(
        FilterOutput(250, 100), target, dt=1/120,
        intent_confirmed=True, confidence=0.99,
    )
    inside = apply_magnetic_assistance(
        FilterOutput(100, 100), target, dt=1/120,
        intent_confirmed=True, confidence=0.99,
    )
    assert outside.x == 250
    assert inside.x == 100
