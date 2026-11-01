from interaction.target_geometry import (actual_rect, expanded_rect, inside_actual,
    inside_virtual, isolated_small_icon_margin, point_rect_distance, regions_overlap)
from shared.contracts import Target


def test_virtual_target_expands_without_mutating_actual_geometry():
    target = Target("tiny", 100, 80, 20, 10)
    before = actual_rect(target)
    virtual = expanded_rect(target, 30)
    assert (virtual.width, virtual.height) == (80, 70)
    assert actual_rect(target) == before
    assert inside_actual(target, 100, 80)
    assert not inside_actual(target, 125, 80)
    assert inside_virtual(target, 125, 80, 30)


def test_point_distance_and_overlapping_regions():
    a, b = Target("a", 0, 0, 20, 20), Target("b", 45, 0, 20, 20)
    assert point_rect_distance(20, 0, actual_rect(a)) == 10
    assert not regions_overlap(a, 5, b, 5)
    assert regions_overlap(a, 13, b, 13)


def test_extra_dwell_margin_only_for_isolated_small_close_icon():
    close = Target("window:close", 50, 50, 32, 24)
    far = Target("other", 130, 50, 30, 30)
    near = Target("near", 100, 50, 30, 30)
    assert isolated_small_icon_margin(close, [close, far]) == 16
    assert isolated_small_icon_margin(close, [close, near]) == 0
    ordinary = Target("toolbar:button", 50, 50, 32, 24)
    assert isolated_small_icon_margin(ordinary, [ordinary, far]) == 0
