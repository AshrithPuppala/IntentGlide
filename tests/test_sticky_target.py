from interaction.sticky_target import StickyTargetController
from shared.contracts import Target


def test_brief_excursion_retains_then_releases_after_grace():
    lock = StickyTargetController(exit_margin_px=10, grace_seconds=.4)
    a = Target("a", 50, 50, 20, 20)
    lock.acquire("a")
    assert lock.retain(x=75, y=50, now=1.0, targets=[a]) is a
    assert lock.retain(x=75, y=50, now=1.5, targets=[a]) is None


def test_deliberate_outward_motion_releases_and_missing_target_resets():
    lock = StickyTargetController(exit_margin_px=30, deliberate_speed_px_s=100)
    a = Target("a", 50, 50, 20, 20)
    lock.acquire("a")
    assert lock.retain(x=62, y=50, now=1, targets=[a], velocity_x=150) is None
    lock.acquire("a")
    assert lock.retain(x=50, y=50, now=2, targets=[]) is None
    assert not lock.locked


def test_switch_requires_sustained_challenger_margin():
    lock = StickyTargetController(switch_confirmation_seconds=.25)
    a, b = Target("a", 0, 0, 20, 20), Target("b", 100, 0, 20, 20)
    lock.acquire("a")
    assert lock.retain(x=0,y=0,now=1,targets=[a,b],challenger_id="b",challenger_margin=.3) is a
    assert lock.retain(x=0,y=0,now=1.3,targets=[a,b],challenger_id="b",challenger_margin=.3) is None
    assert lock.target_id == "b"
