from interaction.dwell_controller import DwellController
from shared.contracts import Target


def target():
    return Target("calculator", 100, 100, 80, 60)


def update(controller, now, x=100, y=100, **kwargs):
    args = dict(target=target(), confidence=.9, cursor_x=x, cursor_y=y,
                filtered_speed_px_s=10, enabled=True, override=False,
                cancel_requested=False)
    args.update(kwargs)
    return controller.update(now=now, **args)


def tick(controller, start, end, x=100, y=100, **kwargs):
    result = None
    steps = int(round((end-start)/.05))
    for i in range(1, steps+1):
        result = update(controller, start+i*.05, x=x, y=y, **kwargs)
    return result


def test_default_three_second_dwell_activates_once():
    c = DwellController()
    assert c.dwell_seconds == 3
    update(c, 0)
    result = tick(c, 0, 3)
    assert result.state == "activated" and result.activated
    assert not update(c, 4.1).activated


def test_brief_excursion_keeps_timer_running_without_paused_state():
    c = DwellController(dwell_seconds=3, tolerance_px=32)
    update(c, 0)
    tick(c, 0, 1)
    result = tick(c, 1, 2, x=158)  # 18 px beyond actual right edge
    assert result.state == "dwelling"
    assert result.progress >= .49
    assert result.state != "paused"


def test_exit_grace_keeps_elapsed_time_but_long_exit_resets():
    c = DwellController(dwell_seconds=3, tolerance_px=20, exit_grace_seconds=.75)
    update(c, 0)
    tick(c, 0, 1)
    result = update(c, 1.2, x=180, target=None)
    assert result.progress >= .29
    assert result.state == "dwelling"
    result = update(c, 2.0, x=180, target=None)
    assert result.state == "idle" and result.progress == 0


def test_completed_dwell_outside_actual_is_ready_then_activates_on_return():
    c = DwellController(dwell_seconds=1, tolerance_px=32)
    update(c, 0, x=158)
    result = tick(c, 0, 1, x=158)
    assert result.state == "ready" and result.progress == 1 and not result.activated
    result = update(c, 1.05, x=139)
    assert result.state == "activated" and result.activated


def test_space_right_click_and_override_cancel():
    for kwargs in ({"cancel_requested": True}, {"override": True}):
        c = DwellController()
        update(c, 0)
        result = update(c, .1, **kwargs)
        assert result.state == "cancelled"


def test_missing_target_expires_grace_and_activation_is_one_shot():
    c = DwellController(dwell_seconds=.2)
    update(c, 0)
    result = tick(c, 0, .2)
    assert result.activated
    assert not update(c, .25).activated
    update(c, 1, x=180)
    result = update(c, 1.1, target=None, x=180)
    assert result.state == "idle"


def test_fast_deliberate_outward_motion_resets_attempt():
    c = DwellController(tolerance_px=32, max_stable_speed_px_s=65)
    update(c, 0, x=135)
    result = update(c, .05, x=148, filtered_speed_px_s=200)
    assert result.state == "idle"
    assert result.reason == "deliberate movement away"


def test_virtual_activation_can_be_enabled_only_for_safe_enlarged_controls():
    c = DwellController(dwell_seconds=.2, tolerance_px=16)
    update(c, 0, x=141, allow_virtual_activation=True)
    result = update(c, .2, x=141, allow_virtual_activation=True)
    assert result.state == "activated" and result.activated

    guarded = DwellController(dwell_seconds=.2, tolerance_px=16)
    update(guarded, 0, x=141, allow_virtual_activation=False)
    result = update(guarded, .2, x=141, allow_virtual_activation=False)
    assert result.state == "ready" and not result.activated
