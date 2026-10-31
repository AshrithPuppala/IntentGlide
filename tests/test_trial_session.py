import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from app.trial_session import KEY_PREFIX, TrialSession, key_label
from input.input_provider import InputManager
from sandbox.desktop import Desktop
from shared.contracts import InputSample


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class StaticSource:
    def get_sample(self):
        return InputSample(timestamp=0.0, x=0.0, y=0.0, source="mouse")


def make(auto=False, seed=1, **kw):
    clock = FakeClock()
    desktop = Desktop()
    inputs = InputManager(mouse=StaticSource())
    resets = []
    session = TrialSession(desktop, inputs, on_reset=lambda: resets.append(1), clock=clock, seed=seed, **kw)
    session.auto = auto
    return session, desktop, inputs, clock, resets


def goal_center(desktop, session):
    return next(t for t in desktop.keypad_targets() if t.id == session.goal_id)


def test_start_opens_calculator_and_highlights_a_key():
    s, d, *_ = make()
    assert s.start("full", (50, 50))
    assert [w.id for w in d.windows] == ["calculator"]
    assert s.active and s.goal_id == d.goal_id
    assert s.goal_id in {t.id for t in d.keypad_targets()}
    assert s.goal_id.startswith(KEY_PREFIX)


def test_clicking_the_goal_is_a_success_and_clears_the_highlight():
    s, d, _i, clock, _r = make()
    s.start("full", (50, 50))
    t = goal_center(d, s)
    clock.advance(1.5)
    r = s.click((t.center_x, t.center_y))
    assert r.outcome == "success" and abs(r.elapsed - 1.5) < 1e-9
    assert not s.active and d.goal_id is None
    assert s.stats("full")["success_rate"] == 1.0 and s.stats("full")["median_time"] == 1.5


def test_wrong_key_miss_and_timeout_are_distinguished():
    s, d, _i, clock, _r = make()
    s.start("raw", (50, 50))
    other = next(t for t in d.keypad_targets() if t.id != s.goal_id)
    assert s.click((other.center_x, other.center_y)).outcome == "wrong_target"
    s.start("raw", (50, 50))
    assert s.click((5, 5)).outcome == "miss"            # empty space: one click ends the trial
    s.start("raw", (50, 50))
    clock.advance(s.timeout + 0.1)
    assert s.update((5, 5), "raw").outcome == "timeout"
    st = s.stats("raw")
    assert (st["wrong_target"], st["miss"], st["timeout"], st["success"]) == (1, 1, 1, 0)
    assert st["median_time"] is None


def test_stats_are_kept_per_mode():
    s, d, _i, clock, _r = make()
    for mode, dt in (("raw", 1.0), ("raw", 3.0), ("full", 2.0)):
        s.start(mode, (50, 50))
        t = goal_center(d, s)
        clock.advance(dt)
        s.click((t.center_x, t.center_y))
    assert s.stats("raw")["n"] == 2 and s.stats("raw")["median_time"] == 2.0
    assert s.stats("full")["n"] == 1 and s.stats()["n"] == 3


def test_cancel_discards_the_trial():
    s, d, *_ = make()
    s.start("full", (50, 50))
    s.cancel()
    assert not s.active and d.goal_id is None and s.stats()["n"] == 0


def test_goal_never_repeats_back_to_back():
    s, *_ = make(seed=7)
    goals = []
    for _ in range(40):
        s.start("full", (50, 50))
        goals.append(s.goal_id)
    assert all(a != b for a, b in zip(goals, goals[1:]))


def test_auto_reach_drives_the_cursor_clicks_and_chains():
    s, d, inputs, clock, resets = make(auto=True, tremor_amp=0.0)
    s.start("full", (60, 300))
    assert inputs.active == "reach" and resets
    cursor, results = (60, 300), []
    for _ in range(400):                                 # 3.2 s at 125 Hz
        clock.advance(0.008)
        sample = inputs.get_sample() if inputs.active == "reach" else None
        if sample is not None:
            cursor = (sample.x, sample.y)                # no filtering in this test
        r = s.update(cursor, "full")
        if r:
            results.append(r)
    assert results and results[0].outcome == "success"
    assert len(results) >= 2                             # trials chained after the pause
    assert inputs.active in ("mouse", "reach")


def test_auto_click_presses_the_calculator_key():
    s, d, inputs, clock, _r = make(auto=True, tremor_amp=0.0)
    s.start("full", (60, 300))
    pressed = []
    d.windows[-1].calc.press = pressed.append             # record what the click actually pressed
    cursor = (60, 300)
    goal = s.goal_id
    for _ in range(300):
        clock.advance(0.008)
        if inputs.active == "reach":
            sample = inputs.get_sample()
            cursor = (sample.x, sample.y)
        if s.update(cursor, "full"):
            break
    assert pressed == [goal.removeprefix(KEY_PREFIX)]


def test_source_is_restored_when_a_reach_trial_ends_or_is_cancelled():
    s, d, inputs, clock, _r = make(auto=True)
    s.start("full", (60, 300))
    assert inputs.active == "reach"
    s.stop_auto()
    assert inputs.active == "mouse" and not s.auto


def test_key_labels_are_readable():
    assert key_label("calculator:key:plus") == "+"
    assert key_label("calculator:key:7") == "7"
    assert key_label(None) == "-"
