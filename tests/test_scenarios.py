# tests/test_scenarios.py
from shared.contracts import Target
from sandbox.tasks import TrialManager
from input.synthetic_generator import SyntheticReach

TARGETS = [
    Target("btn_1", 500.0, 500.0, 50.0, 50.0),
    Target("btn_2", 900.0, 200.0, 50.0, 50.0),
]

def run_trial(forced_id=None, click_offset=(0.0, 0.0)):
    t = [0.0]
    clock = lambda: t[0]                      # shared fake clock
    trials = TrialManager(TARGETS, timeout=10.0, clock=clock)
    trials.start_trial(forced_id)
    gen = SyntheticReach((100, 100), trials.intended_target, clock=clock)

    while not trials.is_complete:
        t[0] += 0.008
        s = gen.get_sample()
        if gen.arrived:
            trials.report_click(s.x + click_offset[0], s.y + click_offset[1])
        trials.update()
    return trials

def test_success():
    r = run_trial("btn_1").result
    assert r.outcome == "success" and r.success
    assert r.elapsed > 0

def test_wrong_target():
    # click lands on btn_2 instead of the intended btn_1
    r = run_trial("btn_1", click_offset=(400, -300)).result
    assert r.outcome == "wrong_target" and r.clicked_target_id == "btn_2"

def test_timeout():
    t = [0.0]
    trials = TrialManager(TARGETS, timeout=1.0, clock=lambda: t[0])
    trials.start_trial("btn_1")
    t[0] = 1.5
    assert trials.update().outcome == "timeout"

def test_labels_align_and_ambiguity():
    t = [0.0]
    clock = lambda: t[0]
    tgt = Target("btn_1", 500.0, 500.0, 50.0, 50.0)
    rival = Target("btn_2", 520.0, 480.0, 50.0, 50.0)   # nearly the same bearing
    far = Target("btn_3", 100.0, 900.0, 50.0, 50.0)
    gen = SyntheticReach((100, 100), tgt, trial_id=1,
                         candidates=[tgt, rival, far], clock=clock)
    pairs = []
    while not gen.arrived:
        t[0] += 0.008
        s = gen.get_sample()
        pairs.append((s, gen.get_label()))

    for s, l in pairs:
        assert abs(l.timestamp - s.timestamp) < 1e-9
        assert l.intended_target_id == "btn_1"
    assert pairs[0][1].phase == "onset"
    assert pairs[-1][1].phase == "dwell"
    assert pairs[5][1].ambiguous and pairs[5][1].rival_target_id == "btn_2"
    assert pairs[-1][1].dist_to_target < 1.0

def test_miss_then_success():
    t = [0.0]
    trials = TrialManager(TARGETS, timeout=10.0, clock=lambda: t[0])
    trials.start_trial("btn_1")
    t[0] = 1.0
    assert trials.report_click(0.0, 0.0) is None      # empty space, trial continues
    assert not trials.is_complete
    t[0] = 2.0
    r = trials.report_click(500.0, 500.0)             # now hit btn_1
    assert r.outcome == "success" and r.miss_count == 1

def test_miss_ends_trial_when_configured():
    t = [0.0]
    trials = TrialManager(TARGETS, end_on_miss=True, clock=lambda: t[0])
    trials.start_trial("btn_1")
    r = trials.report_click(0.0, 0.0)
    assert r.outcome == "miss" and not r.success