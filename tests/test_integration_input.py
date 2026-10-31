from shared.contracts import Target
from sandbox.tasks import TrialManager
from input.synthetic_generator import SyntheticReach
from input.input_provider import InputManager

TARGETS = [
    Target("btn_1", 500.0, 500.0, 50.0, 50.0),
    Target("btn_2", 900.0, 200.0, 50.0, 50.0),
]

def test_full_loop_with_manager():
    t = [0.0]
    clock = lambda: t[0]
    trials = TrialManager(TARGETS, timeout=10.0, clock=clock)
    trial = trials.start_trial("btn_1")
    gen = SyntheticReach((100, 100), trials.intended_target,
                         trial_id=trial.trial_id,
                         candidates=list(trials.targets.values()),
                         clock=clock)
    inputs = InputManager(reach=gen)
    inputs.set_source("reach")

    labels = []
    while not trials.is_complete:
        t[0] += 0.008
        s = inputs.get_sample()
        labels.append(gen.get_label())
        cx, cy = s.x, s.y            # stand-in for DSP -> intent -> assist
        if gen.arrived:
            trials.report_click(cx, cy)
        trials.update()

    r = trials.result
    assert r.outcome == "success"
    assert all(l.trial_id == trial.trial_id for l in labels)
    assert labels[-1].phase == "dwell"