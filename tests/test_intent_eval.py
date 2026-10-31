from app.orchestrator import IntentGlideOrchestrator
from evaluation.intent_eval import evaluate, keypad_goals, make_desktop


def test_intent_eval_harness_runs_and_counts_trials():
    goal = next(iter(keypad_goals(make_desktop())))
    res = evaluate(IntentGlideOrchestrator, seeds=(0,), starts=1, goals=[goal])
    assert res.trials == 1
    assert sum(res.clicks["full"].values()) == 1
    assert res.phase["dwell"]["frames"] > 0
