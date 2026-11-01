import numpy as np
from interaction.target_analyzer import MovementHistory, rank_targets
from interaction.intent_controller import IntentController
from shared.contracts import FilterOutput, MotionFeatures, Target


def test_velocity_history_points_to_target_and_ranks_it_first():
    history=MovementHistory()
    target=Target("right",200,100,60,60)
    other=Target("left",20,100,60,60)
    for i in range(20):
        history.add(i/120,100+i*2,100)
    estimate=history.estimate(target)
    assert estimate.velocity_x > 100
    ranked=rank_targets(138,100,estimate.velocity_x,estimate.velocity_y,[other,target],history.points)
    assert ranked[0][0].id=="right"


def test_equal_targets_remain_ambiguous():
    controller=IntentController(persistence_seconds=0.0)
    features=MotionFeatures(0.2,None,0.1,0.8,100.0)
    targets=[Target("a",200,100,60,60),Target("b",200,100,60,60)]
    result=None
    for i in range(30):
        pos=FilterOutput(100+i,100)
        result=controller.decide(pos.x,pos.y,pos,features,targets,timestamp=i/120)
    assert result.state=="target_ambiguous"
    assert result.target_id is None


def test_intent_does_not_use_raw_minus_filtered_as_velocity():
    controller=IntentController(persistence_seconds=0.0)
    features=MotionFeatures(0.1,None,0.1,0.9,150.0)
    targets=[Target("right",300,100,60,60),Target("left",20,100,60,60)]
    result=None
    for i in range(40):
        # Filtered path moves right; raw position has an alternating tremor residual.
        fx=100+i*2
        raw=fx+(12 if i%2 else -12)
        result=controller.decide(raw,100,FilterOutput(fx,100),features,targets,timestamp=i/120)
    assert controller.speed_px_s > 100
    assert controller.policy.get_status()["source"] in {"ml","heuristic_fallback"}
