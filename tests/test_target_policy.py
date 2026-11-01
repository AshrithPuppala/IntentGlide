import numpy as np
from interaction.target_analyzer import TARGET_FEATURE_NAMES, target_feature_dict
from ml.target_policy import TargetIntentPolicy
from shared.contracts import Target
from sandbox.desktop import Desktop


def test_target_features_include_velocity_and_history():
    history=[(i/120,100+i*2,100+i) for i in range(20)]
    f=target_feature_dict(history,138,119,Target("t",220,160,60,60))
    assert set(f)==set(TARGET_FEATURE_NAMES)
    assert f["speed_norm"]>0
    assert f["history_alignment"]>0.5


def test_policy_uses_heuristic_when_no_model():
    policy=TargetIntentPolicy(model_path="missing-intent-model.joblib")
    features={name:0.0 for name in TARGET_FEATURE_NAMES}
    features["distance_over_target_diagonal"]=0.1
    assert 0 <= policy.score_features(features) <= 1
    assert policy.get_status()["source"]=="heuristic_fallback"


def test_desktop_dwell_activation_uses_visible_semantic_target():
    desktop=Desktop()
    target=next(t for t in desktop.targets if t.id=="icon:calculator")
    assert desktop.target_at(target.center_x,target.center_y)=="icon:calculator"
    assert desktop.activate_target("icon:calculator",source="dwell") is True
    assert [w.id for w in desktop.windows]==["calculator"]


def test_dwell_can_close_one_window_but_not_bulk_close_actions():
    desktop=Desktop()
    desktop.open_window("calculator")
    assert desktop.activate_target("calculator:close",source="dwell") is True
    assert desktop.windows == []
    desktop.open_window("calculator")
    assert desktop.activate_target("calculator:close",source="physical") is True
    assert desktop.windows == []
    desktop.toggle_start()
    assert desktop.activate_target("start:close_all",source="dwell") is False
    assert desktop.activate_target("taskbar:desktop",source="dwell") is False
