import csv
import pytest

from personalization.calibration import CalibrationSession
from personalization.user_profile import UserProfile
from shared.contracts import Target


def test_calibration_preserves_instructed_ground_truth_and_collects_metrics(tmp_path):
    targets=[Target("wanted",50,50,20,20),Target("other",120,50,40,40)]
    c=CalibrationSession(UserProfile(),targets,trial_count=10,clock=lambda:0.0)
    instructed=c.current_target_id
    for i,(x,y) in enumerate(((0,50),(39,50),(58,50),(72,50))):
        t=.05*(i+1)
        c.record_sample(timestamp=t,raw_x=x+1,raw_y=y,filtered_x=x,filtered_y=y,
                        velocity_x=120,velocity_y=0)
    c.finish_trial("other",dwell_outcome="physical_click")
    record=c.trials[0]
    assert record.intended_target_id==instructed
    assert record.actual_selected_target_id=="other"
    assert record.metrics()["success"] is False
    assert record.metrics()["overshoot_px"]>0
    j=c.export_jsonl(tmp_path/"trials.jsonl")
    assert "intended_target_id" in j.read_text()
    dataset=c.export_intent_csv(tmp_path/"intent.csv")
    row=next(csv.DictReader(dataset.open(encoding="utf-8")))
    assert row["data_source"]=="calibration"
    assert row["is_intended"] in {"0","1"}


def test_incomplete_calibration_trial_rejected_safely():
    c=CalibrationSession(UserProfile(),[Target("a",1,1,20,20)],trial_count=10,clock=lambda:0)
    with pytest.raises(ValueError):
        c.finish_trial(None,dwell_outcome="miss")
    assert len(c.trials)==0
