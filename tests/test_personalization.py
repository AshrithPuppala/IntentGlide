import json
import pytest

from personalization.adaptation_engine import AdaptationEngine
from personalization.profile_manager import ProfileManager
from personalization.user_profile import UserProfile


def test_profile_roundtrip_reset_and_export(tmp_path):
    manager=ProfileManager(tmp_path/"profiles")
    profile=manager.load()
    profile.dwell_seconds=5.5
    manager.save(profile)
    assert manager.load().dwell_seconds==5.5
    exported=manager.export(tmp_path/"export.json")
    assert json.loads(exported.read_text())["profile_id"]==profile.profile_id
    reset=manager.reset()
    assert reset.dwell_seconds==3


def test_invalid_profile_values_rejected_and_corrupt_file_recovers(tmp_path):
    with pytest.raises(ValueError):
        UserProfile(dwell_seconds=99)
    manager=ProfileManager(tmp_path)
    manager.path.write_text("{bad",encoding="utf-8")
    assert manager.load().dwell_seconds==3


def test_adaptation_is_gradual_and_cancel_is_not_negative_feedback():
    profile=UserProfile()
    engine=AdaptationEngine(profile,learning_rate=.1)
    before=profile.magnetic_strength_scale
    engine.observe_event("user_cancelled")
    assert profile.magnetic_strength_scale==before
    result=engine.apply_feedback("magnetic_too_strong",explicit=True)
    assert result.changed and before-profile.magnetic_strength_scale < .02
    assert profile.dwell_seconds==3


def test_calibration_uses_bounded_ewma_not_instant_replacement():
    profile=UserProfile()
    engine=AdaptationEngine(profile,learning_rate=.1)
    initial_dwell_tolerance=profile.dwell_tolerance_px
    engine.apply_calibration({"trial_count":12,"motion_variability_rms":20,
        "median_overshoot_px":40,"median_acquisition_time_s":2.5,
        "median_movement_speed_px_s":400,"correction_frequency":5,"success_rate":.6})
    assert 0 < profile.motion_variability_rms < 20
    assert profile.acquisition_radius_min_px <= profile.acquisition_margin_px <= profile.acquisition_radius_max_px
    assert initial_dwell_tolerance < profile.dwell_tolerance_px < 60
    assert profile.calibration_trial_count==12
