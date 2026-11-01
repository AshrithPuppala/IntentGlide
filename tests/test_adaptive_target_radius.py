from interaction.target_analyzer import target_feature_dict
from personalization.adaptation_engine import AdaptationEngine
from personalization.user_profile import UserProfile
from shared.contracts import Target


def test_radius_increases_with_calibrated_variability_and_overshoot_and_is_bounded():
    profile = UserProfile(acquisition_base_margin_px=30,
                          motion_variability_rms=10, median_overshoot_px=20)
    engine = AdaptationEngine(profile)
    assert engine.acquisition_margin() == 50
    profile.motion_variability_rms = 100
    profile.median_overshoot_px = 300
    assert engine.acquisition_margin() == profile.acquisition_radius_max_px


def test_profile_features_are_exposed_to_target_ranker():
    features = target_feature_dict([(0, 20, 0), (.01, 25, 0)], 25, 0,
        Target("t", 50, 0, 20, 20), 30,
        {"motion_variability_rms": 10, "median_overshoot_px": 20,
         "median_movement_speed": 300})
    assert features["profile_variability_norm"] > 0
    assert features["profile_overshoot_norm"] > 0
    assert features["inside_virtual_target"] == 1
