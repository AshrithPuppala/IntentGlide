from app.orchestrator import IntentGlideOrchestrator
from personalization.user_profile import UserProfile
from shared.contracts import InputSample, Target


def test_profile_applies_to_intent_and_magnetic_layers_in_all_filter_modes():
    pipeline=IntentGlideOrchestrator()
    profile=UserProfile(acquisition_base_margin_px=35,
        maximum_assistance_distance_px=90,maximum_magnetic_speed_px_s=120,
        sticky_grace_ms=500)
    pipeline.apply_user_profile(profile)
    assert pipeline.intent.acquisition_base_px==35
    assert pipeline.magnetic.user_radius_px==90
    assert pipeline.magnetic.user_speed_px_s==120
    target=[Target("a",200,200,60,60)]
    for mode in pipeline.MODES:
        pipeline.set_mode(mode)
        sample=InputSample(timestamp=1.0,x=100,y=100,source="synthetic")
        result=pipeline.process(sample,targets=target,assist_enabled=True)
        assert result["decision"] is not None
        assert result["personalization_source"]==profile.adaptation_source
        assert "target_lock" in result


def test_manual_override_returns_raw_and_disables_magnetic_assistance():
    pipeline=IntentGlideOrchestrator()
    sample=InputSample(timestamp=1.0,x=42,y=71,source="synthetic")
    result=pipeline.process(sample,targets=[Target("a",100,100,50,50)],override=True)
    assert result["output"].x==42 and result["output"].y==71
    assert result["decision"].state=="override"
