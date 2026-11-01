from shared.contracts import InputSample
from app.orchestrator import IntentGlideOrchestrator


def test_orchestrator_exposes_ml_butterworth_mode_even_without_model():
    pipeline = IntentGlideOrchestrator()
    pipeline.set_mode("ml_butterworth")
    result = pipeline.process(InputSample(0.0, 100.0, 120.0, "synthetic"))
    assert pipeline.mode == "ml_butterworth"
    assert (result["output"].x, result["output"].y) == (result["ml_butterworth"].x,
                                                          result["ml_butterworth"].y)
    assert result["filter_source"] in {"warmup", "heuristic", "heuristic_fallback", "ml"}
    assert result["cutoff_hz"] is not None


def test_orchestrator_preserves_existing_modes():
    pipeline = IntentGlideOrchestrator()
    for mode in ("raw", "fixed", "adaptive", "full", "ml_butterworth"):
        pipeline.set_mode(mode)
        assert pipeline.mode == mode
