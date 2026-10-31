from app.orchestrator import IntentGlideOrchestrator
from input.synthetic_generator import SyntheticInput
from sandbox.targets import make_targets


def test_mock_pipeline_runs_without_hardware():
    pipeline = IntentGlideOrchestrator()
    pipeline.mode = "full"
    targets = make_targets("standard", 900, 600)

    result = None
    generator = SyntheticInput()

    for _ in range(20):
        sample = generator.get_sample()
        result = pipeline.process(sample, targets=targets)

    assert result is not None
    assert set(result) >= {
        "raw", "fixed", "adaptive", "output", "features",
        "decision", "processing_time_ms",
    }
    assert result["processing_time_ms"] >= 0
def test_override_returns_raw_in_every_mode():
    targets = make_targets("standard", 900, 600)
    for mode in ("raw", "fixed", "adaptive", "full"):
        pipeline = IntentGlideOrchestrator()
        pipeline.set_mode(mode)
        for sample in [SyntheticInput().get_sample() for _ in range(5)]:
            result = pipeline.process(sample, targets=targets, override=True)
        assert result["output"].x == result["raw"].x
        assert result["decision"].state == "override"


def test_reset_and_filter_source_reported():
    pipeline = IntentGlideOrchestrator()
    result = pipeline.process(SyntheticInput().get_sample())
    expected_source = "ml" if pipeline.alpha_policy.source == "ml" else "heuristic"
    assert result["filter_source"] == expected_source
    pipeline.reset()
