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
