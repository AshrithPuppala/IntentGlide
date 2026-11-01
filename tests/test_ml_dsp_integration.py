from pathlib import Path

import joblib
import numpy as np

from app.orchestrator import IntentGlideOrchestrator
from ml.features import FEATURE_NAMES, extract_features
from ml.filter_policy import FilterPolicy
from shared.contracts import InputSample, MotionFeatures, to_vector


class ConstantAlphaModel:
    n_features_in_ = len(FEATURE_NAMES)

    def predict(self, values):
        assert values.shape == (1, len(FEATURE_NAMES))
        return np.array([0.5])


def test_training_extractor_matches_live_motion_analyzer():
    sample_rate = 120.0
    t = np.arange(128) / sample_rate
    xy = np.column_stack((100 + 20 * t + 3 * np.sin(2 * np.pi * 8 * t),
                          150 + 10 * t + 2 * np.cos(2 * np.pi * 8 * t)))

    result = extract_features(xy, sample_rate)
    pipeline = IntentGlideOrchestrator()
    live = None
    for i, (x, y) in enumerate(xy):
        live = pipeline.motion.update(InputSample(i / sample_rate, x, y, "synthetic"))

    assert list(result) == FEATURE_NAMES
    np.testing.assert_allclose(list(result.values()), to_vector(live), rtol=1e-10, atol=1e-10)


def test_filter_policy_uses_live_motion_features_and_rejects_old_schema(tmp_path: Path):
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": ConstantAlphaModel(), "feature_names": FEATURE_NAMES,
                 "status": "evaluated_candidate", "target": "alpha_optimal"}, model_path)
    policy = FilterPolicy(model_path)
    features = MotionFeatures(0.2, 8.0, 0.8, 0.7, 120.0)
    assert policy.predict_alpha(features) == 0.5
    assert policy.get_status()["source"] == "ml"

    joblib.dump({"model": ConstantAlphaModel(), "feature_names": [*FEATURE_NAMES, "old_feature"]}, model_path)
    fallback = FilterPolicy(model_path)
    alpha = fallback.predict_alpha(features)
    assert 0.12 <= alpha <= 0.85
    assert fallback.get_status()["source"] == "heuristic_fallback"


def test_orchestrator_has_policy_and_runs_without_model():
    pipeline = IntentGlideOrchestrator()
    result = pipeline.process(InputSample(0.0, 10.0, 20.0, "synthetic"))
    assert pipeline.alpha_policy is not None
    assert result["model_source"] in {"ml", "heuristic_fallback"}
    expected_source = "ml" if pipeline.alpha_policy.source == "ml" else "heuristic"
    assert result["filter_source"] == expected_source
