import numpy as np

from ml.features import FEATURE_NAMES, extract_features
from ml.labeling import select_optimal_alpha
from ml.synthetic_data import generate_trials
from ml.dataset import build_training_rows, align_input_and_labels
from ml.synthetic_data import _write_csv, INPUT_FIELDS, LABEL_FIELDS
from pathlib import Path


def test_feature_extractor_outputs_finite_named_features():
    t = np.arange(128) / 120.0
    xy = np.column_stack((100 + 20*t + 3*np.sin(2*np.pi*8*t),
                          150 + 10*t + 2*np.cos(2*np.pi*8*t)))
    result = extract_features(xy, 120)
    assert list(result) == FEATURE_NAMES
    assert all(np.isfinite(list(result.values())))


def test_optimal_alpha_is_bounded():
    t = np.arange(96) / 120.0
    truth = np.column_stack((t*100, t*50))
    obs = truth + np.column_stack((3*np.sin(2*np.pi*8*t), 2*np.cos(2*np.pi*8*t)))
    alpha, metrics = select_optimal_alpha(obs[32:], truth[32:], obs[31], sample_rate=120)
    assert 0.12 <= alpha <= 0.85
    assert metrics["label_cost"] >= 0


def test_synthetic_pipeline_aligns_and_builds_rows(tmp_path: Path):
    inputs, labels = generate_trials(n_trials=12, sample_rate=120, duration_s=3.5, seed=7)
    in_path, lab_path = tmp_path / "input.csv", tmp_path / "labels.csv"
    _write_csv(in_path, inputs, INPUT_FIELDS)
    _write_csv(lab_path, labels, LABEL_FIELDS)
    aligned, missing = align_input_and_labels(in_path, lab_path)
    rows, skipped = build_training_rows(aligned, sample_rate=120, window_size=128,
                                        horizon=64, stride=32)
    assert missing == 0
    assert len(rows) > 0
    assert 0.12 <= rows[0]["alpha_optimal"] <= 0.85
    assert all(name in rows[0] for name in FEATURE_NAMES)
    assert "truth_x" not in FEATURE_NAMES


def test_filter_policy_fallback_is_bounded():
    from ml.filter_policy import FilterPolicy
    policy = FilterPolicy(model_path=None)
    features = {name: 0.0 for name in FEATURE_NAMES}
    features["oscillation_score"] = 1.0
    alpha = policy.predict_alpha(features)
    assert 0.12 <= alpha <= 0.85
    assert alpha == 0.12
    assert policy.get_status()["source"] == "heuristic_fallback"
