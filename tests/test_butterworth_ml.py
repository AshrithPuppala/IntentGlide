from pathlib import Path
import joblib
import numpy as np

from ml.features import FEATURE_NAMES
from ml.butterworth_policy import ButterworthCutoffPolicy
from ml.butterworth_labeling import select_optimal_cutoff
from ml.synthetic_data_butterworth import generate_trials, INPUT_FIELDS, LABEL_FIELDS, _write
from ml.dataset import align_input_and_labels
from ml.butterworth_dataset import build_cutoff_training_rows
from shared.contracts import MotionFeatures


class ConstantCutoffModel:
    n_features_in_ = len(FEATURE_NAMES)
    def predict(self, values):
        assert values.shape == (1, len(FEATURE_NAMES))
        return np.asarray([4.25])


def test_cutoff_policy_loads_schema_and_predicts(tmp_path: Path):
    path = tmp_path / "cutoff.joblib"
    joblib.dump({
        "model": ConstantCutoffModel(),
        "feature_names": list(FEATURE_NAMES),
        "target": "cutoff_optimal_hz",
        "cutoff_bounds_hz": [1.5, 18.0],
        "status": "evaluated_candidate",
    }, path)
    policy = ButterworthCutoffPolicy(path)
    features = MotionFeatures(0.6, 8.0, 0.8, 0.4, 100.0)
    assert policy.predict_cutoff(features) == 4.25
    assert policy.get_status()["source"] == "ml"


def test_cutoff_policy_fallback_is_bounded():
    policy = ButterworthCutoffPolicy(None)
    features = MotionFeatures(1.0, 8.0, 0.9, 0.1, 100.0)
    value = policy.predict_cutoff(features)
    assert 1.5 <= value <= 18.0
    assert policy.get_status()["source"] == "heuristic_fallback"


def test_cutoff_labeler_returns_a_valid_candidate():
    fs = 120.0
    t = np.arange(192) / fs
    truth = np.column_stack((100 + 100 * t, 150 + 50 * t))
    observed = truth.copy()
    observed[:, 0] += 8 * np.sin(2 * np.pi * 8 * t)
    observed[:, 1] += 6 * np.cos(2 * np.pi * 8 * t)
    cutoff, metrics = select_optimal_cutoff(
        observed[128:], truth[128:], history=observed[:128],
        candidates=[2.0, 3.0, 6.0, 12.0], sample_rate=fs,
    )
    assert cutoff in [2.0, 3.0, 6.0, 12.0]
    assert metrics["label_cost"] >= 0


def test_architecture_c_synthetic_dataset_builds_rows(tmp_path: Path):
    inputs, labels = generate_trials(
        n_trials=12, sample_rate=120.0,
        min_duration_s=2.0, max_duration_s=2.2, seed=11,
    )
    input_path, label_path = tmp_path / "input.csv", tmp_path / "labels.csv"
    _write(input_path, inputs, INPUT_FIELDS)
    _write(label_path, labels, LABEL_FIELDS)
    aligned, unmatched = align_input_and_labels(input_path, label_path)
    rows, skipped = build_cutoff_training_rows(
        aligned, sample_rate=120, window_size=128, horizon=64, stride=32,
        candidates=[2.0, 4.0, 8.0],
    )
    assert unmatched == 0
    assert len(rows) > 0
    assert all(1.5 <= row["cutoff_optimal_hz"] <= 18.0 for row in rows)
    assert all(name in rows[0] for name in FEATURE_NAMES)
    assert "profile_id" in rows[0]
    assert "truth_x" not in FEATURE_NAMES
