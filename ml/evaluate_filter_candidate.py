"""Evaluate an EMA-alpha or Butterworth-cutoff candidate on grouped holdout data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.features import FEATURE_NAMES
from ml.train import load_training_csv, grouped_split
from ml.train_cutoff import load_cutoff_dataset, split_by_group


def evaluate(candidate: Path, dataset: Path, report_path: Path, kind: str) -> dict:
    artifact = joblib.load(candidate)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Candidate must contain model and validation metadata")
    if not str(artifact.get("status", "")).startswith("candidate_"):
        raise ValueError("Only an unpromoted candidate can be evaluated")
    if list(artifact.get("feature_names", [])) != list(FEATURE_NAMES):
        raise ValueError("Candidate feature schema mismatch")
    if kind == "alpha":
        target, bounds = "alpha_optimal", [0.12, 0.85]
        if artifact.get("target") != target:
            raise ValueError("Candidate target is not alpha_optimal")
        X, y, groups = load_training_csv(dataset)
        _, _, test_idx = grouped_split(X, y, groups)
        metric_key, baseline = "mae", np.clip(.85-X[:, 0]*(.85-.12), .12, .85)
    else:
        target, bounds = "cutoff_optimal_hz", artifact.get("cutoff_bounds_hz", [1.5, 18.0])
        if artifact.get("target") != target:
            raise ValueError("Candidate target is not cutoff_optimal_hz")
        X, y, groups, _ = load_cutoff_dataset(dataset, float(bounds[0]), float(bounds[1]))
        _, _, test_idx = split_by_group(X, y, groups)
        metric_key, baseline = "mae_hz", np.full(len(y), float(np.median(y)))
    if len(set(groups[test_idx])) < 1:
        raise ValueError("No independent grouped test trials")
    pred = np.clip(artifact["model"].predict(X[test_idx]), float(bounds[0]), float(bounds[1]))
    candidate_mae = float(mean_absolute_error(y[test_idx], pred))
    baseline_mae = float(mean_absolute_error(y[test_idx], baseline[test_idx]))
    candidate_rmse = float(np.sqrt(mean_squared_error(y[test_idx], pred)))
    baseline_rmse = float(np.sqrt(mean_squared_error(y[test_idx], baseline[test_idx])))
    passed = candidate_mae <= baseline_mae and candidate_rmse <= baseline_rmse
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    with dataset.open("r", newline="", encoding="utf-8-sig") as handle:
        dataset_rows = list(csv.DictReader(handle))
    report = {"schema_version": 1, "kind": kind, "candidate": str(candidate),
        "candidate_sha256": digest, "dataset": str(dataset),
        "data_sources": sorted({str(row.get("data_source", "unknown")) for row in dataset_rows}),
        "grouped_test_groups": len(set(groups[test_idx])),
        "test_metrics": {"candidate_mae": candidate_mae, "baseline_mae": baseline_mae,
                         "candidate_rmse": candidate_rmse, "baseline_rmse": baseline_rmse,
                         "metric": metric_key},
        "evaluation_passed": bool(passed), "eligible_for_promotion": bool(passed),
        "warning": "Held-out regression metrics validate this dataset and objective only; they do not establish user benefit."}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kind", choices=("alpha", "cutoff"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.candidate, args.dataset, args.report, args.kind), indent=2))


if __name__ == "__main__":
    main()
