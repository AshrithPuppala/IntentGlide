"""Train Random Forest (and optional XGBoost) to predict Butterworth cutoff Hz."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from ml.features import FEATURE_NAMES

TARGET = "cutoff_optimal_hz"


def load_cutoff_dataset(path: Path, min_cutoff: float, max_cutoff: float, group_column: str = "profile_id"):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    missing = set(list(FEATURE_NAMES) + [TARGET, "trial_id"]) - set(rows[0])
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    X = np.asarray([[float(r[name]) for name in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.asarray([float(r[TARGET]) for r in rows], dtype=float)
    actual_group = group_column if group_column in rows[0] else "trial_id"
    groups = np.asarray([str(r.get(actual_group, r["trial_id"])) for r in rows])
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Features/targets contain NaN or infinity")
    if np.any((y < min_cutoff - 1e-6) | (y > max_cutoff + 1e-6)):
        raise ValueError("Cutoff labels outside configured bounds")
    return X, y, groups, actual_group


def split_by_group(X, y, groups, seed=42):
    first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    trainval, test = next(first.split(X, y, groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    train_rel, val_rel = next(second.split(X[trainval], y[trainval], groups[trainval]))
    return trainval[train_rel], trainval[val_rel], test


def model_metrics(model, X, y, min_cutoff, max_cutoff):
    pred = np.clip(model.predict(X), min_cutoff, max_cutoff)
    return {
        "mae_hz": float(mean_absolute_error(y, pred)),
        "rmse_hz": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else 0.0,
    }


def train(dataset: Path, output_dir: Path, min_cutoff: float = 1.5,
          max_cutoff: float = 18.0, group_column: str = "profile_id"):
    X, y, groups, actual_group = load_cutoff_dataset(dataset, min_cutoff, max_cutoff, group_column)
    if len(np.unique(groups)) < 4:
        raise ValueError(f"Need at least four independent groups for train/validation/test; found {len(np.unique(groups))}")
    train_idx, val_idx, test_idx = split_by_group(X, y, groups)
    if len(np.unique(groups[test_idx])) < 1:
        raise ValueError("Need independent trial groups for held-out evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=18, min_samples_leaf=3,
            max_features=1.0, random_state=42, n_jobs=-1,
        )
    }
    xgb_available = True
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=350, max_depth=5, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.90,
            objective="reg:squarederror", reg_lambda=1.0,
            random_state=42, n_jobs=4, tree_method="hist",
        )
    except ImportError:
        xgb_available = False

    results = {}
    for name, model in models.items():
        model.fit(X[train_idx], y[train_idx])
        artifact = {
            "model": model,
            "schema_version": 1,
            "model_version": f"filter-cutoff-{name}-1",
            "model_name": name,
            "feature_names": list(FEATURE_NAMES),
            "target": TARGET,
            "cutoff_bounds_hz": [float(min_cutoff), float(max_cutoff)],
            "filter": "second_order_butterworth_lowpass",
            "status": "candidate_requires_evaluation_before_promotion",
        }
        joblib.dump(artifact, output_dir / f"{name}_cutoff.joblib")
        results[name] = {
            "validation": model_metrics(model, X[val_idx], y[val_idx], min_cutoff, max_cutoff),
            "test": model_metrics(model, X[test_idx], y[test_idx], min_cutoff, max_cutoff),
        }

    report = {
        "dataset": str(dataset),
        "rows": int(len(y)),
        "unique_split_groups": int(len(np.unique(groups))),
        "split_group_column": actual_group,
        "split_groups": {
            "train": int(len(np.unique(groups[train_idx]))),
            "validation": int(len(np.unique(groups[val_idx]))),
            "test": int(len(np.unique(groups[test_idx]))),
        },
        "features": list(FEATURE_NAMES),
        "target": TARGET,
        "cutoff_bounds_hz": [float(min_cutoff), float(max_cutoff)],
        "label_distribution_hz": {
            "min": float(y.min()), "median": float(np.median(y)),
            "mean": float(y.mean()), "max": float(y.max()),
        },
        "models": results,
        "xgboost_available": xgb_available,
        "warning": "Synthetic-only development metrics; not real-user or clinical performance.",
    }
    report_path = output_dir / "cutoff_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/butterworth_training_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/candidates/butterworth"))
    parser.add_argument("--min-cutoff", type=float, default=1.5)
    parser.add_argument("--max-cutoff", type=float, default=18.0)
    parser.add_argument("--group-column", type=str, default="profile_id",
                        help="Group split by this column (default profile_id; falls back to trial_id if absent)")
    args = parser.parse_args()
    train(args.dataset, args.output_dir, args.min_cutoff, args.max_cutoff, args.group_column)


if __name__ == "__main__":
    main()
