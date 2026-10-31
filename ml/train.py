"""Train Random Forest and XGBoost alpha regressors on grouped trial splits."""
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

TARGET = "alpha_optimal"


def load_training_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    missing = set(FEATURE_NAMES + [TARGET, "trial_id"]) - set(rows[0])
    if missing:
        raise ValueError(f"Training CSV missing required columns: {sorted(missing)}")
    X = np.array([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=float)
    y = np.array([float(row[TARGET]) for row in rows], dtype=float)
    groups = np.array([str(row["trial_id"]) for row in rows])
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Features/labels contain NaN or infinite values")
    if np.any((y < 0.12) | (y > 0.85)):
        raise ValueError("alpha_optimal outside expected range [0.12, 0.85]")
    return X, y, groups


def grouped_split(X, y, groups, seed=42):
    split1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    trainval_idx, test_idx = next(split1.split(X, y, groups))
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    train_rel, val_rel = next(split2.split(X[trainval_idx], y[trainval_idx], groups[trainval_idx]))
    return trainval_idx[train_rel], trainval_idx[val_rel], test_idx


def metrics_for(model, X, y):
    pred = np.clip(model.predict(X), 0.12, 0.85)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else 0.0,
    }


def train_all(dataset_path: Path, output_dir: Path):
    X, y, groups = load_training_csv(dataset_path)
    train_idx, val_idx, test_idx = grouped_split(X, y, groups)
    if len(np.unique(groups[train_idx])) < 2 or len(np.unique(groups[test_idx])) < 1:
        raise ValueError("Need multiple independent trial_id groups for grouped validation")
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=250, max_depth=16, min_samples_leaf=3,
            max_features=1.0, random_state=42, n_jobs=-1,
        )
    }
    xgb_available = True
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.90,
            objective="reg:squarederror", reg_lambda=1.0,
            random_state=42, n_jobs=4, tree_method="hist",
        )
    except ImportError:
        xgb_available = False

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        artifact = {
            "model": model,
            "model_name": name,
            "feature_names": FEATURE_NAMES,
            "target": TARGET,
            "alpha_bounds": [0.12, 0.85],
            "status": "synthetic_development_model_not_real_world_validated",
        }
        joblib.dump(artifact, output_dir / f"{name}_alpha.joblib")
        results[name] = {
            "validation": metrics_for(model, X_val, y_val),
            "test": metrics_for(model, X_test, y_test),
        }

    report = {
        "dataset": str(dataset_path),
        "rows": int(len(y)),
        "unique_trials": int(len(np.unique(groups))),
        "split_rows": {"train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))},
        "split_trials": {
            "train": int(len(np.unique(groups[train_idx]))),
            "validation": int(len(np.unique(groups[val_idx]))),
            "test": int(len(np.unique(groups[test_idx]))),
        },
        "features": FEATURE_NAMES,
        "target": TARGET,
        "models": results,
        "xgboost_available": xgb_available,
        "warning": "Synthetic-only development metrics. Do not claim real-user or clinical performance.",
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/training_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts"))
    args = parser.parse_args()
    train_all(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
