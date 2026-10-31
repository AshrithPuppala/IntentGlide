"""Re-evaluate saved alpha regressors on the same held-out trial split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib

from ml.train import load_training_csv, grouped_split, metrics_for


def evaluate(dataset: Path, artifact_dir: Path):
    X, y, groups = load_training_csv(dataset)
    _, _, test_idx = grouped_split(X, y, groups)
    report = {}
    for name in ("random_forest", "xgboost"):
        path = artifact_dir / f"{name}_alpha.joblib"
        if not path.exists():
            continue
        artifact = joblib.load(path)
        model = artifact["model"] if isinstance(artifact, dict) else artifact
        report[name] = metrics_for(model, X[test_idx], y[test_idx])
    if not report:
        raise FileNotFoundError(f"No model artifacts found in {artifact_dir}")
    print(json.dumps({"test_trials": int(len(set(groups[test_idx]))), "results": report,
                       "caution": "Synthetic-only held-out metrics."}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/training_dataset.csv"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("ml/artifacts"))
    args = parser.parse_args()
    evaluate(args.dataset, args.artifact_dir)

if __name__ == "__main__":
    main()
