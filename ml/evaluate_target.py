"""Evaluate target intent on untouched trial/session groups and users."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from interaction.target_analyzer import TARGET_FEATURE_NAMES, heuristic_target_score


def _load(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = set(TARGET_FEATURE_NAMES) | {"trial_id", "timestamp", "target_id", "intended_target_id", "is_intended"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"Dataset is empty or missing fields: {sorted(required-set(rows[0]) if rows else required)}")
    X = np.asarray([[float(row[name]) for name in TARGET_FEATURE_NAMES] for row in rows])
    y = np.asarray([int(row["is_intended"]) for row in rows])
    if not np.isfinite(X).all() or set(np.unique(y)) - {0, 1}:
        raise ValueError("Invalid feature values or labels")
    return rows, X, y


def _rank_metrics(rows, scores, threshold=0.65, margin=0.10):
    groups = {}
    for row, score in zip(rows, scores):
        groups.setdefault((row.get("session_id", ""), row["trial_id"], row["timestamp"]), []).append((float(score), row))
    correct, accepted, wrong = [], 0, 0
    for candidates in groups.values():
        candidates.sort(key=lambda item: item[0], reverse=True)
        lead = candidates[0][0] - (candidates[1][0] if len(candidates) > 1 else 0.0)
        winner = candidates[0][1]
        top_correct = int(winner["is_intended"]) == 1
        correct.append(top_correct)
        if candidates[0][0] >= threshold and lead >= margin:
            accepted += 1
            wrong += not top_correct
    return {"candidate_top1_accuracy": float(np.mean(correct)) if correct else 0.0,
            "accepted_coverage": accepted/len(groups) if groups else 0.0,
            "wrong_target_rate_among_accepted": wrong/max(accepted, 1),
            "groups": len(groups)}


def evaluate(dataset: Path, artifact_path: Path, report_path: Path):
    rows, X, y = _load(dataset)
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    if list(artifact.get("feature_names", [])) != list(TARGET_FEATURE_NAMES):
        raise ValueError("Artifact feature schema differs from current target-intent schema")
    group_column = "session_id" if "session_id" in rows[0] and len({r.get("session_id") for r in rows}) >= 3 else "trial_id"
    groups = np.asarray([str(row.get(group_column, row["trial_id"])) for row in rows])
    if len(set(groups)) < 3:
        raise ValueError("Need at least three independent groups for train/test evaluation")
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(X, y, groups))
    # Evaluate classifier only on held-out rows; task metrics are grouped per trial/time.
    probs = model.predict_proba(X[test])[:, list(model.classes_).index(1)]
    test_rows = [rows[i] for i in test]
    heuristic = [heuristic_target_score({k: float(row[k]) for k in TARGET_FEATURE_NAMES}) for row in test_rows]
    ml_metrics = _rank_metrics(test_rows, probs)
    baseline_metrics = _rank_metrics(test_rows, heuristic)
    users = sorted({str(row.get("profile_id", "unknown")) for row in rows})
    logo_results = {}
    if len(users) >= 3:
        y_profiles = np.asarray([str(row.get("profile_id", "unknown")) for row in rows])
        for train_idx, test_idx in LeaveOneGroupOut().split(X, y, y_profiles):
            if len(set(y[train_idx])) < 2:
                continue
            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X[train_idx], y[train_idx])
            fold_prob = fold_model.predict_proba(X[test_idx])[:, list(fold_model.classes_).index(1)]
            key = str(y_profiles[test_idx[0]])
            logo_results[key] = _rank_metrics([rows[i] for i in test_idx], fold_prob)
    metrics_passed = (ml_metrics["candidate_top1_accuracy"] >= baseline_metrics["candidate_top1_accuracy"]
              and ml_metrics["wrong_target_rate_among_accepted"] <= 0.20
              and ml_metrics["accepted_coverage"] > 0)
    data_sources = sorted({str(row.get("data_source", "unknown")) for row in rows})
    # Synthetic runs prove the evaluation code works, but cannot authorize a
    # runtime artifact. Require multiple genuine-user profiles and sessions.
    promotion_data_ready = ("real_user" in data_sources and len(users) >= 3
                            and len(set(groups)) >= 5)
    passed = metrics_passed and promotion_data_ready
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    report = {"schema_version": 1, "dataset": str(dataset),
        "data_sources": data_sources,
        "split_strategy": f"grouped holdout by {group_column}; random_state=42",
        "test_groups": len(set(groups[test])), "users": len(users),
        "candidate_model": str(artifact_path), "candidate_sha256": digest,
        "thresholds": {"score": .65, "score_margin": .10},
        "ranking_scores_are_calibrated_probabilities": False,
        "held_out": {"candidate_model": ml_metrics, "heuristic_baseline": baseline_metrics},
        "leave_one_user_out": logo_results,
        "evaluation_passed": bool(metrics_passed),
        "eligible_for_promotion": bool(passed),
        "promotion_block_reason": None if passed else ("Requires passing metrics plus at least three profiles, five independent sessions, and real_user-labelled observations." if metrics_passed else "Held-out target metrics did not meet promotion thresholds."),
        "warning": "Performance is only as representative as the labeled source data; synthetic results are not real-user validation."}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--model",type=Path,default=Path("ml/candidates/intent_target_model.candidate.joblib"))
    parser.add_argument("--report",type=Path,default=Path("ml/candidates/intent_target_evaluation.json"))
    args=parser.parse_args()
    report=evaluate(args.dataset,args.model,args.report)
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
