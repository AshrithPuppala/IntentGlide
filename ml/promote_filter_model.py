"""Promote an alpha/cutoff candidate only when its hash-bound evaluation passes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib

from ml.features import FEATURE_NAMES


def promote(candidate: Path, report_path: Path, artifact_dir: Path) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("eligible_for_promotion"):
        raise ValueError("Filter evaluation did not authorize promotion")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest != report.get("candidate_sha256"):
        raise ValueError("Candidate changed after evaluation; evaluate again")
    artifact = joblib.load(candidate)
    if list(artifact.get("feature_names", [])) != list(FEATURE_NAMES):
        raise ValueError("Candidate feature schema mismatch")
    expected = "alpha_optimal" if report["kind"] == "alpha" else "cutoff_optimal_hz"
    if artifact.get("target") != expected:
        raise ValueError("Candidate target does not match evaluation report")
    artifact["status"] = "evaluated_candidate"
    artifact["evaluation_report"] = str(report_path)
    artifact["evaluation_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    name = "random_forest_alpha.joblib" if report["kind"] == "alpha" else "random_forest_cutoff.joblib"
    dest = artifact_dir / name
    joblib.dump(artifact, dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("ml/artifacts"))
    args = parser.parse_args()
    print(promote(args.candidate, args.report, args.artifact_dir))


if __name__ == "__main__":
    main()
