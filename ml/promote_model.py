"""Promote a model candidate only when its matching evaluation report passes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import joblib

from interaction.target_analyzer import TARGET_FEATURE_NAMES


def promote(candidate: Path, report_path: Path, artifact_dir: Path) -> Path:
    report=json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("eligible_for_promotion"):
        raise ValueError("Evaluation report does not authorize promotion")
    digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest != report.get("candidate_sha256"):
        raise ValueError("Candidate changed after evaluation; re-evaluate before promotion")
    artifact=joblib.load(candidate)
    if list(artifact.get("feature_names", [])) != list(TARGET_FEATURE_NAMES):
        raise ValueError("Candidate feature schema mismatch")
    artifact["status"]="evaluated_candidate"
    artifact["evaluation_report"] = str(report_path)
    artifact_dir.mkdir(parents=True,exist_ok=True)
    dest=artifact_dir/"intent_target_model.joblib"
    joblib.dump(artifact,dest)
    return dest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",type=Path,default=Path("ml/candidates/intent_target_model.candidate.joblib"))
    parser.add_argument("--report",type=Path,default=Path("ml/candidates/intent_target_evaluation.json"))
    parser.add_argument("--artifact-dir",type=Path,default=Path("ml/artifacts"))
    args=parser.parse_args()
    print(promote(args.candidate,args.report,args.artifact_dir))

if __name__=="__main__": main()
