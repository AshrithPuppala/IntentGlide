"""Evaluate logged personalization conditions without inventing outcomes.

Input is a CSV containing one row per completed instructed task. Required columns:
profile_id, session_id, trial_id, condition, intended_target_id,
actual_selected_target_id, completion_time_s, wrong_target, overshoot_px,
corrections, accidental_dwell, dwell_completion_time_s,
magnetic_displacement_px, latency_ms.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

CONDITIONS = ("default", "calibrated", "statistical_adaptation", "future_ml")
NUMERIC = ("completion_time_s", "overshoot_px", "corrections",
           "dwell_completion_time_s", "magnetic_displacement_px", "latency_ms")


def _mean(rows, key):
    vals=[float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return statistics.mean(vals) if vals else None


def evaluate_rows(rows):
    if not rows:
        raise ValueError("No real or synthetic logged trials supplied")
    grouped={condition:[r for r in rows if r["condition"]==condition] for condition in CONDITIONS}
    report={"schema_version":1,"input_rows":len(rows),"data_sources":sorted({r.get("data_source","unknown") for r in rows}),
            "metrics_by_condition":{},"within_user_chronological":{},"leave_one_user_out":{},
            "future_ml_status":"not trained; requires sufficient consented real-user sessions"}
    for condition, subset in grouped.items():
        if not subset:
            report["metrics_by_condition"][condition]={"trials":0,"status":"no data"}
            continue
        successes=[r["actual_selected_target_id"]==r["intended_target_id"] for r in subset]
        report["metrics_by_condition"][condition]={"trials":len(subset),
            "target_acquisition_success_rate":statistics.mean(successes),
            "wrong_target_activation_rate":_mean(subset,"wrong_target"),
            "mean_task_completion_time_s":_mean(subset,"completion_time_s"),
            "mean_overshoot_px":_mean(subset,"overshoot_px"),
            "mean_corrective_movements":_mean(subset,"corrections"),
            "accidental_dwell_activation_rate":_mean(subset,"accidental_dwell"),
            "mean_dwell_completion_time_s":_mean(subset,"dwell_completion_time_s"),
            "mean_magnetic_displacement_px":_mean(subset,"magnetic_displacement_px"),
            "mean_end_to_end_latency_ms":_mean(subset,"latency_ms")}
    # Chronological first 70% vs final 30% within each user/session/condition.
    for condition, subset in grouped.items():
        for profile_id in sorted({r.get("profile_id","unknown") for r in subset}):
            user=sorted([r for r in subset if r.get("profile_id")==profile_id],
                        key=lambda r:(r.get("session_id",""),int(r["trial_id"])))
            cut=max(1,int(len(user)*.7)) if user else 0
            report["within_user_chronological"].setdefault(condition,{})[profile_id]={
                "early_trials":cut,"later_trials":len(user)-cut,
                "early_success_rate":statistics.mean(r["actual_selected_target_id"]==r["intended_target_id"] for r in user[:cut]) if cut else None,
                "later_success_rate":statistics.mean(r["actual_selected_target_id"]==r["intended_target_id"] for r in user[cut:]) if len(user)>cut else None}
    # User holdouts are reported only when at least three pseudonymous users exist.
    profiles=sorted({r.get("profile_id","unknown") for r in rows})
    if len(profiles)>=3:
        for heldout in profiles:
            user=[r for r in rows if r.get("profile_id")==heldout]
            report["leave_one_user_out"][heldout]={"held_out_trials":len(user),
                "observed_success_rate":statistics.mean(r["actual_selected_target_id"]==r["intended_target_id"] for r in user)}
    report["interpretation_warning"]="This summarizes observed task records; it does not establish causality or clinical benefit. Synthetic data must remain labeled synthetic."
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("evaluation/results/personalization_report.json"))
    args=parser.parse_args()
    with args.dataset.open("r",newline="",encoding="utf-8-sig") as handle:
        rows=list(csv.DictReader(handle))
    required={"profile_id","session_id","trial_id","condition","intended_target_id","actual_selected_target_id",*NUMERIC,"wrong_target","accidental_dwell"}
    if rows and required-set(rows[0]):
        raise ValueError(f"Dataset missing columns: {sorted(required-set(rows[0]))}")
    report=evaluate_rows(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
