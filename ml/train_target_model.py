"""Train a trial/session-grouped candidate target ranker (candidate only)."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from interaction.target_analyzer import TARGET_FEATURE_NAMES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("data/generated/intent_candidates.csv"))
    parser.add_argument("--output-dir",type=Path,default=Path("ml/candidates"))
    parser.add_argument("--group-column",choices=["session_id","trial_id","profile_id"],default="session_id")
    args=parser.parse_args()
    with args.dataset.open("r",newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    required=set(TARGET_FEATURE_NAMES+ ("is_intended","trial_id","timestamp","target_id"))
    missing=required-set(rows[0]) if rows else required
    if not rows or missing: raise ValueError(f"Dataset empty or missing columns: {sorted(missing)}")
    X=np.asarray([[float(r[n]) for n in TARGET_FEATURE_NAMES] for r in rows],dtype=float)
    y=np.asarray([int(r["is_intended"]) for r in rows],dtype=int)
    group_column=args.group_column
    if group_column not in rows[0] or len({str(r.get(group_column, "")) for r in rows}) < 3:
        group_column="trial_id"
    groups=np.asarray([str(r[group_column]) for r in rows])
    if not np.isfinite(X).all() or set(np.unique(y))!={0,1}: raise ValueError("Expected finite features and both target classes")
    outer=GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=42)
    trainval,test=next(outer.split(X,y,groups))
    inner=GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=43)
    tr_rel,val_rel=next(inner.split(X[trainval],y[trainval],groups[trainval]))
    train=trainval[tr_rel]; val=trainval[val_rel]
    model=RandomForestClassifier(n_estimators=300,max_depth=14,min_samples_leaf=3,
        class_weight="balanced_subsample",random_state=42,n_jobs=-1)
    model.fit(X[train],y[train])
    pred=model.predict(X[test]); proba=model.predict_proba(X[test])[:,list(model.classes_).index(1)]
    val_proba=model.predict_proba(X[val])[:,list(model.classes_).index(1)]
    # Evaluate the actual task: among candidates at each trial/time, rank the instructed target first.
    def grouped_top1(indices, scores):
        grouped={}
        for idx,p in zip(indices,scores):
            row=rows[int(idx)]
            key=(row.get("session_id", ""),row["trial_id"],row["timestamp"])
            grouped.setdefault(key,[]).append((float(p),int(row["is_intended"])))
        return float(np.mean([max(items,key=lambda z:z[0])[1] for items in grouped.values()]))
    top1=grouped_top1(test,proba)
    report={"schema_version":2,"dataset":str(args.dataset),"rows":len(rows),
        "data_sources":sorted({str(r.get("data_source", "unknown")) for r in rows}),
        "profiles":len({r.get("profile_id", "unknown") for r in rows}),
        "group_column":group_column,"split_groups":{
            "train":len(set(groups[train])),"validation":len(set(groups[val])),"test":len(set(groups[test]))},
        "validation_metrics":{"accuracy":float(accuracy_score(y[val],model.predict(X[val]))),
          "top1_target_accuracy":grouped_top1(val,val_proba)},
        "candidate_metrics":{"accuracy":float(accuracy_score(y[test],pred)),
          "precision":float(precision_score(y[test],pred,zero_division=0)),
          "recall":float(recall_score(y[test],pred,zero_division=0)),
          "f1":float(f1_score(y[test],pred,zero_division=0)),
          "roc_auc":float(roc_auc_score(y[test],proba))},
        "top1_target_accuracy":top1,
        "warning":"Synthetic/calibration development metric; not proof of real-user benefit."}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":model,"feature_names":list(TARGET_FEATURE_NAMES),"target":"is_intended",
                 "schema_version":2,"model_version":"intent-target-2",
                 "data_sources":report["data_sources"],
                 "status":"candidate_requires_evaluation_before_promotion"},
                args.output_dir/"intent_target_model.candidate.joblib")
    (args.output_dir/"intent_target_training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
