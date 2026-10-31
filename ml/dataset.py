"""Align P1 input/label CSVs and construct causal feature/alpha-label rows."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from ml.features import FEATURE_NAMES, extract_features
from ml.labeling import select_optimal_alpha


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def align_input_and_labels(input_path: Path, labels_path: Path,
                           timestamp_decimals: int = 8):
    inputs, labels = _read_csv(input_path), _read_csv(labels_path)
    required_input = {"trial_id", "timestamp", "x", "y"}
    required_labels = {"trial_id", "timestamp", "truth_x", "truth_y"}
    if not inputs or not labels:
        raise ValueError("Input and label CSVs must both contain rows")
    if not required_input.issubset(inputs[0]):
        raise ValueError(f"Input CSV missing columns: {sorted(required_input - set(inputs[0]))}")
    if not required_labels.issubset(labels[0]):
        raise ValueError(f"Label CSV missing columns: {sorted(required_labels - set(labels[0]))}")

    label_index = {}
    for row in labels:
        key = (str(row["trial_id"]), round(float(row["timestamp"]), timestamp_decimals))
        if key in label_index:
            raise ValueError(f"Duplicate label timestamp key: {key}")
        label_index[key] = row

    aligned = []
    missing = 0
    for row in inputs:
        key = (str(row["trial_id"]), round(float(row["timestamp"]), timestamp_decimals))
        lab = label_index.get(key)
        if lab is None:
            missing += 1
            continue
        merged = dict(row)
        merged.update({f"label_{k}": v for k, v in lab.items() if k not in ("trial_id", "timestamp")})
        merged["trial_id"] = str(row["trial_id"])
        merged["timestamp"] = float(row["timestamp"])
        for key_name in ("x", "y", "label_truth_x", "label_truth_y"):
            merged[key_name] = float(merged[key_name])
        aligned.append(merged)
    if not aligned:
        raise ValueError("No input rows matched labels. Check trial_id/timestamp values.")
    return aligned, missing


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_training_rows(aligned_rows, sample_rate: float = 120.0,
                        window_size: int = 128, horizon: int = 64,
                        stride: int = 32, alpha_candidates=None):
    if window_size < 32 or horizon < 16 or stride < 1:
        raise ValueError("Require window_size>=32, horizon>=16, stride>=1")
    trials = defaultdict(list)
    for row in aligned_rows:
        trials[str(row["trial_id"])].append(row)

    output = []
    skipped_override = 0
    for trial_id, rows in trials.items():
        rows.sort(key=lambda r: r["timestamp"])
        obs = np.array([[r["x"], r["y"]] for r in rows], dtype=float)
        truth = np.array([[r["label_truth_x"], r["label_truth_y"]] for r in rows], dtype=float)
        timestamps = np.array([r["timestamp"] for r in rows], dtype=float)
        if len(rows) < window_size + horizon:
            continue
        # Reject clearly irregular traces for this initial fixed-rate feature implementation.
        dt = np.diff(timestamps)
        if len(dt) and not np.allclose(dt, 1.0 / sample_rate, rtol=2e-3, atol=2e-5):
            raise ValueError(f"Trial {trial_id} is not sampled at {sample_rate} Hz; resample explicitly first")
        for start in range(0, len(rows) - window_size - horizon + 1, stride):
            end = start + window_size
            future_end = end + horizon
            future_rows = rows[end:future_end]
            if any(_as_bool(r.get("label_override", "false")) for r in future_rows):
                skipped_override += 1
                continue
            features = extract_features(obs[start:end], sample_rate)
            alpha, metrics = select_optimal_alpha(
                observed_future=obs[end:future_end],
                truth_future=truth[end:future_end],
                initial_position=obs[end - 1],
                candidates=alpha_candidates,
                sample_rate=sample_rate,
            )
            endpoint = rows[end - 1]
            first = rows[start]
            item = {
                "trial_id": trial_id,
                "window_timestamp": float(timestamps[end - 1]),
                "phase": endpoint.get("label_phase", ""),
                "true_state": endpoint.get("label_true_state", ""),
                "tremor_active": endpoint.get("label_tremor_active", "false"),
                "tremor_freq": float(endpoint.get("label_tremor_freq", 0) or 0),
                "tremor_amp": float(endpoint.get("label_tremor_amp", 0) or 0),
                "ambiguous": endpoint.get("label_ambiguous", "false"),
                "override": endpoint.get("label_override", "false"),
                "window_start_timestamp": float(timestamps[start]),
                "alpha_optimal": alpha,
                **metrics,
                **features,
            }
            output.append(item)
    if not output:
        raise ValueError("No training windows were produced; check trial length, sample rate and labels")
    return output, skipped_override


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("data/generated/synthetic_input.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/generated/synthetic_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/training_dataset.csv"))
    parser.add_argument("--sample-rate", type=float, default=120.0)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    args = parser.parse_args()
    aligned, missing = align_input_and_labels(args.inputs, args.labels)
    rows, skipped = build_training_rows(aligned, args.sample_rate, args.window, args.horizon, args.stride)
    write_rows(args.output, rows)
    print(f"Matched {len(aligned):,} rows; unmatched input rows: {missing:,}")
    print(f"Created {len(rows):,} feature/alpha examples; skipped override horizons: {skipped:,}")
    print(f"Saved: {args.output}")
    print("Features:", ", ".join(FEATURE_NAMES))


if __name__ == "__main__":
    main()
