"""Build training examples for the ML-predicted Butterworth cutoff policy."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

from ml.dataset import align_input_and_labels, _as_bool
from ml.features import FEATURE_NAMES, extract_features
from ml.butterworth_labeling import select_optimal_cutoff, default_cutoff_candidates


def build_cutoff_training_rows(aligned_rows, sample_rate: float = 120.0,
                               window_size: int = 128, horizon: int = 64,
                               stride: int = 32, candidates=None,
                               jitter_weight: float = 0.5,
                               lag_weight: float = 0.5):
    if window_size < 32 or horizon < 16 or stride < 1:
        raise ValueError("Require window_size>=32, horizon>=16 and stride>=1")
    trials = defaultdict(list)
    for row in aligned_rows:
        trials[str(row["trial_id"])].append(row)

    rows_out, skipped_override = [], 0
    for trial_id, rows in trials.items():
        rows.sort(key=lambda row: row["timestamp"])
        observed = np.asarray([[r["x"], r["y"]] for r in rows], dtype=float)
        truth = np.asarray([[r["label_truth_x"], r["label_truth_y"]] for r in rows], dtype=float)
        timestamps = np.asarray([r["timestamp"] for r in rows], dtype=float)
        if len(rows) < window_size + horizon:
            continue
        dt = np.diff(timestamps)
        if len(dt) and not np.allclose(dt, 1.0 / sample_rate, rtol=2e-3, atol=2e-5):
            raise ValueError(f"Trial {trial_id} is not sampled at {sample_rate} Hz")

        for start in range(0, len(rows) - window_size - horizon + 1, stride):
            end = start + window_size
            future_end = end + horizon
            if any(_as_bool(r.get("label_override", "false")) for r in rows[end:future_end]):
                skipped_override += 1
                continue
            history = observed[start:end]
            features = extract_features(history, sample_rate)
            cutoff, metrics = select_optimal_cutoff(
                observed_future=observed[end:future_end],
                truth_future=truth[end:future_end],
                history=history,
                candidates=candidates,
                sample_rate=sample_rate,
                jitter_weight=jitter_weight,
                lag_weight=lag_weight,
            )
            endpoint = rows[end - 1]
            item = {
                "trial_id": str(trial_id),
                "window_timestamp": float(timestamps[end - 1]),
                "window_start_timestamp": float(timestamps[start]),
                "phase": endpoint.get("label_phase", ""),
                "true_state": endpoint.get("label_true_state", ""),
                "tremor_active": endpoint.get("label_tremor_active", "false"),
                "tremor_freq": float(endpoint.get("label_tremor_freq", 0) or 0),
                "tremor_amp": float(endpoint.get("label_tremor_amp", 0) or 0),
                "ambiguous": endpoint.get("label_ambiguous", "false"),
                "override": endpoint.get("label_override", "false"),
                "profile_id": endpoint.get("label_profile_id", "unknown"),
                "cutoff_optimal_hz": float(cutoff),
                **metrics,
                **features,
            }
            rows_out.append(item)

    if not rows_out:
        raise ValueError("No cutoff training windows produced")
    return rows_out, skipped_override


def write_cutoff_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("data/generated_arch_c/synthetic_input.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/generated_arch_c/synthetic_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/butterworth_training_dataset.csv"))
    parser.add_argument("--sample-rate", type=float, default=120.0)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--min-cutoff", type=float, default=1.5)
    parser.add_argument("--max-cutoff", type=float, default=18.0)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--jitter-weight", type=float, default=0.5)
    parser.add_argument("--lag-weight", type=float, default=0.5)
    args = parser.parse_args()
    aligned, unmatched = align_input_and_labels(args.inputs, args.labels)
    candidates = default_cutoff_candidates(args.min_cutoff, args.max_cutoff, args.candidates)
    rows, skipped = build_cutoff_training_rows(
        aligned, args.sample_rate, args.window, args.horizon, args.stride,
        candidates, args.jitter_weight, args.lag_weight,
    )
    write_cutoff_rows(args.output, rows)
    values = np.asarray([r["cutoff_optimal_hz"] for r in rows])
    print(f"Matched {len(aligned):,} rows; unmatched input rows: {unmatched:,}")
    print(f"Created {len(rows):,} cutoff examples; skipped override horizons: {skipped:,}")
    at_min = int(np.isclose(values, args.min_cutoff).sum())
    at_max = int(np.isclose(values, args.max_cutoff).sum())
    print(f"Cutoff labels Hz: min={values.min():.3f}, median={np.median(values):.3f}, mean={values.mean():.3f}, max={values.max():.3f}")
    print(f"Labels at lower/upper bound: {at_min} ({100*at_min/len(values):.1f}%) / {at_max} ({100*at_max/len(values):.1f}%)")
    print(f"Saved: {args.output}")
    print("Features:", ", ".join(FEATURE_NAMES))


if __name__ == "__main__":
    main()
