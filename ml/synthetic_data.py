"""Generate synthetic observed cursor traces plus Person-1-style labels.

This is a development data generator, not a physiological tremor simulator.
It creates controlled trajectories with injected sinusoidal disturbances.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

INPUT_FIELDS = ["trial_id", "timestamp", "x", "y", "source"]
LABEL_FIELDS = [
    "trial_id", "timestamp", "intended_target_id", "true_state", "phase",
    "progress", "truth_x", "truth_y", "dist_to_target", "tremor_active",
    "tremor_freq", "tremor_amp", "ambiguous", "rival_target_id", "override",
    "generation_seed",
]


def generate_trials(n_trials: int = 400, sample_rate: float = 120.0,
                    duration_s: float = 3.5, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = int(round(sample_rate * duration_s))
    if n < 220:
        raise ValueError("duration_s must provide at least 220 samples at this sample rate")
    input_rows, label_rows = [], []

    for trial_id in range(n_trials):
        x0, y0 = rng.uniform(100, 350), rng.uniform(100, 300)
        target_x, target_y = rng.uniform(550, 1000), rng.uniform(350, 620)
        target_id = f"target-{int(rng.integers(1, 5))}"
        rival_id = f"target-{int(rng.integers(5, 9))}"
        ambiguous = bool(rng.random() < 0.20)
        override = bool(rng.random() < 0.035)
        tremor_active = bool(rng.random() < 0.72)
        tremor_freq = float(rng.uniform(4.0, 12.0)) if tremor_active else 0.0
        tremor_amp = float(rng.uniform(1.0, 7.5)) if tremor_active else 0.0
        phase_x, phase_y = rng.uniform(0, 2*np.pi, size=2)
        axis_mix = rng.uniform(0.35, 1.0)
        white_noise = rng.uniform(0.15, 0.8)
        curve_x, curve_y = rng.normal(0, 35), rng.normal(0, 35)
        t = np.arange(n, dtype=float) / sample_rate
        progress = np.linspace(0.0, 1.0, n)

        # Smooth point-to-point intended path with a modest curved component.
        truth_x = x0 + (target_x - x0) * progress + curve_x * np.sin(np.pi * progress)
        truth_y = y0 + (target_y - y0) * progress + curve_y * np.sin(np.pi * progress)
        if override:
            # Simulate the user changing direction late in the reach.
            mask = progress >= 0.72
            reversal = progress[mask] - 0.72
            truth_x[mask] -= (target_x - x0) * 0.45 * reversal
            truth_y[mask] += (target_y - y0) * 0.30 * reversal

        disturbance_x = tremor_amp * np.sin(2*np.pi*tremor_freq*t + phase_x) if tremor_active else 0.0
        disturbance_y = tremor_amp * axis_mix * np.sin(2*np.pi*tremor_freq*t + phase_y) if tremor_active else 0.0
        observed_x = truth_x + disturbance_x + rng.normal(0, white_noise, n)
        observed_y = truth_y + disturbance_y + rng.normal(0, white_noise, n)
        dist = np.hypot(target_x - truth_x, target_y - truth_y)

        for i in range(n):
            p = float(progress[i])
            if p < 0.08:
                motion_phase = "onset"
            elif p < 0.66:
                motion_phase = "transit"
            elif p < 0.90:
                motion_phase = "approach"
            else:
                motion_phase = "dwell"

            if override and p >= 0.72:
                state = "override"
            elif ambiguous and motion_phase in ("approach", "dwell"):
                state = "target_ambiguous"
            elif tremor_active:
                state = "oscillation_detected"
            elif motion_phase in ("approach", "dwell"):
                state = "target_confident"
            else:
                state = "stable"

            ts = round(float(t[i]), 9)
            input_rows.append({
                "trial_id": trial_id, "timestamp": ts,
                "x": round(float(observed_x[i]), 6),
                "y": round(float(observed_y[i]), 6), "source": "synthetic",
            })
            label_rows.append({
                "trial_id": trial_id, "timestamp": ts,
                "intended_target_id": target_id, "true_state": state,
                "phase": motion_phase, "progress": round(p, 6),
                "truth_x": round(float(truth_x[i]), 6),
                "truth_y": round(float(truth_y[i]), 6),
                "dist_to_target": round(float(dist[i]), 6),
                "tremor_active": str(tremor_active).lower(),
                "tremor_freq": round(tremor_freq, 6),
                "tremor_amp": round(tremor_amp, 6),
                "ambiguous": str(ambiguous).lower(),
                "rival_target_id": rival_id if ambiguous else "",
                "override": str(override and p >= 0.72).lower(),
                "generation_seed": seed,
            })
    return input_rows, label_rows


def _write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--sample-rate", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/generated"))
    args = parser.parse_args()
    inputs, labels = generate_trials(args.trials, args.sample_rate, args.duration, args.seed)
    _write_csv(args.output / "synthetic_input.csv", inputs, INPUT_FIELDS)
    _write_csv(args.output / "synthetic_labels.csv", labels, LABEL_FIELDS)
    print(f"Wrote {len(inputs):,} input samples and {len(labels):,} label rows to {args.output}")


if __name__ == "__main__":
    main()
