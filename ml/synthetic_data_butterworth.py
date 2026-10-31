"""Generate a broader set of synthetic cursor trials for Architecture C.

The generator uses controlled engineering disturbances, not physiological
simulations. It creates varied tremor frequencies/amplitudes, X/Y phases,
movement speeds, trajectory shapes, and trial durations. The CSV schema stays
compatible with ml.dataset.align_input_and_labels().
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
    "profile_id", "trajectory_type", "amp_x", "amp_y", "generation_seed",
]

# Simulated user profiles deliberately span different signal characteristics.
PROFILES = [
    {"freq": 4.8, "amp_scale": 0.65, "speed_scale": 0.75},
    {"freq": 6.0, "amp_scale": 0.90, "speed_scale": 1.00},
    {"freq": 7.2, "amp_scale": 1.20, "speed_scale": 1.15},
    {"freq": 8.4, "amp_scale": 1.35, "speed_scale": 0.85},
    {"freq": 9.6, "amp_scale": 0.80, "speed_scale": 1.30},
    {"freq": 11.2, "amp_scale": 1.10, "speed_scale": 1.00},
]


def generate_trials(n_trials: int = 500, sample_rate: float = 120.0,
                    min_duration_s: float = 2.0, max_duration_s: float = 5.0,
                    seed: int = 2026):
    if n_trials < 1 or sample_rate <= 0 or min_duration_s < 1.6 or max_duration_s < min_duration_s:
        raise ValueError("Invalid trial count, sample rate or duration range")
    rng = np.random.default_rng(seed)
    input_rows, label_rows = [], []

    for trial_id in range(n_trials):
        profile_id = int(trial_id % len(PROFILES))
        profile = PROFILES[profile_id]
        duration = float(rng.uniform(min_duration_s, max_duration_s))
        n = int(round(sample_rate * duration))
        t = np.arange(n, dtype=float) / sample_rate
        progress = np.linspace(0.0, 1.0, n)

        x0, y0 = rng.uniform(80, 260), rng.uniform(90, 280)
        target_x, target_y = rng.uniform(550, 980), rng.uniform(350, 630)
        dx, dy = target_x - x0, target_y - y0
        curve_x, curve_y = rng.normal(0, 45, size=2)
        trajectory_type = str(rng.choice(["minimum_jerk", "linear", "curved"], p=[0.55, 0.20, 0.25]))
        if trajectory_type == "minimum_jerk":
            p = 10 * progress**3 - 15 * progress**4 + 6 * progress**5
            truth_x = x0 + dx * p
            truth_y = y0 + dy * p
        elif trajectory_type == "linear":
            truth_x = x0 + dx * progress
            truth_y = y0 + dy * progress
        else:
            truth_x = x0 + dx * progress + curve_x * np.sin(np.pi * progress)
            truth_y = y0 + dy * progress + curve_y * np.sin(np.pi * progress)

        tremor_active = bool(rng.random() < 0.78)
        tremor_freq = float(np.clip(profile["freq"] + rng.normal(0, 1.4), 4.0, 12.0)) if tremor_active else 0.0
        amp_scale = profile["amp_scale"]
        amp_x = float(min(15.0, rng.uniform(0.8, 15.0) * amp_scale)) if tremor_active else 0.0
        amp_y = float(min(15.0, rng.uniform(0.8, 15.0) * amp_scale)) if tremor_active else 0.0
        phase_x, phase_y = rng.uniform(0, 2 * np.pi, size=2)
        amp_mod_x = float(rng.uniform(0.0, 0.30))
        amp_mod_y = float(rng.uniform(0.0, 0.30))
        noise_sd = float(rng.uniform(0.08, 1.0))
        speed_scale = float(profile["speed_scale"] * rng.uniform(0.75, 1.30))
        # Adjust the time parameter independently of the duration to vary movement speed.
        movement_progress = np.clip(progress * speed_scale, 0.0, 1.0)
        if trajectory_type == "minimum_jerk":
            p2 = 10 * movement_progress**3 - 15 * movement_progress**4 + 6 * movement_progress**5
        else:
            p2 = movement_progress
        truth_x = x0 + dx * p2 + (curve_x * np.sin(np.pi * movement_progress) if trajectory_type == "curved" else 0.0)
        truth_y = y0 + dy * p2 + (curve_y * np.sin(np.pi * movement_progress) if trajectory_type == "curved" else 0.0)

        override = bool(rng.random() < 0.04)
        if override:
            mask = progress >= 0.72
            late = progress[mask] - 0.72
            truth_x[mask] -= dx * 0.45 * late
            truth_y[mask] += dy * 0.30 * late

        mod_x = 1.0 + amp_mod_x * np.sin(2 * np.pi * 0.35 * t + rng.uniform(0, 2*np.pi))
        mod_y = 1.0 + amp_mod_y * np.sin(2 * np.pi * 0.28 * t + rng.uniform(0, 2*np.pi))
        disturbance_x = amp_x * mod_x * np.sin(2*np.pi*tremor_freq*t + phase_x) if tremor_active else 0.0
        disturbance_y = amp_y * mod_y * np.sin(2*np.pi*tremor_freq*t + phase_y) if tremor_active else 0.0
        observed_x = truth_x + disturbance_x + rng.normal(0, noise_sd, n)
        observed_y = truth_y + disturbance_y + rng.normal(0, noise_sd, n)
        dist = np.hypot(target_x - truth_x, target_y - truth_y)
        target_id = f"target-{int(rng.integers(1, 9))}"
        ambiguous = bool(rng.random() < 0.18)
        rival_id = f"rival-{int(rng.integers(1, 9))}" if ambiguous else ""
        amp_summary = float(math.hypot(amp_x, amp_y) / math.sqrt(2.0))

        for i in range(n):
            p = float(progress[i])
            phase = "onset" if p < 0.08 else "transit" if p < 0.66 else "approach" if p < 0.90 else "dwell"
            if override and p >= 0.72:
                state = "override"
            elif ambiguous and phase in ("approach", "dwell"):
                state = "target_ambiguous"
            elif tremor_active:
                state = "oscillation_detected"
            elif phase in ("approach", "dwell"):
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
                "phase": phase, "progress": round(p, 6),
                "truth_x": round(float(truth_x[i]), 6),
                "truth_y": round(float(truth_y[i]), 6),
                "dist_to_target": round(float(dist[i]), 6),
                "tremor_active": str(tremor_active).lower(),
                "tremor_freq": round(tremor_freq, 6),
                "tremor_amp": round(amp_summary, 6),
                "ambiguous": str(ambiguous).lower(),
                "rival_target_id": rival_id,
                "override": str(bool(override and p >= 0.72)).lower(),
                "profile_id": profile_id, "trajectory_type": trajectory_type,
                "amp_x": round(amp_x, 6), "amp_y": round(amp_y, 6),
                "generation_seed": seed,
            })
    return input_rows, label_rows


def _write(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--sample-rate", type=float, default=120.0)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("data/generated_arch_c"))
    args = parser.parse_args()
    inputs, labels = generate_trials(args.trials, args.sample_rate,
                                     args.min_duration, args.max_duration, args.seed)
    _write(args.output / "synthetic_input.csv", inputs, INPUT_FIELDS)
    _write(args.output / "synthetic_labels.csv", labels, LABEL_FIELDS)
    print(f"Wrote {len(inputs):,} input samples and {len(labels):,} labels to {args.output}")


if __name__ == "__main__":
    main()
