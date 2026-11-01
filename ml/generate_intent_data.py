"""Generate labelled candidate-target examples for a first intent ranker.

Synthetic development data only. The intended target is known to the simulator;
it is never provided to runtime inference.
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import numpy as np
from shared.contracts import Target
from interaction.target_analyzer import TARGET_FEATURE_NAMES, target_feature_dict

FIELDS = ["schema_version", "data_source", "session_id", "trial_id", "profile_id", "timestamp",
          "target_id", "intended_target_id", "is_intended", *TARGET_FEATURE_NAMES]


def generate_rows(trials: int = 400, sample_rate: float = 120.0, seed: int = 2026):
    rng = np.random.default_rng(seed)
    rows = []
    profile_params = [(4.5 + i * 0.9, 1.5 + i * 1.7, 1.6 + i * 0.35) for i in range(8)]
    for trial_id in range(trials):
        profile_index = trial_id % len(profile_params)
        profile_id = f"synthetic-user-{profile_index:02d}"
        session_id = f"synthetic-session-{trial_id // 25:04d}"
        base_freq, base_amp, base_duration = profile_params[profile_index]
        n_targets = int(rng.integers(3, 6))
        centres = []
        for _ in range(n_targets):
            candidate = (float(rng.uniform(100, 900)), float(rng.uniform(90, 620)))
            centres.append(candidate)
        start = (float(rng.uniform(70, 260)), float(rng.uniform(80, 580)))
        intended_index = int(rng.integers(0, n_targets))
        intended_id = f"target-{intended_index}"
        targets = [Target(f"target-{i}", cx, cy,
                          float(rng.uniform(42, 90)), float(rng.uniform(38, 82)))
                   for i, (cx, cy) in enumerate(centres)]
        end = centres[intended_index]
        duration = float(np.clip(base_duration + rng.normal(0, 0.25), 1.3, 5.0))
        n = max(160, int(round(duration * sample_rate)))
        t = np.arange(n, dtype=float) / sample_rate
        u = np.clip(t / duration, 0.0, 1.0)
        minimum_jerk = 10*u**3 - 15*u**4 + 6*u**5
        curve = rng.normal(0, 28, size=2)
        truth_x = start[0] + minimum_jerk*(end[0]-start[0]) + curve[0]*np.sin(np.pi*u)
        truth_y = start[1] + minimum_jerk*(end[1]-start[1]) + curve[1]*np.sin(np.pi*u)
        freq = float(np.clip(base_freq + rng.normal(0, 1.0), 4.0, 12.0))
        amp_x = float(np.clip(base_amp + rng.normal(0, 1.5), 0.0, 15.0))
        amp_y = float(np.clip(amp_x*rng.uniform(0.45, 1.15), 0.0, 15.0))
        phase_x, phase_y = rng.uniform(0, 2*np.pi, size=2)
        observed_x = truth_x + amp_x*np.sin(2*np.pi*freq*t + phase_x) + rng.normal(0, 0.45, n)
        observed_y = truth_y + amp_y*np.sin(2*np.pi*freq*t + phase_y) + rng.normal(0, 0.45, n)
        for i in range(18, n, 6):  # about 20 snapshots/sec, after history warm-up
            history_start = max(0, i - int(0.45*sample_rate))
            history = [(float(t[j]), float(observed_x[j]), float(observed_y[j]))
                       for j in range(history_start, i+1)]
            x, y = float(observed_x[i]), float(observed_y[i])
            for target in targets:
                profile_stats = {"motion_variability_rms": float(base_amp * 0.35),
                                 "median_overshoot_px": float(base_amp * 0.8),
                                 "median_movement_speed": float(np.linalg.norm(np.asarray(end)-np.asarray(start)) / duration)}
                feats = target_feature_dict(history, x, y, target,
                                            acquisition_margin_px=30.0 + base_amp,
                                            profile_features=profile_stats)
                rows.append({
                    "schema_version": 2,
                    "data_source": "synthetic",
                    "session_id": session_id,
                    "trial_id": trial_id, "profile_id": profile_id,
                    "timestamp": float(t[i]), "target_id": target.id,
                    "intended_target_id": intended_id,
                    "is_intended": int(target.id == intended_id),
                    **feats,
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--sample-rate", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("data/generated/intent_candidates.csv"))
    args = parser.parse_args()
    rows = generate_rows(args.trials, args.sample_rate, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows):,} candidate examples from {args.trials} trials to {args.output}")
    print("Profiles:", len(set(r['profile_id'] for r in rows)), "Positive examples:", sum(r['is_intended'] for r in rows))

if __name__ == "__main__":
    main()
