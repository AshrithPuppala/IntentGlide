"""Compare raw, fixed, heuristic-adaptive and ML-adaptive filtering on one known synthetic reach.

Run from the IntentGlide repository root after retraining the model against the
current DSP feature definitions:
    python benchmark_filtering.py --frequency 8 --amplitude 12

Outputs CSV and PNG files under data/processed/.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from dsp.adaptive_filter import AdaptiveFilter
from dsp.fixed_filter import FixedFilter
from dsp.motion_analyzer import MotionAnalyzer
from input.synthetic_generator import SyntheticReach
from ml.filter_policy import FilterPolicy
from shared.contracts import Target


def sinusoid_amplitude(t: np.ndarray, signal: np.ndarray, frequency: float) -> float:
    """Least-squares amplitude of one frequency, allowing DC and linear drift."""
    centred_t = t - np.mean(t)
    design = np.column_stack((
        np.sin(2.0 * np.pi * frequency * t),
        np.cos(2.0 * np.pi * frequency * t),
        np.ones_like(t),
        centred_t,
    ))
    coeff, *_ = np.linalg.lstsq(design, signal, rcond=None)
    return float(np.hypot(coeff[0], coeff[1]))


def rmse_2d(actual: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((actual - truth) ** 2, axis=1))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=8.0)
    parser.add_argument("--amplitude", type=float, default=12.0)
    parser.add_argument("--sample-rate", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--model", type=Path, default=Path("ml/artifacts/random_forest_alpha.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if args.sample_rate <= 2 * args.frequency:
        raise ValueError("Sample rate must exceed twice the disturbance frequency (Nyquist condition).")
    if args.duration < 1.5:
        raise ValueError("Use at least 1.5 seconds so the feature window can warm up and the tone can be measured.")

    policy = FilterPolicy(args.model)
    if policy.source != "ml":
        raise RuntimeError(
            f"ML model did not load: {policy.get_status()}. Retrain against the current DSP features first."
        )

    clock_value = [0.0]
    clock = lambda: clock_value[0]
    target = Target("benchmark-target", 700.0, 450.0, 50.0, 50.0)
    reach = SyntheticReach(
        start=(100.0, 100.0),
        target=target,
        duration=args.duration,
        tremor_freq=args.frequency,
        tremor_amp=args.amplitude,
        trial_id=1,
        candidates=[target],
        clock=clock,
    )

    analyzer = MotionAnalyzer()
    ml_filter = AdaptiveFilter(nominal_fs=args.sample_rate)
    heuristic_filter = AdaptiveFilter(nominal_fs=args.sample_rate)
    fixed_filter = FixedFilter(cutoff_hz=3.0, nominal_fs=args.sample_rate)

    rows: list[dict[str, float | str | bool]] = []
    total_ms: list[float] = []
    model_alpha_values: list[float] = []

    n_samples = int(round(args.duration * args.sample_rate))
    for index in range(n_samples):
        clock_value[0] = index / args.sample_rate
        sample = reach.get_sample()
        truth = reach.last_truth

        start = time.perf_counter()
        features = analyzer.update(sample)
        alpha_target = None
        if features.valid:
            alpha_target = policy.predict_alpha(features)
            if policy.source == "ml":
                model_alpha_values.append(alpha_target)
            else:
                alpha_target = None

        out_ml = ml_filter.update(sample, features, alpha_ml=alpha_target)
        out_heuristic = heuristic_filter.update(sample, features, alpha_ml=None)
        out_fixed = fixed_filter.update(sample)
        total_ms.append((time.perf_counter() - start) * 1000.0)

        rows.append({
            "timestamp": sample.timestamp,
            "raw_x": sample.x,
            "raw_y": sample.y,
            "truth_x": truth.x,
            "truth_y": truth.y,
            "fixed_x": out_fixed.x,
            "fixed_y": out_fixed.y,
            "heuristic_x": out_heuristic.x,
            "heuristic_y": out_heuristic.y,
            "ml_x": out_ml.x,
            "ml_y": out_ml.y,
            "features_valid": features.valid,
            "oscillation_score": features.oscillation_score,
            "dominant_frequency": features.dominant_frequency or 0.0,
            "model_alpha_target": alpha_target if alpha_target is not None else "",
            "ml_applied_alpha": out_ml.alpha if out_ml.alpha is not None else "",
            "ml_cutoff_hz": ml_filter.cutoff_hz if ml_filter.cutoff_hz is not None else "",
            "filter_source": ml_filter.last_source,
        })

    # Exclude initial analyzer warm-up and initial filter settling from the tone metrics.
    t_all = np.asarray([float(r["timestamp"]) for r in rows])
    truth_all = np.asarray([[float(r["truth_x"]), float(r["truth_y"])] for r in rows])
    signals = {
        "raw": np.asarray([[float(r["raw_x"]), float(r["raw_y"])] for r in rows]),
        "fixed": np.asarray([[float(r["fixed_x"]), float(r["fixed_y"])] for r in rows]),
        "heuristic": np.asarray([[float(r["heuristic_x"]), float(r["heuristic_y"])] for r in rows]),
        "ml": np.asarray([[float(r["ml_x"]), float(r["ml_y"])] for r in rows]),
    }
    mask = (t_all >= 1.0) & (t_all <= args.duration - 0.05)
    if int(mask.sum()) < int(args.sample_rate * 0.75):
        raise RuntimeError("Not enough post-warm-up samples for a reliable comparison.")

    t_eval = t_all[mask]
    truth_eval = truth_all[mask]
    amplitude_raw = sinusoid_amplitude(t_eval, signals["raw"][mask, 0] - truth_eval[:, 0], args.frequency)

    print("\nMODEL STATUS:", policy.get_status())
    print(f"Trial: {args.frequency:.2f} Hz disturbance, {args.amplitude:.2f} px X/Y amplitude, {args.sample_rate:.1f} Hz sampling")
    print(f"Evaluation samples after warm-up: {int(mask.sum())}")
    print("\nMETHOD       2D-RMSE(px)   RESIDUAL-TONE(px)   TONE-ATTENUATION(%)")
    print("----------------------------------------------------------------")
    for name in ("raw", "fixed", "heuristic", "ml"):
        path = signals[name][mask]
        err = path - truth_eval
        tone_amp = sinusoid_amplitude(t_eval, err[:, 0], args.frequency)
        attenuation = 100.0 * (1.0 - tone_amp / amplitude_raw) if amplitude_raw > 1e-9 else 0.0
        print(f"{name.upper():<11} {rmse_2d(path, truth_eval):>10.3f} {tone_amp:>19.3f} {attenuation:>20.1f}")

    if model_alpha_values:
        alpha_arr = np.asarray(model_alpha_values)
        print("\nML predicted alpha: "
              f"median={np.median(alpha_arr):.3f}, mean={np.mean(alpha_arr):.3f}, "
              f"min={np.min(alpha_arr):.3f}, max={np.max(alpha_arr):.3f}")
    print("Applied ML filter cutoff at end: "
          f"{ml_filter.cutoff_hz:.3f} Hz; last applied alpha={ml_filter._last_out.alpha:.3f}")
    print("Pipeline time per sample: "
          f"median={np.median(total_ms):.3f} ms, p95={np.percentile(total_ms, 95):.3f} ms")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "filter_benchmark_8hz.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved sample-by-sample results:", csv_path)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped plot. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t_all, [r["raw_x"] for r in rows], label="Raw", alpha=0.55)
    axes[0].plot(t_all, [r["truth_x"] for r in rows], label="Intended truth", linewidth=2)
    axes[0].plot(t_all, [r["fixed_x"] for r in rows], label="Fixed Butterworth")
    axes[0].plot(t_all, [r["heuristic_x"] for r in rows], label="Heuristic adaptive")
    axes[0].plot(t_all, [r["ml_x"] for r in rows], label="ML adaptive")
    axes[0].set_ylabel("X position (pixels)")
    axes[0].set_title("Raw vs filtered cursor position")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(t_all, [r["ml_applied_alpha"] if r["ml_applied_alpha"] != "" else np.nan for r in rows], label="Applied ML alpha")
    axes[1].plot(t_all, [r["ml_cutoff_hz"] if r["ml_cutoff_hz"] != "" else np.nan for r in rows], label="ML cutoff (Hz)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Alpha / cutoff (different units)")
    axes[1].set_title("Adaptive filter control values (separate scales; inspect CSV for exact values)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    plot_path = args.output_dir / "filter_benchmark_8hz.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print("Saved plot:", plot_path)


if __name__ == "__main__":
    main()
