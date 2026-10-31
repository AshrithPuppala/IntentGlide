"""Compare raw, fixed BW, heuristic EMA, ML-EMA and ML-Butterworth paths.

Run after training ml/artifacts/random_forest_cutoff.joblib:
  python benchmark_butterworth.py --frequency 8 --amplitude 12
Outputs are written to a fresh timestamped directory to avoid overwriting an
open CSV/PNG on Windows.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from time import perf_counter
import math

import matplotlib.pyplot as plt
import numpy as np

from dsp.adaptive_filter import AdaptiveFilter
from dsp.adaptive_butterworth import AdaptiveButterworthFilter
from dsp.fixed_filter import FixedFilter
from dsp.motion_analyzer import MotionAnalyzer
from input.synthetic_generator import SyntheticReach
from ml.filter_policy import FilterPolicy
from ml.butterworth_policy import ButterworthCutoffPolicy
from shared.contracts import InputSample, Target


def _tone_amplitude(error: np.ndarray, times: np.ndarray, frequency: float) -> float:
    """Fit sine/cosine at the injected frequency; return combined XY amplitude."""
    columns = [np.sin(2 * np.pi * frequency * times),
               np.cos(2 * np.pi * frequency * times),
               np.ones_like(times)]
    design = np.column_stack(columns)
    amps = []
    for axis in range(2):
        coeff, *_ = np.linalg.lstsq(design, error[:, axis], rcond=None)
        amps.append(float(math.hypot(coeff[0], coeff[1])))
    return float(math.hypot(amps[0], amps[1]))


def _rmse(output: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    delta = output[mask] - truth[mask]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _summarize(name: str, output: np.ndarray, truth: np.ndarray,
               times: np.ndarray, frequency: float, mask: np.ndarray,
               raw_tone: float) -> dict:
    residual_tone = _tone_amplitude(output[mask] - truth[mask], times[mask], frequency)
    return {
        "method": name,
        "tracking_rmse_px": _rmse(output, truth, mask),
        "residual_tone_amplitude_px_2d": residual_tone,
        "tone_attenuation_pct": 100.0 * (1.0 - residual_tone / max(raw_tone, 1e-9)),
    }


def run_benchmark(frequency: float = 8.0, amplitude: float = 12.0,
                  duration: float = 3.5, sample_rate: float = 120.0,
                  output_dir: Path | None = None):
    artifact_dir = Path("ml/artifacts")
    cutoff_path = artifact_dir / "random_forest_cutoff.joblib"
    if not cutoff_path.is_file():
        raise FileNotFoundError(
            f"Missing {cutoff_path}. Train the Architecture-C cutoff model first: "
            "python -m ml.train_cutoff"
        )

    alpha_policy = FilterPolicy(artifact_dir / "random_forest_alpha.joblib")
    alpha_model_loaded = alpha_policy.source == "ml"
    alpha_policy_label = "ML_EMA" if alpha_model_loaded else "HEURISTIC_EMA_FALLBACK"
    cutoff_policy = ButterworthCutoffPolicy(cutoff_path)
    if cutoff_policy.source != "ml":
        raise RuntimeError(f"Cutoff model failed to load: {cutoff_policy.get_status()}")

    fs = float(sample_rate)
    n = int(round(duration * fs))
    target = Target("benchmark-target", 700.0, 650.0, 40.0, 40.0)
    now = [0.0]
    reach = SyntheticReach(
        start=(100.0, 110.0), target=target, duration=duration,
        tremor_freq=frequency, tremor_amp=amplitude, trial_id=1,
        candidates=[target], clock=lambda: now[0],
    )

    analyzer = MotionAnalyzer()
    fixed = FixedFilter(cutoff_hz=3.0, nominal_fs=fs)
    ema = AdaptiveFilter(nominal_fs=fs)
    ml_ema = AdaptiveFilter(nominal_fs=fs)
    ml_butterworth = AdaptiveButterworthFilter(nominal_fs=fs)

    times, raw, truth = [], [], []
    outputs = {"fixed": [], "heuristic_ema": [], "ml_ema": [], "ml_butterworth": []}
    alpha_predictions, cutoff_predictions, cutoff_applied = [], [], []
    timings = []

    for i in range(n):
        now[0] = i / fs
        sample = reach.get_sample()
        gt = reach.last_truth
        start = perf_counter()
        features = analyzer.update(sample)
        fixed_out = fixed.update(sample)
        heuristic_out = ema.update(sample, features)

        alpha_target = None
        cutoff_target = None
        if features.valid:
            predicted_alpha = alpha_policy.predict_alpha(features)
            if alpha_policy.source == "ml":
                alpha_target = predicted_alpha
            predicted_cutoff = cutoff_policy.predict_cutoff(features)
            if cutoff_policy.source == "ml":
                cutoff_target = predicted_cutoff
            else:
                predicted_cutoff = float("nan")
        else:
            predicted_alpha = float("nan")
            predicted_cutoff = float("nan")

        # Dedicated ML-EMA comparator; separate state so heuristic and ML paths
        # see exactly the same raw stream.
        ml_ema_out = ml_ema.update(sample, features, alpha_ml=alpha_target)
        bw_out = ml_butterworth.update(sample, features, cutoff_target_hz=cutoff_target)
        timings.append((perf_counter() - start) * 1000.0)

        times.append(sample.timestamp)
        raw.append([sample.x, sample.y])
        truth.append([gt.x, gt.y])
        outputs["fixed"].append([fixed_out.x, fixed_out.y])
        outputs["heuristic_ema"].append([heuristic_out.x, heuristic_out.y])
        outputs["ml_ema"].append([ml_ema_out.x, ml_ema_out.y])
        outputs["ml_butterworth"].append([bw_out.x, bw_out.y])
        alpha_predictions.append(predicted_alpha)
        cutoff_predictions.append(predicted_cutoff)
        cutoff_applied.append(ml_butterworth.cutoff_hz)

    # Avoid counting initial feature/filter warm-up in the primary scores.
    times = np.asarray(times, dtype=float)
    raw = np.asarray(raw, dtype=float)
    truth = np.asarray(truth, dtype=float)
    outputs = {key: np.asarray(value, dtype=float) for key, value in outputs.items()}
    warmup_seconds = min(1.0, max(0.5, duration * 0.25))
    mask = times >= warmup_seconds
    raw_tone = _tone_amplitude(raw[mask] - truth[mask], times[mask], frequency)

    summary = [_summarize("RAW", raw, truth, times, frequency, mask, raw_tone)]
    for key, label in [
        ("fixed", "FIXED_BUTTERWORTH_3HZ"),
        ("heuristic_ema", "HEURISTIC_EMA"),
        ("ml_ema", alpha_policy_label),
        ("ml_butterworth", "ML_BUTTERWORTH"),
    ]:
        summary.append(_summarize(label, outputs[key], truth, times, frequency, mask, raw_tone))

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/processed") / f"architecture_c_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    csv_path = output_dir / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["time", "raw_x", "raw_y", "truth_x", "truth_y"]
        for key in outputs:
            fields.extend([f"{key}_x", f"{key}_y"])
        fields += ["alpha_prediction", "cutoff_prediction_hz", "cutoff_applied_hz"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(times)):
            row = {"time": times[i], "raw_x": raw[i, 0], "raw_y": raw[i, 1],
                   "truth_x": truth[i, 0], "truth_y": truth[i, 1],
                   "alpha_prediction": alpha_predictions[i],
                   "cutoff_prediction_hz": cutoff_predictions[i],
                   "cutoff_applied_hz": cutoff_applied[i]}
            for key, values in outputs.items():
                row[f"{key}_x"], row[f"{key}_y"] = values[i]
            writer.writerow(row)

    report_path = output_dir / "summary.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    axes[0].plot(truth[:, 0], truth[:, 1], label="Intended truth", linewidth=2)
    axes[0].plot(raw[:, 0], raw[:, 1], label="Raw", alpha=0.5)
    axes[0].plot(outputs["fixed"][:, 0], outputs["fixed"][:, 1], label="Fixed Butterworth 3 Hz")
    axes[0].plot(outputs["heuristic_ema"][:, 0], outputs["heuristic_ema"][:, 1], label="Heuristic EMA")
    axes[0].plot(outputs["ml_ema"][:, 0], outputs["ml_ema"][:, 1], label="ML EMA")
    axes[0].plot(outputs["ml_butterworth"][:, 0], outputs["ml_butterworth"][:, 1], label="ML Butterworth", linewidth=2)
    axes[0].set_title("Cursor trajectory: raw, baselines and Architecture C")
    axes[0].set_ylabel("Y position (px)")
    axes[0].legend(ncol=2)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(times, truth[:, 0], label="Truth X", linewidth=2)
    axes[1].plot(times, raw[:, 0], label="Raw X", alpha=0.45)
    axes[1].plot(times, outputs["fixed"][:, 0], label="Fixed Butterworth")
    axes[1].plot(times, outputs["heuristic_ema"][:, 0], label="Heuristic EMA")
    axes[1].plot(times, outputs["ml_ema"][:, 0], label="ML EMA")
    axes[1].plot(times, outputs["ml_butterworth"][:, 0], label="ML Butterworth", linewidth=2)
    axes[1].set_ylabel("X position (px)")
    axes[1].legend(ncol=3)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(times, cutoff_predictions, label="ML target cutoff (Hz)")
    axes[2].plot(times, cutoff_applied, label="Applied smoothed cutoff (Hz)", linewidth=2)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Cutoff (Hz)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    plot_path = output_dir / "benchmark.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Alpha model: {alpha_policy.get_status()}")
    print(f"Cutoff model: {cutoff_policy.get_status()}")
    print(f"Trial: {frequency:.2f} Hz disturbance, {amplitude:.2f} px per axis, {fs:.1f} Hz sampling")
    print(f"Evaluation begins after {warmup_seconds:.2f} s ({int(mask.sum())} samples)")
    print("\nMETHOD                  2D-RMSE(px)  RESIDUAL-TONE(px)  ATTENUATION(%)")
    for row in summary:
        print(f"{row['method']:<23} {row['tracking_rmse_px']:>10.3f} {row['residual_tone_amplitude_px_2d']:>18.3f} {row['tone_attenuation_pct']:>15.1f}")
    timing = np.asarray(timings)
    print(f"\nPipeline timing median={np.median(timing):.3f} ms, p95={np.percentile(timing,95):.3f} ms")
    print(f"Median applied ML cutoff: {np.nanmedian(cutoff_applied):.3f} Hz")
    print(f"Saved: {csv_path}, {report_path}, {plot_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=8.0)
    parser.add_argument("--amplitude", type=float, default=12.0, help="Per-axis injected amplitude in pixels")
    parser.add_argument("--duration", type=float, default=3.5)
    parser.add_argument("--sample-rate", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_benchmark(args.frequency, args.amplitude, args.duration, args.sample_rate, args.output_dir)


if __name__ == "__main__":
    main()
