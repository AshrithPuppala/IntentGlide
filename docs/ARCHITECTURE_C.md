# Architecture C: ML-controlled Butterworth low-pass

## What changes

Architecture C is an additional filter path. It does not replace the existing
raw, fixed Butterworth, heuristic/ML EMA, or full intent-assisted modes.

1. `MotionAnalyzer` extracts the shared 11 runtime features.
2. `ButterworthCutoffPolicy` predicts a target low-pass cutoff in Hz.
3. `AdaptiveButterworthFilter` applies a second-order Butterworth low-pass.
4. The cutoff moves smoothly in log-frequency with a time-based slew limit.
5. If the model is missing, invalid, or not yet warm, a bounded fallback is used.

The existing ML-EMA model continues to predict alpha. The new cutoff model is a
separate artifact: `ml/artifacts/random_forest_cutoff.joblib`.

## Training data and labels

`ml.synthetic_data_butterworth` generates controlled synthetic trajectories
with 4-12 Hz disturbance, independent X/Y amplitudes/phases (up to about 15 px
per axis), varied trajectory shapes, 2-5 second trial durations, and six
simulated user-like profiles. These are engineering disturbances, not medical
or physiological tremor simulations.

`ml.butterworth_dataset` extracts the exact shared 11 features from the past
window. It then searches candidate cutoff frequencies on the following future
horizon, using the past observed window to warm each candidate Butterworth
filter state. Future ground truth is used only to produce the offline target
`cutoff_optimal_hz`; it is never an inference feature.

The training split groups by `profile_id` by default, so test/validation
profiles are not present in the training subset. If profile metadata is absent,
training falls back to grouping by trial ID. This is a first simulated-profile
generalisation test, not evidence of real-user generalisation.

## Dependency note

The patch includes `scipy>=1.11` in `requirements.txt`. SciPy is already used by the project DSP implementation and is also used for offline candidate Butterworth replay. Matplotlib is optional and needed only by the benchmark plot (`pip install matplotlib`).

## Commands (Windows CMD, repository root)

```cmd
python -m ml.synthetic_data_butterworth --trials 500 --sample-rate 120 --min-duration 2 --max-duration 5 --seed 2026 --output data/generated_arch_c
python -m ml.butterworth_dataset --inputs data/generated_arch_c/synthetic_input.csv --labels data/generated_arch_c/synthetic_labels.csv --output data/processed/butterworth_training_dataset.csv
python -m ml.train_cutoff --dataset data/processed/butterworth_training_dataset.csv --group-column profile_id
python -m pytest -q tests/test_adaptive_butterworth.py tests/test_butterworth_ml.py tests/test_architecture_c_orchestrator.py
python -m app.main
```

Press **5** to select ML Butterworth. Key 3 remains the existing adaptive EMA;
key 4 remains the existing full intent-assisted EMA path.

## Benchmark

```cmd
python benchmark_butterworth.py --frequency 8 --amplitude 12 --duration 3.5
```

The benchmark writes to a new timestamped directory under `data/processed`,
so it does not overwrite a CSV or PNG that may be open in Excel or an image
viewer. It compares raw, fixed 3 Hz Butterworth, heuristic EMA, ML-EMA, and
ML-Butterworth on the same synthetic trajectory. It reports tracking RMSE,
residual tone amplitude, attenuation, and pipeline timing.

## Interpretation cautions

- The candidate cutoff is selected from a finite grid (default 1.5-18 Hz), not
  a mathematically continuous global optimum.
- Label cost defaults to tracking RMSE + 0.5 * residual 4-12 Hz error RMS +
  0.5 * estimated lag in pixels. These are configurable in the dataset command;
  they are initial engineering weights, not validated user-preference weights.
- The feature extractor is a one-second window; warm-up uses a neutral 8 Hz
  cutoff until valid features are available.
- A second-order Butterworth low-pass has frequency-dependent phase delay.
  Compare both tremor attenuation and tracking/lag. Do not judge success from
  visual smoothness alone.
- The Pygame cursor is still an in-window cursor. This mode does not inject
  filtered coordinates into the Windows system pointer.
- No clinical or real-user performance claim is supported by synthetic tests.
