# Person 3 — ML Filter Controller

**Status:** First synthetic development pipeline is implemented in this starter bundle. It is not yet integrated with Person 2's final feature extractor or the Pygame app.

## Run locally from the IntentGlide repository root

Install the project's dependencies plus XGBoost:

```cmd
pip install -r requirements-ml.txt
```

Generate controlled trajectories using Person-1-style labels:

```cmd
python -m ml.synthetic_data --trials 400 --sample-rate 120 --duration 3.5 --seed 42
```

Align input and label rows, extract causal observed-motion features, and label each window with its offline best alpha:

```cmd
python -m ml.dataset --inputs data/generated/synthetic_input.csv --labels data/generated/synthetic_labels.csv
```

Train both regressors using trial-grouped train/validation/test splits:

```cmd
python -m ml.train --dataset data/processed/training_dataset.csv
```

Re-evaluate saved models on the same held-out trial split:

```cmd
python -m ml.evaluate --dataset data/processed/training_dataset.csv
```

Run tests:

```cmd
python -m pytest -q tests/test_ml.py
```

## Files

- `synthetic_data.py`: creates observed X/Y plus Person-1-style label CSVs.
- `features.py`: prototype window-level feature extraction from observed X/Y only.
- `labeling.py`: offline candidate-alpha replay and objective calculation.
- `dataset.py`: timestamp/trial alignment and feature/alpha table creation.
- `train.py`: grouped-split Random Forest and XGBoost training.
- `filter_policy.py`: inference wrapper with alpha bounds, rate limiting and heuristic fallback.
- `evaluate.py`: held-out regression metric re-evaluation.
- `dataset_sources.md`: how public datasets should and should not be used.

## Current prototype settings

- Synthetic input sample rate: 120 Hz.
- Feature window: 128 samples (~1.07 s).
- Label horizon: 64 future samples (~0.53 s).
- Window stride: 32 samples.
- Alpha candidates: 16 values from 0.12 to 0.85.
- Label objective: tracking RMSE + 0.50 × residual 4–12 Hz error RMS + 0.50 × estimated lag displacement in pixels.
- Model inputs: 13 prototype features generated from observed cursor X/Y only.
- Ground-truth fields (`truth_x`, `truth_y`), disturbance parameters and task labels are not model inputs.
- Override horizons are excluded from alpha regression.

These values are initial experimental choices. Agree feature definitions and sampling behaviour with Person 2 before integration. Confirm with Person 1 that `tremor_freq` is in Hz and `tremor_amp` is in pixels before applying this generator's units to real scenario files.

## Important limitations

1. Synthetic data is an engineering testbed, not clinically representative tremor data.
2. The alpha label is optimal only for the stated cost function, window, horizon, candidate grid and filter initialization.
3. A low alpha prediction error does not prove lower cursor error. Evaluate filtered trajectories and task outcomes too.
4. Public datasets without intended-path ground truth cannot directly produce supervised optimal-alpha targets.
5. The runtime policy currently accepts the prototype feature dictionary. Align its schema with Person 2 before integrating it.
6. The main app and `dsp/adaptive_filter.py` are not changed by this bundle.
