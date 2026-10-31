# ML Filter Controller — Development Notes

**Status: planned.** The current repository does not yet contain a trained model or runtime ML inference. Keep the heuristic adaptive filter as the working baseline until this module is implemented and evaluated.

## Objective

Train a lightweight regression model to predict a suitable target smoothing parameter (`alpha_target`) from causal cursor-motion features. The model recommends the filter parameter; DSP applies it. It is not a medical classifier or a future-tremor predictor.

## Initial model

Use `sklearn.ensemble.RandomForestRegressor`. Freeze the feature order, units, window length, preprocessing, random seed and model hyperparameters for each experiment. Consider temporal CNN/GRU models only after evaluating the engineered-feature baseline.

## Pipeline

1. Read synthetic traces with timestamps, observed X/Y and known intended X/Y.
2. Build causal features from the current/past movement window using the P2 feature schema.
3. Offline, evaluate candidate alpha values with the same stateful smoother over a defined subsequent evaluation horizon.
4. Calculate a documented cost such as `w_track * tracking_error + w_jitter * residual_jitter + w_lag * lag_penalty`.
5. Select the alpha with the lowest objective as that window's regression target.
6. Do not put future samples from the label horizon into the model input features.
7. Split by complete trajectory/scenario/seed, not individual overlapping windows.
8. Train on train data, select on validation data, and evaluate once on a held-out test set.
9. Save the model and feature metadata with joblib.
10. Validate inference, clamp alpha, rate-limit changes, and fall back to the heuristic policy if the model is unavailable or invalid.

## Runtime interface (planned)

```python
alpha_target = filter_policy.predict_alpha(motion_features)
filtered = adaptive_filter.update(sample, alpha_target=alpha_target)
```

The policy should expose whether the active source is `ml` or `heuristic_fallback`. Do not report model confidence unless an uncertainty estimate has been designed and calibrated.

## Suggested files

- `dataset.py`: trace/window assembly, optimal-alpha labels and grouped splitting.
- `train.py`: model fitting, validation metrics and joblib export.
- `filter_policy.py`: model loading, inference validation, alpha bounds and fallback.
- `evaluate.py`: held-out comparisons against fixed and heuristic baselines.

## Evaluation

Compare raw, fixed Butterworth, heuristic adaptive and ML adaptive modes on identical held-out paths. Report tracking error, residual jitter, lag, task outcomes and inference time. Include intentional fast movement and trajectory changes.

## Artifact policy

Keep generated datasets and trained models out of Git by default. Do not commit private recordings or unvalidated `.joblib` files. Store reproducible scripts, configuration, feature metadata and metrics instead.
