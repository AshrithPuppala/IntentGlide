# IntentGlide personalization and ML workflow

## Independent learning components

1. **Adaptive filtering** predicts EMA alpha or Butterworth cutoff from the existing causal motion feature schema (`ml/features.py`). It is independent of target choice. Calibration statistics do not enter this trained regressor today: the filtering dataset has no validated user-level labels for that relationship, so inventing the connection would risk leakage and schema-incompatible artifacts. The shared filter datasets, trial/session grouped training and held-out evaluation remain the source for future filtering improvements.
2. **Target intent** ranks current reachable targets using cursor history, velocity/direction, distance, target geometry, approach rate, tremor summaries and calibrated profile features. Only the instructed target identity supplies positive labels. Runtime inference can abstain if confidence or score margin is too low; filter predictions are never used as target intent.
3. **Personalisation** is an interpretable, bounded statistical engine. Calibration initializes profile statistics and adapts acquisition margin, dwell tolerance/speed, sticky grace, magnetic speed and confidence. Explicit preference feedback tunes only the relevant preference gradually. Cancellation and inferred correction are logged and never treated as negative feedback by themselves.

## Calibration and profile

Press **C** in the sandbox to start 12 guided pointing trials (the API permits 10–15). Follow each highlighted target and physically click it; outside clicks are retained as misses. The local profile is pseudonymous (`local-…`), versioned, validated and stored in `data/user_profiles/default.json`. Calibration produces observations and initial statistics; it does not train an individual model. Use **F11** to export a profile and **F12** to reset it. Calibration traces are written under `data/calibration_sessions/`; interaction events are JSONL under `data/interaction_sessions/`.

Explicit feedback shortcuts: **F1/F3** magnetic too strong/weak, **F4/F5** dwell too short/long, **F6** wrong target corrected, **F7** correct acceptance, and **F8/F9** sticky grace too short/long. Inferred behavior and user feedback are separate records.

## Reproducible commands (Windows CMD, repository root)

```cmd
python -m ml.synthetic_data --trials 400 --sample-rate 120 --duration 3.5 --seed 42
python -m ml.dataset --inputs data/generated/synthetic_input.csv --labels data/generated/synthetic_labels.csv
python -m ml.train --dataset data/processed/training_dataset.csv --output-dir ml/candidates/filtering
python -m ml.evaluate --dataset data/processed/training_dataset.csv --artifact-dir ml/candidates/filtering
python -m ml.evaluate_filter_candidate --kind alpha --candidate ml/candidates/filtering/random_forest_alpha.joblib --dataset data/processed/training_dataset.csv --report ml/candidates/filtering/alpha-evaluation.json
python -m ml.promote_filter_model --candidate ml/candidates/filtering/random_forest_alpha.joblib --report ml/candidates/filtering/alpha-evaluation.json --artifact-dir ml/artifacts
python -m ml.synthetic_data_butterworth --trials 400 --sample-rate 120 --min-duration 2 --max-duration 5 --seed 42 --output data/generated_arch_c
python -m ml.butterworth_dataset --inputs data/generated_arch_c/synthetic_input.csv --labels data/generated_arch_c/synthetic_labels.csv
python -m ml.train_cutoff --dataset data/processed/butterworth_training_dataset.csv --output-dir ml/candidates/butterworth --group-column profile_id
python -m ml.evaluate_filter_candidate --kind cutoff --candidate ml/candidates/butterworth/random_forest_cutoff.joblib --dataset data/processed/butterworth_training_dataset.csv --report ml/candidates/butterworth/cutoff-evaluation.json
python -m ml.promote_filter_model --candidate ml/candidates/butterworth/random_forest_cutoff.joblib --report ml/candidates/butterworth/cutoff-evaluation.json --artifact-dir ml/artifacts
python -m ml.generate_intent_data --trials 250 --sample-rate 120 --seed 42
python -m ml.train_target_model --dataset data/generated/intent_candidates.csv --output-dir ml/candidates/intent
python -m ml.evaluate_target --dataset data/generated/intent_candidates.csv --model ml/candidates/intent/intent_target_model.candidate.joblib --report ml/candidates/intent/evaluation.json
python -m evaluation.personalization --dataset data/processed/personalization_trials.csv --output data/processed/personalization_report.json
python -m pytest -q --basetemp=.pytest-tmp
```

Training creates candidate artifacts only. The target-intent candidate can be promoted with `python -m ml.promote_model --candidate ... --report ...` only when its SHA-matched evaluation authorizes promotion. Runtime rejects candidate, legacy and unvalidated synthetic artifacts. Filtering artifacts use the existing grouped train/evaluate workflow; they must also be validated and promoted before runtime loading. Never describe synthetic development scores as real-user benefit.

## Evaluation protocol and metrics

Rows are grouped by trial/session before splitting, so windows from one trial cannot enter train and test. Target evaluation reports grouped top-1 accuracy, abstention coverage and wrong-target rate among accepted targets, with a heuristic baseline. If there are multiple users it additionally fits leave-one-user-out folds. `evaluation.personalization` compares default, calibrated, statistical-adaptation and future-ML conditions on target acquisition, wrong activation, completion time, overshoot, corrections, accidental dwell, dwell time, magnetic displacement and latency. It reports within-user chronological summaries and descriptive held-out-user summaries; these require recorded real trials to be meaningful.

Future ML-based personalization is intentionally not active. It requires sufficient real-user, consented, session-separated data and passing held-out evaluation. Models are loaded at startup, never retrained during interaction. Session outcomes are saved for offline curation.

## Limits

Synthetic experiments check software and split behavior only. Calibration summaries describe the current local session and are not clinical measurements. Small-button virtual acquisition can rank an enlarged candidate area, but activation still requires actual hitbox entry or a high-confidence instructed intent; overlap is resolved by the ranker and abstention. Real accessibility benefit, population robustness and latency impact remain to be evaluated with users.
