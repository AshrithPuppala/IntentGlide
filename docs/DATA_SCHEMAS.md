# Local data schemas

All runtime records are local by default. `profile_id` is pseudonymous; it is not an account identity. Synthetic data is explicitly tagged `data_source=synthetic`. Calibration is tagged `data_source=calibration`. Do not mix or relabel sources when reporting results.

## Target-intent candidate dataset (CSV, schema version 2)

Identity/label columns: `schema_version`, `data_source`, `session_id`, `trial_id`, `profile_id`, `timestamp`, `target_id`, `intended_target_id`, `is_intended`. `is_intended` is one only for the instructed target. It is never generated from the model decision. Feature columns are the ordered `TARGET_FEATURE_NAMES` in `interaction/target_analyzer.py`: movement history and velocity/direction, approach rate, geometry and virtual acquisition membership, plus profile variability/overshoot/speed/confidence features. Preserve the feature-name list in every artifact.

Example (schema illustration only; not a user observation):

```csv
schema_version,data_source,session_id,trial_id,profile_id,timestamp,target_id,intended_target_id,is_intended
2,synthetic,session-example,trial-0001,synthetic-user-00,0.250,button-a,button-a,1
```

The actual CSV also includes every required feature column. Calibration export produces per-sample/per-reachable-target candidates and uses the trial's instructed target as the label.

## Calibration JSONL

One JSON object per instructed trial. Fields include schema/session/profile/trial identifiers; intended and actual selected target; target dimensions and center; ordered samples; dwell result; explicit feedback; and outcome metrics. Each sample records raw, filtered and assisted coordinates, velocity, acceleration, local detrended residual/tremor statistics, acquisition margin, magnetic strength and displacement. Null selected-target is a miss. `explicit_feedback` is only set after an explicit user action.

## Interaction session JSONL

One object per event with `schema_version`, pseudonymous `profile_id`, monotonic timestamp, event name and event-specific fields. Events include mode changes, target decisions, physical/dwell activations, explicit feedback, calibration progress and session end. Cancellation/outcomes are observational records and do not update preferences automatically.

## Personalisation evaluation CSV

Required columns: `profile_id`, `session_id`, `trial_id`, `condition`, `intended_target_id`, `actual_selected_target_id`, and numeric `completion_time_s`, `overshoot_px`, `corrections`, `accidental_dwell`, `dwell_completion_time_s`, `magnetic_displacement_px`, `latency_ms`, `success`, `wrong_target`. Conditions: `default`, `calibrated`, `statistical_adaptation`, or `future_ml`. The future-ML condition is reserved and should not be populated until such a model is eligible for use. Successes/latency measurements must come from recorded trials, never from generated model outputs.

## Profile JSON

`UserProfile` validates schema version, finite bounded settings and local pseudonymous ID before loading/saving. The profile contains user-adjustable preferences, bounded adapted parameters, calibration summary statistics and a monotonically increasing profile version. Reset creates a new default profile; export writes a JSON copy.
