# IntentGlide Data Plan

Person 1 generates repeatable synthetic movement scenarios and known intended trajectories. Person 3 assembles processed feature/label tables for ML training. Person 2 owns the feature schema; Person 4 owns end-to-end evaluation.

## Suggested generated trace fields

- `trace_id`, `scenario_id`, `generation_seed`
- `timestamp_s`, `observed_x`, `observed_y`
- `intended_x`, `intended_y` (ground truth for synthetic traces only)
- `source` (`synthetic`, `mouse`, or `hardware`)
- disturbance configuration: frequency, amplitude, axis, phase and noise level
- task type/layout and target metadata where relevant

Do not write synthetic ground-truth fields for real mouse or hardware traces unless a valid independent ground-truth measurement exists. Clearly mark simulated data.

## Ownership

- **P1:** scenario generation, trace reproducibility, intended path and disturbance metadata.
- **P2:** feature definitions, units, windowing, normalization and feature extraction.
- **P3:** processed feature tables, optimal-alpha labels, grouped splits and model input preparation.
- **P4:** evaluation logs and task outcome records.

## Storage and privacy

`generated/` and `processed/` are local/generated data folders and should normally be ignored by Git. Do not commit participant recordings or identifiable data. Keep scripts, configuration, schema documentation and non-sensitive summary metrics under version control.
