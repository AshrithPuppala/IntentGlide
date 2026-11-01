# IntentGlide experiment report

## Smoke experiment (synthetic development data)

- Date: 2026-11-01
- Dataset generation: `.venv` Python, `ml.generate_intent_data --trials 80 --sample-rate 120 --seed 42`
- Data source: synthetic only; 80 trials, 8 synthetic profiles, 17,248 candidate rows.
- Split: grouped by `session_id`, seed 42; 4 sessions yielded 2 train, 1 validation and 1 test session. The test therefore represents one session and is too small to support generalization claims.
- Training report: validation top-1 target accuracy 0.779; held-out top-1 0.791.
- Held-out target ranking: candidate top-1 0.791, acceptance coverage 0.642, wrong-target rate among accepted 0.097; heuristic top-1 0.730, coverage 0.140 and wrong-target rate 0.042.
- Leave-one-user-out folds were computed for the 8 synthetic profiles. They are available in the generated evaluation JSON and are not real-user evidence.
- Evaluation metric gates: passed on this synthetic smoke run. Runtime eligibility: **blocked**. Promotion requires at least three genuine-user profiles, five independent sessions and passing metrics; this run is synthetic-only and has four sessions.
- Limitation: no real-user benefit, assistive effectiveness or clinical safety is inferred from these metrics.

Reproduction:

```cmd
python -m ml.generate_intent_data --trials 80 --sample-rate 120 --seed 42 --output data/generated/intent_candidates.csv
python -m ml.train_target_model --dataset data/generated/intent_candidates.csv --output-dir ml/candidates/intent
python -m ml.evaluate_target --dataset data/generated/intent_candidates.csv --model ml/candidates/intent/intent_target_model.candidate.joblib --report ml/candidates/intent/evaluation.json
```

## Real-user experiment template

Complete this section only from actual recorded, consented tasks. Record dataset hashes and software/model versions. Use separate sessions for chronological holdout, and users for leave-one-user-out. Report per-user deltas as well as aggregate means/medians and uncertainty intervals.

| Condition | Trials | Success | Wrong activation | Completion time | Overshoot | Corrections | Accidental dwell | Dwell time | Magnetic displacement | p50/p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default | — | — | — | — | — | — | — | — | — | — |
| Calibrated | — | — | — | — | — | — | — | — | — | — |
| Statistical adaptation | — | — | — | — | — | — | — | — | — | — |
| Future ML (only if separately evaluated) | — | — | — | — | — | — | — | — | — | — |
