# IntentGlide ML

IntentGlide keeps filtering and target-intent learning independent. Filter policies predict EMA alpha or Butterworth cutoff using the shared causal motion features. The intent policy separately ranks candidate targets. Personalisation remains bounded, interpretable statistics; it is not another model and does not retrain at runtime.

See [Personalization and ML workflow](../docs/PERSONALIZATION_AND_ML.md) for reproducible commands, split rules, artifact promotion and runtime behavior, and [Data schemas](../docs/DATA_SCHEMAS.md) for dataset fields. The initial experiment is recorded in [Experiment report](../docs/EXPERIMENT_REPORT.md).

Training writes only to candidate directories. Runtime policies accept only artifacts carrying `status=evaluated_candidate` and the exact current feature schema. Target promotion additionally requires a matching SHA-256 evaluation report; synthetic-only or insufficient-session runs do not authorize promotion. Legacy artifacts without this validation metadata fall back to heuristics.

All synthetic results are development checks, not real-user or clinical evidence. Calibration initializes local profile statistics but does not train a per-user model. Offline training and evaluation are separate from interaction.
