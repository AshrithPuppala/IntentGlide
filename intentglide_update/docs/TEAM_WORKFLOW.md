# IntentGlide — Team Workflow & Work Distribution

This document expands the team ownership and 24-hour execution plan. The README is the overall source of truth; update both when ownership or interfaces change.

## 1. Person 1 — Input, Hardware, Data Generation & Sandbox Scenarios

**Owns:** `input/`, `hardware/`, `sandbox/targets.py`, `sandbox/tasks.py`, input/scenario tests.

Tasks:
- Synthetic intended trajectories and configurable disturbance profiles.
- Mouse input adapter with real timestamps.
- Optional ESP32 + ERM motor rig with an appropriate motor driver.
- Ground-truth trajectories and scenario metadata for ML training/evaluation.
- Target geometry/layouts and task definitions (standard, small-target, close-target, ambiguous).
- Keep synthetic input and scenarios usable without hardware.

**Deliverables:** `input/input_provider.py`, `input/synthetic_generator.py`, planned `input/disturbance_profiles.py`, `hardware/esp32_controller.ino`, `sandbox/targets.py`, `sandbox/tasks.py`, `tests/test_input.py`, planned `tests/test_scenarios.py`.

**Handoff:** `InputSample` stream, target geometry, task configuration, and ground-truth scenario metadata.

**Boundary:** P1 defines targets and tasks; P4 draws them in `sandbox/scene.py` and owns the Pygame event loop.

## 2. Person 2 — DSP & Feature Engineering

**Owns:** `dsp/` and DSP tests.

Tasks:
- Preserve timestamps and handle nonuniform real mouse samples; resample explicitly when necessary.
- Expand motion analysis to both X and Y.
- Implement documented spectral, amplitude, velocity, acceleration, jerk and directional features.
- Maintain Butterworth baseline and rule-based adaptive filter.
- Publish a stable feature schema for ML training and inference.
- Test filter state, edge cases, and latency.

**Deliverables:** `dsp/motion_analyzer.py`, planned `dsp/feature_extractor.py` (only if a separate module is useful), `dsp/fixed_filter.py`, `dsp/adaptive_filter.py`, `tests/test_dsp.py`.

**Handoff:** documented feature vector and filtered coordinates.

## 3. Person 3 — ML & Adaptive Intelligence

**Owns:** `ml/`, ML training/evaluation code, processed dataset preparation, ML tests.

Tasks:
- Freeze feature schema with P2.
- Read synthetic traces and ground truth supplied by P1.
- Generate optimal-alpha labels from a documented tracking/jitter/lag objective.
- Train Random Forest Regressor and evaluate on held-out trajectories/scenarios.
- Compare with rule-based adaptive policy.
- Save/load model with joblib.
- Implement bounded `predict_alpha(features)` with validation, rate limiting support, policy-source diagnostics, and heuristic fallback.

**Deliverables:** planned `ml/dataset.py`, `ml/train.py`, `ml/filter_policy.py`, `ml/evaluate.py`, `ml/artifacts/`, `tests/test_ml.py`.

**Handoff:** model artifact, inference interface, feature metadata, and evaluation report.

## 4. Person 4 — Application, Interaction, Integration & Evaluation

**Owns:** `app/`, `sandbox/scene.py`, `interaction/`, `visualization/`, `evaluation/`, integration tests.

Tasks:
- Keep the Pygame application runnable from the start using mocks.
- Render Person 1's target layouts and task scenarios.
- Orchestrate input → features → filter policy → DSP filter → intent/safety → output.
- Keep raw, fixed, heuristic-adaptive, ML-adaptive, and full-system modes available as implemented.
- Own target ranking, intent confidence, ambiguity abstention, bounded assistance, and override integration.
- Display raw/output paths, features, alpha, policy source, target confidence, and controller state.
- Benchmark tracking error, residual jitter, lag, task outcomes, processing time, and end-to-end latency.
- Integrate one module at a time and run regression tests.

**Deliverables:** existing/planned `app/`, `sandbox/scene.py`, `interaction/`, `visualization/`, `evaluation/`, `tests/test_integration.py`.

## 5. Shared contracts and handoffs

```text
P1: InputSample + scenario metadata + Target geometry
             ↓
P2: documented MotionFeatures + DSP outputs
             ↓
P3: alpha_target + ML/heuristic policy source
             ↓
P2: stateful filter applies alpha_target
             ↓
P4: intent/safety + app + visualization + benchmarks
```

Proposed ML interface:

```python
alpha_target = filter_policy.predict_alpha(motion_features)
filtered = adaptive_filter.update(sample, alpha_target=alpha_target)
```

This interface is planned; the current adaptive filter computes alpha internally. Refactor only after P2/P3 agree on the contract.

Shared rules:
- Coordinates: screen pixels. Timestamps: monotonic seconds.
- Never assume live mouse events arrive at a fixed rate.
- Freeze feature order, units, normalization and missing-value rules.
- Keep filter state across samples and reset between independent trials.
- Do not mix processing time with end-to-end latency.
- Review changes to `shared/contracts.py` with all four members.

## 6. 24-hour timeline

| Time | P1 — Input / Hardware / Scenarios | P2 — DSP / Features | P3 — ML | P4 — App / Integration |
|---|---|---|---|---|
| 0–1 h | Agree input/scenario contracts | Agree feature definitions | Agree ML feature/label schema | Contracts, mocks, runnable app shell |
| 1–4 h | Synthetic generator, disturbance profiles, target layouts | FFT analysis and fixed baseline | Dataset/label design | Pygame scene and mock pipeline |
| 4–7 h | Task scenarios, ground truth, mouse adapter | Adaptive filter and feature extraction | Alpha-label generation | Orchestrator and mode selector |
| 7–10 h | Optional ESP32 rig; sample data | Verify features/filter state | Train first Random Forest | Integrate input and DSP |
| 10–12 h | Input/scenario handoff | Feature/DSP handoff | Model inference or fallback | Connect policy and interaction |
| 12–15 h | Test disturbances/scenarios | Tune and validate filters | Evaluate held-out scenarios | Complete end-to-end pipeline |
| 15–18 h | Hardware reliability | Diagnostics and latency | ML-vs-rule comparison | Benchmark dashboard/demo |
| 18–20 h | Bug fixes | Bug fixes | Model freeze/bug fixes | Feature freeze and regression tests |
| 20–22 h | Demo support | Performance measurements | Final model measurements | Presentation and recording |
| 22–24 h | Final tests | Final tests | Final tests | Submission and pitch rehearsal |

If the ML model is not demonstrably better by the evaluation checkpoint, retain the heuristic policy as the default and report ML as experimental. A working, honest baseline is better than an unvalidated model.

## 7. Git workflow

```text
main
├── person1/input-hardware-sandbox
├── person2/dsp-feature-engineering
├── person3/ml-filter-policy
└── person4/app-integration
```

- Keep `main` runnable.
- Use focused commits and pull requests.
- Work in owned files where possible.
- Do not commit secrets, local environments, generated datasets, private recordings, or unvalidated model binaries.
- Keep data generation reproducible with scripts/configuration/seeds.
- Person 4 integrates incrementally and runs integration tests after merges.

## 8. Definition of done

- P1 produces reproducible input/scenarios, targets/tasks, and ground truth.
- P2 delivers causal, documented motion features and working baselines.
- P3 delivers an evaluated model, grouped held-out validation, bounded inference and fallback.
- P4 delivers a runnable app, visualization, intent/safety integration and benchmark logs.
- The full synthetic demo works without ESP32 hardware.
- Synthetic-only results are clearly labelled and not presented as human-study evidence.
