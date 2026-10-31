# IntentGlide — GitHub Folder Structure

This is the target layout for the ML + DSP version. **Existing** marks files currently in the repository scaffold; **NEW / PLANNED** marks additions for the next development stage.

```text
IntentGlide/
├── README.md
├── START_HERE.md
├── MANIFEST.txt
├── requirements.txt
├── .gitignore
├── app/                         # P4: application and orchestration
│   ├── main.py                  # Existing
│   └── orchestrator.py          # Existing; planned ML integration
├── input/                       # P1: input and scenario data
│   ├── input_provider.py        # Existing
│   ├── synthetic_generator.py   # Existing
│   └── disturbance_profiles.py  # NEW / PLANNED
├── hardware/                    # P1: optional physical disturbance rig
│   └── esp32_controller.ino    # Existing starter firmware
├── sandbox/                     # P1 scenario data; P4 rendering
│   ├── targets.py               # Existing; P1 owns geometry/layouts
│   ├── tasks.py                 # Existing; P1 owns task definitions
│   └── scene.py                 # Existing; P4 owns Pygame rendering
├── dsp/                         # P2: preprocessing and signal processing
│   ├── motion_analyzer.py       # Existing baseline
│   ├── feature_extractor.py    # NEW / PLANNED (optional split)
│   ├── fixed_filter.py          # Existing Butterworth baseline
│   └── adaptive_filter.py      # Existing heuristic smoother; refactor for alpha input
├── ml/                          # P3: trained filter policy
│   ├── __init__.py              # NEW / PLANNED
│   ├── README.md
│   ├── dataset.py               # NEW / PLANNED
│   ├── train.py                 # NEW / PLANNED
│   ├── filter_policy.py         # NEW / PLANNED
│   ├── evaluate.py              # NEW / PLANNED
│   └── artifacts/               # Local model artifacts; normally ignored by Git
├── data/                        # P1 generates; P3 prepares
│   ├── README.md
│   ├── generated/               # Local synthetic traces; ignored by Git
│   └── processed/               # Local feature/label tables; ignored by Git
├── interaction/                 # P4: target intent and safety
│   ├── target_analyzer.py       # Existing
│   ├── intent_controller.py     # Existing
│   ├── assistance.py            # Existing
│   └── interaction_safety.py    # Existing
├── visualization/               # P4: display and diagnostics
│   ├── cursor_canvas.py         # Existing
│   ├── diagnostics.py           # Existing
│   └── dashboard.py             # Existing
├── evaluation/                  # P4: whole-system benchmarks
│   ├── benchmark.py             # Existing
│   ├── metrics.py               # Existing
│   └── results/                 # Generated results; avoid private outputs
├── shared/                      # ALL: shared contracts and mocks
│   ├── contracts.py             # Existing; update only by team agreement
│   └── mocks.py                 # Existing
└── tests/                       # ALL: module and integration tests
    ├── test_input.py            # Existing
    ├── test_scenarios.py         # NEW / PLANNED
    ├── test_dsp.py               # Existing
    ├── test_ml.py                # NEW / PLANNED
    ├── test_intent.py            # Existing
    └── test_integration.py       # Existing
```

## Ownership

| Person | Folders/files | Workstream |
|---|---|---|
| P1 | `input/`, `hardware/`, `sandbox/targets.py`, `sandbox/tasks.py` | Input capture, synthetic disturbances, ground-truth traces, ESP32, target layouts and task scenarios |
| P2 | `dsp/` | Preprocessing, feature engineering, Butterworth and heuristic adaptive filters |
| P3 | `ml/`, processed dataset preparation | Random Forest filter policy, alpha labels, training, evaluation and inference |
| P4 | `app/`, `sandbox/scene.py`, `interaction/`, `visualization/`, `evaluation/` | Pygame rendering, intent/safety integration, orchestration and whole-system benchmarking |
| All | `shared/`, `tests/` | Contracts, mocks, unit tests and integration tests |

## Sandbox ownership boundary

Person 1 defines **what** targets and scenarios exist (`targets.py`, `tasks.py`). Person 4 defines **how** they are drawn and connected to the application (`scene.py`, `app/`). P1 should not duplicate the Pygame renderer or the main app event loop.

## Repository hygiene

- Use `.gitkeep` for empty folders that must be tracked.
- Do not commit generated datasets, private recordings, local environment files, caches, or unvalidated model artifacts.
- Keep dataset generation reproducible from scripts, parameters, and seeds.
- Keep `main` runnable and use one branch per person.
- Planned files should be created incrementally as part of the assigned workstream.
