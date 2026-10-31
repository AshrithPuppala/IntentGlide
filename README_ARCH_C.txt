INTENTGLIDE - ARCHITECTURE C PATCH
=================================

This patch adds a separate ML-controlled Butterworth filter mode while keeping
existing RAW, FIXED, adaptive EMA and FULL modes.

CONTENTS
--------
app/orchestrator.py                         updated: add ml_butterworth mode
app/main.py                                 updated: key 5 selects it
visualization/panel.py                      updated: display ML-BW mode/status

dsp/adaptive_butterworth.py                 new adaptive second-order Butterworth
ml/butterworth_policy.py                    new cutoff-in-Hz model policy
ml/butterworth_labeling.py                  new offline candidate cutoff search
ml/butterworth_dataset.py                   new cutoff training dataset builder
ml/train_cutoff.py                          new cutoff regressor training script
ml/synthetic_data_butterworth.py            broader synthetic data generator
benchmark_butterworth.py                    new five-way evaluation benchmark

tests/test_adaptive_butterworth.py           new DSP tests
tests/test_butterworth_ml.py                 new policy/label/dataset tests
tests/test_architecture_c_orchestrator.py    new orchestrator tests
docs/ARCHITECTURE_C.md                      detailed commands/design notes

INSTALL (WINDOWS CMD)
---------------------
1. Commit/stash any uncommitted local changes first.
2. Extract this ZIP into the IntentGlide repository root, preserving folders.
   Allow the three listed existing files to be replaced:
     app/orchestrator.py
     app/main.py
     visualization/panel.py
   All other files are new.
3. Ensure SciPy is installed (already required by the current DSP project):
     pip install -r requirements.txt
4. Generate the wider synthetic dataset:
     python -m ml.synthetic_data_butterworth --trials 500 --sample-rate 120 --min-duration 2 --max-duration 5 --seed 2026 --output data/generated_arch_c
5. Build cutoff labels:
     python -m ml.butterworth_dataset --inputs data/generated_arch_c/synthetic_input.csv --labels data/generated_arch_c/synthetic_labels.csv --output data/processed/butterworth_training_dataset.csv
6. Train the cutoff model:
     python -m ml.train_cutoff --dataset data/processed/butterworth_training_dataset.csv --group-column profile_id
7. Run new tests:
     python -m pytest -q tests/test_adaptive_butterworth.py tests/test_butterworth_ml.py tests/test_architecture_c_orchestrator.py
8. Run the full suite:
     python -m pytest -q
9. Launch:
     python -m app.main
   Press 5 for ML Butterworth; 3 remains ML/heuristic EMA; 4 remains FULL.
10. Run benchmark:
     python benchmark_butterworth.py --frequency 8 --amplitude 12 --duration 3.5

IMPORTANT
---------
The new cutoff model is independent of random_forest_alpha.joblib. It predicts
cutoff_optimal_hz, not alpha. Training and benchmark commands create data/model
artifacts locally; do not commit generated CSVs or .joblib files by default.

Architecture C is an experimental synthetic-development branch. It is not yet
real-user validated. It filters the simulated/in-window cursor only; it does
not control the Windows system pointer.
