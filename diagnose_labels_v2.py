
import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/processed/training_dataset_aryan.csv"
)

print("\n========== ALPHA AT BOUNDS ==========")

print(
    "At minimum:",
    np.isclose(df.alpha_optimal, 0.12).sum()
)

print(
    "At maximum:",
    np.isclose(df.alpha_optimal, 0.85).sum()
)

print(
    "Percentage at maximum:",
    round(100 * np.isclose(df.alpha_optimal, 0.85).mean(), 2),
    "%"
)

print("\n========== ALPHA BY TREMOR STATUS ==========")

print(
    df.groupby("tremor_active")["alpha_optimal"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(3)
)

print("\n========== ALPHA BY MOVEMENT PHASE ==========")

print(
    df.groupby("phase")["alpha_optimal"]
    .agg(["count", "mean", "median"])
    .round(3)
)

print("\n========== TREMOR-ONLY FEATURE CORRELATIONS ==========")

tremor = df[df["tremor_active"].astype(str).str.lower()
            .isin(["true", "1"])]

cols = [
    "alpha_optimal",
    "oscillation_score",
    "est_amplitude",
    "dominant_frequency",
    "velocity",
    "directional_consistency",
    "tracking_rmse",
    "residual_jitter_rms",
    "lag_penalty_px",
]

print(
    tremor[cols]
    .corr()["alpha_optimal"]
    .sort_values()
    .round(3)
)

print("\n========== HIGH-DIRECTIONAL-CONSISTENCY CASES ==========")

high_direction = df[
    (df.directional_consistency >= 0.8) &
    (df.oscillation_score >= 0.75)
]

print("Number of examples:", len(high_direction))

print(
    high_direction[
        ["alpha_optimal", "oscillation_score",
         "est_amplitude", "velocity",
         "directional_consistency"]
    ].describe().round(3)
)
