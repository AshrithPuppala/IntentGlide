
import pandas as pd

path = "data/processed/training_dataset_aryan.csv"
df = pd.read_csv(path)

print("\n========== DATASET ==========")
print("Rows:", len(df))
print("Unique trials:", df["trial_id"].nunique())

print("\n========== ALPHA DISTRIBUTION ==========")
print(
    df["alpha_optimal"].describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
    )
)

features = [
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

print("\n========== CORRELATION WITH ALPHA ==========")
print(
    df[features]
    .corr(numeric_only=True)["alpha_optimal"]
    .sort_values()
    .round(3)
    .to_string()
)

print("\n========== ALPHA BY OSCILLATION ==========")

osc_bins = [-0.01, 0.25, 0.50, 0.75, 1.01]
osc_groups = pd.cut(df["oscillation_score"], osc_bins)

print(
    df.groupby(osc_groups, observed=False)["alpha_optimal"]
    .agg(["count", "mean", "median"])
    .round(3)
    .to_string()
)

print("\n========== ALPHA BY VELOCITY ==========")

velocity_bins = [0, 50, 100, 200, 400, 800, 2000]
velocity_groups = pd.cut(
    df["velocity"],
    velocity_bins,
    include_lowest=True
)

print(
    df.groupby(velocity_groups, observed=False)["alpha_optimal"]
    .agg(["count", "mean", "median"])
    .round(3)
    .to_string()
)

print("\n========== ALPHA BY ESTIMATED AMPLITUDE ==========")

amp_bins = [-0.01, 1, 3, 6, 10, 16, 30]
amp_groups = pd.cut(df["est_amplitude"], amp_bins)

print(
    df.groupby(amp_groups, observed=False)["alpha_optimal"]
    .agg(["count", "mean", "median"])
    .round(3)
    .to_string()
)
