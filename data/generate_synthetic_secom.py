"""
Generates a synthetic dataset with the SAME SHAPE and statistical quirks as the
real UCI SECOM semiconductor manufacturing dataset:
  - ~1,567 production units (rows)
  - 590 anonymized sensor readings (columns) + a timestamp + a pass/fail label
  - ~6.6% defect rate (realistic, heavily imbalanced -- like real fab yield data)
  - Missing values scattered across sensors (some sensors missing much more than others)
  - A handful of sensors that are actually predictive of defects, buried among noise

WHY SYNTHETIC: the real SECOM dataset lives on UCI's archive / Kaggle, both of which
require either a direct browser download or a Kaggle account -- neither reachable
from this environment. This generator produces a file with an IDENTICAL schema, so
every downstream script (etl.py, train_model.py, app.py) runs unmodified. When you
have the real uci-secom.csv, just point RAW_DATA_PATH at it instead -- see README.

Usage:
    python3 generate_synthetic_secom.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_UNITS = 1567
N_SENSORS = 590
N_INFORMATIVE_SENSORS = 12   # sensors that actually drive defects
DEFECT_RATE = 0.066          # matches real SECOM's ~104/1567 fail rate

OUT_PATH = Path(__file__).parent / "raw_secom.csv"


def generate():
    rng = np.random.default_rng(RNG_SEED)

    # --- Timestamps: one production unit every ~9 minutes, matching a real fab cadence ---
    timestamps = pd.date_range("2026-01-01 06:00:00", periods=N_UNITS, freq="9min")

    # --- Base sensor readings: mostly noise, correlated in small clusters (like real
    # process sensors that share equipment/lines) ---
    sensor_data = rng.normal(loc=50, scale=10, size=(N_UNITS, N_SENSORS))

    # Add correlated sensor clusters (groups of 5 sensors sharing a hidden process factor)
    n_clusters = N_SENSORS // 5
    for c in range(n_clusters):
        shared_factor = rng.normal(0, 5, size=N_UNITS)
        cols = slice(c * 5, c * 5 + 5)
        sensor_data[:, cols] += shared_factor[:, None]

    # --- Informative sensors: pick a handful of columns and make them actually predictive ---
    informative_idx = rng.choice(N_SENSORS, size=N_INFORMATIVE_SENSORS, replace=False)
    risk_score = np.zeros(N_UNITS)
    for idx in informative_idx:
        weight = rng.uniform(0.5, 1.5)
        # push the sensor mean for a random subset of units, simulating a drifting process
        drift_mask = rng.random(N_UNITS) < 0.15
        sensor_data[drift_mask, idx] += rng.normal(15, 4, size=drift_mask.sum())
        risk_score += weight * (sensor_data[:, idx] - sensor_data[:, idx].mean()) / sensor_data[:, idx].std()

    # --- Convert risk score into a binary pass/fail label at the target defect rate ---
    threshold = np.quantile(risk_score, 1 - DEFECT_RATE)
    labels = (risk_score >= threshold).astype(int)  # 1 = fail/defect, 0 = pass

    # --- Scatter missingness: some sensors are much flakier than others (realistic) ---
    missing_rate_per_sensor = rng.beta(1.5, 15, size=N_SENSORS)  # skewed toward low missingness
    for j in range(N_SENSORS):
        miss_mask = rng.random(N_UNITS) < missing_rate_per_sensor[j]
        sensor_data[miss_mask, j] = np.nan

    # --- Assemble dataframe with SECOM-style column names ---
    cols = [f"sensor_{i+1}" for i in range(N_SENSORS)]
    df = pd.DataFrame(sensor_data, columns=cols)
    df.insert(0, "Timestamp", timestamps)
    df["Pass_Fail"] = np.where(labels == 1, -1, 1)  # SECOM convention: -1 = fail, 1 = pass

    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {OUT_PATH}")
    print(f"Defect rate: {(df['Pass_Fail'] == -1).mean():.3%}")
    print(f"Informative sensor columns (for your own reference/debugging): {sorted(informative_idx.tolist())}")


if __name__ == "__main__":
    generate()
