"""
ETL step: raw multi-sensor CSV -> clean, analysis-ready table.

This mirrors the real-world problem stated in the job description almost exactly:
"preparing... coding tasks... assist with development of dashboards and ETL pipelines."

Steps:
  1. Load raw sensor data (590 columns, lots of missingness, no structure yet).
  2. Drop sensors that are missing too often to be trustworthy.
  3. Impute remaining gaps with the median (robust to outliers, standard for sensor data).
  4. Drop near-constant sensors (zero information -- pure noise/dead sensors).
  5. Save a clean parquet/csv ready for feature engineering + modeling.

Usage:
    python3 etl.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "data" / "raw_secom.csv"
CLEAN_PATH = Path(__file__).parent.parent / "outputs" / "clean_secom.csv"

MAX_MISSING_FRACTION = 0.35   # drop a sensor if more than 35% of its readings are missing
MIN_VARIANCE = 1e-3           # drop a sensor if it barely varies at all


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    meta_cols = [c for c in df.columns if not c.startswith("sensor_")]

    # 1. Drop sensors with too much missing data
    missing_frac = df[sensor_cols].isna().mean()
    keep_cols = missing_frac[missing_frac <= MAX_MISSING_FRACTION].index.tolist()
    dropped_missing = len(sensor_cols) - len(keep_cols)

    # 2. Impute remaining gaps with per-sensor median
    df_clean = df[meta_cols + keep_cols].copy()
    df_clean[keep_cols] = df_clean[keep_cols].fillna(df_clean[keep_cols].median())

    # 3. Drop near-constant (zero-information) sensors
    variances = df_clean[keep_cols].var()
    informative_cols = variances[variances > MIN_VARIANCE].index.tolist()
    dropped_constant = len(keep_cols) - len(informative_cols)

    final_cols = meta_cols + informative_cols
    df_final = df_clean[final_cols]

    print(f"Started with {len(sensor_cols)} sensors")
    print(f"Dropped {dropped_missing} sensors for >{MAX_MISSING_FRACTION:.0%} missingness")
    print(f"Dropped {dropped_constant} near-constant sensors")
    print(f"Kept {len(informative_cols)} sensors for modeling")

    return df_final


def run():
    df_raw = load_raw()
    df_clean = clean(df_raw)
    CLEAN_PATH.parent.mkdir(exist_ok=True)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"Saved clean dataset to {CLEAN_PATH} ({df_clean.shape[0]} rows x {df_clean.shape[1]} cols)")


if __name__ == "__main__":
    run()
