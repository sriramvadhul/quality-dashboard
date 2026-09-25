"""
Trains a defect classifier and produces a risk-ranked alert list under a fixed
inspection budget -- the same "alert prioritization" framing used in the thesis,
scaled down to a simple, explainable project.

Real fab quality teams can't manually inspect every unit. So instead of just
reporting accuracy, this script answers the operational question:
    "If we can only pull aside the top N riskiest units for inspection,
     how many real defects do we actually catch?"

Steps:
  1. Load the clean dataset from etl.py.
  2. Chronological train/test split (no shuffling -- mirrors production reality
     where you train on the past and predict the future, and avoids leakage).
  3. Train an XGBoost classifier.
  4. Rank the test set by predicted defect probability (the "risk score").
  5. Report precision/recall at a fixed inspection budget (e.g. top 100 units).
  6. Save per-unit risk scores + top contributing sensors for the dashboard.

Usage:
    python3 train_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

CLEAN_PATH = Path(__file__).parent.parent / "outputs" / "clean_secom.csv"
SCORES_PATH = Path(__file__).parent.parent / "outputs" / "risk_scores.csv"
IMPORTANCE_PATH = Path(__file__).parent.parent / "outputs" / "feature_importance.csv"

INSPECTION_BUDGET = 100  # how many units the "quality team" can afford to inspect


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def train_and_score(df: pd.DataFrame):
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    X = df[sensor_cols]
    y = (df["Pass_Fail"] == -1).astype(int)  # 1 = defect, 0 = pass

    # Chronological split: first 75% to train, last 25% to test -- no shuffling.
    split_idx = int(len(df) * 0.75)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    meta_test = df.iloc[split_idx:].reset_index(drop=True)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    risk_scores = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, risk_scores)
    ap = average_precision_score(y_test, risk_scores)
    print(f"Test ROC-AUC: {auc:.4f}")
    print(f"Test Average Precision: {ap:.4f}")

    # --- Alert prioritization: rank by risk, evaluate at a fixed inspection budget ---
    results = meta_test[["Timestamp"]].copy()
    results["actual_defect"] = y_test.reset_index(drop=True)
    results["risk_score"] = risk_scores
    results = results.sort_values("risk_score", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1
    results["flagged_for_inspection"] = results["rank"] <= INSPECTION_BUDGET

    budget = min(INSPECTION_BUDGET, len(results))
    flagged = results.iloc[:budget]
    caught = flagged["actual_defect"].sum()
    total_defects = results["actual_defect"].sum()
    precision_at_budget = caught / budget if budget else 0
    recall_at_budget = caught / total_defects if total_defects else 0
    baseline_rate = total_defects / len(results)
    lift = (precision_at_budget / baseline_rate) if baseline_rate else 0

    print(f"\n--- Alert prioritization @ top {budget} units ---")
    print(f"Precision: {precision_at_budget:.1%}  (of flagged units, how many were real defects)")
    print(f"Recall:    {recall_at_budget:.1%}  (of all real defects, how many we caught)")
    print(f"Lift over random inspection: {lift:.1f}x")

    results.to_csv(SCORES_PATH, index=False)
    print(f"\nSaved per-unit risk scores to {SCORES_PATH}")

    # --- Feature importance for the "why is this flagged" dashboard panel ---
    importance = pd.Series(model.feature_importances_, index=sensor_cols)
    importance = importance.sort_values(ascending=False).head(15).reset_index()
    importance.columns = ["sensor", "importance"]
    importance.to_csv(IMPORTANCE_PATH, index=False)
    print(f"Saved top-15 feature importances to {IMPORTANCE_PATH}")

    return {
        "auc": auc, "ap": ap, "precision_at_budget": precision_at_budget,
        "recall_at_budget": recall_at_budget, "lift": lift, "budget": budget,
    }


def run():
    df = load_clean()
    train_and_score(df)


if __name__ == "__main__":
    run()
