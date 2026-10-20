import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import FEATURE_COLUMNS
from db_utils import get_engine

MODEL_PATH = ROOT / "models" / "risk_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
OUTPUT_PATH = ROOT / "models" / "threshold_evaluation.csv"

def main():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)

    model = joblib.load(MODEL_PATH)
    test_start = saved_metrics["test_start_date"]
    test_end = saved_metrics["test_end_date"]

    columns = ", ".join(f'"{column}"' for column in FEATURE_COLUMNS)
    query = text(f"""
        SELECT {columns}, fails_within_horizon
        FROM features
        WHERE timestamp >= :start_date
          AND timestamp <= :end_date
          AND fails_within_horizon IS NOT NULL
        ORDER BY timestamp
    """)

    with get_engine().connect() as conn:
        test_df = pd.read_sql(
            query, conn,
            params={"start_date": test_start, "end_date": test_end}
        )

    if test_df.empty:
        raise RuntimeError("No test rows found for the saved evaluation dates.")

    X = test_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = test_df["fails_within_horizon"].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        raise RuntimeError("The selected test data does not contain both classes.")

    probabilities = model.predict_proba(X)[:, 1]
    rows = []

    for threshold in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = metrics.confusion_matrix(
            y, predictions, labels=[0, 1]
        ).ravel()

        rows.append({
            "threshold": threshold,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "precision": float(metrics.precision_score(
                y, predictions, zero_division=0
            )),
            "recall": float(metrics.recall_score(
                y, predictions, zero_division=0
            )),
            "f1_score": float(metrics.f1_score(
                y, predictions, zero_division=0
            )),
            "predicted_failure_count": int(predictions.sum()),
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Test rows evaluated: {len(y):,}")
    print(f"Actual failures in selected data: {int(y.sum())}")
    print(f"ROC-AUC: {metrics.roc_auc_score(y, probabilities):.4f}")
    print(f"PR-AUC: {metrics.average_precision_score(y, probabilities):.6f}")
    print("\nThreshold comparison:")
    print(result.to_string(index=False, formatters={
        "precision": "{:.4f}".format,
        "recall": "{:.4f}".format,
        "f1_score": "{:.4f}".format,
    }))
    print(f"\nReport saved: {OUTPUT_PATH}")
    print("Existing model and metrics.json were not modified.")

if __name__ == "__main__":
    main()
