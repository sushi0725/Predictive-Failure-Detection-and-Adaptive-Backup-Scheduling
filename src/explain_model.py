import sys
from pathlib import Path
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import text
from config import FEATURE_COLUMNS
from db_utils import get_engine

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
OUTPUT_PATH = ROOT / "models" / "shap_feature_importance.png"
SAMPLE_SIZE = 1000

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    columns = ", ".join(f'"{column}"' for column in FEATURE_COLUMNS)

    query = text(f"""
        SELECT {columns}
        FROM features
        WHERE fails_within_horizon IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT :sample_size
    """)

    with get_engine().connect() as conn:
        data = pd.read_sql(query, conn, params={"sample_size": SAMPLE_SIZE})

    if data.empty:
        raise RuntimeError("No labeled feature rows were found in the database.")

    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.replace([float("inf"), float("-inf")], float("nan"))
    data = data.dropna(subset=FEATURE_COLUMNS)

    if data.empty:
        raise RuntimeError("No valid SMART feature rows remain after cleaning.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data[FEATURE_COLUMNS])

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    elif getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, -1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        data[FEATURE_COLUMNS],
        plot_type="bar",
        show=False
    )
    plt.title("Global SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Valid samples analyzed: {len(data)}")
    print(f"Features analyzed: {len(FEATURE_COLUMNS)}")
    print(f"SHAP chart saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SHAP analysis failed: {exc}", file=sys.stderr)
        raise
