import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FEATURE_COLUMNS
from db_utils import get_engine

BASELINE_ROWS = 5000
RECENT_ROWS = 1000
P_VALUE_THRESHOLD = 0.01
MODEL_VERSION = "xgb-7d"

def load_samples(conn, columns):
    selected = ", ".join(f'"{column}"' for column in columns)
    query = text(f"""
        SELECT {selected}
        FROM features
        WHERE fails_within_horizon IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT :limit
        OFFSET :offset
    """)

    recent = pd.read_sql(
        query, conn, params={"limit": RECENT_ROWS, "offset": 0}
    )
    baseline = pd.read_sql(
        query, conn, params={"limit": BASELINE_ROWS, "offset": RECENT_ROWS}
    )
    return baseline, recent

def main():
    engine = get_engine()

    with engine.connect() as conn:
        baseline, recent = load_samples(conn, FEATURE_COLUMNS)

    if len(baseline) < 100 or len(recent) < 100:
        raise RuntimeError(
            f"Insufficient samples: baseline={len(baseline)}, recent={len(recent)}"
        )

    results = []

    for feature in FEATURE_COLUMNS:
        base = pd.to_numeric(baseline[feature], errors="coerce")
        curr = pd.to_numeric(recent[feature], errors="coerce")
        base = base.replace([np.inf, -np.inf], np.nan).dropna()
        curr = curr.replace([np.inf, -np.inf], np.nan).dropna()

        if len(base) < 100 or len(curr) < 100:
            print(f"Skipping {feature}: insufficient valid values")
            continue

        test = ks_2samp(base, curr)
        shifted = bool(test.pvalue < P_VALUE_THRESHOLD)

        results.append({
            "feature": feature,
            "baseline_n": len(base),
            "recent_n": len(curr),
            "ks_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "drift_detected": shifted,
        })

    if not results:
        raise RuntimeError("No features had enough valid values for drift testing.")

    drift_detected = any(item["drift_detected"] for item in results)
    evaluated_at = datetime.now()

    insert_sql = text("""
        INSERT INTO model_drift_monitoring
            (evaluated_at, model_version, roc_auc, pr_auc, f1_score,
             baseline_failure_rate, drift_detected)
        VALUES
            (:evaluated_at, :model_version, NULL, NULL, NULL, NULL, :drift_detected)
    """)

    with engine.begin() as conn:
        conn.execute(
            insert_sql,
            {
                "evaluated_at": evaluated_at,
                "model_version": MODEL_VERSION,
                "drift_detected": drift_detected,
            },
        )

    report_path = Path(__file__).resolve().parents[1] / "models" / "drift_report.csv"
    pd.DataFrame(results).to_csv(report_path, index=False)

    print(f"Baseline rows: {len(baseline)}")
    print(f"Recent rows: {len(recent)}")
    print(f"KS p-value threshold: {P_VALUE_THRESHOLD}")
    print("\nPer-feature results:")
    for item in results:
        print(
            f"{item['feature']}: KS={item['ks_statistic']:.4f}, "
            f"p={item['p_value']:.6g}, drift={item['drift_detected']}"
        )

    print(f"\nOverall input drift detected: {drift_detected}")
    print(f"Detailed report saved: {report_path}")
    print("Monitoring result saved to model_drift_monitoring.")

if __name__ == "__main__":
    main()
