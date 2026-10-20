import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_utils import get_engine
from decision_layer import decide, estimate_backup_load

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "models" / "backup_policy_comparison.csv"
HORIZON_DAYS = 7
SAMPLE_LIMIT = 10000

def main():
    query = text("""
        SELECT
            d.serial_number,
            r.risk_score,
            r.timestamp
        FROM risk_scores r
        JOIN disks d ON d.disk_id = r.disk_id
        WHERE r.risk_score >= 0 AND r.risk_score <= 1
        ORDER BY r.timestamp DESC
        LIMIT :sample_limit
    """)

    with get_engine().connect() as conn:
        data = pd.read_sql(
            query, conn, params={"sample_limit": SAMPLE_LIMIT}
        )

    if data.empty:
        raise RuntimeError("No usable risk scores found in the database.")

    data["risk_score"] = pd.to_numeric(data["risk_score"], errors="coerce")
    data = data.dropna(subset=["serial_number", "risk_score"])
    data = data.drop_duplicates(subset=["serial_number"], keep="first")

    if data.empty:
        raise RuntimeError("No valid per-disk risk scores remain.")

    decisions = [
        decide(str(row.serial_number), float(row.risk_score))
        for row in data.itertuples(index=False)
    ]

    load = estimate_backup_load(decisions, horizon_days=HORIZON_DAYS)

    total = len(decisions)
    high_risk = [d for d in decisions if d.risk_score >= 0.50]
    critical = [d for d in decisions if d.risk_score >= 0.80]
    adaptive_ops = sum(
        HORIZON_DAYS * 24 / d.backup_interval_hours for d in decisions
    )
    fixed_ops = total * HORIZON_DAYS
    high_risk_adaptive = sum(
        HORIZON_DAYS * 24 / d.backup_interval_hours for d in high_risk
    )
    high_risk_fixed = len(high_risk) * HORIZON_DAYS

    summary = pd.DataFrame([{
        "evaluation_type": "estimated_policy_simulation",
        "horizon_days": HORIZON_DAYS,
        "disks_evaluated": total,
        "high_risk_disks_score_ge_0_50": len(high_risk),
        "critical_disks_score_ge_0_80": len(critical),
        "adaptive_estimated_operations": round(adaptive_ops, 2),
        "fixed_daily_estimated_operations": round(fixed_ops, 2),
        "overall_operation_change_percent": round(
            100 * (adaptive_ops - fixed_ops) / max(fixed_ops, 1), 2
        ),
        "high_risk_adaptive_estimated_operations": round(high_risk_adaptive, 2),
        "high_risk_fixed_estimated_operations": round(high_risk_fixed, 2),
        "high_risk_operation_change_percent": round(
            100 * (high_risk_adaptive - high_risk_fixed)
            / max(high_risk_fixed, 1), 2
        ) if high_risk else 0.0,
        "limitation": "Estimated operations only; physical backups are not measured."
    }])

    details = pd.DataFrame([{
        "serial_number": d.serial_number,
        "risk_score": d.risk_score,
        "risk_tier": d.risk_tier,
        "action": d.action.value,
        "urgency": d.urgency.value,
        "backup_interval_hours": d.backup_interval_hours,
        "adaptive_estimated_operations_7d":
            round(HORIZON_DAYS * 24 / d.backup_interval_hours, 2),
        "fixed_daily_operations_7d": HORIZON_DAYS,
    } for d in decisions])

    summary.to_csv(OUTPUT_PATH, index=False)
    details.to_csv(OUTPUT_PATH.with_name("backup_policy_disk_details.csv"), index=False)

    print("Adaptive vs fixed-policy evaluation completed.")
    print(f"Disks evaluated: {total}")
    print(f"High-risk disks (score >= 0.50): {len(high_risk)}")
    print(f"Critical disks (score >= 0.80): {len(critical)}")
    print(f"Adaptive estimated operations over {HORIZON_DAYS} days: {adaptive_ops:.2f}")
    print(f"Fixed daily estimated operations over {HORIZON_DAYS} days: {fixed_ops:.2f}")
    print(f"Overall operation change: {summary.iloc[0]['overall_operation_change_percent']:.2f}%")
    print(f"Summary saved: {OUTPUT_PATH}")
    print(f"Per-disk details saved: {OUTPUT_PATH.with_name('backup_policy_disk_details.csv')}")
    print("Note: These are scheduling estimates, not verified physical backups.")

if __name__ == "__main__":
    main()
