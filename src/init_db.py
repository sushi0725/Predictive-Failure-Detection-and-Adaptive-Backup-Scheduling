
import os
import argparse
from sqlalchemy import create_engine, text
from config import DATABASE_URL
from decision_layer import POLICY_TIERS


def main():
    parser = argparse.ArgumentParser(
        description="Infrastructure DB Initialization Engine"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate project database objects (DESTRUCTIVE)"
    )
    args = parser.parse_args()
    engine = create_engine(DATABASE_URL)

    print("[-] Commencing database architecture sync...")

    if args.reset:
        confirmation = input(
            "WARNING: This deletes project database data. Type RESET to continue: "
        )
        if confirmation != "RESET":
            print("Reset cancelled.")
            return

        print("[!] Resetting project database objects...")
        views_to_drop = [
            "vw_urgent_backup_queue",
            "vw_disk_latest_health",
            "v_latest_risk"
        ]
        tables_to_drop = [
            "adaptive_backup_events",
            "model_drift_monitoring",
            "adaptive_backup_policies",
            "risk_scores",
            "features",
            "smart_telemetry",
            "disks",
            "storage_nodes"
        ]

        with engine.begin() as conn:
            for view in views_to_drop:
                conn.execute(text(f"DROP VIEW IF EXISTS {view} CASCADE"))
            for table in tables_to_drop:
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))

        print("[+] Existing project objects dropped.")

    print("[-] Applying db/schema.sql...")
    schema_path = os.path.join("db", "schema.sql")
    with open(schema_path, "r", encoding="utf-8-sig") as file:
        schema_sql = file.read().strip()

    raw_connection = engine.raw_connection()
    try:
        cursor = raw_connection.cursor()
        try:
            if schema_sql:
                cursor.execute(schema_sql)
            raw_connection.commit()
        except Exception:
            raw_connection.rollback()
            raise
        finally:
            cursor.close()
    finally:
        raw_connection.close()

    print("[+] Tables and views established.")

    print("[-] Ensuring backup policies exist...")
    normalized_tiers = []

    if isinstance(POLICY_TIERS, dict):
        for tier, tier_config in POLICY_TIERS.items():
            normalized_tiers.append((tier, tier_config))
    elif isinstance(POLICY_TIERS, list):
        for index, tier_config in enumerate(POLICY_TIERS):
            if isinstance(tier_config, dict):
                tier = tier_config.get("risk_tier", f"TIER_{index}")
                normalized_tiers.append((tier, tier_config))

    with engine.begin() as conn:
        for tier, tier_config in normalized_tiers:
            conn.execute(
                text("""
                    INSERT INTO adaptive_backup_policies
                    (policy_version, risk_tier, threshold_low, threshold_high,
                     action, urgency, backup_interval_hours)
                    SELECT
                    :version, :tier, :low, :high, :action, :urgency, :interval
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM adaptive_backup_policies
                        WHERE policy_version = :version
                          AND risk_tier = :tier
                    )
                """),
                {
                    "version": "v1.0-live",
                    "tier": tier,
                    "low": float(tier_config.get("threshold_low", 0.0)),
                    "high": float(tier_config.get("threshold_high", 1.0)),
                    "action": tier_config.get(
                        "action", "Backup Override Enforced"
                    ),
                    "urgency": tier_config.get("urgency", "LOW"),
                    "interval": int(
                        tier_config.get("backup_interval_hours", 24)
                    )
                }
            )

    engine.dispose()
    print("[+] Database initialization completed successfully.")


if __name__ == "__main__":
    main()
