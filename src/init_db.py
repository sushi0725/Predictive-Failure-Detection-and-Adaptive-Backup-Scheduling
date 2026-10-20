import os
import argparse
from sqlalchemy import create_engine, text
from config import DATABASE_URL
from decision_layer import POLICY_TIERS

def main():
    parser = argparse.ArgumentParser(description="Infrastructure DB Initialization Engine")
    parser.add_argument("--reset", action="store_true", help="Run teardown scripts")
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    print("[-] Commencing database architecture sync...")

    # SESSION 1: Teardown legacy environment tracking contexts cleanly
    if args.reset:
        print("[!] Wiping database catalog via programmatic object inspection...")
        views_to_drop = ["vw_urgent_backup_queue", "vw_disk_latest_health", "v_latest_risk", "smart_telemetry", "risk_scores"]
        for view_name in views_to_drop:
            try:
                with engine.begin() as conn:
                    conn.execute(text(f"DROP VIEW IF EXISTS {view_name} CASCADE;"))
            except Exception:
                pass
        
        tables_to_drop = [
            "model_drift_monitoring", "adaptive_backup_events", "adaptive_backup_policies",
            "risk_scores", "risk_scores_records", "features", "smart_telemetry", "smart_telemetry_records", "disks", "storage_nodes"
        ]
        for table_name in tables_to_drop:
            try:
                with engine.begin() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE;"))
            except Exception:
                pass
                
        print("[+] Database catalog cleaned successfully.")

    # SESSION 2: Independent session block compiling the relational tables 
    print("[-] Constructing schema fields and views layer from db/schema.sql...")
    with engine.connect() as conn:
        raw_conn = conn.connection.driver_connection
        with raw_conn.cursor() as cursor:
            with open(os.path.join("db", "schema.sql"), "r", encoding="utf-8-sig") as f:
                schema_sql = f.read().strip()
            if schema_sql:
                cursor.execute(schema_sql)
        raw_conn.commit()
    print("[+] Operational tables and analytical views established successfully.")

    # SESSION 3: Multi-format policy dictionary/list seeding wrapper engine
    print("[-] Programmatically seeding policies from decision_layer module...")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM adaptive_backup_policies;"))
        
        # FIX: Adapt execution if POLICY_TIERS is configured as a list structure natively
        normalized_tiers = []
        if isinstance(POLICY_TIERS, dict):
            for tier, config in POLICY_TIERS.items():
                config["risk_tier_name"] = tier
                normalized_tiers.append(config)
        elif isinstance(POLICY_TIERS, list):
            for idx, item in enumerate(POLICY_TIERS):
                if isinstance(item, dict):
                    if "risk_tier" not in item:
                        item["risk_tier_name"] = f"TIER_{idx}"
                    else:
                        item["risk_tier_name"] = item["risk_tier"]
                    normalized_tiers.append(item)

        for config in normalized_tiers:
            tier_label = config.get("risk_tier_name", "UNKNOWN_TIER")
            conn.execute(
                text("""
                    INSERT INTO adaptive_backup_policies 
                    (policy_version, risk_tier, threshold_low, threshold_high, action, urgency, backup_interval_hours)
                    VALUES (:version, :tier, :low, :high, :act, :urg, :interval);
                """),
                {
                    "version": "v1.0-live", 
                    "tier": tier_label,
                    "low": float(config.get("threshold_low", 0.0)), 
                    "high": float(config.get("threshold_high", 1.0)),
                    "act": config.get("action", "Backup Override Enforced"), 
                    "urg": config.get("urgency", "LOW"), 
                    "interval": int(config.get("backup_interval_hours", 24))
                }
            )
            
    print("[+] Database initialized successfully.")

if __name__ == "__main__":
    main()
