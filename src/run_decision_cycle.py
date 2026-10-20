import os
import sys
from sqlalchemy import create_engine, text
import pandas as pd

# Structural bindings mapping config parameters
from config import DATABASE_URL
import decision_layer

def main():
    print("\n================ STARTING ADAPTIVE BACKUP SCHEDULER LIFECYCLE ================")
    engine = create_engine(DATABASE_URL)
    
    print("[-] Fetching active infrastructure telemetry risk scores from PostgreSQL views...")
    try:
        df = pd.read_sql("SELECT * FROM v_latest_risk;", engine)
    except Exception:
        df = pd.read_sql("SELECT * FROM risk_scores;", engine)

    if df.empty:
        print("[!] Notice: Zero active drive risk profiles found inside storage. Skipping decision step.")
        return

    scores = {}
    factors = {}
    
    for idx, row in df.iterrows():
        serial = str(row["serial_number"])
        raw_score = float(row["risk_score"])
        
        # Self-correcting probability scaling wrapper block
        if raw_score > 1.0:
            normalized_score = raw_score / 100.0
        else:
            normalized_score = raw_score
            
        scores[serial] = normalized_score
        factors[serial] = str(row["top_factors"])

    print(f"[-] Forwarding {len(scores)} tracking entries to the automated triage matrix...")
    
    decisions = decision_layer.decide_batch(scores, factors=factors)
    
    if decisions:
        print(f"[-] Writing {len(decisions)} scheduled backup events rows to database storage tables...")
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE adaptive_backup_events RESTART IDENTITY CASCADE;"))
            
            for item in decisions:
                try:
                    v_serial = getattr(item, 'serial_number', getattr(item, 'serial', None))
                    if v_serial is None:
                        v_serial = item['serial_number'] if isinstance(item, dict) else str(item)
                except Exception:
                    continue

                disk_info = conn.execute(
                    text("SELECT disk_id FROM disks WHERE serial_number = :sn LIMIT 1;"),
                    {"sn": str(v_serial)}
                ).fetchone()
                
                if not disk_info:
                    continue
                    
                v_disk_id = disk_info[0]
                
                # FIXED: Pointed to the actual 'risk_scores' database table to resolve relation does not exist error
                score_info = conn.execute(
                    text("SELECT score_id FROM risk_scores WHERE disk_id = :id ORDER BY timestamp DESC LIMIT 1;"),
                    {"id": v_disk_id}
                ).fetchone()
                
                v_score_id = score_info[0] if score_info else None

                if v_score_id:
                    v_ver = getattr(item, 'policy_version', 'v1.0-custom')
                    v_tier = getattr(item, 'risk_tier', 'UNDEFINED')
                    v_act = getattr(item, 'action', 'Backup Track Enforced')
                    v_urg = getattr(item, 'urgency', 'LOW')
                    v_interval = getattr(item, 'backup_interval_hours', 24)
                    v_due = getattr(item, 'next_backup_due', None)
                    v_reason = getattr(item, 'reason', 'Dynamic Risk Threshold Exceeded')

                    conn.execute(
                        text("""
                            INSERT INTO adaptive_backup_events 
                            (score_id, policy_version, risk_tier, action, urgency, backup_interval_hours, next_backup_due, reason)
                            VALUES (:score, :ver, :tier, :act, :urg, :interval, :due, :reason)
                            ON CONFLICT (score_id, policy_version) DO NOTHING;
                        """),
                        {
                            "score": v_score_id,
                            "ver": str(v_ver),
                            "tier": str(v_tier),
                            "act": str(v_act),
                            "urg": str(v_urg),
                            "interval": int(v_interval) if v_interval else 24,
                            "due": v_due,
                            "reason": str(v_reason)
                        }
                    )
        print("[+] Adaptive backup policies compiled and committed cleanly to PostgreSQL tables.")
    print("[+] Decision life cycle scheduler run executed successfully.\n")

if __name__ == "__main__":
    main()
