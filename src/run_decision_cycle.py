"""
Glue between the database and your teammate's decision_layer.py.

    python src/run_decision_cycle.py
    python src/run_decision_cycle.py --dry-run                 # show what would be logged, write nothing
    python src/run_decision_cycle.py --policy-version tiers-v2 # after you tune POLICY_TIERS

Reads the newest risk score of every drive from the database, asks decision_layer.decide_batch()
what to do, and logs every decision into `backup_events` -- all in ONE transaction, so either every
decision is saved or none is. decision_layer.py itself is unchanged and knows nothing about the database.
"""

import argparse
from collections import Counter

import pandas as pd

from db_utils import get_connection, get_engine
from decision_layer import decide_batch, estimate_backup_load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-version", default="tiers-v1",
                    help="label stored with each decision; bump it when you change POLICY_TIERS")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    latest = pd.read_sql("SELECT score_id, serial_number, risk_score, top_factors FROM v_latest_risk", get_engine())
    if latest.empty:
        raise SystemExit("risk_scores is empty. The model must write scores first "
                         "(try: python src/demo_score_drives.py).")

    scores = dict(zip(latest["serial_number"], latest["risk_score"].astype(float)))
    factors = {
        r.serial_number: [f.strip() for f in r.top_factors.split(",") if f.strip()]
        for r in latest.itertuples(index=False) if isinstance(r.top_factors, str)
    }
    score_id = dict(zip(latest["serial_number"], latest["score_id"].astype(int)))

    decisions = decide_batch(scores, factors=factors)
    print("Decisions by tier:", dict(Counter(d.risk_tier for d in decisions)))
    load = estimate_backup_load(decisions)
    print(f"Backup operations per week: adaptive {load['adaptive_backup_ops']} "
          f"vs fixed daily {load['fixed_daily_backup_ops']}")

    if args.dry_run:
        print("Dry run: nothing written.")
        return

    rows = []
    for d in decisions:
        r = d.to_row()
        rows.append((score_id[d.serial_number], args.policy_version, r["risk_tier"], r["action"],
                     r["urgency"], r["backup_interval_hours"], d.next_backup_due, r["reason"]))

    with get_connection() as conn:                       # one transaction: commit at the end, or roll back on error
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO backup_events (score_id, policy_version, risk_tier, action, urgency,
                                           backup_interval_hours, next_backup_due, reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (score_id, policy_version) DO NOTHING""", rows)
            logged = cur.rowcount
    print(f"Logged {len(rows)} decisions ({'all new' if logged == len(rows) else 'some already logged'}) "
          f"under policy '{args.policy_version}'.")


if __name__ == "__main__":
    main()
