"""
STEP 1 -- load Backblaze Drive Stats into PostgreSQL.
Maps data to the project tables: `disks`, `storage_nodes`, and `smart_telemetry`.

    python src/load_backblaze.py --data data/raw
    python src/load_backblaze.py --data data/raw --max-files 1     # trial run, newest day only
    python src/load_backblaze.py --data data/raw --healthy-sample 0.05

What it does, per daily CSV (newest day first):
  1. reads only the columns we need (the files have ~500, mostly empty);
  2. keeps EVERY drive that has failed, plus a random `--healthy-sample` share of the others
     (chosen by a hash of the serial number, so a drive is either kept for all days or none);
  3. turns impossible numbers into NULL (negative or beyond what a BIGINT can hold);
  4. bulk-loads with COPY into staging tables, then INSERT ... ON CONFLICT DO UPDATE/NOTHING,
     mapping physical locations to `storage_nodes`, drive assets to `disks`, and daily health
     metrics to `smart_telemetry`.
"""

import argparse
import io
import re
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import config
from db_utils import get_connection

DATE_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.csv$")
BASE_COLS = ["date", "serial_number", "model", "capacity_bytes", "failure"]
LOCATION_COLS = ["datacenter", "cluster_id", "vault_id", "pod_id", "pod_slot_num"]
BIGINT_LIMIT = 9.0e18  # anything larger (or negative) cannot be a real counter


# ----------------------------------------------------------------------
# Finding the daily files (inside a zip, or in a folder)
# ----------------------------------------------------------------------

def list_sources(path: Path):
    """Return [(date_string, open_function)] for every YYYY-MM-DD.csv, newest first."""
    found = []
    if path.is_file() and path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        for name in zf.namelist():
            base = name.rsplit("/", 1)[-1]
            if name.startswith("__MACOSX") or base.startswith("._"):
                continue
            m = DATE_FILE.match(base)
            if m:
                found.append((m.group(1), (lambda n=name: zf.open(n))))
    elif path.is_dir():
        for p in path.rglob("*.csv"):
            m = DATE_FILE.match(p.name)
            if m:
                found.append((m.group(1), (lambda p=p: open(p, "rb"))))
    else:
        raise SystemExit(f"--data must be a .zip file or a folder of CSVs, got: {path}")
    found.sort(key=lambda x: x[0], reverse=True)
    return found


# ----------------------------------------------------------------------
# Reading and cleaning one day
# ----------------------------------------------------------------------

def to_clean_int(series: pd.Series) -> pd.Series:
    """Text/float -> nullable integer. Negative or absurdly large values become NULL."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.where((s >= 0) & (s <= BIGINT_LIMIT))
    return s.round().astype("Int64")


def read_day(opener) -> pd.DataFrame:
    with opener() as f:
        header = pd.read_csv(f, nrows=0).columns.tolist()
    missing = [c for c in BASE_COLS if c not in header]
    if missing:
        raise SystemExit(f"CSV is missing expected columns {missing}. First columns found: {header[:12]}")
    wanted = [c for c in BASE_COLS + LOCATION_COLS + config.SMART_COLS if c in header]
    with opener() as f:
        df = pd.read_csv(f, usecols=wanted, dtype={"serial_number": str, "model": str, "datacenter": str})

    df["record_date"] = df["date"].astype(str).str.slice(0, 10)
    df["failure"] = pd.to_numeric(df["failure"], errors="coerce").fillna(0).astype(int)
    df["capacity_bytes"] = to_clean_int(df["capacity_bytes"]).where(lambda s: s > 0)
    for col in config.SMART_COLS:                       # a column absent from this file = all NULL
        df[col] = to_clean_int(df[col]) if col in df else pd.array([pd.NA] * len(df), dtype="Int64")
    for col in ["cluster_id", "vault_id", "pod_id", "pod_slot_num"]:
        df[col] = to_clean_int(df[col]) if col in df else pd.array([pd.NA] * len(df), dtype="Int64")
    if "datacenter" not in df:
        df["datacenter"] = pd.NA
    return df


def bucket(serial: str) -> int:
    """Stable 0..9999 number for a serial (Python's own hash() changes every run)."""
    return zlib.crc32(serial.encode("utf-8")) % 10_000


def copy_frame(cur, table: str, df: pd.DataFrame, columns: list[str]) -> None:
    buf = io.StringIO()
    df[columns].to_csv(buf, index=False, header=False, na_rep="")   # empty = NULL in COPY csv
    buf.seek(0)
    with cur.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv)") as cp:
        while chunk := buf.read(1 << 20):
            cp.write(chunk)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

STAGE_SQL = f"""
CREATE TEMP TABLE IF NOT EXISTS stage_nodes (
    node_id TEXT, datacenter TEXT, cluster_id TEXT, vault_id TEXT, pod_id TEXT
) ON COMMIT DELETE ROWS;

CREATE TEMP TABLE IF NOT EXISTS stage_disks (
    serial_number TEXT, model TEXT, capacity_bytes BIGINT,
    node_id TEXT, pod_slot_num TEXT, record_date DATE, failure INTEGER
) ON COMMIT DELETE ROWS;

CREATE TEMP TABLE IF NOT EXISTS stage_telemetry (
    serial_number TEXT, record_date DATE, failure INTEGER,
    {', '.join([f'{col} BIGINT' for col in config.SMART_COLS])}
) ON COMMIT DELETE ROWS;
"""

NODE_COLS = ["node_id", "datacenter", "cluster_id", "vault_id", "pod_id"]
DISK_COLS = ["serial_number", "model", "capacity_bytes", "node_id", "pod_slot_num", "record_date", "failure"]
TELEMETRY_COLS = ["serial_number", "record_date", "failure"] + config.SMART_COLS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="a quarterly .zip, or a folder holding YYYY-MM-DD.csv files")
    ap.add_argument("--healthy-sample", type=float, default=0.02,
                    help="share of never-failed drives to keep (default 0.02 = 2%%; 1.0 = all)")
    ap.add_argument("--models", help="comma-separated model names to keep, e.g. ST4000DM000,HGST HMS5C4040BLE640")
    ap.add_argument("--max-files", type=int, help="only the newest N daily files (for a quick trial)")
    args = ap.parse_args()
    if not 0 < args.healthy_sample <= 1:
        raise SystemExit("--healthy-sample must be between 0 (exclusive) and 1")
    keep_models = {m.strip() for m in args.models.split(",")} if args.models else None

    sources = list_sources(Path(args.data))
    if not sources:
        raise SystemExit(f"No YYYY-MM-DD.csv files found in {args.data}")
    if args.max_files:
        sources = sources[: args.max_files]
    print(f"{len(sources)} daily files, {sources[-1][0]} .. {sources[0][0]}  |  "
          f"keeping all failed drives + {args.healthy_sample:.1%} of healthy ones")

    failed: set[str] = set()
    cutoff = int(args.healthy_sample * 10_000)
    total_new, t_start = 0, time.time()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(STAGE_SQL)
            # Drives that failed in data loaded earlier keep their history across runs
            cur.execute("SELECT DISTINCT serial_number FROM smart_telemetry WHERE failure = 1")
            failed.update(row[0] for row in cur.fetchall())
        conn.commit()
        if failed:
            print(f"{len(failed):,} failed drives already in the database: their older history is kept as well")

        for i, (day, opener) in enumerate(sources, 1):
            t0 = time.time()
            df = read_day(opener)
            rows_in = len(df)
            if keep_models:
                df = df[df["model"].isin(keep_models)]

            failed.update(df.loc[df["failure"] == 1, "serial_number"])
            sampled = {s for s in df["serial_number"].unique() if bucket(s) < cutoff}
            df = df[df["serial_number"].isin(sampled) | df["serial_number"].isin(failed)]

            if df.empty:
                print(f"[{i}/{len(sources)}] {day}: nothing kept")
                continue

            # Construct node_id from physical topology
            has_dc = df["datacenter"].notna()
            df["node_id"] = np.where(
                has_dc,
                df["datacenter"].astype(str) + "-" + df["cluster_id"].fillna(0).astype(str) + "-" +
                df["vault_id"].fillna(0).astype(str) + "-" + df["pod_id"].fillna(0).astype(str),
                None
            )

            # 1. Distinct storage nodes
            nodes_df = df.loc[df["node_id"].notna(), NODE_COLS].drop_duplicates("node_id")

            # 2. Distinct disks metadata
            disks_df = (df.sort_values("failure", ascending=False)
                          .drop_duplicates("serial_number")[DISK_COLS])

            # 3. Time-series SMART telemetry readings
            telemetry_df = df[TELEMETRY_COLS]

            with conn.cursor() as cur:
                if not nodes_df.empty:
                    copy_frame(cur, "stage_nodes", nodes_df, NODE_COLS)
                    cur.execute("""
                        INSERT INTO storage_nodes (node_id, datacenter, cluster_id, vault_id, pod_id)
                        SELECT DISTINCT node_id, datacenter, cluster_id, vault_id, pod_id
                        FROM stage_nodes
                        WHERE node_id IS NOT NULL AND datacenter IS NOT NULL
                        ON CONFLICT (node_id) DO NOTHING
                    """)

                copy_frame(cur, "stage_disks", disks_df, DISK_COLS)
                cur.execute("""
                    INSERT INTO disks (serial_number, model, capacity_bytes, node_id, pod_slot_num, first_seen_date, status, failure_date)
                    SELECT s.serial_number, s.model, COALESCE(s.capacity_bytes, 0), s.node_id, s.pod_slot_num, s.record_date,
                           CASE WHEN s.failure = 1 THEN 'failed' ELSE 'active' END,
                           CASE WHEN s.failure = 1 THEN s.record_date ELSE NULL END
                    FROM stage_disks s
                    ON CONFLICT (serial_number) DO UPDATE SET
                        first_seen_date = LEAST(disks.first_seen_date, EXCLUDED.first_seen_date),
                        status = CASE WHEN EXCLUDED.status = 'failed' THEN 'failed' ELSE disks.status END,
                        failure_date = COALESCE(EXCLUDED.failure_date, disks.failure_date),
                        node_id = COALESCE(EXCLUDED.node_id, disks.node_id),
                        pod_slot_num = COALESCE(EXCLUDED.pod_slot_num, disks.pod_slot_num)
                """)

                copy_frame(cur, "stage_telemetry", telemetry_df, TELEMETRY_COLS)
                cols_joined = ", ".join(TELEMETRY_COLS)
                cur.execute(f"""
                    INSERT INTO smart_telemetry ({cols_joined})
                    SELECT {cols_joined} FROM stage_telemetry
                    ON CONFLICT (serial_number, record_date) DO NOTHING
                """)
                new_rows = cur.rowcount
            conn.commit()
            total_new += new_rows
            print(f"[{i}/{len(sources)}] {day}: read {rows_in:,} rows, kept {len(df):,}, "
                  f"inserted {new_rows:,} new into smart_telemetry ({time.time() - t0:.1f}s)")

        with conn.cursor() as cur:
            cur.execute("""
                SELECT (SELECT COUNT(*) FROM disks),
                       (SELECT COUNT(*) FROM smart_telemetry),
                       (SELECT COUNT(*) FROM disks WHERE status = 'failed'),
                       (SELECT COUNT(DISTINCT model) FROM disks),
                       (SELECT COUNT(*) FROM storage_nodes),
                       (SELECT MIN(record_date) FROM smart_telemetry),
                       (SELECT MAX(record_date) FROM smart_telemetry)
            """)
            n_drives, n_rows, n_failed, n_models, n_nodes, d_min, d_max = cur.fetchone()

    print(f"\nDone in {time.time() - t_start:.0f}s. This run added {total_new:,} readings.")
    print(f"Database now holds: {n_rows:,} readings in smart_telemetry | {n_drives:,} disks ({n_failed:,} failed) | "
          f"{n_models} models | {n_nodes} storage nodes | {d_min} .. {d_max}")
    if n_failed == 0:
        print("WARNING: no failed drives yet. A model cannot learn from zero failures; "
              "load more days (or a full quarter).")


if __name__ == "__main__":
    main()
