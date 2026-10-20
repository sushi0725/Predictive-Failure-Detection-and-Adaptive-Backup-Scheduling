
import psycopg
from config import DATABASE_URL, HORIZON_DAYS

SMART_COLUMNS = [
    "smart_1_raw", "smart_3_raw", "smart_4_raw", "smart_5_raw",
    "smart_7_raw", "smart_9_raw", "smart_10_raw", "smart_12_raw",
    "smart_187_raw", "smart_188_raw", "smart_189_raw", "smart_190_raw",
    "smart_192_raw", "smart_193_raw", "smart_194_raw", "smart_197_raw",
    "smart_198_raw", "smart_199_raw", "smart_240_raw", "smart_241_raw",
    "smart_242_raw"
]

def main():
    if HORIZON_DAYS < 1:
        raise ValueError("HORIZON_DAYS must be at least 1.")

    dsn = DATABASE_URL.replace(
        "postgresql+psycopg://", "postgresql://", 1
    )
    smart_columns = ", ".join(SMART_COLUMNS)
    horizon_interval = f"INTERVAL '{HORIZON_DAYS} days'"

    sql = f"""
        INSERT INTO features (
            telemetry_id, disk_id, serial_number, model,
            timestamp, feature_date, {smart_columns},
            failure, fails_within_horizon
        )
        WITH ordered AS (
            SELECT
                t.telemetry_id,
                t.disk_id,
                t.serial_number,
                t.model,
                t.timestamp,
                t.date,
                t.failure,
                {", ".join("t." + c for c in SMART_COLUMNS)},
                MAX(t.timestamp) OVER (
                    PARTITION BY t.serial_number
                ) AS last_timestamp,
                MAX(t.failure) OVER (
                    PARTITION BY t.serial_number
                    ORDER BY t.timestamp
                    RANGE BETWEEN INTERVAL '1 microsecond' FOLLOWING
                    AND {horizon_interval} FOLLOWING
                ) AS future_failure
            FROM smart_telemetry t
            WHERE t.serial_number IS NOT NULL
              AND t.timestamp IS NOT NULL
        )
        SELECT
            telemetry_id,
            disk_id,
            serial_number,
            model,
            timestamp,
            timestamp AS feature_date,
            {smart_columns},
            failure,
            CASE
                WHEN last_timestamp >= timestamp + {horizon_interval}
                THEN COALESCE(future_failure, 0)
                ELSE NULL
            END AS fails_within_horizon
        FROM ordered
    """

    print("=== Backblaze Feature Engineering ===")
    print(f"Prediction horizon: {HORIZON_DAYS} days")
    print("Building features inside PostgreSQL; please wait...")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM smart_telemetry")
            telemetry_count = cur.fetchone()[0]
            if telemetry_count == 0:
                raise ValueError("No telemetry records found.")

            print(f"Source telemetry records: {telemetry_count:,}")

            # No tables reference features, as verified earlier.
            # DELETE and INSERT are in the same transaction.
            cur.execute("DELETE FROM features")
            cur.execute(sql)

            cur.execute("""
                SELECT
                    COUNT(*),
                    COUNT(*) FILTER (
                        WHERE fails_within_horizon IS NOT NULL
                    ),
                    COUNT(*) FILTER (
                        WHERE fails_within_horizon = 1
                    ),
                    COUNT(*) FILTER (
                        WHERE fails_within_horizon = 0
                    ),
                    COUNT(*) FILTER (
                        WHERE fails_within_horizon IS NULL
                    )
                FROM features
            """)
            total, labeled, positive, negative, unlabeled = cur.fetchone()

            if labeled == 0 or positive == 0 or negative == 0:
                raise ValueError(
                    "Feature labels do not contain both classes. "
                    "The transaction will be rolled back."
                )

            print(f"Feature rows: {total:,}")
            print(f"Labeled rows: {labeled:,}")
            print(f"Future-failure positives: {positive:,}")
            print(f"Healthy negatives: {negative:,}")
            print(f"Unlabeled rows: {unlabeled:,}")

    print("[+] Feature engineering completed successfully.")

if __name__ == "__main__":
    main()
