
from functools import lru_cache
import pandas as pd
import psycopg
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
import config


def _sqlalchemy_url() -> str:
    url = make_url(config.DATABASE_URL)
    if url.drivername in ("postgresql", "postgres", "postgresql+psycopg2"):
        url = url.set(drivername="postgresql+psycopg")
    return url.render_as_string(hide_password=False)


def _libpq_url() -> str:
    url = make_url(config.DATABASE_URL).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(_sqlalchemy_url(), pool_pre_ping=True)


def get_connection() -> psycopg.Connection:
    return psycopg.connect(_libpq_url())


def database_available() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def write_risk_scores(scores: pd.DataFrame, model_version: str) -> int:
    required = {"disk_id", "timestamp", "risk_score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing required risk-score columns: {sorted(missing)}")

    df = scores.copy()
    if "top_factors" not in df.columns:
        df["top_factors"] = None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="raise")
    df["model_version"] = model_version

    df = df[
        ["disk_id", "timestamp", "risk_score", "model_version", "top_factors"]
    ].drop_duplicates(subset=["disk_id", "timestamp"], keep="last")

    rows = list(df.itertuples(index=False, name=None))
    sql = """
        INSERT INTO risk_scores
            (disk_id, timestamp, risk_score, model_version, top_factors)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (disk_id, timestamp)
        DO UPDATE SET
            risk_score = EXCLUDED.risk_score,
            model_version = EXCLUDED.model_version,
            top_factors = EXCLUDED.top_factors
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)

    return len(rows)


def load_latest_risk_scores():
    df = pd.read_sql(
        """
        SELECT serial_number, risk_score, top_factors
        FROM v_latest_risk
        """,
        get_engine()
    )

    scores = {
        row.serial_number: float(row.risk_score)
        for row in df.itertuples(index=False)
    }

    factors = {
        row.serial_number: [
            factor.strip()
            for factor in row.top_factors.split(",")
            if factor.strip()
        ]
        for row in df.itertuples(index=False)
        if isinstance(row.top_factors, str) and row.top_factors.strip()
    }

    return scores, factors
