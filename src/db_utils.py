"""
Database helpers shared by every script, the dashboard and (later) the ML code.

    from db_utils import get_engine, get_connection, write_risk_scores, load_latest_risk_scores
"""

from functools import lru_cache

import pandas as pd
import psycopg
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

import config


def _sqlalchemy_url() -> str:
    """SQLAlchemy needs to be told which driver to use. We use psycopg (v3)."""
    url = make_url(config.DATABASE_URL)
    if url.drivername in ("postgresql", "postgres", "postgresql+psycopg2"):
        url = url.set(drivername="postgresql+psycopg")
    return url.render_as_string(hide_password=False)


def _libpq_url() -> str:
    """psycopg itself wants a plain postgresql:// address."""
    url = make_url(config.DATABASE_URL).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


@lru_cache(maxsize=1)
def get_engine():
    """SQLAlchemy engine: use with pandas.read_sql / DataFrame.to_sql."""
    return create_engine(_sqlalchemy_url(), pool_pre_ping=True)


def get_connection() -> psycopg.Connection:
    """A raw psycopg connection: use for COPY, transactions and bulk inserts.
    `with get_connection() as conn:` commits on success and rolls back on error."""
    return psycopg.connect(_libpq_url())


def database_available() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
# Contract with the ML teammate: write predictions HERE.
# ----------------------------------------------------------------------

def write_risk_scores(scores: pd.DataFrame, model_version: str) -> int:
    """
    Save model predictions into the risk_scores table.

    `scores` needs these columns:
        serial_number : str    drive id (must already exist in `drives`)
        as_of_date    : date   the telemetry day the score is based on (features.feature_date)
        risk_score    : float  P(failure within the horizon), between 0 and 1
        top_factors   : str    OPTIONAL, e.g. "smart_197_raw_mean_7d, smart_5_raw_slope_7d"
                               (SHAP's top features, comma-separated)

    Re-running with the same model_version updates the earlier score instead of duplicating it.
    Returns the number of rows written.
    """
    df = scores.copy()
    if "top_factors" not in df:
        df["top_factors"] = None
    rows = [
        (r.serial_number, pd.Timestamp(r.as_of_date).date(), float(r.risk_score), model_version,
         r.top_factors if isinstance(r.top_factors, str) else None)
        for r in df.itertuples(index=False)
    ]
    sql = """
        INSERT INTO risk_scores (serial_number, as_of_date, risk_score, model_version, top_factors)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (serial_number, as_of_date, model_version)
        DO UPDATE SET risk_score = EXCLUDED.risk_score,
                      top_factors = EXCLUDED.top_factors,
                      computed_at = now()
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


# ----------------------------------------------------------------------
# Contract with the dashboard: read the newest score of every drive.
# ----------------------------------------------------------------------

def load_latest_risk_scores():
    """
    Returns (scores, factors) in exactly the shape decision_layer.decide_batch() expects:
        scores  = {serial_number: risk_score}
        factors = {serial_number: ["feature a", "feature b", ...]}   (only where available)
    """
    df = pd.read_sql("SELECT serial_number, risk_score, top_factors FROM v_latest_risk", get_engine())
    scores = {r.serial_number: float(r.risk_score) for r in df.itertuples(index=False)}
    factors = {
        r.serial_number: [f.strip() for f in r.top_factors.split(",") if f.strip()]
        for r in df.itertuples(index=False)
        if isinstance(r.top_factors, str) and r.top_factors.strip()
    }
    return scores, factors
