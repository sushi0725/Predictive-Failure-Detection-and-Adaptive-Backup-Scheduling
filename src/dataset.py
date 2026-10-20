
import numpy as np
import pandas as pd
import config
from db_utils import get_engine

def load_labelled_features():
    query = """
        SELECT *
        FROM features
        WHERE fails_within_horizon IN (0, 1)
        ORDER BY feature_date, serial_number
    """
    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        raise ValueError(
            "No valid prediction labels found. Run src/build_features.py first."
        )

    df["feature_date"] = pd.to_datetime(df["feature_date"], errors="coerce")
    df[config.LABEL_COLUMN] = pd.to_numeric(
        df[config.LABEL_COLUMN], errors="coerce"
    )
    df = df.dropna(subset=["feature_date", config.LABEL_COLUMN])
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].astype(int)

    if not df[config.LABEL_COLUMN].isin([0, 1]).all():
        raise ValueError("Failure labels must contain only 0 or 1.")

    return df

def chronological_split(df, test_fraction=0.20, gap_days=None):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    df = df.copy()
    df["feature_date"] = pd.to_datetime(df["feature_date"])
    dates = np.sort(df["feature_date"].dt.normalize().unique())

    if len(dates) < 2:
        raise ValueError(
            "At least two distinct dates are required for a chronological split."
        )

    if gap_days is None:
        gap_days = config.HORIZON_DAYS

    cut_idx = int(len(dates) * (1 - test_fraction))
    cut_idx = max(1, min(len(dates) - 1, cut_idx))
    cut_date = pd.Timestamp(dates[cut_idx])

    train_end = cut_date - pd.Timedelta(days=gap_days)

    train = df[df["feature_date"] < train_end].copy()
    test = df[df["feature_date"] >= cut_date].copy()

    if train.empty or test.empty:
        raise ValueError(
            "The chronological split produced an empty partition. "
            "Use more telemetry dates or review the prediction horizon."
        )

    return train, test

def class_weights(y):
    y = np.asarray(y)
    n = len(y)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    if positives == 0 or negatives == 0:
        raise ValueError(
            "Training requires both healthy and failed examples."
        )

    return {
        0: n / (2 * negatives),
        1: n / (2 * positives)
    }
