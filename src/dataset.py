"""
For the ML teammate: load the training data from the database and split it correctly.

    from dataset import load_labelled_features, chronological_split, class_weights
    from config import FEATURE_COLUMNS, LABEL_COLUMN

    df = load_labelled_features()
    train, test = chronological_split(df)
    X_train, y_train = train[FEATURE_COLUMNS], train[LABEL_COLUMN]
"""

import numpy as np
import pandas as pd

import config
from db_utils import get_engine


def load_labelled_features() -> pd.DataFrame:
    """All feature rows whose outcome is known (label is 0 or 1), oldest first."""
    df = pd.read_sql(
        "SELECT * FROM features WHERE fails_within_horizon IS NOT NULL ORDER BY feature_date, serial_number",
        get_engine(),
    )
    df["feature_date"] = pd.to_datetime(df["feature_date"])
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].astype(int)
    return df


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.20, gap_days: int | None = None):
    """
    Train on the past, test on the future, never a random shuffle.

    A row's label looks `gap_days` into the future, so the last `gap_days` of the training period
    would 'see' outcomes that fall inside the test period. We therefore leave a gap of that
    length between the two sets. Returns (train, test).
    """
    dates = np.sort(df["feature_date"].unique())
    num_dates = len(dates)
    if gap_days is None:
        gap_days = 0 if num_dates < 14 else config.HORIZON_DAYS

    cut_idx = max(1, min(num_dates - 1, int(num_dates * (1 - test_fraction))))
    cut = dates[cut_idx]            # first day of the test period

    train = df[df["feature_date"] < cut - pd.Timedelta(days=gap_days)]
    if train.empty and gap_days > 0:
        train = df[df["feature_date"] < cut]

    test = df[df["feature_date"] >= cut]
    return train, test


def class_weights(y) -> dict:
    """'Balanced' weights for the rare failure class: {0: w0, 1: w1}."""
    y = np.asarray(y)
    n, n_pos = len(y), int((y == 1).sum())
    if n_pos in (0, n):
        return {0: 1.0, 1: 1.0}
    return {0: n / (2 * (n - n_pos)), 1: n / (2 * n_pos)}
