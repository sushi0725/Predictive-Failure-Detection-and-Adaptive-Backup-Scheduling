"""
Central settings. Every other script imports from here, so there is exactly
one place to change a setting or the list of columns.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # reads DATABASE_URL etc. from the ".env" file in the project root

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://backup_user:change_me@localhost:5432/backup_db",
)

# ----------------------------------------------------------------------
# Prediction horizon: label = "this drive fails within the next N days".
# The whole team must use the SAME value: the model's risk score means
# P(fail within N days) and the decision tiers are built on that meaning.
# ----------------------------------------------------------------------
HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", "14"))

# Backblaze gives ONE row per drive per day, so "24 hours" is simply the current
# row and the rolling windows are measured in days.
ROLLING_WINDOW_DAYS = 7   # changing this also means renaming the *_7d columns in db/schema.sql
MIN_HISTORY_DAYS = 7      # a drive needs a week of history before it gets feature rows
FFILL_LIMIT_DAYS = 7      # forward-fill a missing reading for at most this many rows

# ----------------------------------------------------------------------
# S.M.A.R.T. attributes stored in telemetry_readings (raw values only).
# This list MUST match the smart_*_raw columns in db/schema.sql.
# ----------------------------------------------------------------------
SMART_ATTRS = [1, 3, 4, 5, 7, 9, 10, 12, 187, 188, 189, 190, 192, 193,
               194, 197, 198, 199, 240, 241, 242]
SMART_COLS = [f"smart_{n}_raw" for n in SMART_ATTRS]

# Features used by the model (stored in the `features` table).
CURRENT_SMART = [5, 9, 12, 187, 188, 192, 193, 197, 198]   # today's value
ROLLING_SMART = [5, 187, 188, 197, 198]                    # + mean / slope / delta over the window
ROLLING_STATS = [f"mean_{ROLLING_WINDOW_DAYS}d",
                 f"slope_{ROLLING_WINDOW_DAYS}d",
                 f"delta_{ROLLING_WINDOW_DAYS}d"]

FEATURE_COLUMNS = (
    [f"smart_{n}_raw" for n in CURRENT_SMART]
    + ["temperature_c"]
    + [f"smart_{n}_raw_{s}" for n in ROLLING_SMART for s in ROLLING_STATS]
)
LABEL_COLUMN = "fails_within_horizon"

# Values that are physically impossible become NULL ("drop obviously bad sensor readings").
# Extend this after you look at the real data (see db/queries.sql, section 2).
IMPOSSIBLE_ABOVE = {
    9: 200_000,   # power-on hours: 200,000 h is about 23 years
}
TEMPERATURE_RANGE_C = (0, 100)
