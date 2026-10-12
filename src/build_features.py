"""
STEP 2 -- turn raw daily S.M.A.R.T. readings into model-ready features.

    python src/build_features.py
    python src/build_features.py --horizon 7          # label = fails within 7 days instead of the default

Pipeline (all done in memory with pandas/numpy, then written to the `features` table):
  1. CLEAN   impossible values -> NULL, forward-fill short gaps per drive, extract temperature.
  2. WINDOW  for the key error counters: mean, slope and change over the trailing 7 days.
  3. LABEL   fails_within_horizon = 1 / 0 / NULL (NULL = outcome not known yet).
  4. STORE   the whole `features` table is rebuilt each run, so the result is reproducible.

Class imbalance (failures are rare) is NOT fixed here. The table keeps the true labels and the
script prints the ratio; the model applies class weights while training (see src/dataset.py).
"""

import argparse
import io
import time

import numpy as np
import pandas as pd

import config
from db_utils import get_connection, get_engine


# ----------------------------------------------------------------------
# 1. Rolling window statistics (the "tricky" maths, kept in one tested function)
# ----------------------------------------------------------------------

def rolling_features(code: np.ndarray, day: np.ndarray, values: np.ndarray, window: int):
    """
    For every row, look at the same drive's readings in the trailing `window` calendar days
    (the row itself plus up to window-1 earlier days; missing days are simply absent) and return

        mean   average of the available readings
        slope  least-squares trend, in "units per day" (0 if only one reading)
        delta  today's value minus the oldest available value in the window

    Inputs must be sorted by (code, day). `code` = integer drive id, `day` = integer day number,
    `values` = float array (rows x columns) with NaN where a reading is missing.

    Instead of one slow Python call per row, we loop over the `window` possible lags (0..window-1 rows
    back) and update every row at once. Coordinates are taken relative to the current row (dx <= 0,
    dy = older - current), which keeps the sums small and exact for integer counters.
    """
    n, m = values.shape
    cnt = np.zeros((n, m))
    sdx = np.zeros((n, m))
    sdy = np.zeros((n, m))
    sdxdy = np.zeros((n, m))
    sdxdx = np.zeros((n, m))
    oldest_dy = np.zeros((n, m))
    ref = np.nan_to_num(values, nan=0.0)   # reference point = today's value (0 if today's reading is missing)

    for k in range(min(window, n)):
        cur = slice(k, n)                  # row i
        old = slice(0, n - k)              # row i-k
        gap = day[cur] - day[old]          # calendar days between the two rows
        in_window = (code[cur] == code[old]) & (gap <= window - 1)
        dx = -gap.astype(float)[:, None]
        dy = values[old] - ref[cur]
        ok = in_window[:, None] & ~np.isnan(values[old])

        cnt[cur] += ok
        sdx[cur] += np.where(ok, dx, 0.0)
        sdy[cur] += np.where(ok, dy, 0.0)
        sdxdy[cur] += np.where(ok, dx * dy, 0.0)
        sdxdx[cur] += np.where(ok, dx * dx, 0.0)
        oldest_dy[cur] = np.where(ok, dy, oldest_dy[cur])   # a larger k overwrites: ends as the oldest

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(cnt > 0, ref + sdy / cnt, np.nan)
        denom = cnt * sdxdx - sdx ** 2
        slope = np.where(denom > 0, (cnt * sdxdy - sdx * sdy) / denom, 0.0)
        slope = np.where(cnt > 0, slope, np.nan)
        delta = np.where((cnt > 0) & ~np.isnan(values), -oldest_dy, np.nan)   # needs today's value
    return mean, slope, delta


# ----------------------------------------------------------------------
# 2. Labels
# ----------------------------------------------------------------------

def make_labels(code: np.ndarray, day: np.ndarray, failure: np.ndarray, horizon: int):
    """
    label = 1  the drive fails within `horizon` days (including today)
    label = 0  we know it does not (it fails later, or the data runs on for >= horizon more days)
    label = NaN  unknown: no failure seen and the data ends within `horizon` days
    Also returns a mask of rows AFTER a drive's failure day (should not exist; dropped for safety).
    """
    n = len(code)
    last_day = day.max()
    failed_rows = failure == 1
    first_fail = pd.Series(day[failed_rows]).groupby(code[failed_rows]).min()
    fail_day = pd.Series(code).map(first_fail).to_numpy(dtype=float)     # NaN = never failed in the data
    until = fail_day - day

    label = np.full(n, np.nan)
    has_failure = ~np.isnan(until)
    label[has_failure & (until >= 0) & (until <= horizon)] = 1.0
    label[has_failure & (until > horizon)] = 0.0
    label[~has_failure & (day <= last_day - horizon)] = 0.0
    after_failure = has_failure & (until < 0)
    return label, after_failure


# ----------------------------------------------------------------------
# 3. The whole pipeline as one function (easy to test)
# ----------------------------------------------------------------------

def build_feature_frame(raw: pd.DataFrame, horizon: int, min_history: int) -> pd.DataFrame:
    """raw = rows of smart_telemetry (serial_number, record_date, failure, smart_*_raw)."""
    date_col = "record_date" if "record_date" in raw.columns else "reading_date"
    df = raw.sort_values(["serial_number", date_col]).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])
    day = (df[date_col] - df[date_col].min()).dt.days.to_numpy(dtype=np.int64)
    code = pd.factorize(df["serial_number"])[0].astype(np.int64)

    # --- clean ---------------------------------------------------------
    needed = sorted(set(config.CURRENT_SMART) | set(config.ROLLING_SMART) | {194})
    clean = pd.DataFrame({f"smart_{n}_raw": df[f"smart_{n}_raw"].astype("float64") for n in needed})
    for attr, limit in config.IMPOSSIBLE_ABOVE.items():
        col = f"smart_{attr}_raw"
        clean[col] = clean[col].where(clean[col] <= limit)

    # Seagate-style drives pack min/max temperature into the high bytes of smart_194_raw;
    # the lowest byte is the current temperature in degrees C.
    temp = np.mod(clean.pop("smart_194_raw").to_numpy(), 256)
    lo, hi = config.TEMPERATURE_RANGE_C
    clean["temperature_c"] = np.where((temp >= lo) & (temp <= hi), temp, np.nan)

    clean = clean.groupby(code).ffill(limit=config.FFILL_LIMIT_DAYS)     # per-drive forward fill

    # --- window --------------------------------------------------------
    roll_cols = [f"smart_{n}_raw" for n in config.ROLLING_SMART]
    mean, slope, delta = rolling_features(code, day, clean[roll_cols].to_numpy(dtype=float),
                                          config.ROLLING_WINDOW_DAYS)
    out = pd.DataFrame({"serial_number": df["serial_number"], "feature_date": df[date_col].dt.date})
    for col in config.FEATURE_COLUMNS:
        if col in clean:
            out[col] = clean[col].to_numpy()
    w = config.ROLLING_WINDOW_DAYS
    for j, n in enumerate(config.ROLLING_SMART):
        out[f"smart_{n}_raw_mean_{w}d"] = mean[:, j]
        out[f"smart_{n}_raw_slope_{w}d"] = slope[:, j]
        out[f"smart_{n}_raw_delta_{w}d"] = delta[:, j]

    # --- label ---------------------------------------------------------
    label, after_failure = make_labels(code, day, df["failure"].to_numpy(), horizon)
    out[config.LABEL_COLUMN] = label

    # --- keep only rows with enough history, and none after a failure ----
    first_day = pd.Series(day).groupby(code).transform("min").to_numpy()
    keep = (day - first_day >= min_history - 1) & ~after_failure
    out = out.loc[keep].reset_index(drop=True)
    out[config.LABEL_COLUMN] = out[config.LABEL_COLUMN].astype("Int64")
    return out[["serial_number", "feature_date"] + config.FEATURE_COLUMNS + [config.LABEL_COLUMN]]


# ----------------------------------------------------------------------
# 4. Database in / out
# ----------------------------------------------------------------------

def read_telemetry() -> pd.DataFrame:
    cols = ", ".join(["serial_number", "record_date", "failure"] + config.SMART_COLS)
    return pd.read_sql(f"SELECT {cols} FROM smart_telemetry", get_engine())


def write_features(feats: pd.DataFrame) -> None:
    cols = list(feats.columns)
    buf = io.StringIO()
    feats.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    with get_connection() as conn, conn.cursor() as cur:
        col_defs = []
        for c in cols:
            if c == "serial_number":
                col_defs.append(f"{c} VARCHAR(64) NOT NULL")
            elif c == "feature_date":
                col_defs.append(f"{c} DATE NOT NULL")
            elif c == config.LABEL_COLUMN:
                col_defs.append(f"{c} INTEGER")
            else:
                col_defs.append(f"{c} DOUBLE PRECISION")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS features (
                {', '.join(col_defs)},
                PRIMARY KEY (serial_number, feature_date)
            )
        """)
        cur.execute("TRUNCATE features")
        with cur.copy(f"COPY features ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv)") as cp:
            while chunk := buf.read(1 << 20):
                cp.write(chunk)


def report(feats: pd.DataFrame, horizon: int) -> None:
    lab = feats[config.LABEL_COLUMN]
    known = lab.notna().sum()
    pos = int((lab == 1).sum())
    neg = int((lab == 0).sum())
    print(f"\nFeature table: {len(feats):,} rows, {feats['serial_number'].nunique():,} drives, "
          f"{feats['feature_date'].min()} .. {feats['feature_date'].max()}")
    print(f"Label = fails within {horizon} days: {pos:,} positive, {neg:,} negative, "
          f"{len(feats) - known:,} unknown (too close to the end of the data)")
    if pos:
        print(f"Positive share of labelled rows: {pos / known:.2%}   "
          f"-> use class weights; suggested positive-class weight ~ {neg / pos:.0f}")
    else:
        print("WARNING: no positive labels. Load more data (a full quarter) before training.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=config.HORIZON_DAYS)
    ap.add_argument("--min-history", type=int, default=config.MIN_HISTORY_DAYS)
    args = ap.parse_args()

    t0 = time.time()
    raw = read_telemetry()
    if raw.empty:
        raise SystemExit("smart_telemetry is empty. Run src/load_backblaze.py first.")
    print(f"Read {len(raw):,} readings in {time.time() - t0:.1f}s; building features ...")
    feats = build_feature_frame(raw, args.horizon, args.min_history)
    write_features(feats)
    report(feats, args.horizon)
    print(f"Done in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
