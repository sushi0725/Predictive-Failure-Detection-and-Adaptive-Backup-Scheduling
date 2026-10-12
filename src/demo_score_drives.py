"""
PLACEHOLDER model -- NOT the project's real model (that is the ML teammate's job).

It exists for one reason: to prove the whole chain works end to end
(features -> risk_scores -> decisions -> dashboard) before the real model is ready.
Delete it, or replace the model below, once the ML teammate's model exists.

    python src/demo_score_drives.py

It trains a small gradient-boosted classifier on the older data, prints how it does on the
newer data, then writes one risk score per drive (newest feature row) into `risk_scores`.
"""

import numpy as np
import argparse
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import config
from dataset import chronological_split, load_labelled_features
from db_utils import get_engine, write_risk_scores

MODEL_VERSION = "demo-hgb-v0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-fraction", type=float, default=0.20,
                    help="Fraction of dates for test set (default 0.20)")
    ap.add_argument("--gap-days", type=int, default=0,
                    help="Gap days between train and test sets (default 0 for compact ranges)")
    ap.add_argument("--train-end", type=str, default=None,
                    help="Explicit last date for training, e.g. '2026-03-29'")
    ap.add_argument("--test-start", type=str, default=None,
                    help="Explicit first date for testing, e.g. '2026-03-30'")
    args = ap.parse_args()

    df = load_labelled_features()
    if df.empty:
        raise SystemExit("No labelled features found in 'features' table. Run src/build_features.py first.")

    # Apply date split fitting within active range (2026-03-26 to 2026-03-31)
    if args.train_end and args.test_start:
        t_end = pd.to_datetime(args.train_end)
        t_start = pd.to_datetime(args.test_start)
        train = df[df["feature_date"] <= t_end]
        test = df[df["feature_date"] >= t_start]
    else:
        train, test = chronological_split(df, test_fraction=args.test_fraction, gap_days=args.gap_days)

    print(f"train: {len(train):,} rows ({train[config.LABEL_COLUMN].sum()} failures) | "
          f"test: {len(test):,} rows ({test[config.LABEL_COLUMN].sum()} failures)")
    if train[config.LABEL_COLUMN].nunique() < 2 or test[config.LABEL_COLUMN].nunique() < 2:
        raise SystemExit("Need both failing and healthy rows in the train AND test period. Load more data.")

    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0)
    model.fit(train[config.FEATURE_COLUMNS], train[config.LABEL_COLUMN])
    p = model.predict_proba(test[config.FEATURE_COLUMNS])[:, 1]
    print(f"test ROC-AUC {roc_auc_score(test[config.LABEL_COLUMN], p):.3f} | "
          f"PR-AUC {average_precision_score(test[config.LABEL_COLUMN], p):.3f} "
          f"(PR-AUC is the honest metric when failures are rare)")

    latest = pd.read_sql(
        "SELECT DISTINCT ON (serial_number) * FROM features ORDER BY serial_number, feature_date DESC",
        get_engine(),
    )
    # Drives that already failed are no longer in service: do not schedule backups for them.
    failed = pd.read_sql("SELECT serial_number FROM v_drive_lifecycle WHERE failed", get_engine())
    latest = latest[~latest["serial_number"].isin(failed["serial_number"])].reset_index(drop=True)

    latest["risk_score"] = model.predict_proba(latest[config.FEATURE_COLUMNS])[:, 1]
    latest["as_of_date"] = latest["feature_date"]

    # Stand-in for SHAP (NOT an explanation of the model): the 3 features sitting furthest above
    # the fleet median, measured in standard deviations. The real thing is shap.TreeExplainer.
    X = latest[config.FEATURE_COLUMNS]
    z = (X - X.median()) / X.std().replace(0, np.nan)
    latest["top_factors"] = [
        ", ".join(z.loc[i].nlargest(3).dropna().index) if latest.loc[i, "risk_score"] >= 0.10 else None
        for i in latest.index
    ]

    n = write_risk_scores(latest[["serial_number", "as_of_date", "risk_score", "top_factors"]], MODEL_VERSION)
    print(f"Wrote {n:,} scores to risk_scores (model_version={MODEL_VERSION}). "
          f"Highest: {latest['risk_score'].max():.3f}, mean: {latest['risk_score'].mean():.3f}")


if __name__ == "__main__":
    main()
