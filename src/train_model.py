
import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from sqlalchemy import text
from config import DATABASE_URL, FEATURE_COLUMNS, HORIZON_DAYS
from db_utils import get_engine
import dataset

def require_two_classes(y, name):
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(
            f"{name} must contain both healthy (0) and failed (1) "
            f"examples. Found classes: {classes.tolist()}. "
            "No artificial labels will be generated."
        )

def main():
    print("=== Predictive Failure Detection: Training and Evaluation ===")
    print(f"Prediction horizon: {HORIZON_DAYS} days")

    df = dataset.load_labelled_features()
    print(f"Valid labeled rows: {len(df):,}")

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required feature column is missing: {col}")

    train_df, test_df = dataset.chronological_split(df)

    X_train = train_df[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    X_test = test_df[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    y_train = train_df["fails_within_horizon"].astype(int).to_numpy()
    y_test = test_df["fails_within_horizon"].astype(int).to_numpy()

    require_two_classes(y_train, "Training labels")
    require_two_classes(y_test, "Test labels")

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")
    print(f"Training failures: {int(y_train.sum()):,}")
    print(f"Testing failures:  {int(y_test.sum()):,}")

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    pos_weight = negatives / positives

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    print("Training XGBoost on the training period...")
    model.fit(X_train, y_train)

    print("Evaluating on the held-out future period...")
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    roc_auc = float(metrics.roc_auc_score(y_test, probabilities))
    pr_auc = float(metrics.average_precision_score(y_test, probabilities))
    precision = float(metrics.precision_score(
        y_test, predictions, zero_division=0
    ))
    recall = float(metrics.recall_score(
        y_test, predictions, zero_division=0
    ))
    f1 = float(metrics.f1_score(
        y_test, predictions, zero_division=0
    ))
    accuracy = float(metrics.accuracy_score(y_test, predictions))
    failure_rate = float(np.mean(y_test == 1))

    payload = {
        "data_source": "database_backed_telemetry",
        "prediction_horizon_days": HORIZON_DAYS,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "training_failures": int(y_train.sum()),
        "testing_failures": int(y_test.sum()),
        "test_start_date": str(test_df["feature_date"].min()),
        "test_end_date": str(test_df["feature_date"].max()),
        "threshold": 0.5,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "baseline_failure_rate": failure_rate
    }

    os.makedirs("models", exist_ok=True)

    with open("models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    joblib.dump(model, "models/risk_model.joblib")

    plt.figure(figsize=(7, 5))
    fpr, tpr, _ = metrics.roc_curve(y_test, probabilities)
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Held-out Test Set: ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/evaluation.png", dpi=150)
    plt.close()

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO model_drift_monitoring
                (model_version, roc_auc, pr_auc, f1_score,
                 baseline_failure_rate, drift_detected)
                VALUES (:version, :roc, :pr, :f1, :rate, false)
            """),
            {
                "version": f"xgb-{HORIZON_DAYS}d",
                "roc": roc_auc,
                "pr": pr_auc,
                "f1": f1,
                "rate": failure_rate
            }
        )

    # Generate risk scores for the latest available feature snapshot
    # of each drive. The scores are probabilities between 0 and 1.
    all_features = df.sort_values("feature_date").copy()
    latest = all_features.groupby("serial_number").tail(1).copy()

    X_latest = latest[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    X_latest = X_latest.replace([np.inf, -np.inf], np.nan).fillna(0)
    risk_probabilities = model.predict_proba(X_latest)[:, 1]

    importance = model.feature_importances_
    top_features = [
        FEATURE_COLUMNS[i]
        for i in np.argsort(importance)[::-1][:3]
        if importance[i] > 0
    ]

    score_rows = []
    for (_, row), probability in zip(
        latest.iterrows(), risk_probabilities
    ):
        score_rows.append({
            "disk_id": int(row["disk_id"]),
            "timestamp": pd.to_datetime(row["timestamp"]),
            "risk_score": float(probability),
            "model_version": f"xgb-{HORIZON_DAYS}d",
            "top_factors": ", ".join(top_features)
        })

    if score_rows:
        score_df = pd.DataFrame(score_rows)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM risk_scores"))
            score_df.to_sql(
                "risk_scores",
                con=conn,
                if_exists="append",
                index=False
            )

    print("\n=== Evaluation Results ===")
    for key in [
        "accuracy", "roc_auc", "pr_auc", "precision",
        "recall", "f1_score", "baseline_failure_rate"
    ]:
        print(f"{key}: {payload[key]:.4f}")

    print("\nSaved:")
    print("  models/metrics.json")
    print("  models/risk_model.joblib")
    print("  models/evaluation.png")
    print("  risk scores and model metrics in PostgreSQL")

if __name__ == "__main__":
    main()
