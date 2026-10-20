
import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import shap

from sqlalchemy import text
from sklearn.metrics import precision_recall_curve
from config import FEATURE_COLUMNS, HORIZON_DAYS
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

    # Generate ROC and precision-recall plots using held-out test data.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = metrics.roc_curve(y_test, probabilities)
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    pr_precision, pr_recall, _ = precision_recall_curve(
        y_test, probabilities
    )
    axes[1].plot(pr_recall, pr_precision, label=f"PR-AUC = {pr_auc:.4f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("models/evaluation.png", dpi=150)
    plt.close(fig)

    engine = get_engine()

    # Compare the current F1 score with the most recent monitoring record.
    with engine.begin() as conn:
        previous_f1 = conn.execute(
            text("""
                SELECT f1_score
                FROM model_drift_monitoring
                ORDER BY evaluated_at DESC
                LIMIT 1
            """)
        ).scalar()

        drift_detected = (
            previous_f1 is not None
            and float(previous_f1) - f1 > 0.10
        )

        conn.execute(
            text("""
                INSERT INTO model_drift_monitoring
                (model_version, roc_auc, pr_auc, f1_score,
                 baseline_failure_rate, drift_detected)
                VALUES (:version, :roc, :pr, :f1, :rate, :drift)
            """),
            {
                "version": f"xgb-{HORIZON_DAYS}d",
                "roc": roc_auc,
                "pr": pr_auc,
                "f1": f1,
                "rate": failure_rate,
                "drift": drift_detected
            }
        )

    print(f"Model degradation detected: {drift_detected}")

    # Generate predictions for the latest available feature row per drive.
    all_features = df.sort_values("feature_date").copy()
    latest = all_features.groupby("serial_number").tail(1).copy()

    X_latest = latest[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    X_latest = X_latest.replace([np.inf, -np.inf], np.nan).fillna(0)

    risk_probabilities = model.predict_proba(X_latest)[:, 1]

    # Calculate global SHAP feature importance on a sample of latest drives.
    print("Calculating SHAP feature importance...")
    explainer = shap.TreeExplainer(model)
    sample_size = min(1000, len(X_latest))
    shap_sample = X_latest.sample(n=sample_size, random_state=42)
    shap_values = explainer.shap_values(shap_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    shap_importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_importance)[::-1][:3]
    top_features = [
        FEATURE_COLUMNS[i]
        for i in top_indices
        if shap_importance[i] > 0
    ]

    # Store risk scores using the verified database column names.
    score_df = pd.DataFrame({
        "disk_id": latest["disk_id"].astype(int).to_numpy(),
        "timestamp": pd.to_datetime(latest["timestamp"]).to_numpy(),
        "risk_score": risk_probabilities.astype(float),
        "model_version": f"xgb-{HORIZON_DAYS}d",
        "top_factors": ", ".join(top_features)
    })

    if not score_df.empty:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO risk_scores
                        (disk_id, timestamp, risk_score, model_version, top_factors)
                    VALUES
                        (:disk_id, :timestamp, :risk_score, :model_version, :top_factors)
                    ON CONFLICT (disk_id, timestamp)
                    DO UPDATE SET
                        risk_score = EXCLUDED.risk_score,
                        model_version = EXCLUDED.model_version,
                        top_factors = EXCLUDED.top_factors
                """),
                score_df.to_dict(orient="records")
            )

if __name__ == "__main__":
    main()


