
import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sqlalchemy import text
from config import FEATURE_COLUMNS, HORIZON_DAYS
from db_utils import get_engine
import dataset

def require_two_classes(y, name):
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"{name} must contain both classes; found {classes.tolist()}.")

def main():
    print("=== Predictive Failure Detection: Training and Evaluation ===")
    print(f"Prediction horizon: {HORIZON_DAYS} days")
    df = dataset.load_labelled_features()
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")
    train_df, test_df = dataset.chronological_split(df)
    X_train = train_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df["fails_within_horizon"].astype(int).to_numpy()
    y_test = test_df["fails_within_horizon"].astype(int).to_numpy()
    require_two_classes(y_train, "Training labels")
    require_two_classes(y_test, "Test labels")
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    if positives == 0:
        raise ValueError("Training data contains no positive failures.")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negatives / positives,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    print("Training XGBoost...")
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    roc_auc = float(metrics.roc_auc_score(y_test, probabilities))
    pr_auc = float(metrics.average_precision_score(y_test, probabilities))
    precision = float(metrics.precision_score(y_test, predictions, zero_division=0))
    recall = float(metrics.recall_score(y_test, predictions, zero_division=0))
    f1 = float(metrics.f1_score(y_test, predictions, zero_division=0))
    accuracy = float(metrics.accuracy_score(y_test, predictions))
    failure_rate = float(np.mean(y_test == 1))
    payload = {
        "data_source": "database_backed_telemetry",
        "prediction_horizon_days": HORIZON_DAYS,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "training_failures": positives,
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
    model_version = f"xgb-{HORIZON_DAYS}d-{pd.Timestamp.now():%Y%m%d}"
    joblib.dump(model, "models/risk_model.joblib")

    # Save ROC and precision-recall curves from held-out test data.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = metrics.roc_curve(y_test, probabilities)
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "--")
    axes[0].set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[0].legend()
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, probabilities)
    axes[1].plot(pr_recall, pr_precision, label=f"PR-AUC = {pr_auc:.4f}")
    axes[1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("models/evaluation.png", dpi=150)
    plt.close(fig)

    engine = get_engine()
    with engine.begin() as conn:
        previous = conn.execute(
            text("SELECT f1_score FROM model_drift_monitoring ORDER BY evaluated_at DESC LIMIT 1")
        ).scalar()
        drift_detected = previous is not None and (float(previous) - f1) > 0.10
        conn.execute(
            text("""
                INSERT INTO model_drift_monitoring
                (model_version, roc_auc, pr_auc, f1_score, baseline_failure_rate, drift_detected)
                VALUES (:version, :roc, :pr, :f1, :rate, :drift)
            """),
            {"version": model_version, "roc": roc_auc, "pr": pr_auc, "f1": f1, "rate": failure_rate, "drift": drift_detected}
        )

    # SHAP explanations on a bounded test sample to avoid excessive runtime.
    sample_size = min(1000, len(X_test))
    shap_sample = X_test.sample(n=sample_size, random_state=42) if sample_size else X_test
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(shap_importance)[::-1]
    top_features = [FEATURE_COLUMNS[i] for i in order[:3]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([FEATURE_COLUMNS[i] for i in order[::-1]], shap_importance[order[::-1]])
    ax.set_title("Mean Absolute SHAP Importance")
    fig.tight_layout()
    fig.savefig("models/shap_importance.png", dpi=150)
    plt.close(fig)

    # Score the latest feature row for each drive; never delete historical scores.
    latest = df.sort_values("feature_date").groupby("serial_number").tail(1).copy()
    X_latest = latest[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    latest_probabilities = model.predict_proba(X_latest)[:, 1]
    score_rows = []
    for (_, row), probability in zip(latest.iterrows(), latest_probabilities):
        score_rows.append({
            "disk_id": int(row["disk_id"]),
            "timestamp": pd.to_datetime(row["timestamp"]),
            "risk_score": float(probability),
            "model_version": model_version,
            "top_factors": ", ".join(top_features)
        })
    if score_rows:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO risk_scores (disk_id, timestamp, risk_score, model_version, top_factors)
                    VALUES (:disk_id, :timestamp, :risk_score, :model_version, :top_factors)
                    ON CONFLICT (disk_id, timestamp) DO UPDATE SET
                        risk_score = EXCLUDED.risk_score,
                        model_version = EXCLUDED.model_version,
                        top_factors = EXCLUDED.top_factors
                """),
                score_rows
            )

    print("\n=== Evaluation Results ===")
    for key in ["accuracy", "roc_auc", "pr_auc", "precision", "recall", "f1_score", "baseline_failure_rate"]:
        print(f"{key}: {payload[key]:.6f}")
    print(f"Drift detected: {drift_detected}")
    print("Saved model, metrics, evaluation plots, SHAP plot, and risk scores.")

if __name__ == "__main__":
    main()
