import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib

# Team mate-in decision layer-ilirundhu functions-ai import seyyuthal[cite: 2]
from decision_layer import decide_batch, estimate_backup_load

print("=== 1. Generating / Loading SMART Data (Slide 1 & 2) ===")
np.random.seed(42)
n_samples = 10
serials = [f"DRIVE{i:04d}" for i in range(n_samples)]

# Smart telemetry features (rolling means, slopes)
X = pd.DataFrame({
    'smart_197_raw_mean': np.random.rand(n_samples),
    'smart_5_raw_slope': np.random.rand(n_samples),
    'smart_187_raw_mean': np.random.rand(n_samples),
    'smart_198_raw_slope': np.random.rand(n_samples),
    'disk_io_latency_ms_mean': np.random.rand(n_samples),
    'smart_188_raw_mean': np.random.rand(n_samples)
})
y = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

print("=== 2. Training Baseline Risk Model (Slide 3) ===")
model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "baseline_risk_model.joblib")
print("Model saved successfully as 'baseline_risk_model.joblib'!")

# Predict risk probabilities for each drive
risk_probabilities = model.predict_proba(X)[:, 1]
risk_scores_dict = {serials[i]: float(risk_probabilities[i]) for i in range(n_samples)}

print("=== 3. Adding Explainability via SHAP (Slide 5) ===")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
feature_names = np.array(X.columns)

factors_dict = {}
for i, serial in enumerate(serials):
    top_indices = np.argsort(shap_values[i])[-3:][::-1]
    factors_dict[serial] = list(feature_names[top_indices])

print("=== 4. Running Decision Layer & Action Queue (Slide 4) ===")
decisions = decide_batch(risk_scores_dict, factors=factors_dict)

print(f"\n{'DRIVE':<12}{'RISK':>7}  {'TIER':<10}{'ACTION':<28}")
print("-" * 65)
for d in decisions:
    print(f"{d.serial_number:<12}{d.risk_score:>7.2f}  {d.risk_tier:<10}{d.action.value:<28}")

print("\n=== 5. Backup Load Comparison ===")
load_stats = estimate_backup_load(decisions)
for k, v in load_stats.items():
    print(f"  {k:<32} {v}")

print("\nAll steps executed successfully!")