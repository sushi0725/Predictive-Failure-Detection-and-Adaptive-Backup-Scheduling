import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt

# Team mate-in decision layer-ilirundhu functions import
from decision_layer import decide_batch, estimate_backup_load

print("=== 1. Data Generation ===")
np.random.seed(42)
n_samples = 10
serials = [f"DRIVE{i:04d}" for i in range(n_samples)]

X = pd.DataFrame({
    'smart_197_raw_mean': np.random.rand(n_samples),
    'smart_5_raw_slope': np.random.rand(n_samples),
    'smart_187_raw_mean': np.random.rand(n_samples),
    'smart_198_raw_slope': np.random.rand(n_samples),
    'disk_io_latency_ms_mean': np.random.rand(n_samples),
    'smart_188_raw_mean': np.random.rand(n_samples)
})
y = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

print("=== 2. Training Baseline Model (Slide 3) ===")
model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
model.fit(X, y)
joblib.dump(model, "baseline_risk_model.joblib")

# SLIDE 3 GRAPH: Feature Importance Plot
plt.figure(figsize=(8, 5))
xgb.plot_importance(model, max_num_features=6)
plt.title("Slide 3: Baseline Risk Model - Feature Importance")
plt.tight_layout()
plt.savefig("slide3_feature_importance.png")
plt.close()
print("Saved Slide 3 Graph as 'slide3_feature_importance.png'")

print("=== 3. Calculating SHAP Values (Slide 5) ===")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
feature_names = np.array(X.columns)

factors_dict = {}
for i, serial in enumerate(serials):
    top_indices = np.argsort(shap_values[i])[-3:][::-1]
    factors_dict[serial] = list(feature_names[top_indices])

# SLIDE 5 GRAPH: SHAP Summary Plot
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X, show=False)
plt.title("Slide 5: SHAP Feature Impact Analysis", fontsize=12)
plt.tight_layout()
plt.savefig("slide5_shap_summary.png")
plt.close()
print("Saved Slide 5 Graph as 'slide5_shap_summary.png'")

print("=== 4. Running Decision Layer ===")
risk_probabilities = model.predict_proba(X)[:, 1]
risk_scores_dict = {serials[i]: float(risk_probabilities[i]) for i in range(n_samples)}
decisions = decide_batch(risk_scores_dict, factors=factors_dict)

print("\n--- RESULTS ---")
for d in decisions:
    print(f"{d.serial_number:<12} Risk: {d.risk_score:.2f} | Tier: {d.risk_tier:<10} | Action: {d.action.value}")

print("\nAll steps completed successfully!")