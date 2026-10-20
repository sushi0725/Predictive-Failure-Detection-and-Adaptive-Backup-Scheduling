
# Predictive Failure Detection and Adaptive Backup Scheduling

An ML-powered system that analyzes hard-disk telemetry to estimate near-term failure risk and prioritize backup scheduling based on predicted risk levels.

## Overview

The system uses Backblaze hard-drive telemetry, feature engineering, an XGBoost classifier, PostgreSQL storage, and an interactive Streamlit dashboard. Its objective is to identify potentially failing disks early and use their estimated risk to prioritize backup operations.

## Key Features

- **Telemetry ingestion:** Imports Backblaze daily hard-drive CSV data into PostgreSQL.
- **Feature engineering:** Generates model-ready features from SMART telemetry.
- **Failure prediction:** Uses XGBoost to estimate whether a disk may fail within the next seven days.
- **Chronological evaluation:** Evaluates the model using a time-based train/test split.
- **Risk scoring:** Stores disk risk scores and associated explanatory factors.
- **Adaptive backup scheduling:** Applies risk-based policies to prioritize backup events.
- **Interactive dashboard:** Displays disk health, risk categories, urgent backup priorities, and policy controls.
- **Model monitoring:** Records evaluation metrics and supports F1-based performance-degradation checks.
- **Explainability:** Includes SHAP-based global feature-importance analysis in the training pipeline.

## Architecture

```text
Backblaze Daily CSV Files
          |
          v
   Data Ingestion
          |
          v
      PostgreSQL
          |
          v
  Feature Engineering
          |
          v
 Chronological Train/Test Split
          |
          v
   XGBoost Classifier
          |
          v
     Risk Scoring
          |
          v
 Adaptive Backup Policies
          |
          v
  Backup Priority Events
          |
          v
 Streamlit Dashboard
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Machine learning | XGBoost, scikit-learn |
| Data processing | pandas, NumPy |
| Explainability | SHAP |
| Database | PostgreSQL |
| Database integration | SQLAlchemy, psycopg |
| Dashboard | Streamlit |
| Visualization | Matplotlib |
| Model persistence | joblib |

## Project Structure

```text
Predictive Failure Detection and Adaptive Backup Scheduling/
├── db/
│   ├── schema.sql
│   └── drop_all.sql
├── data/
│   └── raw/
├── models/
│   ├── risk_model.joblib
│   ├── metrics.json
│   └── evaluation.png
├── src/
│   ├── config.py
│   ├── db_utils.py
│   ├── init_db.py
│   ├── dataset.py
│   ├── build_features.py
│   ├── train_model.py
│   ├── decision_layer.py
│   ├── run_decision_cycle.py
│   └── dashboard.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: The exact directory structure may vary slightly depending on your local repository.*

## Dataset

The project uses Backblaze hard-drive telemetry, including SMART attributes and disk failure labels.

The current database contains telemetry from 91 daily CSV files covering April 1 to June 30, 2026. Feature generation also includes available historical records required for calculating rolling features.

The model uses selected SMART attributes, including:

- `smart_1_raw`
- `smart_5_raw`
- `smart_9_raw`
- `smart_187_raw`
- `smart_197_raw`

The prediction target is whether a disk failure occurs within the following seven days.

## Model Training and Evaluation

The training pipeline uses an XGBoost classifier and a chronological train/test split to preserve the time ordering of telemetry.

Class weighting is used to address the severe imbalance between failure and healthy observations.

The training pipeline is designed to save:

- `models/risk_model.joblib` — trained model
- `models/metrics.json` — evaluation metrics
- `models/evaluation.png` — evaluation plots

The updated training code also includes SHAP-based global feature importance and F1-based performance-degradation monitoring. These additions must be run and validated before their results can be considered verified.

### Previously recorded evaluation

The existing `models/metrics.json` reports the following results from an earlier training run:

| Metric | Value |
|---|---:|
| Training rows | 1,042,265 |
| Testing rows | 115,703 |
| Training failures | 54 |
| Testing failures | 7 |
| Accuracy | 95.54% |
| ROC-AUC | 0.8244 |
| PR-AUC | 0.00995 |
| Precision | 0.000775 |
| Recall | 57.14% |
| F1-score | 0.00155 |

These are previously recorded results, not a validation of any subsequent code changes.

**Important limitation:** Accuracy alone is not sufficient for evaluating this highly imbalanced failure-prediction problem. The recorded precision and F1-score are very low, and the test set contains only seven failures. The model therefore requires further evaluation and improvement before production use.

## Adaptive Backup Scheduling

The decision layer uses risk-based policies to prioritize disks according to their predicted risk.

The system stores risk scores and corresponding backup-priority events in PostgreSQL. The dashboard displays the current priorities and allows users to explore how risk thresholds affect classification and scheduling decisions.

The current implementation represents scheduling decisions and generated events. It does not establish that physical backups, snapshots, or recovery operations have actually executed.

## Dashboard

The Streamlit dashboard provides views of:

- Latest disk risk scores
- Disk health and risk classifications
- Urgent backup queue
- Risk-based policy thresholds
- Backup allocation and workload comparisons
- Model evaluation information

The workload comparison is an estimate based on the scheduling logic, not a measurement of completed physical backup operations.

## Setup

### 1. Prerequisites

Install the following:

- Python 3.10 or another Python version supported by the project's dependencies
- PostgreSQL
- GitHub Desktop or Git
- A terminal such as PowerShell

### 2. Create and activate a virtual environment

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example configuration:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and provide your PostgreSQL connection details. Ensure the database name, username, password, host, and port match your local PostgreSQL setup.

**Security:** Never commit `.env` or real database credentials to GitHub.

### 5. Initialize the database

Initialize the schema only when appropriate for your existing database:

```powershell
python src/init_db.py
```

Do not use the `--reset` option on a database containing data you need to preserve.

### 6. Prepare the data

Place the required Backblaze CSV files in the configured data directory and use the project's ingestion and feature-generation workflow.

If your telemetry and feature tables are already populated, do not repeat the import or feature-generation steps unnecessarily.

### 7. Run the decision cycle

```powershell
python src/run_decision_cycle.py
```

This generates or updates risk-based scheduling information using the configured pipeline.

### 8. Launch the dashboard

```powershell
python -m streamlit run src/dashboard.py
```

Open the local URL printed in the terminal, normally:

`http://localhost:8501`

## Limitations and Future Work

The current implementation is a prototype and has the following limitations:

1. **Rare failure events:** The dataset contains very few failure-positive observations relative to healthy records.
2. **Low precision:** The recorded evaluation shows that false-positive predictions are a major concern.
3. **Limited telemetry:** The current pipeline focuses on Backblaze SMART telemetry rather than CPU, memory, operating-system, and application logs.
4. **Model comparisons:** LSTM, Transformer, and other alternative models have not been established as completed comparative experiments.
5. **Decision optimization:** The current backup decision logic uses policies and rules; reinforcement learning or a Markov decision process is not implemented as a verified capability.
6. **Monitoring:** The current F1-based degradation check is not a complete statistical data-drift detection system. Automatic retraining is not implemented as a verified end-to-end workflow.
7. **Explainability validation:** SHAP-based global feature importance is included in the updated training code, but its successful execution and outputs must be validated.
8. **Physical backup execution:** Generated scheduling events do not prove that backups, snapshots, or failover operations have actually executed.
9. **Robustness testing:** Additional corruption, ablation, and external-dataset experiments remain future work.

## Future Enhancements

- Improve failure recall and precision through additional failure examples, temporal validation, and threshold tuning.
- Compare XGBoost with alternative machine-learning and time-series models.
- Add statistically validated data-drift monitoring and a controlled retraining workflow.
- Extend telemetry with operating-system and system-resource metrics.
- Implement and evaluate optimization methods beyond rule-based scheduling.
- Integrate real backup execution and recovery verification.
- Evaluate robustness on additional datasets and under simulated failure conditions.

## Safety and Data Handling

- Keep database credentials in `.env`.
- Do not run destructive database reset scripts unless intentionally rebuilding the database.
- Preserve existing risk scores and backup events when updating model predictions.
- Validate model metrics and database outputs before presenting results as verified.

## Conclusion

Predictive Failure Detection and Adaptive Backup Scheduling demonstrates a database-backed prototype that combines hard-drive telemetry, machine-learning-based failure-risk estimation, risk-based backup prioritization, and an interactive dashboard.

Its current results demonstrate the implemented pipeline, while low precision, limited failure examples, and the absence of verified physical backup execution remain important limitations for future development.
