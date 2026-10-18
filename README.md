# Predictive Failure Detection and Adaptive Backup Scheduling for Server Infrastructure

An enterprise-grade, predictive disaster recovery (DR) framework that continuously monitors multi-layer server telemetry to forecast imminent hardware failures and dynamically schedules risk-proportionate, cost-aware backup actions. 

---

## 👥 Academic Context & Team Details
* **Course:** Database Systems (BACSE202)
* **Institution:** Vellore Institute of Technology (VIT)
* **Faculty Guide:** Dr. Saritha Murali
* **Team Members:**
  * Abinaya J (25BCE2779)
  * Ayushi Sahu (25BAI0167)
  * Ashwini J (25BBS0127)

---

## 📌 Project Overview
Traditional backup systems operate reactively—triggering snapshots on rigid calendars or *after* a crash occurs. This project moves server infrastructure management from reactive to predictive. It continuously fuses multi-layer signal streams to predict failures days in advance, mapping risk thresholds to dynamic, cost-aware data protection workflows.

### Core Architecture Layers:
1. **Telemetry Ingestion Layer:** Append-only ingestion of hardware, OS, and system log data.
2. **Predictive Analytics Layer:** Gradient-Boosted Trees (GBT) and deep sequential models (LSTM) that evaluate rolling time-series window attributes.
3. **Cost-Aware Decision Layer:** A Markov Decision Process (MDP) that weights data loss gravity against operational storage and network costs.
4. **Feedback & Drift Layer:** A continuous monitor assessing accuracy degradation over time to trigger proactive model retraining.

---

## 🗄️ Database Architecture (PostgreSQL 18)
The storage layer is engineered inside **PostgreSQL 18** to handle incoming metrics, calculated features, and automated scheduling logs natively:

### System Tables
* `storage_nodes`: Maps physical data centers, cluster nodes, vaults, and machine pods.
* `disks`: Core hardware registry tracking drive models, slots, device capacities, and live lifecycle state.
* `smart_telemetry`: High-frequency metrics table ingesting critical S.M.A.R.T. health identifiers (Attributes 5, 9, 187, 188, 194, 197, 198, 199).
* `features`: Engineered analytical data holding 7-day rolling means, trends, and delta variations.
* `risk_scores`: Real-time machine learning inference store running prediction tracking variants.
* `adaptive_backup_policies`: Cost-aware rule lookup matrix defining threshold actions.
* `adaptive_backup_events`: Execution ledger recording snapshot triggers, backup overhead volumes, and run costs.
* `model_drift_monitoring`: Evaluates metric rolling metrics (F1/Precision) to identify system drift.

### Analytical Database Views
* `vw_disk_latest_health`: Performs highly optimized internal table joins to return active status roll-ups for all devices.
* `vw_urgent_backup_queue`: Filters high-risk operational hardware items to generate priority execution lines for the automation layers.

---

## 🚀 Setup & Execution Guide

### Prerequisites
* Python 3.10+
* PostgreSQL 18 Local Instance
* [Backblaze Hard Drive Stats Public Dataset](https://backblaze.com) (Download a single quarter/month CSV dataset)

### 1. Installation & Environment Map
Clone this repository onto your machine and initialize dependencies:
```bash
pip install -r requirements.txt
```
Duplicate `.env.example`, save it as `.env`, and assign your database access parameters:
```env
DATABASE_URL=postgresql://<username>:<password>@localhost:5432/postgres
PYTHONPATH=src
HORIZON_DAYS=1
```

### 2. File Organization
Place a selection of unzipped daily Backblaze data tracking logs (`.csv` format) straight into your system folders:
```text
📁 data/
└── 📁 raw/
    ├── 2026-03-26.csv
    ├── 2026-03-27.csv
    └── 2026-03-28.csv
```

### 3. Initialize the Engine Pipeline
Set up the tables, ingest data rows, compile rolling window features, score the drive models, and trigger decisions sequentially:
```powershell
# Build tables and schema views
python src/init_db.py

# Run the ETL script to parse data/raw CSVs into postgres
python src/load_backblaze.py

# Compile windowing features and trends
python src/build_features.py

# Execute ML model inferences
python src/demo_score_drives.py

# Process risk constraints against database decision matrices
python src/run_decision_cycle.py
```

### 4. Launch the Management Interface Dashboard
Fire up the web console to view current risk metrics, active alerts, and database tracking tables:
```bash
streamlit run src/dashboard.py
```
Open your internet browser and navigate to: **`http://localhost:8501`**

---

## 📊 Live System Status & Policy Actions
The automated decision layer matches calculated risk metrics against specialized protection thresholds:

| Risk Scale | Graded Action | Operational System Logic |
| :--- | :--- | :--- |
| **≥ 0.80 (Critical)** | `prepare_failover` | Imminent danger. Initiate data rerouting protocols every 2 hours. |
| **≥ 0.50 (High)** | `immediate_full_snapshot` | Risk verified. Capture a full system backup snapshot every 6 hours. |
| **≥ 0.25 (Moderate)**| `schedule_incremental_backup`| Minor deviations. Run incremental file copies every 12 hours. |
| **≥ 0.10 (Low)** | `increase_monitoring` | Watch list. Escalate telemetry tracking frequencies to 24-hour cycles. |
| **< 0.10 (Nominal)**  | `no_action` | Drive healthy. Standard logging parameters apply. |

---

## 🌿 Sustainability & Global Impact
* **SDG 9 (Industry, Innovation, and Infrastructure):** Boosts infrastructure runtime stability by eliminating unexpected hardware failures via algorithmic foresight.
* **SDG 12 (Responsible Consumption and Production):** Prevents storage array overhead, computing runtime costs, and data clutter by triggering backups selectively rather than across massive blank arrays.
