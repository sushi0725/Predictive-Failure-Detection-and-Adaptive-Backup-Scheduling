# 🚀 Morning Resume Checklist & Pipeline Execution Guide

**Project:** Predictive Failure Detection and Adaptive Backup Scheduling for Server Infrastructure  
**Course:** Database Systems (BACSE202)  
**Database:** Local PostgreSQL 18 (`localhost:5432 / postgres`)

---

## 📌 Status Summary
* **Database Schema Initialized:** Tables `disks`, `storage_nodes`, `smart_telemetry`, `features`, `risk_scores`, `adaptive_backup_policies`, and `backup_events` are created and verified.
* **Dataset Loaded:** `41,408` rows of Backblaze S.M.A.R.T. telemetry ingested across 6 daily logs (`2026-03-26` to `2026-03-31`), registering `6,993` unique disks (`80` failed drives captured).
* **Feature Engineering Done:** `src/build_features.py` processed all `41,408` records with rolling windows, temperature normalization, and horizon labeling (`34,412` healthy labels, `125` failure labels).
* **Model Training & Drive Scoring Done:** `src/demo_score_drives.py` adjusted for the active date range (`2026-03-26` to `2026-03-31`), trained on `27,605` rows, evaluated on `6,932` rows, and wrote `6,913` risk scores into `risk_scores`.
* **Adaptive Decision Cycle Executed:** `src/run_decision_cycle.py` executed and logged `6,913` decisions into `backup_events` (5 critical failover, 16 full snapshots, 26 incremental backups, 26 increased monitoring, 6,840 nominal).

---

## 📋 Exact Commands for Pipeline Execution

Open a PowerShell terminal in the project root directory:
```powershell
cd "c:\Users\user\Desktop\VIT Material\2nd year\sem3\databse systems\project\Predictive Failure Detection and Adaptive Backup Scheduling"
```

### 1. Score Drives with the Predictive Model
Train the failure-prediction classifier on historical telemetry and score active drives:
```powershell
$env:PYTHONPATH="src"
python src/demo_score_drives.py

# Optional: customize explicit date thresholds
python src/demo_score_drives.py --train-end 2026-03-29 --test-start 2026-03-30
```

---

### 2. Run the Adaptive Decision Cycle
Evaluate risk scores against cost-aware policy tiers and log actions into `backup_events`:
```powershell
$env:PYTHONPATH="src"
# Dry-run to inspect decision distribution:
python src/run_decision_cycle.py --dry-run

# Commit decisions to PostgreSQL:
python src/run_decision_cycle.py
```

---

### 3. Launch the Project Dashboard
Start the interactive Streamlit dashboard to visualize disk health, risk scores, and adaptive scheduling policies in your web browser:
```powershell
$env:PYTHONPATH="src"
streamlit run src/dashboard.py
```
> The dashboard will open automatically at: **http://localhost:8501**

---

### 4. Sync Changes with GitHub Desktop
1. Open the **GitHub Desktop** app on your laptop.
2. Select repository: `Predictive-Failure-Detection-and-Adaptive-Backup-Scheduling`
3. Review changed files:
   * `db/schema.sql` (Full PostgreSQL schema)
   * `src/load_backblaze.py` (ETL loader for Backblaze data)
   * `src/build_features.py` (Feature engineering pipeline)
   * `src/dataset.py` (Chronological data splitting for short spans)
   * `src/demo_score_drives.py` (Predictive model & scoring script)
   * `RESUME_WORK.md` (Pipeline documentation)
   *(Note: `.env` and `data/raw/` CSV files should remain local).*
4. In the commit summary box, type:
   ```text
   feat: implement full predictive failure detection and adaptive backup pipeline
   ```
5. Click **Commit to main** and then **Push origin**.

---

## 🔍 Useful Database Quick-Check Commands

To inspect your PostgreSQL database at any time from PowerShell:
```powershell
$env:PGPASSWORD="ayushi@25"
& "D:\postgres SQL\bin\psql.exe" -U postgres -d postgres -c "
SELECT count(*) AS total_disks FROM disks;
SELECT count(*) AS total_telemetry FROM smart_telemetry;
SELECT count(*) AS total_features FROM features;
SELECT count(*) AS total_risk_scores FROM risk_scores;
SELECT action, count(*) FROM backup_events GROUP BY action;
"
$env:PGPASSWORD=$null
```
