# Predictive Failure Detection and Adaptive Backup Scheduling

An intelligent, data-driven system that uses machine learning to forecast hard drive failures using SMART telemetry data and adaptively prioritizes backup frequencies based on real-time hardware risk scores. By integrating failure forecasting with a policy-driven decision layer, the system minimizes data loss while optimizing network and storage bandwidth utilization.

---

## 🚀 Key Features

* **Predictive Risk Modeling:** Uses an optimized machine learning pipeline to analyze telemetry data and classify individual hard drives into varying risk bands (Low, Medium, High).
* **Adaptive Scheduling Engine:** Dynamically calculates optimal backup intervals based on drive risk metrics instead of static calendars.
* **Production Database Integration:** Automates failure state historical logs and asset mapping using standard relational SQL databases.
* **Interactive Operational Dashboard:** Visualizes machine metrics, predicted failure horizons, and active schedules cleanly using an embedded UI framework.

---

## 📂 Repository Structure

The project codebase is organized into modular pipelines for model ingestion, training, database management, and policy execution:

```text
├── db/
│   ├── schema.sql              # Relational tables configuration for assets & logs
│   └── drop_all.sql            # Utility script to safely teardown local environments
├── src/
│   ├── config.py               # Shared parameters, file paths, and environment settings
│   ├── dataset.py              # Ingestion utilities and feature generation templates
│   ├── load_backblaze.py       # Live data collector tailored for public telemetry schemas
│   ├── build_features.py       # Cleans SMART logs and flags target degradation vectors
│   ├── train_model.py          # Trains and registers the predictive failure engine
│   ├── db_utils.py             # Reusable abstraction layer for raw SQL transactions
│   ├── init_db.py              # Bootstrap automation for local/target database clusters
│   ├── decision_layer.py       # Orchestrates adaptive scheduling rules per asset risk band
│   ├── run_decision_cycle.py   # Continuous engine cron executor 
│   └── dashboard.py            # Operational visual interface logic
├── baseline_risk_model.joblib  # Pre-compiled deployment package of the risk estimator
├── requirements.txt.txt        # System library requirements manifest
└── OUTCOME GRAPHS.pdf          # Performance verification plots and accuracy curves
```

---

## 🛠️ System Components

### 1. Ingestion & Transformation Pipeline
* **`load_backblaze.py`**: Fetches external hardware failure benchmarks and matches raw hard drive formats to local storage matrices.
* **`dataset.py` & `build_features.py`**: Reads raw time-series measurements, handles missing sensor data, and extracts relevant indicators (such as reallocated sector counts, command timeouts, and current pending sector counts).

### 2. Predictive Engine
* **`train_model.py`**: Implements training loops, applies stratified sampling to handle data imbalances, and serializes the resulting model into `baseline_risk_model.joblib`.

### 3. Management Layer
* **`decision_layer.py`**: Houses the dynamic scheduling logic:
    * **Low Risk:** Normal backup cycles (e.g., weekly).
    * **Medium Risk:** Increased telemetry logging and shortened backup cycles (e.g., daily).
    * **High Risk:** Triggers immediate data evacuation tasks and alerts system administrators.
* **`run_decision_cycle.py`**: Runs regularly to update risks, check storage metrics, and trigger alerts.

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.8 or higher
* An active relational database instance (PostgreSQL / SQLite / MySQL compatible)

### 1. Clone & Environment Setup
Clone this repository to your local system and configure your environment variables:
```bash
git clone https://github.com
cd Predictive-Failure-Detection-and-Adaptive-Backup-Scheduling

# Copy and update required system connection parameters
cp .env.example .env
```

### 2. Install Required Dependencies
Install the required packages using the project requirements file:
```bash
pip install -r requirements.txt.txt
```

### 3. Database Initialization
Prepare the target tables by executing the database bootstrap module:
```bash
python src/init_db.py
```

---

## 📈 Running the Application

### Phase 1: Train the Risk Predictor
To build a custom model using local hardware profiles, process your features and run the training pipeline:
```bash
python src/build_features.py
python src/train_model.py
```

### Phase 2: Start the Continuous Scheduling Engine
Run the core control loop to begin tracking hardware risks and managing backup windows:
```bash
python src/run_decision_cycle.py
```

### Phase 3: Launch the Operational Dashboard
Launch the interface to monitor current drive risks, schedule changes, and infrastructure performance metrics:
```bash
python src/dashboard.py
```
