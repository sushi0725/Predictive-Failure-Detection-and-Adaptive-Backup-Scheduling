-- ============================================================================
-- Database Schema: Predictive Failure Detection and Adaptive Backup Scheduling
-- Course: Database Systems (BACSE202)
-- DBMS: PostgreSQL 18
-- ============================================================================

-- Drop existing tables/views if they exist (in reverse dependency order)
DROP VIEW IF EXISTS vw_urgent_backup_queue CASCADE;
DROP VIEW IF EXISTS vw_disk_latest_health CASCADE;
DROP TABLE IF EXISTS model_drift_monitoring CASCADE;
DROP TABLE IF EXISTS adaptive_backup_events CASCADE;
DROP TABLE IF EXISTS adaptive_backup_policies CASCADE;
DROP TABLE IF EXISTS health_predictions CASCADE;
DROP TABLE IF EXISTS server_system_metrics CASCADE;
DROP TABLE IF EXISTS smart_telemetry CASCADE;
DROP TABLE IF EXISTS disks CASCADE;
DROP TABLE IF EXISTS storage_nodes CASCADE;

-- ----------------------------------------------------------------------------
-- 1. Storage Nodes (Physical & Infrastructure Topology)
-- ----------------------------------------------------------------------------
CREATE TABLE storage_nodes (
    node_id VARCHAR(64) PRIMARY KEY,
    datacenter VARCHAR(32) NOT NULL,
    cluster_id VARCHAR(32),
    vault_id VARCHAR(32),
    pod_id VARCHAR(32),
    status VARCHAR(32) DEFAULT 'active' CHECK (status IN ('active', 'degraded', 'offline', 'maintenance')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE storage_nodes IS 'Physical server node or pod hosting storage drives';

-- ----------------------------------------------------------------------------
-- 2. Disks (Hardware Asset Registry)
-- ----------------------------------------------------------------------------
CREATE TABLE disks (
    serial_number VARCHAR(64) PRIMARY KEY,
    model VARCHAR(128) NOT NULL,
    capacity_bytes BIGINT NOT NULL,
    node_id VARCHAR(64) REFERENCES storage_nodes(node_id) ON DELETE SET NULL,
    pod_slot_num VARCHAR(32),
    first_seen_date DATE NOT NULL,
    status VARCHAR(32) DEFAULT 'active' CHECK (status IN ('active', 'warning', 'critical', 'failed', 'decommissioned')),
    failure_date DATE NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_disks_model ON disks(model);
CREATE INDEX idx_disks_node ON disks(node_id);
CREATE INDEX idx_disks_status ON disks(status);
COMMENT ON TABLE disks IS 'Hard disk drives and solid-state drives registered in infrastructure';

-- ----------------------------------------------------------------------------
-- 3. SMART Telemetry (Time-series drive health signals from Backblaze logs)
-- ----------------------------------------------------------------------------
CREATE TABLE smart_telemetry (
    telemetry_id BIGSERIAL PRIMARY KEY,
    serial_number VARCHAR(64) NOT NULL REFERENCES disks(serial_number) ON DELETE CASCADE,
    record_date DATE NOT NULL,
    failure INT NOT NULL DEFAULT 0 CHECK (failure IN (0, 1)),
    
    -- Critical S.M.A.R.T. metrics strongly correlated with hard drive failure
    smart_1_raw BIGINT,            -- Read Error Rate
    smart_1_normalized INT,
    smart_3_raw BIGINT,            -- Spin-Up Time
    smart_4_raw BIGINT,            -- Start/Stop Count
    smart_5_raw BIGINT,            -- Reallocated Sectors Count (critical)
    smart_5_normalized INT,
    smart_7_raw BIGINT,            -- Seek Error Rate
    smart_9_raw BIGINT,            -- Power-On Hours
    smart_9_normalized INT,
    smart_10_raw BIGINT,           -- Spin Retry Count
    smart_12_raw BIGINT,           -- Power Cycle Count
    smart_12_normalized INT,
    smart_187_raw BIGINT,          -- Reported Uncorrectable Errors (critical)
    smart_187_normalized INT,
    smart_188_raw BIGINT,          -- Command Timeout (critical)
    smart_188_normalized INT,
    smart_189_raw BIGINT,          -- High Fly Writes
    smart_190_raw BIGINT,          -- Airflow Temperature
    smart_192_raw BIGINT,          -- Power-Off Retract Count
    smart_193_raw BIGINT,          -- Load/Unload Cycle Count
    smart_194_raw BIGINT,          -- Temperature Celsius
    smart_194_normalized INT,
    smart_197_raw BIGINT,          -- Current Pending Sector Count (critical)
    smart_197_normalized INT,
    smart_198_raw BIGINT,          -- Offline Uncorrectable Sector Count (critical)
    smart_198_normalized INT,
    smart_199_raw BIGINT,          -- UltraDMA CRC Error Count
    smart_199_normalized INT,
    smart_240_raw BIGINT,          -- Head Flying Hours
    smart_241_raw BIGINT,          -- Total LBAs Written
    smart_242_raw BIGINT,          -- Total LBAs Read

    -- Flexible JSONB store for remaining arbitrary SMART attributes
    additional_smart_attributes JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_disk_record_date UNIQUE (serial_number, record_date)
);

CREATE INDEX idx_telemetry_serial_date ON smart_telemetry(serial_number, record_date DESC);
CREATE INDEX idx_telemetry_date ON smart_telemetry(record_date);
CREATE INDEX idx_telemetry_failure ON smart_telemetry(failure) WHERE failure = 1;
COMMENT ON TABLE smart_telemetry IS 'Daily hardware telemetry logs recording S.M.A.R.T. statistics';

-- ----------------------------------------------------------------------------
-- 4. Server System Metrics (OS-level multi-signal telemetry)
-- ----------------------------------------------------------------------------
CREATE TABLE server_system_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    node_id VARCHAR(64) NOT NULL REFERENCES storage_nodes(node_id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL,
    cpu_utilization_pct NUMERIC(5, 2) CHECK (cpu_utilization_pct >= 0 AND cpu_utilization_pct <= 100),
    memory_utilization_pct NUMERIC(5, 2) CHECK (memory_utilization_pct >= 0 AND memory_utilization_pct <= 100),
    disk_io_latency_ms NUMERIC(8, 2),
    queue_depth NUMERIC(8, 2),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sys_metrics_node_time ON server_system_metrics(node_id, recorded_at DESC);
COMMENT ON TABLE server_system_metrics IS 'System and OS-level telemetry signals';

-- ----------------------------------------------------------------------------
-- 5. Health Predictions (ML inference output & Explainability)
-- ----------------------------------------------------------------------------
CREATE TABLE health_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    serial_number VARCHAR(64) NOT NULL REFERENCES disks(serial_number) ON DELETE CASCADE,
    prediction_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    risk_score NUMERIC(5, 4) NOT NULL CHECK (risk_score >= 0.0 AND risk_score <= 1.0),
    predicted_horizon_days INT NOT NULL DEFAULT 7,
    model_name VARCHAR(64) NOT NULL,
    shap_feature_attributions JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_serial ON health_predictions(serial_number, prediction_timestamp DESC);
CREATE INDEX idx_predictions_risk ON health_predictions(risk_score DESC);
COMMENT ON TABLE health_predictions IS 'Machine learning failure probability forecasts and SHAP feature attributions';

-- ----------------------------------------------------------------------------
-- 6. Adaptive Backup Policies (Cost-aware MDP / Decision Layer Configuration)
-- ----------------------------------------------------------------------------
CREATE TABLE adaptive_backup_policies (
    policy_id SERIAL PRIMARY KEY,
    policy_name VARCHAR(64) NOT NULL UNIQUE,
    min_risk_threshold NUMERIC(5, 4) NOT NULL,
    max_risk_threshold NUMERIC(5, 4) NOT NULL,
    recommended_action VARCHAR(64) NOT NULL CHECK (recommended_action IN ('do_nothing', 'monitor_closely', 'snapshot_now', 'prepare_failover')),
    backup_frequency_hours INT NOT NULL,
    action_cost_weight NUMERIC(6, 2) NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE adaptive_backup_policies IS 'Rule-based and RL policy mappings from risk score to backup action';

-- Seed default adaptive policies matching the project decision layer
INSERT INTO adaptive_backup_policies (policy_name, min_risk_threshold, max_risk_threshold, recommended_action, backup_frequency_hours, action_cost_weight) VALUES
('Normal / Low Risk', 0.0000, 0.2000, 'do_nothing', 24, 0.00),
('Moderate Risk / Monitor', 0.2001, 0.5000, 'monitor_closely', 12, 10.00),
('High Risk / Urgent Snapshot', 0.5001, 0.8000, 'snapshot_now', 2, 50.00),
('Critical Risk / Prepare Failover', 0.8001, 1.0000, 'prepare_failover', 1, 150.00);

-- ----------------------------------------------------------------------------
-- 7. Adaptive Backup Events (Triggered backup executions and audit log)
-- ----------------------------------------------------------------------------
CREATE TABLE adaptive_backup_events (
    backup_event_id BIGSERIAL PRIMARY KEY,
    serial_number VARCHAR(64) NOT NULL REFERENCES disks(serial_number) ON DELETE CASCADE,
    action_taken VARCHAR(64) NOT NULL CHECK (action_taken IN ('do_nothing', 'monitor_closely', 'snapshot_now', 'prepare_failover')),
    trigger_source VARCHAR(32) NOT NULL DEFAULT 'adaptive_policy' CHECK (trigger_source IN ('adaptive_policy', 'manual_operator', 'scheduled')),
    risk_score_at_trigger NUMERIC(5, 4),
    backup_status VARCHAR(32) NOT NULL DEFAULT 'completed' CHECK (backup_status IN ('pending', 'running', 'completed', 'failed')),
    snapshot_size_bytes BIGINT,
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    action_cost NUMERIC(8, 2),
    notes TEXT
);

CREATE INDEX idx_backup_events_serial ON adaptive_backup_events(serial_number, started_at DESC);
CREATE INDEX idx_backup_events_status ON adaptive_backup_events(backup_status);
COMMENT ON TABLE adaptive_backup_events IS 'Log of backup and failover actions triggered by the adaptive scheduler';

-- ----------------------------------------------------------------------------
-- 8. Model Drift Monitoring (Feedback & Retraining Loop)
-- ----------------------------------------------------------------------------
CREATE TABLE model_drift_monitoring (
    drift_id SERIAL PRIMARY KEY,
    model_name VARCHAR(64) NOT NULL,
    evaluation_date DATE NOT NULL,
    rolling_f1_score NUMERIC(5, 4) NOT NULL,
    rolling_precision NUMERIC(5, 4) NOT NULL,
    rolling_recall NUMERIC(5, 4) NOT NULL,
    drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
    retraining_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    evaluated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_drift_model_date ON model_drift_monitoring(model_name, evaluation_date DESC);
COMMENT ON TABLE model_drift_monitoring IS 'Tracking accuracy decay and concept drift to trigger model retraining';

-- ----------------------------------------------------------------------------
-- 9. Analytical Views
-- ----------------------------------------------------------------------------

-- View: Disk Latest Health Overview
CREATE OR REPLACE VIEW vw_disk_latest_health AS
SELECT 
    d.serial_number,
    d.model,
    d.capacity_bytes,
    d.status AS disk_status,
    d.node_id,
    latest_tel.record_date AS last_telemetry_date,
    latest_tel.smart_5_raw AS reallocated_sectors,
    latest_tel.smart_187_raw AS reported_uncorrectable,
    latest_tel.smart_197_raw AS pending_sectors,
    latest_tel.smart_194_raw AS temperature_c,
    latest_pred.risk_score,
    latest_pred.prediction_timestamp AS last_prediction_at
FROM disks d
LEFT JOIN LATERAL (
    SELECT * FROM smart_telemetry st 
    WHERE st.serial_number = d.serial_number 
    ORDER BY record_date DESC LIMIT 1
) latest_tel ON TRUE
LEFT JOIN LATERAL (
    SELECT * FROM health_predictions hp 
    WHERE hp.serial_number = d.serial_number 
    ORDER BY prediction_timestamp DESC LIMIT 1
) latest_pred ON TRUE;

-- View: Urgent Backup Action Queue
CREATE OR REPLACE VIEW vw_urgent_backup_queue AS
SELECT 
    d.serial_number,
    d.model,
    d.node_id,
    hp.risk_score,
    p.recommended_action,
    p.backup_frequency_hours,
    hp.prediction_timestamp,
    hp.shap_feature_attributions
FROM health_predictions hp
JOIN disks d ON d.serial_number = hp.serial_number
JOIN adaptive_backup_policies p 
    ON hp.risk_score >= p.min_risk_threshold 
   AND hp.risk_score <= p.max_risk_threshold
WHERE p.recommended_action IN ('snapshot_now', 'prepare_failover')
ORDER BY hp.risk_score DESC;

-- View: Drive Lifecycle Compatibility View
CREATE OR REPLACE VIEW v_drive_lifecycle AS
SELECT
    d.serial_number,
    d.model,
    d.status,
    (d.status = 'failed' OR EXISTS (
        SELECT 1 FROM smart_telemetry st WHERE st.serial_number = d.serial_number AND st.failure = 1
    )) AS failed,
    d.first_seen_date,
    d.failure_date
FROM disks d;

-- ----------------------------------------------------------------------------
-- 10. ML Engineered Feature Store & Risk Scores
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    serial_number VARCHAR(64) NOT NULL REFERENCES disks(serial_number) ON DELETE CASCADE,
    feature_date DATE NOT NULL,
    smart_5_raw DOUBLE PRECISION,
    smart_9_raw DOUBLE PRECISION,
    smart_12_raw DOUBLE PRECISION,
    smart_187_raw DOUBLE PRECISION,
    smart_188_raw DOUBLE PRECISION,
    smart_192_raw DOUBLE PRECISION,
    smart_193_raw DOUBLE PRECISION,
    smart_197_raw DOUBLE PRECISION,
    smart_198_raw DOUBLE PRECISION,
    temperature_c DOUBLE PRECISION,
    smart_5_raw_mean_7d DOUBLE PRECISION,
    smart_5_raw_slope_7d DOUBLE PRECISION,
    smart_5_raw_delta_7d DOUBLE PRECISION,
    smart_187_raw_mean_7d DOUBLE PRECISION,
    smart_187_raw_slope_7d DOUBLE PRECISION,
    smart_187_raw_delta_7d DOUBLE PRECISION,
    smart_188_raw_mean_7d DOUBLE PRECISION,
    smart_188_raw_slope_7d DOUBLE PRECISION,
    smart_188_raw_delta_7d DOUBLE PRECISION,
    smart_197_raw_mean_7d DOUBLE PRECISION,
    smart_197_raw_slope_7d DOUBLE PRECISION,
    smart_197_raw_delta_7d DOUBLE PRECISION,
    smart_198_raw_mean_7d DOUBLE PRECISION,
    smart_198_raw_slope_7d DOUBLE PRECISION,
    smart_198_raw_delta_7d DOUBLE PRECISION,
    fails_within_horizon INTEGER,
    PRIMARY KEY (serial_number, feature_date)
);

CREATE TABLE IF NOT EXISTS risk_scores (
    serial_number VARCHAR(64) NOT NULL REFERENCES disks(serial_number) ON DELETE CASCADE,
    as_of_date DATE NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    top_factors TEXT,
    computed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (serial_number, as_of_date, model_version)
);

CREATE OR REPLACE VIEW v_latest_risk AS
SELECT DISTINCT ON (serial_number)
    serial_number, as_of_date, risk_score, top_factors, model_version, computed_at
FROM risk_scores
ORDER BY serial_number, as_of_date DESC, computed_at DESC;
