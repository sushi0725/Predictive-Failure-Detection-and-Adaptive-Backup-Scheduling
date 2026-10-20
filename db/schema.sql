-- Production Infrastructure Storage Schema Definition Matrix
-- Structured with safe conditional creation execution controls

CREATE TABLE IF NOT EXISTS storage_nodes (
    node_id INT PRIMARY KEY,
    datacenter VARCHAR(100),
    cluster_id INT,
    vault_id INT,
    pod_id INT,
    node_name VARCHAR(100) DEFAULT 'unnamed_node',
    ip_address VARCHAR(45),
    status VARCHAR(20) DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS disks (
    disk_id SERIAL PRIMARY KEY,
    serial_number VARCHAR(100) UNIQUE NOT NULL,
    model VARCHAR(100),
    capacity_bytes BIGINT,
    node_id INT REFERENCES storage_nodes(node_id),
    status VARCHAR(20) DEFAULT 'active',
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS smart_telemetry (
    telemetry_id BIGSERIAL PRIMARY KEY,
    disk_id INT REFERENCES disks(disk_id),
    serial_number VARCHAR(100) NOT NULL,
    model VARCHAR(100),
    timestamp TIMESTAMP NOT NULL,
    date DATE,
    record_date TIMESTAMP NOT NULL,
    smart_1_raw BIGINT DEFAULT 0,
    smart_3_raw BIGINT DEFAULT 0,
    smart_4_raw BIGINT DEFAULT 0,
    smart_5_raw BIGINT DEFAULT 0,
    smart_7_raw BIGINT DEFAULT 0,
    smart_9_raw BIGINT DEFAULT 0,
    smart_10_raw BIGINT DEFAULT 0,
    smart_12_raw BIGINT DEFAULT 0,
    smart_187_raw BIGINT DEFAULT 0,
    smart_188_raw BIGINT DEFAULT 0,
    smart_189_raw BIGINT DEFAULT 0,
    smart_190_raw BIGINT DEFAULT 0,
    smart_192_raw BIGINT DEFAULT 0,
    smart_193_raw BIGINT DEFAULT 0,
    smart_194_raw BIGINT DEFAULT 0,
    smart_197_raw BIGINT DEFAULT 0,
    smart_198_raw BIGINT DEFAULT 0,
    smart_199_raw BIGINT DEFAULT 0,
    smart_240_raw BIGINT DEFAULT 0,
    smart_241_raw BIGINT DEFAULT 0,
    smart_242_raw BIGINT DEFAULT 0,
    failure INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS features (
    feature_id BIGSERIAL PRIMARY KEY,
    telemetry_id BIGINT REFERENCES smart_telemetry(telemetry_id),
    disk_id INT REFERENCES disks(disk_id),
    serial_number VARCHAR(100) NOT NULL,
    model VARCHAR(100),
    timestamp TIMESTAMP NOT NULL,
    feature_date TIMESTAMP NOT NULL,
    smart_1_raw BIGINT DEFAULT 0,
    smart_3_raw BIGINT DEFAULT 0,
    smart_4_raw BIGINT DEFAULT 0,
    smart_5_raw BIGINT DEFAULT 0,
    smart_7_raw BIGINT DEFAULT 0,
    smart_9_raw BIGINT DEFAULT 0,
    smart_10_raw BIGINT DEFAULT 0,
    smart_12_raw BIGINT DEFAULT 0,
    smart_187_raw BIGINT DEFAULT 0,
    smart_188_raw BIGINT DEFAULT 0,
    smart_189_raw BIGINT DEFAULT 0,
    smart_190_raw BIGINT DEFAULT 0,
    smart_192_raw BIGINT DEFAULT 0,
    smart_193_raw BIGINT DEFAULT 0,
    smart_194_raw BIGINT DEFAULT 0,
    smart_197_raw BIGINT DEFAULT 0,
    smart_198_raw BIGINT DEFAULT 0,
    smart_199_raw BIGINT DEFAULT 0,
    smart_240_raw BIGINT DEFAULT 0,
    smart_241_raw BIGINT DEFAULT 0,
    smart_242_raw BIGINT DEFAULT 0,
    failure INT DEFAULT 0,
    fails_within_horizon INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS risk_scores (
    score_id BIGSERIAL UNIQUE,
    disk_id INT REFERENCES disks(disk_id),
    timestamp TIMESTAMP NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(50),
    top_factors TEXT,
    PRIMARY KEY (disk_id, timestamp)
);

CREATE TABLE IF NOT EXISTS adaptive_backup_policies (
    policy_id SERIAL PRIMARY KEY,
    policy_version VARCHAR(50) UNIQUE NOT NULL,
    risk_tier VARCHAR(50) NOT NULL,
    threshold_low DOUBLE PRECISION NOT NULL,
    threshold_high DOUBLE PRECISION NOT NULL,
    action VARCHAR(100),
    urgency VARCHAR(50),
    backup_interval_hours INT
);

CREATE TABLE IF NOT EXISTS adaptive_backup_events (
    event_id BIGSERIAL PRIMARY KEY,
    score_id BIGINT REFERENCES risk_scores(score_id),
    policy_version VARCHAR(50),
    risk_tier VARCHAR(50),
    action VARCHAR(100),
    urgency VARCHAR(50),
    backup_interval_hours INT,
    next_backup_due TIMESTAMP,
    reason TEXT,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(score_id, policy_version)
);

CREATE TABLE IF NOT EXISTS model_drift_monitoring (
    monitoring_id SERIAL PRIMARY KEY,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    roc_auc DOUBLE PRECISION,
    pr_auc DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    baseline_failure_rate DOUBLE PRECISION,
    drift_detected BOOLEAN DEFAULT FALSE
);

-- Core Operational Performance Optimization Indexes
CREATE INDEX IF NOT EXISTS idx_telemetry_lookup ON smart_telemetry (disk_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_features_lookup ON features (disk_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_scores_lookup ON risk_scores (disk_id, timestamp);

-- Unified Core Analytical Infrastructure Views
CREATE OR REPLACE VIEW v_latest_risk AS
WITH ranked_scores AS (
    SELECT r.*, d.serial_number, d.model, d.status as disk_status,
           ROW_NUMBER() OVER (PARTITION BY r.disk_id ORDER BY r.timestamp DESC) as rn
    FROM risk_scores r
    JOIN disks d ON r.disk_id = d.disk_id
)
SELECT score_id, disk_id, serial_number, model, timestamp, risk_score, model_version, top_factors, disk_status
FROM ranked_scores
WHERE rn = 1;

CREATE OR REPLACE VIEW vw_disk_latest_health AS
SELECT disk_id, serial_number, model, risk_score,
       CASE 
           WHEN risk_score >= 0.75 THEN 'CRITICAL'
           WHEN risk_score >= 0.15 THEN 'WARNING'
           ELSE 'HEALTHY'
       END as operational_health_tier,
       top_factors
FROM v_latest_risk;

CREATE OR REPLACE VIEW vw_urgent_backup_queue AS
SELECT v.score_id, v.disk_id, v.serial_number, v.risk_score, v.top_factors, p.action, p.urgency
FROM v_latest_risk v
JOIN adaptive_backup_policies p ON 
    (v.risk_score >= p.threshold_low AND v.risk_score < p.threshold_high)
WHERE p.urgency IN ('HIGH', 'CRITICAL')
ORDER BY v.risk_score DESC;
