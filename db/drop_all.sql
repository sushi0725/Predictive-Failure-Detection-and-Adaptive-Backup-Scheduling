-- Comprehensive Master Teardown Cascade Matrix
-- Views must be dropped first to avoid transaction lock errors

DROP VIEW IF EXISTS vw_urgent_backup_queue CASCADE;
DROP VIEW IF EXISTS vw_disk_latest_health CASCADE;
DROP VIEW IF EXISTS v_latest_risk CASCADE;
DROP VIEW IF EXISTS risk_scores CASCADE;
DROP VIEW IF EXISTS smart_telemetry CASCADE;

-- Clear out any physical data tables cleanly
DROP TABLE IF EXISTS model_drift_monitoring CASCADE;
DROP TABLE IF EXISTS adaptive_backup_events CASCADE;
DROP TABLE IF EXISTS adaptive_backup_policies CASCADE;
DROP TABLE IF EXISTS risk_scores_records CASCADE;
DROP TABLE IF EXISTS risk_scores CASCADE;
DROP TABLE IF EXISTS features CASCADE;
DROP TABLE IF EXISTS smart_telemetry_records CASCADE;
DROP TABLE IF EXISTS smart_telemetry CASCADE;
DROP TABLE IF EXISTS disks CASCADE;
DROP TABLE IF EXISTS storage_nodes CASCADE;
