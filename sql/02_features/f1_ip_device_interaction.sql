-- f1_ip_device_interaction: (ip, device) pair as a single categorical.
-- Same rationale as f1_ip_app_interaction — a device fleet behind one IP
-- is an infrastructure fingerprint.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id, CAST(ip AS VARCHAR) || '_' || CAST(device AS VARCHAR) AS value FROM {source}
