-- f1_device_id: pass-through of the device column (device type code).
-- device=0 converts at ~60x baseline (Week 1 EDA) — the model should see it raw.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id, device AS value FROM {source}
