-- f1_os_id: pass-through of the os column (OS version code).
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id, os AS value FROM {source}
