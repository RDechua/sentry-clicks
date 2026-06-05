-- f1_app_id: pass-through of the app column (app being advertised).
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id, app AS value FROM {source}
