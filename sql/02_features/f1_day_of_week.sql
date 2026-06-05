-- f1_day_of_week: ISO day of week (Monday=1 .. Sunday=7) from click_time.
-- isodow over DuckDB's Sunday-zero dayofweek: ISO is unambiguous across
-- engines and readers. Only Mon-Thu appear in the TalkingData window.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id, isodow(click_time) AS value FROM {source}
