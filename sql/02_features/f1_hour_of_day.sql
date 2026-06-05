-- f1_hour_of_day: hour (0-23) extracted from click_time.
-- Raw integer, no cyclic encoding: LightGBM trees split on thresholds and
-- handle non-monotonic hour effects natively (decisions.md, Task 2.3).
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id, EXTRACT(hour FROM click_time) AS value FROM {source}
