-- f2_inter_click_time_seconds: seconds since this IP's previous click.
-- 0 = same-second pair, the EDA #5 bot fingerprint. First click from an
-- IP -> NULL (the honest no-prior-data value; LightGBM handles NULL
-- natively — decisions.md, Task 2.4). The row_id tiebreaker makes tie
-- order among same-second clicks deterministic; without it, lag() order
-- is unspecified.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       date_diff('second',
         lag(click_time) OVER (PARTITION BY ip ORDER BY click_time, row_id),
         click_time) AS value
FROM {source}
