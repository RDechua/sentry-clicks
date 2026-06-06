-- f2_ip_click_std_inter_arrival: stddev of this IP's inter-arrival gaps
-- over the trailing 24h, PRIOR gaps only. Low stddev = metronomic pacing
-- (automation); humans jitter. Two layers because windows can't nest:
-- lag() computes per-row gaps in the CTE, then a windowed stddev_samp
-- aggregates them. The window uses the same strictly-prior frame as the
-- counts; the current row's own gap is deliberately NOT included — it is
-- already its own feature (f2_inter_click_time_seconds), and this one
-- summarizes the rhythm BEFORE this click. NULL gaps (first clicks) are
-- ignored by the aggregate; fewer than two usable gaps -> NULL.
-- Contract: returns (row_id, value), one row per {source} row.
WITH gaps AS (
  SELECT row_id, ip, click_time,
         date_diff('second',
           lag(click_time) OVER (PARTITION BY ip ORDER BY click_time, row_id),
           click_time) AS gap_s
  FROM {source}
)
SELECT row_id,
       stddev_samp(gap_s) OVER (
         PARTITION BY ip ORDER BY click_time
         RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM gaps
