-- f3_app_clicks_alltime: strictly-prior all-time click count for
-- this app (all IPs) — the denominator companion to the all-time rate, same
-- thin-history-discounting role as the 24h pair denominator.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       COUNT(*) OVER (
         PARTITION BY app ORDER BY click_time
         RANGE BETWEEN UNBOUNDED PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
