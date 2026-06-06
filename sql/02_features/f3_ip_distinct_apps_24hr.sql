-- f3_ip_distinct_apps_24hr: distinct apps this IP clicked in the trailing
-- 24h, strictly prior. EDA: fraud spreads thin across many apps; a human
-- clicks a handful. No labels involved; empty window -> honest 0.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       COUNT(DISTINCT app) OVER (
         PARTITION BY ip ORDER BY click_time
         RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
