-- f3_ip_distinct_oses_24hr: distinct OS codes behind this IP in the
-- trailing 24h, strictly prior. Same fingerprint family as devices.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       COUNT(DISTINCT os) OVER (
         PARTITION BY ip ORDER BY click_time
         RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
