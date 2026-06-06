-- f2_clicks_per_ip_last_24hr: clicks from this IP in the trailing day.
-- Catches paced botnets (1 click/minute beats the 1h window; it does not
-- beat the 24h one). Same strictly-prior frame as the 1h variant.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       COUNT(*) OVER (
         PARTITION BY ip ORDER BY click_time
         RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
