-- f3_app_conversion_rate_24hr: the app's conversion rate across ALL IPs
-- over the trailing 24h, strictly prior. A persistently low app rate means
-- the app's traffic is being inflated (publisher-side fraud), regardless
-- of which IP is clicking right now. Same frame discipline and NULL
-- convention as the ip variant.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       AVG(is_attributed) OVER (
         PARTITION BY app ORDER BY click_time
         RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
