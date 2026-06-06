-- f3_ip_app_clicks_24hr: the denominator companion to
-- f3_ip_app_conversion_rate_24hr — how many strictly-prior clicks the
-- rate was computed over. Lets the model discount thin-history rates
-- (a 0/1 pair and a 0/40 pair are different animals) without baking a
-- smoothing prior into the feature itself.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       COUNT(*) OVER (
         PARTITION BY ip, app ORDER BY click_time
         RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
