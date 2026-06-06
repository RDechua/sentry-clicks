-- f3_ip_app_conversion_rate_24hr: the (ip, app) PAIR's conversion rate
-- over the trailing 24h, strictly prior. The expected dominant feature
-- (public solutions on this dataset agree): "this IP clicks this app and
-- never installs" is the core fraud signature — EDA: 99.7% of pairs with
-- >=5 clicks never convert.
--
-- Small denominators are NOT smoothed here: the raw rate ships alongside
-- its denominator (f3_ip_app_clicks_24hr) and the tree model learns its
-- own discounting (decisions.md, Task 3.2). Same frame discipline and
-- NULL convention as every F3 rate.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       AVG(is_attributed) OVER (
         PARTITION BY ip, app ORDER BY click_time
         RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
