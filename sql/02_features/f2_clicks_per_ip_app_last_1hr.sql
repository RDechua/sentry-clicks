-- f2_clicks_per_ip_app_last_1hr: clicks from this (ip, app) pair in the
-- trailing hour. Distinguishes "one IP hammering one app" (targeted
-- inflation) from "one IP spraying many apps" (EDA: fraud spreads thin).
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       COUNT(*) OVER (
         PARTITION BY ip, app ORDER BY click_time
         RANGE BETWEEN INTERVAL 1 HOUR PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
