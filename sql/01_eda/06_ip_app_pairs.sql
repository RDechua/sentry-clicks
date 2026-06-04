-- IP-app pair conversion rates: the build guide flags this as the strongest
-- single feature signal. For pairs with >= 5 clicks, what does the conversion
-- rate distribution look like?
-- Output: bucketed by positive_rate, showing how many (ip, app) pairs sit at
-- each level. The "0% pairs" and "100% pairs" buckets are the predictive ones.

COPY (
    WITH pairs AS (
        SELECT
            ip,
            app,
            COUNT(*)                       AS n_clicks,
            SUM(is_attributed)             AS n_attributed,
            AVG(is_attributed::DOUBLE)     AS positive_rate
        FROM clicks
        GROUP BY ip, app
        HAVING COUNT(*) >= 5
    )
    SELECT
        CASE
            WHEN positive_rate = 0.0  THEN '01: 0% (never converts)'
            WHEN positive_rate < 0.05 THEN '02: 0-5%'
            WHEN positive_rate < 0.20 THEN '03: 5-20%'
            WHEN positive_rate < 0.50 THEN '04: 20-50%'
            WHEN positive_rate < 1.0  THEN '05: 50-99%'
            ELSE                           '06: 100% (always converts)'
        END                                AS positive_rate_bucket,
        COUNT(*)                           AS n_pairs,
        SUM(n_clicks)                      AS n_clicks_total,
        SUM(n_attributed)                  AS n_attributed_total
    FROM pairs
    GROUP BY 1
    ORDER BY 1
)
TO 'reports/eda/06_ip_app_pairs.csv' (HEADER, FORMAT CSV);
