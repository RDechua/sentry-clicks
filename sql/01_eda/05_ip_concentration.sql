-- IP click concentration: bucketed by clicks-per-IP, with the fraction of all
-- clicks coming from each bucket. The whole F2 (velocity) feature story rests
-- on whether the click distribution is heavy-tailed by IP.

COPY (
    WITH ip_counts AS (
        SELECT
            ip,
            COUNT(*)                       AS n_clicks,
            AVG(is_attributed::DOUBLE)     AS positive_rate
        FROM clicks
        GROUP BY ip
    ),
    bucketed AS (
        SELECT
            CASE
                WHEN n_clicks =      1                       THEN '01: 1'
                WHEN n_clicks BETWEEN  2 AND   10            THEN '02: 2-10'
                WHEN n_clicks BETWEEN 11 AND  100            THEN '03: 11-100'
                WHEN n_clicks BETWEEN 101 AND 1000           THEN '04: 101-1000'
                ELSE                                              '05: >1000'
            END                                AS clicks_bucket,
            COUNT(*)                           AS n_ips,
            SUM(n_clicks)                      AS n_clicks_total,
            AVG(positive_rate)                 AS mean_ip_positive_rate
        FROM ip_counts
        GROUP BY 1
    )
    SELECT
        clicks_bucket,
        n_ips,
        n_clicks_total,
        n_clicks_total::DOUBLE / SUM(n_clicks_total) OVER ()  AS fraction_of_all_clicks,
        mean_ip_positive_rate
    FROM bucketed
    ORDER BY clicks_bucket
)
TO 'reports/eda/05_ip_concentration.csv' (HEADER, FORMAT CSV);
