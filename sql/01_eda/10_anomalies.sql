-- Anomalous IPs — the F2/F3 signal candidates. Two buckets:
--   (a) IPs with unusually high click counts
--   (b) IPs touching many distinct apps
-- For the 100k-row train_sample, "high click count" is calibrated down from
-- the build guide's >10k (which assumes the full 200M-row dataset) to >500
-- so the sample produces interpretable rows. For full data, raise the floor.

COPY (
    WITH ip_stats AS (
        SELECT
            ip,
            COUNT(*)                              AS n_clicks,
            COUNT(DISTINCT app)                   AS n_distinct_apps,
            COUNT(DISTINCT channel)               AS n_distinct_channels,
            AVG(is_attributed::DOUBLE)            AS positive_rate
        FROM clicks
        GROUP BY ip
    )
    SELECT * FROM (
        SELECT
            'high_click_count' AS bucket,
            ip::BIGINT, n_clicks, n_distinct_apps, n_distinct_channels, positive_rate
        FROM ip_stats
        WHERE n_clicks > 500
        ORDER BY n_clicks DESC
        LIMIT 50
    )
    UNION ALL
    SELECT * FROM (
        SELECT
            'high_distinct_apps' AS bucket,
            ip::BIGINT, n_clicks, n_distinct_apps, n_distinct_channels, positive_rate
        FROM ip_stats
        WHERE n_distinct_apps >= 10
        ORDER BY n_distinct_apps DESC, n_clicks DESC
        LIMIT 50
    )
    ORDER BY bucket, n_clicks DESC
)
TO 'reports/eda/10_anomalies.csv' (HEADER, FORMAT CSV);
