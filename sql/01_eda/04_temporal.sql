-- Clicks per actual hour-bucket across the dataset (one row per hour of wall time).
-- Distinguishes from 01_class_balance's `by_hour_of_day` (which folds all days
-- together): here we keep the day distinction so you can see the actual time
-- series shape and any day-over-day shift in positive rate.

COPY (
    SELECT
        DATE_TRUNC('hour', click_time)        AS hour_bucket,
        COUNT(*)                              AS n_clicks,
        SUM(is_attributed)                    AS n_attributed,
        AVG(is_attributed::DOUBLE)            AS positive_rate
    FROM clicks
    GROUP BY DATE_TRUNC('hour', click_time)
    ORDER BY hour_bucket
)
TO 'reports/eda/04_temporal.csv' (HEADER, FORMAT CSV);
