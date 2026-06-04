-- Time from click to attribution (only for clicks where is_attributed = 1).
-- A very-short lag suggests "click already had install intent" rather than the
-- click *causing* the install. A long lag may indicate delayed attribution from
-- a re-engagement campaign. Both shapes matter for evaluating whether the
-- model's prediction window matches the attribution window.

COPY (
    WITH lags AS (
        SELECT EXTRACT(epoch FROM attributed_time - click_time) AS lag_seconds
        FROM clicks
        WHERE attributed_time IS NOT NULL
    )
    SELECT 'count'  AS stat, COUNT(*)::DOUBLE                  AS value FROM lags
    UNION ALL SELECT 'min',    MIN(lag_seconds)                FROM lags
    UNION ALL SELECT 'p01',    QUANTILE_CONT(lag_seconds, 0.01) FROM lags
    UNION ALL SELECT 'p10',    QUANTILE_CONT(lag_seconds, 0.10) FROM lags
    UNION ALL SELECT 'p25',    QUANTILE_CONT(lag_seconds, 0.25) FROM lags
    UNION ALL SELECT 'p50',    QUANTILE_CONT(lag_seconds, 0.50) FROM lags
    UNION ALL SELECT 'p75',    QUANTILE_CONT(lag_seconds, 0.75) FROM lags
    UNION ALL SELECT 'p90',    QUANTILE_CONT(lag_seconds, 0.90) FROM lags
    UNION ALL SELECT 'p95',    QUANTILE_CONT(lag_seconds, 0.95) FROM lags
    UNION ALL SELECT 'p99',    QUANTILE_CONT(lag_seconds, 0.99) FROM lags
    UNION ALL SELECT 'mean',   AVG(lag_seconds)                FROM lags
    UNION ALL SELECT 'max',    MAX(lag_seconds)                FROM lags
)
TO 'reports/eda/08_attributed_time_lag.csv' (HEADER, FORMAT CSV);
