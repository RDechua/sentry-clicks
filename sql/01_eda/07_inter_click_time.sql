-- Inter-click time per IP: seconds between consecutive clicks from the same IP.
-- This is the raw signal F2 (velocity / rolling-window counts) will lean on.
-- Build guide explicitly calls out NOT skipping this — even though the query
-- uses a window function and is "harder", the distribution shape (especially
-- the low-percentile values) tells you what time windows F2 should use.

COPY (
    WITH gaps AS (
        SELECT
            EXTRACT(epoch FROM click_time - LAG(click_time) OVER (
                PARTITION BY ip ORDER BY click_time
            )) AS gap_seconds
        FROM clicks
    ),
    nonnull_gaps AS (
        SELECT gap_seconds FROM gaps WHERE gap_seconds IS NOT NULL
    )
    SELECT 'count'  AS stat, COUNT(*)::DOUBLE                  AS value FROM nonnull_gaps
    UNION ALL SELECT 'min',    MIN(gap_seconds)                FROM nonnull_gaps
    UNION ALL SELECT 'p01',    QUANTILE_CONT(gap_seconds, 0.01) FROM nonnull_gaps
    UNION ALL SELECT 'p10',    QUANTILE_CONT(gap_seconds, 0.10) FROM nonnull_gaps
    UNION ALL SELECT 'p25',    QUANTILE_CONT(gap_seconds, 0.25) FROM nonnull_gaps
    UNION ALL SELECT 'p50',    QUANTILE_CONT(gap_seconds, 0.50) FROM nonnull_gaps
    UNION ALL SELECT 'p75',    QUANTILE_CONT(gap_seconds, 0.75) FROM nonnull_gaps
    UNION ALL SELECT 'p90',    QUANTILE_CONT(gap_seconds, 0.90) FROM nonnull_gaps
    UNION ALL SELECT 'p95',    QUANTILE_CONT(gap_seconds, 0.95) FROM nonnull_gaps
    UNION ALL SELECT 'p99',    QUANTILE_CONT(gap_seconds, 0.99) FROM nonnull_gaps
    UNION ALL SELECT 'mean',   AVG(gap_seconds)                FROM nonnull_gaps
    UNION ALL SELECT 'max',    MAX(gap_seconds)                FROM nonnull_gaps
)
TO 'reports/eda/07_inter_click_time.csv' (HEADER, FORMAT CSV);
