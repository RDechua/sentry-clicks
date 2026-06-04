-- Null patterns. Specifically confirms the documented invariant that
-- attributed_time IS NULL iff is_attributed = 0.
-- If any is_attributed=1 row has NULL attributed_time, or any is_attributed=0
-- row has a non-null attributed_time, the source data has been tampered with.

COPY (
    SELECT
        is_attributed,
        COUNT(*)                                                       AS n,
        COUNT(*) FILTER (WHERE attributed_time IS NULL)                AS n_null_attributed_time,
        COUNT(*) FILTER (WHERE attributed_time IS NOT NULL)            AS n_nonnull_attributed_time
    FROM clicks
    GROUP BY is_attributed
    ORDER BY is_attributed
)
TO 'reports/eda/09_null_patterns.csv' (HEADER, FORMAT CSV);
