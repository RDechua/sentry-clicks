-- Top 20 values for each categorical column with their fraud rates.
-- Combines all five columns into one CSV with a `col` discriminator so the
-- output stays compact (one file, ~100 rows).
-- Each block uses a subquery so the inner LIMIT applies per-column before
-- the UNION concatenates them.

COPY (
    SELECT * FROM (
        SELECT 'ip' AS col, ip::BIGINT AS value,
               COUNT(*) AS n_clicks, AVG(is_attributed::DOUBLE) AS positive_rate
        FROM clicks GROUP BY ip
        ORDER BY n_clicks DESC LIMIT 20
    )
    UNION ALL
    SELECT * FROM (
        SELECT 'app', app::BIGINT,
               COUNT(*), AVG(is_attributed::DOUBLE)
        FROM clicks GROUP BY app
        ORDER BY COUNT(*) DESC LIMIT 20
    )
    UNION ALL
    SELECT * FROM (
        SELECT 'device', device::BIGINT,
               COUNT(*), AVG(is_attributed::DOUBLE)
        FROM clicks GROUP BY device
        ORDER BY COUNT(*) DESC LIMIT 20
    )
    UNION ALL
    SELECT * FROM (
        SELECT 'os', os::BIGINT,
               COUNT(*), AVG(is_attributed::DOUBLE)
        FROM clicks GROUP BY os
        ORDER BY COUNT(*) DESC LIMIT 20
    )
    UNION ALL
    SELECT * FROM (
        SELECT 'channel', channel::BIGINT,
               COUNT(*), AVG(is_attributed::DOUBLE)
        FROM clicks GROUP BY channel
        ORDER BY COUNT(*) DESC LIMIT 20
    )
    ORDER BY col, n_clicks DESC
)
TO 'reports/eda/03_top_values.csv' (HEADER, FORMAT CSV);
