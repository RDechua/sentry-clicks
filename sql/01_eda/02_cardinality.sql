-- Distinct value counts for each categorical column.
-- Sets the search space for feature engineering — IPs are the high-cardinality
-- column where velocity and aggregate features will live.

COPY (
    SELECT 'ip'      AS column_name, COUNT(DISTINCT ip)      AS n_distinct FROM clicks UNION ALL
    SELECT 'app',     COUNT(DISTINCT app)     FROM clicks UNION ALL
    SELECT 'device',  COUNT(DISTINCT device)  FROM clicks UNION ALL
    SELECT 'os',      COUNT(DISTINCT os)      FROM clicks UNION ALL
    SELECT 'channel', COUNT(DISTINCT channel) FROM clicks
    ORDER BY n_distinct DESC
)
TO 'reports/eda/02_cardinality.csv' (HEADER, FORMAT CSV);
