-- f3_app_distinct_ips_24hr: distinct IPs that clicked this app in the
-- trailing 24h, strictly prior. The app-side degree from the bipartite
-- ip-app click graph (the one F4 "graph feature" that is mechanically
-- just a window aggregate — see decisions.md, F4 descope): a sudden
-- audience explosion on one app is publisher-side inflation.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       COUNT(DISTINCT ip) OVER (
         PARTITION BY app ORDER BY click_time
         RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
