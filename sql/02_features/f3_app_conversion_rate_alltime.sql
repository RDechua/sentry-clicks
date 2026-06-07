-- f3_app_conversion_rate_alltime: conversion rate of this app (all IPs) over ALL
-- strictly-prior history (no 24h cap). Added after the Week 4 density
-- gate: the 24h cap forgets days 1-2 of the dataset, and long memory is
-- where the strongest honest signal lives. Same strictly-prior frame
-- discipline; NULL with no history.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       AVG(is_attributed) OVER (
         PARTITION BY app ORDER BY click_time
         RANGE BETWEEN UNBOUNDED PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
