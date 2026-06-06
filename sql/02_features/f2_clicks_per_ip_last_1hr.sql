-- f2_clicks_per_ip_last_1hr: clicks from this IP in the trailing hour.
-- Catches click-farm bursts (Tier 1: many clicks, minutes apart).
--
-- Strictly-prior frame per CLAUDE.md 3.4: [t-1h, t-1ms] excludes the
-- current row AND its same-second peers. (EXCLUDE GROUP is the verified
-- equivalent — 0 diffs over the train split; AND CURRENT ROW and
-- EXCLUDE CURRENT ROW both leak, the latter via same-second peers.)
-- Bounds are inclusive both ends: a click exactly 1h prior counts.
-- Empty window -> 0: an honest "no history", not a NULL case.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       COUNT(*) OVER (
         PARTITION BY ip ORDER BY click_time
         RANGE BETWEEN INTERVAL 1 HOUR PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
