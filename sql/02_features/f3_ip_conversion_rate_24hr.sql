-- f3_ip_conversion_rate_24hr: this IP's conversion rate over its clicks in
-- the trailing 24h, STRICTLY PRIOR. This aggregates LABELS (is_attributed)
-- of past clicks — the highest-leakage-risk feature family in the project:
--   * the current row is outside the frame (ms-preceding bound), so its
--     own label can never enter its own value;
--   * same-second peers are outside too (the EXCLUDE CURRENT ROW trap);
--   * an IP with no prior clicks gets NULL (AVG over an empty frame) —
--     never 0.0 ("known bad") or 1.0 ("its own label leaked").
-- Offline caveat (decisions.md, Avoiding label leakage in F3): prior
-- labels are assumed knowable at decision time; in production attribution
-- arrives late and this feature needs a label-maturation window.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       AVG(is_attributed) OVER (
         PARTITION BY ip ORDER BY click_time
         RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
