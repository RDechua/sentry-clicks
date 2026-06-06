-- f3_ip_distinct_devices_24hr: distinct device codes behind this IP in the
-- trailing 24h, strictly prior. Many devices on one IP = NAT or a device
-- farm; combined with velocity the model can tell which.
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id,
       COUNT(DISTINCT device) OVER (
         PARTITION BY ip ORDER BY click_time
         RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                   AND INTERVAL 1 MILLISECOND PRECEDING
       ) AS value
FROM {source}
