-- f1_ip_app_interaction: (ip, app) pair as a single categorical.
-- 99.7% of (ip, app) pairs with >=5 clicks never convert (Week 1 EDA) —
-- the pair carries signal neither column has alone. High cardinality is
-- deliberate; LightGBM handles categoricals natively (encoding decided at
-- the model layer, Week 4).
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id, CAST(ip AS VARCHAR) || '_' || CAST(app AS VARCHAR) AS value FROM {source}
