# Data Dictionary — TalkingData Click Stream

Source: [TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection) on Kaggle.

Canonical Python representation: `src/sentry/data/schema.py:ClickRecord`.
Canonical DuckDB representation: the `clicks` table built by `src/sentry/data/ingestion.py:ingest_csv_to_duckdb`.

If you change a column, update both.

## Columns

| Column | DuckDB type | Nullable | Semantics |
|---|---|---|---|
| `ip` | `INTEGER` | No | Anonymized integer hash of the click source's IP address. Observed range in the public sample: roughly 1 to 360,000. Heavy-tailed distribution — a small number of IPs account for a large share of clicks, and the head of the distribution is fraud-heavy. |
| `app` | `INTEGER` | No | App identifier — the mobile app showing the ad. Range 0 to ~768. |
| `device` | `INTEGER` | No | Device type identifier (handset model / form factor). Heavily skewed; most clicks come from a few common devices. |
| `os` | `INTEGER` | No | Operating system version identifier. |
| `channel` | `INTEGER` | No | Mobile ad channel identifier (publisher / inventory source). |
| `click_time` | `TIMESTAMP` | No | UTC timestamp of the click event. Train set covers 2017-11-06 to 2017-11-09; the (unlabeled) test set extends through 2017-11-11. CSV format is ISO-8601, unambiguous. |
| `attributed_time` | `TIMESTAMP` | **Yes** | UTC timestamp of the resulting app install, if any. `NULL` for every row where `is_attributed = 0`. When non-null, always strictly after `click_time`. |
| `is_attributed` | `TINYINT` | No | Binary label. `1` if the click resulted in an app install, `0` otherwise. See the gotcha below — the label is *attribution*, not *fraud*. |

## Class balance

Roughly **0.2% positive rate** (`is_attributed = 1`) on the full train set. The 100k-row `train_sample.csv` runs at about 0.00227. `validate_ingestion` checks that `mean(is_attributed)` lands in [0.001, 0.01] as a sanity gate — anything outside that range signals a sampling or label-flip bug.

## Date range

| File | Window |
|---|---|
| `train.csv` (full, ~200M rows) | 2017-11-06 to 2017-11-09 |
| `train_sample.csv` (100k rows) | uniform sample of the above |
| `test.csv` (unlabeled, leaderboard set) | 2017-11-10 to 2017-11-11 |

`validate_ingestion` accepts click times up to 2017-11-11 23:59:59, so the same validator works on either train or test.

## Gotchas

1. **The label is "attribution", not "fraud".** `is_attributed = 0` means the click never resulted in an install — that's the *fraud-suspect* case in TalkingData's framing, and it's the majority class. A classifier trained to predict `is_attributed = 1` is predicting "this click is legitimate", and "fraud probability" is `1 - p`. Easy to flip by accident; double-check the direction of every probability before reporting.
2. **`attributed_time` is null-correlated with the label.** Every row with `is_attributed = 0` has `attributed_time = NULL` by definition. Using `attributed_time` (or `attributed_time IS NULL`) as a feature is label leakage. The `NOT_NULL_COLUMNS` constant in `schema.py` deliberately excludes it.
3. **Pandas dtype trap.** Reading the CSV without explicit types makes pandas infer `is_attributed` as `float64` (it sees the `NaT` rows in `attributed_time` and gets confused about adjacent columns). Always use the DuckDB schema in `DUCKDB_COLUMN_TYPES` or pass `dtype={"is_attributed": "Int8"}` to `read_csv`.
4. **IP is not a stable user identifier.** Behind NAT, many users share one IP; behind mobile-carrier rotation, one user occupies many IPs. Treat `ip` as a noisy signal for fraud-cluster detection, not as a user join key.
5. **`device` and `os` are categorical, despite being integers.** Don't sum them, don't average them — they're identifiers, not measurements.
