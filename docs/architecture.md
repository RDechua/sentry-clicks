# Sentry — System Architecture

*Started in Week 2 (Task 2.2) with the feature pipeline; the full document
gets written in Week 7. Sections below are added as the components they
describe become real.*

## Data layer (Week 1)

One DuckDB database (`artifacts/sentry.duckdb`) holds the `clicks` table,
ingested from the TalkingData CSV by `sentry.data.ingestion`. Ingestion
assigns every row a stable `row_id` (contiguous 1..N, ascending with
`click_time`) — the join key everything downstream uses. Validation
(`sentry.data.validation`) gates the ingest with structural checks before
anything reads the table.

Three views partition the table by time (`sentry.data.splits`):
`clicks_train` / `clicks_val` / `clicks_test`, half-open intervals on
pinned hour boundaries (60/20/20 by rows). All feature and model work reads
the views, never the base table — the split is the leakage firewall.

## Feature pipeline (Week 2)

`sentry.features.pipeline` computes feature tables from a split view.

Two feature kinds implement one contract:

- **SqlFeature** — a query template run against the split view, returning
  `(row_id, value)` for every row. Aggregations, joins, window functions.
- **PythonFeature** — a function over the accumulated frame (base columns
  plus already-computed features). Cross-row statistics, composite logic.

`FeaturePipeline` topo-sorts features by their declared dependencies at
construction (bad graphs fail before any compute starts), runs them in
order, and joins SQL results back on `row_id` — result order never matters,
and a feature query that drops or duplicates rows fails loudly instead of
silently misaligning the table.

Leakage protection is layered, and the layers are independent:

1. The pipeline is split-agnostic — callers name a split view as the
   source, so a train run physically cannot read val/test rows.
2. Each feature's own window must be strictly prior to the current row
   (CLAUDE.md §3.4) — owned and tested per-feature, not by the framework.

A global `FEATURE_REGISTRY` collects definitions at import time; pipelines
are built from the registry or any explicit subset of it.

## Planned (placeholders, filled in as built)

- **Feature store** (Task 2.5): versioned Parquet snapshots per split under
  `artifacts/features/`.
- **Model layer** (Week 4): LightGBM training, isotonic calibration on val.
- **Triage** (Weeks 5-6): cost-based thresholds, routing, audit logging —
  the audit schema and writers already exist (`sentry.audit`, Week 1).
- **Reporting** (Week 7): HTML report generation.
