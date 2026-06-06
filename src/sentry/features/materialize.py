"""Full-scale feature materialization (Task 4.1).

`FeaturePipeline` materializes a whole split in pandas — correct and
convenient at sample scale, impossible at 110M rows. This module computes
the SAME 22 features in one composed DuckDB query and writes Parquet via
`COPY`, never touching pandas. Equivalence with the pipeline is asserted
by test (`test_materialize.py`), so the individually-tested feature
definitions remain the source of truth and this query is their proven
composition.

Sampling (the Day-1 decision: ~10% time-stratified training rows) happens
AFTER window computation: every window sees the full-density history, and
only the emitted rows are sampled — per-hour, deterministically, by
hashing row_id. Sampling the source instead would thin every window by
the sample factor and resurrect the Task 3.4 sparsity problem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import duckdb
import structlog

from sentry.data._db import fetch_one
from sentry.features.f1_per_click import F1_FEATURES
from sentry.features.f2_velocity import (
    BURST_MAX_GAP_SECONDS,
    BURST_MIN_CLICKS_1H,
    F2_FEATURES,
)
from sentry.features.f3_aggregates import F3_FEATURES

logger = structlog.get_logger(__name__)

#: Every feature this module materializes, in pipeline order. The
#: equivalence test iterates this list against FeaturePipeline output.
ALL_FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    f.name for f in (*F1_FEATURES, *F2_FEATURES, *F3_FEATURES)
)

#: The composed query. Three layers, because windows can't nest and
#: SELECT aliases can't be referenced in the same SELECT:
#:   gaps  — lag() per ip (the inter-click gap feeds two features)
#:   feat  — every window feature in one pass over five shared frames
#:   outer — burst score (a CASE over two feat columns) + sampling
#: All frames are the strictly-prior CLAUDE.md §3.4 pattern. NULL gap in
#: the burst CASE: `NULL < 60` is NULL, CASE falls to ELSE 0 — identical
#: to the PythonFeature's NaN semantics.
_QUERY_TEMPLATE: Final[str] = """
WITH gaps AS (
    SELECT *,
           date_diff('second',
             lag(click_time) OVER (PARTITION BY ip ORDER BY click_time, row_id),
             click_time) AS f2_inter_click_time_seconds
    FROM {source}
),
feat AS (
    SELECT
        row_id, ip, app, device, os, channel, click_time, attributed_time, is_attributed,
        app AS f1_app_id,
        channel AS f1_channel_id,
        device AS f1_device_id,
        os AS f1_os_id,
        EXTRACT(hour FROM click_time) AS f1_hour_of_day,
        isodow(click_time) AS f1_day_of_week,
        CAST(ip AS VARCHAR) || '_' || CAST(app AS VARCHAR) AS f1_ip_app_interaction,
        CAST(ip AS VARCHAR) || '_' || CAST(device AS VARCHAR) AS f1_ip_device_interaction,
        COUNT(*) OVER w_ip_1h AS f2_clicks_per_ip_last_1hr,
        COUNT(*) OVER w_ip_24h AS f2_clicks_per_ip_last_24hr,
        COUNT(*) OVER w_pair_1h AS f2_clicks_per_ip_app_last_1hr,
        f2_inter_click_time_seconds,
        stddev_samp(f2_inter_click_time_seconds) OVER w_ip_24h
            AS f2_ip_click_std_inter_arrival,
        AVG(is_attributed) OVER w_ip_24h AS f3_ip_conversion_rate_24hr,
        AVG(is_attributed) OVER w_app_24h AS f3_app_conversion_rate_24hr,
        COUNT(DISTINCT app) OVER w_ip_24h AS f3_ip_distinct_apps_24hr,
        COUNT(DISTINCT device) OVER w_ip_24h AS f3_ip_distinct_devices_24hr,
        COUNT(DISTINCT os) OVER w_ip_24h AS f3_ip_distinct_oses_24hr,
        AVG(is_attributed) OVER w_pair_24h AS f3_ip_app_conversion_rate_24hr,
        COUNT(*) OVER w_pair_24h AS f3_ip_app_clicks_24hr,
        COUNT(DISTINCT ip) OVER w_app_24h AS f3_app_distinct_ips_24hr
    FROM gaps
    WINDOW
        w_ip_1h AS (PARTITION BY ip ORDER BY click_time
            RANGE BETWEEN INTERVAL 1 HOUR PRECEDING
                      AND INTERVAL 1 MILLISECOND PRECEDING),
        w_ip_24h AS (PARTITION BY ip ORDER BY click_time
            RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                      AND INTERVAL 1 MILLISECOND PRECEDING),
        w_pair_1h AS (PARTITION BY ip, app ORDER BY click_time
            RANGE BETWEEN INTERVAL 1 HOUR PRECEDING
                      AND INTERVAL 1 MILLISECOND PRECEDING),
        w_pair_24h AS (PARTITION BY ip, app ORDER BY click_time
            RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                      AND INTERVAL 1 MILLISECOND PRECEDING),
        w_app_24h AS (PARTITION BY app ORDER BY click_time
            RANGE BETWEEN INTERVAL 24 HOURS PRECEDING
                      AND INTERVAL 1 MILLISECOND PRECEDING)
)
SELECT *,
       CASE WHEN f2_clicks_per_ip_last_1hr > {burst_min_clicks}
                 AND f2_inter_click_time_seconds < {burst_max_gap}
            THEN 1 ELSE 0 END AS f2_burst_score
FROM feat
{sample_clause}
"""

#: Deterministic per-hour sampling: rank rows within each hour by a hash of
#: row_id and keep the first `fraction` of each hour. No RNG state, no seed
#: files — re-running always selects the same rows (§3.8).
_SAMPLE_CLAUSE: Final[str] = """
QUALIFY row_number() OVER (
            PARTITION BY date_trunc('hour', click_time) ORDER BY hash(row_id)
        ) <= CAST(CEIL({fraction} * COUNT(*) OVER (
            PARTITION BY date_trunc('hour', click_time))) AS BIGINT)
"""


def full_feature_query(source: str, sample_fraction: float | None = None) -> str:
    """The composed 22-feature query against `source` (a split view)."""
    sample_clause = ""
    if sample_fraction is not None:
        if not 0 < sample_fraction < 1:
            raise ValueError(f"sample fraction must be in (0, 1), got {sample_fraction}")
        sample_clause = _SAMPLE_CLAUSE.format(fraction=sample_fraction)
    return _QUERY_TEMPLATE.format(
        source=source,
        burst_min_clicks=BURST_MIN_CLICKS_1H,
        burst_max_gap=BURST_MAX_GAP_SECONDS,
        sample_clause=sample_clause,
    )


def materialize_features(
    db_path: Path | str,
    source: str,
    out_path: Path | str,
    sample_fraction: float | None = None,
) -> int:
    """Compute all 22 features over `source` and COPY them to Parquet.

    Runs entirely inside DuckDB — bounded memory regardless of split size.
    Returns the number of rows written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    query = full_feature_query(source, sample_fraction)

    with duckdb.connect(str(db_path), read_only=True) as conn:
        conn.execute(f"COPY ({query}) TO '{out_path}' (FORMAT PARQUET)")
        n_rows: int = fetch_one(conn, "SELECT COUNT(*) FROM read_parquet(?)", [str(out_path)])[0]

    logger.info(
        "features_materialized",
        source=source,
        out_path=str(out_path),
        rows=n_rows,
        sample_fraction=sample_fraction,
    )
    return n_rows
