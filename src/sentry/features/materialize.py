"""Full-scale feature materialization (Task 4.1).

`FeaturePipeline` materializes a whole split in pandas — correct and
convenient at sample scale, impossible at 110M rows. This module computes
the SAME 22 features in DuckDB and writes Parquet via `COPY`, never
touching pandas. Equivalence with the pipeline is asserted by test
(`test_materialize.py`), so the individually-tested feature definitions
remain the source of truth.

Why multiple passes instead of one composed query: a single query stacking
five window frames, three DISTINCT aggregates, and a lag CTE holds all
operator state at once and gets the container OOM-killed at 184M-row scale
(measured: died at 24% of the val split in a 3 GB container, with and
without a memory_limit). Each individual window runs comfortably (~15s
per 37M rows in 2.2 GB) — so the materializer runs one small query per
window family into intermediate Parquet, then joins on row_id. Same
result, bounded memory; the equivalence test proves the decomposition.

Sampling (the Day-1 decision: ~10% time-stratified training rows) happens
in the final assembly, AFTER all windows are computed: every window sees
full-density history and only the emitted rows are sampled — per hour,
deterministically, by hashing row_id. Sampling the source instead would
thin every window by the sample factor and resurrect the Task 3.4
sparsity problem.
"""

from __future__ import annotations

import shutil
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

#: Every feature this module materializes. The equivalence test iterates
#: this list against FeaturePipeline output.
ALL_FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    f.name for f in (*F1_FEATURES, *F2_FEATURES, *F3_FEATURES)
)

#: Strictly-prior frames (CLAUDE.md §3.4), shared by the pass queries.
_FRAMES: Final[dict[str, str]] = {
    "w_ip_1h": (
        "PARTITION BY ip ORDER BY click_time RANGE BETWEEN INTERVAL 1 HOUR "
        "PRECEDING AND INTERVAL 1 MILLISECOND PRECEDING"
    ),
    "w_ip_24h": (
        "PARTITION BY ip ORDER BY click_time RANGE BETWEEN INTERVAL 24 HOURS "
        "PRECEDING AND INTERVAL 1 MILLISECOND PRECEDING"
    ),
    "w_pair_1h": (
        "PARTITION BY ip, app ORDER BY click_time RANGE BETWEEN INTERVAL 1 HOUR "
        "PRECEDING AND INTERVAL 1 MILLISECOND PRECEDING"
    ),
    "w_pair_24h": (
        "PARTITION BY ip, app ORDER BY click_time RANGE BETWEEN INTERVAL 24 HOURS "
        "PRECEDING AND INTERVAL 1 MILLISECOND PRECEDING"
    ),
    "w_app_24h": (
        "PARTITION BY app ORDER BY click_time RANGE BETWEEN INTERVAL 24 HOURS "
        "PRECEDING AND INTERVAL 1 MILLISECOND PRECEDING"
    ),
}

#: One entry per pass: (pass name, SELECT body). `{source}` is the split
#: view. Every pass emits row_id plus its features; the base pass also
#: carries the base columns, the f1 row-locals, and the lag-derived gap
#: (cheap streaming operators that don't compound memory). DISTINCT
#: window aggregates each get their own pass — they are the heavy operator.
_PASS_QUERIES: Final[tuple[tuple[str, str], ...]] = (
    (
        "base",
        f"""
        SELECT row_id, ip, app, device, os, channel, click_time, attributed_time,
               is_attributed,
               app AS f1_app_id,
               channel AS f1_channel_id,
               device AS f1_device_id,
               os AS f1_os_id,
               EXTRACT(hour FROM click_time) AS f1_hour_of_day,
               isodow(click_time) AS f1_day_of_week,
               CAST(ip AS VARCHAR) || '_' || CAST(app AS VARCHAR)
                   AS f1_ip_app_interaction,
               CAST(ip AS VARCHAR) || '_' || CAST(device AS VARCHAR)
                   AS f1_ip_device_interaction,
               date_diff('second',
                 lag(click_time) OVER (PARTITION BY ip ORDER BY click_time, row_id),
                 click_time) AS f2_inter_click_time_seconds,
               COUNT(*) OVER w_ip_1h AS f2_clicks_per_ip_last_1hr
        FROM {{source}}
        WINDOW w_ip_1h AS ({_FRAMES["w_ip_1h"]})
        """,
    ),
    (
        "ip_24h",
        f"""
        WITH gaps AS (
            SELECT row_id, ip, click_time, is_attributed,
                   date_diff('second',
                     lag(click_time) OVER (PARTITION BY ip ORDER BY click_time, row_id),
                     click_time) AS gap_s
            FROM {{source}}
        )
        SELECT row_id,
               COUNT(*) OVER w AS f2_clicks_per_ip_last_24hr,
               stddev_samp(gap_s) OVER w AS f2_ip_click_std_inter_arrival,
               AVG(is_attributed) OVER w AS f3_ip_conversion_rate_24hr
        FROM gaps
        WINDOW w AS ({_FRAMES["w_ip_24h"]})
        """,
    ),
    (
        "pair",
        f"""
        SELECT row_id,
               COUNT(*) OVER w1 AS f2_clicks_per_ip_app_last_1hr,
               COUNT(*) OVER w24 AS f3_ip_app_clicks_24hr,
               AVG(is_attributed) OVER w24 AS f3_ip_app_conversion_rate_24hr
        FROM {{source}}
        WINDOW w1 AS ({_FRAMES["w_pair_1h"]}), w24 AS ({_FRAMES["w_pair_24h"]})
        """,
    ),
    (
        "app",
        f"""
        SELECT row_id,
               AVG(is_attributed) OVER w AS f3_app_conversion_rate_24hr
        FROM {{source}}
        WINDOW w AS ({_FRAMES["w_app_24h"]})
        """,
    ),
    (
        "distinct_apps",
        f"""
        SELECT row_id, COUNT(DISTINCT app) OVER w AS f3_ip_distinct_apps_24hr
        FROM {{source}} WINDOW w AS ({_FRAMES["w_ip_24h"]})
        """,
    ),
    (
        "distinct_devices",
        f"""
        SELECT row_id, COUNT(DISTINCT device) OVER w AS f3_ip_distinct_devices_24hr
        FROM {{source}} WINDOW w AS ({_FRAMES["w_ip_24h"]})
        """,
    ),
    (
        "distinct_oses",
        f"""
        SELECT row_id, COUNT(DISTINCT os) OVER w AS f3_ip_distinct_oses_24hr
        FROM {{source}} WINDOW w AS ({_FRAMES["w_ip_24h"]})
        """,
    ),
    (
        "distinct_ips",
        f"""
        SELECT row_id, COUNT(DISTINCT ip) OVER w AS f3_app_distinct_ips_24hr
        FROM {{source}} WINDOW w AS ({_FRAMES["w_app_24h"]})
        """,
    ),
)

#: Deterministic per-hour sampling, applied in the final assembly. No RNG
#: state — re-running always selects the same rows (§3.8).
_SAMPLE_CLAUSE: Final[str] = """
QUALIFY row_number() OVER (
            PARTITION BY date_trunc('hour', click_time) ORDER BY hash(row_id)
        ) <= CAST(CEIL({fraction} * COUNT(*) OVER (
            PARTITION BY date_trunc('hour', click_time))) AS BIGINT)
"""


def _assembly_query(pass_dir: Path, sample_fraction: float | None) -> str:
    """Join all pass outputs on row_id; add burst score and sampling.

    The burst CASE lives here because it reads two columns from different
    passes. `NULL < threshold` is NULL, so CASE falls to ELSE 0 — identical
    to the PythonFeature's NaN semantics.
    """
    sample_clause = ""
    if sample_fraction is not None:
        if not 0 < sample_fraction < 1:
            raise ValueError(f"sample fraction must be in (0, 1), got {sample_fraction}")
        sample_clause = _SAMPLE_CLAUSE.format(fraction=sample_fraction)

    joins = "\n".join(
        f"JOIN read_parquet('{pass_dir / name}.parquet') AS {name} USING (row_id)"
        for name, _ in _PASS_QUERIES[1:]
    )
    return f"""
        SELECT base.*, {", ".join(f"{name}.* EXCLUDE (row_id)" for name, _ in _PASS_QUERIES[1:])},
               CASE WHEN f2_clicks_per_ip_last_1hr > {BURST_MIN_CLICKS_1H}
                         AND f2_inter_click_time_seconds < {BURST_MAX_GAP_SECONDS}
                    THEN 1 ELSE 0 END AS f2_burst_score
        FROM read_parquet('{pass_dir / "base"}.parquet') AS base
        {joins}
        {sample_clause}
    """


def materialize_features(
    db_path: Path | str,
    source: str,
    out_path: Path | str,
    sample_fraction: float | None = None,
    memory_limit: str | None = None,
    threads: int | None = None,
) -> int:
    """Compute all 22 features over `source` and COPY them to Parquet.

    Runs as one DuckDB query per window family (bounded memory at any
    split size) plus a final row_id join. Intermediate pass files live in
    a sibling directory of `out_path` and are removed on success.
    Returns the number of rows written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_fraction is not None and not 0 < sample_fraction < 1:
        raise ValueError(f"sample fraction must be in (0, 1), got {sample_fraction}")
    pass_dir = out_path.parent / f".{out_path.stem}_passes"
    pass_dir.mkdir(exist_ok=True)

    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            if memory_limit is not None:
                conn.execute(f"SET memory_limit = '{memory_limit}'")
            if threads is not None:
                conn.execute(f"SET threads = {threads}")
            conn.execute("SET enable_progress_bar = false")

            for name, body in _PASS_QUERIES:
                query = body.format(source=source)
                conn.execute(f"COPY ({query}) TO '{pass_dir / name}.parquet' (FORMAT PARQUET)")
                logger.info("materialize_pass_done", pass_name=name, source=source)

            assembly = _assembly_query(pass_dir, sample_fraction)
            conn.execute(f"COPY ({assembly}) TO '{out_path}' (FORMAT PARQUET)")
            n_rows: int = fetch_one(conn, "SELECT COUNT(*) FROM read_parquet(?)", [str(out_path)])[
                0
            ]
    finally:
        shutil.rmtree(pass_dir, ignore_errors=True)

    logger.info(
        "features_materialized",
        source=source,
        out_path=str(out_path),
        rows=n_rows,
        sample_fraction=sample_fraction,
    )
    return n_rows
