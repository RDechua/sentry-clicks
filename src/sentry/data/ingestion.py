"""CSV → DuckDB ingestion for the TalkingData click stream.

DuckDB streams the CSV file natively (no Python-side chunking required), so the
same function works on the 100k-row `train_sample.csv` and the full 200M-row
`train.csv`. Memory footprint is bounded by DuckDB's internal buffering, not by
the input file size.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import structlog

from sentry.data._db import fetch_one
from sentry.data.schema import DUCKDB_COLUMN_TYPES, TABLE_NAME

logger = structlog.get_logger(__name__)


def ingest_csv_to_duckdb(
    csv_path: Path | str, db_path: Path | str, create_indexes: bool = True
) -> int:
    """Load a TalkingData CSV into the `clicks` table of a DuckDB database.

    The table is created (replacing any existing `clicks` table) with the
    explicit schema in `DUCKDB_COLUMN_TYPES`, plus a derived `row_id` column
    (contiguous 1..N, ascending with click_time) that gives every click a
    stable identity — the join key for SQL-computed features. Indexes are
    built on `click_time` and `ip` since those are the columns features
    filter and join on most.

    Parameters
    ----------
    csv_path:
        Path to the input CSV file. Must have a header row matching the
        TalkingData column names.
    db_path:
        Path to the DuckDB database file. Created if it does not exist.

    Returns
    -------
    Number of rows ingested.
    """
    csv_path_str = str(Path(csv_path))
    db_path_str = str(Path(db_path))

    # DuckDB struct-literal form: {'col': 'TYPE', ...}. Keys/values are static,
    # built from DUCKDB_COLUMN_TYPES — no user input flows into this clause.
    columns_clause = (
        "{" + ", ".join(f"'{name}': '{dtype}'" for name, dtype in DUCKDB_COLUMN_TYPES.items()) + "}"
    )

    with duckdb.connect(db_path_str) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        # ORDER BY click_time at load time: most feature queries are
        # time-windowed, so physical row ordering by click_time avoids
        # re-sorting on every scan at 200M rows.
        #
        # row_id is the stable per-row identity the feature pipeline joins on
        # (Task 2.2). (ip, click_time) is NOT unique — the EDA found same-IP
        # clicks in the same second — so features computed in separate SQL
        # queries need an unambiguous key. The window's ORDER BY is total
        # (click_time plus every dimension column as tie-breaker); fully
        # identical rows are interchangeable, so any stable assignment among
        # them is correct.
        conn.execute(
            f"""
            CREATE TABLE {TABLE_NAME} AS
            SELECT
                row_number() OVER (
                    ORDER BY click_time, ip, app, device, os, channel
                ) AS row_id,
                *
            FROM read_csv(?, columns={columns_clause}, header=true)
            ORDER BY click_time
            """,
            [csv_path_str],
        )
        # Indexes per build guide Task 1.6 — optional since Task 4.1: the ART
        # index build OOMs at 184M rows in a 3 GB container, and the feature
        # workload is full-scan window queries that never use them. Kept (on
        # by default) for the dev-scale DB where point-lookup exploration is
        # common and the cost is negligible.
        if create_indexes:
            for col in ("click_time", "ip"):
                conn.execute(f"CREATE INDEX idx_clicks_{col} ON {TABLE_NAME}({col})")

        row_count: int = fetch_one(conn, f"SELECT COUNT(*) FROM {TABLE_NAME}")[0]

    logger.info(
        "ingested_csv",
        csv_path=csv_path_str,
        db_path=db_path_str,
        rows=row_count,
    )
    return row_count
