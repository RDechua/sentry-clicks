"""Unit tests for `sentry.data.ingestion.ingest_csv_to_duckdb`.

The fixture-based tests use `tiny_sample_data` (40 rows) written to a temp CSV
and then ingested into a temp DuckDB file. Schema correctness and null handling
are exercised; the real-data round-trip lives in the integration test.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from sentry.data._db import fetch_one
from sentry.data.ingestion import ingest_csv_to_duckdb
from sentry.data.schema import TABLE_NAME


def test_ingest_creates_clicks_table(tmp_path: Path, tiny_sample_data: pd.DataFrame) -> None:
    """Happy path — fixture round-trip through CSV → DuckDB preserves row count."""
    csv_path = tmp_path / "clicks.csv"
    db_path = tmp_path / "clicks.duckdb"
    tiny_sample_data.to_csv(csv_path, index=False)

    row_count = ingest_csv_to_duckdb(csv_path, db_path)

    assert row_count == len(tiny_sample_data) == 40
    with duckdb.connect(str(db_path), read_only=True) as conn:
        n = fetch_one(conn, f"SELECT COUNT(*) FROM {TABLE_NAME}")[0]
    assert n == 40


def test_ingest_assigns_stable_row_ids(tmp_path: Path, tiny_sample_data: pd.DataFrame) -> None:
    """row_id is the join key for SQL-computed features (Task 2.2).

    Contract: contiguous 1..N, unique, and ascending with click_time so the
    physical ORDER BY and the id order agree.
    """
    csv_path = tmp_path / "clicks.csv"
    db_path = tmp_path / "clicks.duckdb"
    tiny_sample_data.to_csv(csv_path, index=False)

    ingest_csv_to_duckdb(csv_path, db_path)

    with duckdb.connect(str(db_path), read_only=True) as conn:
        n_rows, n_ids, min_id, max_id = fetch_one(
            conn,
            f"SELECT COUNT(*), COUNT(DISTINCT row_id), MIN(row_id), MAX(row_id) "
            f"FROM {TABLE_NAME}",
        )
        out_of_order = fetch_one(
            conn,
            f"""
            SELECT COUNT(*) FROM (
                SELECT click_time,
                       lag(click_time) OVER (ORDER BY row_id) AS prev_time
                FROM {TABLE_NAME}
            ) WHERE prev_time > click_time
            """,
        )[0]

    assert (n_ids, min_id, max_id) == (n_rows, 1, n_rows), "row_id must be contiguous 1..N"
    assert out_of_order == 0, "row_id order must agree with click_time order"


def test_ingest_can_skip_indexes(tmp_path: Path, tiny_sample_data: pd.DataFrame) -> None:
    """create_indexes=False: full-scale path (index build OOMs at 184M rows
    and the window-scan workload never uses them)."""
    csv_path = tmp_path / "clicks.csv"
    db_path = tmp_path / "clicks.duckdb"
    tiny_sample_data.to_csv(csv_path, index=False)

    ingest_csv_to_duckdb(csv_path, db_path, create_indexes=False)

    with duckdb.connect(str(db_path), read_only=True) as conn:
        n_indexes = fetch_one(conn, "SELECT COUNT(*) FROM duckdb_indexes()")[0]
    assert n_indexes == 0


def test_ingest_handles_attributed_time_nulls(
    tmp_path: Path, tiny_sample_data: pd.DataFrame
) -> None:
    """attributed_time is NaT for is_attributed=0 rows; must round-trip as NULL."""
    csv_path = tmp_path / "clicks.csv"
    db_path = tmp_path / "clicks.duckdb"
    tiny_sample_data.to_csv(csv_path, index=False)

    ingest_csv_to_duckdb(csv_path, db_path)

    with duckdb.connect(str(db_path), read_only=True) as conn:
        null_attr = fetch_one(
            conn, f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE attributed_time IS NULL"
        )[0]
        zero_label = fetch_one(conn, f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE is_attributed = 0")[
            0
        ]
    # Every is_attributed=0 row should have NULL attributed_time, and vice versa.
    assert null_attr == zero_label > 0
