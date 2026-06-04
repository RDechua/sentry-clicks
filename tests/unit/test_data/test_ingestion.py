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
