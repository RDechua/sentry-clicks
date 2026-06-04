"""Unit tests for `sentry.data.validation.validate_ingestion`.

The fixture-based tests build a freshly-ingested DuckDB from `tiny_sample_data`,
then either validate it directly or mutate one row to verify the failure path.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from sentry.data.ingestion import ingest_csv_to_duckdb
from sentry.data.schema import TABLE_NAME
from sentry.data.validation import validate_ingestion


def _ingest_fixture(tmp_path: Path, df: pd.DataFrame) -> Path:
    csv_path = tmp_path / "clicks.csv"
    db_path = tmp_path / "clicks.duckdb"
    df.to_csv(csv_path, index=False)
    ingest_csv_to_duckdb(csv_path, db_path)
    return db_path


def test_validate_ok_for_fixture(tmp_path: Path, tiny_sample_data: pd.DataFrame) -> None:
    """Hand-crafted fixture passes structural checks (with class-balance skipped).

    The fixture's ~25% positive rate is intentionally non-representative for
    fraud-detection coverage; pass expected_balance=None to skip that check.
    """
    db_path = _ingest_fixture(tmp_path, tiny_sample_data)

    result = validate_ingestion(db_path, expected_balance=None)

    assert result.ok, f"unexpected validation failures: {result.errors}"
    assert result.row_count == 40


def test_validate_flags_invalid_is_attributed(
    tmp_path: Path, tiny_sample_data: pd.DataFrame
) -> None:
    """Inserting is_attributed=2 (outside {0, 1}) is caught by the validator."""
    db_path = _ingest_fixture(tmp_path, tiny_sample_data)
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            f"INSERT INTO {TABLE_NAME} VALUES "
            f"(1, 1, 1, 1, 1, TIMESTAMP '2017-11-07 09:00:00', NULL, 2)"
        )

    result = validate_ingestion(db_path, expected_balance=None)

    assert not result.ok
    assert any(c.name == "is_attributed_in_0_1" and not c.ok for c in result.checks)


def test_validate_flags_null_in_required_column(
    tmp_path: Path, tiny_sample_data: pd.DataFrame
) -> None:
    """Inserting a NULL into a non-nullable column is caught."""
    db_path = _ingest_fixture(tmp_path, tiny_sample_data)
    with duckdb.connect(str(db_path)) as conn:
        # ip is not nullable per the schema; DuckDB allows NULL at the column level
        # because the DDL doesn't add NOT NULL constraints — validate is the gate.
        conn.execute(
            f"INSERT INTO {TABLE_NAME} VALUES "
            f"(NULL, 1, 1, 1, 1, TIMESTAMP '2017-11-07 09:00:00', NULL, 0)"
        )

    result = validate_ingestion(db_path, expected_balance=None)

    assert not result.ok
    assert any(c.name == "no_nulls_in_ip" and not c.ok for c in result.checks)


def test_validate_flags_out_of_range_dates(tmp_path: Path, tiny_sample_data: pd.DataFrame) -> None:
    """A click_time outside the TalkingData date range is flagged."""
    db_path = _ingest_fixture(tmp_path, tiny_sample_data)
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            f"INSERT INTO {TABLE_NAME} VALUES "
            f"(1, 1, 1, 1, 1, TIMESTAMP '2020-01-01 00:00:00', NULL, 0)"
        )

    result = validate_ingestion(db_path, expected_balance=None)

    assert not result.ok
    assert any(c.name == "click_time_in_expected_range" and not c.ok for c in result.checks)
