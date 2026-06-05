"""Tests for `sentry.data.splits` — time-based train/val/test split.

The split is the leakage firewall for the whole project (CLAUDE.md §3.1):
time-based, 60/20/20, half-open intervals with exclusive upper bounds.
These tests pin the three properties that matter:

- the three views partition the source table exactly (no loss, no overlap)
- boundary timestamps land on the correct side of each half-open interval
- `apply_split` (pandas path) agrees with the views (SQL path) row-for-row
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import structlog.testing

from sentry.data._db import fetch_one
from sentry.data.splits import (
    SPLIT_NAMES,
    TRAIN_END_EXCLUSIVE,
    VAL_END_EXCLUSIVE,
    apply_split,
    create_split_views,
    get_splits,
)


def _boundary_df() -> pd.DataFrame:
    """Six rows straddling both split boundaries by exactly one second.

    Two rows sit ON each boundary — half-open semantics must push them into
    the LATER split (exclusive upper bound on the earlier one).
    """
    times = [
        TRAIN_END_EXCLUSIVE - timedelta(seconds=1),  # last train row
        TRAIN_END_EXCLUSIVE,  # ON boundary -> val
        VAL_END_EXCLUSIVE - timedelta(seconds=1),  # last val row
        VAL_END_EXCLUSIVE,  # ON boundary -> test
        datetime(2017, 11, 6, 16, 0, 0),  # data start -> train
        datetime(2017, 11, 9, 15, 59, 51),  # data end -> test
    ]
    return pd.DataFrame(
        {
            "ip": range(len(times)),
            "app": 1,
            "device": 1,
            "os": 1,
            "channel": 1,
            "click_time": pd.to_datetime(times),
            "attributed_time": pd.NaT,
            "is_attributed": 0,
        }
    )


def _db_with_clicks(tmp_path: Path, df: pd.DataFrame) -> Path:
    db_path = tmp_path / "splits.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        conn.register("source_df", df)
        conn.execute("CREATE TABLE clicks AS SELECT * FROM source_df")
    return db_path


def test_views_partition_dataset(tmp_path: Path, tiny_sample_data: pd.DataFrame) -> None:
    """AC: train + val + test row counts equal the full dataset, no overlap."""
    df = pd.concat([tiny_sample_data, _boundary_df()], ignore_index=True)
    db_path = _db_with_clicks(tmp_path, df)

    create_split_views(db_path)

    with duckdb.connect(str(db_path), read_only=True) as conn:
        total = fetch_one(conn, "SELECT COUNT(*) FROM clicks")[0]
        counts = {
            name: fetch_one(conn, f"SELECT COUNT(*) FROM clicks_{name}")[0] for name in SPLIT_NAMES
        }
        # No overlap: every (ip, click_time) pair appears in exactly one view.
        overlap = fetch_one(
            conn,
            """
            SELECT COUNT(*) FROM (
                SELECT ip, click_time FROM clicks_train
                INTERSECT
                SELECT ip, click_time FROM clicks_val
                UNION ALL
                SELECT * FROM (
                    SELECT ip, click_time FROM clicks_val
                    INTERSECT
                    SELECT ip, click_time FROM clicks_test
                )
                UNION ALL
                SELECT * FROM (
                    SELECT ip, click_time FROM clicks_train
                    INTERSECT
                    SELECT ip, click_time FROM clicks_test
                )
            )
            """,
        )[0]

    assert sum(counts.values()) == total
    assert overlap == 0
    assert all(counts[name] > 0 for name in SPLIT_NAMES), counts


def test_boundary_rows_land_exclusively(tmp_path: Path) -> None:
    """A row AT a boundary belongs to the later split (half-open intervals)."""
    db_path = _db_with_clicks(tmp_path, _boundary_df())
    create_split_views(db_path)

    with duckdb.connect(str(db_path), read_only=True) as conn:
        train_max = fetch_one(conn, "SELECT MAX(click_time) FROM clicks_train")[0]
        val_min = fetch_one(conn, "SELECT MIN(click_time) FROM clicks_val")[0]
        val_max = fetch_one(conn, "SELECT MAX(click_time) FROM clicks_val")[0]
        test_min = fetch_one(conn, "SELECT MIN(click_time) FROM clicks_test")[0]

    assert train_max == TRAIN_END_EXCLUSIVE - timedelta(seconds=1)
    assert val_min == TRAIN_END_EXCLUSIVE
    assert val_max == VAL_END_EXCLUSIVE - timedelta(seconds=1)
    assert test_min == VAL_END_EXCLUSIVE


def test_apply_split_matches_views(tmp_path: Path, tiny_sample_data: pd.DataFrame) -> None:
    """The pandas path and the SQL path must agree row-for-row."""
    df = pd.concat([tiny_sample_data, _boundary_df()], ignore_index=True)
    db_path = _db_with_clicks(tmp_path, df)
    create_split_views(db_path)

    with duckdb.connect(str(db_path), read_only=True) as conn:
        for name in SPLIT_NAMES:
            view_times = sorted(
                row[0] for row in conn.execute(f"SELECT click_time FROM clicks_{name}").fetchall()
            )
            split_times = sorted(apply_split(df, name)["click_time"])
            # pd.Timestamp == datetime compares by instant, so the lists match
            # elementwise despite the different types.
            assert split_times == view_times, f"pandas/SQL disagree on split {name!r}"


def test_apply_split_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        apply_split(_boundary_df(), "holdout")


def test_apply_split_test_emits_loud_warning() -> None:
    """Accessing the test split warns — one of the two §3.1 guard mechanisms."""
    with structlog.testing.capture_logs() as logs:
        apply_split(_boundary_df(), "test")
        apply_split(_boundary_df(), "train")

    warnings = [e for e in logs if e["log_level"] == "warning"]
    assert len(warnings) == 1, "exactly one warning: test access warns, train access doesn't"
    assert warnings[0]["event"] == "test_split_accessed"


def test_get_splits_is_contiguous_and_covers_data(tmp_path: Path) -> None:
    df = _boundary_df()
    db_path = _db_with_clicks(tmp_path, df)

    train, val, test = get_splits(db_path)

    assert train.start == df["click_time"].min()
    assert train.end_exclusive == val.start == TRAIN_END_EXCLUSIVE
    assert val.end_exclusive == test.start == VAL_END_EXCLUSIVE
    assert test.end_exclusive is None  # unbounded: test owns everything after VAL_END
    assert (train.name, val.name, test.name) == SPLIT_NAMES
