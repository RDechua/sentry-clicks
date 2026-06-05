"""Time-based train/validation/test split (CLAUDE.md §3.1).

The split is defined by TIME, not by row count: two pinned hour-aligned
boundary constants partition `click_time` into three half-open intervals.
Pinned timestamps mean the same split applies identically to the 100k
sample and the full 184M-row dataset, and nothing about the split changes
when the data is re-ingested.

    train: [data start,          TRAIN_END_EXCLUSIVE)
    val:   [TRAIN_END_EXCLUSIVE, VAL_END_EXCLUSIVE)
    test:  [VAL_END_EXCLUSIVE,   ...)

The boundaries were chosen once from the sample's row quantiles, rounded
to the hour: they yield 60.3 / 20.1 / 19.6 by rows — the locked 60/20/20
(CLAUDE.md §3.1) on human-readable boundaries. Full derivation in
`docs/decisions.md` (2026-06-05: Time-based split boundaries).

Test-set guard mechanisms (build guide Task 2.1 requires two):
1. View separation — all feature/model work queries `clicks_train` /
   `clicks_val`; nothing imports a test-split path by accident.
2. Loud access warning — `apply_split(df, "test")` emits a structlog
   WARNING (`test_split_accessed`) every time, so a stray test-set read
   during development is visible in the logs, not silent.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Final, NamedTuple

import duckdb
import pandas as pd
import structlog

from sentry.data._db import fetch_one
from sentry.data.schema import TABLE_NAME

logger = structlog.get_logger(__name__)

#: Split names in temporal order. View names are `clicks_<name>`.
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "val", "test")

#: First click_time that is NOT train (train upper bound, exclusive).
TRAIN_END_EXCLUSIVE: Final[datetime] = datetime(2017, 11, 8, 13, 0, 0)

#: First click_time that is NOT val (val upper bound, exclusive).
VAL_END_EXCLUSIVE: Final[datetime] = datetime(2017, 11, 9, 5, 0, 0)


class SplitRange(NamedTuple):
    """One split's half-open time interval. `end_exclusive=None` = unbounded."""

    name: str
    start: datetime
    end_exclusive: datetime | None


def get_splits(db_path: Path | str) -> tuple[SplitRange, SplitRange, SplitRange]:
    """Return the three split ranges, anchored to the data actually in the DB.

    The boundaries are the pinned module constants; only the start of train
    is read from the data (MIN of click_time). Test is unbounded above —
    everything from `VAL_END_EXCLUSIVE` on belongs to test.
    """
    with duckdb.connect(str(db_path), read_only=True) as conn:
        data_start = fetch_one(conn, f"SELECT MIN(click_time) FROM {TABLE_NAME}")[0]

    return (
        SplitRange("train", data_start, TRAIN_END_EXCLUSIVE),
        SplitRange("val", TRAIN_END_EXCLUSIVE, VAL_END_EXCLUSIVE),
        SplitRange("test", VAL_END_EXCLUSIVE, None),
    )


def apply_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Filter a clicks DataFrame to one split by `click_time`.

    The pandas mirror of the SQL views — same boundaries, same half-open
    semantics. Accessing the test split logs a WARNING every time (guard
    mechanism 2 above); the formal final evaluation in Task 4.7 is the
    only intended reader.
    """
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split_name!r}; expected one of {SPLIT_NAMES}")
    if split_name == "test":
        logger.warning(
            "test_split_accessed",
            reminder="test is touched once, at the Task 4.7 formal evaluation",
        )

    times = df["click_time"]
    if split_name == "train":
        mask = times < TRAIN_END_EXCLUSIVE
    elif split_name == "val":
        mask = (times >= TRAIN_END_EXCLUSIVE) & (times < VAL_END_EXCLUSIVE)
    else:
        mask = times >= VAL_END_EXCLUSIVE
    return df[mask]


def create_split_views(db_path: Path | str) -> dict[str, int]:
    """Create (or replace) the `clicks_train` / `clicks_val` / `clicks_test` views.

    Views, not tables: zero storage cost, always consistent with the source
    table, and re-running after a re-ingest is free. Returns the per-view
    row counts so callers can log/assert the partition.
    """
    train_end = TRAIN_END_EXCLUSIVE.isoformat()
    val_end = VAL_END_EXCLUSIVE.isoformat()
    predicates = {
        "train": f"click_time < TIMESTAMP '{train_end}'",
        "val": (f"click_time >= TIMESTAMP '{train_end}' AND click_time < TIMESTAMP '{val_end}'"),
        "test": f"click_time >= TIMESTAMP '{val_end}'",
    }

    counts: dict[str, int] = {}
    with duckdb.connect(str(db_path)) as conn:
        for name in SPLIT_NAMES:
            conn.execute(
                f"CREATE OR REPLACE VIEW clicks_{name} AS "
                f"SELECT * FROM {TABLE_NAME} WHERE {predicates[name]}"
            )
            counts[name] = fetch_one(conn, f"SELECT COUNT(*) FROM clicks_{name}")[0]

    logger.info("split_views_created", db_path=str(db_path), **counts)
    return counts
