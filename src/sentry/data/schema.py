"""TalkingData click stream schema.

Two coupled representations, kept in sync by hand:

  - `ClickRecord` (pydantic) documents the per-row schema, used for row-level
    sanity checks (and later for audit-log payloads).
  - `DUCKDB_COLUMN_TYPES` is the dict used by ingestion to build the DuckDB
    CREATE TABLE and read_csv calls.

If you change a column, update both. The human-facing description lives in
`docs/data-dictionary.md`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

from pydantic import BaseModel, Field

#: Name of the DuckDB table the ingestion function creates.
TABLE_NAME: Final[str] = "clicks"

#: Earliest click_time seen in the Kaggle TalkingData train set.
TRAIN_DATE_MIN: Final[datetime] = datetime(2017, 11, 6)

#: Latest click_time the validator will accept, as an EXCLUSIVE upper bound.
#: Covers the Kaggle test-set window (through 2017-11-11) — using exclusive
#: midnight 2017-11-12 avoids losing the final second that a `<=` to 23:59:59
#: silently misses.
TRAIN_DATE_MAX_EXCLUSIVE: Final[datetime] = datetime(2017, 11, 12)

#: Expected lower bound on mean(is_attributed). TalkingData is documented at
#: ~0.2% positive rate; below 0.1% suggests label-flip or upstream filter bug.
EXPECTED_CLASS_BALANCE_MIN: Final[float] = 0.001

#: Expected upper bound on mean(is_attributed). Above 1% on real data suggests
#: stratified sampling that bypassed the natural distribution.
EXPECTED_CLASS_BALANCE_MAX: Final[float] = 0.01


class ClickRecord(BaseModel):
    """One click event from the TalkingData dataset.

    Field constraints document the per-row schema; bulk type enforcement at
    ingest time lives in DuckDB via `DUCKDB_COLUMN_TYPES`.
    """

    ip: int = Field(ge=0)
    app: int = Field(ge=0)
    device: int = Field(ge=0)
    os: int = Field(ge=0)
    channel: int = Field(ge=0)
    click_time: datetime
    attributed_time: datetime | None
    is_attributed: int = Field(ge=0, le=1)


#: DuckDB column types for the `clicks` table. Keys are also the CSV column
#: order — both the DDL and the read_csv call use this dict.
DUCKDB_COLUMN_TYPES: Final[dict[str, str]] = {
    "ip": "INTEGER",
    "app": "INTEGER",
    "device": "INTEGER",
    "os": "INTEGER",
    "channel": "INTEGER",
    "click_time": "TIMESTAMP",
    "attributed_time": "TIMESTAMP",
    "is_attributed": "TINYINT",
}

#: Columns where NULL is a validation error (everything except attributed_time,
#: which is legitimately NULL for is_attributed=0 rows).
NOT_NULL_COLUMNS: Final[tuple[str, ...]] = (
    "ip",
    "app",
    "device",
    "os",
    "channel",
    "click_time",
    "is_attributed",
)
