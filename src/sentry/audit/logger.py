"""Audit log writer.

`log_event(entry, db_path=...)` appends one row to the `audit_log` table in
a DuckDB database, creating the database and table lazily on first write.
`log_events(entries, db_path=...)` is the batch variant for callers scoring
whole datasets (rationale in its docstring).

The audit DB lives at `artifacts/audit.duckdb` by default — separate from
`artifacts/sentry.duckdb` (the clicks DB) because the two have different
lifecycles: clicks is loaded once per ingest, audit grows monotonically
with every triage decision.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import duckdb
import structlog

from sentry.audit.schema import AuditLogEntry

logger = structlog.get_logger(__name__)

#: Name of the audit log table inside the DuckDB database.
AUDIT_TABLE_NAME = "audit_log"

#: Default location for the audit log database.
DEFAULT_AUDIT_DB_PATH = Path("artifacts/audit.duckdb")

# `top_features` is JSON, not STRUCT/LIST, because different models have
# different feature sets and forcing a fixed nested type would couple the
# audit schema to the current model's feature names.
_AUDIT_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {AUDIT_TABLE_NAME} (
    event_id              UUID PRIMARY KEY,
    event_timestamp       TIMESTAMP NOT NULL,
    case_id               VARCHAR   NOT NULL,
    click_timestamp       TIMESTAMP NOT NULL,
    model_version         VARCHAR   NOT NULL,
    policy_version        VARCHAR   NOT NULL,
    raw_score             DOUBLE    NOT NULL,
    calibrated_score      DOUBLE    NOT NULL,
    threshold_block       DOUBLE    NOT NULL,
    threshold_review      DOUBLE    NOT NULL,
    action                VARCHAR   NOT NULL,
    top_features          JSON      NOT NULL,
    reviewer_disposition  VARCHAR,
    reviewer_timestamp    TIMESTAMP,
    notes                 VARCHAR
)
"""


# The INSERT column list and parameter order are derived from
# `AuditLogEntry.model_fields` — the pydantic schema is the single source
# of truth. Reordering fields in the schema does not silently corrupt
# written rows, which would happen with a positional INSERT.
_FIELD_NAMES = list(AuditLogEntry.model_fields.keys())
_INSERT_SQL = (
    f"INSERT INTO {AUDIT_TABLE_NAME} ({', '.join(_FIELD_NAMES)}) "
    f"VALUES ({', '.join('?' for _ in _FIELD_NAMES)})"
)


def _entry_row(entry: AuditLogEntry) -> list[object]:
    """Serialize one entry into an INSERT parameter row.

    `mode="json"` makes pydantic serialize UUIDs to strings, datetimes to
    ISO format, StrEnums to their string values, and nested BaseModels to
    dicts. Then we JSON-encode top_features (a list of dicts) into the one
    string DuckDB's JSON column wants.
    """
    data = entry.model_dump(mode="json")
    data["top_features"] = json.dumps(data["top_features"])
    return [data[name] for name in _FIELD_NAMES]


def _write_rows(rows: Sequence[Sequence[object]], db_path: Path) -> None:
    """The single write path both public writers share.

    Creates the parent directory, the database file, and the `audit_log`
    table if they don't exist. Idempotent on table creation (`IF NOT EXISTS`).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(_AUDIT_TABLE_DDL)
        conn.executemany(_INSERT_SQL, rows)


def log_event(
    entry: AuditLogEntry,
    db_path: Path | str = DEFAULT_AUDIT_DB_PATH,
) -> None:
    """Append one audit log entry to the audit DB.

    For one-decision-at-a-time callers (the Task 6 triage router). Callers
    scoring whole datasets should use `log_events`.

    Parameters
    ----------
    entry:
        The audit log entry to write.
    db_path:
        Optional override for the audit DB location. Defaults to
        `DEFAULT_AUDIT_DB_PATH` (`artifacts/audit.duckdb`).
    """
    _write_rows([_entry_row(entry)], Path(db_path))
    logger.info(
        "audit_event_logged",
        event_id=str(entry.event_id),
        case_id=entry.case_id,
        action=entry.action.value,
    )


def log_events(
    entries: Sequence[AuditLogEntry],
    db_path: Path | str = DEFAULT_AUDIT_DB_PATH,
) -> int:
    """Append many audit log entries to the audit DB in one connection.

    Same write path as `log_event`, batched. CLAUDE.md §3.9 requires every
    decision to produce a log entry — batching is what makes that affordable
    when a run scores an entire dataset (the Task 1.10 tracer surfaced this:
    ~20k per-call connections would have blown the 5-minute pipeline budget;
    one batched call takes ~1s).

    An empty batch is a no-op: returns 0 without creating the DB file.

    Parameters
    ----------
    entries:
        The audit log entries to write.
    db_path:
        Optional override for the audit DB location. Defaults to
        `DEFAULT_AUDIT_DB_PATH` (`artifacts/audit.duckdb`).

    Returns
    -------
    int
        Number of rows written.
    """
    if not entries:
        return 0

    _write_rows([_entry_row(entry) for entry in entries], Path(db_path))
    logger.info("audit_events_logged", n_entries=len(entries), db_path=str(db_path))
    return len(entries)
