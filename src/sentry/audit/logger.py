"""Audit log writer.

`log_event(entry, db_path=...)` appends one row to the `audit_log` table in
a DuckDB database, creating the database and table lazily on first write.

The audit DB lives at `artifacts/audit.duckdb` by default — separate from
`artifacts/sentry.duckdb` (the clicks DB) because the two have different
lifecycles: clicks is loaded once per ingest, audit grows monotonically
with every triage decision.
"""

from __future__ import annotations

import json
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


def log_event(
    entry: AuditLogEntry,
    db_path: Path | str = DEFAULT_AUDIT_DB_PATH,
) -> None:
    """Append one audit log entry to the audit DB.

    Creates the database file, the parent directory, and the `audit_log`
    table if they don't exist. Idempotent on table creation (`IF NOT EXISTS`).

    The INSERT column list and parameter order are derived from
    `AuditLogEntry.model_fields` — the pydantic schema is the single source
    of truth. Reordering fields in the schema does not silently corrupt
    written rows, which would happen with a positional INSERT.

    Production note: a real high-volume audit pipeline would batch writes
    and reuse a single connection. For Sentry-Clicks's max ~100 calls/run
    the per-call connection cost is invisible; the simpler function is
    the right altitude. See `docs/decisions.md`.

    Parameters
    ----------
    entry:
        The audit log entry to write.
    db_path:
        Optional override for the audit DB location. Defaults to
        `DEFAULT_AUDIT_DB_PATH` (`artifacts/audit.duckdb`).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # `mode="json"` makes pydantic serialize UUIDs to strings, datetimes to
    # ISO format, StrEnums to their string values, and nested BaseModels to
    # dicts. Then we JSON-encode top_features (a list of dicts) into the one
    # string DuckDB's JSON column wants.
    data = entry.model_dump(mode="json")
    data["top_features"] = json.dumps(data["top_features"])

    field_names = list(AuditLogEntry.model_fields.keys())
    columns_clause = ", ".join(field_names)
    placeholders = ", ".join("?" for _ in field_names)

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(_AUDIT_TABLE_DDL)
        conn.execute(
            f"INSERT INTO {AUDIT_TABLE_NAME} ({columns_clause}) VALUES ({placeholders})",
            [data[name] for name in field_names],
        )

    logger.info(
        "audit_event_logged",
        event_id=str(entry.event_id),
        case_id=entry.case_id,
        action=entry.action.value,
    )
