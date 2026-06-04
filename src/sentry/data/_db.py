"""Internal DuckDB helpers shared across `sentry.data` modules.

The Python DuckDB driver's `fetchone()` returns `tuple | None` because the
contract has to cover queries that return zero rows. For aggregate queries
(COUNT, AVG, MIN/MAX) the row is always present, but mypy doesn't know that.
`fetch_one` narrows the type and raises an explicit RuntimeError if the
invariant is violated — safer than the bare `assert` form that disappears
under `python -O`.
"""

from __future__ import annotations

from typing import Any

import duckdb


def fetch_one(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[Any] | None = None,
) -> tuple[Any, ...]:
    """Execute SQL and return the single result row.

    Use for queries that always return exactly one row (aggregates like
    COUNT/AVG/MIN/MAX, or `SELECT ... LIMIT 1`). Raises RuntimeError if the
    query returns no rows.
    """
    row = conn.execute(sql, params or []).fetchone()
    if row is None:
        raise RuntimeError(f"expected one row, got none. SQL: {sql}")
    return row
