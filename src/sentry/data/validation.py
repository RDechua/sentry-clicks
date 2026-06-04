"""Post-ingest validation checks for the `clicks` DuckDB table.

`validate_ingestion(db_path)` runs the checks listed in build guide Task 1.6:
non-zero row count, no nulls in required columns, `is_attributed` in {0, 1},
`click_time` in the expected date range, and class balance within bounds.

All checks are computed by a single SQL aggregation pass (one full table scan
with parallel `COUNT(*) FILTER` aggregates) so cost is bounded by the data,
not by the check count. At 200M rows this is ~10x faster than per-check
queries.

The result is a structured `ValidationResult` so each check is testable in
isolation and the structure slots into the future audit log (Task 1.9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel

from sentry.data.schema import (
    EXPECTED_CLASS_BALANCE_MAX,
    EXPECTED_CLASS_BALANCE_MIN,
    NOT_NULL_COLUMNS,
    TABLE_NAME,
    TRAIN_DATE_MAX_EXCLUSIVE,
    TRAIN_DATE_MIN,
)


class CheckResult(BaseModel):
    """One validation check's outcome."""

    name: str
    ok: bool
    detail: str


class ValidationResult(BaseModel):
    """Aggregate result of all validation checks against an ingested DB."""

    row_count: int
    class_balance: float
    checks: list[CheckResult]

    @property
    def ok(self) -> bool:
        """True iff every individual check passed."""
        return all(c.ok for c in self.checks)

    @property
    def errors(self) -> list[CheckResult]:
        """Subset of checks that failed (empty when `ok` is True)."""
        return [c for c in self.checks if not c.ok]


def validate_ingestion(
    db_path: Path | str,
    expected_balance: tuple[float, float] | None = (
        EXPECTED_CLASS_BALANCE_MIN,
        EXPECTED_CLASS_BALANCE_MAX,
    ),
) -> ValidationResult:
    """Run sanity checks against an ingested clicks table.

    Parameters
    ----------
    db_path:
        Path to the DuckDB database file containing a `clicks` table.
    expected_balance:
        `(min, max)` bounds for mean(is_attributed). Defaults to TalkingData's
        documented ~0.2% positive rate. Pass `None` to skip this check — useful
        for fixture-based tests whose hand-crafted positive rate is not
        representative of production.

    Returns
    -------
    ValidationResult with one CheckResult per check. `result.ok` is True iff
    all checks passed.
    """
    # Build the combined aggregation query. Each check becomes a column in the
    # single resulting row; aliases let us look up by name after.
    null_aggs = ",\n            ".join(
        f"COUNT(*) FILTER (WHERE {col} IS NULL) AS nulls_{col}" for col in NOT_NULL_COLUMNS
    )
    sql = f"""
        SELECT
            COUNT(*) AS row_count,
            {null_aggs},
            COUNT(*) FILTER (WHERE is_attributed NOT IN (0, 1)) AS bad_labels,
            MIN(click_time) AS min_ct,
            MAX(click_time) AS max_ct,
            AVG(CAST(is_attributed AS DOUBLE)) AS class_balance
        FROM {TABLE_NAME}
    """

    with duckdb.connect(str(db_path), read_only=True) as conn:
        cursor = conn.execute(sql)
        col_names = [d[0] for d in cursor.description]
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"validation query returned no rows (table empty?). SQL: {sql}")
        agg: dict[str, Any] = dict(zip(col_names, row, strict=True))

    row_count = int(agg["row_count"])
    bal = agg["class_balance"]
    class_balance = float(bal) if bal is not None else 0.0
    min_ct, max_ct = agg["min_ct"], agg["max_ct"]

    checks: list[CheckResult] = [
        CheckResult(
            name="row_count_nonzero",
            ok=row_count > 0,
            detail=f"row_count={row_count}",
        ),
    ]

    for col in NOT_NULL_COLUMNS:
        n = int(agg[f"nulls_{col}"])
        checks.append(
            CheckResult(
                name=f"no_nulls_in_{col}",
                ok=n == 0,
                detail=("ok" if n == 0 else f"{n} nulls in {col}"),
            )
        )

    bad_labels = int(agg["bad_labels"])
    bad_label_detail = (
        "ok" if bad_labels == 0 else f"{bad_labels} rows with is_attributed not in (0, 1)"
    )
    checks.append(
        CheckResult(
            name="is_attributed_in_0_1",
            ok=bad_labels == 0,
            detail=bad_label_detail,
        )
    )

    # Exclusive upper bound on the date range avoids losing the final second
    # that a `<= 23:59:59` boundary silently misses.
    date_ok = (
        min_ct is not None
        and max_ct is not None
        and min_ct >= TRAIN_DATE_MIN
        and max_ct < TRAIN_DATE_MAX_EXCLUSIVE
    )
    checks.append(
        CheckResult(
            name="click_time_in_expected_range",
            ok=date_ok,
            detail=f"min={min_ct}, max={max_ct}",
        )
    )

    if expected_balance is not None:
        lo, hi = expected_balance
        balance_ok = lo <= class_balance <= hi
        checks.append(
            CheckResult(
                name="class_balance_in_range",
                ok=balance_ok,
                detail=(f"mean(is_attributed)={class_balance:.5f}, expected [{lo}, {hi}]"),
            )
        )

    return ValidationResult(
        row_count=row_count,
        class_balance=class_balance,
        checks=checks,
    )
