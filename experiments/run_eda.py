"""Run every SQL file in sql/01_eda/ against artifacts/sentry.duckdb.

Each SQL file is expected to end with a `COPY (...) TO 'reports/eda/NN_name.csv'`
statement, so this script just opens the DB and exec's each file in order.

Invoked via `make eda` from the project root. Outputs land in reports/eda/
(gitignored — local-only artifact). The committed EDA deliverables are the
SQL queries themselves and the corresponding decisions.md entry.

Promoted to a proper CLI command in Task 1.10; lives in experiments/ for now.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

_DB_PATH = Path("artifacts/sentry.duckdb")
_SQL_DIR = Path("sql/01_eda")
_OUT_DIR = Path("reports/eda")


def main() -> int:
    if not _DB_PATH.exists():
        print(f"DB not found: {_DB_PATH}. Ingest train_sample.csv first.", file=sys.stderr)
        return 1

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    sql_files = sorted(_SQL_DIR.glob("*.sql"))
    if not sql_files:
        print(f"No SQL files in {_SQL_DIR}.", file=sys.stderr)
        return 1

    with duckdb.connect(str(_DB_PATH), read_only=False) as conn:
        for sql_file in sql_files:
            print(f"→ {sql_file}")
            conn.execute(sql_file.read_text())

    print(f"Done. {len(sql_files)} files run; outputs in {_OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
