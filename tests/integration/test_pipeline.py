"""Integration test for the Task 1.10 tracer-bullet pipeline.

Skipped (via the `train_sample_path` conftest fixture) when the Kaggle
`train_sample.csv` isn't mounted at `/data/`. When data is present, invokes
the documented `sentry pipeline --sample` command via typer's `CliRunner`
in-process — no subprocess — and verifies:

- exit code 0
- metrics JSON file exists and round-trips through `EvaluationResult`
- audit DB has exactly one row per VAL-split decision (CLAUDE.md §3.9;
  the canonical test split is untouched until Task 4.7)
- clicks DB has the expected 100k rows
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from sentry.audit.logger import AUDIT_TABLE_NAME
from sentry.cli import app
from sentry.data._db import fetch_one
from sentry.data.schema import TABLE_NAME
from sentry.evaluation.harness import EvaluationResult


@pytest.mark.integration
def test_pipeline_runs_end_to_end(tmp_path: Path, train_sample_path: Path) -> None:
    db_path = tmp_path / "tracer_clicks.duckdb"
    audit_db_path = tmp_path / "tracer_audit.duckdb"
    metrics_path = tmp_path / "tracer_metrics.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "pipeline",
            "--sample",  # exercises the documented invocation: --sample selects the input
            "--db-path",
            str(db_path),
            "--audit-db-path",
            str(audit_db_path),
            "--metrics-path",
            str(metrics_path),
        ],
    )

    assert result.exit_code == 0, f"Pipeline failed:\n{result.output}"

    # Metrics JSON exists and is a valid EvaluationResult.
    assert metrics_path.exists()
    evaluation = EvaluationResult.model_validate_json(metrics_path.read_text())
    assert evaluation.n_samples > 0
    assert evaluation.n_positive > 0
    # The tracer model is terrible by design — PR-AUC will be tiny. Just
    # assert it's in [0, 1].
    assert 0.0 <= evaluation.pr_auc <= 1.0

    # Clicks DB has the expected 100k rows.
    assert db_path.exists()
    with duckdb.connect(str(db_path), read_only=True) as conn:
        n_clicks = fetch_one(conn, f"SELECT COUNT(*) FROM {TABLE_NAME}")[0]
    assert n_clicks == 100_000

    # Every triage decision produced an audit entry — the §3.9 completeness
    # invariant. The canonical split boundaries (Task 2.1) put exactly
    # 20,087 of the 100k rows in val, and the pipeline decides on every
    # val row (test stays untouched until Task 4.7).
    assert audit_db_path.exists()
    with duckdb.connect(str(audit_db_path), read_only=True) as conn:
        n_audit = fetch_one(conn, f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}")[0]
    assert n_audit == 20_087, f"audit rows ({n_audit}) != val decisions (20087)"
