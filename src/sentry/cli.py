"""Sentry-Clicks CLI.

Currently exposes one command — the Task 1.10 tracer-bullet pipeline:

    sentry pipeline --sample

Fires a single shot through the whole system (raw CSV → trivial feature →
trivial model → evaluation → triage → audit log) before any layer is real.
The model and metrics are intentionally terrible (one feature, plain
logistic regression); the point is that every component is connected
and tested. Subsequent weeks replace each layer with a real implementation.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import duckdb
import pandas as pd
import structlog
import typer
from sklearn.linear_model import LogisticRegression

from sentry.audit.logger import DEFAULT_AUDIT_DB_PATH, log_events
from sentry.audit.schema import Action, AuditLogEntry, FeatureContribution
from sentry.data.ingestion import ingest_csv_to_duckdb
from sentry.data.schema import TABLE_NAME
from sentry.evaluation.harness import evaluate

app = typer.Typer(add_completion=False, help="Sentry-Clicks CLI.")
logger = structlog.get_logger(__name__)


@app.callback()
def main() -> None:
    """Sentry-Clicks: mobile ad click fraud detection.

    The explicit callback keeps typer in subcommand mode — with exactly one
    command and no callback, typer collapses the app into a root command and
    `sentry pipeline` breaks with "unexpected extra argument". Found by the
    tracer-bullet integration test; later weeks add more commands here.
    """


# Trivial triage threshold — non-real. Real values come from the cost-based
# sweep in Task 5.2. The tracer's triage is two-way (build guide: AUTO_BLOCK
# above 0.5, else ALLOW), so there is no review band — the audit entries log
# threshold_review == threshold_block so the recorded policy replays to the
# actions actually taken.
_TRACER_THRESHOLD_BLOCK = 0.5

# Default paths as module-level constants so the signature has no calls in
# argument defaults (flake8-bugbear B008).
_SAMPLE_CSV_PATH = Path("/data/train_sample.csv")
_DEFAULT_CLICKS_DB_PATH = Path("artifacts/sentry.duckdb")
_DEFAULT_METRICS_PATH = Path("reports/tracer_metrics.json")


@app.command("pipeline")
def pipeline(
    sample: Annotated[
        bool,
        typer.Option("--sample", help="Run on the 100k train_sample.csv."),
    ] = False,
    csv_path: Annotated[
        Path | None, typer.Option(help="Path to the input CSV (overrides --sample).")
    ] = None,
    db_path: Annotated[
        Path, typer.Option(help="Path to the clicks DuckDB database.")
    ] = _DEFAULT_CLICKS_DB_PATH,
    audit_db_path: Annotated[
        Path, typer.Option(help="Path to the audit log DuckDB database.")
    ] = DEFAULT_AUDIT_DB_PATH,
    metrics_path: Annotated[
        Path, typer.Option(help="Where to write the JSON metrics.")
    ] = _DEFAULT_METRICS_PATH,
) -> None:
    """Tracer-bullet end-to-end: ingest → feature → model → eval → triage + audit."""
    # `--sample` owns input selection. A bare `sentry pipeline` fails loudly
    # instead of silently defaulting — when Week 4 gives the bare invocation
    # full-data semantics, nobody is surprised by a multi-hour 200M-row run.
    if csv_path is None:
        if not sample:
            raise typer.BadParameter("pass --sample (full-data mode arrives in Week 4)")
        csv_path = _SAMPLE_CSV_PATH

    start = time.time()
    typer.echo(f"Tracer-bullet pipeline starting (--sample={sample})")

    typer.echo(f"[1/6] Ingesting {csv_path} → {db_path}")
    n_rows = ingest_csv_to_duckdb(csv_path, db_path)
    typer.echo(f"      {n_rows:,} rows ingested")

    # 2. Trivial feature: clicks per IP across the WHOLE dataset (the build
    # guide's "clicks_per_ip_in_dataset" — the suffix is the point: it spans
    # all splits and all time, so it leaks. Deliberate for the tracer only;
    # real features in Weeks 2-3 use strictly-prior windows per CLAUDE.md
    # §3.4).
    typer.echo("[2/6] Computing trivial feature (clicks_per_ip)")
    with duckdb.connect(str(db_path), read_only=True) as conn:
        df = conn.execute(f"""
            SELECT ip, click_time, is_attributed,
                   COUNT(*) OVER (PARTITION BY ip) AS clicks_per_ip
            FROM {TABLE_NAME}
            ORDER BY click_time
            """).fetch_df()
    typer.echo(f"      {len(df):,} rows with feature")

    # 3. Time-based 60/20/20 split per CLAUDE.md §3.1. Even the tracer bullet
    # uses time-based splits — random splitting on a time-series problem is
    # the methodology error this project specifically avoids.
    typer.echo("[3/6] Time-based 60/20/20 split")
    n = len(df)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    typer.echo(f"      train={len(train):,}, val={len(val):,}, test={len(test):,}")

    typer.echo("[4/6] Training logistic regression on clicks_per_ip")
    x_train = train[["clicks_per_ip"]].to_numpy()
    y_train = train["is_attributed"].to_numpy()
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(x_train, y_train)

    typer.echo("[5/6] Predicting + evaluating on test")
    x_test = test[["clicks_per_ip"]].to_numpy()
    y_test = test["is_attributed"].to_numpy()
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    result = evaluate(y_test, y_pred_proba, name="tracer-logreg-clicks_per_ip")
    typer.echo(
        f"      PR-AUC={result.pr_auc:.4f}, ROC-AUC={result.roc_auc:.4f}, "
        f"Brier={result.brier_score:.4f}"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(result.model_dump_json(indent=2))
    typer.echo(f"      Wrote {metrics_path}")

    # 6. Trivial triage + audit logging for EVERY decision (CLAUDE.md §3.9:
    # decisions without log entries are bugs), written via one `log_events` batch.
    typer.echo(f"[6/6] Triage + audit ({len(test):,} decisions)")
    coef = float(model.coef_[0][0])
    logged_at = datetime.now(tz=UTC)  # one batch, one write timestamp
    entries = []
    for proba, ip, click_ts, ip_count in zip(
        y_pred_proba,
        test["ip"].to_numpy(),
        test["click_time"].to_numpy(),
        test["clicks_per_ip"].to_numpy(),
        strict=True,
    ):
        click_at = pd.Timestamp(click_ts)
        count = int(ip_count)
        action = Action.AUTO_BLOCK if proba > _TRACER_THRESHOLD_BLOCK else Action.ALLOW
        entries.append(
            AuditLogEntry(
                event_timestamp=logged_at,
                case_id=f"tracer-ip{int(ip)}-{click_at.isoformat()}",
                click_timestamp=click_at.to_pydatetime(),
                model_version="tracer-logreg-v0",
                policy_version="tracer-policy-v0",
                raw_score=float(proba),
                calibrated_score=float(proba),  # no calibration in tracer
                threshold_block=_TRACER_THRESHOLD_BLOCK,
                threshold_review=_TRACER_THRESHOLD_BLOCK,  # no review band — see top of file
                action=action,
                top_features=[
                    FeatureContribution(
                        feature_name="clicks_per_ip",
                        value=count,
                        shap_contribution=coef * count,
                    )
                ],
            )
        )
    n_logged = log_events(entries, db_path=audit_db_path)
    typer.echo(f"      {n_logged:,} audit entries written to {audit_db_path}")

    typer.echo(f"\nDone in {time.time() - start:.1f}s. PR-AUC={result.pr_auc:.4f}")
    typer.echo("  (Tracer is barely above random by design — Week 2-4 replaces each layer.)")


if __name__ == "__main__":
    app()
