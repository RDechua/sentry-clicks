"""Sentry-Clicks CLI.

Currently exposes one command — the tracer-bullet pipeline (Task 1.10,
upgraded to real features in Task 2.7):

    sentry pipeline --sample

Fires one shot through the whole system: raw CSV → canonical split views →
F1+F2 feature pipeline → logistic-regression baseline → evaluation harness
on the VAL split → triage → an audit entry for every decision. The model
is still a deliberate baseline (the real LightGBM arrives in Week 4); the
point is that every layer stays connected as each gets replaced.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import duckdb
import numpy as np
import pandas as pd
import structlog
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sentry.audit.logger import DEFAULT_AUDIT_DB_PATH, log_events
from sentry.audit.schema import Action, AuditLogEntry, FeatureContribution
from sentry.data.ingestion import ingest_csv_to_duckdb
from sentry.data.splits import create_split_views
from sentry.evaluation.harness import evaluate
from sentry.features.f1_per_click import F1_FEATURES
from sentry.features.f2_velocity import F2_FEATURES
from sentry.features.pipeline import FeaturePipeline

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

    # 2. Canonical time-based split views (Task 2.1). The tracer trains on
    # train and evaluates on VAL — the test view exists but is touched only
    # at the Task 4.7 formal evaluation, per CLAUDE.md §3.1.
    typer.echo("[2/6] Creating split views")
    counts = create_split_views(db_path)
    typer.echo(f"      {counts}")

    # 3. Real features: F1 per-click + F2 velocity through the pipeline.
    typer.echo("[3/6] Computing F1+F2 features (train, val)")
    feature_pipeline = FeaturePipeline([*F1_FEATURES, *F2_FEATURES])
    with duckdb.connect(str(db_path), read_only=True) as conn:
        train = feature_pipeline.compute(conn, source="clicks_train")
        val = feature_pipeline.compute(conn, source="clicks_val")
    typer.echo(f"      train={len(train):,}, val={len(val):,}")

    # 4. Baseline model. Logistic regression can't consume the two string
    # interaction features (LightGBM will, in Week 4) and can't route NULLs,
    # so the tracer uses numeric features with a documented -1 sentinel
    # (§3.4 allows a sentinel when recorded; decisions.md Task 2.7) and a
    # scaler so the L2 penalty treats unitless columns comparably.
    numeric_features = [
        name
        for name in feature_pipeline.feature_names
        if name not in ("f1_ip_app_interaction", "f1_ip_device_interaction")
    ]
    typer.echo(f"[4/6] Training logistic regression on {len(numeric_features)} features")
    x_train = train[numeric_features].astype("float64").fillna(-1.0).to_numpy()
    y_train = train["is_attributed"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
    )
    model.fit(x_train, y_train)

    typer.echo("[5/6] Predicting + evaluating on val")
    x_val = val[numeric_features].astype("float64").fillna(-1.0).to_numpy()
    y_val = val["is_attributed"].to_numpy()
    y_pred_proba = model.predict_proba(x_val)[:, 1]
    result = evaluate(y_val, y_pred_proba, name="tracer-logreg-f1f2-val")
    typer.echo(
        f"      PR-AUC={result.pr_auc:.4f}, ROC-AUC={result.roc_auc:.4f}, "
        f"Brier={result.brier_score:.4f}"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(result.model_dump_json(indent=2))
    typer.echo(f"      Wrote {metrics_path}")

    # 6. Triage + an audit entry for EVERY decision (CLAUDE.md §3.9), one
    # log_events batch. Per-row contributions: coef_j * scaled_x_ij is the
    # exact additive term in a linear model's logit — the linear analogue of
    # a SHAP value (the real SHAP arrives with the tree model in Week 4).
    typer.echo(f"[6/6] Triage + audit ({len(val):,} decisions)")
    scaled = model.named_steps["standardscaler"].transform(x_val)
    coefs = model.named_steps["logisticregression"].coef_[0]
    contributions = scaled * coefs  # (n_rows, n_features)
    top5_idx = np.argsort(-np.abs(contributions), axis=1)[:, :5]
    logged_at = datetime.now(tz=UTC)  # one batch, one write timestamp

    entries = []
    for i, (proba, row_id, click_ts) in enumerate(
        zip(y_pred_proba, val["row_id"].to_numpy(), val["click_time"].to_numpy(), strict=True)
    ):
        action = Action.AUTO_BLOCK if proba > _TRACER_THRESHOLD_BLOCK else Action.ALLOW
        top_features = [
            FeatureContribution(
                feature_name=numeric_features[j],
                value=float(x_val[i, j]),
                shap_contribution=float(contributions[i, j]),
            )
            for j in top5_idx[i]
        ]
        entries.append(
            AuditLogEntry(
                event_timestamp=logged_at,
                case_id=f"tracer-row{int(row_id)}",  # row_id: stable, collision-free
                click_timestamp=pd.Timestamp(click_ts).to_pydatetime(),
                model_version="tracer-logreg-f1f2-v1",
                policy_version="tracer-policy-v0",
                raw_score=float(proba),
                calibrated_score=float(proba),  # no calibration until Task 4.5
                threshold_block=_TRACER_THRESHOLD_BLOCK,
                threshold_review=_TRACER_THRESHOLD_BLOCK,  # no review band — see top of file
                action=action,
                top_features=top_features,
            )
        )
    n_logged = log_events(entries, db_path=audit_db_path)
    typer.echo(f"      {n_logged:,} audit entries written to {audit_db_path}")

    typer.echo(f"\nDone in {time.time() - start:.1f}s. PR-AUC={result.pr_auc:.4f}")
    typer.echo("  (Baseline on real features — Week 4 replaces the model itself.)")


if __name__ == "__main__":
    app()
