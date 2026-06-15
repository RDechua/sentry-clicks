"""Sentry-Clicks CLI.

Currently exposes one command — the tracer-bullet pipeline (Task 1.10,
upgraded to real features in Task 2.7):

    sentry pipeline --sample

Fires one shot through the whole system: raw CSV → canonical split views →
F1+F2+F3 feature pipeline → logistic-regression baseline → evaluation
harness on the VAL split → triage → an audit entry for every decision. The
model is still a deliberate baseline (the real LightGBM arrives in Week 4);
the point is that every layer stays connected as each gets replaced.
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
from sentry.features.f3_aggregates import F3_FEATURES
from sentry.features.pipeline import FeaturePipeline
from sentry.models.predict import load_model, top_contributions
from sentry.models.train import LABEL, MODEL_FEATURES, train_model
from sentry.triage.cost import fraud_probability
from sentry.triage.report import write_review_report
from sentry.triage.router import TriageThresholds, route_batch

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

    # 3. Real features: F1 per-click + F2 velocity + F3 aggregates.
    typer.echo("[3/6] Computing F1+F2+F3 features (train, val)")
    feature_pipeline = FeaturePipeline([*F1_FEATURES, *F2_FEATURES, *F3_FEATURES])
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
    result = evaluate(y_val, y_pred_proba, name="tracer-logreg-f1f2f3-val")
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
                model_version="tracer-logreg-f1f2f3-v2",
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


@app.command("train")
def train(
    features_version: Annotated[
        str, typer.Option(help="Feature-store version to train on (e.g. v0.5.0).")
    ],
    output: Annotated[Path, typer.Option(help="Directory for model.txt + metadata.json.")] = Path(
        "artifacts/models/lgbm-v0.1.0"
    ),
    model_version: Annotated[str, typer.Option(help="Model version label.")] = "lgbm-v0.1.0",
) -> None:
    """Train the LightGBM model on a feature version (Task 4.3)."""
    start = time.time()
    typer.echo(f"Training {model_version} on features {features_version}")
    result = train_model(features_version, output, model_version=model_version)
    typer.echo(
        f"Done in {time.time() - start:.1f}s. best_iteration={result.best_iteration}, "
        f"val PR-AUC={result.val_pr_auc:.4f}, ROC-AUC={result.val_roc_auc:.4f}"
    )
    typer.echo(f"  Wrote {output}/model.txt + metadata.json")


@app.command("tune")
def tune(
    features_version: Annotated[
        str, typer.Option(help="Feature-store version to tune on (e.g. v0.5.0).")
    ],
    n_trials: Annotated[int, typer.Option(help="Optuna trial budget (50-100).")] = 50,
    study_db: Annotated[Path, typer.Option(help="SQLite path for the resumable study.")] = Path(
        "artifacts/tuning/study.db"
    ),
    output: Annotated[Path, typer.Option(help="Final tuned-model directory.")] = Path(
        "artifacts/models/lgbm-v0.2.0"
    ),
) -> None:
    """Optuna search on a subsample, then final-fit the best params on full data (Task 4.4)."""
    from sentry.models.tune import best_full_params, tune_lgbm

    start = time.time()
    typer.echo(f"Tuning on {features_version} ({n_trials} trials, study {study_db})")
    study = tune_lgbm(features_version, study_db, n_trials=n_trials)
    typer.echo(
        f"  Best val PR-AUC (subsample) = {study.best_value:.4f} @ trial {study.best_trial.number}"
    )
    typer.echo(f"  Best params: {study.best_params}")

    typer.echo("Final fit on FULL data with best params -> lgbm-v0.2.0")
    result = train_model(
        features_version, output, model_version="lgbm-v0.2.0", params=best_full_params(study)
    )
    typer.echo(
        f"Done in {time.time() - start:.1f}s. tuned val PR-AUC={result.val_pr_auc:.4f}, "
        f"ROC-AUC={result.val_roc_auc:.4f}, best_iteration={result.best_iteration}"
    )


# Enforcement DEMO thresholds are set at fraud-score QUANTILES so all three
# tiers are populated for the review-queue artifact. They are explicitly NOT
# the operating policy: a fixed fraud-score threshold can't make a balanced
# split here because val is 99.78% non-attributed and the model scores those
# confidently (fraud scores pile up near 1.0), so any cutoff below ~0.99
# blocks almost everything. The real thresholds are cost-based (Week 5 /
# tradeoffs.md), where at illustrative costs the optimum auto-decides with no
# review tier. This demo exercises the routing/SHAP/audit/report machinery,
# not the policy. (§3.6 forbids percentile thresholds for SELECTION; this is
# artifact population, not selection.)
_ENFORCE_BLOCK_FRAC = 0.005  # top 0.5% of fraud scores -> AUTO_BLOCK
_ENFORCE_REVIEW_FRAC = 0.020  # next 2% -> HUMAN_REVIEW
_ENFORCE_QA_RATE = 0.005
_ENFORCE_SEED = 42


@app.command("enforce")
def enforce(
    features_version: Annotated[str, typer.Option(help="Feature-store version (e.g. v0.5.0).")],
    split: Annotated[str, typer.Option(help="Split to score: train/val/test.")] = "val",
    model_dir: Annotated[Path, typer.Option(help="Trained model directory.")] = Path(
        "artifacts/models/lgbm-v0.1.0"
    ),
    limit: Annotated[
        int, typer.Option(help="Rows to enforce on (demo tractability; 0 = all).")
    ] = 100_000,
    audit_db_path: Annotated[Path, typer.Option(help="Audit DB.")] = DEFAULT_AUDIT_DB_PATH,
    report_path: Annotated[Path, typer.Option(help="Review-queue HTML.")] = Path(
        "reports/sample_review_queue.html"
    ),
) -> None:
    """Score a split, route to enforcement actions, write the audit log + review report.

    The enforcement (inference) pipeline: load the trained model + calibrator,
    score the split, route via the three-tier policy, log every enforcement
    ACTION (block/review/QA) with full SHAP top-5, and render the human-review
    queue. ALLOWs are audited via the QA sample, not individually (logging
    every allow at stream scale is infeasible and unrealistic). Decisions in
    docs/decisions.md (Task 6.5).
    """
    start = time.time()
    bundle = load_model(model_dir)

    typer.echo(f"[1/4] Loading {split} features (v{features_version}, limit={limit or 'all'})")
    cols = ", ".join(
        [*(f"CAST({n} AS FLOAT) AS {n}" for n in MODEL_FEATURES), "row_id", "click_time", LABEL]
    )
    limit_clause = f"LIMIT {limit}" if limit else ""
    parquet = f"artifacts/features/{features_version}/{split}.parquet"
    with duckdb.connect() as conn:
        conn.execute("SET memory_limit = '1.5GB'")
        df = conn.execute(f"SELECT {cols} FROM read_parquet('{parquet}') {limit_clause}").fetch_df()
    typer.echo(f"      {len(df):,} rows")

    typer.echo("[2/4] Scoring + routing")
    raw_p = np.asarray(bundle.booster.predict(df[list(MODEL_FEATURES)]))
    calibrated = bundle.calibrator.predict(raw_p) if bundle.calibrator else raw_p
    # Route the DEMO on the RAW fraud score (1 - raw P): isotonic calibration
    # ties a large mass of cases at calibrated fraud = 1.0, so calibrated
    # quantiles collapse and no review band can form. The raw score has
    # continuous spread, so quantile thresholds populate all three tiers. The
    # audit log still records BOTH raw and calibrated per case; production
    # would route on calibrated per the cost model (which auto-decides anyway).
    raw_fraud = fraud_probability(raw_p)
    t_block = float(np.quantile(raw_fraud, 1.0 - _ENFORCE_BLOCK_FRAC))
    t_review = float(np.quantile(raw_fraud, 1.0 - _ENFORCE_BLOCK_FRAC - _ENFORCE_REVIEW_FRAC))
    thresholds = TriageThresholds(block=t_block, review=t_review)
    typer.echo(
        f"      demo thresholds (raw fraud score): block={t_block:.4f}, review={t_review:.4f}"
    )
    actions = route_batch(
        raw_fraud, thresholds, _ENFORCE_QA_RATE, np.random.default_rng(_ENFORCE_SEED)
    )
    routed_mask = actions != Action.ALLOW.value
    n_routed = int(routed_mask.sum())
    typer.echo(
        f"      block={int((actions == Action.AUTO_BLOCK.value).sum()):,} "
        f"review={int((actions == Action.HUMAN_REVIEW.value).sum()):,} "
        f"qa={int((actions == Action.QA_SAMPLE.value).sum()):,} "
        f"allow={int((actions == Action.ALLOW.value).sum()):,}"
    )

    typer.echo(f"[3/4] SHAP + audit for {n_routed:,} enforcement actions")
    routed = df[routed_mask].reset_index(drop=True)
    routed_actions = actions[routed_mask]
    routed_cal = calibrated[routed_mask]
    routed_raw = raw_p[routed_mask]
    contributions = top_contributions(bundle, routed)
    logged_at = datetime.now(tz=UTC)
    entries = [
        AuditLogEntry(
            event_timestamp=logged_at,
            case_id=f"{split}-row{int(routed['row_id'].iloc[i])}",
            click_timestamp=pd.Timestamp(routed["click_time"].iloc[i]).to_pydatetime(),
            model_version=bundle_version(model_dir),
            policy_version="enforce-demo-v1",
            raw_score=float(routed_raw[i]),
            calibrated_score=float(routed_cal[i]),
            threshold_block=thresholds.block,
            threshold_review=thresholds.review,
            action=Action(routed_actions[i]),
            top_features=contributions[i],
        )
        for i in range(n_routed)
    ]
    n_logged = log_events(entries, db_path=audit_db_path)
    typer.echo(f"      {n_logged:,} audit entries -> {audit_db_path}")

    typer.echo("[4/4] Writing human-review report")
    n_cases = write_review_report(audit_db_path, report_path)
    typer.echo(f"      {n_cases:,} review cases -> {report_path}")
    typer.echo(f"\nDone in {time.time() - start:.1f}s.")


def bundle_version(model_dir: Path) -> str:
    """Read the model_version from a model dir's metadata, or fall back to its name."""
    import json

    meta = model_dir / "metadata.json"
    if meta.exists():
        version: str = json.loads(meta.read_text()).get("model_version", model_dir.name)
        return version
    return model_dir.name


if __name__ == "__main__":
    app()
