"""Task 4.2: evaluate B1/B2/B3 on v0.5.0 and write reports/baselines.md.

B1/B2/B3 score the val split through the same evaluation harness as every
real model (Task 4.2 = baselines only; the LightGBM is Task 4.3). For
context the table also cites the untuned default-LightGBM number the
density gate already computed on these exact v0.5.0 features — recomputing
it here would mean importing LightGBM and holding its full-train fit
alongside B3's float64 copies, which OOMs the 3.8 GB container.

Memory-frugal by necessity (3.8 GB container, 11M train rows): each step
loads only the columns it needs and frees them before the next. Run:
    docker compose run --rm sentry python experiments/run_baselines.py
"""

from __future__ import annotations

import gc
import resource
from pathlib import Path

import duckdb
import numpy as np


def mem(tag: str) -> None:
    # ru_maxrss is bytes on macOS, KB on Linux (container is Linux).
    mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [mem] {tag}: peak RSS {mb:.0f} MB", flush=True)

from sentry.evaluation.harness import EvaluationResult, compare, evaluate
from sentry.models.baselines import (
    B2_FEATURE,
    F1F2_NUMERIC_FEATURES,
    LABEL,
    SEED,
    fit_logreg_scores,
    score_random,
)

VERSION = "v0.5.0"
TRAIN = f"artifacts/features/{VERSION}/train.parquet"
VAL = f"artifacts/features/{VERSION}/val.parquet"
# Default LightGBM on all numeric v0.5.0 features, from experiments/density_gate.py
# (same data, same untuned params) — the 4.3/4.4 starting point, cited not recomputed.
DEFAULT_LGBM_PR_AUC = 0.5095


def load(path: str, cols: list[str], sample_pct: int | None = None):  # noqa: ANN201
    select = ", ".join(f"CAST({c} AS FLOAT) AS {c}" for c in cols)
    sample = f"USING SAMPLE {sample_pct} PERCENT (reservoir, {SEED})" if sample_pct else ""
    with duckdb.connect() as conn:
        # Cap DuckDB so it doesn't reserve ~80% of the 3.8 GB VM and starve
        # the Python/sklearn side; the loads are simple scans that fit easily.
        conn.execute("SET memory_limit = '900MB'")
        return conn.execute(f"SELECT {select} FROM read_parquet('{path}') {sample}").fetch_df()


def scalar(sql: str) -> float:
    with duckdb.connect() as conn:
        return float(conn.execute(sql).fetchone()[0])


# Linear baseline trains on a seeded 20% sample of train (~2.2M rows): a
# logreg's coefficients are statistically indistinguishable at that size
# (~5.6k positives, ample for 12 features) and the float64 upcast +
# StandardScaler copy of the full 11M matrix OOMs the 3.8 GB container.
# The GBM context row uses full train (LightGBM is histogram-based, frugal).
B3_TRAIN_SAMPLE_PCT = 20

y_val = load(VAL, [LABEL])[LABEL].to_numpy()
n_val = len(y_val)
n_pos = int(y_val.sum())
results: list[EvaluationResult] = []

# B1 — random floor.
results.append(evaluate(y_val, score_random(n_val, seed=SEED), name="B1_random"))
mem("after B1")

# B2 — pair conversion rate as the score; NULL -> train base rate (via SQL,
# so train is never loaded into pandas).
prior = scalar(f"SELECT AVG({LABEL}) FROM read_parquet('{TRAIN}')")
b2_scores = load(VAL, [B2_FEATURE])[B2_FEATURE].fillna(prior).to_numpy()
results.append(evaluate(y_val, b2_scores, name="B2_pair_conversion_rate"))
del b2_scores
gc.collect()
mem("after B2")

# B3 — logreg on F1+F2 only (sampled train; full val). fit_logreg_scores
# frees its internal matrices on return; the input frames are freed BEFORE
# evaluate(), so only the score array is resident during the harness call
# (the harness is the memory hog at multi-M rows — Task 1.8 deferred debt,
# fixed in 4.4).
tr = load(TRAIN, [*F1F2_NUMERIC_FEATURES, LABEL], sample_pct=B3_TRAIN_SAMPLE_PCT)
va = load(VAL, [*F1F2_NUMERIC_FEATURES, LABEL])
b3 = fit_logreg_scores(tr, va, F1F2_NUMERIC_FEATURES, seed=SEED)
del tr, va
gc.collect()
mem("B3 fit done, inputs freed")
results.append(evaluate(y_val, b3, name="B3_logreg_f1f2"))
del b3
gc.collect()
print(f"B3 done (trained on {B3_TRAIN_SAMPLE_PCT}% sample)", flush=True)

table = compare(results)
print(table.to_string(index=False))

lines = [
    "# Baselines (Task 4.2)",
    "",
    f"Evaluated on the **{VERSION}** val split ({n_val:,} rows, {n_pos:,} positives, "
    f"{np.mean(y_val):.4%} base rate). All scores via the Week 1 evaluation harness; "
    "PR-AUC is the headline (§3.2).",
    "",
    "| model | PR-AUC | ROC-AUC | Brier |",
    "|---|---|---|---|",
]
for r in sorted(results, key=lambda x: x.pr_auc):
    lines.append(f"| {r.name} | {r.pr_auc:.4f} | {r.roc_auc:.4f} | {r.brier_score:.4f} |")
lines += [
    f"| default_lgbm_all_features* | {DEFAULT_LGBM_PR_AUC:.4f} | — | — |",
    "",
    f"B3 uses {len(F1F2_NUMERIC_FEATURES)} F1+F2 numeric features, trained on a seeded "
    f"{B3_TRAIN_SAMPLE_PCT}% sample of train (linear-fit coefficients are stable at "
    "~2.2M rows; the full-matrix float64 upcast OOMs the container).",
    "",
    "*Default LightGBM on all numeric F1+F2+F3 features (untuned) — the Task 4.3/4.4 "
    "starting point, cited from `experiments/density_gate.py` on the same v0.5.0 data, "
    "shown for context (not a baseline).",
    "",
]
out = Path("reports/baselines.md")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines))
print(f"\nwrote {out}")
