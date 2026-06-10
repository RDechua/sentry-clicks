"""Task 4.6: feature-family ablation.

Retrain the model with each family (F1 per-click, F2 velocity, F3
aggregates) removed, evaluate val PR-AUC, and compare to the full-feature
model (lgbm-v0.1.0, val PR-AUC 0.5618 on the same val). The delta per
family is the most direct evidence of which features carry the signal.

Same fit (`fit_lightgbm`, DEFAULT_PARAMS) as the full model, so the only
variable is the feature set. Families are run sequentially with the
matrices freed between them to stay inside the 3.8 GB container. Run:
    docker compose run --rm sentry python experiments/ablation.py
"""

from __future__ import annotations

import gc
from pathlib import Path

import duckdb
from sklearn.metrics import average_precision_score, roc_auc_score

from sentry.models.train import LABEL, MODEL_FEATURES, fit_lightgbm

VERSION = "v0.5.0"
FULL_PR_AUC = 0.5618  # lgbm-v0.1.0 on full v0.5.0 val (Task 4.3)
FAMILIES = ("f1", "f2", "f3")


def _load(split: str, features: list[str]):
    cols = ", ".join(
        [*(f"CAST({n} AS FLOAT) AS {n}" for n in features), f"{LABEL}::TINYINT {LABEL}"]
    )
    with duckdb.connect() as conn:
        conn.execute("SET memory_limit = '1.2GB'")
        return conn.execute(
            f"SELECT {cols} FROM read_parquet('artifacts/features/{VERSION}/{split}.parquet')"
        ).fetch_df()


rows = [("full (all families)", len(MODEL_FEATURES), FULL_PR_AUC, FULL_PR_AUC - FULL_PR_AUC)]

for family in FAMILIES:
    feats = [f for f in MODEL_FEATURES if not f.startswith(f"{family}_")]
    tr = _load("train", feats)
    va = _load("val", feats)
    model = fit_lightgbm(tr[feats], tr[LABEL].to_numpy(), va[feats], va[LABEL].to_numpy())
    proba = model.predict_proba(va[feats])[:, 1]
    pr = float(average_precision_score(va[LABEL], proba))
    roc = float(roc_auc_score(va[LABEL], proba))
    rows.append((f"without {family.upper()}", len(feats), pr, pr - FULL_PR_AUC))
    print(
        f"without {family.upper()}: {len(feats)} feats, val PR-AUC={pr:.4f} (Δ={pr - FULL_PR_AUC:+.4f}), ROC={roc:.4f}",
        flush=True,
    )
    del tr, va, model, proba
    gc.collect()

lines = [
    "# Feature-family ablation (Task 4.6)",
    "",
    f"Each family removed in turn; model retrained on full **{VERSION}** with the same "
    "fit and hyperparameters as the full model, evaluated on full val. Delta vs the "
    f"full-feature model (lgbm-v0.1.0, PR-AUC {FULL_PR_AUC:.4f}).",
    "",
    "| model | n_features | val PR-AUC | Δ PR-AUC |",
    "|---|---|---|---|",
]
for name, nfeat, pr, delta in rows:
    lines.append(f"| {name} | {nfeat} | {pr:.4f} | {delta:+.4f} |")
lines.append("")
Path("reports/ablation.md").write_text("\n".join(lines))
print("\nwrote reports/ablation.md")
