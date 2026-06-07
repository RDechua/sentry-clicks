"""The Week 4 density gate (decisions.md, Task 3.4).

At dev-sample scale the pair conversion rate ranked last because 84% of
its windows were empty. v0.3.0 features have FULL-density windows; the
Task 3.4 diagnosis predicts the pair rate now lands top-3 and PR-AUC
clears 0.80. If it doesn't, the sparsity explanation was wrong and
something real is broken — stop Week 4 and dig.

Loads numerics as float32 straight from the store parquets via DuckDB
(11M train rows must fit a ~3.9 GB container). The two string interaction
features are excluded: near-zero importance at every scale measured, and
their pandas memory cost at 11M rows is the flagged Task 2.3 revisit.

Run: docker compose run --rm sentry python experiments/density_gate.py [version]
(default v0.4.0 — full-history windows per the amended §3.4; pass v0.3.0
to reproduce the cold-start run this gate first failed on)
"""

from __future__ import annotations

import sys

import duckdb
import lightgbm as lgb
import numpy as np
import shap

from sentry.evaluation.harness import evaluate
from sentry.features.materialize import ALL_FEATURE_NAMES

SEED = 42
SHAP_SAMPLE = 100_000
EXCLUDED = ("f1_ip_app_interaction", "f1_ip_device_interaction")
FEATURES = [n for n in ALL_FEATURE_NAMES if n not in EXCLUDED]
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v0.4.0"


def load(split: str):  # noqa: ANN201 — experiment script
    cols = ", ".join(f"CAST({n} AS FLOAT) AS {n}" for n in FEATURES)
    with duckdb.connect() as conn:
        df = conn.execute(
            f"SELECT {cols}, is_attributed FROM "
            f"read_parquet('artifacts/features/{VERSION}/{split}.parquet')"
        ).fetch_df()
    y = df.pop("is_attributed")
    return df, y


print(f"gate version: {VERSION}")


x_train, y_train = load("train")
print(f"train: {x_train.shape}, positives={int(y_train.sum()):,} ({y_train.mean():.4%})")

model = lgb.LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1)
model.fit(x_train, y_train)
del x_train, y_train

x_val, y_val = load("val")
print(f"val:   {x_val.shape}, positives={int(y_val.sum()):,} ({y_val.mean():.4%})")
result = evaluate(y_val.to_numpy(), model.predict_proba(x_val)[:, 1], name="density-gate-val")
print(
    f"\nval: PR-AUC={result.pr_auc:.4f}, ROC-AUC={result.roc_auc:.4f}, "
    f"Brier={result.brier_score:.4f}"
)

rng = np.random.default_rng(SEED)
idx = rng.choice(len(x_val), size=min(SHAP_SAMPLE, len(x_val)), replace=False)
shap_values = shap.TreeExplainer(model).shap_values(x_val.iloc[idx])
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]
mean_abs = dict(zip(FEATURES, np.abs(shap_values).mean(axis=0), strict=True))

ranked = sorted(FEATURES, key=lambda n: -mean_abs[n])
print(f"\ntop 10 by mean |SHAP| (val sample n={len(idx):,}):")
for name in ranked[:10]:
    print(f"  {name:36s} {mean_abs[name]:8.4f}")

pair_rank = ranked.index("f3_ip_app_conversion_rate_24hr") + 1
print(f"\nGATE: pair rate rank #{pair_rank} (need <=3), PR-AUC {result.pr_auc:.4f} (need >=0.80)")
print("GATE PASSED" if pair_rank <= 3 and result.pr_auc >= 0.80 else "GATE FAILED — STOP AND DEBUG")
