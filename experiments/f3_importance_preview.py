"""Task 3.4: feature importance preview on F1+F2+F3.

Quick-and-dirty default LightGBM — NOT the Week 4 model. The questions:
does f3_ip_app_conversion_rate_24hr land in the top 3, and is val PR-AUC
in a sane band (too low = feature bug, ~0.95+ = suspect leakage)?

Run: docker compose run --rm sentry python experiments/f3_importance_preview.py
"""

from __future__ import annotations

import duckdb
import lightgbm as lgb
import numpy as np
import shap

from sentry.evaluation.harness import evaluate
from sentry.features.f1_per_click import F1_FEATURES
from sentry.features.f2_velocity import F2_FEATURES
from sentry.features.f3_aggregates import F3_FEATURES
from sentry.features.pipeline import FeaturePipeline

SEED = 42
SHAP_SAMPLE = 10_000

pipeline = FeaturePipeline([*F1_FEATURES, *F2_FEATURES, *F3_FEATURES])
with duckdb.connect("artifacts/sentry.duckdb", read_only=True) as conn:
    train = pipeline.compute(conn, source="clicks_train")
    val = pipeline.compute(conn, source="clicks_val")

feature_cols = list(pipeline.feature_names)


def prep(table):  # noqa: ANN001, ANN201 — experiment script
    x = table[feature_cols].copy()
    for col in ("f1_ip_app_interaction", "f1_ip_device_interaction"):
        x[col] = x[col].astype("category")
    return x


x_train, y_train = prep(train), train["is_attributed"]
x_val, y_val = prep(val), val["is_attributed"]

model = lgb.LGBMClassifier(
    class_weight="balanced",  # CLAUDE.md 3.3
    random_state=SEED,
    verbose=-1,
)
model.fit(x_train, y_train)

result = evaluate(y_val.to_numpy(), model.predict_proba(x_val)[:, 1], name="f1f2f3-preview-val")
print(
    f"\nval: PR-AUC={result.pr_auc:.4f}, ROC-AUC={result.roc_auc:.4f}, "
    f"Brier={result.brier_score:.4f}, positives={result.n_positive}/{result.n_samples}"
)

# Importance two ways: LightGBM split gain and mean |SHAP| on a val sample.
gain = dict(zip(feature_cols, model.booster_.feature_importance("gain"), strict=True))

rng = np.random.default_rng(SEED)
idx = rng.choice(len(x_val), size=min(SHAP_SAMPLE, len(x_val)), replace=False)
shap_values = shap.TreeExplainer(model).shap_values(x_val.iloc[idx])
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]
mean_abs = dict(zip(feature_cols, np.abs(shap_values).mean(axis=0), strict=True))

print(f"\n{'feature':38s} {'gain':>12s} {'mean|SHAP|':>10s}")
for name in sorted(feature_cols, key=lambda n: -mean_abs[n])[:20]:
    print(f"  {name:36s} {gain[name]:12.1f} {mean_abs[name]:10.4f}")

rank = sorted(feature_cols, key=lambda n: -mean_abs[n]).index("f3_ip_app_conversion_rate_24hr") + 1
print(f"\nf3_ip_app_conversion_rate_24hr SHAP rank: #{rank} (expected top 3)")
