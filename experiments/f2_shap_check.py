"""Task 2.4 AC check: F2 features must show non-trivial SHAP importance.

Trains a small LightGBM on F1+F2 over the train split, computes SHAP values
on a sample, prints mean |SHAP| per feature. This is a sanity check that F2
carries signal — NOT the real model (Week 4) nor the real ablation (Week 5).

Run: docker compose run --rm sentry python experiments/f2_shap_check.py
"""

from __future__ import annotations

import duckdb
import lightgbm as lgb
import numpy as np
import shap

from sentry.features.f1_per_click import F1_FEATURES
from sentry.features.f2_velocity import F2_FEATURES
from sentry.features.pipeline import FeaturePipeline

SEED = 42
SHAP_SAMPLE = 10_000

pipeline = FeaturePipeline([*F1_FEATURES, *F2_FEATURES])
with duckdb.connect("artifacts/sentry.duckdb", read_only=True) as conn:
    table = pipeline.compute(conn, source="clicks_train")

feature_cols = list(pipeline.feature_names)
x = table[feature_cols].copy()
# The two interaction features are strings; LightGBM wants category dtype.
for col in ("f1_ip_app_interaction", "f1_ip_device_interaction"):
    x[col] = x[col].astype("category")
y = table["is_attributed"]

model = lgb.LGBMClassifier(
    n_estimators=100,
    class_weight="balanced",  # CLAUDE.md 3.3: class weights, never resampling
    random_state=SEED,
    verbose=-1,
)
model.fit(x, y)

rng = np.random.default_rng(SEED)
idx = rng.choice(len(x), size=min(SHAP_SAMPLE, len(x)), replace=False)
sample = x.iloc[idx]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)
# Binary classification: some shap versions return a list/3-d array per
# class — take the positive-class slice if so.
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

mean_abs = np.abs(shap_values).mean(axis=0)
ranked = sorted(zip(feature_cols, mean_abs, strict=True), key=lambda t: -t[1])

print(f"\nmean |SHAP| per feature (train sample n={len(sample)}):")
for name, val in ranked:
    marker = " <-- F2" if name.startswith("f2_") else ""
    print(f"  {name:35s} {val:8.4f}{marker}")

f2_total = sum(v for n, v in ranked if n.startswith("f2_"))
all_total = sum(v for _, v in ranked)
print(f"\nF2 share of total |SHAP|: {f2_total / all_total:.1%}")
