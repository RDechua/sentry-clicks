"""Tasks 5.2/5.3: sweep thresholds on val, pick cost-optimal + capacity-capped.

Predicts calibrated P(is_attributed=1) on val, converts to fraud score
(1-p), sweeps the (T_block, T_review) grid against the cost model, and
reports the unconstrained optimum and the optimum under the SM6 reviewer
cap (review ≤ 0.5% of clicks). Saves the cost surface plot.

Run: docker compose run --rm sentry python experiments/threshold_sweep.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import duckdb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np

from sentry.models.calibration import Calibrator
from sentry.models.train import LABEL, MODEL_FEATURES
from sentry.triage.cost import CostModel, expected_cost, fraud_probability
from sentry.triage.thresholds import sweep_thresholds

VERSION = "v0.5.0"
MODEL_DIR = Path("artifacts/models/lgbm-v0.1.0")
REVIEW_CAP = 0.005  # SM6: human review ≤ 0.5% of total clicks
COSTS = CostModel()

cols = ", ".join([*(f"CAST({n} AS FLOAT) AS {n}" for n in MODEL_FEATURES), f"{LABEL}::TINYINT {LABEL}"])
with duckdb.connect() as conn:
    conn.execute("SET memory_limit = '1.5GB'")
    val = conn.execute(f"SELECT {cols} FROM read_parquet('artifacts/features/{VERSION}/val.parquet')").fetch_df()

booster = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
calibrator = Calibrator.load(MODEL_DIR / "calibrator.json")
y = val[LABEL].to_numpy()
p_cal = calibrator.predict(booster.predict(val[list(MODEL_FEATURES)]))
fraud = fraud_probability(p_cal)

res = sweep_thresholds(fraud, y, COSTS)
unconstrained = res.best()
capped = res.best_within_capacity(REVIEW_CAP)

# Full breakdown + a do-nothing baseline (allow everything) for ROI framing.
allow_all = expected_cost(fraud, y, threshold_block=2.0, threshold_review=2.0, costs=COSTS)
chosen = expected_cost(fraud, y, capped.threshold_block, capped.threshold_review, COSTS)

print(f"val rows: {len(y):,}, fraud (label=0): {int((y == 0).sum()):,}")
for name, c in [("unconstrained", unconstrained), ("capacity-capped (<=0.5%)", capped)]:
    print(f"{name}: T_block={c.threshold_block:.3f} T_review={c.threshold_review:.3f} "
          f"cost=${c.total_cost:,.0f} (${c.cost_per_click:.4f}/click) review={c.review_fraction:.4%}")
print(f"allow-everything baseline cost: ${allow_all.total:,.0f}")
binding = "reviewer capacity" if capped.total_cost > unconstrained.total_cost + 1e-6 else "cost (capacity slack)"
print(f"binding constraint: {binding}")

# Cost surface heatmap (log-scaled; infeasible cells masked).
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(np.log10(res.total_cost), origin="lower", extent=[0, 1, 0, 1], aspect="auto")
ax.scatter([capped.threshold_review], [capped.threshold_block], c="red", marker="*", s=200, label="capacity-capped optimum")
ax.set_xlabel("T_review (fraud score)")
ax.set_ylabel("T_block (fraud score)")
ax.set_title("Expected cost surface (log10 $) — val")
ax.legend(loc="lower right")
fig.colorbar(im, ax=ax, label="log10 total cost ($)")
fig.savefig(MODEL_DIR / "cost_surface.png", dpi=100, bbox_inches="tight")

lines = [
    "# Threshold sweep & reviewer capacity (Tasks 5.2/5.3)",
    "",
    f"Cost-optimal triage thresholds on the **{VERSION}** val split "
    f"({len(y):,} clicks, {int((y == 0).sum()):,} fraud), calibrated fraud scores, "
    f"cost model c_fp={COSTS.c_fp} c_fn={COSTS.c_fn} c_review={COSTS.c_review}. "
    "Selected on val, never test (§3.6).",
    "",
    "| policy | T_block | T_review | total cost | $/click | review load |",
    "|---|---|---|---|---|---|",
    f"| allow-everything (baseline) | – | – | ${allow_all.total:,.0f} | ${allow_all.total / len(y):.4f} | 0% |",
    f"| cost-optimal (unconstrained) | {unconstrained.threshold_block:.3f} | {unconstrained.threshold_review:.3f} | ${unconstrained.total_cost:,.0f} | ${unconstrained.cost_per_click:.4f} | {unconstrained.review_fraction:.4%} |",
    f"| **selected (review ≤ 0.5%)** | **{capped.threshold_block:.3f}** | **{capped.threshold_review:.3f}** | ${capped.total_cost:,.0f} | ${capped.cost_per_click:.4f} | {capped.review_fraction:.4%} |",
    "",
    f"**Binding constraint: {binding}.** "
    + (
        "The unconstrained optimum routes more than 0.5% of clicks to review, so reviewer "
        "capacity — not cost — sets the operating point; with free reviewers the system would "
        "review more and pay less."
        if binding == "reviewer capacity"
        else "The unconstrained optimum already satisfies the 0.5% review cap, so cost is the "
        "binding constraint and capacity has slack."
    ),
    "",
    "Selected policy vs allow-everything: "
    f"${allow_all.total - chosen.total:,.0f} saved on val "
    f"({(allow_all.total - chosen.total) / allow_all.total:.1%}). Surface plot: `cost_surface.png`.",
    "",
]
Path("reports/threshold_sweep.md").write_text("\n".join(lines))
print("\nwrote reports/threshold_sweep.md")
