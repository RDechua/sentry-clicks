"""Task 4.7: final test-set evaluation — ONE SHOT.

The test split has not been touched in any prior task (§3.1). This script
applies the frozen model + calibrator to test exactly once and records the
official numbers. It is NOT to be re-run to chase a better number — that is
attention leakage. If a genuine CODE bug is found, fix it and re-run; never
re-run to tune.

Best model: lgbm-v0.1.0 (the default beat the subsample-tuned candidate,
Task 4.4) + the isotonic calibrator fit on val (Task 4.5).

Requires v0.5.0/test.parquet (materialized just before this runs — the test
split's features were deliberately not built earlier). Run:
    docker compose run --rm sentry python experiments/final_evaluation.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import lightgbm as lgb

from sentry.evaluation.harness import evaluate, plot_calibration, plot_pr_curve
from sentry.models.calibration import Calibrator
from sentry.models.train import LABEL, MODEL_FEATURES

VERSION = "v0.5.0"
MODEL_DIR = Path("artifacts/models/lgbm-v0.1.0")
TEST = f"artifacts/features/{VERSION}/test.parquet"
# Val calibrated Brier (Task 4.5 holdout) for the val/test gap discussion.
VAL_CALIBRATED_BRIER = 0.00130
VAL_PR_AUC = 0.5618

cols = ", ".join([*(f"CAST({n} AS FLOAT) AS {n}" for n in MODEL_FEATURES), f"{LABEL}::TINYINT {LABEL}"])
with duckdb.connect() as conn:
    conn.execute("SET memory_limit = '1.5GB'")
    test = conn.execute(f"SELECT {cols} FROM read_parquet('{TEST}')").fetch_df()

booster = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
calibrator = Calibrator.load(MODEL_DIR / "calibrator.json")
y_test = test[LABEL].to_numpy()
raw = booster.predict(test[list(MODEL_FEATURES)])
calibrated = calibrator.predict(raw)

raw_result = evaluate(y_test, raw, name="test_raw")
cal_result = evaluate(y_test, calibrated, name="test_calibrated")
plot_pr_curve([cal_result], MODEL_DIR / "test_pr_curve.png")
plot_calibration([raw_result, cal_result], MODEL_DIR / "test_calibration.png")

print("=" * 60)
print("FINAL TEST EVALUATION — ONE SHOT (do not re-run to tune)")
print("=" * 60)
print(f"test rows: {len(y_test):,}  positives: {int(y_test.sum()):,} ({y_test.mean():.4%})")
print(f"PR-AUC (raw):        {raw_result.pr_auc:.4f}")
print(f"PR-AUC (calibrated): {cal_result.pr_auc:.4f}")
print(f"ROC-AUC:             {cal_result.roc_auc:.4f}")
print(f"Brier (raw):         {raw_result.brier_score:.5f}")
print(f"Brier (calibrated):  {cal_result.brier_score:.5f}")


def _cm_rows() -> list[str]:
    out = []
    for cm in cal_result.confusion_matrices:
        out.append(
            f"| {cm.threshold:.2f} | {cm.tp:,} | {cm.fp:,} | "
            f"{cm.fn:,} | {cm.tn:,} | {cm.precision:.4f} | {cm.recall:.4f} |"
        )
    return out


lines = [
    "# Final test evaluation (Task 4.7) — ONE SHOT",
    "",
    "Official held-out numbers for **lgbm-v0.1.0 + isotonic calibrator**, "
    f"applied once to the **{VERSION}** test split "
    f"({len(y_test):,} rows, {int(y_test.sum()):,} positives, {y_test.mean():.4%} base rate). "
    "The test set was untouched until this run.",
    "",
    "| metric | value |",
    "|---|---|",
    f"| **PR-AUC** (primary, §3.2) | **{cal_result.pr_auc:.4f}** |",
    f"| ROC-AUC | {cal_result.roc_auc:.4f} |",
    f"| Brier (calibrated) | {cal_result.brier_score:.5f} |",
    f"| Brier (raw, uncalibrated) | {raw_result.brier_score:.5f} |",
    f"| PR-AUC (raw, pre-calibration) | {raw_result.pr_auc:.4f} |",
    "",
    f"Val reference: PR-AUC {VAL_PR_AUC:.4f}, calibrated Brier {VAL_CALIBRATED_BRIER:.5f}. "
    f"Val→test PR-AUC gap = {VAL_PR_AUC - cal_result.pr_auc:+.4f}.",
    "",
    "## Confusion matrices at operating thresholds (calibrated)",
    "",
    "| threshold | TP | FP | FN | TN | precision | recall |",
    "|---|---|---|---|---|---|---|",
    *_cm_rows(),
    "",
    "Thresholds here are the harness's fixed diagnostic set; the cost-optimal "
    "operating threshold is selected in Week 5 against the reviewer-capacity "
    "cost model. Plots: `test_pr_curve.png`, `test_calibration.png`.",
    "",
]
Path("reports/final_evaluation.md").write_text("\n".join(lines))
print("\nwrote reports/final_evaluation.md")
