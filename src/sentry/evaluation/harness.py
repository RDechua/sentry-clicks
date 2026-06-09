"""Model-agnostic evaluation harness.

`evaluate(y_true, y_pred_proba, name)` returns a structured `EvaluationResult`
covering the primary metric (PR-AUC), the secondary metrics (ROC-AUC for
completeness, Brier score for calibration), and the diagnostic curves
(PR curve, calibration curve, confusion matrices at a fixed threshold set).

Built before any model exists, per CLAUDE.md §3.2 and build guide Task 1.8:
"Building the harness when you have a model creates a temptation to design
the harness to flatter your model." Rationale for the specific metric picks
lives in `docs/decisions.md` ("Evaluation metrics — choice and rationale").

The plot helpers use matplotlib's non-interactive Agg backend (forced via
`MPLBACKEND=Agg` set below) so the harness works in containers, CI, and
notebook/script callers on headless boxes — not just under pytest.
"""

from __future__ import annotations

import os

# Pin the matplotlib backend before pyplot is imported. Setting via env var
# (instead of `matplotlib.use("Agg")`) avoids the ruff E402 import-ordering
# tangle and works for any caller that imports this module before matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

# Fixed threshold set for the *diagnostic* confusion matrices — a quick
# sanity view at common operating points, NOT the operational sweep.
# Skewed low because the operating-point picker in Task 5.2 (cost-based)
# will mostly care about thresholds well below 0.5 on a ~0.2%-positive
# problem. Task 5.2 owns its own threshold grid; don't reuse this one.
_SANITY_THRESHOLDS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


class PRPoint(BaseModel):
    """One point on the precision-recall curve."""

    precision: float
    recall: float
    threshold: float | None  # None for the anchor (precision=1.0, recall=0.0)


class CalibrationPoint(BaseModel):
    """One point on the reliability diagram (predicted vs observed)."""

    predicted_mean: float
    fraction_positive: float


class ConfusionMatrix(BaseModel):
    """Confusion matrix at a specific binarization threshold."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0


class EvaluationResult(BaseModel):
    """Standardized metric bundle for a single (y_true, y_pred_proba) pair."""

    name: str
    n_samples: int
    n_positive: int
    pr_auc: float
    roc_auc: float  # NaN when only one class is present
    brier_score: float
    pr_curve: list[PRPoint]
    calibration_curve_points: list[CalibrationPoint]
    confusion_matrices: list[ConfusionMatrix]


def evaluate(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    name: str,
) -> EvaluationResult:
    """Compute the full metric bundle for one set of predictions.

    Parameters
    ----------
    y_true:
        1-D array of binary labels (0 or 1).
    y_pred_proba:
        1-D array of predicted probabilities in [0, 1] with the same length.
    name:
        Identifier for the result (e.g., `"lightgbm_v3"`, `"baseline_majority"`).

    Returns
    -------
    `EvaluationResult` with PR-AUC, ROC-AUC, Brier, the PR and calibration
    curves, and confusion matrices at the diagnostic threshold set.

    Raises
    ------
    ValueError:
        If shapes mismatch, or if `y_true` has no positive examples (PR-AUC
        is undefined and silently returning NaN is the wrong thing to do —
        a split with zero positives is almost always a bug worth catching).
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if y_true.shape != y_pred_proba.shape:
        raise ValueError(
            f"y_true and y_pred_proba shape mismatch: {y_true.shape} vs {y_pred_proba.shape}"
        )

    n_samples = len(y_true)
    n_positive = int(y_true.sum())

    if n_positive == 0:
        raise ValueError(
            "y_true has no positive examples; PR-AUC is undefined and "
            "several other metrics degenerate. Almost always indicates a "
            "split-construction bug — fix the caller rather than this function."
        )

    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    # ROC-AUC needs both classes present. Don't raise — return NaN so a
    # caller evaluating an all-positive holdout can still inspect Brier etc.
    roc_auc = float(roc_auc_score(y_true, y_pred_proba)) if n_positive < n_samples else float("nan")
    brier = float(brier_score_loss(y_true, y_pred_proba))

    pr_curve = _build_pr_curve(y_true, y_pred_proba)
    cal_points = _build_calibration_curve(y_true, y_pred_proba, n_samples)
    cms = _build_confusion_matrices(y_true, y_pred_proba)

    return EvaluationResult(
        name=name,
        n_samples=n_samples,
        n_positive=n_positive,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        brier_score=brier,
        pr_curve=pr_curve,
        calibration_curve_points=cal_points,
        confusion_matrices=cms,
    )


#: Cap on stored PR-curve points. `precision_recall_curve` returns up to one
#: point per distinct score — ~n_samples points — and materializing millions
#: of pydantic PRPoints costs gigabytes (measured: ~2.7 GB per evaluate() on a
#: 3.7M-row val set, the Task 1.8 deferred-efficiency debt coming due at full
#: scale). The stored curve is only for plotting/threshold inspection; the
#: pr_auc scalar is computed by average_precision_score over the FULL data and
#: is unaffected. A few thousand points render an identical-looking curve.
_MAX_PR_POINTS: Final[int] = 2000


def _build_pr_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> list[PRPoint]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # sklearn returns thresholds with one fewer element than precision/recall
    # because the last point is the (precision=1.0, recall=0.0) anchor.
    # Pad once with None so the strict zip below stays honest about lengths
    # AND the anchor naturally gets threshold=None.
    padded: list[float | None] = [float(t) for t in thresholds] + [None]

    # Downsample to a bounded number of evenly-spaced points (keeping the
    # first and last anchors) before building pydantic objects. Plot-only;
    # no metric depends on these stored points.
    n = len(precisions)
    if n > _MAX_PR_POINTS:
        idx = np.unique(np.linspace(0, n - 1, _MAX_PR_POINTS).astype(int))
        precisions, recalls = precisions[idx], recalls[idx]
        padded = [padded[i] for i in idx]

    return [
        PRPoint(precision=float(p), recall=float(r), threshold=t)
        for p, r, t in zip(precisions, recalls, padded, strict=True)
    ]


def _build_calibration_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_samples: int
) -> list[CalibrationPoint]:
    # Need at least 2 samples to form 2 bins. Below that, skip the curve.
    if n_samples < 2:
        return []
    # Quantile binning distributes samples evenly across bins; uniform binning
    # would put almost everything in the lowest bin on a 0.2%-positive problem.
    n_bins = min(10, max(2, n_samples // 10))
    frac_pos, mean_pred = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy="quantile"
    )
    return [
        CalibrationPoint(predicted_mean=float(m), fraction_positive=float(f))
        for m, f in zip(mean_pred, frac_pos, strict=True)
    ]


def _build_confusion_matrices(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> list[ConfusionMatrix]:
    out: list[ConfusionMatrix] = []
    for t in _SANITY_THRESHOLDS:
        y_pred_bin = (y_pred_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out.append(ConfusionMatrix(threshold=t, tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn)))
    return out


def compare(results: list[EvaluationResult]) -> pd.DataFrame:
    """Build a comparison table from multiple `EvaluationResult`s.

    Sorted by PR-AUC descending (the primary metric).
    """
    rows = [
        {
            "name": r.name,
            "n_samples": r.n_samples,
            "n_positive": r.n_positive,
            "pr_auc": r.pr_auc,
            "roc_auc": r.roc_auc,
            "brier_score": r.brier_score,
        }
        for r in results
    ]
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def plot_pr_curve(results: list[EvaluationResult], output_path: Path | str) -> None:
    """Plot precision-recall curves for one or more results to a PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        recalls = [p.recall for p in r.pr_curve]
        precisions = [p.precision for p in r.pr_curve]
        ax.plot(recalls, precisions, label=f"{r.name} (PR-AUC={r.pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=120)
    plt.close(fig)


def plot_calibration(results: list[EvaluationResult], output_path: Path | str) -> None:
    """Plot calibration (reliability) curves for one or more results to a PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    for r in results:
        if not r.calibration_curve_points:
            continue
        mean_pred = [p.predicted_mean for p in r.calibration_curve_points]
        frac_pos = [p.fraction_positive for p in r.calibration_curve_points]
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            label=f"{r.name} (Brier={r.brier_score:.3f})",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=120)
    plt.close(fig)
