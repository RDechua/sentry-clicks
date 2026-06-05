"""Tests for `sentry.evaluation.harness` — the model-agnostic metric bundle.

Covers the edge cases the build guide Task 1.8 calls out: all positive,
all negative, perfect, random, dummy 0.5 baseline. Also: the PR-AUC vs
ROC-AUC demonstration on imbalanced data, JSON round-trip, the
`compare()` shape, and that the plot helpers actually produce files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentry.evaluation.harness import (
    EvaluationResult,
    compare,
    evaluate,
    plot_calibration,
    plot_pr_curve,
)


def _rng() -> np.random.Generator:
    """Pinned RNG per CLAUDE.md §3.8 — same input, same output, every time."""
    return np.random.default_rng(42)


def test_perfect_predictions_pr_auc_is_one() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0.05, 0.95, 0.10, 0.90, 0.15, 0.85])
    result = evaluate(y_true, y_pred, name="perfect")
    assert result.pr_auc == pytest.approx(1.0)
    assert result.roc_auc == pytest.approx(1.0)


def test_no_positives_raises_value_error() -> None:
    """PR-AUC is undefined without positives — fail loud, not silently NaN.

    This is the canonical "test split with zero positives" bug; silent NaN
    hides it for weeks.
    """
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="no positive"):
        evaluate(y_true, y_pred, name="no_positives")


def test_all_positives_pr_auc_is_one() -> None:
    """No negatives means precision is always 1.0 → PR-AUC is exactly 1.0."""
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.9, 0.8, 0.7, 0.6])
    result = evaluate(y_true, y_pred, name="all_positives")
    assert result.pr_auc == pytest.approx(1.0)
    # ROC-AUC is undefined with one class — should be NaN, not raised.
    assert np.isnan(result.roc_auc)


def test_dummy_0_5_baseline_brier_is_quarter() -> None:
    """Brier of constant 0.5 against any binary label is exactly 0.25."""
    rng = _rng()
    y_true = rng.integers(0, 2, size=100)
    # Ensure at least one positive (avoid no-positives ValueError).
    if y_true.sum() == 0:
        y_true[0] = 1
    y_pred = np.full_like(y_true, 0.5, dtype=float)
    result = evaluate(y_true, y_pred, name="dummy_0_5")
    assert result.brier_score == pytest.approx(0.25)


def test_pr_auc_below_roc_auc_on_imbalanced() -> None:
    """The 'why PR-AUC' demonstration.

    On a heavily imbalanced dataset, ROC-AUC can look respectable while
    PR-AUC reveals the model is actually poor at picking out the rare class.
    """
    rng = _rng()
    n = 10_000
    # ~0.5% positive rate
    y_true = (rng.uniform(size=n) < 0.005).astype(int)
    # Predictor: positives get a small bump above the noise.
    y_pred = rng.beta(2, 5, size=n) + y_true * 0.15
    y_pred = np.clip(y_pred, 0.0, 1.0)

    result = evaluate(y_true, y_pred, name="imbalanced")
    assert result.pr_auc < result.roc_auc
    # The predictor is above chance, so ROC-AUC is informative-looking...
    assert result.roc_auc > 0.6
    # ...but PR-AUC reveals it's actually quite poor on the rare class.
    assert result.pr_auc < 0.30


def test_evaluation_result_round_trips_through_json() -> None:
    """Pydantic serialization preserves all fields including the curve lists."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75])
    result = evaluate(y_true, y_pred, name="json_test")

    serialized = result.model_dump_json()
    restored = EvaluationResult.model_validate_json(serialized)

    assert restored.name == "json_test"
    assert restored.pr_auc == pytest.approx(result.pr_auc)
    assert restored.brier_score == pytest.approx(result.brier_score)
    assert len(restored.pr_curve) == len(result.pr_curve)
    assert len(restored.confusion_matrices) == len(result.confusion_matrices)


def test_compare_returns_dataframe_sorted_by_pr_auc() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    r_good = evaluate(y_true, np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75]), name="good")
    r_dummy = evaluate(y_true, np.full(8, 0.5), name="dummy")

    df = compare([r_dummy, r_good])  # pass in non-sorted order

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df.columns) >= {"name", "pr_auc", "roc_auc", "brier_score"}
    # Sorted descending by pr_auc — `good` should be first.
    assert df.iloc[0]["name"] == "good"


def test_plot_pr_curve_writes_png(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 0, 1])
    r = evaluate(y_true, np.array([0.1, 0.9, 0.2, 0.8]), name="x")
    out = tmp_path / "pr.png"
    plot_pr_curve([r], out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_calibration_writes_png(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    r = evaluate(
        y_true,
        np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75]),
        name="x",
    )
    out = tmp_path / "cal.png"
    plot_calibration([r], out)
    assert out.exists() and out.stat().st_size > 0


def test_shape_mismatch_raises() -> None:
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0.1, 0.9])
    with pytest.raises(ValueError, match="shape"):
        evaluate(y_true, y_pred, name="mismatch")
