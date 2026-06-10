"""Tests for the threshold sweep + capacity constraint (Tasks 5.2/5.3)."""

from __future__ import annotations

import numpy as np
import pytest

from sentry.triage.cost import CostModel, expected_cost
from sentry.triage.thresholds import sweep_thresholds

COSTS = CostModel(c_fp=0.30, c_fn=0.30, c_review=0.50)


def _data(n: int = 4000, seed: int = 0):
    """Fraud scores where fraud (label 0) skews high and legit (1) skews low,
    so a sensible block/review/allow split exists."""
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.2).astype(int)  # 20% legit
    f = np.where(y == 1, rng.uniform(0.0, 0.5, n), rng.uniform(0.5, 1.0, n))
    return f, y


def test_sweep_matches_expected_cost_cell_by_cell() -> None:
    """The fast searchsorted surface must equal looping expected_cost."""
    f, y = _data()
    grid = np.linspace(0.0, 1.0, 11)
    res = sweep_thresholds(f, y, COSTS, grid=grid)
    for i, tb in enumerate(grid):
        for j, tr in enumerate(grid):
            if tr > tb:
                assert np.isnan(res.total_cost[i, j])
                continue
            ref = expected_cost(f, y, threshold_block=tb, threshold_review=tr, costs=COSTS)
            assert res.total_cost[i, j] == pytest.approx(ref.total)
            assert res.review_fraction[i, j] == pytest.approx(ref.n_reviewed / len(y))


def test_best_is_the_global_grid_minimum() -> None:
    f, y = _data()
    res = sweep_thresholds(f, y, COSTS)
    best = res.best()
    assert best.total_cost == pytest.approx(np.nanmin(res.total_cost))
    assert best.threshold_review <= best.threshold_block


def test_capacity_constraint_caps_review_volume() -> None:
    f, y = _data()
    res = sweep_thresholds(f, y, COSTS)
    capped = res.best_within_capacity(max_review_fraction=0.05)
    assert capped.review_fraction <= 0.05 + 1e-9
    # Constrained optimum can't be cheaper than the unconstrained one.
    assert capped.total_cost >= res.best().total_cost - 1e-9


def test_capacity_too_tight_raises() -> None:
    f, y = _data()
    res = sweep_thresholds(f, y, COSTS)
    # Even allow-everything has 0 reviews, so 0.0 is always feasible; an
    # impossible NEGATIVE cap has no feasible cell.
    with pytest.raises(ValueError, match="review"):
        res.best_within_capacity(max_review_fraction=-0.01)


def test_zero_review_capacity_picks_a_two_way_policy() -> None:
    """With no review budget, the optimum has T_block == T_review (no band)."""
    f, y = _data()
    res = sweep_thresholds(f, y, COSTS)
    capped = res.best_within_capacity(max_review_fraction=0.0)
    assert capped.review_fraction == pytest.approx(0.0)
