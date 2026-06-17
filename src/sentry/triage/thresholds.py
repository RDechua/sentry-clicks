"""Cost-based threshold selection (Tasks 5.2 + 5.3).

Sweeps a grid of (T_block, T_review) fraud-score cutoffs, scoring each by the
Task 5.1 cost model, and selects the minimum-cost policy — optionally subject
to a reviewer-capacity cap (5.3). Thresholds are selected on VAL, never test:
threshold choice is a model decision, and optimizing it on test would leak
test information into the deployed system (§3.6, build-guide stop-and-think).

Efficiency: a naive double loop would re-scan all val rows per grid cell. We
sort fraud scores once per label and answer every "how many scores ≥ t" with
`searchsorted`, so the whole grid costs O(n log n + grid). Identical results
to looping `expected_cost`, asserted by test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from sentry.triage.cost import CostModel

#: Default candidate cutoffs swept on each axis (fraud-score space [0, 1]).
DEFAULT_GRID: Final[np.ndarray] = np.linspace(0.0, 1.0, 51)


@dataclass(frozen=True)
class ThresholdChoice:
    """One selected policy: the cutoffs, its cost, and its review load.

    Block/allow volumes aren't carried here — the caller recomputes the full
    `CostBreakdown` via `expected_cost` for the single chosen policy when it
    wants every count; the sweep only needs cost and review fraction to rank.
    """

    threshold_block: float
    threshold_review: float
    total_cost: float
    cost_per_click: float
    review_fraction: float
    n_reviewed: int


@dataclass(frozen=True)
class SweepResult:
    """The full cost surface over the (T_block, T_review) grid."""

    grid: np.ndarray
    # Square matrices indexed [i_block, j_review]; cells with review > block
    # are np.nan (infeasible: review cutoff above block cutoff).
    total_cost: np.ndarray
    review_fraction: np.ndarray
    n_total: int

    def best(self) -> ThresholdChoice:
        """Unconstrained minimum-cost policy."""
        return self._argmin(np.ones_like(self.total_cost, dtype=bool))

    def best_within_capacity(self, max_review_fraction: float) -> ThresholdChoice:
        """Minimum-cost policy whose review volume ≤ `max_review_fraction`."""
        feasible = self.review_fraction <= max_review_fraction
        if not np.any(feasible & ~np.isnan(self.total_cost)):
            raise ValueError(f"no grid policy keeps review ≤ {max_review_fraction:.4%}")
        return self._argmin(feasible)

    def _argmin(self, mask: np.ndarray) -> ThresholdChoice:
        costs = np.where(mask & ~np.isnan(self.total_cost), self.total_cost, np.inf)
        i, j = np.unravel_index(int(np.argmin(costs)), costs.shape)
        return ThresholdChoice(
            threshold_block=float(self.grid[i]),
            threshold_review=float(self.grid[j]),
            total_cost=float(self.total_cost[i, j]),
            cost_per_click=float(self.total_cost[i, j]) / self.n_total,
            review_fraction=float(self.review_fraction[i, j]),
            n_reviewed=round(float(self.review_fraction[i, j]) * self.n_total),
        )


def sweep_thresholds(
    fraud_scores: np.ndarray,
    labels: np.ndarray,
    costs: CostModel,
    grid: np.ndarray = DEFAULT_GRID,
) -> SweepResult:
    """Compute the cost surface over all (T_block ≥ T_review) grid pairs.

    `fraud_scores` = fraud probability (1 - calibrated p); `labels` =
    is_attributed (1 legit, 0 fraud).
    """
    f = np.asarray(fraud_scores, dtype="float64")
    y = np.asarray(labels)
    n = len(f)
    legit = np.sort(f[y == 1])
    fraud = np.sort(f[y == 0])

    def ge(sorted_scores: np.ndarray, t: float) -> int:
        """Count of ``sorted_scores`` >= ``t``, via searchsorted on the sorted array."""
        return len(sorted_scores) - int(np.searchsorted(sorted_scores, t, side="left"))

    g = grid
    total = np.full((len(g), len(g)), np.nan)
    rev_frac = np.full((len(g), len(g)), np.nan)

    # Precompute per-threshold counts.
    legit_ge = np.array([ge(legit, t) for t in g])  # legit blocked if T_block=t
    fraud_lt = np.array([len(fraud) - ge(fraud, t) for t in g])  # fraud allowed if T_review=t
    all_ge = np.array([ge(legit, t) + ge(fraud, t) for t in g])  # all with f >= t

    for i, _t_block in enumerate(g):
        for j, _t_review in enumerate(g):
            if g[j] > g[i]:
                continue  # infeasible: review cutoff above block cutoff
            fp = legit_ge[i]  # blocked legit
            fn = fraud_lt[j]  # allowed fraud
            n_reviewed = all_ge[j] - all_ge[i]  # in [T_review, T_block)
            total[i, j] = fp * costs.c_fp + fn * costs.c_fn + n_reviewed * costs.c_review
            rev_frac[i, j] = n_reviewed / n if n else 0.0

    return SweepResult(grid=g, total_cost=total, review_fraction=rev_frac, n_total=n)
