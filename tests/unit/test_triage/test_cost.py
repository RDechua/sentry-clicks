"""Tests for the triage cost model (Task 5.1).

Hand-crafted scenarios with costs computed by hand. The direction matters
most: fraud score is HIGH for fraud (is_attributed=0), so blocking targets
high scores, and a blocked legit click is the false positive.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentry.triage.cost import CostModel, expected_cost, fraud_probability

COSTS = CostModel(c_fp=0.30, c_fn=0.30, c_review=0.50)


def test_fraud_probability_inverts_attribution() -> None:
    p = np.array([0.0, 0.25, 1.0])
    assert np.allclose(fraud_probability(p), [1.0, 0.75, 0.0])


def test_hand_computed_three_tier_costs() -> None:
    # fraud scores and labels (1=legit, 0=fraud):
    #   0.95 fraud  -> block   (correct, free)
    #   0.90 legit  -> block   -> FALSE POSITIVE (c_fp)
    #   0.50 fraud  -> review  (c_review)
    #   0.40 legit  -> review  (c_review)
    #   0.10 fraud  -> allow   -> FALSE NEGATIVE (c_fn)
    #   0.05 legit  -> allow   (correct, free)
    f = np.array([0.95, 0.90, 0.50, 0.40, 0.10, 0.05])
    y = np.array([0, 1, 0, 1, 0, 1])
    r = expected_cost(f, y, threshold_block=0.8, threshold_review=0.2, costs=COSTS)

    assert (r.n_blocked, r.n_reviewed, r.n_allowed) == (2, 2, 2)
    assert r.false_positives == 1  # the 0.90 legit click, blocked
    assert r.false_negatives == 1  # the 0.10 fraud click, allowed
    assert r.fp_cost == pytest.approx(0.30)
    assert r.fn_cost == pytest.approx(0.30)
    assert r.review_cost == pytest.approx(1.00)  # 2 reviews x 0.50
    assert r.total == pytest.approx(1.60)
    assert r.cost_per_click == pytest.approx(1.60 / 6)


def test_block_everything_costs_every_legit_as_fp() -> None:
    """threshold_block=0 blocks all; cost = c_fp x (number of legit clicks)."""
    f = np.array([0.9, 0.1, 0.5, 0.2])
    y = np.array([1, 1, 0, 0])  # 2 legit, 2 fraud
    r = expected_cost(f, y, threshold_block=0.0, threshold_review=0.0, costs=COSTS)
    assert r.n_blocked == 4
    assert r.false_positives == 2
    assert r.false_negatives == 0
    assert r.total == pytest.approx(2 * 0.30)


def test_allow_everything_costs_every_fraud_as_fn() -> None:
    """threshold_review > all scores allows all; cost = c_fn x (n fraud)."""
    f = np.array([0.9, 0.1, 0.5, 0.2])
    y = np.array([1, 1, 0, 0])
    r = expected_cost(f, y, threshold_block=1.1, threshold_review=1.1, costs=COSTS)
    assert r.n_allowed == 4
    assert r.false_negatives == 2
    assert r.false_positives == 0
    assert r.total == pytest.approx(2 * 0.30)


def test_boundaries_are_block_inclusive_review_inclusive() -> None:
    """A score exactly at threshold_block blocks; exactly at review reviews."""
    f = np.array([0.8, 0.2])
    y = np.array([0, 0])
    r = expected_cost(f, y, threshold_block=0.8, threshold_review=0.2, costs=COSTS)
    assert r.n_blocked == 1  # the 0.80 -> block (>=)
    assert r.n_reviewed == 1  # the 0.20 -> review (>=)
    assert r.n_allowed == 0


def test_review_threshold_above_block_raises() -> None:
    with pytest.raises(ValueError, match="threshold_review"):
        expected_cost(
            np.array([0.5]), np.array([0]), threshold_block=0.3, threshold_review=0.6, costs=COSTS
        )


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        expected_cost(
            np.array([0.5, 0.6]),
            np.array([0]),
            threshold_block=0.5,
            threshold_review=0.2,
            costs=COSTS,
        )
