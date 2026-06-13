"""Tests for the triage router (Task 6.1) — all four outcomes + edges."""

from __future__ import annotations

import numpy as np
import pytest

from sentry.audit.schema import Action
from sentry.triage.router import TriageThresholds, route_case

THR = TriageThresholds(block=0.8, review=0.4)


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_block_review_allow_outcomes() -> None:
    assert route_case(0.95, THR, qa_sample_rate=0.0, rng=_rng()) == Action.AUTO_BLOCK
    assert route_case(0.50, THR, qa_sample_rate=0.0, rng=_rng()) == Action.HUMAN_REVIEW
    assert route_case(0.10, THR, qa_sample_rate=0.0, rng=_rng()) == Action.ALLOW


def test_qa_sample_outcome() -> None:
    # Below review and qa_sample_rate=1 -> always diverted to QA.
    assert route_case(0.10, THR, qa_sample_rate=1.0, rng=_rng()) == Action.QA_SAMPLE


def test_thresholds_are_block_inclusive_review_inclusive() -> None:
    # Exactly at block -> block; exactly at review -> review.
    assert route_case(0.8, THR, qa_sample_rate=0.0, rng=_rng()) == Action.AUTO_BLOCK
    assert route_case(0.4, THR, qa_sample_rate=0.0, rng=_rng()) == Action.HUMAN_REVIEW


def test_qa_rate_zero_never_samples() -> None:
    rng = _rng()
    assert all(
        route_case(0.1, THR, qa_sample_rate=0.0, rng=rng) == Action.ALLOW for _ in range(100)
    )


def test_qa_rate_one_always_samples() -> None:
    rng = _rng()
    assert all(
        route_case(0.1, THR, qa_sample_rate=1.0, rng=rng) == Action.QA_SAMPLE for _ in range(100)
    )


def test_qa_sampling_fraction_is_approximately_the_rate() -> None:
    """Over many allowed cases, QA diversion rate ≈ qa_sample_rate (seeded)."""
    rng = np.random.default_rng(42)
    n = 20_000
    qa = sum(
        route_case(0.1, THR, qa_sample_rate=0.05, rng=rng) == Action.QA_SAMPLE for _ in range(n)
    )
    assert qa / n == pytest.approx(0.05, abs=0.01)


def test_qa_sampling_is_reproducible_with_seeded_rng() -> None:
    seq1 = [route_case(0.1, THR, 0.3, np.random.default_rng(7)) for _ in range(5)]
    seq2 = [route_case(0.1, THR, 0.3, np.random.default_rng(7)) for _ in range(5)]
    assert seq1 == seq2


def test_invalid_qa_rate_raises() -> None:
    with pytest.raises(ValueError, match="qa_sample_rate"):
        route_case(0.1, THR, qa_sample_rate=1.5, rng=_rng())


def test_review_above_block_raises() -> None:
    with pytest.raises(ValueError, match="review"):
        TriageThresholds(block=0.3, review=0.6)
