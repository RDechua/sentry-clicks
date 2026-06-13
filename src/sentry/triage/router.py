"""Triage router — score to enforcement action (Task 6.1).

Implements the three-tier logic from PRD §8.1 on a click's FRAUD score
(higher = more fraud-likely, = `1 - calibrated P(is_attributed=1)`; see
`cost.fraud_probability`):

    score >= block            -> AUTO_BLOCK
    review <= score < block   -> HUMAN_REVIEW
    score <  review, sampled  -> QA_SAMPLE   (a random slice of would-be-allows)
    otherwise                 -> ALLOW

QA sampling pulls a small fraction of otherwise-allowed clicks for
quality-assurance review — the only way to estimate the false-negative rate
on traffic the model is confident about. The draw is taken from a passed-in
`numpy` Generator so batch routing is reproducible (§3.8); rates of 0 and 1
are deterministic and don't depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentry.audit.schema import Action


@dataclass(frozen=True)
class TriageThresholds:
    """Fraud-score cutoffs for the router. `review <= block` is required.

    Built from a cost sweep's selected policy (`ThresholdChoice`) but kept
    as its own small type so the router doesn't depend on the sweep machinery.
    """

    block: float
    review: float

    def __post_init__(self) -> None:
        if self.review > self.block:
            raise ValueError(f"review ({self.review}) must be <= block ({self.block})")


def route_case(
    score: float,
    thresholds: TriageThresholds,
    qa_sample_rate: float,
    rng: np.random.Generator,
) -> Action:
    """Map one fraud score to an enforcement Action (PRD §8.1).

    Parameters
    ----------
    score:
        Fraud score in [0, 1]; higher = more fraud-likely.
    thresholds:
        The block/review cutoffs.
    qa_sample_rate:
        Fraction of would-be-allowed clicks to divert to QA review, in [0, 1].
    rng:
        Source for the QA sampling draw (seed it for reproducible batches).
    """
    if not 0.0 <= qa_sample_rate <= 1.0:
        raise ValueError(f"qa_sample_rate must be in [0, 1], got {qa_sample_rate}")

    if score >= thresholds.block:
        return Action.AUTO_BLOCK
    if score >= thresholds.review:
        return Action.HUMAN_REVIEW
    if rng.random() < qa_sample_rate:
        return Action.QA_SAMPLE
    return Action.ALLOW
