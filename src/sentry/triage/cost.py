"""Cost model for triage decisions (Task 5.1).

Threshold selection is a business decision, not a metric one — so it is made
against an explicit dollar cost, not max-F1 or a fixed percentile (§3.6).
This module defines that cost.

THE DIRECTION (the label-inversion gotcha, in the triage layer): the model
predicts `p = P(is_attributed = 1) = P(legitimate)`. Fraud is the absence of
attribution, so **fraud probability = 1 - p** (`fraud_probability` below).
Triage acts on the FRAUD score: a HIGH fraud score (low p) is blocked, a LOW
fraud score (high p) is allowed. Get this backwards and the whole cost model
inverts — hence the named helper and this paragraph.

The three-tier decision on a click's fraud score `f`:
    f >= threshold_block            -> AUTO_BLOCK
    threshold_review <= f < block   -> HUMAN_REVIEW
    f <  threshold_review           -> ALLOW

Cost accounting (the documented simplification): a correctly-blocked fraud
and a correctly-allowed legit click cost nothing; a human review costs
`c_review` and is assumed to resolve correctly (so reviewed cases incur only
the review cost). The two error costs:
    - block a legitimate click (is_attributed=1)  -> c_fp  (false positive)
    - allow a fraudulent click (is_attributed=0)  -> c_fn  (false negative)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Per-event costs in dollars. ILLUSTRATIVE values — the METHODOLOGY is
    the deliverable, not these numbers (build guide §5.1).

    c_fp/c_fn are treated as equal here (both ≈ one mid-range CPC) as a
    documented simplification: in reality a blocked legit click and an
    allowed fraud click carry different direct + reputational costs, but
    modeling that asymmetry needs revenue data this project doesn't have.
    c_review is loaded reviewer time: ~90s at ~$20/hr ≈ $0.50.
    """

    c_fp: float = 0.30  # block a legitimate click (lost CPC + advertiser annoyance)
    c_fn: float = 0.30  # allow a fraudulent click (advertiser charged for nothing)
    c_review: float = 0.50  # one human review (opportunity cost of reviewer time)


@dataclass(frozen=True)
class CostBreakdown:
    """The cost of one (threshold_block, threshold_review) policy on a set."""

    total: float
    n_blocked: int
    n_reviewed: int
    n_allowed: int
    false_positives: int  # blocked but legitimate
    false_negatives: int  # allowed but fraudulent
    fp_cost: float
    fn_cost: float
    review_cost: float

    @property
    def cost_per_click(self) -> float:
        n = self.n_blocked + self.n_reviewed + self.n_allowed
        return self.total / n if n else 0.0


def fraud_probability(p_attributed: np.ndarray) -> np.ndarray:
    """Convert model P(is_attributed=1) to fraud probability (1 - p).

    The one place the label inversion lives. Callers pass calibrated
    P(legitimate); triage and cost reason about fraud score.
    """
    return 1.0 - np.asarray(p_attributed, dtype="float64")


def expected_cost(
    fraud_scores: np.ndarray,
    labels: np.ndarray,
    threshold_block: float,
    threshold_review: float,
    costs: CostModel,
) -> CostBreakdown:
    """Total expected cost of a triage policy on a scored, labelled set.

    Parameters
    ----------
    fraud_scores:
        Fraud probability per click (= `fraud_probability(calibrated_p)`),
        higher = more fraud-likely.
    labels:
        `is_attributed` (1 = legitimate, 0 = fraud).
    threshold_block, threshold_review:
        Fraud-score cutoffs; must satisfy `threshold_review <= threshold_block`.
    costs:
        The per-event `CostModel`.
    """
    if threshold_review > threshold_block:
        raise ValueError(
            f"threshold_review ({threshold_review}) must be <= threshold_block ({threshold_block})"
        )
    f = np.asarray(fraud_scores, dtype="float64")
    y = np.asarray(labels)
    if f.shape != y.shape:
        raise ValueError(f"fraud_scores and labels shape mismatch: {f.shape} vs {y.shape}")

    is_legit = y == 1
    is_fraud = y == 0
    blocked = f >= threshold_block
    reviewed = (f >= threshold_review) & (f < threshold_block)
    allowed = f < threshold_review

    false_positives = int(np.sum(blocked & is_legit))  # blocked a legit click
    false_negatives = int(np.sum(allowed & is_fraud))  # allowed a fraud click
    n_reviewed = int(np.sum(reviewed))

    fp_cost = false_positives * costs.c_fp
    fn_cost = false_negatives * costs.c_fn
    review_cost = n_reviewed * costs.c_review

    return CostBreakdown(
        total=fp_cost + fn_cost + review_cost,
        n_blocked=int(np.sum(blocked)),
        n_reviewed=n_reviewed,
        n_allowed=int(np.sum(allowed)),
        false_positives=false_positives,
        false_negatives=false_negatives,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        review_cost=review_cost,
    )
