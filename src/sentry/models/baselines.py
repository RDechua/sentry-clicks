"""Baselines B1/B2/B3 (Task 4.2).

Baselines exist BEFORE the real model on purpose (build guide §4.2): they
set the floor and the "is that good?" reference for the LightGBM number,
and building them first removes the temptation to make them weak.

The positive class is `is_attributed = 1` (a click that led to a
download) — the rare ~0.2% class the harness scores PR-AUC on. Each
function returns a 1-D array of P(is_attributed=1) for the val rows, fed
through the same `evaluation.harness.evaluate` as every real model.

- **B1 random** — uniform scores. PR-AUC floor ≈ the base rate.
- **B2 pair conversion rate** — the (ip, app) pair's historical conversion
  rate IS an estimate of P(is_attributed=1); use it directly as the score.
  One feature, no model. Expected to be surprisingly strong — the point.
- **B3 logistic regression on F1+F2** — no F3. Isolates whether the F3
  aggregates earn their place over velocity/per-click features alone.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sentry.features.f1_per_click import F1_FEATURES
from sentry.features.f2_velocity import F2_FEATURES

LABEL: Final[str] = "is_attributed"
SEED: Final[int] = 42

#: B2's feature. The 24h pair rate is the canonical "historical conversion
#: rate of the (ip, app) pair"; a no-history pair is NULL and gets the train
#: base rate (the only honest prior for an unseen pair).
B2_FEATURE: Final[str] = "f3_ip_app_conversion_rate_24hr"

#: String interaction features LogisticRegression can't consume (LightGBM
#: will, in Week 4). Excluded from the linear baseline's matrix.
_STRING_FEATURES: Final[frozenset[str]] = frozenset(
    {"f1_ip_app_interaction", "f1_ip_device_interaction"}
)

#: F1+F2 numeric feature names, derived from the registries so this can't
#: drift from the actual feature set.
F1F2_NUMERIC_FEATURES: Final[tuple[str, ...]] = tuple(
    f.name for f in (*F1_FEATURES, *F2_FEATURES) if f.name not in _STRING_FEATURES
)

#: NULL fill for the linear baseline. A documented sentinel (CLAUDE.md §3.4):
#: logreg can't route NULLs, and -1 sits below every genuine feature value
#: (counts ≥ 0, rates ∈ [0, 1]), so the scaler maps it to a consistent
#: out-of-band point rather than distorting a real mean.
_SENTINEL: Final[float] = -1.0


def score_random(n_rows: int, seed: int = SEED) -> np.ndarray:
    """B1: uniform random scores in [0, 1]. Seeded for reproducibility (§3.8)."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=n_rows)


def score_pair_conversion_rate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature: str = B2_FEATURE,
) -> np.ndarray:
    """B2: the pair conversion-rate feature used directly as the score.

    No-history pairs (NULL) take the train base rate — the population prior
    is the best guess for an unseen pair, and it's a constant computed from
    train only, so no val information leaks.
    """
    prior = float(train[LABEL].mean())
    scores: np.ndarray = val[feature].fillna(prior).to_numpy(dtype="float64")
    return scores


def fit_logreg_scores(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: tuple[str, ...],
    seed: int = SEED,
) -> np.ndarray:
    """Scaled, class-balanced logistic regression; returns val P(positive).

    Shared by B3 (features=F1F2) and reusable for any feature subset. Class
    weights, not resampling (CLAUDE.md §3.3). The scaler keeps the L2
    penalty comparable across unitless columns; the sentinel fills NULLs.
    """
    x_train = train[list(features)].astype("float64").fillna(_SENTINEL).to_numpy()
    x_val = val[list(features)].astype("float64").fillna(_SENTINEL).to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=seed, max_iter=1000, class_weight="balanced"),
    )
    model.fit(x_train, train[LABEL].to_numpy())
    proba: np.ndarray = model.predict_proba(x_val)[:, 1]
    return proba


def score_logreg_f1f2(train: pd.DataFrame, val: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """B3: logistic regression on F1+F2 only (no F3 aggregates)."""
    return fit_logreg_scores(train, val, F1F2_NUMERIC_FEATURES, seed=seed)
