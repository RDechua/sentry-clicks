"""Tests for the B1/B2/B3 baselines (Task 4.2).

Baseline LOGIC only — small synthetic feature frames with exactly the
columns each baseline reads. Feature correctness is covered by the
feature-module tests; these pin: determinism (§3.8), the NULL-prior
handling, score ranges, and that B3 uses F1+F2 only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentry.models.baselines import (
    B2_FEATURE,
    F1F2_NUMERIC_FEATURES,
    score_logreg_f1f2,
    score_pair_conversion_rate,
    score_random,
)


def _feature_frame(n: int = 200) -> pd.DataFrame:
    """A frame carrying every F1+F2 numeric column, the B2 feature, and a
    learnable label (positives concentrated where the pair rate is high)."""
    rng = np.random.default_rng(0)
    pair_rate = rng.uniform(0, 1, n)
    df = pd.DataFrame({B2_FEATURE: pair_rate})
    for name in F1F2_NUMERIC_FEATURES:
        df[name] = rng.normal(size=n)
    # Label correlates with the pair rate so the linear model has signal.
    df["is_attributed"] = (rng.uniform(0, 1, n) < pair_rate * 0.4).astype(int)
    return df


# --- B1 random ----------------------------------------------------------


def test_random_is_deterministic_and_bounded() -> None:
    a = score_random(1000, seed=42)
    b = score_random(1000, seed=42)
    assert a.shape == (1000,)
    assert np.array_equal(a, b), "same seed must reproduce (§3.8)"
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_random_differs_across_seeds() -> None:
    assert not np.array_equal(score_random(1000, seed=1), score_random(1000, seed=2))


# --- B2 pair conversion rate -------------------------------------------


def test_pair_rate_used_directly_as_score() -> None:
    train = pd.DataFrame({B2_FEATURE: [0.5, 0.5], "is_attributed": [1, 0]})
    val = pd.DataFrame({B2_FEATURE: [0.9, 0.1], "is_attributed": [1, 0]})
    scores = score_pair_conversion_rate(train, val)
    assert list(scores) == [0.9, 0.1]


def test_pair_rate_null_takes_train_base_rate() -> None:
    """A no-history pair (NULL) must score the train base rate, not 0 or NaN."""
    train = pd.DataFrame({B2_FEATURE: [1.0, 0.0, 0.0, 0.0], "is_attributed": [1, 0, 0, 0]})
    val = pd.DataFrame({B2_FEATURE: [np.nan, 0.7], "is_attributed": [0, 1]})
    scores = score_pair_conversion_rate(train, val)
    assert scores[0] == pytest.approx(0.25)  # train base rate = 1/4
    assert scores[1] == pytest.approx(0.7)
    assert not np.isnan(scores).any()


# --- B3 logistic regression on F1+F2 -----------------------------------


def test_f1f2_features_exclude_strings_and_f3() -> None:
    assert "f1_ip_app_interaction" not in F1F2_NUMERIC_FEATURES
    assert "f1_ip_device_interaction" not in F1F2_NUMERIC_FEATURES
    assert not any(n.startswith("f3_") for n in F1F2_NUMERIC_FEATURES)
    # 6 numeric F1 + 6 F2.
    assert len(F1F2_NUMERIC_FEATURES) == 12


def test_logreg_is_deterministic_and_bounded() -> None:
    train, val = _feature_frame(300), _feature_frame(120)
    a = score_logreg_f1f2(train, val, seed=42)
    b = score_logreg_f1f2(train, val, seed=42)
    assert a.shape == (120,)
    assert np.array_equal(a, b), "fixed seed must reproduce predictions (§3.8)"
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_logreg_tolerates_nulls() -> None:
    """NULLs in a feature must not crash or produce NaN scores (sentinel fill)."""
    train, val = _feature_frame(300), _feature_frame(80)
    val.loc[:10, F1F2_NUMERIC_FEATURES[0]] = np.nan
    scores = score_logreg_f1f2(train, val)
    assert not np.isnan(scores).any()
