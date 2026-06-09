"""Tests for the Optuna tuning module (Task 4.4).

Tuning correctness is hard to unit-test (it's stochastic search), so these
pin the contract: the study runs the requested trials, explores the full
search space, persists to SQLite and resumes without re-running trials,
and the best params merge cleanly for the final fit. Run on a tiny temp
store so it's fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from sentry.features.materialize import ALL_FEATURE_NAMES
from sentry.features.store import FeatureStore
from sentry.models.tune import _suggest_params, best_full_params, tune_lgbm

LABEL = "is_attributed"
_SEARCH_KEYS = {
    "num_leaves",
    "learning_rate",
    "min_child_samples",
    "colsample_bytree",
    "subsample",
    "reg_alpha",
    "reg_lambda",
}


def _feature_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rate = rng.uniform(size=n)
    df = pd.DataFrame()
    for name in ALL_FEATURE_NAMES:
        if name.endswith("_interaction"):
            df[name] = "k0"
        elif "conversion_rate" in name:
            df[name] = rate
        else:
            df[name] = rng.normal(size=n)
    df[LABEL] = (rng.uniform(size=n) < rate * 0.3).astype(int)
    return df


@pytest.fixture
def store(tmp_path: Path) -> FeatureStore:
    s = FeatureStore(root=tmp_path / "features")
    (s.root / "v9.9.9").mkdir(parents=True)
    _feature_frame(2000, 0).to_parquet(s.root / "v9.9.9" / "train.parquet", index=False)
    _feature_frame(800, 1).to_parquet(s.root / "v9.9.9" / "val.parquet", index=False)
    return s


def test_suggested_params_are_in_range_and_complete() -> None:
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    params = _suggest_params(study.ask())
    assert set(params) >= _SEARCH_KEYS
    assert 15 <= params["num_leaves"] <= 255
    assert 0.01 <= params["learning_rate"] <= 0.2
    assert 0.5 <= params["subsample"] <= 1.0
    # Base (non-tuned) params survive.
    assert params["objective"] == "binary"
    assert params["deterministic"] is True


def test_study_runs_requested_trials_and_maximizes(tmp_path: Path, store: FeatureStore) -> None:
    storage = tmp_path / "study.db"
    study = tune_lgbm(
        "v9.9.9", storage, n_trials=4, study_name="t", store=store, train_frac=0.8, val_frac=0.8
    )
    assert len(study.trials) == 4
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert 0.0 <= study.best_value <= 1.0
    assert set(study.best_params) == _SEARCH_KEYS


def test_study_is_resumable(tmp_path: Path, store: FeatureStore) -> None:
    """Re-running with a higher budget adds trials; it does not restart."""
    storage = tmp_path / "study.db"
    s1 = tune_lgbm(
        "v9.9.9", storage, n_trials=2, study_name="t", store=store, train_frac=0.8, val_frac=0.8
    )
    assert len(s1.trials) == 2
    s2 = tune_lgbm(
        "v9.9.9", storage, n_trials=5, study_name="t", store=store, train_frac=0.8, val_frac=0.8
    )
    assert len(s2.trials) == 5  # 2 reused + 3 new, not 7


def test_best_full_params_merges_over_defaults(tmp_path: Path, store: FeatureStore) -> None:
    study = tune_lgbm(
        "v9.9.9",
        tmp_path / "s.db",
        n_trials=3,
        study_name="t",
        store=store,
        train_frac=0.8,
        val_frac=0.8,
    )
    merged = best_full_params(study)
    assert merged["objective"] == "binary"  # base survives
    assert merged["deterministic"] is True
    for k in _SEARCH_KEYS:  # tuned values present
        assert k in merged
    # The tuned values actually came from the study, not the defaults.
    assert merged["num_leaves"] == study.best_params["num_leaves"]
