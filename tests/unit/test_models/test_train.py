"""Tests for the LightGBM training pipeline (Task 4.3).

The headline AC is reproducibility (§3.8): training twice on the same data
produces identical val predictions. Tested at two levels — the pure
`fit_lightgbm` on tiny arrays, and `train_model` end-to-end through a temp
feature store (artifact + metadata written, two runs byte-identical).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sentry.features.materialize import ALL_FEATURE_NAMES
from sentry.features.store import FeatureStore
from sentry.models.train import DEFAULT_PARAMS, LABEL, fit_lightgbm, train_model

_CATEGORICAL = ("f1_ip_app_interaction", "f1_ip_device_interaction")
_FAST_PARAMS = {**DEFAULT_PARAMS, "n_estimators": 40, "num_leaves": 15}


def _xy(n: int, seed: int):
    rng = np.random.default_rng(seed)
    signal = rng.uniform(size=n)
    x = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(6)})
    x["signal"] = signal
    y = (rng.uniform(size=n) < signal * 0.3).astype(int)
    return x, y


def test_fit_lightgbm_is_reproducible() -> None:
    xt, yt = _xy(2000, 0)
    xv, yv = _xy(600, 1)
    p1 = fit_lightgbm(xt, yt, xv, yv, params=_FAST_PARAMS).predict_proba(xv)[:, 1]
    p2 = fit_lightgbm(xt, yt, xv, yv, params=_FAST_PARAMS).predict_proba(xv)[:, 1]
    assert np.array_equal(p1, p2), "fixed seeds must reproduce predictions (§3.8)"


def test_early_stopping_uses_pr_auc_not_auc() -> None:
    xt, yt = _xy(2000, 0)
    xv, yv = _xy(600, 1)
    model = fit_lightgbm(xt, yt, xv, yv, params=_FAST_PARAMS)
    assert "average_precision" in model.best_score_["valid_0"]
    assert "auc" not in model.best_score_["valid_0"]


def _feature_frame(n: int, seed: int) -> pd.DataFrame:
    """A frame with every v0.x feature column + label; the pair conversion
    rate carries the signal so the model has something to learn."""
    rng = np.random.default_rng(seed)
    rate = rng.uniform(size=n)
    df = pd.DataFrame()
    for name in ALL_FEATURE_NAMES:
        if name in _CATEGORICAL:
            df[name] = [f"k{v}" for v in rng.integers(0, 5, n)]
        elif "conversion_rate" in name:
            df[name] = rate
        else:
            df[name] = rng.normal(size=n)
    df[LABEL] = (rng.uniform(size=n) < rate * 0.3).astype(int)
    return df


def _seed_store(tmp_path: Path) -> FeatureStore:
    store = FeatureStore(root=tmp_path / "features")
    (store.root / "v9.9.9").mkdir(parents=True)
    _feature_frame(1500, 0).to_parquet(store.root / "v9.9.9" / "train.parquet", index=False)
    _feature_frame(500, 1).to_parquet(store.root / "v9.9.9" / "val.parquet", index=False)
    return store


def test_train_model_writes_artifact_and_metadata(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    stamp = datetime(2026, 6, 9, tzinfo=UTC)
    result = train_model(
        "v9.9.9", tmp_path / "model", params=_FAST_PARAMS, store=store, trained_at=stamp
    )

    assert (tmp_path / "model" / "model.txt").exists()
    assert (tmp_path / "model" / "metadata.json").exists()
    assert result.features_version == "v9.9.9"
    assert result.n_train == 1500
    assert result.n_val == 500
    assert result.best_iteration >= 1
    assert 0.0 <= result.val_pr_auc <= 1.0
    # The two interaction features must be in the trained feature set.
    assert all(c in result.feature_names for c in _CATEGORICAL)


def test_train_model_reproducible_end_to_end(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    r1 = train_model("v9.9.9", tmp_path / "m1", params=_FAST_PARAMS, store=store)
    r2 = train_model("v9.9.9", tmp_path / "m2", params=_FAST_PARAMS, store=store)

    assert r1.val_pr_auc == r2.val_pr_auc
    assert r1.best_iteration == r2.best_iteration

    # Stronger: the saved boosters predict identically on a fresh matrix.
    val = pd.read_parquet(store.root / "v9.9.9" / "val.parquet")
    feats = r1.feature_names
    for c in _CATEGORICAL:
        val[c] = val[c].astype("category")
    b1 = lgb.Booster(model_file=str(tmp_path / "m1" / "model.txt"))
    b2 = lgb.Booster(model_file=str(tmp_path / "m2" / "model.txt"))
    assert np.array_equal(b1.predict(val[feats]), b2.predict(val[feats]))
