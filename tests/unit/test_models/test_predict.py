"""Tests for prediction + SHAP explanation (Task 6.3)."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from sentry.models.calibration import fit_calibrator
from sentry.models.predict import ModelBundle, load_model, predict_calibrated, top_contributions

FEATURES = [f"feat_{i}" for i in range(8)]


def _xy(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    # feat_0 drives the label, so SHAP should rank it highly.
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-2 * x["feat_0"]))).astype(int)
    return x, y


def _bundle(with_calibrator: bool = True) -> ModelBundle:
    x, y = _xy(3000, 0)
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 15, "verbose": -1, "seed": 0},
        lgb.Dataset(x, y),
        num_boost_round=40,
    )
    cal = fit_calibrator(np.asarray(booster.predict(x)), y) if with_calibrator else None
    return ModelBundle(booster=booster, feature_names=FEATURES, calibrator=cal)


def test_top_contributions_shape_and_ranking() -> None:
    bundle = _bundle()
    x, _ = _xy(50, 1)
    contribs = top_contributions(bundle, x, k=5)

    assert len(contribs) == 50
    for row in contribs:
        assert len(row) == 5
        # Sorted by descending |SHAP|.
        mags = [abs(c.shap_contribution) for c in row]
        assert mags == sorted(mags, reverse=True)
        assert all(c.feature_name in FEATURES for c in row)


def test_top_feature_is_the_driver() -> None:
    """feat_0 generates the label, so it should usually top the SHAP ranking."""
    bundle = _bundle()
    x, _ = _xy(200, 2)
    tops = [row[0].feature_name for row in top_contributions(bundle, x, k=5)]
    assert tops.count("feat_0") > 100  # the majority


def test_predict_calibrated_matches_manual() -> None:
    bundle = _bundle()
    x, _ = _xy(100, 3)
    got = predict_calibrated(bundle, x)
    assert bundle.calibrator is not None
    expected = bundle.calibrator.predict(np.asarray(bundle.booster.predict(x[FEATURES])))
    assert np.array_equal(got, expected)
    assert got.min() >= 0.0 and got.max() <= 1.0


def test_predict_calibrated_without_calibrator_raises() -> None:
    bundle = _bundle(with_calibrator=False)
    with pytest.raises(ValueError, match="calibrator"):
        predict_calibrated(bundle, _xy(10, 4)[0])


def test_shap_is_reproducible() -> None:
    bundle = _bundle()
    x, _ = _xy(40, 5)
    a = top_contributions(bundle, x)
    b = top_contributions(bundle, x)
    assert [c.shap_contribution for r in a for c in r] == [
        c.shap_contribution for r in b for c in r
    ]


def test_load_model_round_trip(tmp_path: Path) -> None:
    bundle = _bundle()
    assert bundle.calibrator is not None
    bundle.booster.save_model(str(tmp_path / "model.txt"))
    (tmp_path / "metadata.json").write_text(f'{{"feature_names": {FEATURES!r}}}'.replace("'", '"'))
    bundle.calibrator.save(tmp_path / "calibrator.json")

    loaded = load_model(tmp_path)
    assert loaded.feature_names == FEATURES
    assert loaded.calibrator is not None
    x, _ = _xy(20, 6)
    assert np.allclose(predict_calibrated(loaded, x), predict_calibrated(bundle, x))


def test_load_model_without_calibrator(tmp_path: Path) -> None:
    bundle = _bundle(with_calibrator=False)
    bundle.booster.save_model(str(tmp_path / "model.txt"))
    (tmp_path / "metadata.json").write_text(f'{{"feature_names": {FEATURES!r}}}'.replace("'", '"'))
    loaded = load_model(tmp_path)
    assert loaded.calibrator is None
