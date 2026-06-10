"""Tests for isotonic calibration (Task 4.5).

Pinned: calibration lowers Brier OUT-OF-SAMPLE (fit on one half, score the
other — in-sample improvement would be vacuous), the JSON Calibrator
round-trips and matches sklearn's predict, monotonicity holds, and clip
bounds keep outputs in [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss

from sentry.models.calibration import Calibrator, fit_calibrator, fit_isotonic


def _miscalibrated(n: int, seed: int):
    """Monotonic-but-miscalibrated scores: ranking is good, magnitudes are
    squashed (raw = true_prob**2), so isotonic has real work to do."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < true_p).astype(int)
    raw = true_p**2  # systematically too low — the thing calibration fixes
    return raw, y


def test_calibration_lowers_brier_out_of_sample() -> None:
    raw_fit, y_fit = _miscalibrated(5000, 0)
    raw_eval, y_eval = _miscalibrated(5000, 1)

    cal = fit_calibrator(raw_fit, y_fit)
    brier_pre = brier_score_loss(y_eval, raw_eval)
    brier_post = brier_score_loss(y_eval, cal.predict(raw_eval))

    assert (
        brier_post < brier_pre
    ), f"calibration must help out-of-sample: {brier_post} vs {brier_pre}"


def test_calibrator_json_round_trip(tmp_path: Path) -> None:
    raw, y = _miscalibrated(3000, 0)
    cal = fit_calibrator(raw, y)
    probe = np.linspace(0, 1, 200)

    path = tmp_path / "calibrator.json"
    cal.save(path)
    reloaded = Calibrator.load(path)

    assert np.array_equal(cal.predict(probe), reloaded.predict(probe))


def test_calibrator_matches_sklearn_predict() -> None:
    raw, y = _miscalibrated(3000, 0)
    iso = fit_isotonic(raw, y)
    cal = Calibrator.from_isotonic(iso)
    probe = np.linspace(raw.min(), raw.max(), 500)
    # np.interp over the knots reproduces sklearn's clipped isotonic predict.
    assert np.allclose(cal.predict(probe), iso.predict(probe), atol=1e-9)


def test_calibration_is_monotonic_and_bounded() -> None:
    raw, y = _miscalibrated(3000, 0)
    cal = fit_calibrator(raw, y)
    out = cal.predict(np.linspace(-0.5, 1.5, 1000))  # includes out-of-range inputs
    assert np.all(np.diff(out) >= -1e-12), "isotonic output must be non-decreasing"
    assert out.min() >= 0.0 and out.max() <= 1.0
