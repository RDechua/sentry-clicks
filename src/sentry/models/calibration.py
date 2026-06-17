"""Score calibration via isotonic regression (Task 4.5).

LightGBM scores rank well but are not probabilities — Week 5's cost-based
thresholding needs scores that read as P(is_attributed=1). Isotonic
regression maps raw scores → empirical probabilities.

Isotonic, not Platt (CLAUDE.md §3.5, build-guide stop-and-think): Platt
assumes the score→probability link is a sigmoid; isotonic assumes only
monotonicity, so it fits whatever shape the data shows. Platt's stronger
prior helps on small calibration sets; ours is ~3.7M val rows, far more
than enough for isotonic's flexibility to win.

Discipline (§3.5): the calibrator is FIT ON VAL and APPLIED TO TEST, never
fit on test. The fitted calibrator is saved as JSON thresholds (portable,
no pickle — same rationale as model.txt) alongside the model artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(raw_scores: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Fit isotonic raw-score → probability on a calibration set.

    `out_of_bounds="clip"` so scores beyond the fitted range at apply time
    map to the nearest endpoint probability rather than extrapolating.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_scores, y_true)
    return iso


class Calibrator:
    """A fitted isotonic map, stored as interpolation knots.

    Applying is `np.interp` over the monotonic knots (clipped to [0, 1]) —
    identical to sklearn's `IsotonicRegression.predict` with clip bounds,
    but serializable to plain JSON so the artifact carries no pickle.
    """

    def __init__(self, x_thresholds: np.ndarray, y_thresholds: np.ndarray) -> None:
        self.x_thresholds = np.asarray(x_thresholds, dtype="float64")
        self.y_thresholds = np.asarray(y_thresholds, dtype="float64")

    @classmethod
    def from_isotonic(cls, iso: IsotonicRegression) -> Calibrator:
        """Build a Calibrator from a fitted sklearn ``IsotonicRegression``."""
        return cls(iso.X_thresholds_, iso.y_thresholds_)

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities, clipped to [0, 1]."""
        raw = np.asarray(raw_scores, dtype="float64")
        calibrated = np.interp(raw, self.x_thresholds, self.y_thresholds)
        clipped: np.ndarray = np.clip(calibrated, 0.0, 1.0)
        return clipped

    def save(self, path: Path | str) -> None:
        """Serialize the calibrator knots to JSON at ``path`` (no pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "method": "isotonic",
                    "x_thresholds": self.x_thresholds.tolist(),
                    "y_thresholds": self.y_thresholds.tolist(),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path | str) -> Calibrator:
        """Load a Calibrator from a JSON knots file written by ``save``."""
        data = json.loads(Path(path).read_text())
        return cls(np.array(data["x_thresholds"]), np.array(data["y_thresholds"]))


def fit_calibrator(raw_scores: np.ndarray, y_true: np.ndarray) -> Calibrator:
    """Fit isotonic and return the serializable Calibrator in one step."""
    return Calibrator.from_isotonic(fit_isotonic(raw_scores, y_true))
