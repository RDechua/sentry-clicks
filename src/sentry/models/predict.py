"""Prediction and SHAP explanation (Task 6.3).

Loads a trained model bundle (booster + calibrator + feature order) and
produces, for a batch of cases:
  - calibrated P(is_attributed=1), via the isotonic calibrator (Task 4.5);
  - the top-5 contributing features per case by |SHAP|, for the audit log.

SHAP units: for a binary LightGBM Booster, `TreeExplainer.shap_values`
returns contributions in LOG-ODDS (margin) space — they sum with the
explainer's base value to the raw model margin, NOT to the probability.
The audit log records this; a reviewer reads a positive SHAP as "pushed the
log-odds of legitimacy up." SHAP is computed per BATCH (per-case calls are
slow); top-5 because that's enough to explain a decision and small enough
to keep audit entries manageable (build-guide §6.2 stop-and-think).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from sentry.audit.schema import FeatureContribution
from sentry.models.calibration import Calibrator

#: Features logged per case (CLAUDE.md §3.9 convention).
TOP_K: Final[int] = 5


@dataclass(frozen=True)
class ModelBundle:
    """A trained model ready to score and explain."""

    booster: lgb.Booster
    feature_names: list[str]
    calibrator: Calibrator | None = None


def load_model(model_dir: Path | str) -> ModelBundle:
    """Load model.txt + metadata.json (feature order) + calibrator.json if present."""
    model_dir = Path(model_dir)
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    metadata = json.loads((model_dir / "metadata.json").read_text())
    calibrator_path = model_dir / "calibrator.json"
    calibrator = Calibrator.load(calibrator_path) if calibrator_path.exists() else None
    return ModelBundle(booster, list(metadata["feature_names"]), calibrator)


def predict_calibrated(bundle: ModelBundle, x: pd.DataFrame) -> np.ndarray:
    """Calibrated P(is_attributed=1) for each row (requires a calibrator)."""
    if bundle.calibrator is None:
        raise ValueError("bundle has no calibrator; cannot produce calibrated probabilities")
    raw = np.asarray(bundle.booster.predict(x[bundle.feature_names]))
    return bundle.calibrator.predict(raw)


def top_contributions(
    bundle: ModelBundle, x: pd.DataFrame, k: int = TOP_K
) -> list[list[FeatureContribution]]:
    """Top-`k` features per row by |SHAP| (log-odds), for the audit log.

    Computed once over the whole batch. Each returned `FeatureContribution`
    carries the feature's value for that row and its SHAP value.
    """
    feats = bundle.feature_names
    matrix = x[feats]
    shap_values = np.asarray(shap.TreeExplainer(bundle.booster).shap_values(matrix))
    # Binary Booster -> single (n, n_features) array; guard the list form too.
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    values = matrix.to_numpy()
    # Indices of the top-k by absolute SHAP, per row.
    top_idx = np.argsort(-np.abs(shap_values), axis=1)[:, :k]

    out: list[list[FeatureContribution]] = []
    for row in range(len(matrix)):
        out.append(
            [
                FeatureContribution(
                    feature_name=feats[j],
                    value=float(values[row, j]),
                    shap_contribution=float(shap_values[row, j]),
                )
                for j in top_idx[row]
            ]
        )
    return out
