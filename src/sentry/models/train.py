"""LightGBM training pipeline (Task 4.3).

`train_model(features_version, output_dir)` loads a feature-store version,
trains a LightGBM classifier on train with early stopping on val, and
writes a versioned, reproducible model artifact plus training metadata.

Reproducibility is the gate (CLAUDE.md §3.8): `deterministic=True` +
`force_row_wise=True` + fixed seeds + fixed `num_threads` make two runs on
the same data produce byte-identical predictions. A test asserts it.

Imbalance is handled with class weights, never resampling (§3.3): SMOTE
interpolates meaningless points in categorical-heavy space and, done before
the split, leaks; undersampling discards data. `class_weight="balanced"`
is a weighted loss — the principled approach. Rationale in decisions.md.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import pandas as pd
import structlog
from pydantic import BaseModel

from sentry.evaluation.harness import evaluate
from sentry.features.materialize import ALL_FEATURE_NAMES
from sentry.features.store import FeatureStore

logger = structlog.get_logger(__name__)

LABEL: Final[str] = "is_attributed"
SEED: Final[int] = 42

#: String interaction features carried as pandas `category` so LightGBM uses
#: its native categorical split (no one-hot); everything else is numeric.
_CATEGORICAL: Final[tuple[str, ...]] = ("f1_ip_app_interaction", "f1_ip_device_interaction")

#: Default hyperparameters (build guide §4.3) plus the determinism trio.
#: Each choice, briefly (the full paragraph-per-param rationale is in
#: docs/decisions.md — interview material):
#:   num_leaves=63        — 2^6-1; capacity without the overfveritting of
#:                          very deep trees on a 0.2%-positive target.
#:   learning_rate=0.05   — slow enough that early stopping finds a good
#:                          round count; paired with many estimators.
#:   min_data_in_leaf=100 — leaves must cover ≥100 clicks, so a leaf can't
#:                          memorize a handful of rare positives.
#:   feature/bagging_fraction=0.8 — row+column subsampling for regularization
#:                          and decorrelated trees.
#:   class_weight=balanced — §3.3 weighted loss, not resampling.
DEFAULT_PARAMS: Final[dict[str, Any]] = {
    "objective": "binary",
    "n_estimators": 2000,  # upper bound; early stopping picks the real count
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_child_samples": 100,  # sklearn alias for min_data_in_leaf
    "subsample": 0.8,  # bagging_fraction
    "subsample_freq": 1,  # bag every iteration (else subsample is ignored)
    "colsample_bytree": 0.8,  # feature_fraction
    "class_weight": "balanced",
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 2,
    "verbose": -1,
}

#: Early-stopping patience (rounds without val PR-AUC improvement).
_EARLY_STOPPING_ROUNDS: Final[int] = 50


class TrainingResult(BaseModel):
    """What a training run produced — serialized to metadata.json."""

    model_version: str
    features_version: str
    seed: int
    n_train: int
    n_val: int
    n_features: int
    feature_names: list[str]
    params: dict[str, Any]
    best_iteration: int
    val_pr_auc: float
    val_roc_auc: float
    val_brier: float
    trained_at: datetime


def _seeded_params(params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Pin every RNG LightGBM exposes to `seed` (§3.8)."""
    return {
        **params,
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
    }


def fit_lightgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any] | None = None,
    seed: int = SEED,
) -> lgb.LGBMClassifier:
    """Fit one LightGBM classifier with early stopping on val PR-AUC.

    Pure (no I/O) so reproducibility can be tested on tiny arrays. Early
    stopping uses `average_precision` (= PR-AUC, the §3.2 headline), NOT
    auc — the build-guide failure mode of optimizing/reporting ROC-AUC on
    an imbalanced target.
    """
    model = lgb.LGBMClassifier(**_seeded_params(params or DEFAULT_PARAMS, seed))
    categorical = [c for c in _CATEGORICAL if c in x_train.columns]
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="average_precision",
        categorical_feature=categorical or "auto",
        callbacks=[lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def _load_features(store: FeatureStore, version: str, split: str) -> pd.DataFrame:
    """Load a split's features lean: numeric as float32, interactions as
    category, label as int8 — so 11M rows fit the container."""
    path = store.root / version / f"{split}.parquet"
    numeric = [n for n in ALL_FEATURE_NAMES if n not in _CATEGORICAL]
    cols = ", ".join(
        [
            *(f"CAST({n} AS FLOAT) AS {n}" for n in numeric),
            *_CATEGORICAL,
            f"{LABEL}::TINYINT {LABEL}",
        ]
    )
    import duckdb

    with duckdb.connect() as conn:
        conn.execute("SET memory_limit = '1.5GB'")
        df = conn.execute(f"SELECT {cols} FROM read_parquet('{path}')").fetch_df()
    for c in _CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def train_model(
    features_version: str,
    output_dir: Path | str,
    model_version: str = "lgbm-v0.1.0",
    params: dict[str, Any] | None = None,
    seed: int = SEED,
    store: FeatureStore | None = None,
    trained_at: datetime | None = None,
) -> TrainingResult:
    """Train on a feature version, write model.txt + metadata.json, return result."""
    store = store or FeatureStore()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_features(store, features_version, "train")
    val_df = _load_features(store, features_version, "val")
    feature_cols = [c for c in train_df.columns if c != LABEL]

    model = fit_lightgbm(
        train_df[feature_cols],
        train_df[LABEL].to_numpy(),
        val_df[feature_cols],
        val_df[LABEL].to_numpy(),
        params=params,
        seed=seed,
    )

    val_proba = model.predict_proba(val_df[feature_cols])[:, 1]
    metrics = evaluate(val_df[LABEL].to_numpy(), val_proba, name=f"{model_version}-val")

    model.booster_.save_model(str(output_dir / "model.txt"))
    result = TrainingResult(
        model_version=model_version,
        features_version=features_version,
        seed=seed,
        n_train=len(train_df),
        n_val=len(val_df),
        n_features=len(feature_cols),
        feature_names=feature_cols,
        params=_seeded_params(params or DEFAULT_PARAMS, seed),
        best_iteration=int(model.best_iteration_),
        val_pr_auc=metrics.pr_auc,
        val_roc_auc=metrics.roc_auc,
        val_brier=metrics.brier_score,
        trained_at=trained_at or datetime.now(tz=UTC),
    )
    (output_dir / "metadata.json").write_text(result.model_dump_json(indent=2))

    logger.info(
        "model_trained",
        model_version=model_version,
        best_iteration=result.best_iteration,
        val_pr_auc=round(result.val_pr_auc, 4),
        output=str(output_dir),
    )
    return result
