"""Hyperparameter tuning with Optuna (Task 4.4).

`tune_lgbm()` runs a TPE (Bayesian) search over LightGBM hyperparameters,
maximizing val PR-AUC, and persists the study to a SQLite file so it is
resumable across crashes. The final tuned model is trained separately on
the FULL feature version with the best params (`train_model` with
`params=study.best_params`-merged); tuning itself runs on a SUBSAMPLE of
train+val because the measured full-data train is ~12 min/trial, which
trips the Day-1 cloud-pivot threshold — tuning on a representative
subsample keeps 50 trials to ~2 h locally, and the final fit uses full
data (decisions.md, Task 4.4).

Tuning uses the val set, not cross-validation: the split is time-based
(CV would need TimeSeriesSplit), val is large enough to be a stable
estimate, and CV at this scale is needlessly expensive (build-guide
stop-and-think). The test set is never touched here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from sentry.features.store import FeatureStore
from sentry.models.train import DEFAULT_PARAMS, LABEL, MODEL_FEATURES, SEED, fit_lightgbm

#: Default trial budget. 50-100 is the build-guide range; the first ~30
#: trials cover the space and the rest refine marginally — 1000 would burn
#: compute for diminishing returns on a portfolio project (decisions.md).
DEFAULT_N_TRIALS: Final[int] = 50

#: Tuning subsample fractions (of the already-10%-sampled v0.5.0 splits).
#: ~1.7M train / ~0.9M val keeps trials fast and PR-AUC stable (~2k val
#: positives); the FINAL model trains on the full split.
TRAIN_SAMPLE_FRAC: Final[float] = 0.15
VAL_SAMPLE_FRAC: Final[float] = 0.25


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Search space (build guide §4.4), ranges defensible from LightGBM docs.

    Tunables override the DEFAULT_PARAMS; objective/determinism/n_estimators
    ceiling are kept from the base so trials stay reproducible and bounded.
    """
    return {
        **DEFAULT_PARAMS,
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _load_subsample(
    store: FeatureStore, version: str, split: str, frac: float, seed: int
) -> pd.DataFrame:
    """Load a seeded subsample of MODEL_FEATURES + label, float32, lean."""
    import duckdb

    path = store.root / version / f"{split}.parquet"
    cols = ", ".join(
        [*(f"CAST({n} AS FLOAT) AS {n}" for n in MODEL_FEATURES), f"{LABEL}::TINYINT {LABEL}"]
    )
    with duckdb.connect() as conn:
        conn.execute("SET memory_limit = '1.2GB'")
        return conn.execute(
            f"SELECT {cols} FROM read_parquet('{path}') "
            f"USING SAMPLE {int(frac * 100)} PERCENT (reservoir, {seed})"
        ).fetch_df()


def tune_lgbm(
    features_version: str,
    storage_path: Path | str,
    n_trials: int = DEFAULT_N_TRIALS,
    study_name: str = "lgbm-v0.5.0",
    seed: int = SEED,
    store: FeatureStore | None = None,
    train_frac: float = TRAIN_SAMPLE_FRAC,
    val_frac: float = VAL_SAMPLE_FRAC,
) -> optuna.Study:
    """Run (or resume) a TPE study maximizing val PR-AUC on a subsample.

    The study is keyed by `study_name` in the SQLite file at `storage_path`;
    re-running with the same args resumes and adds trials up to `n_trials`
    total. Returns the study (best_params / best_value / trials_dataframe).
    """
    store = store or FeatureStore()
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    train = _load_subsample(store, features_version, "train", train_frac, seed)
    val = _load_subsample(store, features_version, "val", val_frac, seed)
    x_train, y_train = train[list(MODEL_FEATURES)], train[LABEL].to_numpy()
    x_val, y_val = val[list(MODEL_FEATURES)], val[LABEL].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        model = fit_lightgbm(
            x_train, y_train, x_val, y_val, params=_suggest_params(trial), seed=seed
        )
        proba = model.predict_proba(x_val)[:, 1]
        return float(average_precision_score(y_val, proba))

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        load_if_exists=True,
    )
    n_remaining = max(0, n_trials - len(study.trials))
    study.optimize(objective, n_trials=n_remaining)
    return study


def best_full_params(study: optuna.Study) -> dict[str, Any]:
    """Merge the study's best tunables over DEFAULT_PARAMS for the final fit."""
    return {**DEFAULT_PARAMS, **study.best_params}
