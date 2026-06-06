"""Tests for full-scale feature materialization (Task 4.1).

The composed single-query materializer must be provably equivalent to the
FeaturePipeline (which is the tested source of truth but materializes
whole splits in pandas and dies at 110M rows). The equivalence test IS
the trust story: same substrate, same rows, same values, feature by
feature.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from sentry.features.f1_per_click import F1_FEATURES
from sentry.features.f2_velocity import F2_FEATURES
from sentry.features.f3_aggregates import F3_FEATURES
from sentry.features.materialize import ALL_FEATURE_NAMES, materialize_features
from sentry.features.pipeline import FeaturePipeline


@pytest.fixture
def full_table(clicks_db_path: Path) -> pd.DataFrame:
    """Reference output: the (tested) FeaturePipeline over the fixture."""
    pipeline = FeaturePipeline([*F1_FEATURES, *F2_FEATURES, *F3_FEATURES])
    with duckdb.connect(str(clicks_db_path), read_only=True) as conn:
        return pipeline.compute(conn, source="clicks")


def test_materialized_features_equal_pipeline_output(
    clicks_db_path: Path, tmp_path: Path, full_table: pd.DataFrame
) -> None:
    """The one test that makes the composed SQL trustworthy."""
    out = tmp_path / "features.parquet"
    n = materialize_features(clicks_db_path, source="clicks", out_path=out)

    materialized = pd.read_parquet(out).sort_values("row_id").reset_index(drop=True)
    reference = full_table.sort_values("row_id").reset_index(drop=True)

    assert n == len(reference)
    assert set(ALL_FEATURE_NAMES) <= set(materialized.columns)
    for name in ALL_FEATURE_NAMES:
        pd.testing.assert_series_equal(
            materialized[name],
            reference[name],
            check_dtype=False,  # parquet round-trip vs fetch_df may differ in nullable repr
            check_names=False,
            obj=f"feature {name}",
        )


def test_sampling_is_deterministic_subset(
    clicks_db_path: Path, tmp_path: Path, full_table: pd.DataFrame
) -> None:
    """Sampled output: same rows on every run, a subset of the full rows,
    with features identical to the full-density values (sampling happens
    AFTER window computation)."""
    out_a = tmp_path / "a.parquet"
    out_b = tmp_path / "b.parquet"
    n_a = materialize_features(clicks_db_path, source="clicks", out_path=out_a, sample_fraction=0.5)
    n_b = materialize_features(clicks_db_path, source="clicks", out_path=out_b, sample_fraction=0.5)

    a = pd.read_parquet(out_a).sort_values("row_id").reset_index(drop=True)
    b = pd.read_parquet(out_b).sort_values("row_id").reset_index(drop=True)

    assert n_a == n_b == len(a)
    assert list(a["row_id"]) == list(b["row_id"]), "sampling must be deterministic"
    assert 0 < len(a) < len(full_table)

    # Full-density check: the sampled rows' features match the unsampled run.
    reference = full_table.set_index("row_id")
    sampled = a.set_index("row_id")
    for name in ("f2_clicks_per_ip_last_24hr", "f3_ip_app_conversion_rate_24hr"):
        ref_vals = reference.loc[sampled.index, name]
        pd.testing.assert_series_equal(
            sampled[name], ref_vals, check_dtype=False, check_names=False, obj=name
        )


def test_bad_fraction_raises(clicks_db_path: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fraction"):
        materialize_features(
            clicks_db_path, source="clicks", out_path=tmp_path / "x.parquet", sample_fraction=1.5
        )
