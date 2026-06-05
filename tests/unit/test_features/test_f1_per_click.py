"""Tests for the F1 per-click features (Task 2.3).

Every expectation is computed independently in pandas from the fixture, so
the SQL is checked against a second implementation, not against itself.
Row alignment comes free from the pipeline's row_id join — these tests
compare against base columns of the SAME returned table.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from sentry.features.f1_per_click import F1_FEATURES
from sentry.features.pipeline import FeaturePipeline


@pytest.fixture
def f1_table(clicks_db_path: Path) -> pd.DataFrame:
    """The full F1 feature table computed once over the fixture DB."""
    pipeline = FeaturePipeline(F1_FEATURES)
    with duckdb.connect(str(clicks_db_path), read_only=True) as conn:
        return pipeline.compute(conn, source="clicks")


def test_f1_produces_all_eight_features(f1_table: pd.DataFrame) -> None:
    """AC: the pipeline produces a feature table joining all F1 features."""
    expected = {
        "f1_app_id",
        "f1_channel_id",
        "f1_device_id",
        "f1_os_id",
        "f1_hour_of_day",
        "f1_day_of_week",
        "f1_ip_app_interaction",
        "f1_ip_device_interaction",
    }
    assert expected <= set(f1_table.columns)
    assert len(F1_FEATURES) == len(expected)


@pytest.mark.parametrize(
    ("feature", "base_column"),
    [
        ("f1_app_id", "app"),
        ("f1_channel_id", "channel"),
        ("f1_device_id", "device"),
        ("f1_os_id", "os"),
    ],
)
def test_f1_passthroughs_equal_base_columns(
    f1_table: pd.DataFrame, feature: str, base_column: str
) -> None:
    assert (f1_table[feature] == f1_table[base_column]).all()


def test_f1_hour_of_day_matches_pandas(f1_table: pd.DataFrame) -> None:
    expected = f1_table["click_time"].dt.hour
    assert (f1_table["f1_hour_of_day"] == expected).all()
    assert f1_table["f1_hour_of_day"].between(0, 23).all()


def test_f1_day_of_week_is_iso(f1_table: pd.DataFrame) -> None:
    """isodow: Monday=1 .. Sunday=7 (pandas dayofweek is Monday=0, so +1)."""
    expected = f1_table["click_time"].dt.dayofweek + 1
    assert (f1_table["f1_day_of_week"] == expected).all()
    assert f1_table["f1_day_of_week"].between(1, 7).all()


def test_f1_interactions_concatenate_pairs(f1_table: pd.DataFrame) -> None:
    expected_app = f1_table["ip"].astype(str) + "_" + f1_table["app"].astype(str)
    expected_device = f1_table["ip"].astype(str) + "_" + f1_table["device"].astype(str)
    assert (f1_table["f1_ip_app_interaction"] == expected_app).all()
    assert (f1_table["f1_ip_device_interaction"] == expected_device).all()


def test_f1_features_carry_documentation() -> None:
    """Every feature must be explainable: non-empty description and dtype."""
    for feat in F1_FEATURES:
        assert feat.description, f"{feat.name} has no description"
        assert feat.output_dtype, f"{feat.name} has no output_dtype"
        assert "{source}" in feat.sql, f"{feat.name} SQL is not split-parameterized"
