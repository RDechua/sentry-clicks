"""Property-level feature tests (Task 2.6).

Computation correctness lives in test_f1_per_click.py / test_f2_velocity.py
(hand-crafted known answers). This file pins the cross-cutting properties:

- edge cases: empty input, single row, all-same-IP, all-different-IP
- idempotence: computing twice yields identical tables
- no-leakage: train features are identical whether or not val/test data
  exists in the database at all
- schema stability: feature columns and dtypes match a frozen reference
- a deliberately BROKEN feature (the AND CURRENT ROW trap) fails the
  strictly-prior expectations — proof the tests do real work
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from sentry.data.splits import TRAIN_END_EXCLUSIVE, create_split_views
from sentry.features.f1_per_click import F1_FEATURES
from sentry.features.f2_velocity import F2_FEATURES
from sentry.features.f3_aggregates import F3_FEATURES
from sentry.features.pipeline import FeaturePipeline, SqlFeature

ALL_FEATURES = [*F1_FEATURES, *F2_FEATURES, *F3_FEATURES]

_T = datetime(2017, 11, 7, 9, 0, 0)

BuildDb = Callable[[Path, pd.DataFrame], Path]


def _clicks(rows: list[tuple[int, int, datetime]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ip", "app", "click_time"])
    df["device"] = 1
    df["os"] = 1
    df["channel"] = 1
    df["attributed_time"] = pd.NaT
    df["is_attributed"] = 0
    return df


@pytest.fixture
def make_db(tmp_path: Path, build_clicks_db: BuildDb) -> Callable[[pd.DataFrame], Path]:
    """Curried builder: df -> ingestion-faithful temp clicks DB."""
    counter = iter(range(100))

    def _make(df: pd.DataFrame) -> Path:
        return build_clicks_db(tmp_path / f"props_{next(counter)}.duckdb", df)

    return _make


def _compute(db_path: Path, source: str = "clicks") -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return FeaturePipeline(ALL_FEATURES).compute(conn, source=source)


# --- edge cases ---------------------------------------------------------


def test_empty_input(make_db: Callable[[pd.DataFrame], Path]) -> None:
    table = _compute(make_db(_clicks([]).astype({"click_time": "datetime64[us]"})))
    assert len(table) == 0
    for feat in ALL_FEATURES:
        assert feat.name in table.columns


def test_single_row(make_db: Callable[[pd.DataFrame], Path]) -> None:
    table = _compute(make_db(_clicks([(7, 3, _T)])))
    row = table.iloc[0]
    assert row["f2_clicks_per_ip_last_1hr"] == 0
    assert row["f2_clicks_per_ip_last_24hr"] == 0
    assert pd.isna(row["f2_inter_click_time_seconds"])
    assert pd.isna(row["f2_ip_click_std_inter_arrival"])
    assert row["f2_burst_score"] == 0


def test_all_same_ip(make_db: Callable[[pd.DataFrame], Path]) -> None:
    """Five clicks, one IP, 10s apart: the k-th click has k prior clicks."""
    df = _clicks([(7, 1, _T + timedelta(seconds=10 * k)) for k in range(5)])
    table = _compute(make_db(df)).sort_values("click_time")
    assert list(table["f2_clicks_per_ip_last_1hr"]) == [0, 1, 2, 3, 4]
    assert list(table["f2_inter_click_time_seconds"].iloc[1:]) == [10, 10, 10, 10]


def test_all_different_ips(make_db: Callable[[pd.DataFrame], Path]) -> None:
    """Every IP's first click: all counts 0, all gaps NULL."""
    df = _clicks([(k, 1, _T + timedelta(seconds=k)) for k in range(5)])
    table = _compute(make_db(df))
    assert (table["f2_clicks_per_ip_last_1hr"] == 0).all()
    assert table["f2_inter_click_time_seconds"].isna().all()


# --- idempotence --------------------------------------------------------


def test_recompute_is_identical(
    make_db: Callable[[pd.DataFrame], Path], tiny_sample_data: pd.DataFrame
) -> None:
    db_path = make_db(tiny_sample_data)
    first = _compute(db_path)
    second = _compute(db_path)
    pd.testing.assert_frame_equal(first, second)


# --- no-leakage ---------------------------------------------------------


def test_train_features_blind_to_val_and_test_data(
    make_db: Callable[[pd.DataFrame], Path],
) -> None:
    """Train features must be identical whether val/test rows exist or not.

    Same train-period rows in both databases; the second database also
    holds future (val/test-period) rows. If any feature's value changes,
    something is reading beyond the train view — leakage by construction.
    """
    train_rows = [(7, 1, _T + timedelta(minutes=k)) for k in range(10)]
    future_rows = [(7, 1, TRAIN_END_EXCLUSIVE + timedelta(minutes=k)) for k in range(50)]

    path_a = make_db(_clicks(train_rows))
    path_b = make_db(_clicks(train_rows + future_rows))
    create_split_views(path_a)
    create_split_views(path_b)

    features_a = _compute(path_a, source="clicks_train")
    features_b = _compute(path_b, source="clicks_train")

    pd.testing.assert_frame_equal(features_a, features_b)


# --- schema stability ---------------------------------------------------

#: Frozen reference: feature name -> pandas dtype as computed through the
#: pipeline. A change here is a contract change for every model downstream —
#: update deliberately, with a decisions.md note, not by accident.
FROZEN_SCHEMA = {
    "f1_app_id": "int32",  # INTEGER pass-throughs keep the ingestion type
    "f1_channel_id": "int32",
    "f1_device_id": "int32",
    "f1_os_id": "int32",
    "f1_hour_of_day": "int64",  # EXTRACT/isodow return BIGINT
    "f1_day_of_week": "int64",
    "f1_ip_app_interaction": "str",  # pandas 3 native string dtype
    "f1_ip_device_interaction": "str",
    "f2_clicks_per_ip_last_1hr": "int64",  # COUNT(*) is BIGINT, never NULL
    "f2_clicks_per_ip_last_24hr": "int64",
    "f2_clicks_per_ip_app_last_1hr": "int64",
    "f2_inter_click_time_seconds": "Int64",  # nullable: BIGINT with first-click NULLs
    "f2_ip_click_std_inter_arrival": "float64",
    "f2_burst_score": "int8",
    "f3_ip_conversion_rate_24hr": "float64",
    "f3_app_conversion_rate_24hr": "float64",
    "f3_ip_distinct_apps_24hr": "int64",
    "f3_ip_distinct_devices_24hr": "int64",
    "f3_ip_distinct_oses_24hr": "int64",
    "f3_ip_app_conversion_rate_24hr": "float64",
    "f3_ip_app_clicks_24hr": "int64",
    "f3_app_distinct_ips_24hr": "int64",
    "f3_ip_conversion_rate_alltime": "float64",
    "f3_ip_clicks_alltime": "int64",
    "f3_app_conversion_rate_alltime": "float64",
    "f3_app_clicks_alltime": "int64",
    "f3_ip_app_conversion_rate_alltime": "float64",
    "f3_ip_app_clicks_alltime": "int64",
}


def test_schema_matches_frozen_reference(clicks_db_path: Path) -> None:
    table = _compute(clicks_db_path)
    actual = {name: str(table[name].dtype) for name in FROZEN_SCHEMA}
    assert actual == FROZEN_SCHEMA


# --- the deliberately broken feature ------------------------------------


def test_broken_frame_is_caught_by_known_answer_expectations(
    make_db: Callable[[pd.DataFrame], Path],
) -> None:
    """The AND CURRENT ROW trap: self-inclusion shifts every count up by
    one (plus same-second peers). Running it through the SAME expectations
    that the real feature passes must FAIL — if this test ever starts
    passing, the test suite has stopped doing real work.
    """
    broken = SqlFeature(
        name="broken_1h_count",
        sql=(
            "SELECT row_id, COUNT(*) OVER ("
            "  PARTITION BY ip ORDER BY click_time"
            "  RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW"
            ") AS value FROM {source}"
        ),
        output_dtype="int64",
        description="deliberately leaky: includes the current row",
    )
    df = _clicks([(7, 1, _T + timedelta(seconds=10 * k)) for k in range(5)])
    with duckdb.connect(str(make_db(df)), read_only=True) as conn:
        table = FeaturePipeline([broken]).compute(conn, source="clicks")

    strictly_prior_expectation = [0, 1, 2, 3, 4]  # what the REAL feature yields
    actual = list(table.sort_values("click_time")["broken_1h_count"])
    assert actual != strictly_prior_expectation, "leaky frame passed the strict expectations"
    assert actual == [1, 2, 3, 4, 5], "self-inclusion shifts every count by exactly one"
