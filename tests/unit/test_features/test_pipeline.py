"""Tests for `sentry.features.pipeline` — the feature pipeline framework.

Framework-level behavior only (mock features). Real feature definitions get
their own tests in Tasks 2.3/2.4. The properties pinned here:

- features compute in dependency order regardless of registration order
- SQL features align to base rows via row_id, not result order
- misalignment, cycles, unknown deps, and duplicate names fail loud
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from sentry.features.pipeline import (
    FEATURE_REGISTRY,
    FeaturePipeline,
    PythonFeature,
    SqlFeature,
    python_feature,
)


def _db_with_clicks(tmp_path: Path, df: pd.DataFrame) -> Path:
    """Temp DuckDB with the ingestion row_id contract: contiguous, time-ordered."""
    db_path = tmp_path / "pipeline.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        conn.register("source_df", df)
        conn.execute(
            "CREATE TABLE clicks AS "
            "SELECT row_number() OVER (ORDER BY click_time, ip) AS row_id, * "
            "FROM source_df"
        )
    return db_path


def _ip_doubled(df: pd.DataFrame) -> pd.Series:
    return df["ip"] * 2


def _ip_doubled_plus_one(df: pd.DataFrame) -> pd.Series:
    return df["ip_doubled"] + 1


def test_python_features_compute_in_dependency_order(
    tmp_path: Path, tiny_sample_data: pd.DataFrame
) -> None:
    """Registration order is B-before-A; dependency order must win."""
    db_path = _db_with_clicks(tmp_path, tiny_sample_data)
    pipeline = FeaturePipeline(
        [
            PythonFeature(
                name="ip_doubled_plus_one",
                fn=_ip_doubled_plus_one,
                output_dtype="int64",
                description="ip * 2 + 1, depends on ip_doubled",
                dependencies=("ip_doubled",),
            ),
            PythonFeature(
                name="ip_doubled",
                fn=_ip_doubled,
                output_dtype="int64",
                description="ip * 2",
            ),
        ]
    )

    with duckdb.connect(str(db_path), read_only=True) as conn:
        table = pipeline.compute(conn, source="clicks")

    assert (table["ip_doubled"] == table["ip"] * 2).all()
    assert (table["ip_doubled_plus_one"] == table["ip"] * 2 + 1).all()


def test_sql_feature_aligns_on_row_id_not_result_order(
    tmp_path: Path, tiny_sample_data: pd.DataFrame
) -> None:
    """The SQL result is deliberately mis-ordered; row_id join must fix it."""
    db_path = _db_with_clicks(tmp_path, tiny_sample_data)
    pipeline = FeaturePipeline(
        [
            SqlFeature(
                name="ip_times_ten",
                sql="SELECT row_id, ip * 10 AS value FROM {source} ORDER BY value DESC",
                output_dtype="int64",
                description="ip * 10, returned in scrambled order",
            )
        ]
    )

    with duckdb.connect(str(db_path), read_only=True) as conn:
        table = pipeline.compute(conn, source="clicks")

    assert (table["ip_times_ten"] == table["ip"] * 10).all()


def test_python_feature_can_depend_on_sql_feature(
    tmp_path: Path, tiny_sample_data: pd.DataFrame
) -> None:
    db_path = _db_with_clicks(tmp_path, tiny_sample_data)
    pipeline = FeaturePipeline(
        [
            PythonFeature(
                name="ip_times_ten_log",
                fn=lambda df: df["ip_times_ten"] - df["ip"],
                output_dtype="int64",
                description="difference of sql feature and base column",
                dependencies=("ip_times_ten",),
            ),
            SqlFeature(
                name="ip_times_ten",
                sql="SELECT row_id, ip * 10 AS value FROM {source}",
                output_dtype="int64",
                description="ip * 10",
            ),
        ]
    )

    with duckdb.connect(str(db_path), read_only=True) as conn:
        table = pipeline.compute(conn, source="clicks")

    assert (table["ip_times_ten_log"] == table["ip"] * 9).all()


def test_sql_feature_row_mismatch_raises(tmp_path: Path, tiny_sample_data: pd.DataFrame) -> None:
    """A feature query that drops rows is a leakage/misalignment bug — fail loud."""
    db_path = _db_with_clicks(tmp_path, tiny_sample_data)
    pipeline = FeaturePipeline(
        [
            SqlFeature(
                name="partial",
                sql="SELECT row_id, ip AS value FROM {source} WHERE ip > 100",
                output_dtype="int64",
                description="deliberately drops rows",
            )
        ]
    )

    with (
        duckdb.connect(str(db_path), read_only=True) as conn,
        pytest.raises(ValueError, match="partial"),
    ):
        pipeline.compute(conn, source="clicks")


def test_cycle_raises() -> None:
    a = PythonFeature("a", lambda df: df["ip"], "int64", "cyclic a", dependencies=("b",))
    b = PythonFeature("b", lambda df: df["ip"], "int64", "cyclic b", dependencies=("a",))
    with pytest.raises(ValueError, match=r"cycle"):
        FeaturePipeline([a, b])


def test_unknown_dependency_raises() -> None:
    a = PythonFeature("a", lambda df: df["ip"], "int64", "dep missing", dependencies=("ghost",))
    with pytest.raises(ValueError, match="ghost"):
        FeaturePipeline([a])


def test_duplicate_name_raises() -> None:
    a1 = PythonFeature("a", lambda df: df["ip"], "int64", "first")
    a2 = PythonFeature("a", lambda df: df["ip"], "int64", "second")
    with pytest.raises(ValueError, match="duplicate"):
        FeaturePipeline([a1, a2])


def test_python_feature_decorator_registers() -> None:
    before = dict(FEATURE_REGISTRY)
    try:

        @python_feature(name="reg_test_feature", output_dtype="int64", description="registry")
        def reg_test_feature(df: pd.DataFrame) -> pd.Series:
            return df["ip"]

        assert "reg_test_feature" in FEATURE_REGISTRY
        registered = FEATURE_REGISTRY["reg_test_feature"]
        assert isinstance(registered, PythonFeature)
        assert registered.description == "registry"
    finally:
        FEATURE_REGISTRY.clear()
        FEATURE_REGISTRY.update(before)


def test_output_keeps_base_columns_and_row_count(
    tmp_path: Path, tiny_sample_data: pd.DataFrame
) -> None:
    db_path = _db_with_clicks(tmp_path, tiny_sample_data)
    pipeline = FeaturePipeline(
        [SqlFeature("f", "SELECT row_id, 1 AS value FROM {source}", "int64", "constant")]
    )

    with duckdb.connect(str(db_path), read_only=True) as conn:
        table = pipeline.compute(conn, source="clicks")

    assert len(table) == len(tiny_sample_data)
    for col in ("row_id", "ip", "click_time", "is_attributed", "f"):
        assert col in table.columns
