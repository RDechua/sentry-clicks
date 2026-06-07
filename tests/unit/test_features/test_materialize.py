"""Tests for full-scale feature materialization (Task 4.1).

The composed single-query materializer must be provably equivalent to the
FeaturePipeline (which is the tested source of truth but materializes
whole splits in pandas and dies at 110M rows). The equivalence test IS
the trust story: same substrate, same rows, same values, feature by
feature.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
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


def test_equivalence_on_boundary_adversarial_timeline(
    tmp_path: Path,
    build_clicks_db: object,
) -> None:
    """A timeline built to break the prefix-sum/segment-event rewrites:
    same-second peers, clicks exactly 24h apart (segment merge edge),
    clicks 24h+1s apart (segment split edge), and one busy app shared by
    many ips. If the ASOF formulations differ from the RANGE frames by
    even a millisecond, this catches it."""
    t0 = datetime(2017, 11, 6, 16, 0, 0)
    rows = [
        # (ip, app, click_time, label)
        (1, 1, t0, 1),
        (2, 1, t0, 0),  # same-second, different ip, same app
        (1, 1, t0, 0),  # same-second same-pair peer
        (3, 1, t0 + timedelta(hours=12), 1),
        (1, 1, t0 + timedelta(hours=24), 0),  # exactly 24h after first
        (2, 1, t0 + timedelta(hours=24, seconds=1), 1),  # 24h+1s: segment split
        (4, 2, t0 + timedelta(hours=1), 0),  # second app, single click
        (3, 1, t0 + timedelta(hours=36, seconds=30), 0),
        (1, 2, t0 + timedelta(hours=48), 1),
    ]
    df = pd.DataFrame(rows, columns=["ip", "app", "click_time", "is_attributed"])
    df["device"] = 1
    df["os"] = 1
    df["channel"] = 1
    df["attributed_time"] = pd.NaT

    db_path = build_clicks_db(tmp_path / "adv.duckdb", df)  # type: ignore[operator]
    pipeline = FeaturePipeline([*F1_FEATURES, *F2_FEATURES, *F3_FEATURES])
    with duckdb.connect(str(db_path), read_only=True) as conn:
        reference = pipeline.compute(conn, source="clicks").sort_values("row_id")

    out = tmp_path / "adv.parquet"
    materialize_features(db_path, source="clicks", out_path=out)
    materialized = pd.read_parquet(out).sort_values("row_id").reset_index(drop=True)
    reference = reference.reset_index(drop=True)

    for name in ALL_FEATURE_NAMES:
        pd.testing.assert_series_equal(
            materialized[name],
            reference[name],
            check_dtype=False,
            check_names=False,
            obj=f"feature {name}",
        )


def test_split_assembly_uses_full_history(
    tmp_path: Path,
    build_clicks_db: object,
) -> None:
    """The amended §3.4 (2026-06-07): split_name assigns ROWS to a split,
    but windows see all strictly-prior history — a val row whose IP
    clicked during the train period must have a populated window, not the
    cold-start the per-split-source rule produced. Also: only val-period
    rows are emitted, and future (test-period) rows change nothing."""
    from sentry.data.splits import TRAIN_END_EXCLUSIVE, VAL_END_EXCLUSIVE

    train_rows = [(7, 1, TRAIN_END_EXCLUSIVE - timedelta(hours=2, minutes=k), 0) for k in range(5)]
    val_row = [(7, 1, TRAIN_END_EXCLUSIVE + timedelta(minutes=30), 0)]
    test_rows = [(7, 1, VAL_END_EXCLUSIVE + timedelta(hours=1), 1)]

    def _df(rows: Sequence[tuple[int, int, datetime, int]]) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["ip", "app", "click_time", "is_attributed"])
        df["device"] = 1
        df["os"] = 1
        df["channel"] = 1
        df["attributed_time"] = pd.NaT
        return df

    db_a = build_clicks_db(tmp_path / "a.duckdb", _df(train_rows + val_row))  # type: ignore[operator]
    db_b = build_clicks_db(  # type: ignore[operator]
        tmp_path / "b.duckdb", _df(train_rows + val_row + test_rows)
    )

    out_a = tmp_path / "a.parquet"
    out_b = tmp_path / "b.parquet"
    materialize_features(db_a, source="clicks", out_path=out_a, split_name="val")
    materialize_features(db_b, source="clicks", out_path=out_b, split_name="val")

    a = pd.read_parquet(out_a)
    b = pd.read_parquet(out_b)

    # Only the val-period row is emitted.
    assert len(a) == len(b) == 1
    # Its window sees the five train-period clicks — full history, no cold-start.
    assert a["f2_clicks_per_ip_last_24hr"].iloc[0] == 5
    assert a["f3_ip_conversion_rate_24hr"].iloc[0] == 0.0  # five prior labels, all 0
    # Future (test-period) rows change nothing: strictly-prior is intact.
    pd.testing.assert_frame_equal(a, b)


def test_bad_fraction_raises(clicks_db_path: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fraction"):
        materialize_features(
            clicks_db_path, source="clicks", out_path=tmp_path / "x.parquet", sample_fraction=1.5
        )
