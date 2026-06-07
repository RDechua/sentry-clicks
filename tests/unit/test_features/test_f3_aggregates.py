"""Tests for the F3 aggregate behavioral features (Task 3.1).

F3 is where leakage risk peaks: conversion-rate features aggregate the
LABELS of prior clicks. The tests here are hand-crafted known answers plus
the two anti-leakage assertions that matter most:

- the build-guide canary: a first click with label=1 gets conversion rate
  NULL, never 1.0
- the label-flip test: changing a row's own label must not change that
  row's features (labels flow forward in time only)

Timeline (ip, app, device, os, label):
  A ip1 09:00 app1 d1 o1 L=1     E ip2 09:15 app1 d1 o1 L=1 (first click)
  B ip1 09:30 app2 d2 o1 L=0     F ip3 12:00 app1 d1 o1 L=0
  C ip1 10:00 app1 d1 o2 L=1     G ip3 12:30 app1 d1 o1 L=0
  D ip1 11:00 app3 d1 o1 L=0
  H ip1 next-day 09:30 (A aged out of 24h; B exactly 24h prior -> included)
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from sentry.features.f3_aggregates import F3_FEATURES
from sentry.features.pipeline import FeaturePipeline

_T = datetime(2017, 11, 7, 9, 0, 0)
_DAY = timedelta(hours=24)


def _f3_df() -> pd.DataFrame:
    rows = [
        # (ip, app, device, os, click_time, is_attributed)
        (1, 1, 1, 1, _T, 1),  # A
        (1, 2, 2, 1, _T + timedelta(minutes=30), 0),  # B
        (1, 1, 1, 2, _T + timedelta(hours=1), 1),  # C
        (1, 3, 1, 1, _T + timedelta(hours=2), 0),  # D
        (2, 1, 1, 1, _T + timedelta(minutes=15), 1),  # E — first click, label 1
        (3, 1, 1, 1, _T + timedelta(hours=3), 0),  # F
        (3, 1, 1, 1, _T + timedelta(hours=3, minutes=30), 0),  # G
        (1, 1, 1, 1, _T + _DAY + timedelta(minutes=30), 0),  # H
    ]
    df = pd.DataFrame(rows, columns=["ip", "app", "device", "os", "click_time", "is_attributed"])
    df["channel"] = 1
    df["attributed_time"] = pd.NaT
    return df


def _compute(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return FeaturePipeline(F3_FEATURES).compute(conn, source="clicks")


@pytest.fixture
def f3_table(tmp_path: Path, build_clicks_db: Callable[[Path, pd.DataFrame], Path]) -> pd.DataFrame:
    return _compute(build_clicks_db(tmp_path / "f3.duckdb", _f3_df()))


def _row(table: pd.DataFrame, ip: int, ts: datetime) -> pd.Series:
    match = table[(table["ip"] == ip) & (table["click_time"] == ts)]
    assert len(match) == 1
    return match.iloc[0]


def test_first_click_with_positive_label_gets_null_not_one(f3_table: pd.DataFrame) -> None:
    """THE build-guide canary: E has label=1 and zero prior clicks. If its
    own label leaked into its window, the rate would be 1.0; it must be NULL."""
    e = _row(f3_table, ip=2, ts=_T + timedelta(minutes=15))
    assert pd.isna(e["f3_ip_conversion_rate_24hr"])


def test_ip_conversion_rate_known_answers(f3_table: pd.DataFrame) -> None:
    """D's priors are A(1), B(0), C(1) -> 2/3. B's prior is A(1) -> 1.0."""
    d = _row(f3_table, ip=1, ts=_T + timedelta(hours=2))
    b = _row(f3_table, ip=1, ts=_T + timedelta(minutes=30))
    assert d["f3_ip_conversion_rate_24hr"] == pytest.approx(2 / 3)
    assert b["f3_ip_conversion_rate_24hr"] == pytest.approx(1.0)


def test_ip_conversion_rate_ages_out(f3_table: pd.DataFrame) -> None:
    """H's 24h window: A (24h30m prior) aged out; B exactly 24h prior is on
    the inclusive bound; priors = B(0), C(1), D(0) -> 1/3."""
    h = _row(f3_table, ip=1, ts=_T + _DAY + timedelta(minutes=30))
    assert h["f3_ip_conversion_rate_24hr"] == pytest.approx(1 / 3)


def test_app_conversion_rate_partitions_by_app_across_ips(f3_table: pd.DataFrame) -> None:
    """App 1's clicks in time order: A(1), E(1), C(1), F(0), G, H.
    G's priors = A, E, C, F -> 3/4. F's priors = A, E, C -> 1.0."""
    g = _row(f3_table, ip=3, ts=_T + timedelta(hours=3, minutes=30))
    f = _row(f3_table, ip=3, ts=_T + timedelta(hours=3))
    assert g["f3_app_conversion_rate_24hr"] == pytest.approx(3 / 4)
    assert f["f3_app_conversion_rate_24hr"] == pytest.approx(1.0)


def test_distinct_counts_known_answers(f3_table: pd.DataFrame) -> None:
    """D's priors A, B, C: apps {1,2} -> 2, devices {1,2} -> 2, oses {1,2} -> 2.
    First clicks get an honest 0 (a count, unlike the undefined rate)."""
    d = _row(f3_table, ip=1, ts=_T + timedelta(hours=2))
    e = _row(f3_table, ip=2, ts=_T + timedelta(minutes=15))
    assert d["f3_ip_distinct_apps_24hr"] == 2
    assert d["f3_ip_distinct_devices_24hr"] == 2
    assert d["f3_ip_distinct_oses_24hr"] == 2
    assert e["f3_ip_distinct_apps_24hr"] == 0


def test_flipping_own_label_does_not_move_own_features(
    tmp_path: Path, build_clicks_db: Callable[[Path, pd.DataFrame], Path]
) -> None:
    """Labels must flow forward in time only. Flip D's label (0 -> 1):
    D's OWN feature values must be identical; H (which has D in its
    window) must move — proof the label propagates forward, not inward."""
    df_orig = _f3_df()
    df_flip = _f3_df()
    d_mask = (df_flip["ip"] == 1) & (df_flip["click_time"] == _T + timedelta(hours=2))
    df_flip.loc[d_mask, "is_attributed"] = 1

    table_orig = _compute(build_clicks_db(tmp_path / "orig.duckdb", df_orig))
    table_flip = _compute(build_clicks_db(tmp_path / "flip.duckdb", df_flip))

    d_ts = _T + timedelta(hours=2)
    h_ts = _T + _DAY + timedelta(minutes=30)
    f3_cols = [feat.name for feat in F3_FEATURES]

    d_orig = _row(table_orig, ip=1, ts=d_ts)[f3_cols]
    d_flip = _row(table_flip, ip=1, ts=d_ts)[f3_cols]
    pd.testing.assert_series_equal(d_orig, d_flip, check_names=False)

    h_orig = _row(table_orig, ip=1, ts=h_ts)
    h_flip = _row(table_flip, ip=1, ts=h_ts)
    assert h_orig["f3_ip_conversion_rate_24hr"] == pytest.approx(1 / 3)
    assert h_flip["f3_ip_conversion_rate_24hr"] == pytest.approx(2 / 3)


def test_pair_conversion_rate_known_answers(f3_table: pd.DataFrame) -> None:
    """The Task 3.2 headline feature, on hand-crafted data.

    Pair (ip1, app1) in time order: A(1), C(1), H. H's 24h pair window
    holds C only (A aged out 30 minutes earlier) -> rate 1.0, denom 1.
    Pair (ip3, app1): G's prior is F(0) -> rate 0.0, denom 1.
    D is (ip1, app3)'s first click -> rate NULL, denom 0 — even though
    ip1 itself has plenty of history.
    """
    h = _row(f3_table, ip=1, ts=_T + _DAY + timedelta(minutes=30))
    g = _row(f3_table, ip=3, ts=_T + timedelta(hours=3, minutes=30))
    d = _row(f3_table, ip=1, ts=_T + timedelta(hours=2))

    assert h["f3_ip_app_conversion_rate_24hr"] == pytest.approx(1.0)
    assert h["f3_ip_app_clicks_24hr"] == 1
    assert g["f3_ip_app_conversion_rate_24hr"] == pytest.approx(0.0)
    assert g["f3_ip_app_clicks_24hr"] == 1
    assert pd.isna(d["f3_ip_app_conversion_rate_24hr"])
    assert d["f3_ip_app_clicks_24hr"] == 0


def test_pair_rate_canary_first_pair_click_with_positive_label(f3_table: pd.DataFrame) -> None:
    """Same canary as the IP rate, at pair granularity: E is (ip2, app1)'s
    first click with label=1 -> NULL, never 1.0."""
    e = _row(f3_table, ip=2, ts=_T + timedelta(minutes=15))
    assert pd.isna(e["f3_ip_app_conversion_rate_24hr"])


def test_app_degree_known_answers(f3_table: pd.DataFrame) -> None:
    """G: app 1's strictly-prior clickers are A,C (ip1), E (ip2), F (ip3)
    -> 3 distinct IPs. H next day: A and E have aged out of the 24h window
    (24h30m and 24h15m prior); C, F, G remain -> ips {1, 3} -> 2."""
    g = _row(f3_table, ip=3, ts=_T + timedelta(hours=3, minutes=30))
    h = _row(f3_table, ip=1, ts=_T + _DAY + timedelta(minutes=30))
    assert g["f3_app_distinct_ips_24hr"] == 3
    assert h["f3_app_distinct_ips_24hr"] == 2


def test_alltime_rates_keep_what_24h_forgot(f3_table: pd.DataFrame) -> None:
    """The long-memory variants (added after the Week 4 density gate): H's
    24h ip window dropped A, but the all-time window keeps it — priors
    A(1), B(0), C(1), D(0) -> 1/2 vs the 24h value of 1/3. Same strictly-
    prior discipline, no time cap."""
    h = _row(f3_table, ip=1, ts=_T + _DAY + timedelta(minutes=30))
    assert h["f3_ip_conversion_rate_alltime"] == pytest.approx(1 / 2)
    assert h["f3_ip_clicks_alltime"] == 4
    # Pair (ip1, app1): priors A(1), C(1) -> 1.0 over 2 (24h saw only C).
    assert h["f3_ip_app_conversion_rate_alltime"] == pytest.approx(1.0)
    assert h["f3_ip_app_clicks_alltime"] == 2
    # App 1 all-time at H: A(1), E(1), C(1), F(0), G(0) -> 3/5.
    assert h["f3_app_conversion_rate_alltime"] == pytest.approx(3 / 5)
    assert h["f3_app_clicks_alltime"] == 5


def test_alltime_rates_share_the_null_canary(f3_table: pd.DataFrame) -> None:
    """First click with label=1 -> NULL at all-time granularity too."""
    e = _row(f3_table, ip=2, ts=_T + timedelta(minutes=15))
    assert pd.isna(e["f3_ip_conversion_rate_alltime"])
    assert pd.isna(e["f3_ip_app_conversion_rate_alltime"])
    assert e["f3_ip_clicks_alltime"] == 0


def test_f3_features_carry_documentation() -> None:
    for feat in F3_FEATURES:
        assert feat.description, f"{feat.name} has no description"
        assert "{source}" in feat.sql, f"{feat.name} SQL is not split-parameterized"
