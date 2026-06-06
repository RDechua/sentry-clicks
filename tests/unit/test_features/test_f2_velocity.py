"""Tests for the F2 velocity features (Task 2.4).

Every test runs against a hand-crafted timeline where the right answer is
computable by hand (build-guide AC). The timeline encodes the edge cases
that matter:

ip=1 (app in parens):           ip=2:            ip=3 (the burst):
  A 09:00:00 (1) ┐ same second    F 09:15:00 (1)   G0..G11 12:00:00 + 10s*k (3)
  B 09:00:00 (1) ┘                (only click)     twelve clicks, 10s apart
  C 09:30:00 (1)
  D 10:00:00 (2)   exactly 1h after A/B — tests the inclusive lower bound
  E 11:00:01 (1)   3601s after D — everything has aged out of the 1h window
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from sentry.features.f2_velocity import F2_FEATURES
from sentry.features.pipeline import FeaturePipeline

_T = datetime(2017, 11, 7, 9, 0, 0)


def _velocity_df() -> pd.DataFrame:
    rows = [
        # (ip, app, click_time)
        (1, 1, _T),  # A
        (1, 1, _T),  # B — same-second peer of A
        (1, 1, _T + timedelta(minutes=30)),  # C
        (1, 2, _T + timedelta(hours=1)),  # D — exactly 1h after A/B
        (1, 1, _T + timedelta(hours=2, seconds=1)),  # E — 3601s after D
        (2, 1, _T + timedelta(minutes=15)),  # F — ip 2's only click
    ]
    burst_start = _T + timedelta(hours=3)
    rows += [(3, 3, burst_start + timedelta(seconds=10 * k)) for k in range(12)]  # G0..G11
    df = pd.DataFrame(rows, columns=["ip", "app", "click_time"])
    df["device"] = 1
    df["os"] = 1
    df["channel"] = 1
    df["attributed_time"] = pd.NaT
    df["is_attributed"] = 0
    return df


@pytest.fixture(scope="module")
def f2_table(
    tmp_path_factory: pytest.TempPathFactory,
    build_clicks_db: Callable[[Path, pd.DataFrame], Path],
) -> pd.DataFrame:
    """F2 features computed once over the hand-crafted timeline."""
    db_path = build_clicks_db(tmp_path_factory.mktemp("f2") / "velocity.duckdb", _velocity_df())
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return FeaturePipeline(F2_FEATURES).compute(conn, source="clicks")


def _row(table: pd.DataFrame, ip: int, ts: datetime, app: int | None = None) -> pd.DataFrame:
    mask = (table["ip"] == ip) & (table["click_time"] == ts)
    if app is not None:
        mask &= table["app"] == app
    return table[mask]


def test_1h_count_excludes_same_second_peers(f2_table: pd.DataFrame) -> None:
    """A and B are each other's peers; strictly-prior means neither counts."""
    pair = _row(f2_table, ip=1, ts=_T)
    assert len(pair) == 2
    assert list(pair["f2_clicks_per_ip_last_1hr"]) == [0, 0]


def test_1h_count_includes_click_exactly_one_hour_prior(f2_table: pd.DataFrame) -> None:
    """D at 10:00: window [09:00:00.000, 09:59:59.999] — A, B (on the
    boundary, inclusive) and C are all inside -> 3."""
    d = _row(f2_table, ip=1, ts=_T + timedelta(hours=1))
    assert list(d["f2_clicks_per_ip_last_1hr"]) == [3]


def test_1h_count_ages_out(f2_table: pd.DataFrame) -> None:
    """E at 11:00:01: D (3601s before) just aged out -> 0."""
    e = _row(f2_table, ip=1, ts=_T + timedelta(hours=2, seconds=1))
    assert list(e["f2_clicks_per_ip_last_1hr"]) == [0]


def test_24h_count_keeps_what_1h_dropped(f2_table: pd.DataFrame) -> None:
    """E sees all four prior ip-1 clicks in the 24h window."""
    e = _row(f2_table, ip=1, ts=_T + timedelta(hours=2, seconds=1))
    assert list(e["f2_clicks_per_ip_last_24hr"]) == [4]


def test_ip_app_count_partitions_by_pair(f2_table: pd.DataFrame) -> None:
    """D is ip 1's fourth click but its FIRST on app 2 -> 0; C is the third
    ip-1 click and the third on app 1 -> 2 prior."""
    d = _row(f2_table, ip=1, ts=_T + timedelta(hours=1))
    c = _row(f2_table, ip=1, ts=_T + timedelta(minutes=30))
    assert list(d["f2_clicks_per_ip_app_last_1hr"]) == [0]
    assert list(c["f2_clicks_per_ip_app_last_1hr"]) == [2]


def test_inter_click_time_first_click_is_null(f2_table: pd.DataFrame) -> None:
    """No prior click -> NULL (LightGBM-native), never a sentinel."""
    f = _row(f2_table, ip=2, ts=_T + timedelta(minutes=15))
    assert f["f2_inter_click_time_seconds"].isna().all()


def test_inter_click_time_same_second_pair(f2_table: pd.DataFrame) -> None:
    """One of A/B is first (NULL), the other follows at gap 0 — the EDA #5
    bot fingerprint. Order within the pair is pinned by row_id."""
    pair = _row(f2_table, ip=1, ts=_T)["f2_inter_click_time_seconds"]
    assert pair.isna().sum() == 1
    assert (pair == 0).sum() == 1


def test_inter_click_time_known_gaps(f2_table: pd.DataFrame) -> None:
    c = _row(f2_table, ip=1, ts=_T + timedelta(minutes=30))
    e = _row(f2_table, ip=1, ts=_T + timedelta(hours=2, seconds=1))
    assert list(c["f2_inter_click_time_seconds"]) == [1800]
    assert list(e["f2_inter_click_time_seconds"]) == [3601]


def test_std_inter_arrival_needs_two_prior_gaps(f2_table: pd.DataFrame) -> None:
    """C has one usable prior gap (B's 0; A's is NULL) -> stddev undefined -> NULL."""
    c = _row(f2_table, ip=1, ts=_T + timedelta(minutes=30))
    assert c["f2_ip_click_std_inter_arrival"].isna().all()


def test_std_inter_arrival_hand_computed(f2_table: pd.DataFrame) -> None:
    """D's prior gaps: {0 (B), 1800 (C)} -> stddev_samp = 1800/sqrt(2).
    E's prior gaps: {0, 1800, 1800} -> stddev_samp = sqrt(1_080_000)."""
    d = _row(f2_table, ip=1, ts=_T + timedelta(hours=1))
    e = _row(f2_table, ip=1, ts=_T + timedelta(hours=2, seconds=1))
    assert d["f2_ip_click_std_inter_arrival"].iloc[0] == pytest.approx(1800 / np.sqrt(2))
    assert e["f2_ip_click_std_inter_arrival"].iloc[0] == pytest.approx(np.sqrt(1_080_000))


def test_std_inter_arrival_uniform_burst_is_zero(f2_table: pd.DataFrame) -> None:
    """G11's ten prior gaps are all exactly 10s -> stddev 0 — metronomic
    pacing, the opposite of human jitter."""
    g11 = _row(f2_table, ip=3, ts=_T + timedelta(hours=3, seconds=110))
    assert g11["f2_ip_click_std_inter_arrival"].iloc[0] == pytest.approx(0.0)


def test_burst_score_fires_only_above_both_thresholds(f2_table: pd.DataFrame) -> None:
    """G11 is the only row with c1h > 10 AND gap < 60: eleven prior clicks
    in the hour, 10s gap. G10 sits exactly at c1h == 10 — not strictly
    greater, no fire. Everything else (including NULL-gap first clicks)
    must be 0."""
    burst_rows = f2_table[f2_table["f2_burst_score"] == 1]
    assert len(burst_rows) == 1
    g11 = _row(f2_table, ip=3, ts=_T + timedelta(hours=3, seconds=110))
    assert g11["f2_burst_score"].iloc[0] == 1
    g10 = _row(f2_table, ip=3, ts=_T + timedelta(hours=3, seconds=100))
    assert g10["f2_burst_score"].iloc[0] == 0


def test_burst_score_null_gap_is_zero_not_null(f2_table: pd.DataFrame) -> None:
    """A first click can't be a burst: NULL gap -> score 0."""
    f = _row(f2_table, ip=2, ts=_T + timedelta(minutes=15))
    assert f["f2_burst_score"].iloc[0] == 0


def test_f2_features_carry_documentation() -> None:
    for feat in F2_FEATURES:
        assert feat.description, f"{feat.name} has no description"
        assert feat.output_dtype, f"{feat.name} has no output_dtype"
