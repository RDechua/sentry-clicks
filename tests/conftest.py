"""Shared pytest fixtures for Sentry-Clicks tests.

The `tiny_sample_data` fixture is the canonical test substrate for every feature
test. It is hand-crafted (40 rows) to exercise the specific edge cases the
feature engineers care about — see the per-block comments inline below for
the intent behind each block.

Block layout (40 rows): L1 (1-3), F2 burst (4-8), F2 slow (9-15), L2 (16-25),
F3 skew (26-30), Borderline (31-35), Edge cases (36-40).

Design intent recorded in `docs/decisions.md` (2026-05-25: Test framework).
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def train_sample_path() -> Path:
    """The real Kaggle 100k sample, or skip.

    Container-internal path: /data is mounted read-only from DATA_DIR on the
    host (see docker-compose.yml). If DATA_DIR isn't set, compose falls back
    to the empty repo ./data — the file won't exist and integration tests
    that request this fixture are skipped.
    """
    path = Path("/data/train_sample.csv")
    if not path.exists():
        pytest.skip(f"{path} not found — download the TalkingData files")
    return path


# Anchor inside the TalkingData dataset's actual date range (2017-11-06 to
# 2017-11-09) so any downstream date-bucketing logic doesn't have to special-case
# fixture data.
_T0 = datetime(2017, 11, 7, 9, 0, 0)


def _at(*, days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0) -> datetime:
    """Click time at _T0 plus the given offset.

    Keyword-only args (so `_at(hours=2, minutes=15)` reads naturally).
    """
    return _T0 + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


_COLUMNS = [
    "ip",
    "app",
    "device",
    "os",
    "channel",
    "click_time",
    "attributed_time",
    "is_attributed",
]


# Each row is (ip, app, device, os, channel, click_time, attributed_time, is_attributed).
# pd.NaT marks rows where the click did not convert (attributed_time is null).
_ROWS = [
    # ===== L1 — Clearly legitimate baseline (rows 1-3) =====
    # Single clicks from distinct IPs/apps, all converted ~30s after click.
    # Establishes the "no fraud signal" control group for F2/F3 tests.
    (100, 12, 1, 13, 466, _T0, _at(seconds=30), 1),
    (101, 15, 1, 19, 213, _at(seconds=5), _at(seconds=45), 1),
    (102, 18, 2, 17, 280, _at(seconds=12), _at(seconds=58), 1),
    # ===== F2 burst fraud (rows 4-8) =====
    # ip=999 fires 5 clicks in 10s across different apps. Classic click-farm
    # burst → must be flagged by F2 (velocity / rolling-window counts).
    (999, 12, 1, 13, 466, _at(seconds=20), pd.NaT, 0),
    (999, 15, 1, 13, 213, _at(seconds=22), pd.NaT, 0),
    (999, 18, 1, 13, 280, _at(seconds=24), pd.NaT, 0),
    (999, 12, 1, 13, 466, _at(seconds=27), pd.NaT, 0),
    (999, 15, 1, 13, 213, _at(seconds=30), pd.NaT, 0),
    # ===== F2 slow fraud (rows 9-15) =====
    # ip=888 fires 7 clicks on the SAME app over an hour. Sub-burst rate, but
    # the (ip, app) combo's zero conversion rate is also the signal for F3.
    (888, 42, 1, 13, 100, _at(minutes=5), pd.NaT, 0),
    (888, 42, 1, 13, 100, _at(minutes=15), pd.NaT, 0),
    (888, 42, 1, 13, 100, _at(minutes=25), pd.NaT, 0),
    (888, 42, 1, 13, 100, _at(minutes=33), pd.NaT, 0),
    (888, 42, 1, 13, 100, _at(minutes=42), pd.NaT, 0),
    (888, 42, 1, 13, 100, _at(minutes=51), pd.NaT, 0),
    (888, 42, 1, 13, 100, _at(minutes=58), pd.NaT, 0),
    # ===== L2 mixed legit (rows 16-25) =====
    # ip=100 (same as row 1) — 10 clicks across an hour, ~3 conversions.
    # Engaged legitimate user. Models should NOT flag these as fraud.
    (100, 12, 1, 13, 466, _at(minutes=10), _at(minutes=12), 1),
    (100, 15, 1, 13, 213, _at(minutes=15), pd.NaT, 0),
    (100, 18, 1, 13, 280, _at(minutes=20), pd.NaT, 0),
    (100, 12, 1, 13, 466, _at(minutes=24), _at(minutes=26), 1),
    (100, 15, 1, 13, 213, _at(minutes=27), pd.NaT, 0),
    (100, 22, 1, 13, 105, _at(minutes=33), pd.NaT, 0),
    (100, 18, 1, 13, 280, _at(minutes=38), _at(minutes=40), 1),
    (100, 12, 1, 13, 466, _at(minutes=44), pd.NaT, 0),
    (100, 22, 1, 13, 105, _at(minutes=51), pd.NaT, 0),
    (100, 15, 1, 13, 213, _at(minutes=58), pd.NaT, 0),
    # ===== F3 skewed conversion (rows 26-30) =====
    # ip=777: 4 clicks on app=99 (0 conversions), 1 click on app=42 (1 conversion).
    # Per-(ip, app) conversion rate varies wildly per app — the F3 signal.
    (777, 99, 1, 13, 500, _at(hours=1), pd.NaT, 0),
    (777, 99, 1, 13, 500, _at(hours=1, minutes=15), pd.NaT, 0),
    (777, 99, 1, 13, 500, _at(hours=1, minutes=30), pd.NaT, 0),
    (777, 99, 1, 13, 500, _at(hours=1, minutes=45), pd.NaT, 0),
    (777, 42, 1, 13, 100, _at(hours=2), _at(hours=2, seconds=45), 1),
    # ===== Borderline (rows 31-35) =====
    # Five distinct IPs, single click each across the day, mixed labels.
    # Hard to classify from a single click — used for threshold/calibration tests.
    (200, 12, 1, 13, 466, _at(hours=3), pd.NaT, 0),
    (201, 15, 1, 19, 213, _at(hours=4), _at(hours=4, minutes=2), 1),
    (202, 18, 2, 17, 280, _at(hours=5), pd.NaT, 0),
    (203, 22, 1, 13, 105, _at(hours=6), pd.NaT, 0),
    (204, 12, 1, 19, 466, _at(hours=7), _at(hours=7, minutes=1), 1),
    # ===== Edge cases (rows 36-40) =====
    # 36: sub-2s conversion — suspiciously fast even though labeled positive.
    # 37: rare device + os combination (99/99).
    # 38: channel=0 — likely a default/garbage sentinel value.
    # 39-40: ip=1 (very common in the real dataset), no conversions.
    (300, 12, 1, 13, 466, _at(hours=8), _at(hours=8, seconds=1), 1),
    (301, 15, 99, 99, 213, _at(hours=9), pd.NaT, 0),
    (302, 18, 1, 13, 0, _at(hours=10), pd.NaT, 0),
    (1, 12, 1, 13, 466, _at(hours=11), pd.NaT, 0),
    (1, 12, 1, 13, 466, _at(hours=12), pd.NaT, 0),
]


@pytest.fixture(scope="session")
def _tiny_sample_data_base() -> pd.DataFrame:
    """Session-scoped immutable base for `tiny_sample_data`.

    Built once per pytest session. The user-facing function-scope fixture
    returns a `.copy()` of this so individual tests can mutate freely without
    paying the construction cost on every call.
    """
    df = pd.DataFrame(_ROWS, columns=_COLUMNS)
    df["click_time"] = pd.to_datetime(df["click_time"])
    df["attributed_time"] = pd.to_datetime(df["attributed_time"])
    return df


@pytest.fixture
def tiny_sample_data(_tiny_sample_data_base: pd.DataFrame) -> pd.DataFrame:
    """A 40-row hand-crafted DataFrame matching the TalkingData schema.

    Returns a fresh copy on each call (function scope) so tests can mutate freely.
    """
    return _tiny_sample_data_base.copy()
