"""F2 — velocity and burst features (Task 2.4).

Trailing-window behavior of the IP behind a click. These target Tier 1
adversaries: click farms (dense bursts → 1h window) and paced botnets
(steady drip → 24h window). Both windows exist because the two patterns
are invisible to each other's window (decisions.md, Task 2.4).

Every window uses the strictly-prior frame from CLAUDE.md §3.4 —
`RANGE BETWEEN INTERVAL <X> PRECEDING AND INTERVAL 1 MILLISECOND
PRECEDING` — which excludes both the current row and its same-second
peers. The empirically-verified traps and equivalences are documented in
the SQL files themselves.

`f2_burst_score` is the one PythonFeature: a composite over two computed
features, which is exactly the case the SQL-vs-Python rule assigns to
Python (decisions.md, Task 2.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from sentry.features.pipeline import (
    FEATURE_REGISTRY,
    Feature,
    SqlFeature,
    python_feature,
    register_feature,
)

#: Repo-level sql/ directory; __file__ is src/sentry/features/f2_velocity.py.
_SQL_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "sql" / "02_features"


def _load_sql(feature_name: str) -> str:
    return (_SQL_DIR / f"{feature_name}.sql").read_text()


#: Burst thresholds, pinned from train-split EDA (2026-06-05): the 1h count's
#: p99 is 10 and a <60s gap sits in the fastest ~2% of inter-click gaps. At
#: `count > 10 AND gap < 60`, 127 train rows match — with zero conversions
#: against a 0.237% baseline. Real operating thresholds come from Task 5.2;
#: these only define the feature.
BURST_MIN_CLICKS_1H: Final[int] = 10
BURST_MAX_GAP_SECONDS: Final[int] = 60

#: (name, output_dtype, description) for the SQL-computed F2 features.
_F2_SQL_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "f2_clicks_per_ip_last_1hr",
        "int32",
        "Clicks from this IP in the trailing hour (strictly prior). Click-farm bursts.",
    ),
    (
        "f2_clicks_per_ip_last_24hr",
        "int32",
        "Clicks from this IP in the trailing 24h (strictly prior). Paced botnets.",
    ),
    (
        "f2_clicks_per_ip_app_last_1hr",
        "int32",
        "Clicks from this (ip, app) pair in the trailing hour (strictly prior).",
    ),
    (
        "f2_inter_click_time_seconds",
        "float64",
        "Seconds since this IP's previous click; NULL for an IP's first click; 0 = bot.",
    ),
    (
        "f2_ip_click_std_inter_arrival",
        "float64",
        "Stddev of this IP's prior inter-arrival gaps over 24h; low = metronomic pacing.",
    ),
)

_F2_SQL_FEATURES: Final[tuple[SqlFeature, ...]] = tuple(
    register_feature(
        SqlFeature(name=name, sql=_load_sql(name), output_dtype=dtype, description=desc)
    )
    for name, dtype, desc in _F2_SQL_SPECS
)


@python_feature(
    name="f2_burst_score",
    output_dtype="int8",
    description=(
        f"1 if >{BURST_MIN_CLICKS_1H} clicks from the IP in the trailing hour AND "
        f"<{BURST_MAX_GAP_SECONDS}s since its previous click; else 0. NULL gap -> 0 "
        "(a first click cannot be a burst)."
    ),
    dependencies=("f2_clicks_per_ip_last_1hr", "f2_inter_click_time_seconds"),
)
def _burst_score(df: pd.DataFrame) -> pd.Series:
    fast = df["f2_inter_click_time_seconds"] < BURST_MAX_GAP_SECONDS  # NaN -> False
    dense = df["f2_clicks_per_ip_last_1hr"] > BURST_MIN_CLICKS_1H
    return (fast & dense).astype("int8")


#: All F2 features. The decorator registered the burst score; pull the
#: instance from the registry. (Burst listed last for readability; the
#: pipeline topo-sorts by its declared dependencies regardless.)
F2_FEATURES: Final[tuple[Feature, ...]] = (
    *_F2_SQL_FEATURES,
    FEATURE_REGISTRY["f2_burst_score"],
)
