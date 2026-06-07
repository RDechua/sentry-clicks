"""F3 — aggregate behavioral features (Tasks 3.1/3.2).

Rolling aggregates of an entity's history — including, for the conversion
rates, the LABELS of prior clicks. This is the highest-signal and
highest-leakage-risk feature family in the project: get the window wrong
and the model trains on its own answers.

The discipline, stated once (each SQL file restates its own version):

- frame is strictly prior — `RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND
  INTERVAL 1 MILLISECOND PRECEDING` — so neither the current row nor its
  same-second peers can contribute;
- the current row's label NEVER enters its own feature;
- no prior history → NULL for rates (undefined), 0 for counts (true zero);
- tests pin the canary case (first click with label=1 → NULL, not 1.0)
  and the label-flip property (changing a row's label moves only LATER
  rows' features, never its own).

Full rationale in decisions.md ("Avoiding label leakage in F3").
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from sentry.features.pipeline import SqlFeature, register_feature

#: Repo-level sql/ directory; __file__ is src/sentry/features/f3_aggregates.py.
_SQL_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "sql" / "02_features"


def _load_sql(feature_name: str) -> str:
    return (_SQL_DIR / f"{feature_name}.sql").read_text()


#: (name, output_dtype, description) for each F3 feature; SQL loaded by name.
_F3_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "f3_ip_conversion_rate_24hr",
        "float64",
        "This IP's strictly-prior 24h conversion rate; NULL with no history.",
    ),
    (
        "f3_app_conversion_rate_24hr",
        "float64",
        "The app's strictly-prior 24h conversion rate across all IPs; NULL with no history.",
    ),
    (
        "f3_ip_distinct_apps_24hr",
        "int64",
        "Distinct apps this IP clicked in the strictly-prior 24h; fraud spreads thin.",
    ),
    (
        "f3_ip_distinct_devices_24hr",
        "int64",
        "Distinct device codes behind this IP in the strictly-prior 24h.",
    ),
    (
        "f3_ip_distinct_oses_24hr",
        "int64",
        "Distinct OS codes behind this IP in the strictly-prior 24h.",
    ),
    (
        "f3_ip_app_conversion_rate_24hr",
        "float64",
        "The (ip, app) pair's strictly-prior 24h conversion rate — the expected "
        "dominant feature; NULL with no pair history.",
    ),
    (
        "f3_ip_app_clicks_24hr",
        "int64",
        "Denominator companion to the pair rate: strictly-prior 24h pair clicks, "
        "so the model can discount thin-history rates.",
    ),
    (
        "f3_app_distinct_ips_24hr",
        "int64",
        "Distinct IPs clicking this app in the strictly-prior 24h (app-side degree).",
    ),
    # Long-memory variants (post-density-gate, 2026-06-07): the 24h cap
    # forgets days 1-2; these keep all strictly-prior history.
    (
        "f3_ip_conversion_rate_alltime",
        "float64",
        "This IP's all-time strictly-prior conversion rate; NULL with no history.",
    ),
    (
        "f3_ip_clicks_alltime",
        "int64",
        "This IP's all-time strictly-prior click count (denominator companion).",
    ),
    (
        "f3_app_conversion_rate_alltime",
        "float64",
        "The app's all-time strictly-prior conversion rate; NULL with no history.",
    ),
    (
        "f3_app_clicks_alltime",
        "int64",
        "The app's all-time strictly-prior click count (denominator companion).",
    ),
    (
        "f3_ip_app_conversion_rate_alltime",
        "float64",
        "The pair's all-time strictly-prior conversion rate; NULL with no history.",
    ),
    (
        "f3_ip_app_clicks_alltime",
        "int64",
        "The pair's all-time strictly-prior click count (denominator companion).",
    ),
)

#: All F3 features, registered in the global registry at import time.
F3_FEATURES: Final[tuple[SqlFeature, ...]] = tuple(
    register_feature(
        SqlFeature(name=name, sql=_load_sql(name), output_dtype=dtype, description=desc)
    )
    for name, dtype, desc in _F3_SPECS
)
