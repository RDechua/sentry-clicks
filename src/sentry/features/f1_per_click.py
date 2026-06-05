"""F1 — per-click features (Task 2.3).

Direct column values and simple transformations of a single click. No
windows, no aggregates, no other rows involved — which makes F1 trivially
leakage-free: every value derives from the row itself.

The SQL for each feature lives in its own file under `sql/02_features/`
(build-guide convention: the SQL is reviewable documentation). This module
loads those files and registers one `SqlFeature` per file.

Notably absent: raw `ip`. Millions of distinct values, most appearing a
handful of times — as a categorical it teaches the model to memorize
training-set IPs that won't recur. IPs enter the model only through derived
features (F2 velocity, F3 aggregates) and through the bounded interaction
pairs below. Rationale in decisions.md (Task 2.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from sentry.features.pipeline import SqlFeature, register_feature

#: Repo-level sql/ directory; __file__ is src/sentry/features/f1_per_click.py.
_SQL_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "sql" / "02_features"


def _load_sql(feature_name: str) -> str:
    return (_SQL_DIR / f"{feature_name}.sql").read_text()


#: (name, output_dtype, description) for each F1 feature; SQL loaded by name.
_F1_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    ("f1_app_id", "int32", "App being advertised (raw categorical code)."),
    ("f1_channel_id", "int32", "Publisher channel serving the ad (raw categorical code)."),
    ("f1_device_id", "int32", "Device type code; device=0 converts at ~60x baseline."),
    ("f1_os_id", "int32", "OS version code (raw categorical code)."),
    ("f1_hour_of_day", "int8", "Hour 0-23 of the click; raw int — trees handle cycles."),
    ("f1_day_of_week", "int8", "ISO day of week, Monday=1..Sunday=7."),
    ("f1_ip_app_interaction", "string", "(ip, app) pair as one categorical fingerprint."),
    ("f1_ip_device_interaction", "string", "(ip, device) pair as one categorical fingerprint."),
)

#: All F1 features, registered in the global registry at import time.
F1_FEATURES: Final[tuple[SqlFeature, ...]] = tuple(
    register_feature(
        SqlFeature(name=name, sql=_load_sql(name), output_dtype=dtype, description=desc)
    )
    for name, dtype, desc in _F1_SPECS
)
