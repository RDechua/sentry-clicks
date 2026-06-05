"""Feature pipeline framework (Task 2.2).

Two feature kinds, one pipeline:

- `SqlFeature` — a SQL query template executed against a split view
  (`{source}` placeholder). Must return exactly `(row_id, value)` and cover
  every source row; the pipeline joins it back on `row_id`, so result order
  is irrelevant. For aggregations, joins, and window functions.
- `PythonFeature` — a function over the accumulated DataFrame (base columns
  plus every already-computed feature). For cross-row statistics and logic
  that is awkward in SQL.

The SQL-vs-Python rule (decisions.md, Task 2.2): if it can be a window
function or join, it's SQL; Python is for what SQL can't express cleanly.

Leakage discipline composes from two independent layers: the pipeline is
split-agnostic (callers pass `clicks_train` / `clicks_val` as `source`, so
cross-split contamination is impossible by construction), and each feature's
own window must be strictly prior per CLAUDE.md §3.4 (owned by the feature
definitions in Tasks 2.3/2.4, tested there).

Dependencies are resolved by topological sort at construction time, so a
bad graph fails when the pipeline is built, not halfway through an hour-long
feature run. Intermediate results live in the accumulating frame — each
feature computes exactly once, and later features read earlier ones by
column name.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

import duckdb
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

#: Column name SQL features must use for the per-row value they return.
SQL_VALUE_COLUMN: Final[str] = "value"


@dataclass(frozen=True)
class SqlFeature:
    """A feature computed by SQL against the split view.

    `sql` must contain a `{source}` placeholder for the view name and return
    exactly two columns: `row_id` and `value`, one row per source row.
    """

    name: str
    sql: str
    output_dtype: str
    description: str
    dependencies: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class PythonFeature:
    """A feature computed by a Python function over the accumulated frame.

    `fn` receives the base columns plus every feature computed before this
    one (declare what you read in `dependencies`) and returns a Series
    aligned to the frame's index.
    """

    name: str
    fn: Callable[[pd.DataFrame], pd.Series]
    output_dtype: str
    description: str
    dependencies: tuple[str, ...] = field(default=())


Feature = SqlFeature | PythonFeature

#: Global registry. Feature modules (f1_*, f2_*, ...) register at import time;
#: callers build a FeaturePipeline from the registry or any explicit subset.
FEATURE_REGISTRY: dict[str, Feature] = {}


def register_feature(feature: Feature) -> Feature:
    """Add a feature to the global registry. Duplicate names fail loud."""
    if feature.name in FEATURE_REGISTRY:
        raise ValueError(f"duplicate feature name in registry: {feature.name!r}")
    FEATURE_REGISTRY[feature.name] = feature
    return feature


def python_feature(
    name: str,
    output_dtype: str,
    description: str,
    dependencies: tuple[str, ...] = (),
) -> Callable[[Callable[[pd.DataFrame], pd.Series]], Callable[[pd.DataFrame], pd.Series]]:
    """Decorator: define and register a PythonFeature in one step."""

    def decorate(fn: Callable[[pd.DataFrame], pd.Series]) -> Callable[[pd.DataFrame], pd.Series]:
        register_feature(
            PythonFeature(
                name=name,
                fn=fn,
                output_dtype=output_dtype,
                description=description,
                dependencies=dependencies,
            )
        )
        return fn

    return decorate


def _topological_order(features: Sequence[Feature]) -> list[Feature]:
    """Kahn's algorithm, stable: among ready features, input order is kept.

    Dependencies may name other features only — base columns are always
    available and never need declaring. Unknown names and cycles raise.
    """
    names = [f.name for f in features]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate feature name(s): {dupes}")
    by_name = {f.name: f for f in features}

    for f in features:
        unknown = [d for d in f.dependencies if d not in by_name]
        if unknown:
            raise ValueError(f"feature {f.name!r} depends on unknown feature(s): {unknown}")

    remaining = {f.name: set(f.dependencies) for f in features}
    ordered: list[Feature] = []
    while remaining:
        ready = {name for name in remaining if not remaining[name]}
        if not ready:
            raise ValueError(f"cycle among features: {sorted(remaining)}")
        # Stable: process ready features in original input order.
        for name in (n for n in names if n in ready):
            ordered.append(by_name[name])
            del remaining[name]
            for deps in remaining.values():
                deps.discard(name)
    return ordered


class FeaturePipeline:
    """Computes a feature table from a source view, dependency-ordered."""

    def __init__(self, features: Sequence[Feature]) -> None:
        self._ordered = _topological_order(features)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature column names in computation order."""
        return tuple(f.name for f in self._ordered)

    def compute(
        self, conn: duckdb.DuckDBPyConnection, source: str = "clicks_train"
    ) -> pd.DataFrame:
        """Compute every feature against `source` and return the full table.

        The returned frame holds the base columns plus one column per
        feature, with `row_id` as a regular column. `source` should be a
        split view (`clicks_train` / `clicks_val`) — passing the raw table
        is for tests only.
        """
        # `source` is a code-owned view/table name, never user input.
        table = conn.execute(f"SELECT * FROM {source} ORDER BY row_id").fetch_df()
        table = table.set_index("row_id", drop=False)

        for feat in self._ordered:
            if isinstance(feat, SqlFeature):
                result = conn.execute(feat.sql.format(source=source)).fetch_df()
                if list(result.columns) != ["row_id", SQL_VALUE_COLUMN]:
                    raise ValueError(
                        f"SQL feature {feat.name!r} must return exactly "
                        f"(row_id, {SQL_VALUE_COLUMN}); got {list(result.columns)}"
                    )
                series = result.set_index("row_id")[SQL_VALUE_COLUMN]
                if len(series) != len(table) or not series.index.isin(table.index).all():
                    raise ValueError(
                        f"SQL feature {feat.name!r} returned {len(series)} rows "
                        f"for {len(table)} source rows — every source row needs a value"
                    )
                table[feat.name] = series
            else:
                series = feat.fn(table)
                if len(series) != len(table):
                    raise ValueError(
                        f"python feature {feat.name!r} returned {len(series)} values "
                        f"for {len(table)} rows"
                    )
                table[feat.name] = series

        logger.info(
            "features_computed",
            source=source,
            n_rows=len(table),
            n_features=len(self._ordered),
            features=list(self.feature_names),
        )
        return table.reset_index(drop=True)
