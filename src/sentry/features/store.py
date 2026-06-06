"""Versioned feature store (Task 2.5).

Computed feature tables are expensive (at full scale, hours); models iterate
cheaply on top of them. The store persists one Parquet file per split under
a semver-ish version directory, with one human-readable `metadata.json` per
version:

    artifacts/features/
      v0.1.0/
        metadata.json
        train.parquet
        val.parquet

Versions are IMMUTABLE by default: re-saving an existing version+split
raises. A "versioned" store that silently overwrites is just a cache with
extra steps — if a feature definition changes, that's a new version
(v0.1.0 = F1+F2; v0.2.0 = F1+F2+F3; the metadata says exactly what's
inside either way).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
import structlog

from sentry.data.splits import SPLIT_NAMES

logger = structlog.get_logger(__name__)

#: Default on-disk location (gitignored, like every artifact).
DEFAULT_STORE_ROOT: Final[Path] = Path("artifacts/features")

_VERSION_RE: Final[re.Pattern[str]] = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

#: Feature columns are identified by the project-wide naming convention
#: (f1_*, f2_*, ...) — the metadata lists them so a reader knows what a
#: version contains without opening the Parquet.
_FEATURE_COLUMN_RE: Final[re.Pattern[str]] = re.compile(r"^f\d+_")


def _version_key(version: str) -> tuple[int, int, int]:
    match = _VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"version must look like v0.1.0, got {version!r}")
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))


class FeatureStore:
    """Save/load feature tables as versioned, split-keyed Parquet files."""

    def __init__(self, root: Path | str = DEFAULT_STORE_ROOT) -> None:
        self.root = Path(root)

    def save(
        self,
        table: pd.DataFrame,
        version: str,
        split_name: str,
        source: str,
    ) -> Path:
        """Persist one split's feature table under `version`.

        Parameters
        ----------
        table:
            The feature table (base columns + f*_ feature columns).
        version:
            Semver-ish version string (`v0.1.0`). New feature set = new version.
        split_name:
            One of `train` / `val` / `test`.
        source:
            Human-readable provenance (e.g. `train_sample.csv@100k`), recorded
            in the metadata so a reader can trace what produced the table.

        Returns
        -------
        Path to the written Parquet file.
        """
        _version_key(version)  # validates format
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {split_name!r}; expected one of {SPLIT_NAMES}")

        version_dir = self.root / version
        parquet_path = version_dir / f"{split_name}.parquet"
        if parquet_path.exists():
            raise FileExistsError(
                f"{version}/{split_name} already exists at {parquet_path} — versions are "
                f"immutable; bump the version if the feature set changed"
            )
        version_dir.mkdir(parents=True, exist_ok=True)

        table.to_parquet(parquet_path, index=False)

        digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        metadata = self._read_metadata(version_dir) or {
            "version": version,
            "features": [c for c in table.columns if _FEATURE_COLUMN_RE.match(c)],
            "splits": {},
        }
        metadata["splits"][split_name] = {
            "rows": len(table),
            "source": source,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "file_sha256": digest,
        }
        (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        logger.info(
            "features_saved",
            version=version,
            split=split_name,
            rows=len(table),
            path=str(parquet_path),
        )
        return parquet_path

    def load(self, version: str, split_name: str) -> pd.DataFrame:
        """Load one split's feature table from `version`."""
        parquet_path = self.root / version / f"{split_name}.parquet"
        if not parquet_path.exists():
            available = ", ".join(self.list_versions()) or "none"
            raise FileNotFoundError(
                f"no saved features for {version}/{split_name} under {self.root} "
                f"(available versions: {available})"
            )
        table = pd.read_parquet(parquet_path)
        logger.info("features_loaded", version=version, split=split_name, rows=len(table))
        return table

    def list_versions(self) -> list[str]:
        """All saved versions, numerically ordered (v0.10.0 after v0.2.0)."""
        if not self.root.exists():
            return []
        versions = [p.name for p in self.root.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)]
        return sorted(versions, key=_version_key)

    @staticmethod
    def _read_metadata(version_dir: Path) -> dict[str, Any] | None:
        meta_path = version_dir / "metadata.json"
        if not meta_path.exists():
            return None
        loaded: dict[str, Any] = json.loads(meta_path.read_text())
        return loaded
