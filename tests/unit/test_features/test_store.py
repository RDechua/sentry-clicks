"""Tests for the versioned feature store (Task 2.5).

The properties pinned: save/load round-trips the exact table (hash-verified
per the AC), versions are immutable by default, metadata is human-readable
JSON with the fields the build guide names, and bad inputs fail loud.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sentry.features.store import FeatureStore


@pytest.fixture
def table(tiny_sample_data: pd.DataFrame) -> pd.DataFrame:
    """A plausible feature table: base columns plus two feature columns."""
    df = tiny_sample_data.copy()
    df["row_id"] = range(1, len(df) + 1)
    df["f1_app_id"] = df["app"]
    df["f2_inter_click_time_seconds"] = 12.5
    return df


@pytest.fixture
def store(tmp_path: Path) -> FeatureStore:
    return FeatureStore(root=tmp_path / "features")


def test_round_trip_is_exact(store: FeatureStore, table: pd.DataFrame) -> None:
    """AC: loading produces the same table that was saved, hash-verified."""
    store.save(table, version="v0.1.0", split_name="train", source="fixture")

    loaded = store.load(version="v0.1.0", split_name="train")

    original_hash = pd.util.hash_pandas_object(table, index=False)
    loaded_hash = pd.util.hash_pandas_object(loaded, index=False)
    assert list(loaded.columns) == list(table.columns)
    assert (original_hash.to_numpy() == loaded_hash.to_numpy()).all()


def test_metadata_is_human_readable(store: FeatureStore, table: pd.DataFrame) -> None:
    """AC: metadata.json exists, parses, and carries the build-guide fields."""
    store.save(table, version="v0.1.0", split_name="train", source="train_sample.csv")
    store.save(table, version="v0.1.0", split_name="val", source="train_sample.csv")

    meta_path = store.root / "v0.1.0" / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())

    assert meta["version"] == "v0.1.0"
    assert "f1_app_id" in meta["features"]
    assert "f2_inter_click_time_seconds" in meta["features"]
    assert meta["splits"]["train"]["rows"] == len(table)
    assert meta["splits"]["train"]["source"] == "train_sample.csv"
    assert meta["splits"]["train"]["created_at"]  # ISO timestamp present
    assert meta["splits"]["train"]["file_sha256"]
    assert set(meta["splits"]) == {"train", "val"}


def test_versions_are_immutable_by_default(store: FeatureStore, table: pd.DataFrame) -> None:
    """Re-saving an existing version+split must raise, not silently replace."""
    store.save(table, version="v0.1.0", split_name="train", source="fixture")
    with pytest.raises(FileExistsError, match=r"v0\.1\.0"):
        store.save(table, version="v0.1.0", split_name="train", source="fixture")


def test_list_versions_sorted(store: FeatureStore, table: pd.DataFrame) -> None:
    for version in ("v0.2.0", "v0.1.0", "v0.10.0"):
        store.save(table, version=version, split_name="train", source="fixture")

    # Numeric-aware ordering: v0.10.0 sorts after v0.2.0.
    assert store.list_versions() == ["v0.1.0", "v0.2.0", "v0.10.0"]


def test_load_missing_version_raises(store: FeatureStore) -> None:
    with pytest.raises(FileNotFoundError, match=r"v9\.9\.9"):
        store.load(version="v9.9.9", split_name="train")


def test_bad_version_string_raises(store: FeatureStore, table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="version"):
        store.save(table, version="latest", split_name="train", source="fixture")


def test_bad_split_name_raises(store: FeatureStore, table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="split"):
        store.save(table, version="v0.1.0", split_name="vall", source="fixture")


def test_add_parquet_registers_external_file(
    store: FeatureStore, table: pd.DataFrame, tmp_path: Path
) -> None:
    """The full-scale path: DuckDB-written parquet moves into the store
    with the same metadata and immutability as save()."""
    external = tmp_path / "materialized.parquet"
    table.to_parquet(external, index=False)

    dest = store.add_parquet(external, version="v0.3.0", split_name="train", source="full@10pct")

    assert not external.exists(), "file must MOVE into the store, not copy"
    loaded = store.load(version="v0.3.0", split_name="train")
    assert len(loaded) == len(table)

    meta = json.loads((store.root / "v0.3.0" / "metadata.json").read_text())
    assert meta["splits"]["train"]["rows"] == len(table)
    assert "f1_app_id" in meta["features"]
    assert dest.exists()

    # Same immutability as save().
    table.to_parquet(external, index=False)
    with pytest.raises(FileExistsError, match=r"v0\.3\.0"):
        store.add_parquet(external, version="v0.3.0", split_name="train", source="x")
