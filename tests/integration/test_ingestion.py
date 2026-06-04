"""Integration test for ingesting the real Kaggle `train_sample.csv`.

Skipped automatically when the file isn't mounted at `/data/train_sample.csv`
(the path inside the container per docker-compose's data mount). Once you've
downloaded the TalkingData files and pointed DATA_DIR at the directory
containing `train_sample.csv`, this test runs and exercises the full
ingest → validate → query loop on real production-shaped data.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from sentry.data._db import fetch_one
from sentry.data.ingestion import ingest_csv_to_duckdb
from sentry.data.schema import TABLE_NAME
from sentry.data.validation import validate_ingestion

# Container-internal path. /data is mounted read-only from DATA_DIR on the host;
# see docker-compose.yml. If DATA_DIR isn't set, compose falls back to the empty
# repo ./data — in which case the file won't exist and this test is skipped.
TRAIN_SAMPLE_PATH = Path("/data/train_sample.csv")


@pytest.mark.integration
def test_ingest_real_train_sample(tmp_path: Path) -> None:
    """Round-trip the Kaggle 100k-row sample and verify the headline stats."""
    if not TRAIN_SAMPLE_PATH.exists():
        pytest.skip(f"{TRAIN_SAMPLE_PATH} not found — download the TalkingData files")

    db_path = tmp_path / "real.duckdb"

    row_count = ingest_csv_to_duckdb(TRAIN_SAMPLE_PATH, db_path)
    assert row_count == 100_000, f"expected 100k rows, got {row_count}"

    result = validate_ingestion(db_path)
    assert result.ok, f"validation failed: {[(c.name, c.detail) for c in result.errors]}"

    # The build-guide-specified spot check.
    with duckdb.connect(str(db_path), read_only=True) as conn:
        n, mean_label = fetch_one(conn, f"SELECT COUNT(*), AVG(is_attributed) FROM {TABLE_NAME}")
    assert n == 100_000
    # TalkingData's documented positive rate is ~0.2%; the 100k sample's actual
    # value is around 0.00227. The bounds match validate_ingestion's defaults.
    assert 0.001 <= mean_label <= 0.01
