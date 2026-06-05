"""Tests for `sentry.audit.logger.log_event` and `log_events`.

Each test uses a temp DB path so they don't touch the real `artifacts/`
audit log. Verify: DB+table created lazily, append semantics, round-trip
through the DB preserves fields, JSON column round-trips the feature list.
(Single/batch row parity needs no test — both writers serialize via
`_entry_row` and write via `_write_rows`, so parity is structural.)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import pytest

from sentry.audit.logger import AUDIT_TABLE_NAME, log_event, log_events
from sentry.audit.schema import Action, AuditLogEntry, FeatureContribution
from sentry.data._db import fetch_one


def _entry(case_id: str = "click-1", action: Action = Action.AUTO_BLOCK) -> AuditLogEntry:
    return AuditLogEntry(
        event_timestamp=datetime(2017, 11, 7, 9, 0, 30),
        case_id=case_id,
        click_timestamp=datetime(2017, 11, 7, 9, 0, 0),
        model_version="lgbm-v0.1.0",
        policy_version="policy-v0.1.0",
        raw_score=0.92,
        calibrated_score=0.81,
        threshold_block=0.80,
        threshold_review=0.40,
        action=action,
        top_features=[
            FeatureContribution(feature_name="ip_count_1h", value=120, shap_contribution=0.31),
            FeatureContribution(feature_name="app", value=12, shap_contribution=-0.04),
        ],
    )


def test_log_event_creates_db_and_table(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.duckdb"
    assert not db_path.exists()

    log_event(_entry(), db_path=db_path)

    assert db_path.exists()
    with duckdb.connect(str(db_path), read_only=True) as conn:
        n = fetch_one(conn, f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}")[0]
    assert n == 1


def test_log_event_appends_not_replaces(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.duckdb"
    log_event(_entry(case_id="click-1", action=Action.AUTO_BLOCK), db_path=db_path)
    log_event(_entry(case_id="click-2", action=Action.HUMAN_REVIEW), db_path=db_path)
    log_event(_entry(case_id="click-3", action=Action.ALLOW), db_path=db_path)

    with duckdb.connect(str(db_path), read_only=True) as conn:
        n = fetch_one(conn, f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}")[0]
        actions = conn.execute(f"SELECT action FROM {AUDIT_TABLE_NAME} ORDER BY case_id").fetchall()
    assert n == 3
    assert [row[0] for row in actions] == ["AUTO_BLOCK", "HUMAN_REVIEW", "ALLOW"]


def test_logged_row_round_trips_field_values(tmp_path: Path) -> None:
    """The whole point of the log: a reader can reproduce the decision."""
    db_path = tmp_path / "audit.duckdb"
    entry = _entry(case_id="click-roundtrip", action=Action.AUTO_BLOCK)
    log_event(entry, db_path=db_path)

    with duckdb.connect(str(db_path), read_only=True) as conn:
        row = fetch_one(
            conn,
            f"""
            SELECT case_id, model_version, raw_score, calibrated_score,
                   threshold_block, threshold_review, action, top_features
            FROM {AUDIT_TABLE_NAME} WHERE case_id = ?
            """,
            ["click-roundtrip"],
        )

    case_id, model_v, raw, cal, t_block, t_review, action, features_json = row
    assert case_id == "click-roundtrip"
    assert model_v == "lgbm-v0.1.0"
    assert raw == pytest.approx(0.92)
    assert cal == pytest.approx(0.81)
    assert t_block == pytest.approx(0.80)
    assert t_review == pytest.approx(0.40)
    assert action == "AUTO_BLOCK"
    # top_features stored as JSON column — round-trip parse.
    features = json.loads(features_json)
    assert len(features) == 2
    assert features[0]["feature_name"] == "ip_count_1h"
    assert features[0]["shap_contribution"] == pytest.approx(0.31)


def test_db_path_parent_created_if_missing(tmp_path: Path) -> None:
    """Logger creates the parent dir; a fresh artifacts/ should not be a barrier."""
    db_path = tmp_path / "nested" / "subdir" / "audit.duckdb"
    log_event(_entry(), db_path=db_path)
    assert db_path.exists()


def test_log_events_writes_all_rows_in_one_call(tmp_path: Path) -> None:
    """Batch writer exists so 'log every decision' survives 20k+ entries/run."""
    db_path = tmp_path / "audit.duckdb"
    entries = [_entry(case_id=f"click-{i}") for i in range(50)]

    n_written = log_events(entries, db_path=db_path)

    assert n_written == 50
    with duckdb.connect(str(db_path), read_only=True) as conn:
        n = fetch_one(conn, f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}")[0]
    assert n == 50


def test_log_events_appends_to_existing_log(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.duckdb"
    log_event(_entry(case_id="click-prior"), db_path=db_path)
    log_events([_entry(case_id=f"click-{i}") for i in range(3)], db_path=db_path)

    with duckdb.connect(str(db_path), read_only=True) as conn:
        n = fetch_one(conn, f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}")[0]
    assert n == 4


def test_log_events_empty_batch_is_a_noop(tmp_path: Path) -> None:
    """Nothing to log → no rows, no DB file side effect."""
    db_path = tmp_path / "audit.duckdb"
    assert log_events([], db_path=db_path) == 0
    assert not db_path.exists()
