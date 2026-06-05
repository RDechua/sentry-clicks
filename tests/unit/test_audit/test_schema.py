"""Tests for `sentry.audit.schema` — `AuditLogEntry` + supporting types.

Covers field validation (score range, action enum), JSON round-trip,
defaults (event_id auto-generated as UUID), and that all four `Action`
values exist per build guide line 826.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from sentry.audit.schema import Action, AuditLogEntry, FeatureContribution


def _minimal_entry(**overrides: Any) -> AuditLogEntry:
    """Helper: an AuditLogEntry with the smallest valid payload."""
    defaults: dict[str, Any] = {
        "event_timestamp": datetime(2017, 11, 7, 9, 0, 30),
        "case_id": "click-12345",
        "click_timestamp": datetime(2017, 11, 7, 9, 0, 0),
        "model_version": "lgbm-v0.1.0",
        "policy_version": "policy-v0.1.0",
        "raw_score": 0.87,
        "calibrated_score": 0.65,
        "threshold_block": 0.90,
        "threshold_review": 0.40,
        "action": Action.HUMAN_REVIEW,
        "top_features": [
            FeatureContribution(feature_name="ip_click_count_1h", value=42, shap_contribution=0.18),
        ],
    }
    defaults.update(overrides)
    return AuditLogEntry(**defaults)


def test_action_has_four_values() -> None:
    """Build guide line 826: AUTO_BLOCK, HUMAN_REVIEW, QA_SAMPLE, ALLOW."""
    assert {a.value for a in Action} == {
        "AUTO_BLOCK",
        "HUMAN_REVIEW",
        "QA_SAMPLE",
        "ALLOW",
    }


def test_event_id_defaults_to_uuid() -> None:
    entry = _minimal_entry()
    assert isinstance(entry.event_id, uuid.UUID)


def test_event_id_unique_across_constructions() -> None:
    a = _minimal_entry()
    b = _minimal_entry()
    assert a.event_id != b.event_id


def test_raw_score_above_one_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal_entry(raw_score=1.5)


def test_calibrated_score_below_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal_entry(calibrated_score=-0.1)


def test_threshold_block_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        _minimal_entry(threshold_block=1.01)


def test_action_string_coerces_to_enum() -> None:
    """StrEnum lets callers pass the raw string and pydantic narrows it."""
    entry = _minimal_entry(action="ALLOW")
    assert entry.action is Action.ALLOW


def test_reviewer_fields_default_to_none() -> None:
    entry = _minimal_entry()
    assert entry.reviewer_disposition is None
    assert entry.reviewer_timestamp is None
    assert entry.notes is None


def test_round_trip_through_json() -> None:
    """The whole entry must serialize and reconstruct losslessly — this is
    what makes the audit log a durable artifact."""
    entry = _minimal_entry(
        top_features=[
            FeatureContribution(feature_name="ip_count", value=42, shap_contribution=0.18),
            FeatureContribution(feature_name="app", value=12, shap_contribution=-0.04),
            FeatureContribution(feature_name="channel", value="rare", shap_contribution=0.07),
        ],
    )
    serialized = entry.model_dump_json()
    restored = AuditLogEntry.model_validate_json(serialized)
    assert restored.event_id == entry.event_id
    assert restored.case_id == entry.case_id
    assert restored.action is Action.HUMAN_REVIEW
    assert len(restored.top_features) == 3
    assert restored.top_features[0].shap_contribution == pytest.approx(0.18)
    # Mixed-type feature value preserved (str passes through).
    assert restored.top_features[2].value == "rare"


def test_feature_contribution_accepts_int_float_str_values() -> None:
    """A feature can be int (count), float (rate), or str (categorical)."""
    FeatureContribution(feature_name="x_int", value=10, shap_contribution=0.1)
    FeatureContribution(feature_name="x_float", value=0.5, shap_contribution=0.1)
    FeatureContribution(feature_name="x_str", value="cat", shap_contribution=0.1)


def test_audit_sample_file_matches_schema() -> None:
    """The committed sample at `reports/audit_sample.json` must load as a
    valid `AuditLogEntry`. Regression check against schema drift: if a future
    edit renames a field or tightens a constraint, this fails immediately
    instead of the sample silently rotting out of sync with the schema.
    """
    from pathlib import Path

    sample_path = Path("reports/audit_sample.json")
    if not sample_path.exists():
        pytest.skip(f"{sample_path} not present (run from repo root)")
    AuditLogEntry.model_validate_json(sample_path.read_text())
