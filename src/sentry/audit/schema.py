"""Audit log entry schema for triage decisions.

Every model decision in the triage workflow (Task 6) produces one
`AuditLogEntry`. The entry must be self-contained: a reader of the log
alone should be able to reproduce the model's decision (build guide
line 842 — "logging features by name only without values" is the
canonical failure mode this schema protects against).

Field rationale captured in `docs/decisions.md` (2026-06-04: Audit log
schema design).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Action(StrEnum):
    """Triage decision applied to a click. The four values are exhaustive."""

    AUTO_BLOCK = "AUTO_BLOCK"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    QA_SAMPLE = "QA_SAMPLE"
    ALLOW = "ALLOW"


class FeatureContribution(BaseModel):
    """One feature's contribution to the model's score.

    `shap_contribution` is the SHAP value — the per-feature, per-prediction
    attribution that lets the audit log reconstruct *why* the model decided
    what it did, not just *what* it decided.

    `value` is intentionally permissive (int / float / str) so both numeric
    features (`ip_click_count_1h`) and categorical ones (`channel`) round-trip
    through the same record shape.
    """

    feature_name: str
    value: int | float | str
    shap_contribution: float


class AuditLogEntry(BaseModel):
    """One row of the audit log — a single triage decision.

    `event_id` is auto-generated as a UUID4 if not supplied, so callers can
    construct entries without coordinating IDs across processes.
    """

    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_timestamp: datetime  # when the audit entry was written (UTC)
    case_id: str  # identifier for the click being scored
    click_timestamp: datetime  # when the click actually occurred (UTC)

    # Model and policy versions change at different cadences and must be logged
    # separately — model retrains weekly, policy shifts monthly. Reconstructing
    # a decision from a policy-version lookup alone would lose the model
    # identity. Rationale in `docs/decisions.md`.
    model_version: str
    policy_version: str

    # Both raw and calibrated scores logged: raw for model-debugging, calibrated
    # for threshold comparison. The two diverge after Task 4.5 fits the isotonic
    # calibrator, and you'll want to see both when investigating a decision.
    raw_score: float = Field(ge=0.0, le=1.0)
    calibrated_score: float = Field(ge=0.0, le=1.0)

    # Thresholds are logged per-entry (not derived from policy_version) because
    # thresholds can be adjusted in-flight without a policy bump (e.g., emergency
    # tightening during an attack); the live value is the ground truth.
    threshold_block: float = Field(ge=0.0, le=1.0)
    threshold_review: float = Field(ge=0.0, le=1.0)

    action: Action

    # Top-N SHAP contributors per CLAUDE.md §3.9 (N=5 by convention; not enforced
    # here so the schema works for ablation studies and baseline models with
    # different feature counts).
    top_features: list[FeatureContribution]

    # Filled in after the fact if the case goes through human review.
    reviewer_disposition: str | None = None
    reviewer_timestamp: datetime | None = None
    notes: str | None = None
