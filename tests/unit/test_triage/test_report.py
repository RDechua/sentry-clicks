"""Tests for the human-review HTML report (Task 6.4)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb

from sentry.audit.logger import log_events
from sentry.audit.schema import Action, AuditLogEntry, FeatureContribution
from sentry.data._db import fetch_one
from sentry.triage.report import ReviewCase, render_review_queue, write_review_report


def _case(case_id: str = "case-1") -> ReviewCase:
    return ReviewCase(
        case_id=case_id,
        calibrated_score=0.12,
        raw_score=0.20,
        click_timestamp="2017-11-08 13:30:00",
        model_version="lgbm-v0.1.0",
        top_features=[
            {
                "feature_name": "f3_app_conversion_rate_24hr",
                "value": 0.001,
                "shap_contribution": -2.5,
            },
            {"feature_name": "f1_os_id", "value": 19, "shap_contribution": 1.1},
        ],
    )


def test_render_contains_case_and_features() -> None:
    html = render_review_queue([_case("case-xyz")])
    assert "case-xyz" in html
    assert "f3_app_conversion_rate_24hr" in html
    assert "Human review queue" in html
    assert "log-odds" in html  # the units note for reviewers


def test_render_escapes_markup_in_case_id() -> None:
    """autoescape must neutralize markup so a crafted case_id can't inject HTML."""
    html = render_review_queue([_case("<script>alert(1)</script>")])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_empty_queue_is_valid_html() -> None:
    html = render_review_queue([])
    assert "0 case(s)" in html
    assert html.strip().startswith("<!DOCTYPE html>")


def test_write_review_report_filters_to_human_review(tmp_path: Path) -> None:
    db = tmp_path / "audit.duckdb"

    def entry(case_id: str, action: Action, cal: float) -> AuditLogEntry:
        return AuditLogEntry(
            event_timestamp=datetime(2017, 11, 8, 14, 0, 0),
            case_id=case_id,
            click_timestamp=datetime(2017, 11, 8, 13, 0, 0),
            model_version="lgbm-v0.1.0",
            policy_version="policy-v1",
            raw_score=cal,
            calibrated_score=cal,
            threshold_block=0.7,
            threshold_review=0.3,
            action=action,
            top_features=[
                FeatureContribution(feature_name="f1_os_id", value=19, shap_contribution=0.4)
            ],
        )

    log_events(
        [
            entry("rev-low-p", Action.HUMAN_REVIEW, 0.10),
            entry("rev-high-p", Action.HUMAN_REVIEW, 0.40),
            entry("blocked", Action.AUTO_BLOCK, 0.05),
            entry("allowed", Action.ALLOW, 0.95),
        ],
        db_path=db,
    )

    out = tmp_path / "queue.html"
    n = write_review_report(db, out)

    assert n == 2  # only the two HUMAN_REVIEW cases
    html = out.read_text()
    assert "rev-low-p" in html and "rev-high-p" in html
    assert "blocked" not in html and "allowed" not in html
    # Ordered lowest calibrated P(legit) first (highest fraud).
    assert html.index("rev-low-p") < html.index("rev-high-p")


def test_write_review_report_roundtrips_features(tmp_path: Path) -> None:
    db = tmp_path / "audit.duckdb"
    feats = [
        FeatureContribution(
            feature_name="f3_ip_conversion_rate_24hr", value=0.0, shap_contribution=-1.8
        )
    ]
    log_events(
        [
            AuditLogEntry(
                event_timestamp=datetime(2017, 11, 8, 14, 0, 0),
                case_id="c1",
                click_timestamp=datetime(2017, 11, 8, 13, 0, 0),
                model_version="m",
                policy_version="p",
                raw_score=0.2,
                calibrated_score=0.1,
                threshold_block=0.7,
                threshold_review=0.3,
                action=Action.HUMAN_REVIEW,
                top_features=feats,
            )
        ],
        db_path=db,
    )
    write_review_report(db, tmp_path / "q.html")
    # The feature JSON round-tripped DB -> report.
    with duckdb.connect(str(db), read_only=True) as conn:
        stored = fetch_one(conn, "SELECT top_features FROM audit_log")[0]
    assert json.loads(stored)[0]["feature_name"] == "f3_ip_conversion_rate_24hr"
    assert "f3_ip_conversion_rate_24hr" in (tmp_path / "q.html").read_text()
