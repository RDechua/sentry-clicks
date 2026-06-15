"""Human-review queue HTML report (Task 6.4).

Renders the cases routed to HUMAN_REVIEW into a static HTML page a reviewer
could work from: one row per case with the calibrated fraud score, the
action, the original click metadata, and the top-5 SHAP contributors drawn
as horizontal bars. Even though no human reviews these in this project, the
report is the visible artifact of the enforcement workflow — what a real
T&S tooling layer would produce.

SHAP values are in log-odds of legitimacy (Task 6.3): a positive bar pushed
the model toward "legit," a negative bar toward "fraud." The template labels
this so a reviewer doesn't read a bar as a probability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import duckdb
import structlog
from jinja2 import Environment

from sentry.audit.logger import AUDIT_TABLE_NAME
from sentry.audit.schema import Action

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ReviewCase:
    """One human-review case, flattened from an audit entry for rendering."""

    case_id: str
    calibrated_score: float
    raw_score: float
    click_timestamp: str
    model_version: str
    top_features: list[dict[str, object]]  # [{feature_name, value, shap_contribution}, ...]


# autoescape=True is the security-relevant setting: case_id and feature names
# could in principle carry markup, so the template must escape them.
_ENV = Environment(autoescape=True)

_TEMPLATE = _ENV.from_string("""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Sentry — human review queue</title>
<style>
 body { font: 14px/1.4 system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
 h1 { font-size: 1.3rem; } .meta { color: #666; margin-bottom: 1.5rem; }
 .case { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin-bottom: 1rem; }
 .case h2 { font-size: 1rem; margin: 0 0 .5rem; }
 .score { font-weight: 600; } .kv { color: #555; font-size: .85rem; }
 .bar-row { display: flex; align-items: center; gap: .5rem; margin: .2rem 0; }
 .bar-name { width: 14rem; font-size: .8rem; text-align: right; }
 .bar { height: .8rem; border-radius: 2px; min-width: 1px; }
 .bar.pos { background: #2a7a2a; }
 .bar.neg { background: #b23030; }
 .bar-val { font-size: .75rem; color: #555; }
 .note { margin-top: .5rem; } .note textarea { width: 100%; height: 2.5rem; }
</style></head><body>
<h1>Human review queue</h1>
<p class="meta">{{ cases|length }} case(s) routed to HUMAN_REVIEW. SHAP bars are
in log-odds of legitimacy — <span style="color:#2a7a2a">green pushed toward
legit</span>, <span style="color:#b23030">red toward fraud</span>. Bar width is
proportional to |SHAP| within each case.</p>
{% for c in cases %}
<div class="case">
  <h2>{{ c.case_id }}</h2>
  <div class="kv">calibrated fraud score
    <span class="score">{{ "%.4f"|format(1 - c.calibrated_score) }}</span>
    · raw P(legit) {{ "%.4f"|format(c.raw_score) }}
    · click {{ c.click_timestamp }} · model {{ c.model_version }}</div>
  {% for f in c.top_features %}
  {% set mag = (f.shap_contribution if f.shap_contribution >= 0 else -f.shap_contribution) %}
  <div class="bar-row">
    <span class="bar-name">{{ f.feature_name }} = {{ f.value }}</span>
    <span class="bar {{ 'pos' if f.shap_contribution >= 0 else 'neg' }}"
          style="width: {{ (mag / max_mag * 18)|round(2) }}rem"></span>
    <span class="bar-val">{{ "%+.3f"|format(f.shap_contribution) }}</span>
  </div>
  {% endfor %}
  <div class="note"><label>Reviewer note:<textarea></textarea></label></div>
</div>
{% endfor %}
</body></html>
""")


def render_review_queue(cases: list[ReviewCase]) -> str:
    """Render review cases to an HTML string."""
    max_mag = max(
        (abs(cast(float, f["shap_contribution"])) for c in cases for f in c.top_features),
        default=1.0,
    )
    return _TEMPLATE.render(cases=cases, max_mag=max_mag or 1.0)


def write_review_report(audit_db_path: Path | str, out_path: Path | str, limit: int = 500) -> int:
    """Query HUMAN_REVIEW audit entries and write the HTML report. Returns the
    number of cases rendered (highest fraud score first)."""
    with duckdb.connect(str(audit_db_path), read_only=True) as conn:
        rows = conn.execute(
            f"""
            SELECT case_id, calibrated_score, raw_score, click_timestamp,
                   model_version, top_features
            FROM {AUDIT_TABLE_NAME}
            WHERE action = ?
            ORDER BY calibrated_score ASC  -- lowest P(legit) = highest fraud first
            LIMIT ?
            """,
            [Action.HUMAN_REVIEW.value, limit],
        ).fetchall()

    cases = [
        ReviewCase(
            case_id=r[0],
            calibrated_score=float(r[1]),
            raw_score=float(r[2]),
            click_timestamp=str(r[3]),
            model_version=r[4],
            top_features=json.loads(r[5]),
        )
        for r in rows
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_review_queue(cases))
    logger.info("review_report_written", n_cases=len(cases), path=str(out_path))
    return len(cases)
