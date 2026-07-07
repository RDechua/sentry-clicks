# References

Sources actually used while building Sentry. Grouped by what they informed.

## Dataset

- **TalkingData AdTracking Fraud Detection Challenge** (Kaggle, 2018) —
  the click stream, the `is_attributed` label, and the competition context.
  <https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection>
  The competition was scored on ROC-AUC; that fact mattered for correcting this
  project's original PR-AUC target (see the PRD changelog and `decisions.md`,
  2026-06-10).

## Core libraries (official documentation)

Per the project's API-verification rule, library behavior was checked against
official docs rather than memory — these were the working references:

- **LightGBM** — parameters (`num_leaves`, `min_data_in_leaf`, bagging,
  determinism), the native `average_precision` eval metric, and categorical /
  NULL handling. <https://lightgbm.readthedocs.io/>
- **scikit-learn** — `IsotonicRegression`, `average_precision_score`,
  `brier_score_loss`, and calibration concepts.
  <https://scikit-learn.org/stable/>
- **DuckDB** — window frame semantics (`RANGE ... PRECEDING`, frame
  exclusion), ASOF joins, and memory configuration. <https://duckdb.org/docs/>
- **SHAP** — `TreeExplainer` and the log-odds units of its output for LightGBM
  binary models. <https://shap.readthedocs.io/>
- **Optuna** — TPE sampler and resumable SQLite storage.
  <https://optuna.readthedocs.io/>
- **pydantic / typer / structlog / hypothesis** — schema validation, the CLI,
  structured logging, and property-based tests, respectively (each project's
  official docs).

## Domain background

- Public Kaggle discussion threads and published solution write-ups for the
  TalkingData competition — used as a sanity check on feature families
  (velocity and per-entity aggregates dominate) and on the realistic metric
  range, not as implementation sources.
- Publicly available descriptions of Trust & Safety / integrity engineering
  practice at large platforms (policy-first framing, tiered enforcement,
  review-queue economics, audit requirements) — these shaped the PRD's
  structure and the policy/adversary documents' scope rather than any specific
  algorithm.

## A note on what is *not* cited

No academic papers are cited because none were directly used; the
methodological choices (PR-AUC on imbalanced data, isotonic calibration,
class weights, cost-based thresholds) are standard practice, adopted from the
library documentation and domain experience above, and defended on their own
terms in `decisions.md`.
