# Sentry — Mobile Ad Click Fraud Detection

A click-fraud detection system built to industrial Trust & Safety standards.
On the TalkingData benchmark (~200M clicks, 0.2% fraud) it reaches **ROC-AUC
0.972 / PR-AUC 0.559** on a held-out test set, and pairs detection with the
parts real T&S work is actually made of: an explicit policy, an adversary
model, cost-based threshold tuning, a three-tier triage workflow, and an audit
log for every decision.

## Why this project exists

Most fraud-detection portfolio projects stop at the model. Real Trust & Safety
work is mostly policy, adversary thinking, and operational design — the model
is one piece. I built Sentry to demonstrate the full stack, end to end, the way
the work is actually done, not just the part that fits in a notebook.

## Highlights

- **Detection:** LightGBM + isotonic calibration — test PR-AUC **0.559** (≈220×
  the 0.0025 base rate), ROC-AUC **0.972**, calibrated Brier **0.0015**.
- **Policy** (`docs/policy.md`): what counts as fraud, severity tiers, and
  **12 gray areas** specific to this data — not a generic template.
- **Adversary model** (`docs/adversary-model.md`): three tiers (naïve → adapted
  → sophisticated), each tied to the features that counter it, with honest
  coverage gaps named.
- **Cost-based thresholds** (`docs/tradeoffs.md`): thresholds tuned to dollar
  cost, not max-F1 — including the finding that human review is uneconomic at
  the illustrative costs, and why.
- **Auditable enforcement:** every block/review/QA action is logged with raw +
  calibrated scores, active thresholds, model/policy versions, and the top-5
  SHAP contributors. See [`reports/sample_review_queue.html`](reports/sample_review_queue.html)
  and [`reports/audit_sample.json`](reports/audit_sample.json).
- **Methodology:** time-based splits (never random), PR-AUC as the headline
  metric, class weights (not SMOTE), pinned seeds with verified reproducibility,
  and a 50+ entry decision log.

## Quickstart

Requirements: Docker Desktop. (The Kaggle TalkingData files are kept outside
the repo and mounted read-only.)

```sh
# 1. Point the container at your TalkingData directory.
cp .env.example .env        # then edit DATA_DIR to your local path

# 2. Build the image.
docker compose build

# 3. Run the end-to-end pipeline on the 100k sample
#    (ingest → split views → F1+F2+F3 features → model → eval → triage + audit).
docker compose run --rm sentry sentry pipeline --sample
```

The quickstart runs a **baseline** model end-to-end on the 100k sample to
exercise every layer (it scores low — the sample is tiny and the model is a
deliberate baseline). The headline results below come from the **full LightGBM
model** on the complete feature store: train it with
`sentry train --features-version v0.5.0` and run enforcement with
`sentry enforce --features-version v0.5.0`. Run the full quality gate (ruff,
black, mypy, 165 tests, coverage) with `make check`.

## Document index

| Doc | What's in it |
|---|---|
| [**Project report**](docs/PROJECT_REPORT.md) | **Start here** — a 2–3 page technical overview of the whole project |
| [PRD](docs/PRD.md) | Problem, goals, success metrics, and a plan-vs-reality changelog |
| [Policy](docs/policy.md) | What counts as fraud, severity tiers, 12 gray areas |
| [Adversary model](docs/adversary-model.md) | Three adversary tiers and their counters |
| [Architecture](docs/architecture.md) | Components, data flow, deployment, future work |
| [Tradeoffs](docs/tradeoffs.md) | The cost-based threshold methodology |
| [Decisions](docs/decisions.md) | Running log of every non-obvious choice |
| [Data dictionary](docs/data-dictionary.md) | Columns, gotchas, the label proxy |

## Results

| Held-out test metric | Sentry | Best single-feature | Random |
|---|---|---|---|
| **PR-AUC** (primary) | **0.559** | 0.11 | 0.0025 (base rate) |
| ROC-AUC | 0.972 | — | 0.50 |
| Brier (calibrated) | 0.0015 | — | — |

PR-AUC is the headline because ROC-AUC flatters a 0.2%-positive problem. The
cost-optimal triage policy is **99.8% cheaper** than blocking nothing. PR/ROC
curves and the calibration plot are generated into `reports/` by the evaluation
harness.

## A note on AI assistance

Built with assistance from Claude Code for implementation, test scaffolding,
and boilerplate, and from Claude chat for design critique. Every methodological
decision, every word of the documentation's reasoning, and the project
direction are mine — the decision log (`docs/decisions.md`) records the choices
and the alternatives I weighed. I can defend every line.

## Author

**Rubeno Dechua** — Data Science, University of San Francisco (2025).
<!-- TODO (Rubeno): add LinkedIn URL and confirm the public contact email below -->
[LinkedIn](#) · rubenodechua123@gmail.com
