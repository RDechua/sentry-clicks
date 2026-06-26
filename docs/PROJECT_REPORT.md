# Sentry — Project Report

*A mobile ad click-fraud detection system built end-to-end to industrial Trust
& Safety standards: detection, policy, adversary modeling, cost-based
enforcement, and an audit trail.*

---

## Executive summary

Sentry detects fraudulent mobile ad clicks on the TalkingData benchmark
(~185M clicks, 0.2% positive class). The calibrated LightGBM model reaches
**PR-AUC 0.559 and ROC-AUC 0.972** on a time-based held-out test set, with a
calibrated Brier score of **0.0015**. But the model is one of seven components.
The project also delivers a written fraud policy, a three-tier adversary model,
a cost-based thresholding methodology, a three-tier triage workflow, and an
audit log that records *why* every enforcement decision was made.

The point of the project is to show T&S engineering judgment, not a leaderboard
score: time-based evaluation touched exactly once, an honest primary metric for
a rare-positive problem, leakage discipline enforced structurally, calibrated
probabilities feeding a dollar-cost threshold model, and every non-obvious
choice documented with its alternatives. The codebase is containerized, has
**165 tests at 98% coverage**, pins all random seeds for byte-identical
reproducibility, and was built over **101 commits** with a **51-entry decision
log**.

## The problem, and why it's hard

Click fraud is an adversarial, heavily imbalanced detection problem with a
labeling catch:

- **Extreme imbalance.** Only ~0.2% of clicks are the positive class. Accuracy
  is meaningless; even ROC-AUC flatters the model by crediting it for rejecting
  the obvious negatives.
- **An adversary that adapts.** Fraud operators change behavior the moment a
  detection rule bites, so any feature set has a shelf life and robustness has
  to come from combinations, not single signals.
- **A proxy label.** The dataset's label is *attribution* (did the click lead
  to an install), not *fraud*. Most non-converting clicks are uninterested real
  users, not fraud — so the label is a loose proxy, and every downstream
  decision inherits that looseness. I designed around it and flagged it
  everywhere rather than pretending the label is clean.

## System architecture

A linear, batch pipeline with a clean separation of concerns:

```
raw CSV → ingest (+ row_id) → time-based split views (60/20/20)
        → feature pipeline (F1 per-click, F2 velocity, F3 aggregates)
        → versioned feature store (Parquet)
        → LightGBM + isotonic calibration → model bundle
        → enforce: score → fraud score → three-tier routing
                 → SHAP top-5 → audit log + human-review report
```

The label convention everything agrees on: the model predicts
P(legitimate); **fraud score = 1 − P**, inverted in exactly one place so the
direction is never re-derived ad hoc.

## Key technical decisions

Each was a deliberate choice with a documented rationale and rejected
alternatives (full reasoning in `docs/decisions.md`):

| Decision | Choice | Why (and what I rejected) |
|---|---|---|
| Train/val/test split | **Time-based**, 60/20/20, test touched once | Random splits leak the future and inflate every metric; time-based mirrors production retraining. |
| Primary metric | **PR-AUC** | ROC-AUC misleads on a 0.2%-positive class. Reported both; PR-AUC is the honest headline. |
| Class imbalance | **Class weights** | Rejected SMOTE (invents nonsense on categorical data and leaks across the split) and undersampling (discards 99.8% of the signal). |
| Model | **LightGBM** | Histogram/leaf-wise growth fit the memory budget; native NULL + categorical handling; built-in PR-AUC early stopping. Rejected XGBoost (heavier) and CatBoost (its categorical edge is moot once raw IDs are excluded). |
| Calibration | **Isotonic** on val, applied to test | Cost-based thresholding needs real probabilities. Cut validation Brier from 0.016 to 0.0013. Rejected Platt (assumes a sigmoid we don't need with millions of rows). |
| Thresholds | **Cost-based** sweep | Real T&S tunes to dollar cost, not max-F1, under a reviewer-capacity cap. |
| Explainability | **SHAP top-5** per decision, in log-odds | An auditable decision must be explainable; required for the reviewer workflow and the appeals substrate. |

**Leakage discipline** runs underneath all of it: any time-windowed feature may
read only rows *strictly prior* to the current click's timestamp, enforced
per-feature with exclusive window boundaries and verified by property tests.
This is the highest-risk area in fraud modeling, so it's a structural
guarantee, not a convention.

## Engineering at scale

The system was built and run inside a **3.8 GB container against 185M rows** —
roughly 50× more data than memory. That constraint shaped the engineering:

- **Streaming feature materialization.** Naive in-memory windowing OOMs, so
  features are computed in passes (one query per window family, filtered to a
  batch of row IDs), with sliding-window aggregates rewritten as exact
  streaming equivalents (prefix-sum + ASOF joins, presence-segment events for
  distinct-entity counts). The arithmetic is identical; only the memory profile
  changed.
- **Versioned feature store.** Features are frozen as checksummed Parquet per
  split (`v0.5.0`: ~18.5M rows, a 10% time-stratified sample with full-history
  windows), so a model artifact names exactly the features it trained on.
- **Reproducibility as a test.** Pinned seeds plus LightGBM's deterministic
  mode yield byte-identical predictions on a re-run, and a test asserts it.

## Results

| Metric (held-out test, scored once) | Sentry | Baseline | Random |
|---|---|---|---|
| **PR-AUC** (primary) | **0.559** | 0.11 (best single feature) | 0.0025 (base rate) |
| ROC-AUC | **0.972** | — | 0.50 |
| Calibrated Brier | **0.0015** | — | — |
| Recall @ <0.6% FPR | **~0.78** | — | — |

The detection result is strong and honestly framed: PR-AUC 0.559 is ~220× the
base rate and well above the best single-feature and linear baselines. On the
enforcement side, the cost-optimal policy modeled a **99.8% cost reduction**
versus blocking nothing ($1,854 vs $1.12M on validation).

## Three findings that show judgment

1. **An anti-leakage rule that built a blind spot.** My original rule (features
   read only their own split) created a 24-hour cold-start at every split
   boundary — a >2× feature-distribution shift on the 16-hour validation
   window, which suppressed scores. The fix was to *loosen* the rule to "any
   row strictly prior in time," which prevents real leakage without the
   artifact. Rigor isn't the same as strictness.
2. **Tuning made the model worse, and I shipped the untuned one.** Optuna on a
   subsample found parameters that scored 0.476 on full data — below the
   untuned default's 0.562 — because high `num_leaves` + low learning rate
   undertrains on the full set (310 vs 746 effective rounds). I kept the default
   and documented the negative result rather than manufacturing a win.
3. **The cost-optimal policy reviews nobody.** At illustrative costs, the sweep
   drove the human-review tier to zero width: reviewing a click ($0.50) costs
   more than the worst auto-decision error it prevents ($0.30). Human review
   only earns its place when it's cheaper than the expected error — a result you
   only see once you put real costs on the table.

## Limitations (stated honestly)

- **The label is a proxy.** Detecting non-conversion, not fraud; a production
  system needs a real fraud label. Seen viscerally when enforcement blocked
  ~99.8% of validation traffic, because non-conversion *is* that share of the
  data.
- **A structural blind spot to sophisticated adversaries.** Per-entity features
  can't see coordinated, low-volume distributed fraud; that needs graph/cluster
  features, which were descoped to degree-style counts and named as a gap.
- **Sampled scale.** Final results use a 10% time-stratified sample; the
  methodology is full-scale, the row count is sampled to fit the container.

## What this demonstrates

For a Trust & Safety / Integrity engineering role, this project is evidence of:
end-to-end ownership (data → features → model → calibration → enforcement →
audit); methodological rigor an interviewer can probe line by line; T&S systems
thinking (policy, adversary modeling, cost-based enforcement, auditability);
production-shaped engineering under real memory constraints; and the judgment to
report honest results and keep an instructive negative one.

## Reproducibility and links

A fresh clone reproduces results via Docker. The repo's `docs/` holds the
deeper artifacts: the **PRD** (requirements + a plan-vs-reality changelog),
**policy.md** (12 data-specific gray areas), **adversary-model.md** (three tiers
tied to the actual features), **architecture.md**, **tradeoffs.md** (the cost
methodology), and **decisions.md** (the 51-entry log). The `README` is the
quickstart and front door.
