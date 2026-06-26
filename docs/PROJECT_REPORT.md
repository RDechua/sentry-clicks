# Sentry — Project Report

*A mobile ad click-fraud detection system built end-to-end to industrial Trust
& Safety standards: detection, policy, adversary modeling, cost-based
enforcement, and an audit trail. This report is self-contained — it carries the
context that matters rather than deferring to the repo.*

---

## Executive summary

Sentry detects fraudulent mobile ad clicks on the TalkingData benchmark
(~185M clicks, 0.2% positive class). The calibrated LightGBM model reaches
**PR-AUC 0.559 and ROC-AUC 0.972** on a time-based held-out test set, with a
calibrated Brier score of **0.0015**. But the model is one of seven components.
The project also delivers a written fraud policy, a three-tier adversary model,
a cost-based thresholding methodology, a three-tier triage workflow, and an
audit log that records *why* every enforcement decision was made.

The point is to show T&S engineering judgment, not a leaderboard score:
time-based evaluation touched exactly once, an honest primary metric for a
rare-positive problem, leakage discipline enforced structurally, and calibrated
probabilities feeding a dollar-cost threshold model. The codebase is
containerized, has **165 tests at 98% coverage**, pins all seeds for
byte-identical reproducibility, and was built over **101 commits** with a
**51-entry decision log**.

## The problem, and why it is hard

- **Extreme imbalance.** Only ~0.2% of clicks are positive. Accuracy is
  meaningless; even ROC-AUC flatters the model by crediting it for rejecting
  the obvious negatives.
- **An adversary that adapts.** Fraud operators change behavior the moment a
  rule bites, so robustness has to come from feature combinations, not single
  signals.
- **A proxy label.** The label is *attribution* (did the click lead to an
  install), not *fraud*. Most non-converting clicks are uninterested real
  users — so the label is a loose proxy, and every downstream decision inherits
  that. I designed around it and flagged it everywhere rather than pretending
  it's clean.

## System architecture

A linear, batch pipeline with a clean separation of concerns. The model
predicts P(legitimate); **fraud score = 1 − P**, inverted in exactly one place
so the direction is never re-derived ad hoc.

```
raw CSV → ingest (+ row_id) → time-based split views (60/20/20)
        → feature pipeline (F1 per-click, F2 velocity, F3 aggregates)
        → versioned feature store (Parquet)
        → LightGBM + isotonic calibration → model bundle
        → enforce: score → fraud score → three-tier routing
                 → SHAP top-5 → audit log + human-review report
```

| Layer | Responsibility |
|---|---|
| `data/` | Ingest, validate, time-partition; owns `row_id` and the split firewall |
| `features/` | Define F1–F4, enforce strictly-prior windows, materialize a versioned store |
| `models/` | Train LightGBM, calibrate on val, score, and produce SHAP explanations |
| `triage/` | Cost model, cost-based threshold sweep, score→action router |
| `audit/` | Audit-log schema (pydantic) + DuckDB writer |

## Feature engineering

The model uses **26 numeric features** across three families. (Two
high-cardinality `ip×app` / `ip×device` interaction strings stay in the feature
store but are excluded from the model — they're a memorization trap that OOMs
the container and doesn't generalize.) Each family targets an adversary tier:

- **F1 — per-click attributes** (`app`, `channel`, `device`, `os`, hour, day):
  inventory and temporal context; weak alone, useful in combination.
- **F2 — velocity / burst** (clicks-per-IP over 1h and 24h, clicks-per-IP-app
  over 1h, inter-click time, std of inter-arrival, burst score): targets
  **naive, high-volume** fraud, which lights up every velocity signal at once.
- **F3 — behavioral aggregates** (conversion rates over 24h and all-time for
  IP / app / IP-app pair; distinct apps, devices, OSes per IP; distinct IPs per
  app; click counts): targets **adapted** fraud by keying on *outcome* — paced,
  rotated traffic still doesn't convert.

Every time-windowed feature reads only rows **strictly prior** to the current
click's timestamp (exclusive boundaries), enforced per-feature and verified by
property tests. Leakage is the cardinal sin of fraud modeling, so it's a
structural guarantee, not a convention.

## Results

| Metric (held-out test, scored once) | Sentry | Baseline | Random |
|---|---|---|---|
| **PR-AUC** (primary) | **0.559** | 0.11 (best single feature) | 0.0025 (base rate) |
| ROC-AUC | **0.972** | — | 0.50 |
| Calibrated Brier | **0.0015** | — | — |
| Recall @ <0.6% FPR | **~0.78** | — | — |

PR-AUC 0.559 is ~220× the base rate and well above the best single-feature and
linear baselines. On enforcement, the cost-optimal policy modeled a
**99.8% cost reduction** versus blocking nothing ($1,854 vs $1,116,537 on
validation). See `reports/figs/results.png` for the visual summary.

## Key technical decisions

Each was a deliberate choice with a documented rationale and rejected
alternatives (full reasoning in `docs/decisions.md`):

| Decision | Choice | Why (and what I rejected) |
|---|---|---|
| Train/val/test split | **Time-based**, test touched once | Random splits leak the future; time-based mirrors production retraining. |
| Primary metric | **PR-AUC** | ROC-AUC misleads on a 0.2%-positive class. Both reported; PR-AUC is the headline. |
| Class imbalance | **Class weights** | Rejected SMOTE (invents nonsense on categorical data and leaks) and undersampling (discards 99.8% of the signal). |
| Model | **LightGBM** | Memory/speed fit; native NULL + categorical handling; built-in PR-AUC early stopping. Rejected XGBoost (heavier), CatBoost (edge moot once raw IDs excluded). |
| Calibration | **Isotonic** (fit on val) | Cost-based thresholding needs real probabilities. Cut validation Brier 0.016→0.0013. Rejected Platt (sigmoid assumption unneeded at this volume). |
| Thresholds | **Cost-based** sweep | Real T&S tunes to dollar cost, not max-F1, under a reviewer-capacity cap. |
| Explainability | **SHAP top-5** per decision | An auditable decision must be explainable; the appeals substrate. |

## Enforcement: policy, cost model, and triage

The router maps each click's fraud score to one of four actions:

| Action | Trigger | Effect |
|---|---|---|
| `AUTO_BLOCK` | score ≥ `T_block` | Block the click; don't charge the advertiser |
| `HUMAN_REVIEW` | `T_review` ≤ score < `T_block` | Queue for a reviewer with the top-5 SHAP contributors |
| `QA_SAMPLE` | below `T_review`, random slice | Spot-check to estimate the false-negative rate / drift |
| `ALLOW` | below `T_review`, not sampled | Serve (and, if applicable, charge) |

**The cost model** drives the thresholds. Blocking a legitimate click costs
`c_fp` = $0.30; allowing a fraudulent click costs `c_fn` = $0.30; a human review
costs `c_review` = $0.50 (~90s of reviewer time). For a candidate threshold pair,
expected cost is `FP·c_fp + FN·c_fn + reviews·c_review`, minimized over
(`T_block`, `T_review`) subject to a reviewer-capacity cap. The dollar values are
illustrative and clearly marked as such; the *methodology* is the deliverable.

At these costs the sweep produced a clean two-way split (block ≥ 0.5, allow
below) with **no review tier** — review cost more than the error it prevents
(see Finding 3). The three-tier design and router are fully built; the empty
review tier is a *finding about* the tier, not a missing feature.

## Auditability: a representative audit entry

Every enforcement action emits a structured record (CLAUDE.md §3.9 — decisions
that don't produce one are bugs). A representative `AUTO_BLOCK` entry (schema
fields shown; feature names are the model's actual features):

```json
{
  "event_id": "f7b3c2a1-4d8e-4b6c-9f1a-2e3d4c5b6a7f",
  "event_timestamp": "2017-11-08T13:42:07Z",
  "case_id": "val-row-1048210",
  "click_timestamp": "2017-11-08T13:42:06",
  "model_version": "lgbm-v0.1.0",
  "policy_version": "policy-v0.1.0",
  "raw_score": 0.041,            // raw P(legit); fraud score = 1 − P = 0.959
  "calibrated_score": 0.012,
  "threshold_block": 0.50,
  "threshold_review": 0.50,
  "action": "AUTO_BLOCK",
  "top_features": [               // SHAP in log-odds of legitimacy; − = toward fraud
    {"feature_name": "f3_ip_app_conversion_rate_24hr", "value": 0.0, "shap_contribution": -2.41},
    {"feature_name": "f2_clicks_per_ip_last_1hr",      "value": 312, "shap_contribution": -1.18},
    {"feature_name": "f3_ip_distinct_apps_24hr",       "value": 27,  "shap_contribution": -0.77},
    {"feature_name": "f2_inter_click_time_seconds",    "value": 0.8, "shap_contribution": -0.52},
    {"feature_name": "f1_channel_id",                  "value": 245, "shap_contribution": 0.21}
  ],
  "reviewer_disposition": null
}
```

What the entry enables: **appeals** (the `case_id` and the exact features that
drove the call), **replay** (`model_version` + `policy_version` + thresholds pin
the full decision context), and **false-negative estimation** (allows are
audited via the QA sample, not individually — logging every allow at stream
scale is infeasible). The human-review queue renders these as a static HTML
report (`reports/sample_review_queue.html`) with the SHAP bars drawn per case.

## Adversary model

Modeled at three tiers; each maps to the feature family that counters it:

| Tier | Capability & economics | Counter in Sentry | Residual gap |
|---|---|---|---|
| **1 — Naïve** | Emulators/scripts, few IPs, tight bursts. Near-free, so no camouflage. | F2 velocity (every signal spikes at once) | None — caught easily |
| **2 — Adapted** | Rotating residential proxies, paced, varied fingerprints. Real per-click cost; bulk of volume. | F3 conversion-rate aggregates (faked traffic doesn't install) | Buying real installs to launder conversion rate |
| **3 — Sophisticated** | Real devices, low volume each, blended with genuine use. Most expensive, most damaging. | Only the easy edge (distinct-entity counts) | **Coordinated low-volume fraud** — needs graph/cluster features (descoped) |

The highest-value attack on Sentry specifically is defeating the outcome
features — its strongest family — by buying a baseline of real installs. Naming
that is the point of the exercise: it says where the next engineering
investment goes.

## Engineering at scale

Built and run inside a **3.8 GB container against 185M rows** — ~50× more data
than memory:

- **Streaming feature materialization.** Naive in-memory windowing OOMs, so
  features are computed in passes, with sliding-window aggregates rewritten as
  exact streaming equivalents (prefix-sum + ASOF joins, presence-segment
  events). Identical arithmetic, bounded memory.
- **Versioned feature store.** Checksummed Parquet per split (`v0.5.0`: ~18.5M
  rows, a 10% time-stratified sample), so a model artifact names exactly the
  features it trained on.
- **Reproducibility as a test.** Pinned seeds plus LightGBM deterministic mode
  yield byte-identical predictions, asserted by a test.

## Three findings that show judgment

1. **An anti-leakage rule that built a blind spot.** My original rule (features
   read only their own split) created a 24-hour cold start at every split
   boundary — a >2× feature shift on the 16-hour validation window. The fix was
   to *loosen* the rule to "any row strictly prior in time," which prevents real
   leakage without the artifact. Rigor isn't the same as strictness.
2. **Tuning made the model worse, and I shipped the untuned one.** Optuna on a
   subsample found parameters scoring 0.476 on full data — below the untuned
   default's 0.562 — because high `num_leaves` + low learning rate undertrains
   on the full set. I kept the default and documented the negative result rather
   than manufacturing a win.
3. **The cost-optimal policy reviews nobody.** At illustrative costs the sweep
   drove the review tier to zero width: reviewing a click ($0.50) costs more
   than the worst auto-decision error it prevents ($0.30). Human review only
   earns its place when it's cheaper than the expected error — visible only once
   real costs are on the table.

## Limitations (stated honestly)

- **The label is a proxy** — detecting non-conversion, not fraud; a production
  system needs a real fraud label. Seen viscerally when enforcement blocked
  ~99.8% of validation traffic.
- **A structural blind spot to sophisticated adversaries.** Per-entity features
  can't see coordinated, low-volume distributed fraud; that needs graph/cluster
  features, descoped and named as a gap.
- **Sampled scale.** Final results use a 10% time-stratified sample; the
  methodology is full-scale, the row count is sampled to fit the container.

## Supporting documentation and repository artifacts

The report stands alone, but the repository carries the depth behind each claim:

**Documents (`docs/`):**
- **PRD** — requirements, success metrics, and a plan-vs-reality changelog.
- **policy.md** — what counts as fraud, severity tiers, and 12 data-specific
  gray areas (e.g. the single-click-IP paradox, cold-start NULLs).
- **adversary-model.md** — the three tiers above, expanded, with economics and
  the "if I were the adversary" analysis.
- **tradeoffs.md** — the full cost-thresholding methodology and sensitivity
  analysis.
- **architecture.md** — components, data flow, deployment, and specific future
  work.
- **decisions.md** — a 51-entry running log; every non-obvious choice with its
  alternatives.
- **data-dictionary.md** — columns, the label gotcha, and the proxy caveat.

**Concrete artifacts:**
- **Model bundle** (`artifacts/models/lgbm-v0.1.0/`): `model.txt` (portable
  native format, no pickle) + `metadata.json` (features version, all seeds,
  params, metrics) + `calibrator.json` (isotonic knots).
- **Feature store** (`artifacts/features/v0.5.0/`): per-split Parquet +
  `manifest.json` (feature list, row counts, SHA-256 per file).
- **Audit log** (DuckDB) and the **human-review queue** (`reports/
  sample_review_queue.html`).
- **Test suite**: 165 tests, 98% line coverage on `src/sentry/`.

## What this demonstrates

End-to-end ownership (data → features → model → calibration → enforcement →
audit); methodological rigor an interviewer can probe line by line; T&S systems
thinking (policy, adversary modeling, cost-based enforcement, auditability);
production-shaped engineering under real memory constraints; and the judgment to
report honest results and keep an instructive negative one.
