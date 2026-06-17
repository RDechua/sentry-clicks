# Sentry — System Architecture

This document describes how Sentry is built: the components, how data moves
through them, how the pieces are configured and versioned, and what would
change to run it for real. It reflects what was actually implemented, not the
original plan — where the two diverge, the text says so.

The guiding constraint throughout is **leakage discipline** (CLAUDE.md §3.4):
a feature for any click may read only data strictly prior to that click's
timestamp. Much of the architecture exists to make that property structural
rather than something a reviewer has to take on faith.

A second constraint shaped almost every implementation choice: the whole
system was built and run inside a **3.8 GB Docker container** against a
184.9M-row dataset. That ratio — roughly 50× more data than RAM — is why the
heavy stages stream rather than materialize, and it is called out where it
matters below.

---

## 1. System overview

```
 raw CSV (Kaggle TalkingData, ~200M clicks)
   │
   ▼
 ┌──────────────┐   validate    ┌──────────────────────────────────────┐
 │ ingestion    │──────────────▶│ DuckDB: clicks table (+ row_id)       │
 │ data/        │   schema gate │ artifacts/sentry_full.duckdb (184.9M) │
 └──────────────┘               │ artifacts/sentry.duckdb (100k sample) │
                                └──────────────┬───────────────────────┘
                                               │ 3 time-based views
                                ┌──────────────▼───────────────┐
                                │ clicks_train / _val / _test   │  ← leakage firewall
                                │ data/splits.py (60/20/20)     │
                                └──────────────┬───────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────┐
                    │ feature pipeline / materializer                      │
                    │ features/  (F1 per-click, F2 velocity, F3 aggregates)│
                    │ strictly-prior windows; pass-split at full scale     │
                    └──────────────────────────┬──────────────────────────┘
                                               │ versioned Parquet
                                ┌──────────────▼───────────────┐
                                │ feature store                 │
                                │ artifacts/features/v0.5.0/    │
                                │ train.parquet / val / test    │
                                └──────┬─────────────────┬──────┘
                                       │ train           │ score
                          ┌────────────▼──────┐   ┌──────▼─────────────────────────┐
                          │ models/train.py   │   │ enforcement (inference)        │
                          │ LightGBM + early  │   │ predict → fraud score → route  │
                          │ stop on PR-AUC    │   │ → SHAP top-5 → audit → report  │
                          │ + isotonic calib  │   │ cli.py: `sentry enforce`       │
                          │ → model bundle    │   └──────┬───────────────┬─────────┘
                          └─────────┬─────────┘          │               │
                                    │ model.txt +        ▼               ▼
                                    │ metadata.json + ┌────────────┐  ┌──────────────────┐
                                    └ calibrator.json │ audit log  │  │ review-queue HTML│
                                      (the bundle)    │ DuckDB     │  │ reports/*.html   │
                                                      └────────────┘  └──────────────────┘
```

The flow is linear and batch-oriented: a click stream is ingested once,
partitioned by time, turned into features, used to fit a model, and then
re-scored under an enforcement policy that produces two artifacts — an audit
log and a human-review queue. There is no online serving path; scoring is a
batch job over a split. This is deliberate (PRD §2.2 non-goals): the project
demonstrates detection and enforcement *design*, not a production serving
stack.

The label convention is the one thing every component has to agree on:
`is_attributed = 1` means the click led to an install and is therefore
**legitimate**; it is the rare positive (~0.2%). The model predicts
P(`is_attributed = 1`), and **fraud score = 1 − P**. The single source of
that inversion is `triage/cost.py:fraud_probability`; everything downstream
reasons about fraud score so the direction is never re-derived ad hoc.

---

## 2. Data flow detail

**Ingestion (`data/ingestion.py`).** The CSV is read into a DuckDB `clicks`
table. Every row gets a stable `row_id` assigned by
`row_number() OVER (ORDER BY click_time, ip, app, device, os, channel)` — a
contiguous, deterministic key that is the join target for every feature
computed later. Index creation is optional (`create_indexes=False` at full
scale, because building secondary indexes on 184.9M rows exhausted the
container; the time-ordered scan the features need does not require them).

**Validation (`data/validation.py`).** Before anything reads the table, a
structural gate runs: required columns are non-null, `attributed_time` is
null exactly when `is_attributed = 0`, timestamps fall inside the known data
window, and the positive rate lands in [0.001, 0.01]. A failure here stops
the pipeline rather than letting a sampling or label-flip bug propagate into
a trained model.

**Splits (`data/splits.py`).** Three views partition the table by time on
pinned hour boundaries: `clicks_train` (earliest 60%), `clicks_val` (middle
20%), `clicks_test` (latest 20%), as half-open intervals
(`TRAIN_END_EXCLUSIVE = 2017-11-08 13:00`, `VAL_END_EXCLUSIVE = 2017-11-09
05:00`). Splitting is **time-based, never random** (§3.1): a random split
would let the model see the future and inflate every metric. `apply_split`
emits a warning when code reads the test view, because the test set is
touched exactly once, at the final evaluation (§3.1).

**Feature computation (`features/`).** Two designs coexist behind one
contract. The `FeaturePipeline` (`features/pipeline.py`) is the in-memory
path used by the tracer and on sample data: features declare dependencies,
are topologically sorted at construction (a bad graph fails before any
compute), run in order, and join back on `row_id`. At full scale the
`materialize.py` path takes over — same features, but computed in **passes**,
one query per window family, each filtered to a batch of `row_id`s so peak
memory stays bounded. Sliding-window aggregates that DuckDB could not hold in
RAM were rewritten as exact streaming equivalents (prefix-sum + ASOF join for
per-(app) windows; presence-segment events for distinct-entity counts). The
arithmetic is identical; only the memory profile changed.

**Feature store (`features/store.py`).** Materialized features are written as
versioned Parquet, one file per split, under `artifacts/features/<version>/`,
with a `manifest.json` recording the feature list, row counts, source, and a
SHA-256 per file. The canonical version is **v0.5.0** (train 11.1M / val 3.7M
/ test 3.6M rows — a 10% time-stratified sample of the full stream, with
full-history windows and all-time aggregates). Versioning the store means a
model artifact can name exactly the features it was trained on.

**Training and calibration (`models/`).** `train.py` fits LightGBM on the
26 numeric model features (the manifest lists 28; the two string interaction
features are excluded — see §3). Early stopping watches validation
`average_precision` (PR-AUC), the primary metric (§3.2). `calibration.py`
then fits isotonic regression on the validation split and stores it as JSON
interpolation knots. The three files — `model.txt`, `metadata.json`,
`calibrator.json` — form the **model bundle** (`models/predict.py:ModelBundle`).

**Enforcement (`cli.py:enforce`).** The inference path loads the bundle,
scores a split, converts to fraud score, routes each click through the
three-tier policy (`triage/router.py`), computes SHAP top-5 contributions for
every routed action, writes a full audit entry per action, and renders the
human-review queue HTML. Allowed clicks are audited via the QA sample rather
than individually — logging every allow at stream scale is neither feasible
nor realistic.

---

## 3. Component responsibilities

| Package | Responsibility |
|---|---|
| `data/` | Ingest, validate, and time-partition the click stream. Owns `row_id` and the split firewall. |
| `features/` | Define features (F1–F4), enforce strictly-prior windows per feature, and materialize a versioned feature store. |
| `models/` | Train LightGBM, calibrate on val, load/score a bundle, and produce SHAP explanations. Also baselines and Optuna tuning. |
| `evaluation/` | Compute the metric bundle (PR-AUC primary), run the ablation study, and render evaluation plots. |
| `triage/` | The cost model, the cost-based threshold sweep, and the score→action router. |
| `audit/` | The audit-log schema (pydantic) and the DuckDB writer. |
| `cli.py` | Four typer commands: `pipeline`, `train`, `tune`, `enforce`. |

**Feature families.** F1 (`f1_per_click.py`) are per-row attributes (app, os,
device, channel, hour, day, and two ip×app / ip×device interaction strings).
F2 (`f2_velocity.py`) are strictly-prior velocity windows — clicks-per-ip in
the last 1h/24h, inter-click time, burst score — all using
`RANGE BETWEEN INTERVAL X PRECEDING AND INTERVAL 1 MILLISECOND PRECEDING` so
the current row is never in its own window. F3 (`f3_aggregates.py`) are
counts, conversion rates, and distinct-entity counts over 24h and all-time
windows. F4 (graph features) was descoped to degree-style counts only
(`f3_*_distinct_*`); the full bipartite graph work did not earn its
complexity within the time budget (PRD changelog).

**Why the two string interactions are excluded from the model.** LightGBM
trees on raw high-cardinality interaction strings memorize specific ip×app
pairs — a leakage-adjacent overfitting trap that flattered validation and
would not generalize. The model trains on the 26 numeric features; the
interactions remain in the store for inspection and baselines.

**Calibration is isotonic, not Platt** (§3.5), fit on val and applied to
test, never fit on test. Isotonic assumes only monotonicity; with millions of
validation rows there is ample data for its flexibility to beat Platt's
sigmoid assumption.

**Headline results** (test split, touched once): PR-AUC **0.559**, ROC-AUC
**0.972**, calibrated Brier **0.0015**, against a 0.0025 base rate. The
cost-optimal triage at illustrative costs is a two-way split at fraud score
0.5 — the human-review tier does not pay (review costs more than the error it
prevents); the reasoning is in `docs/tradeoffs.md`.

---

## 4. Configuration and versioning

**Configuration is intentionally thin.** The original plan (CLAUDE.md §4)
had a `config.py` pydantic-settings module; in practice the CLI surface stayed
small enough that it was never needed, and `config.py` is an empty stub. This
is a deliberate YAGNI outcome, not an oversight. Configuration lives in two
places:

- **The Docker mount.** `DATA_DIR` (from `.env`, see `.env.example`) is
  mounted read-only at `/data` by `docker-compose.yml`. Read-only is a safety
  property: the pipeline physically cannot modify the source data. If
  `DATA_DIR` is unset, compose falls back to the repo's empty `./data` so the
  image still builds for a smoke test.
- **CLI flags.** Each command takes explicit paths and version labels with
  sensible defaults (e.g. `--features-version v0.5.0`, model dir
  `artifacts/models/lgbm-v0.1.0`). The defaults encode the canonical run; the
  flags make every input overridable without code changes.

The only environment read in the code itself is the matplotlib backend
(`MPLBACKEND=Agg`), set so plotting works headless in the container.

**Versioning is explicit and layered.** A feature-store version (`v0.5.0`)
names a frozen set of features and rows with per-file checksums. A model
version (`lgbm-v0.1.0`) records, in `metadata.json`, the features version it
was trained on, the full parameter set, all random seeds, the best iteration,
and the validation metrics — so a model is always traceable back to its exact
inputs. A policy version is recorded on every audit entry. These three
versions (features, model, policy) are the coordinates that make a past
decision reproducible and explainable.

**Reproducibility** (§3.8) is built in: every seed (numpy, LightGBM —
including `bagging_seed`, `feature_fraction_seed`, `data_random_seed` — sklearn,
Optuna) is pinned, and LightGBM runs with `deterministic=True` and
`force_row_wise=True`. Training the same data twice yields identical
predictions; a test asserts it.

---

## 5. Deployment considerations

The project is not deployed (PRD §12). What would change for production:

- **Serving.** Batch scoring over a Parquet split would become a streaming or
  micro-batch service. Features like clicks-per-ip-last-1h, currently computed
  with a SQL window over historical rows, would move to an online feature
  store / stream processor (e.g. a windowed aggregation in Flink or a
  feature-store with TTL'd counters) so they are available at decision time
  with the same strictly-prior semantics.
- **Latency budget.** SHAP top-5 per decision is cheap in batch but would need
  a budget online; one option is to compute SHAP only for routed (blocked /
  review) cases, exactly as the batch `enforce` already does, rather than for
  every allow.
- **Model registry.** The file-based bundle (`model.txt` + `metadata.json` +
  `calibrator.json`) maps cleanly onto a registry entry; the metadata already
  carries everything a registry would index.
- **Calibration refresh.** Isotonic calibration is fit on a fixed validation
  window. In production it would be re-fit on a rolling recent window, because
  the score→probability mapping drifts as traffic mix changes.
- **The data mount** would become a governed data source with retention and
  access controls; the read-only mount is the local stand-in for that.

A genuine non-goal worth stating: the enforcement demo routes on a quantile
threshold to populate all three tiers for the review-queue artifact. That is
**artifact population, not policy selection** — the operating thresholds are
the cost-based ones from Week 5. Production would route on the calibrated
score under the cost policy, not on quantiles.

---

## 6. Monitoring and drift (conceptual)

Three drift surfaces would need monitoring (PRD §10):

- **Feature drift.** The distribution of velocity and conversion-rate features
  shifts as traffic and fraud campaigns change. The `manifest.json` row
  counts and per-feature summaries are the baseline; production would track
  population-stability-index per feature against it.
- **Calibration drift.** Brier score and a reliability curve on a labelled
  recent sample detect the score→probability map going stale — the trigger to
  re-fit the isotonic calibrator.
- **Policy drift.** Review-queue volume against reviewer capacity, and the
  block / review / allow mix over time. A sudden change in the routed fraction
  usually means the input distribution moved, not that fraud did.

The **feedback-loop hazard** (PRD §10.3, open question) is the one to respect:
once the system blocks clicks, those clicks never produce attribution labels,
so the training data for the next model is censored by the current model's
decisions. The QA sample (a random, un-blocked slice of allows) exists to keep
an unbiased estimate of the false-negative rate on confidently-allowed traffic
— the only labelled window the policy does not distort.

---

## 7. Future work

Specific extensions, each with what it would actually require:

1. **Online feature serving.** Re-implement F2/F3 windows as TTL'd streaming
   counters keyed by ip and ip×app. Requires a stream processor and a feature
   store; the strictly-prior contract carries over unchanged, but late-arriving
   events need an explicit watermarking policy.
2. **Full graph features (F4).** Build the bipartite ip↔device / ip↔channel
   graph and compute connected-component and shared-neighbor features. Requires
   a graph build step over a time window and a leakage proof for each feature
   (the component a click belongs to must be computed from strictly-prior
   edges only).
3. **Adversarial / drift retraining loop.** Schedule periodic retrains on the
   QA-sampled labelled window, with a champion/challenger gate on test PR-AUC
   before promotion. Requires the registry from §5 and an automated evaluation
   harness run.
4. **Cost-model parameterization from real numbers.** The current costs
   (`c_fp`, `c_fn`, `c_review`) are illustrative. With real advertiser-charge
   and reviewer-time figures, the threshold sweep (`triage/thresholds.py`)
   produces an operating policy directly; the methodology does not change.
5. **Appeals workflow.** Currently conceptual (PRD §8.4 / `docs/policy.md`).
   Requires a case-state store and a reviewer UI; the audit log already holds
   the per-decision evidence an appeal would be adjudicated against.
6. **Pydantic config layer.** If the input surface grows (multiple data
   sources, environment-specific paths), promote the empty `config.py` stub to
   the originally-planned `pydantic-settings` module. Until then, CLI flags are
   the right altitude.

---

*Companion documents: `docs/PRD.md` (requirements), `docs/policy.md` (what
counts as fraud and how the system acts), `docs/adversary-model.md` (who we
defend against), `docs/tradeoffs.md` (the cost methodology), and
`docs/decisions.md` (the running decision log).*
