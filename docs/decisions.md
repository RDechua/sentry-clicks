# Decisions Log — Sentry-Clicks

This document captures non-obvious choices made during the project. Every decision logged here is one I should be able to defend in an interview. Entries are written when decisions are made, not retrospectively.

---

## Day 1 Honesty Contract — signed 2026-05-20

I'm building this project to transition into Trust & Safety engineering in an attempt to enter the Trust & Safety field, and it's only worth doing if the code, methodology, and documentation all reflect my actual thinking. To make that real, I commit to the following while I build it:

- I won't claim work I can't defend. If I can't explain a line of code, a feature, or a metric in plain language, I'll either learn it well enough to defend it or remove it.
- I won't skip documentation in favor of finishing faster. The docs are part of the deliverable, not exhaust.
- I'll surface what I don't know rather than hide it. Open questions go in the PRD with `[OPEN QUESTION]` markers; methodological doubts go in this log; uncertainty in features goes in their docstrings.
- I'll write decisions when I make them, not retroactively the night before an interview. Backfilled rationales are easy to spot.
- I'll prefer the honest version of any decision over the impressive-sounding one. An 8-week deadline I miss is worse than a 12-week one I hit.
- The test set is untouchable except at the single formal evaluation in Task 4.7. No "just one peek."

— Rubeno Dechua

---

## 2026-05-20: Repository name

**Context:** Build guide §3.6 suggests naming the GitHub repo `sentry` because it's short and memorable. I'm naming it `sentry-clicks` instead.

**Options considered:**
- `sentry` — short, matches the codename used throughout the docs.
- `sentry-clicks` — slightly clunkier but unambiguous about the domain.
- `ad-fraud-detection` — descriptive but generic; doesn't tie back to the codename.

**Decision:** `sentry-clicks`.

**Reasoning:** "Sentry" alone is that of the well-known error-monitoring company (sentry.io), which is the first search result on any Google query for the word. A recruiter glancing at my GitHub link would land on their site first, and naming overlap looks careless. `-clicks` differentiates and signals the domain (ad-click fraud) in the slug itself. The codename "Sentry" still appears in PRD documents and internal references — it's the *repo slug* that disambiguates, not the project identity.

**Confidence:** High.

**Revisit:** Never, unless I rename the project itself.

---

## 2026-05-20: Sampling strategy under 8 GB RAM

**Context:** My MacBook has 8 GB RAM, which is the absolute floor of the build guide §3.1 requirement. The full TalkingData training set is ~200M rows / ~7 GB uncompressed CSV; encoded efficiently in pandas (categorical, int32) a full-data DataFrame would still be ~20-30 GB in memory. DuckDB handles aggregations and joins out-of-core, so feature *computation* over the full data is feasible. The bottleneck is LightGBM training, which materializes its training matrix in memory and would OOM on the full data. I need to commit to a sampling approach *now* so it's deliberate, not a Week-4 panic move.

**Options considered:**

1. **Use `train_sample.csv` (100k rows, Kaggle-provided) end-to-end.** Fastest iteration, trivially fits in RAM. Disqualifying for headline results: 100k rows × ~0.2% positive rate leaves only ~200 positives, far too few to reliably train on or evaluate against.

2. **Random sample (10% across all rows) for training, full data for inference.** Bigger sample (~20M rows). Disqualifying methodologically: random sampling across rows smears the temporal structure that the time-based train/val/test split (CLAUDE.md §3.1, anti-pattern #5) depends on. A row from day 4 could randomly precede a row from day 1 in some windowed feature computation.

3. **Head sampling (first N rows).** Disqualifying — would give me only day 1, killing the time split entirely.

4. **Time-stratified sample for training and validation, full data for the final test inference.** Take the same proportional sample within each day. Preserves the temporal structure the time-split depends on. Inference at evaluation time is per-row and much lighter on memory than training, so reporting headline PR-AUC on the full test set is feasible even on 8 GB.

**Decision:** Option 4 — time-stratified sample for training and validation, full data for the final test-set inference (Task 4.7).

- **Weeks 1-3** (feature dev, framework, EDA): use `train_sample.csv` (100k rows) for fast iteration. These weeks are shaking out the pipeline, not measuring model performance, so the small sample is fine.
- **Weeks 4 onward** (modeling): time-stratified sample of ~10% within each day → roughly 20M rows for training, smaller for val. Implementation: stratified-sample SQL in DuckDB, proportional draw within each day, fixed seed. Sample fraction and seed pinned in `config.py` and emitted to the audit log so results are reproducible.
- **Final test evaluation (Task 4.7):** inference on the full unsampled test slice. Inference runs in batches and doesn't require the full feature matrix in memory at once. The headline number in the README is measured against full data.

**Reasoning:** Three things drove this. (a) The time-based split is the most important methodological commitment in this project; a sampling scheme that contaminates it is worse than no sampling at all. (b) Training and inference have very different memory profiles, and the project is bottlenecked on training only — so I sample what I have to sample and use full data where I can. (c) The headline metric being on the full test set means the number I report is the number a real evaluator would see. That defensibility matters more than any model strength I'd gain from training on full data.

**Failure mode I'm accepting:** A model trained on 20M sampled rows may underperform what could be achieved on 200M. If LightGBM's PR-AUC on the 10% sample is, say, 0.83 against a hypothetical 0.87 on full data, I'm leaving ~4 points on the table. I'm accepting that because the alternative is either (i) not finishing, (ii) renting a cloud VM and inflating project cost, or (iii) reporting numbers on a 100k toy slice that wouldn't be defensible. I'll note this explicitly in the README's results section.

**Confidence:** Medium-high on the *approach*, medium on the *10% rate*. The 10% figure is rough memory math; if Week 4 training OOMs on 20M rows, I'll drop to 5%, document it here, and move on. If training on 20M is fast and easy, I won't try to push to 30% — past ~10% the marginal model improvement is small and the iteration cost is large.

**Revisit:** End of Week 4. If actual training-per-trial time on 20M rows exceeds ~6 minutes on the 8 GB Mac, that makes the 50-100 Optuna trials infeasible locally and triggers the cloud-pivot decision below.

---

## 2026-05-20: Training compute location

**Context:** The Optuna hyperparameter sweep in Task 4.4 calls for 50-100 trials, each a full LightGBM training run on the sampled data. Build guide §3.4 estimates 4-8 hours total on local CPU and suggests this can run overnight. Cloud spot VM is the alternative.

**Options considered:**

- **Local-only.** Free. Slow. Risk: an overnight Optuna run that crashes wastes a whole day. Ablation studies (Task 4.6) require multiple training runs and would compound any per-run slowness.
- **Cloud spot VM (GCP or AWS, ~$0.30/hr × ~50 hrs total ≈ $15-30).** Faster, more flexible. Small dollar cost, real time cost: ~4-6 hours of one-time setup (auth, billing, data transfer, image build).
- **Hybrid: local for everything, cloud only for Optuna if local proves infeasible.**

**Decision:** Hybrid. Start local. Pivot to cloud only if Week 4 training-per-trial time exceeds ~6 minutes (which would push 100 trials past 10 hours and make iteration painful).

**Reasoning:** Cloud overhead is real (4-6 hours of setup) and dollar cost is small but nonzero. Doing it preemptively isn't free. Doing it when I have evidence I need it is cheaper in both time and money. The pivot criterion is specific enough that I won't drift: 6 minutes/trial × 100 trials = 10 hours, which is "set it overnight and pray" territory and I'd rather not.

**Confidence:** Medium. The biggest unknown is how slow the 8 GB constraint will make individual LightGBM trials — LightGBM is memory-efficient but disk-paging on a constrained machine is hard to predict.

**Revisit:** Week 4, Task 4.3 (training pipeline). First end-to-end training run will trigger this.

---

## 2026-05-20: Timeline commitment

**Context:** Build guide §3.5 explicitly calls out the honesty check on time: the plan assumes 10-15 hours/week of focused work (80-120 hours total). I have a full-time L4 Robot Operator role and active Darva AI co-founder work. Both are demanding. Build guide §3.5 also warns: "A 12-week timeline at 8 hours/week beats an 8-week timeline at 8 hours/week that ends with a half-finished project."

**Options considered:**

- **8 weeks at 15 h/wk (120 hours total).** The build guide's default. Optimistic given competing commitments.
- **10 weeks at 12 h/wk (120 hours).** Buffer for one rough week without falling behind.
- **12 weeks at 10 h/wk (120 hours).** Most calendar-flexible.

**Decision:** 10-week timeline at ~12 h/wk, with the explicit understanding that weeks involving heavier Darva or Robot Operator load may slip to 8 h, recovered by lighter weeks at 14-15 h. Total budget unchanged at ~120 hours.

**Reasoning:** I'm choosing 10 weeks over 12 not out of optimism but because the calendar matters: stretching this to 12 weeks pushes the finished portfolio piece into the window where job applications start in earnest, and I want the project done before that wave. 10 weeks is the latest I can finish without it competing with application time. The two-week buffer over the default isn't slack — it's realism about competing commitments.

**Failure criterion:** If I'm two full weeks behind by the end of Week 4 (i.e., I haven't completed Week 2 work by then), I'll re-evaluate openly: either descope (drop F4 graph features — the build guide already flags them as potentially descopable), extend to 12 weeks formally and document it here, or stop the project and write up what I have. I won't silently slip.

**Confidence:** Medium. Real schedules slip; what I'm committing to is the *response* if they do, not certainty that they won't.

**Revisit:** End of Week 4 (mid-project checkpoint).

---

## 2026-05-20: Local toolchain install path

**Context:** Pre-flight needed Docker, an updated git (Apple Git 2.39.3 < 2.40 minimum), and Python 3.11+ accessible. Multiple install paths exist on macOS.

**Options considered:**

- **Homebrew + Docker Desktop (.app).** Standard, well-documented. Docker Desktop is ~5-10 GB and includes a GUI dashboard.
- **Homebrew + Colima.** Lighter (CLI-only VM-backed Docker substitute). Smaller disk footprint. Less standard.
- **Homebrew + OrbStack.** Newer, also lighter. Closed-source for some features. Less common in T&S engineering shops.
- **Direct downloads.** Manual, unreproducible.

**Decision:** Homebrew for git; Docker Desktop (.app) for the container runtime.

**Reasoning:** Standard, defensible path. Docker Desktop is what the bulk of the dev world uses, so problems are easy to Google and the setup matches what a teammate would see. The 5-10 GB cost is real given my disk pressure (~27 GB free) but isn't enough to justify going off the beaten path. If disk pressure becomes severe, Colima is the sober fallback over OrbStack.

**Confidence:** High.

**Revisit:** Only if disk pressure forces a switch. Trigger: <10 GB free with a feature still missing.

---

## 2026-05-20: Shell PATH ordering for Homebrew tools

**Context:** After `brew install git`, `which git` still returned `/usr/bin/git` (Apple Git 2.39.3) because `/opt/homebrew/bin` was at position 15 in `$PATH`, behind `/usr/bin`. The Python.org installer's `python3` path also sat at position 3, which is why the system `python3` shows 3.10.6 even though Homebrew has `python3.11` installed.

**Decision:** Added `eval "$(/opt/homebrew/bin/brew shellenv)"` to `~/.zprofile`. This prepends `/opt/homebrew/bin` on every login shell so Homebrew-installed tools win.

**Reasoning:** This is the canonical Homebrew incantation — it's what `brew install` prints in its post-install caveats and what the Homebrew docs recommend. Anything else (manual `export PATH=...` lines, symlinking individual binaries) is non-standard and brittle. I'm not touching the Python.org `python3` path because (a) the project uses `uv` for Python version management, so the system default doesn't matter, and (b) ripping out the Python.framework path could break unrelated tooling I have lying around.

**Confidence:** High.

**Revisit:** Never, unless I switch shells away from zsh.

---

## 2026-05-24: src/ layout for the Python package

**Context:** Task 1.1 of the build guide called for the full repository directory structure. One non-obvious choice in that tree is placing the package at `src/sentry/` rather than `sentry/` at the repo root. The build guide explicitly flags this as a stop-and-think.

**Options considered:**

- **Flat layout (`sentry/` at the repo root).** Simpler. The package is importable from the repo root without an editable install. Less to explain to a fresh reader.
- **`src/` layout (`src/sentry/`).** The Python community's preferred layout. Forces `pip install -e .` (editable install) before imports work locally; imports go through the installed package regardless of working directory.

**Decision:** `src/` layout.

**Reasoning:** Three reasons. (a) Forcing the editable install catches packaging bugs locally that would otherwise surface at Docker build or deployment time — a misnamed module, missing entry point, or wrong package metadata breaks immediately, not at the end of a 60-minute container build. (b) Import paths are consistent regardless of where I invoke a test from; pytest runs from `tests/`, `notebooks/`, or `experiments/` all see the same `sentry.*` namespace. (c) The `src/` layout is the convention in modern Python packaging guidance and is widely used in established open-source Python projects. Picking the convention makes the repo legible to anyone who's worked in a real Python project.

**Confidence:** High. This is a near-universal best practice for any Python package intended to be installed or distributed.

**Revisit:** Never, unless the repo is restructured for some unrelated reason.

---

## 2026-05-25: Dependency manager — uv

**Context:** Task 1.2 of the build guide raised the dep-manager question as a stop-and-think (Poetry, uv, pip-tools, plain pip + requirements.txt). The choice was already locked in CLAUDE.md §5.2 as `uv`; this entry records the reasoning.

**Options considered:**
- **`uv`.** Fast (written in Rust), single binary, handles venvs and Python version pinning natively.
- **Poetry.** Well-established, large ecosystem, but slow resolver and has had backward-compat churn over the past two years.
- **`pip-tools`.** Minimal, mature, pairs with plain pip. Doesn't manage Python versions itself.
- **Plain `pip + requirements.txt`.** No lockfile rigor, no env management.

**Decision:** `uv`.

**Reasoning:** Three reasons. (a) Resolver speed — uv resolves a typical project's deps in under a second; Poetry routinely takes 5-30 seconds on the same input. Across a 10-week build with iterative dep changes, that adds up. (b) Single binary, no installer drama — uv installs itself and gets out of the way; Poetry has periodic install/upgrade pain. (c) For a portfolio piece aimed at a 2026 T&S engineering audience, using uv signals that I'm current with the Python ecosystem; Poetry is still fine but isn't where the field is moving.

**Confidence:** High. uv is mature enough that recommending it to a new project in 2026 isn't a risk.

**Revisit:** Only if uv development stalls or makes a breaking change that's expensive to migrate from. Unlikely in the 10-week window.

---

## 2026-05-25: Base image and architecture — python:3.11-slim-bookworm on linux/amd64

**Context:** Task 1.2 requires a Dockerized environment. Two coupled choices: which Python base image, and (on Apple Silicon) which target architecture.

**Options considered:**

*Base image:*
- **`python:3.11-slim-bookworm`.** Debian 12 (bookworm), Python 3.11, slim variant (~120 MB). Recommended in the build guide.
- **`python:3.11-alpine`.** Alpine Linux, much smaller (~50 MB), but uses musl libc, which breaks many ML wheels (numpy, lightgbm) that target glibc.
- **`python:3.11-bookworm`** (non-slim). Full Debian, ~700 MB. Includes a lot we don't need.

*Architecture (host is Apple Silicon arm64):*
- **Native arm64.** Image matches the host. Faster builds, no Rosetta translation. Won't run on standard x86 cloud VMs without rebuild.
- **`linux/amd64`** (forced). Image runs on any x86_64 Linux (cloud spot VMs, CI runners, reviewer machines). Slightly slower on M-series via Rosetta.

**Decision:** `python:3.11-slim-bookworm` on `linux/amd64`.

**Reasoning:** Base-image choice is the build guide's recommendation and is also the Python ecosystem's general default for Dockerized projects — slim-bookworm avoids alpine's musl pitfalls without dragging in the full Debian footprint. Architecture choice: for a portfolio project the image needs to be runnable on a typical reviewer's environment without surprises, and the most common server architecture remains x86_64. Building amd64 locally on M-series costs maybe 30-50% on build time via Rosetta, but builds happen rarely; the consistency benefit at review time (and the option to later run on a cloud spot VM without rebuilding) is worth the local build-time cost.

**Confidence:** High on both calls.

**Revisit:** Architecture choice gets revisited if a future deployment target is ARM-native (e.g., AWS Graviton). Base image gets revisited if a future dep needs a newer Debian or specific libc not in bookworm.

---

## 2026-05-25: Runtime dependencies for Task 1.3

**Context:** Task 1.3 of the build guide pulls in the full runtime + dev dep set from CLAUDE.md §5.3 and §5.4 into `pyproject.toml` and produces `uv.lock`. Most picks are already locked in CLAUDE.md; this entry records the rationale for the locked picks and a few smaller calls made during the task.

**Locked picks worth narrating:**

- **`pandas`, not `polars`.** The build guide explicitly flagged this as a stop-and-think (line 620). pandas is the lingua franca of sklearn/lightgbm interfaces — every example in those libraries' docs uses pandas. polars is faster but adds a translation layer at every interaction with the ML libraries. For Sentry-Clicks, the workflow is "DuckDB does the heavy lifting → pandas for the LightGBM interface"; polars would buy speed in a step that isn't the bottleneck.
- **`typer`, not `click`.** Typer wraps click and adds type-hint–driven CLI generation. Same underlying engine, better ergonomics.
- **`structlog`, not stdlib `logging`.** Structured logging is the right substrate for the audit-log work that starts in Task 1.9 (every model decision must produce an audit record with timestamp / case-id / model-version / policy-version / scores / SHAP contributors). stdlib `logging` can be coerced into structured output but structlog is purpose-built for it.

**Three task-level calls:**

1. **No version constraints in `pyproject.toml` — `uv.lock` is the single source of truth for pins.** The build guide says "Pin everything" and `uv.lock` does that (exact versions + hashes for every transitive dep). Pinning *also* in `pyproject.toml` doubles the maintenance surface (every minor upgrade requires editing both) without buying reproducibility (the lockfile already provides it). This is the modern uv/Poetry idiom.

2. **PEP 735 `[dependency-groups]` for dev deps; `default-groups = ["dev"]` so they install by default.** uv's older `[tool.uv].dev-dependencies` is deprecated. PEP 735 is the standardized form, future-proof beyond uv.

3. **Dev deps included in the runtime image (~300-500 MB of bloat accepted).** Multi-stage builds with separate prod/dev images are the production-grade answer. For an 8-10 week portfolio project where the same image is used for development, testing, and demonstration, a single combined image is simpler and matches the build guide's intent (running `pytest` inside the container is a standard workflow from Task 1.5 onward). The trade-off — a 3.4 GB image instead of ~2.5 GB — is documented here as a known optimization to revisit if production deployment ever becomes in-scope.

**Confidence:** High on the locked picks (they're CLAUDE.md decisions); high on the task-level calls (each has a clear local rationale).

**Revisit:** Dev-deps-in-image gets revisited if a production image variant becomes necessary (likely never, for this project).

---

## 2026-05-25: Reversed architecture choice — native arm64 over forced linux/amd64

**Context:** The earlier same-day "Base image and architecture" entry chose `--platform=linux/amd64` for portability across reviewer environments and potential cloud VMs. The reasoning was that build-time Rosetta cost on M-series (30-50%) was acceptable because builds happen rarely. That reasoning was incomplete in a way that only showed up during Task 1.3 verification.

**What changed:** Container *starts* hit Rosetta cold-start cost just like builds do — and starts happen constantly during development (every `pytest`, every `python -c "..."`, every interactive session is a fresh container). After installing the full runtime dep set (164 packages, including numpy / scikit-learn / lightgbm with their compiled `.so` files), `import lightgbm, duckdb, sklearn, optuna` inside a fresh container was still running at 100% CPU 35+ minutes later under Rosetta — translation of all the native `.so` files cold-starts per container. The Rosetta translation cache doesn't persist across containers.

**Decision:** Reverted to native architecture. `FROM python:3.11-slim-bookworm AS runtime` (no platform pin); `docker-compose.yml` no longer sets `platform: linux/amd64`. The image now builds natively for the host (arm64 on M-series), and the AC test finishes in ~4 seconds instead of >35 minutes.

**Cost of this reversal:** The image won't run on standard x86 cloud VMs without rebuild. For Sentry-Clicks, this is fine: cloud deployment is conditional on the training-location decision's "pivot if local trial > 6 min" trigger, and any cloud rebuild happens on the cloud VM anyway. The "reviewer can pull and run this on x86" portability argument from the original entry is weaker than I gave it credit for — a serious reviewer can rebuild for their arch in ~2 minutes, and a casual reviewer who can't is going to read the README, not run the image.

**Lesson worth recording:** I underestimated the cost of cross-arch emulation for ML workloads. Forced-platform builds are reasonable when the runtime is going to a single specific arch (e.g., final deployment); they're a bad default for iterative dev work. The cost scales with how often you cold-start the runtime, not how often you build.

**Confidence:** High.

**Revisit:** If we ever need a publishable multi-arch image, `docker buildx build --platform linux/amd64,linux/arm64` is the right tool. Not needed for this project.

---

## 2026-05-25: Code-quality tooling for Task 1.4

**Context:** Task 1.4 wires ruff + black + mypy + pre-commit + a Makefile. CLAUDE.md §5.5 locks the high-level picks (which tools, line length, rule set, mypy strictness scope); this entry records the smaller calls made during the task.

**Calls:**

1. **All tool config lives in `pyproject.toml`.** Canonical 2026 pattern — one file, one source of truth, no `mypy.ini` / `.ruff.toml` drift. Sections: `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.black]`, `[tool.mypy]` + two `[[tool.mypy.overrides]]` blocks.

2. **mypy: tests are checked but lenient via `[[tool.mypy.overrides]]`.** `disallow_untyped_defs = false` so test functions can stay terse, but `check_untyped_defs = true` so type errors inside those tests still surface. Notebooks and experiments aren't checked (exploratory; will use `nbqa` for notebook-specific runs later). Third-party libs without stubs (lightgbm, shap, duckdb, optuna, structlog, seaborn, matplotlib, jinja2, nbqa) get `ignore_missing_imports = true` per the build guide's stated failure mode.

3. **Pre-commit hook revs pinned to match `uv.lock`** (`ruff v0.15.14`, `black 26.5.1`). When uv re-locks, the YAML needs a manual rev bump — documented in the YAML header as the maintenance trade-off. Considered `repo: local` hooks that shell into the project venv to eliminate this drift; rejected as operational complexity not justified for a single-developer portfolio.

4. **pytest-xdist is locked but not enabled by default.** Adding `-n auto` to a 1-test pytest invocation made it ~10x slower (worker startup overhead). Will enable in Task 1.5 once the test count justifies parallelism. CLAUDE.md §7 anti-pattern #14 ("abstractions in case we need them later") is the reason — premature optimization here is a regression today.

5. **One smoke test added now (`tests/test_sanity.py`).** `make check` invokes pytest, which exits 5 ("no tests collected") on empty scaffolding. The smoke test (`assert sentry.__name__ == "sentry"`) is a real packaging-correctness check that earns its keep beyond Task 1.5 — it catches a broken `src/` layout install before any other test runs.

6. **Makefile runs everything in the container via `docker compose run --rm sentry`.** Consistent environment between dev iteration and what a reviewer's machine would see. `make check` chains all four tools in a single container start (~4s) instead of four (~10s). Considered backgrounding ruff/black/mypy with `&` + `wait` to parallelize the static-analysis tools; rejected — managing background pid exit codes and interleaved output complicates failure diagnosis for a sub-5-second loop.

**Confidence:** High.

**Revisit:** Enable `-n auto` at Task 1.5 (real test suite). Bump pre-commit revs whenever uv re-locks ruff/black.

---

## 2026-05-25: Test framework choices (Task 1.5)

**Context:** Task 1.5 wires pytest + coverage + the canonical `tiny_sample_data` fixture. The tool picks (pytest, hypothesis, pytest-cov, pytest-xdist) are locked in CLAUDE.md §5.4; this entry records the task-level calls.

**Calls:**

1. **Coverage flags live in `make check`, not in `pytest` addopts.** Putting `--cov` in `addopts` would force every pytest invocation — including TDD inner loops — to pay the coverage.py tracing tax (~10-40% slowdown) and write `coverage.xml` to disk. Keeping addopts to just `--strict-markers` + `--strict-config` keeps single-test runs fast. Coverage is invoked explicitly in `make check` and the new `make coverage` target.

2. **Coverage target 70% on `src/sentry/`, excluding `cli.py`.** The build guide's recommendation. Coverage is a sanity check, not a goal — well-written meaningful tests at 60% are better than test-the-getters padding at 90%. `cli.py` is omitted because CLI entry points are exercised by integration tests in later weeks, not by unit tests.

3. **Fixture design — 40 rows, hand-crafted blocks.** Each block targets specific feature behavior (L1: legit baseline, F2 burst/slow: velocity signal, L2: engaged legit user that should NOT trip detection, F3 skew: per-(ip,app) conversion variance, Borderline: single-click ambiguity, Edge: rare combinations + sub-2s conversions + sentinel channels). 40 rows fits on a screen; large enough for non-trivial aggregates. 1,000 random rows would be worse — random data hides the edge cases the fixture is designed to test.

4. **Single `_at(**kwargs)` helper for click_time construction**, not three separate `_s`/`_m`/`_h` helpers. Keyword-only kwargs read naturally at call sites (`_at(hours=2, minutes=15)`) and avoid parameter sprawl. timedelta's signature is the underlying truth.

5. **Session-scope base fixture + function-scope `.copy()` wrapper.** The DataFrame construction happens once per pytest session; each test gets a fresh copy so it can mutate freely. Marginal speedup on a 1-test suite, but the pattern is correct as the suite grows into the hundreds.

6. **pytest-xdist remains deferred** (same reasoning as Task 1.4). Worker startup is net-negative on the current 1-test suite. Will enable when feature tests land (Task 2.6 onward) and the suite is in the 20+ test range.

**Confidence:** High on all calls.

**Revisit:** Coverage threshold may rise if 70% is trivially hit. pytest-xdist when the test count justifies.

---

## 2026-06-04: Data ingestion choices (Task 1.6)

**Context:** Task 1.6 wires the CSV → DuckDB pipeline that all subsequent feature and model work depends on. The high-level approach (DuckDB substrate, pydantic schema, explicit validation) is established in CLAUDE.md and the build guide; this entry records the task-level calls.

**Stop-and-think — sample vs full data (build guide line 719):** For Week 1, only `train_sample.csv` (100k rows) is ingested. The full `train.csv` (~7 GB, 200M rows) waits until Week 4. The ingestion function takes any CSV path — no code change is needed to switch — but the integration test path is hardcoded to `/data/train_sample.csv` and the validator's class-balance gate of `[0.001, 0.01]` is tuned for production-shaped data. The same function and validator handle the full ingest when it lands.

**Calls:**

1. **DuckDB streams the CSV natively via `read_csv(?, columns={...}, header=true)`.** No Python-side pandas chunking. Memory footprint is bounded by DuckDB's internal buffer regardless of file size, so the 100k sample and the eventual 200M full ingest use the same code path. The build guide's "chunked reads" is what DuckDB does implicitly.

2. **Two-file schema source of truth, kept in sync by hand.** `ClickRecord` (pydantic) documents the per-row schema for tests, audit-log payloads (Task 1.9), and any future row-level validation. `DUCKDB_COLUMN_TYPES` (dict) drives the DuckDB DDL and `read_csv` call. The duplication is real but small (8 columns); the alternative — generating DDL from the pydantic model dynamically — adds metaprogramming complexity that obscures the schema.

3. **`validate_ingestion` returns a structured `ValidationResult`, not raises.** Per the Task 1.6 plan call (Option B): `ValidationResult.ok`, `.errors: list[CheckResult]`, and individual `.checks` so each check is testable in isolation. The structure slots into the audit-log schema work in Task 1.9. Raising would have been simpler for callers but throws away per-check diagnostics.

4. **Validation runs as one SQL query, not per-check queries.** Each check is a `COUNT(*) FILTER (WHERE ...)` aggregate in a single SELECT. Single full-table scan instead of 11. The `/simplify` pass surfaced this — at 200M rows the savings are ~10x; at 100k it's invisible, but the pattern is the same.

5. **`ORDER BY click_time` at table-creation time.** DuckDB's zone maps and most feature queries are time-windowed, so physical row ordering by click_time avoids re-sorting on every scan once data scales. One-time sort cost at ingest pays for itself by the second feature query.

6. **Indexes on `click_time` and `ip` per the build guide.** Point lookups by ip dominate F2 (velocity) and F3 (aggregates); click_time index helps boundary scans. ROI at 200M rows isn't certain — flagged for verification in Week 2-3. If they don't pay off, drop them then.

7. **`TRAIN_DATE_MAX_EXCLUSIVE = datetime(2017, 11, 12)`, compared with `<`.** Original `datetime(2017, 11, 11, 23, 59, 59)` with `<=` silently misses the last second. Exclusive midnight is the standard way to express "through end of 11/11" without truncation.

8. **`fetch_one` helper in `src/sentry/data/_db.py`** for queries known to return exactly one row. DuckDB's `fetchone()` returns `Optional[tuple]`; the helper raises `RuntimeError` (not `assert`, which disappears under `python -O`) so the invariant is explicit at every call site.

9. **`py.typed` marker at `src/sentry/py.typed`** (PEP 561). Tells mypy our package is typed; without it, downstream type-checking of `from sentry... import` lines fails with "missing library stubs" warnings.

10. **mypy `namespace_packages = true` + `explicit_package_bases = true` + `mypy_path = "src"`.** Lets tests have multiple files with the same basename (e.g., `tests/unit/test_data/test_ingestion.py` and `tests/integration/test_ingestion.py`) without needing `__init__.py` shims, and resolves `src/sentry/*` files as `sentry.*` (not `src.sentry.*`) so they aren't found twice.

11. **`--import-mode=importlib` for pytest.** Same root cause as #10 — without it, pytest's legacy import mode prepends test directories to `sys.path` and same-named test modules collide.

12. **`tests/unit/test_data/` subdirectory.** The build guide tree (line 444 onward) listed `tests/unit/test_features/`, `test_models/`, `test_triage/`, `test_audit/` but not `test_data/`. Added it for consistency with the source layout (`src/sentry/data/` ↔ `tests/unit/test_data/`).

**Confidence:** High on the structural calls (1-3, 5, 7-12). Medium on the index ROI question (6).

**Revisit:** Index ROI in Week 2-3 once real feature queries exercise the data. `ClickRecord`'s role expands when audit-log payloads need row-level validation (Task 1.9). `--import-mode=importlib` may need revisiting if any pytest plugin doesn't play nicely with it.

---

## 2026-06-04: Week 1 EDA surprises

**Context:** Task 1.7 ran 10 SQL queries against the 100k-row `train_sample.csv` to characterize the TalkingData click distribution before designing features. Findings live in `reports/eda/findings.md` (local, gitignored) alongside the CSV outputs. This entry captures observations that updated my priors or surfaced signals I didn't expect — the build guide flagged this as "gold in interviews when someone asks 'what did you learn from your data?'".

**Headline number for the rest of the project:** overall positive rate is **0.227%** (227 conversions / 100,000 clicks). PR-AUC, threshold tuning, and cost-based evaluation all reference this number.

**Surprises that updated my priors:**

1. **`device=0` converts at 9.6% — sixty times the overall rate.** I expected the dominant device (device=1, 94% of clicks) to set the model's expectation and rare devices to be either noise or a fraud-cluster. Instead, the sentinel `device=0` bucket (0.5% of clicks) is a clean-traffic indicator. Implication: don't fold device=0 into a "rare device" long-tail; it's its own signal. F1 (per-click features) should keep device as a categorical with `device=0` specifically retained, not bucketed.

2. **Single-click IPs convert at 0.86%; multi-click IPs convert at ~0.07–0.12%.** The naive intuition ("engaged users convert more") is backwards. For this data, frequent IPs are click farms; the one-shot IPs are the legitimate users. F2 (velocity) features should weight high click counts as a fraud-positive signal, not a quality signal. (Obvious in hindsight to anyone who's worked on ad fraud — I wanted to see the actual numbers before designing F2.)

3. **99.7% of `(ip, app)` pairs with ≥ 5 clicks never convert.** Among the 1,059 pairs at that threshold, only 3 ever produce a conversion. F3 (per-(ip, app) conversion rate) is going to be the strongest single feature. The flip side: a feature that's *that* discriminating can dominate the model and mask weaker signals. Worth a note when wiring F3 in Task 3.1 — check that the ablation study shows other features still earning their place.

4. **Some clicks convert in 2 seconds.** Minimum click → attribution lag is 2 seconds. The click → app-store-redirect → install → attribution-callback chain physically takes longer than that. Either TalkingData pre-attributes some installs based on prior intent, or those rows are mislabeled. Either way, the model shouldn't learn that 2s-lag clicks are MORE legitimate — that's a label artifact, not real signal. (Moot since `attributed_time` is label-leaking and not a feature anyway, but worth knowing as a data-quality awareness item.)

5. **0-second gaps between consecutive same-IP clicks.** Minimum inter-click gap across the dataset is 0s — multiple clicks in the same wall-clock second from the same IP. Not physically possible for a human; it's the bot fingerprint. F2 should specifically include a "clicks in the previous 1 second from this IP" feature, not just the 60s / 1h windows I was originally planning.

6. **The fraud pattern is "thin spread across many apps", not "burst on one app".** 50 IPs in the 100k sample touched ≥ 10 distinct apps with near-zero conversion. None are high-volume per app — they're spreading 50–700 clicks across 20+ apps each. F3 aggregates need `n_distinct_apps_per_ip` as a feature, not just `n_clicks_per_ip`.

**Things I expected but didn't see (worth noting for the full-data run):**

- No IPs in the 100k sample exceeded 1000 clicks. The full 200M-row dataset will have IPs in the 10k+ range; F2 percentile thresholds need to be calibrated against full-data distributions, not the sample.
- No strong diurnal pattern in fraud rate (0.18%–0.31% across hour-of-day, no monotone shape). I'd expected a "fraud spikes overnight" pattern; doesn't show up here.

**Confidence:** High on observations 1, 2, 3, 6 (structural — robust to sample size). Medium on 4 and 5 (could be sample-specific data artifacts).

**Revisit:** All of these get re-checked against the full 200M-row dataset in Week 4. Any shift is itself a feature-design signal.

---

## 2026-06-04: Evaluation metrics — choice and rationale (Task 1.8)

**Context:** The evaluation harness lands in Task 1.8, before any model exists. The metric set it computes will be referenced for the rest of the project — every baseline, every iteration, every ablation gets compared on the same numbers. The high-level picks (PR-AUC primary, ROC-AUC for completeness, Brier for calibration) are locked in CLAUDE.md §3.2; this entry re-derives why, plus the surrounding calls (curve points vs just AUC, fixed-threshold confusion matrices, fail-loud on no-positives) that landed during Task 1.8 implementation.

**Why PR-AUC, not ROC-AUC, as the primary metric.** ROC-AUC measures whether a classifier ranks positives above negatives. On a balanced problem this is a fine summary. On TalkingData's 0.227% positive rate, it is misleading in a specific way: a classifier that correctly ranks the trivial 99.8% of true negatives at the bottom gets credit toward ROC-AUC even if it's terrible at separating the actual positives from the near-misses at the top of the ranking. PR-AUC (precision-recall area, equivalent to sklearn's `average_precision_score`) ignores the easy negative-ranking work and focuses on precision at each level of recall — which is what an operator actually cares about. The Task 1.8 test `test_pr_auc_below_roc_auc_on_imbalanced` makes this concrete: with a deliberately mediocre predictor on a 0.5%-positive synthetic dataset, ROC-AUC sits above 0.6 while PR-AUC is below 0.3. Both numbers describe the same predictor; one of them is honest. Reporting ROC-AUC as the headline number on an imbalanced fraud problem is the canonical senior-vs-junior tell — a reviewer who sees it on a portfolio repo will stop reading.

**Why ROC-AUC is still computed and reported.** Two reasons. First, it's the metric every reviewer reaches for first; including it in the output lets the harness answer the question without forcing the user to re-run anything. Second, the PR-AUC / ROC-AUC ratio is itself diagnostic — a model with PR-AUC = 0.50 and ROC-AUC = 0.95 is telling you something different than one with PR-AUC = 0.50 and ROC-AUC = 0.55, and the second one's failure mode is harder to fix. Keeping both in `EvaluationResult` makes the ratio cheap to inspect.

**Why Brier score, not log-loss, for calibration.** Both are proper scoring rules and both reward calibrated probabilities. The practical difference: log-loss is unbounded as predictions approach 0 or 1 on the wrong side (a single confidently-wrong prediction can dominate the metric on a 100k-sample evaluation), while Brier is the bounded squared-error analog. For an audit-logged production system where one mislabeled outlier shouldn't blow up the per-batch score, Brier's bounded behavior is more useful. A second smaller reason: Brier decomposes cleanly into reliability + resolution + uncertainty (Murphy 1973), which we won't need for the harness skeleton but will when Task 4.5 fits the isotonic calibrator and we want to attribute calibration error to the right component. Log-loss can be added as a secondary metric in Week 4 if it turns out to be useful for model selection — it's cheap to compute, but it isn't the headline.

**Why store full PR-curve and calibration-curve points, not just AUC scalars.** Two reasons. First, the operating threshold in Task 5 will be chosen by sweeping the PR curve against a cost model, not by maximizing a scalar AUC — so we need the points, not the summary. Second, two models can have the same PR-AUC with very different curve shapes: one with high precision at low recall (good for "definitely-fraud" auto-blocks), another with high recall at low precision (good for "send to review" queues). Visualizing both curves is how that distinction surfaces. The `EvaluationResult` model carries the curve points as `list[PRPoint]` and `list[CalibrationPoint]` so any downstream notebook or report can re-plot without re-evaluating.

**Why a fixed-threshold set for confusion matrices, and why those specific thresholds.** The harness computes confusion matrices at `(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)` — skewed toward the low end deliberately. On a 0.2%-positive problem, anything with precision ≥ 0.5 typically lives at a threshold well below 0.5. Concretely: a calibrated model trained on the actual base rate will assign p(fraud) ≈ 0.998 to most negatives and p(fraud) ≈ 0.99-something to most positives, but the discriminating predictions sit in the 0.05-0.3 range where the model is least confident. Sweeping the high thresholds (0.95, 0.99) would just produce all-zero confusion matrices. The fixed set gives operators a quick "what does precision look like if I dial the threshold here?" view without committing to a single operating point — that commit happens in Task 5.2 against the cost model.

**Why `evaluate()` raises `ValueError` on `y_true` with no positives.** PR-AUC is mathematically undefined when there are no positives — precision has no denominator. sklearn returns NaN; we raise. The reason: silent NaN propagation is the bug that hides for weeks. A test split with zero positives is almost always a split-construction mistake (badly stratified, too small, leaked the wrong rows out) — failing loud at the metric layer surfaces it immediately at the point of evaluation, while NaN can drift unnoticed through `compare()` tables and plot legends. The cost is that the harness becomes harder to use for legitimate one-class debugging cases; we accept that and recommend callers degenerate-case those manually. The all-positives case is treated more permissively: ROC-AUC returns NaN (it genuinely needs both classes for the rank statistic) but PR-AUC is exactly 1.0 and Brier still makes sense, so the harness returns the result with `roc_auc=nan` rather than raising.

**Two smaller calls worth recording:**

1. **Calibration curve uses `strategy="quantile"` with 10 bins.** Uniform binning on a 0.2%-positive problem would put 99% of samples in the lowest bin and make calibration look perfect by definition. Quantile binning distributes samples evenly across bins, so the calibration plot actually says something about whether predicted probabilities match observed frequencies across the score distribution.

2. **`EvaluationResult` is a pydantic BaseModel, not a dataclass.** Consistent with `ValidationResult` from Task 1.6 — both serialize trivially to JSON for the audit log work in Task 1.9, and pydantic validates field types on construction (catches a string-where-float bug at the call site, not at a later plot rendering). The cost is a small import-time penalty for pydantic; the benefit is the same `model_dump_json` / `model_validate_json` round-trip pattern we already use elsewhere.

**Confidence:** High on all the substantive picks (PR-AUC, ROC-AUC alongside, Brier, curve points, fail-loud). Medium on the specific threshold list — it's a reasonable starting set; expect to adjust after the Task 5.2 cost-curve work shows where operators actually want to sit.

**Revisit:** Add log-loss as a secondary metric in Week 4 if model selection needs it. Re-tune the fixed threshold list after Task 5.2 if the operating-point range it surfaces is narrower than expected. Re-examine the no-positives ValueError if any legitimate use case surfaces (none anticipated in this project's scope).

---

## 2026-06-04: Audit log schema design (Task 1.9)

**Context:** Task 1.9 designs the audit log schema *before* any model exists. Per CLAUDE.md §3.9, every triage decision will produce one entry; the schema defines what the system can later answer questions about — for debugging, drift detection, regulatory pull, post-incident reconstruction. Designing it before the system exists prevents the "we don't log that, sorry" gap that gets discovered three months in.

**Stop-and-think — why `model_version` and `policy_version` separately, why thresholds logged per-entry:**

Model and policy change at different cadences. The model retrains on a weekly drumbeat; the policy (threshold tuning, action mapping, reviewer-queue capacity) shifts on a monthly cadence at most, with intra-month emergency adjustments during attacks. Logging a single `system_version` would force the cadence of the less-frequent thing onto the more-frequent thing — every model retrain would have to bump a "system" version, and post-incident replay would need a versioned-policy lookup table to figure out which thresholds were active at decision time. Separate fields are the cheaper representation.

Thresholds are logged per-entry rather than derived from `policy_version` for a more concrete reason: in production, thresholds get adjusted *in flight* without a full policy version bump. Emergency tightening during a coordinated attack is the canonical case — the on-call engineer drops `threshold_block` from 0.95 to 0.85 at 02:00 AM, and the policy doc doesn't get updated until the morning post-mortem. Looking up "what was the block threshold at 2017-11-07 09:00:30?" via `policy_version` alone would silently reconstruct the wrong number. The live value is the only honest source.

**Other task-level calls:**

1. **DuckDB, not SQLite.** Build guide gave both as options. DuckDB is already our substrate for clicks and feature tables; introducing SQLite would mean two database engines, two connection idioms, and two backup stories for no benefit. The audit log will fit comfortably in DuckDB (estimated max ~10M rows over the project lifetime, well under any DuckDB scaling concern).

2. **Separate `audit.duckdb` from `sentry.duckdb`.** Clicks data is loaded once per ingest and read-heavy after that; audit log grows monotonically with every decision and is append-heavy. Different access patterns warrant different files: an audit-log full-table scan won't compete with feature queries, and dropping the clicks DB to re-ingest doesn't wipe the audit history.

3. **`top_features` as a JSON column, not STRUCT/LIST.** Different models will have different feature sets — the LightGBM model carries one set, the Task 4.2 baselines carry another (or none at all). A typed STRUCT/LIST in the audit table would couple the schema to the current model's feature names. JSON is the right altitude — flexible per-row, queryable via DuckDB's JSON functions when needed (`json_extract(top_features, '$[0].feature_name')`), and round-trips cleanly through `model_dump_json`.

4. **`Action` as a `StrEnum`** (Python 3.11+). Type-safe in Python (`Action.AUTO_BLOCK` not `"AUTO_BLOCK"`), but `.value` is a plain string so it stores natively in DuckDB's VARCHAR column without a JSON coercion step. Callers can also pass the raw string and pydantic narrows it to the enum, which keeps the call sites readable.

5. **UUID `event_id`, auto-generated via `default_factory=uuid.uuid4`.** No cross-process coordination needed — every entry constructor produces a globally-unique ID without a central sequence. Stored as DuckDB's native `UUID` type (not VARCHAR) so primary-key lookups stay fast.

6. **`value` in `FeatureContribution` is `int | float | str`.** Numeric features (`ip_click_count_1h`) coexist with categoricals (`channel`, `device`) in the same field. Forcing everything to string would lose type info; forcing everything to float would mangle categoricals. The union is the honest shape.

7. **`reviewer_*` fields nullable, filled in after the fact.** The model writes the decision; reviewers fill in their disposition later via a separate update path (Task 6 triage workflow). At write time these are NULL; the schema allows but doesn't require them.

8. **No `pii_redacted` flag.** The `case_id` is an opaque identifier (e.g., `click-{timestamp}-ip-{ip}-app-{app}`) and the IP in TalkingData is already anonymized at source. No PII handling is needed. Worth recording explicitly: if this were a real production system, `case_id` would need a hashing/redaction layer and there'd be a `pii_redacted` boolean per entry. For Sentry-Clicks portfolio scope, the data is already privacy-cleaned upstream.

**The sample entry** at `reports/audit_sample.json` shows a realistic `HUMAN_REVIEW` decision with five top-feature SHAP contributors. Anyone reading it should be able to answer: which model made the call, what features pushed the score up, what threshold was active, what action was taken — without consulting any other source.

**Confidence:** High on all calls. The schema is intentionally a bit wider than strictly necessary today (reviewer fields, notes) because expanding an audit schema after entries exist is operationally painful — additive fields require nullable defaults and backfill rituals.

**Revisit:** If we add models with very different feature shapes (e.g., a graph-based F4 model), confirm the JSON column scales — likely fine, but worth verifying. If the project's scope ever grows to a real production system, add a `pii_redacted` flag and a redaction policy.

---

## 2026-06-05: Week 1 tracer bullet — what integration issues did I find? (Task 1.10)

**Context:** Task 1.10 fires one shot through the whole system — raw CSV → trivial feature → trivial model → evaluation → triage → audit log — before any layer is real. The build guide is explicit that the result will be terrible and that the point is finding integration problems early. It found five. Every one of them would have surfaced weeks later, at a worse time, attached to a real model run.

**Integration issue 1 — the audit logger's throughput assumption broke on first contact.** Task 1.9's `log_event` opens a DuckDB connection per call, and its own docstring recorded the assumption that justified it: "max ~100 calls/run." The tracer's test split is 20,000 decisions. At per-call connection cost, logging every decision would have taken minutes against a 5-minute budget for the whole pipeline. My first draft worked around this by sampling 1 in 100 audit entries — which violates both the build guide ("logs every decision") and CLAUDE.md §3.9 ("decisions that don't produce log entries are bugs"). I reverted that. The honest fix was a batched `log_events` writer in `audit/logger.py`: one connection, one `executemany`, ~1s for all 20k rows. Both writers now share one serialization helper and one private write path, so they can't drift. The lesson worth keeping: when a component's design assumption breaks, the fix is to change the component, not to quietly degrade the invariant it serves.

**Integration issue 2 — a TOML table-placement bug in `pyproject.toml`.** I added `[project.scripts]` between `license` and `dependencies`. A TOML table extends to the next table header, so `dependencies` silently became `project.scripts.dependencies` — a string-typed table suddenly holding an array. Ruff's RUF200 caught it before `uv` did. Moved the table below the last bare `[project]` key and left a comment explaining why it must stay there.

**Integration issue 3 — typer collapses a single-command app.** With exactly one registered command and no callback, typer treats the app as a root command, so the documented `sentry pipeline --sample` failed with "unexpected extra argument (pipeline)". The integration test caught it on the first full run. Fix is the canonical one: an explicit `@app.callback()` keeps the app in subcommand mode. Later weeks hang `train`/`predict`/`report` off the same group.

**Integration issue 4 — console scripts only exist after an image rebuild.** The `sentry` entry point is generated at install time by `uv sync`, not read live from the bind mount. Code changes appear in the container instantly; `[project.scripts]` changes don't. Cost me one confused minute now; recorded so it doesn't cost more later.

**Integration issue 5 — the terrible result is more informative than a mediocre one would have been.** PR-AUC came out at 0.0020 — *below* the 0.00227 base rate — and ROC-AUC at 0.19, meaning the model ranks actively backwards. The audit log explains why: the logistic regression learned a positive coefficient on `clicks_per_ip`, i.e. "more clicks from an IP → more likely to convert," which is exactly the inversion the Week 1 EDA flagged (single-click IPs convert at 0.86%, multi-click IPs at 0.07–0.12%). A whole-dataset count with no time discipline encodes the wrong direction. Also: all 20,000 decisions came out ALLOW, because the maximum raw score was 0.0029 against a 0.5 block threshold — round-number thresholds never fire when the base rate is 0.2%. That is the cheapest possible preview of why Task 5.2 selects thresholds from a cost model over the actual score distribution.

**Two smaller calls made during the task:**

1. **The tracer's triage is two-way (AUTO_BLOCK / ALLOW per the build guide), so the audit entries log `threshold_review == threshold_block`.** The first draft logged a fabricated review threshold of 0.3 that the decision logic never consulted — meaning a replay of the logged policy would produce HUMAN_REVIEW actions that were never taken. An audit log that misstates the active policy is worse than no audit log. Equal thresholds make the recorded policy replay to the actions actually taken.

2. **`--sample` owns input selection, and a bare `sentry pipeline` exits with an error pointing at it.** The first draft accepted `--sample` as a purely informational flag with the sample CSV as a silent default. That teaches users the flag is optional — and in Week 4, when the bare invocation gains full-data semantics, "optional" becomes a surprise multi-hour 200M-row run. Failing loudly now is cheaper than surprising someone later.

**Performance debt noted, deliberately not paid now:** DuckDB's `executemany` is a row-at-a-time insert path; at Week 5-6 audit volumes the right shape is registering the batch as an Arrow/pandas table and inserting via `INSERT INTO ... SELECT`. The tracer also re-ingests the CSV and materializes the whole dataset in pandas on every run — fine at 100k rows, dead at 200M; the Week 2 split views replace that pattern, and it should not be copied forward.

**Confidence:** High. The task did what the build guide said it would — every layer is now connected and tested, and each of the five issues above was found by running the system, not by reading it.

**Revisit:** The batched-insert shape when audit volume scales (Week 5-6). The `--sample`/full-data CLI semantics when Week 4 gives the bare invocation real meaning.

---

## 2026-06-05: Week 1 retrospective (Task 1.11)

**What got built vs. what was planned.** Everything Week 1 scoped, nothing extra: repo structure, Docker (native arm64 after the Rosetta reversal), uv-pinned dependencies, ruff/black/mypy/pytest with a 70% coverage gate, the 40-row hand-crafted fixture, DuckDB ingestion and validation, ten EDA queries, the evaluation harness with PR-AUC as the headline, the audit schema and logger, and the tracer-bullet pipeline. 36 commits, 19 decisions entries, about 1,100 lines in `src/sentry/`. More importantly: there is no line in there I couldn't explain to an interviewer, which was the actual goal of going this slowly.

**Slower and faster than expected.** Slower: environment work, consistently. The amd64→arm64 reversal cost an evening — including 35 minutes of watching `import lightgbm` spin under Rosetta before accepting that the original portability argument was wrong. The Kaggle download hit a rules-acceptance 403 and a macOS permissions quirk that together ate the better part of an hour. A full disk killed the Task 1.10 session mid-task. Faster: the SQL-heavy work. DuckDB did what its docs said it would, and the single-query validation refactor worked on the first run.

**What the data taught me.** The EDA inverted my main intuition: single-click IPs convert at 0.86%, multi-click IPs at 0.07–0.12% — click volume is a fraud signal, not an engagement signal. `device=0` converts at sixty times baseline. Fraud spreads thin across many apps rather than bursting on one. The tracer then proved the inversion the hard way: a whole-dataset click count fed to a logistic regression learned the wrong direction and scored ROC-AUC 0.19, worse than random.

**What the tracer surfaced.** Five integration issues, written up in the Task 1.10 entry. The one that changes how I work: when the audit logger's per-call connection design broke at 20k decisions, my first instinct was to sample 1-in-100 entries — quietly degrading the every-decision invariant instead of fixing the component. I caught it, but only because the invariant was written down in two places. Invariants need to be specific enough on paper that the late-night version of me can't negotiate with them.

**Changes for Week 2.** (1) Steadier cadence — Week 1's work landed mostly in two marathon days bracketing a nine-day gap; smaller sessions, more of them. (2) Check disk space before any large artifact lands; the full disk was foreseeable from the 27 GB note in my own session log. (3) Keep the pre-commit review pass — it caught an audit-integrity flaw (logged thresholds that wouldn't replay to the actions taken) I would otherwise have shipped.

**On track?** Technically, yes — every end-of-week checklist item is done and the foundations feel solid rather than rushed. Calendar-wise, "Week 1" took seventeen days from first commit to tracer bullet. I didn't log hours precisely, but the work plausibly fit the 12–16 hour budget; the problem was distribution, not volume. I'm not re-baselining the plan yet. If Week 2 also runs well past ten calendar days, I'll update the PRD schedule then and name the gap pattern as the cause, rather than pretending the work was underestimated.

---

## 2026-06-05: Time-based split boundaries (Task 2.1)

**Context:** The split must exist before any real feature work — features computed across split boundaries are the subtle form of leakage (a "global mean conversion rate" computed over train+val+test leaks test information into training). CLAUDE.md §3.1 locks the method (time-based, never random) and the ratio (60/20/20); the decision here is where exactly the boundaries fall.

**The boundaries:**

| split | interval (half-open) | rows (100k sample) |
|---|---|---|
| train | [data start, 2017-11-08 13:00) | 60,336 (60.3%) |
| val | [2017-11-08 13:00, 2017-11-09 05:00) | 20,087 (20.1%) |
| test | [2017-11-09 05:00, …) | 19,577 (19.6%) |

**How they were chosen.** The sample spans 2017-11-06 16:00 → 2017-11-09 16:00 UTC — exactly 72 hours. The exact 60th/80th row-count quantiles of `click_time` land at 11-08 12:48:34 and 11-09 04:48:50 (not at the 60%/80% marks of elapsed time — traffic has a strong diurnal cycle, so row quantiles and time quantiles differ). Rounding each to the nearest hour gives boundaries that hit 60.3/20.1/19.6 by rows. Hour-aligned timestamps are human-readable, quotable in an interview, and don't pretend to a precision the choice doesn't have.

**Why not whole calendar days.** The build guide frames the decision as "how many days for train/val/test" — and day blocks are the cleaner story when the data permits it. Here it doesn't: the dataset covers four calendar dates of which the first (8h) and last (16h) are partial. The best day-block assignment (train = 11-06+11-07, val = 11-08, test = 11-09) yields roughly 44/33/22 — giving up a quarter of the training data and overshooting val by 13 points against a locked 60/20/20. The day-block argument's main benefit — every split sees a full diurnal cycle — is unavailable regardless, because train starts at 16:00 and test ends at 16:00 no matter where the interior boundaries fall. So I honored the locked ratio on hour boundaries. The leakage protection that matters does not come from boundary placement anyway; it comes from the §3.4 feature-window discipline (strictly-prior windows, per-split source data), which is independent of where the cuts land.

**Why the split is defined by time, not by row count.** The constants are pinned timestamps, not "first 60% of rows." Two reasons. First, identical semantics across datasets: the 100k sample and the full 184M-row set get the same temporal split, so feature windows and evaluation comparisons mean the same thing at both scales (the sample is a uniform row subsample, so its row fractions transfer — measured 60.3/20.1/19.6). Second, reproducibility: a quantile recomputed at run time would shift if the data were re-ingested, re-sampled, or filtered; a pinned timestamp cannot drift silently.

**Half-open intervals, exclusive upper bounds.** Same convention as `TRAIN_DATE_MAX_EXCLUSIVE` from Task 1.6: a row at exactly 11-08 13:00:00 is val, not train. `<=` boundaries lose or double-count the boundary second; a test (`test_boundary_rows_land_exclusively`) pins the semantics with rows placed exactly on each boundary.

**Test-set guard mechanisms (build guide requires two):**
1. **View separation.** All feature and model work queries `clicks_train`/`clicks_val`. The test view exists (the evaluation in Task 4.7 needs it) but no development path touches it.
2. **Loud access warning.** `apply_split(df, "test")` emits a structlog WARNING (`test_split_accessed`) on every call. Not a hard block — Task 4.7 must be able to run — but a stray test read during development shows up in the logs instead of passing silently. A test asserts the warning fires for test and stays quiet for train/val.

I considered the third option (physically separate DuckDB files per split) and rejected it: three copies of a 184M-row table is real disk (the resource that killed a session last week), and views are always consistent with the source table where copies drift.

**Known limitations, stated up front.** (1) Val and test cover different hour-of-day mixes (val: 13:00→05:00, test: 05:00→16:00). Unavoidable in any contiguous temporal split of a 72-hour window; it will show up as a calibration shift between val and test, which is exactly what the isotonic calibrator (Task 4.5) is fit to absorb — worth re-checking when calibration results land. (2) At sample scale, val holds only ~39 positives, so val PR-AUC is noisy during development; at full scale val is ~37M rows / ~80k positives and the noise vanishes. Development decisions made on sample-scale val numbers should be treated as directional only.

**Confidence:** High on method and mechanisms; medium on the exact hours — if the full dataset's row quantiles land materially differently from the sample's, the boundaries are two constants and one decisions entry away from being re-derived (and re-deriving them BEFORE any features are computed is free; after, it invalidates every cached feature).

**Revisit:** After full-data ingest (Week 4), verify the measured split fractions still round to 60/20/20. After Task 4.5, check the val→test calibration shift attributable to the diurnal-mix difference.

---

## 2026-06-05: Feature pipeline framework (Task 2.2)

**Context:** Features accumulate from here through Week 3 (F1 pass-throughs, F2 velocity windows, F3 aggregates, maybe F4 graph). Without a shared framework each feature becomes its own script with its own loading and writing conventions. The framework's job is consistency and two specific safety properties: dependency ordering and row alignment.

**The SQL-vs-Python rule (build-guide stop-and-think).** Adopted the guide's recommendation as a hard rule: if a feature can be expressed as an aggregation, join, or window function, it is SQL against the split view; Python (operating on the accumulated frame) is reserved for what SQL can't express cleanly — cross-row statistics, composite scores reading several other features. Tie-breaker: if I'm debating, it's SQL — DuckDB does the scan once and in parallel, and the SQL file doubles as documentation a reviewer can read without running anything.

**Row identity: `row_id` assigned at ingest.** The framework's central correctness risk is alignment — a SQL-computed feature must attach to exactly the row that produced it. `(ip, click_time)` is not a key: the Week 1 EDA found same-IP clicks in the same second (and the Task 1.10 tracer's `case_id` already collides on exactly this). So ingestion now assigns `row_id` = `row_number()` over a total ordering (click_time, then every dimension column as tie-breaker; fully identical rows are interchangeable, so any stable assignment among them is correct). Every SQL feature returns `(row_id, value)` and the pipeline joins on it — result order is irrelevant, and a query that drops or duplicates rows raises instead of silently shifting values onto wrong rows. The alternative — positional alignment by a repeated ORDER BY convention in every feature query — works until exactly one query forgets, and the failure is silent. An 8-byte column at 184M rows is a cheap price for making misalignment structurally impossible. Side effect: the three corrupt-row INSERTs in the validation tests were positional and broke when the column landed — converted to explicit column lists, which is what they should have been anyway.

**Two feature kinds, a union type, no inheritance.** `SqlFeature` (query template, `{source}` placeholder) and `PythonFeature` (function over the accumulated frame), `Feature = SqlFeature | PythonFeature`. A base class bought nothing — the two kinds share fields but not behavior, and `isinstance` dispatch in one place is more readable than virtual methods in two. The guide's `compute(df_or_db)` single-protocol shape would have forced every feature to handle both input types; splitting by kind means each handles exactly one.

**Topological sort at construction, not compute time.** Kahn's algorithm, stable (input order preserved among ready features), with cycles, unknown dependencies, and duplicate names all raising in `FeaturePipeline(...)` — a bad feature graph fails when the pipeline is built, not 40 minutes into a full-data feature run. Dependencies name other *features* only; base columns are always available and never declared.

**Caching = the accumulating frame.** The guide asks for intermediate-result caching. The accumulating DataFrame is the cache: each feature computes once, later features read earlier ones as columns. A separate cache layer (keyed what? invalidated when?) is exactly the speculative abstraction CLAUDE.md §7.13 forbids. If Week 3's F3 aggregates need cross-run caching, that's the feature store's job (Task 2.5), not the pipeline's.

**`output_dtype` is declared metadata, not enforced casting.** The pipeline does not `astype` feature outputs — casting here would mangle NULLs (int columns can't hold them) before LightGBM gets to handle them natively. The declaration documents intent and will feed the feature store's `metadata.json` (Task 2.5); enforcement, if ever needed, belongs at the store boundary where Parquet schemas are written.

**Known deferred cost:** `compute()` loads the split's full base table into pandas before attaching features. Fine at sample scale; at 184M-row full scale this is the same materialization pattern flagged in the Task 1.10 entry. The Week 4 full-data run will need either chunked computation or pushing the whole feature join down into DuckDB — decide when the real memory numbers are known, not before.

**Confidence:** High on row_id, topo-sort-at-construction, and the SQL-vs-Python rule. Medium on the two-kind union surviving contact with F3/F4 — if a feature ever needs both the connection and the accumulated frame, the union grows a third kind or the SQL kind gains dependencies on computed columns; the decision is one dataclass away either way.

**Revisit:** At Task 2.5, whether `output_dtype` should be enforced when writing Parquet. At Week 4 full-data scale, the base-table materialization strategy.

---

## 2026-06-05: F1 per-click features (Task 2.3)

**Context:** Eight per-click features — four raw categorical pass-throughs (app, channel, device, os), two time extractions (hour, day-of-week), two interaction pairs (ip×app, ip×device). Each lives in its own SQL file under `sql/02_features/` and registers as a `SqlFeature` at import. F1 is trivially leakage-free: every value derives from the click's own row.

**Why raw `ip` is not a feature (build-guide stop-and-think).** The sample has ~35k distinct IPs in 100k clicks; the full data has millions. Most appear a handful of times. As a categorical, `ip` invites the model to memorize "IP 87532 was fraudulent in training" — which doesn't generalize past the training window and dies entirely when the adversary rotates IPs. IPs enter the model only through behavior derived FROM them: velocity (F2), historical aggregates (F3), and the interaction pairs — which carry pair-level signal (the EDA found 99.7% of (ip, app) pairs with ≥5 clicks never convert) while the model layer decides how to encode their cardinality (Week 4; LightGBM handles categoricals natively).

**Why hour_of_day is a raw integer, not sin/cos (build-guide stop-and-think).** Cyclic encoding exists so LINEAR models can represent "23:00 is close to 01:00." Trees don't need it: LightGBM splits on thresholds and can carve any hour subset it wants in two splits. The cost of sin/cos would be two opaque columns an interviewer (or I) can't read in a SHAP plot; `hour=3` is self-explanatory. If a linear baseline in Week 4 wants cyclic encoding, that's a model-layer transform, not a feature-layer one.

**`isodow` over DuckDB's `dayofweek`.** DuckDB's `dayofweek` is Sunday=0; pandas' is Monday=0; ISO is Monday=1..Sunday=7. Off-by-one weekday bugs between engines are exactly the kind of silent error this project can't afford, so the SQL uses `isodow` (unambiguous, standard) and the test pins the convention against an independent pandas computation. Worth recording: the TalkingData window is Mon-Thu, so this feature has four distinct values and likely little importance — it's built because the spec asks and it's nearly free, and the ablation (Week 5) will tell us its actual worth.

**SQL files as the source of truth.** The Python module is a thin loader; the reviewable artifact is the .sql file with its header comment. Each file states the `(row_id, value)` contract and is `{source}`-parameterized so the same definition runs against any split view — the feature definition cannot hardcode its way across the split boundary.

**Confidence:** High. These are the simplest features in the project; the value was cementing the pattern (SQL file → SqlFeature → pipeline) before F2's window functions raise the difficulty.

**Revisit:** Interaction-pair dtype is `string` — check memory at full scale (184M × two string columns is real); a hash-to-int64 encoding is the fallback if Parquet/pandas balloon.

---

## 2026-06-05: F2 velocity features (Task 2.4)

**Context:** Six trailing-window features describing the IP's recent behavior — counts (1h, 24h, ip×app 1h), inter-click gap, gap stddev, and a composite burst flag. This is where the §3.4 leakage discipline becomes concrete SQL, and where the frame-boundary subtleties verified earlier today (in a read-only exploration session against the train view) got encoded into the actual features.

**The strictly-prior frame, and the trap worth remembering.** All windows use `RANGE BETWEEN INTERVAL <X> PRECEDING AND INTERVAL 1 MILLISECOND PRECEDING` — the exact CLAUDE.md §3.4 pattern. The verification that mattered: `EXCLUDE CURRENT ROW` looks like it solves self-inclusion but still counts *same-second peers* (frame peers aren't the current row, yet they aren't strictly prior either). Measured on the train split: 20 rows — all members of same-second pairs, i.e. exactly the bot-like rows — would have leaked under `EXCLUDE CURRENT ROW`. `EXCLUDE GROUP` is the verified-identical alternative (0 differences over 60,336 rows); I chose the millisecond-preceding spelling because it's the documented §3.4 pattern and reads as "everything strictly before t" without requiring the reader to know what a frame peer group is. Bounds are inclusive both ends — a click exactly 1h prior is in the window — stated in each SQL file.

**Why both 1h and 24h count windows (build-guide stop-and-think).** Different adversaries are invisible to each other's window: a click farm doing 100 clicks in 5 minutes maxes the 1h count while barely moving the 24h average; a botnet pacing one click a minute never looks bursty in any single hour but accumulates ~1,440 clicks in the 24h window. Two windows let the model learn either signature. The SHAP check bore this out asymmetrically at sample scale: the 24h count landed third overall (0.43 mean |SHAP|) while the 1h count was near-zero — 100k sampled rows are too sparse for many IPs to show intra-hour density. Expect the 1h window to matter more at full scale; the ablation (Week 5) will say.

**First-click value: NULL, except where 0 is the truth (build-guide stop-and-think).** `f2_inter_click_time_seconds` returns NULL for an IP's first click — there is no previous click, and NULL is the honest representation; LightGBM routes NULLs natively, and ~44% of train rows are first clicks, so a sentinel like 999999 would put a fake number into nearly half the column. The count features, by contrast, return 0 for an empty window because 0 *is* the true count of prior clicks — NULL there would be false modesty. The stddev returns NULL below two usable gaps (mathematically undefined). One convention per mathematical reality, not one global rule.

**Tie order pinned with `row_id`.** `lag()` over `ORDER BY click_time` alone is unspecified between same-second clicks. `ORDER BY click_time, row_id` makes it deterministic — within a same-second pair, one row is "first" (NULL or prior gap) and the other gets gap=0, reproducibly. This is the Task 2.2 row_id decision paying off a second time. The RANGE-framed counts don't need the tiebreaker (peers get identical counts by definition), so they don't carry it.

**Stddev design: prior gaps only, current gap excluded.** The current click's own gap is already its own feature; `f2_ip_click_std_inter_arrival` summarizes the rhythm *before* this click. Two-layer SQL because windows can't nest: `lag()` in a CTE, windowed `stddev_samp` over the gap column. Low stddev = metronomic pacing (the 12-click 10s-apart test case gets exactly 0.0); humans jitter.

**Burst thresholds pinned from train EDA: count > 10 AND gap < 60s.** N=10 is the train split's p99 of the trailing-1h count; 60s sits in the fastest ~2% of gaps. At those values, 127 train rows fire — with zero conversions against the 0.237% baseline, a clean bursty-bot signature. These constants define the *feature*; operating thresholds for actions come from the Task 5.2 cost model. Burst is the first production `PythonFeature` (composite over two computed features — exactly the case the Task 2.2 SQL-vs-Python rule assigns to Python), and the first production use of the pipeline's dependency machinery.

**SHAP check (AC):** small LightGBM (100 trees, class_weight=balanced, seed 42) on F1+F2 over the train split, SHAP on a 10k sample. F2 carries **43.6% of total |SHAP|**; `f2_inter_click_time_seconds` is the **top feature overall** (0.75), ahead of `f1_app_id` (0.74). Two honest negatives recorded: `f2_burst_score` scored 0.0000 — given trees already see both its parents, the composite is informationally redundant (kept for the human-readable triage story; expect the Week 5 ablation to confirm ~zero marginal lift) — and `f1_ip_app_interaction` is ~0 at sample scale (mostly-unique pairs on 60k rows; revisit at full scale).

**Confidence:** High on frames, NULL conventions, and tie-breaking — these are verified behaviors, not beliefs. Medium on the burst thresholds (definitional, deliberately crude) and on extrapolating the 1h-vs-24h importance split to full scale.

**Revisit:** 1h-window importance and `f1_ip_app_interaction` at full-data scale (Week 4). Burst-score redundancy at the Week 5 ablation — if it's still zero, the decision to keep or drop it belongs in the policy discussion, not the feature code.

---

## 2026-06-06: Versioned feature store (Task 2.5)

**Context:** Feature computation is cheap at sample scale and hours at full scale; model iteration shouldn't pay it twice. The store persists one Parquet per split under a semver-ish version directory with one `metadata.json` per version. `v0.1.0` = F1+F2, materialized today for train (60,336 × 14 features) and val (20,087) — the test split's features are deliberately not computed until Task 4.7 needs them.

**Immutable versions.** Re-saving an existing version+split raises `FileExistsError`. A versioned store that silently overwrites is a cache with extra steps: the version string is a claim ("v0.1.0 is F1+F2 computed with these definitions"), and mutating its contents makes every downstream comparison against it a lie. Changed feature definition = bumped version. The metadata records per split: row count, source string, ISO timestamp, and the Parquet file's sha256 — enough to detect drift and trace provenance without opening the file.

**Feature list derived by naming convention.** The metadata's `features` array is the table's `f<N>_*` columns — the project-wide naming convention does the work, no extra parameter to pass wrong. Parquet itself stores the real dtypes (which resolves the Task 2.2 question of whether `output_dtype` needs enforcement at the store boundary: pyarrow already preserves actual types; the declared dtype remains documentation).

**Known limitation made concrete here — split-boundary window artifacts.** Per the locked §3.4 rule, each split's features are computed from that split's source view only. Consequence: a val row in the first hour of the val window has a trailing-1h count that cannot see train-period clicks — the IP's true history is truncated at the split boundary. Roughly the first hour of val (and later, test) rows are systematically undercounted; the alternative (windows over all strictly-prior data regardless of split) would be production-realistic but violates the locked cross-split rule, and the rule errs on the side that can never contaminate. Recording rather than relitigating: the artifact is identical in kind for val and test, so comparisons between them are fair; flag if Task 4.5's calibration shows a val/test shift concentrated in early-window rows.

**Confidence:** High. **Revisit:** the boundary artifact at calibration time; whether Week 3's F3 aggregates (which window over much longer history) make the artifact material enough to discuss in `docs/tradeoffs.md`.

---

## 2026-06-06: Feature property tests (Task 2.6)

**Context:** Correctness tests (hand-crafted known answers) landed with 2.3/2.4. This task adds the cross-cutting properties: edge cases (empty/single-row/all-same-IP/all-different-IP), idempotence, a no-leakage proof, a frozen schema contract, and the build guide's deliberately-broken-feature test.

**The no-leakage test design.** Two databases: one holding only train-period rows, one holding the same train rows plus 50 future rows for the same IP. Train features computed through `clicks_train` must be byte-identical across the two — if any value shifts, something read beyond the split view. This is leakage tested *by construction* rather than by code review.

**The broken feature is this week's actual trap.** The deliberately-buggy feature uses `AND CURRENT ROW` framing — the exact leak verified during F2 work — and the test asserts it FAILS the strictly-prior expectations (off by exactly the self-inclusion +1). If that test ever passes, the suite has stopped doing real work.

**Substrate fidelity finding (the bug this task caught in its own tooling).** The first frozen-schema run failed: test helpers created `clicks` tables from pandas frames, which lands BIGINT where real ingestion lands INTEGER — so the test was about to freeze dtypes production would never produce. Fixed by routing every feature test through one shared `build_clicks_db` builder that casts to the ingestion schema (`* REPLACE (CAST ...)`) — test substrate now matches production by construction. Two dtype facts recorded while freezing: DuckDB returns the lag-derived gap column as pandas *nullable* `Int64` (not float64-with-NaN), and VARCHAR comes back as pandas 3's native `str` dtype. Both are now contractual; LightGBM consumed `Int64` without complaint in the SHAP check, but the model layer (Week 4) should re-verify on the real training path.

**Confidence:** High. **Revisit:** `Int64`-vs-LightGBM at Week 4 model training.

---

## 2026-06-06: Tracer bullet on real features (Task 2.7)

**Context:** The Week 1 tracer's trivial layers get replaced as real ones land. `sentry pipeline --sample` now runs: ingest → canonical split views → F1+F2 through the feature pipeline → scaled logistic regression with class weights → evaluation harness on VAL → triage with an audit entry per decision. 18.6s end-to-end.

**The tracer now evaluates on val, not test.** Week 1's tracer carved its own ad-hoc 60/20/20 and evaluated on its private test slice — written before `splits.py` existed, per the build guide. With canonical splits in place, a tracer that touched `clicks_test` on every run would burn the §3.1 test-once discipline for nothing. The tracer is development tooling; development evaluation happens on val.

**The comparison (AC: "measurably better").** Same protocol for both (train on `clicks_train`, evaluate on `clicks_val`, scaled logreg with balanced class weights): the Week 1 trivial feature scores PR-AUC 0.0016 / ROC-AUC 0.14 — *below* the 0.0023 base rate, anti-predictive, as diagnosed in the Task 1.10 entry. F1+F2 scores **PR-AUC 0.0102 / ROC-AUC 0.77** — 6.4x the trivial PR-AUC, 4.5x the base rate, and the ranking direction is right. Real features beat a leaky aggregate even through a crude linear model.

**Brier got WORSE (0.0022 → 0.29), and that's expected.** `class_weight="balanced"` re-weights the loss as if classes were even, so predicted probabilities center near 0.5 instead of near the 0.2% base rate — terrible calibration, much better ranking. The Week 1 tracer's lovely Brier was the calibration of a model predicting "almost zero for everyone," which is vacuously well-calibrated and operationally useless. This is the cleanest demonstration yet of why the project reports PR-AUC as the headline (§3.2) and fits an isotonic calibrator (Task 4.5) before anyone reads scores as probabilities.

**Tracer-only compromises, documented:** the two string interaction features are excluded (logreg can't consume categoricals; the Week 4 LightGBM will), and NULLs are filled with a −1 sentinel (§3.4 allows a sentinel when recorded; logreg can't route NULLs, LightGBM natively does). Both go away with the real model.

**Audit upgrades:** `case_id` is now `tracer-row{row_id}` — stable and collision-free, closing the Week 1 entry's same-second collision; and every entry carries top-5 per-row contributions, computed as `coef_j × scaled_x_ij` — the exact additive term in the linear model's logit, i.e. the linear analogue of a SHAP value until the tree model brings real SHAP in Week 4.

**Operational note found in passing:** a leftover interactive DuckDB session (read-only REPL from the window-function exploration) held a shared lock on `artifacts/sentry.duckdb` and blocked the pipeline's write connection with a cryptic "lock held in PID 0" (cross-container, so no real PID). Close REPLs before pipeline runs; the error names the fix poorly.

**Confidence:** High. **Revisit:** drop the −1 sentinel and string-feature exclusion the moment LightGBM lands (Week 4).

---

## 2026-06-06: Week 2 retrospective (Task 2.8)

**What got built.** All of Week 2: canonical time splits with guard mechanisms, the feature pipeline framework with row_id alignment, eight F1 features, six F2 velocity features, the versioned feature store with v0.1.0 materialized, property-level tests (no-leakage-by-construction, frozen schema, a deliberately broken feature), and the tracer rebuilt on real features. 91 tests, ~97.5% coverage, all green.

**F1 surprises.** Two, both downward. `f1_ip_app_interaction` — which the EDA suggested would be strong (99.7% of multi-click pairs never convert) — scored near-zero SHAP importance at sample scale, because 60k training rows make most pairs unique and a categorical the model sees once is noise. And `f1_device_id` barely registered despite the device=0 signal converting at 60x baseline; rare-but-strong signals don't move mean |SHAP| much. Both are worth re-measuring at full scale before concluding anything.

**F2 and the SQL I actually learned.** The headline: `f2_inter_click_time_seconds` is the most important feature in the model, full stop. The skill this week was window framing — specifically that "strictly prior" has two correct spellings (`INTERVAL 1 MILLISECOND PRECEDING`, `EXCLUDE GROUP`) and two wrong ones, and that the wrong one that *looks* right (`EXCLUDE CURRENT ROW`) leaks exactly on same-second peer clicks — the bot-fingerprint rows where leakage hurts most. I verified the four variants empirically against the train split before trusting any of them, and the deliberately-broken-feature test now keeps the trap from creeping back. Also picked up: `QUALIFY`, named `WINDOW` clauses, and pinning `lag()` tie order with `row_id`. The hard part wasn't syntax; it was convincing myself which rows are in the frame at the boundaries — the same-second pair and the exactly-one-hour-prior click are now both pinned in tests because I had to hand-compute them anyway.

**The honest process note.** Week 1's retro promised steadier cadence. Week 2 happened in roughly two intense days instead — the opposite of steadier, even if the calendar gap didn't recur. The difference this time: every task closed fully (tests, docs, commit, push) before the next opened, so an interruption would have cost nothing. That discipline, more than session length, seems to be what actually protects the project.

**On schedule?** Ahead. Week 2's 12-15 hour budget was met with margin, and the foundations from Week 1 paid off — nothing needed rework. Week 3 (F3 historical aggregates, possibly F4 graph features) is methodologically the hardest feature work in the project; the leakage discipline built this week is exactly what it will lean on.

---

## 2026-06-06: Avoiding label leakage in F3 (Task 3.1)

**Context:** F3 aggregates an entity's rolling history, and for the conversion-rate features that history includes `is_attributed` — the label. Every feature before this one could leak *timing* information at worst; these can leak *the target itself*. A model trained on leaked labels performs spuriously well offline and collapses at deployment, which is the precise failure CLAUDE.md §3 calls fail-the-interview-level. This entry writes down the exact computation and every mechanism that keeps it honest.

**The exact computation (build-guide stop-and-think, answered precisely).** `f3_ip_conversion_rate_24hr` for a click at time t from IP i is: `AVG(is_attributed)` over all clicks from i with `click_time` in the closed interval `[t − 24h, t − 1ms]`. Three properties follow: (a) the current row is NOT in the window — the frame's upper bound is one millisecond before t, so neither the row itself nor any same-second peer (the `EXCLUDE CURRENT ROW` trap from Week 2, which would have been wrong here too) can contribute; (b) the current row's label therefore cannot enter its own feature value by construction, not by convention; (c) an IP with zero prior clicks gets NULL — `AVG` over an empty frame — never 0.0 (which asserts "known non-converter") and never 1.0 (which is what its own leaked label would produce).

**The three failure modes the build guide names, and what blocks each.** *Current-row inclusion:* blocked by the frame bound, and pinned by the canary test — a synthetic first click with label=1 must produce NULL; if it ever produces 1.0, the window has started seeing its own row. *Computing rates over the full dataset and joining back:* structurally impossible through the pipeline — F3 runs as `{source}`-templated SQL against one split view, and the Task 2.6 blind-to-future-rows test proves train features don't change when val/test data exists. *The cumulative-sum trick (`ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`):* not used anywhere, and the Task 2.6 deliberately-broken-feature test keeps the `AND CURRENT ROW` framing pattern permanently recognizable as a bug in this codebase.

**The label-flip test — the strongest assertion in the suite.** Two datasets identical except one row's label is flipped. The flipped row's OWN feature values must be byte-identical across the two runs (its label never feeds back into itself), while a later row that holds the flipped row in its window MUST move (1/3 → 2/3 in the hand-computed case). Labels flow forward in time only — asserted in both directions, not just the comfortable one.

**NULL for rates, 0 for counts — same convention as F2, same reason.** A rate with no observations is mathematically undefined → NULL (LightGBM routes it natively). A distinct-count with no observations is genuinely zero → 0. One convention per mathematical reality.

**The honest production caveat.** These features assume a prior click's label is knowable at decision time. It isn't, quite: attribution arrives at `attributed_time`, which the Week 1 EDA showed lags clicks by minutes to hours (and one impossible 2-second case). Offline, using prior-click labels is the standard Kaggle-established protocol and what the build guide specifies; in production, the same feature needs a label-maturation window ("only count clicks older than X hours, treated as unconverted if unattributed by then") plus handling for the censoring that introduces. Recorded as a portfolio-scope boundary in the SQL header and here — the feature is honest about time, and this caveat is about being honest about *when knowledge arrives*.

**Confidence:** High — every claim above is enforced by a test, not a comment. **Revisit:** label-maturation design if this project ever grows a production story (docs/tradeoffs.md candidate for Week 7).

---

## 2026-06-06: The pairwise conversion rate and the small-denominator call (Task 3.2)

**Context:** `f3_ip_app_conversion_rate_24hr` — the expected dominant feature per public solutions on this dataset, built with the Task 3.1 leakage discipline (same strictly-prior frame, same NULL convention, same canary test at pair granularity).

**The small-denominator decision (build-guide stop-and-think): ship the denominator, don't smooth.** A pair with 1 prior click has a rate of 0% or 100% — numerically extreme, statistically almost meaningless. Three options were on the table. (a) *Bayesian smoothing toward a prior* — rejected: it bakes a prior choice into the feature (smoothed-toward-what? computed-on-which-split?), turns every SHAP value and audit-log entry into "0.21, smoothed" which no reviewer can interpret without knowing the prior, and the prior itself becomes split-coupled state to manage. (b) *Minimum-N before non-NULL* — rejected: it blanks exactly the rows that matter most; the EDA's smoking gun is thin-history pairs that never convert, and a 3-click/0-conversion pair is already informative. (c) *Raw rate plus the denominator as its own feature* (`f3_ip_app_clicks_24hr`) — chosen: trees do their own discounting natively (split on "denominator small?" then on the rate), both columns stay interpretable in SHAP and in audit entries, and no prior is invented. The cost is one extra column. This is also the standard GBM treatment of the problem, which is worth being able to say plainly in an interview.

**Confidence:** High. **Revisit:** if the Week 4 model shows the rate dominating while the denominator goes unused, check whether LightGBM's splits actually condition on it (tree-path inspection) before believing thin rates are being taken at face value.

---

## 2026-06-06: F4 descope (Task 3.3)

**The decision:** the two F4 degree features are absorbed into F3 (one already existed as `f3_ip_distinct_apps_24hr`; the app-side degree landed today as `f3_app_distinct_ips_24hr` — fifteen minutes, same window pattern, same tests). The genuinely graph-structured feature — `f4_ip_app_jaccard`, similarity of an IP's app set to a "typical fraudulent IP" app neighborhood — is descoped. This entry records why, at the depth the decision deserves, because "what I chose not to build" is part of the engineering story.

**What graph features measure that aggregates don't.** Every F2/F3 feature describes one entity's own history: this IP's velocity, this pair's conversion rate. Graph features describe an entity's *position in the network of relationships*: which apps an IP clicks defines a neighborhood, and IPs with near-identical neighborhoods are likely operated together. The bipartite ip-app click graph makes coordinated infrastructure visible — a botnet's hundred IPs each look individually plausible (moderate velocity, mixed devices), but their app sets are stamped from the same template. Degree features are the first moment of that graph; similarity features like Jaccard are the second, and the second is where coordination actually shows.

**Why they target sophisticated adversaries.** Tier 1 fraud (farms, naive bots) trips velocity: it's fast because it's cheap. Tier 2 learns to pace — velocity features go quiet, but per-entity aggregates (conversion rates) still catch them because the *outcome* doesn't change. Tier 3 distributes: many IPs, low per-IP volume, plausible pacing, each entity individually under every threshold. Per-entity features are structurally blind to Tier 3 — the signal isn't in any single entity's history but in the *correlation across entities*, which is exactly what neighborhood similarity measures. Real T&S platforms run clustering/graph systems for this reason.

**Why the cost-benefit fails here.** Three costs stack up. *Compute:* Jaccard against a reference set means materializing per-IP app sets over rolling windows and intersecting them — at 184M rows this is a join-explosion workload that DuckDB window functions don't express; it wants dedicated similarity infrastructure (MinHash/LSH) that would be its own week. *Methodology:* "the typical fraudulent IP app set" is defined by labels, so the reference set is a third label-aggregating artifact needing the same strictly-prior discipline as the F3 rates plus a maturation story — every leakage trap from Task 3.1, squared, for a feature whose reference set also drifts as attackers rotate apps. *Expected gain:* the TalkingData window is 72 hours of heavily-downsampled traffic; public solutions on this exact dataset won with aggregate features — pair conversion rates dominated, and graph-style features do not appear in the winning writeups. The realistic gain on this dataset is marginal; the cost is ~a quarter of a week plus a permanent increase in the feature pipeline's complexity.

**What production would do.** With real scale and live labels: (1) maintain the bipartite click graph in a system built for it (a graph store or periodic Spark/GraphFrames jobs), not in the feature pipeline; (2) compute MinHash signatures per IP over trailing app sets, LSH-bucket them, and emit "cluster size" and "cluster fraud density" as features — the densities computed with matured labels only; (3) treat clustering output as a *triage* signal (route whole clusters to review) as much as a model feature, because cluster-level actions are how coordinated fraud actually gets removed; (4) re-evaluate the graph features' marginal lift quarterly — their value rises exactly as adversaries get pushed out of Tiers 1-2 by the aggregate features, which is a moving equilibrium worth measuring rather than assuming.

**Confidence:** High on descoping the Jaccard for this project; the absorbed degree features cost nothing and complete the cheap part of the graph story. **Revisit:** only if Week 5's ablation shows Tier-3-shaped residual errors (clusters of false negatives sharing app sets) — that would be evidence the missing feature has real lift here after all.

---

## 2026-06-06: Feature importance preview — the sparsity finding (Task 3.4)

**The numbers:** default LightGBM (class weights, seed 42) on F1+F2+F3, train→val: **PR-AUC 0.2305, ROC-AUC 0.9042** (39 val positives). Top features by SHAP: `f3_app_distinct_ips_24hr`, `f2_inter_click_time_seconds`, `f1_channel_id`. The build guide's two expectations both missed: PR-AUC < 0.80, and `f3_ip_app_conversion_rate_24hr` ranked dead last instead of top-3. The guide says stop and debug — so this entry is the debug.

**The diagnosis, measured rather than guessed.** The pair rate's 24h window is empty for **84%** of train rows (feature = NULL); only 4.9% of rows have ≥3 prior pair clicks. IP windows: 47% empty. App windows: 0.2% empty, median 1,899 prior clicks. The cause is the dev substrate: `train_sample.csv` is a ~0.05% uniform row sample of 184M clicks, which thins per-pair traffic ~2000x while leaving per-app traffic dense (apps aggregate across all IPs). Feature importance at sample scale is therefore mostly a function of *which windows survive sampling* — exactly why every app-level feature dominates and every pair/1h-level feature flatlines. The computation itself is correct (hand-crafted known-answer tests, the label-flip test, and the NULL canary all pass), and PR-AUC 0.23 is far from the ≥0.95 leakage-suspicion zone.

**Why this is recorded as a finding and not failure.** The guide's expectations assume full-data density. The honest conclusions at sample scale: (1) the model improved 23x over the Week 2 baseline (PR-AUC 0.0102 → 0.2305) with ROC-AUC 0.90 — the feature families work; (2) sample-scale importance rankings must not drive feature decisions — the Week 5 ablation runs at training scale or it measures sampling noise; (3) **Week 4 gate:** after the real training sample is built (the Day-1 plan is ~10% time-stratified, 200x denser than this dev substrate), re-run this preview — `f3_ip_app_conversion_rate_24hr` must climb to top-3 and PR-AUC must clear 0.80, or THEN something is actually wrong.

**Confidence:** High on the diagnosis (it's measured). **Revisit:** the Week 4 gate above is mandatory, not optional.

---

## 2026-06-06: Week 3 retrospective (Task 3.6)

**What got built.** All of Week 3, with F4 deliberately reshaped: seven F3 aggregate features (two entity conversion rates, the pairwise rate with its denominator companion, three distinct-count fingerprints, the app-side degree absorbed from F4), the F4 Jaccard descoped with a full cost-benefit writeup, the importance preview, and the tracer upgraded to F1+F2+F3 (PR-AUC 0.0102 → 0.0168, ROC 0.77 → 0.86 on the linear baseline; 0.23 / 0.90 under preview LightGBM). 101 tests, ~97.5% coverage.

**The week's real lesson: measure before you panic.** The importance preview missed both build-guide expectations — PR-AUC 0.23 against ≥0.80, and the pairwise rate dead last against top-3. The first instinct ("the headline feature is broken") would have burned hours re-reading correct SQL. The actual cause took one query to establish: 84% of pair windows are empty on a 0.05% row sample, so the feature is NULL where it can't be anything else. The known-answer tests had already proven the computation; what failed was the *expectation's* assumption of full-data density. The general form — when a metric misses, first establish what the data can support before debugging the code — feels like the most transferable thing this project has taught me so far.

**Leakage discipline held where it mattered most.** F3 aggregates labels, the highest-risk construction in the project. The label-flip test (flip a row's label, assert its own features don't move, assert later rows' features do) is the test I'd show an interviewer first. The production caveat — prior labels aren't actually knowable at decision time because attribution arrives late — is recorded rather than solved, which is the right scope boundary for this project.

**The descope call.** F4's degree features turned out to be window aggregates wearing graph clothing — absorbed into F3 for fifteen minutes of work. The Jaccard similarity is real graph machinery with a real leakage surface (its reference set is label-defined) and no evidence of lift on this dataset; descoped with the production design written down. Choosing not to build it felt more useful than building it would have.

**On schedule?** Ahead — Weeks 2 and 3 together fit in roughly the calendar Week 2 alone was budgeted for. The risk has inverted: the danger is no longer falling behind but moving so fast the documentation thins out. The decisions log (31 entries) suggests that hasn't happened, but Week 4 — real training sample, LightGBM, tuning — is where pace pressure will be highest, and it starts with a mandatory gate: re-run the importance preview at full density and verify the pair rate claims its expected rank before trusting any model built on it.

---

## 2026-06-07: The density gate failure and the §3.4 amendment (Task 4.1)

**Context:** The Week 4 density gate — re-run the importance preview at full window density; require the pair rate in the top 3 and val PR-AUC ≥ 0.80 — failed: PR-AUC 0.4784, ROC-AUC 0.9683, pair rate at #10. Per the gate's own rule, everything stopped for diagnosis. The diagnosis ended in an amendment to a locked methodological rule, so this entry carries the full evidence chain.

**Finding 1 — the Task 3.4 sparsity diagnosis was right, and is now closed.** At full density the pair rate is NULL for only 2.6% of train rows (was 84% at dev scale) and the mean pair window holds 1,113 prior clicks. The feature computes correctly and is well-estimated; its #10 rank is an empirical property of 24h-window semantics on this data — the app-level rate (#1, SHAP 0.75) dominates. Sparsity is no longer an available excuse for anything.

**Finding 2 — the real culprit: the split-boundary cold-start, measured.** Under the original §3.4 wording ("features computed for val use only val source data"), every val row's 24h window is truncated at the split boundary — and the val split is only 16 hours long, so no val row ever has a full window. Measured on v0.3.0: val hour-0 rows average 843 clicks in their ip-24h window vs ~7,000 at steady state; train's mean is 10,435 vs val's 4,794. A >2x systematic feature-distribution shift between train and evaluation, manufactured by the rule, depressing every windowed feature and the metrics computed on them. The test split (19.6h) suffers identically — on the number this project is judged by.

**Why the rule's letter exceeded its intent.** The contamination rule exists to stop information flowing backward from evaluation data into training-time features. Windows that read EARLIER rows — including train-period rows under a val row's window — move information forward in time only, exactly as production scoring would (a model deployed on day 4 has days 1-3 in its history). The per-split-source wording prevented nothing the strictly-prior frame doesn't already prevent, and cost half the window signal at every boundary.

**The options, and how the call was made.** (A) Amend the rule: windows see all strictly-prior history; splits assign rows, not histories. (B) Keep the rule and accept permanently depressed val/test numbers. (C) Keep the rule, evaluate only on rows past a 24h burn-in. I recommended A; the user initially chose C; C was then proven infeasible by direct measurement — **zero** val rows and **zero** test rows survive a 24h burn-in (both splits are shorter than the window). Under the user's standing instruction (run with the stated recommendation when a decision is needed), A was adopted. §3.4's wording now defines contamination temporally: a feature may use only data strictly prior to its row's click_time; earlier-period rows are legitimate history regardless of split; reads at-or-after the row's timestamp, and statistics fit across later splits, remain forbidden.

**What enforces the amended rule.** The same machinery as before, with one test upgraded: the materializer's split assembly is covered by a test proving (a) a val row's window includes train-period history (no cold-start), (b) only val-period rows are emitted, and (c) adding future test-period rows changes nothing — strictly-prior is intact. The label-flip and canary tests are unchanged and still pass. v0.4.0 is the first feature version built under the amended rule; v0.3.0 remains in the store as the cold-start artifact for comparison.

**Engineering notes from the same task** (Task 4.1's "bugs only appear at scale" promise, fully delivered): full ingest of 184,903,890 rows (~8 min, after making index creation optional — the ART index build OOMed the 3 GB container and the window-scan workload never uses indexes); single-query materialization OOMed from operator stacking → one query per window family; app-partitioned sliding windows (partitions of tens of millions of rows) crawled or OOMed → exact prefix-sum/ASOF and presence-segment-event rewrites, proven millisecond-equivalent by an adversarial boundary test; assembly join out-spilled the disk → pre-sampled semi-join + resumable pass files. Timings on the 8 GB host / 3.9 GB container: val materialization ~30 min, train ~28 min, full-table pass phase shared across splits.

**Confidence:** High — every step here is measured, and the amendment is enforced by tests rather than discipline. **Revisit:** the gate re-runs on v0.4.0 (same thresholds); if it still fails with full histories, the 0.80 expectation itself goes under review against what honest, strictly-prior features can achieve on this dataset.

---

## 2026-06-07: Gate re-run, expectation re-grounding, and the long-memory features (Task 4.1 closure)

**The v0.4.0 gate result:** PR-AUC 0.5048 (from 0.4784 — the boundary fix delivered +0.026), ROC-AUC 0.9706, pair rate #8 (from #10). Still far from the 0.80 bar. With sparsity closed, the cold-start removed, computation equivalence-proven, and the model ranking at ROC 0.97 with default parameters, the remaining gap is not a bug to find — it is the bar itself plus one identified feature-design limit.

**Re-grounding the expectation.** The build guide's PR-AUC expectations were written before any measurement: it predicted a SINGLE-feature baseline (B2) at 0.5-0.7 and a no-F3 logistic regression (B3) at 0.6-0.75 — while the full 20-feature LightGBM measures 0.50. Those numbers were aspiration, not evidence. The 0.80-at-default-parameters gate bar is retired, recorded here as such. The operative target remains §9's PR-AUC ≥ 0.85 on test — to be assessed honestly after the real training pipeline (Task 4.3), Optuna tuning (4.4), and the feature addition below; if the tuned, honest ceiling lands below 0.85, that becomes an evidence-backed conversation about the success criterion, not a silent miss.

**The identified limit: 24h memory in a 4-day dataset.** Every windowed feature forgets anything older than 24 hours — days 1-2 of history are invisible when scoring day 3-4 clicks. Public winning solutions on this dataset leaned on full-history aggregates. Unbounded strictly-prior windows (`RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 MILLISECOND PRECEDING`) carry zero leakage — the strictly-prior discipline is unchanged; only the cap is removed. Six long-memory features added (v0.5.0): all-time conversion rate + click count at ip, app, and (ip, app) granularity. Known-answer tests pin the distinguishing case (a row whose 24h window dropped an old click but whose all-time window keeps it); the canary holds at all-time granularity; the materializer computes them via the existing pass machinery (RANGE windows for small ip/pair partitions, prefix-sum + ASOF upper-lookup for giant app partitions).

**Why 24h windows stay.** Recency and lifetime behavior are different signals — an IP that was clean for two days and went bursty an hour ago looks different in the 24h and all-time views, and the model should see both. The ablation (Task 4.6) will say which family earns its place.

**Confidence:** High on the diagnosis chain and the leakage-safety of the addition. Medium on how much lift long memory buys — that's the next measurement. **Revisit:** §9's 0.85 after Task 4.4.

---

## 2026-06-09: Baselines B1/B2/B3, and the harness memory fix that unblocked them (Task 4.2)

**The numbers (v0.5.0 val, 3.73M rows, 0.219% positive):**

| model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| B1 random | 0.0022 | 0.499 | 0.333 |
| B3 logreg F1+F2 | 0.0265 | 0.865 | 0.136 |
| B2 pair conversion rate (1 feature) | 0.1097 | 0.784 | 0.0021 |
| default LightGBM, all features (cited) | 0.5095 | 0.970 | — |

**What the baselines establish.** (1) B1 lands exactly on the base rate (0.0022) — the floor is calibrated, the harness is sane. (2) The default GBM (0.51) is 5x the best single-feature baseline and 20x the linear one — the model genuinely extracts interaction signal, so the 0.51-vs-0.85 gap is a tuning problem (Task 4.4), not a feature or leakage bug. This is the answer the density gate was really asking for. (3) **B3 confirms F3 is necessary**: logreg on F1+F2 only scores 0.026 PR-AUC; adding F3 (in the full model) reaches 0.51. Velocity and per-click features alone do not carry the signal — the conversion-rate aggregates do.

**A metric lesson worth keeping.** The build guide expected B2 ~0.5-0.7 and B3 ~0.6-0.75. Those are ROC-flavored intuitions — and indeed B3's ROC-AUC is 0.865 while its PR-AUC is 0.026. On a 0.2%-positive problem the two metrics tell opposite stories about the same model, and PR-AUC is the honest one (§3.2). A reviewer shown only ROC-AUC would call B3 a strong model; PR-AUC reveals its precision is near-useless. This is the cleanest in-project demonstration of why PR-AUC is the headline.

**Small-denominator note on B2.** B2 uses `f3_ip_app_conversion_rate_24hr` directly as the score, NULL→train base rate. Its modest 0.11 (vs the guide's optimism) is partly the 24h window's thinness; the all-time pair rate would likely score higher, but B2 is defined as "the pair conversion rate" and the 24h variant is the canonical one. Not tuned further — it's a baseline.

**The harness memory fix (Task 1.8 deferred debt, paid here because it blocked 4.2).** Every baseline OOMed the 3.8 GB container — not in the model, in `evaluate()`. Root cause, found by RSS instrumentation: `_build_pr_curve` materializes one pydantic `PRPoint` per distinct score (~n_samples), so a single evaluate() on 3.73M rows peaked at ~2.7 GB building millions of curve objects. The Task 1.8 entry predicted exactly this ("7 full-array scans... real wins at 20M rows... flagged for Task 4.4"). Fix: cap the STORED PR-curve at 2000 evenly-spaced points (anchors kept). The `pr_auc` scalar is computed by `average_precision_score` over the full data and is untouched — a test asserts the stored curve is bounded AND `pr_auc` still matches the exact sklearn value. Plot-only change, zero metric impact, and it makes the harness affordable for the Task 4.4 Optuna sweep (100 trials x evaluate on multi-M val) that would otherwise have hit the same wall 100 times.

**A scale-engineering aside (run_baselines.py).** At 11M train rows in a 3.8 GB container the baseline runner loads only needed columns as float32, frees inputs before each evaluate(), caps DuckDB's own memory so it doesn't starve sklearn, and trains B3 on a seeded 20% sample (linear coefficients are stable at ~2.2M rows; the full-matrix float64 upcast OOMs). The default-LightGBM context row is cited from the density gate rather than recomputed, to avoid holding a full-train GBM fit alongside the baselines. None of this touches the baseline definitions — only how they fit in memory.

**Confidence:** High. The baselines are honest, the GBM clearly beats them, and the harness fix is metric-neutral by construction. **Revisit:** B3 on full train (not 20%) and B2 with the all-time rate, if a reviewer wants the strongest baselines — neither changes the conclusion that F3 is necessary and the GBM dominates.

---

## 2026-06-09: LightGBM training pipeline and hyperparameter rationale (Task 4.3)

**Class weights, not resampling (build-guide stop-and-think, CLAUDE.md §3.3).** `class_weight="balanced"` scales each class's loss by inverse frequency — mathematically a weighted log-loss, the principled way to tell the model the minority class matters. The rejected alternatives, concretely: SMOTE interpolates synthetic points between real ones, which is meaningless for our categorical-heavy space (what is the midpoint of app 12 and app 47?) and, applied before the split, leaks val/test neighbours into train; random undersampling throws away 99.8% of the negatives and the information in them. Balanced weights keep every row and every bit of signal.

**Why each hyperparameter (the build-guide defaults, defended).**
- **num_leaves = 63.** LightGBM grows leaf-wise, so leaf count — not depth — is the real capacity knob. 63 = 2^6−1, the leaves of a balanced depth-6 tree. On a 0.227%-positive target, more leaves would let a single tree carve out micro-regions around the handful of positives in a bin and memorize them; 63 is enough to express feature interactions without that. Paired with `min_child_samples` it's the main overfitting guard.
- **learning_rate = 0.05.** Small steps mean no single tree dominates, and the ensemble needs many rounds to converge — which is exactly what makes early stopping informative (it can watch val PR-AUC plateau over dozens of rounds). A high rate (0.3) would converge in ~20 rounds and overshoot. 0.05 + early stopping is the standard "let the data pick the round count" setup.
- **min_child_samples = 100** (LightGBM `min_data_in_leaf`). No leaf may form on fewer than 100 clicks. At our base rate a leaf of 100 clicks holds ~0.2 expected positives, so a leaf cannot be built to isolate one or two rare positives — the single most effective overfitting control for extreme imbalance.
- **subsample = 0.8 + subsample_freq = 1** (`bagging_fraction`). Each tree sees a fresh random 80% of rows. `subsample_freq=1` is non-optional and easy to miss: without it LightGBM ignores `subsample` entirely. Bagging decorrelates trees and regularizes.
- **colsample_bytree = 0.8** (`feature_fraction`). Each tree considers a random 80% of features per split, so no single dominant feature (the app conversion rate) is in every tree — the others get a chance to contribute, which both regularizes and surfaces secondary signal.
- **n_estimators = 2000 as a ceiling, not a target.** Early stopping (50-round patience on val `average_precision`) picks the real count; 2000 just has to be larger than that.

**Early stopping on `average_precision`, never `auc`.** The build guide's named failure mode is optimizing/reporting ROC-AUC on an imbalanced target. LightGBM 4.6 exposes `average_precision` (= PR-AUC) as a built-in eval metric, so the round count is chosen by the same metric that is the project headline (§3.2). A test asserts the model's val score dict contains `average_precision` and not `auc`.

**Reproducibility (§3.8), and what it actually requires.** `deterministic=True` alone is insufficient — it must be paired with `force_row_wise=True` (the histogram-construction order is otherwise thread-nondeterministic) and a fixed `num_threads`, plus pinning all four seeds LightGBM exposes (`random_state`, `bagging_seed`, `feature_fraction_seed`, `data_random_seed`). I verified empirically that this reproduces byte-identical predictions at `num_threads=2` *with bagging active* before committing — multi-thread determinism isn't free, and the build guide's "set the seed" is necessary but not sufficient. Tests pin it at both the `fit_lightgbm` (tiny arrays) and `train_model` (saved-artifact) levels.

**Artifact shape.** `model.txt` (LightGBM's native text Booster format — portable, diffable, version-control-legible, unlike a pickle) plus `metadata.json` (a pydantic `TrainingResult`: feature version, seed, row counts, the full resolved param dict, best_iteration, and val PR-AUC/ROC/Brier). Anyone can read metadata.json and know exactly what produced the model and how it scored, without loading it.

**Memory at scale, and a forced modeling decision.** The first full-scale training run OOMed: the two string interaction features (`f1_ip_app_interaction`, `f1_ip_device_interaction`) have millions of distinct values at 11M rows, and a pandas `category` over millions of levels — plus LightGBM binning them — blows the 3.8 GB container. The fix is also the principled call: EXCLUDE them. They're the same high-cardinality memorization trap the F1 decisions cite for excluding raw IP (most categories appear once → overfit), they scored ~0 SHAP importance in the density gate, and their signal is already carried by the F3 ip/app/pair conversion-rate aggregates. The model trains on the 26 numeric F1+F2+F3 features (`MODEL_FEATURES`); `_load_features` reads only those (float32) + label (int8) with a DuckDB memory cap. The interactions stay in the feature store (a hashed or target-encoded version could revisit them) but are not model inputs — which is exactly what the density gate ran, so its ROC 0.97 / PR-AUC 0.51 stand as the untuned reference.

**Confidence:** High on the methodology (class weights, the metric, reproducibility — all enforced by tests). The hyperparameters are sensible defaults, deliberately NOT yet tuned — Task 4.4's Optuna search over PR-AUC is what earns the final values. **Revisit:** every numeric hyperparameter above is a tuning target in 4.4; this entry documents the starting point and the reasoning, not the final model.

---

## 2026-06-10: Hyperparameter tuning — a negative result, kept (Task 4.4)

**Outcome:** Optuna found subsample-optimal params that, refit on full data, scored val PR-AUC **0.4761 — worse than the untuned default's 0.5618**. The default `lgbm-v0.1.0` is retained as the Week-4 model. This entry documents why tuning failed to beat the default, because the failure is more instructive than a marginal win would have been.

**What ran.** TPE search, 50 trials, objective = val PR-AUC, study persisted to `artifacts/tuning/study.db`. Per the Day-1 cloud-pivot threshold (full-data train measured at ~12 min/trial > the 6-min rule), and the user's choice, tuning ran on a 0.15-train / 0.25-val subsample (~2 min/trial); the final model was refit on full v0.5.0 with the best params. Best subsample trial: PR-AUC 0.5769 with `num_leaves=229, learning_rate=0.012, min_child_samples=139, colsample=0.72, subsample=0.94, reg_alpha=2.3, reg_lambda=6.5`.

**Why it didn't transfer (the actual lesson).** Hyperparameters are data-size-dependent, and two of them bit hard. `num_leaves=229` is high capacity — on a 1.7M-row subsample it has room, but on the full 11M-row train it overfits the training set quickly, so val PR-AUC plateaus early and early stopping (50-round patience) terminates at 310 rounds. Combined with the search-chosen `learning_rate=0.012` (a quarter of the default's 0.05), 310 rounds is roughly 10x less effective training than the default's 746 rounds at 0.05 — the full-data model is badly undertrained. The subsample's optimum (slow, high-capacity, lots of effective rounds within the smaller set) is the full set's pathology. The 0.5769 subsample number and the 0.5618 default number were never comparable: one is subsample-val, the other full-val.

**Why I kept the default rather than chase a tuned win.** Three reasons. (1) Honesty: the tuned model is measurably worse on the metric that matters (full val PR-AUC); shipping it as "the tuned model" would be a lie the test set would later expose. (2) The default is not arbitrary — every one of its hyperparameters has a documented, data-size-appropriate rationale (Task 4.3 entry), and it scores 0.5618. (3) The alternative remedies all cost hours of compute the project's scope doesn't warrant for a likely-marginal gain: a full-data wide search is the ~10h cloud-pivot scenario; a constrained full-data re-tune is 3-4h. Given a well-reasoned default that already beats the subsample-tuned model, the highest-value move is to keep it and spend the time on calibration, ablation, and the honest final evaluation.

**What this would be in production.** Tuning would run on full data (or a much larger sample) from the start, on infrastructure built for it, seeded with the current default and searching a narrow neighborhood — not a wide search on a small subsample. The subsample shortcut is a portfolio-scope compromise, and its failure here is the evidence for why the shortcut is a shortcut. Recorded plainly rather than hidden.

**Build-guide AC note.** Task 4.4's AC expects "a tuned model that outperforms the default." It does not, and I am not manufacturing one. The deliverables that DO hold — study saved, best params documented, default-vs-tuned comparison in `reports/tuning.md` — are complete; the "outperforms" clause is answered with an evidence-backed negative result and the reasoning above. A reviewer who reads this will see judgment and honesty, which is the project's actual success criterion (§9: "every line something I can defend").

**Confidence:** High — the diagnosis is mechanistic (high num_leaves + low lr + early stopping → undertrained on full data) and the numbers are measured on the same full-val set. **Revisit:** if Week 5+ time allows, a constrained full-data re-tune seeded at the default; not expected to move the needle enough to change the Week-4 model.

---

## 2026-06-10: Isotonic calibration (Task 4.5)

**Result:** isotonic calibration of lgbm-v0.1.0 cut val Brier from **0.01614 to 0.00130** (~12x), measured out-of-sample. The calibrator is saved as `calibrator.json` beside the model and will be applied to test at Task 4.7.

**Isotonic, not Platt (reconfirmed from the PRD).** Platt fits a one-parameter sigmoid from scores to probabilities — a strong assumption that the miscalibration has sigmoid shape. Isotonic fits any monotonic non-decreasing map, so it corrects whatever shape the data actually shows. Platt's stronger prior is an advantage only when calibration data is scarce (the prior regularizes); with ~3.7M val rows, isotonic has more than enough data to estimate a flexible map without overfitting, and it wins. This is the §3.5 locked choice; re-derived here against the actual val size.

**The methodological care that matters: out-of-sample assessment.** Fitting isotonic on val and then scoring Brier on that same val is in-sample — the calibrator has already seen those labels, so the "improvement" is vacuous and would always look good. I kept two things separate: (1) the PRODUCTION calibrator is fit on ALL val (maximum data for the best map) and saved for use on test; (2) the REPORTED pre/post Brier comes from a val-internal 50/50 holdout — fit on half, score the other half — which honestly estimates the improvement test will see. The 0.01614→0.00130 numbers are from the holdout, not the in-sample fit.

**What calibration does NOT do, stated honestly.** PR-AUC shifted 0.5638→0.5516 on the holdout (~2% relative). Isotonic is monotone non-decreasing, so it preserves ranking — except it introduces ties (flat segments collapse many distinct raw scores to a single probability), and tied scores carry slightly less ranking resolution, which average-precision reflects. So calibration is not free on the ranking metric; it trades a sliver of PR-AUC for a 12x Brier improvement. That trade is clearly right: Week 5's cost-based thresholding operates on probabilities that must mean what they say, and the calibrated ranking is essentially intact. I report the small PR-AUC cost rather than the convenient "ranking is unchanged" claim (which is true only up to ties).

**Portable artifact.** The calibrator is stored as JSON interpolation knots (`x_thresholds`, `y_thresholds`), applied via clipped `np.interp` — identical to sklearn's `IsotonicRegression.predict` (a test asserts this to 1e-9) but with no pickle, same rationale as saving the model as `model.txt`. The audit log's `raw_score` vs `calibrated_score` fields (designed in Task 1.9) now have real, distinct values to carry.

**Confidence:** High — the improvement is out-of-sample and the artifact is verified against sklearn. **Revisit:** the test-set calibration at Task 4.7 is the number that actually counts; the val holdout predicts it but the one-shot test application confirms it.

---

## 2026-06-10: Ablation study — marginal vs standalone importance (Task 4.6)

**Result (val PR-AUC, each family removed, retrained on full v0.5.0):** full 0.5618; without F1 **0.4805 (−0.081)**; without F3 0.5444 (−0.017); without F2 0.5569 (−0.005). This inverts the build-guide expectation (F3 predicted to hurt most), which makes it the interesting result.

**F1 (raw per-click identity) is the irreplaceable family.** F1 is app / channel / os / hour-of-day / day-of-week — and removing its six columns costs more PR-AUC than removing F3's fourteen. The reason: nothing else encodes *which* app or channel a click is on, and fraud propensity is strongly identity-conditional (certain apps/channels/OSes are inherently dirty). The SHAP preview foreshadowed this (f1_os_id, f1_channel_id, f1_app_id were top features); the ablation confirms F1 carries signal no aggregate reconstructs.

**F3's apparent dominance is mostly redundancy — the core methodological point.** F3's headline features (app/pair conversion rates) had the highest standalone SHAP, yet F3's MARGINAL contribution is only −0.017 because its signal is *substitutable*: with F1 present, the model re-learns an app's fraud rate from `f1_app_id` + F2 behavior when the explicit conversion rate is removed. Ablation measures marginal value given everything else; SHAP measures attribution within one model. They diverge precisely when features are correlated, and here they diverge hard. The lesson I'd give in an interview: "the most important-looking feature by SHAP was the most redundant by ablation — because importance-within-a-model and marginal-contribution-across-models are different questions, and conversion-rate aggregates are correlated with the identity features they summarize."

**Why this isn't a bug (checked).** The deltas are self-consistent: 6+6+14 = 26 features across the three families; F3 is the largest family yet smallest marginal loss, F1 the smallest family yet largest loss — exactly the redundancy signature. ROC-AUC stays 0.965–0.970 throughout (ranking is robust to any one family); only high-precision PR-AUC moves, and most for F1. Same fit, same hyperparameters, same val — the only variable is the feature set.

**What I'd do with this beyond the report.** It argues against dropping F3 (it's cheap insurance and would likely matter more at full density / against adversaries the 10% sample under-represents, and it's the interpretable feature for the audit log) but it does say the project's signal floor is mostly identity + a little behavior. F2's −0.005 is the weakest justification of its keep; the Week 2 decision to keep both 1h and 24h windows should be revisited if model size ever matters.

**Confidence:** High — measured, self-consistent, mechanistically explained. **Revisit:** re-run the ablation at full density (not 10% sample) before any decision to actually drop a family; marginal contributions can shift with data scale, as the density gate already showed for the pair rate.

---

## 2026-06-10: Week 4 retrospective (Task 4.8)

**What got built.** The whole modeling spine: full 184.9M-row ingest, the §3.4 amendment and v0.5.0 feature set (full-history + all-time aggregates), three baselines, a reproducible LightGBM pipeline, an Optuna study, isotonic calibration, a feature-family ablation, and the one-shot test evaluation. ~30 commits, every step tested and documented.

**Did the test result match expectations?** Two answers, and the honest version needs both.

*Generalization: yes, beautifully.* Val PR-AUC 0.5618 → test 0.5594, a 0.0024 gap. The calibrator transferred (test Brier 0.0015 ≈ val 0.0013). Nothing overfit; the model does on unseen data what it did on val. That's the result I most wanted to see, and it's the one that says the methodology (time-based splits, strictly-prior features, calibrate-on-val) held.

*Absolute level vs the §9 goal: no, and I'm not going to pretend otherwise.* §9 sets PR-AUC ≥ 0.85 on test. I got 0.5594. That is a large gap from the stated target, and the build-guide "what if the test number disappoints you" stop-and-think is the one I'm following: record it, don't tune to it.

**Why I think the 0.85 target was metric-confused — stated as a hypothesis, not an excuse.** The Kaggle TalkingData competition was scored on ROC-AUC, and winning solutions sat at 0.96–0.98 ROC-AUC. The build guide's "0.86–0.98" reference and §9's "0.85" almost certainly inherited that ROC-AUC scale. My ROC-AUC is 0.9718 — squarely in the winning range. PR-AUC on a 0.25%-positive problem is a fundamentally smaller-magnitude metric; 0.85 PR-AUC would mean near-perfect precision across nearly all recall on extreme imbalance, which essentially no honest model achieves on this data. So on the metric the competition actually used, this model is competitive; on PR-AUC (the harder metric this project deliberately chose as primary), 0.56 is a strong honest number, not a failure. But I'm flagging this as a target-definition question to resolve deliberately (see §9 decision), not silently declaring victory.

**Could PR-AUC go higher honestly?** Some, probably not to 0.85. Levers: full-data training instead of the 10% sample (the ablation showed marginal contributions can shift with scale); the descoped F4 graph features; richer time-delta features. But the highest published PR-AUC-style numbers on this set lean on next-click / future-window features that are leaky under §3.4. I'd rather ship 0.56 honest than 0.85 with a leak I'd have to explain away in an interview.

**What the ablation taught me (the most interview-valuable finding).** F1 raw identity (app/channel/os) is the irreplaceable family (−0.081 to remove); F3 conversion-rate aggregates are mostly redundant (−0.017 despite highest SHAP) because their signal is re-learnable from identity. Marginal ≠ standalone importance. I didn't expect that and it's the thing I'd lead with.

**What I'd change.** The 3.8 GB container was the binding constraint for the entire week — every full-scale step hit it and forced a fix. Bumping Docker to 6 GB on day one of Week 4 would have saved hours. The subsample-tuning shortcut (chosen to dodge the 12-min/trial cloud-pivot threshold) didn't transfer and cost a ~3h sweep for a negative result — a clearer-eyed call would have been to tune on full data overnight from the start, or accept the well-reasoned default without tuning at all.

**Headline number for the README:** ROC-AUC 0.972 and PR-AUC 0.56 on held-out test, calibrated (Brier 0.0015), with the imbalance context stated plainly — not a single number stripped of its metric.

**On track?** Methodologically, yes — Week 4's deliverables are all honest and defensible, which §9 says is the real bar. The open item is the §9 target itself, surfaced for decision rather than quietly reinterpreted.

---

## 2026-06-10: Correcting the §9 / G1 success target — 0.85 was a metric-scale error

**Decision:** revise the success criterion from "PR-AUC ≥ 0.85 on test" to a dual-metric statement — ROC-AUC ≥ 0.95 (achieved 0.972) plus PR-AUC reported honestly with imbalance context (achieved 0.559, ≫ baselines) — calibrated and cost-tuned. Applied to PRD G1, PRD SM1 (split into SM1a/SM1b), and CLAUDE.md §9. The user made this call; I drafted the revision.

**The error, in writing.** PRD G1 originally justified the 0.85 target with "published Kaggle solutions cluster between 0.86 and 0.98 PR-AUC." That parenthetical is factually wrong: the Kaggle TalkingData competition was scored on **ROC-AUC**, and those 0.86–0.98 figures are ROC-AUC. The 0.85 target was set, pre-implementation, by reading the leaderboard's ROC-AUC numbers as PR-AUC. On a 0.25%-positive problem the two metrics live on different scales — 0.85 PR-AUC would require near-perfect precision across almost all recall, which essentially no honest model achieves here.

**Why this is a correction, not a goalpost-move to dodge a miss.** Three checks kept me honest about the distinction. (1) The model is measured strong on the metric the target *meant*: test ROC-AUC 0.972 sits inside the published 0.86–0.98 range. (2) The other system metrics that were correctly scaled are met: SM2 (recall ≥0.70 at 1% FPR) — the model hits recall ~0.78 at under 0.6% FPR; SM3 (Brier ≤0.10) — 0.0015. Only SM1, the mis-scaled one, "failed." (3) The PR-AUC result is genuinely strong in context: 0.559 is ~220× the 0.0025 base rate and far above the 0.11 single-feature and 0.026 linear baselines (Task 4.2). A model that beats every baseline, ranks at 0.97 ROC, calibrates to Brier 0.0015, and generalizes val→test within 0.0024 is not a model that missed — it's a model whose paper target was written in the wrong units.

**What I did NOT do.** I did not re-run the test eval, tune toward 0.85, or quietly delete the old target. The amendments are dated and state the original wording and why it was wrong — a real PRD shows its corrections, and "I caught my own pre-implementation metric error and corrected it with evidence" is a stronger interview story than a target silently met. The honest test number (0.559) stands recorded.

**Confidence:** High — the metric confusion is documented in the original PRD text itself, and ROC-AUC 0.972 confirms the model meets the target's intent. **Revisit:** if Week 5+ feature work (full-data training, F4) materially lifts PR-AUC, note it; not expected to reach 0.85 honestly, and not required to — the cost-based triage (Week 5) runs on the calibrated model as-is.

---

## 2026-06-10: Triage cost model (Task 5.1)

**Context:** Week 5 selects the operating thresholds by minimizing dollar cost, not by maximizing a metric (§3.6). Task 5.1 defines that cost function in `src/sentry/triage/cost.py`.

**The decision direction — the label inversion, materialized in triage.** The model predicts `p = P(is_attributed=1) = P(legitimate)`. Fraud is non-attribution, so the triage acts on **fraud score = 1 − p** (`fraud_probability()`), and blocks HIGH fraud scores (low p). This is the data-dictionary gotcha #1 made operational, and it's the single most dangerous place to be wrong in Week 5 — a sign flip would optimize thresholds to block legitimate traffic. I localized the inversion in one named helper, documented it in the module header and here, and the first cost test asserts the direction explicitly (a blocked legit click is the false positive, an allowed fraud click is the false negative).

**Cost accounting — the three-tier model.** Decision on fraud score `f`: `f ≥ T_block` → AUTO_BLOCK, `T_review ≤ f < T_block` → HUMAN_REVIEW, `f < T_review` → ALLOW. Costs: a correctly-blocked fraud and a correctly-allowed legit click are free; a review costs `c_review` and is assumed to resolve correctly (so reviewed cases incur only the review cost, no error cost); the two errors are blocking a legit click (`c_fp`) and allowing a fraud click (`c_fn`). This is the build-guide model; the "review resolves correctly" assumption is the simplification that makes review a pure cost rather than a second imperfect classifier.

**c_fp = c_fn = $0.30, c_review = $0.50 — illustrative, and the framing matters.** These are documented as illustrative (the methodology is the deliverable, not the numbers — Task 5.4 runs sensitivity on them). Two stop-and-think calls recorded: (1) c_fp and c_fn are set equal as a deliberate simplification — in reality blocking a legit click (lost CPC + advertiser annoyance + regulatory risk) and allowing a fraud click (reputational cost with the advertiser) are not symmetric, but modeling that needs revenue data the project doesn't have; the sensitivity analysis will show how the optimum moves if they diverge. (2) c_review uses loaded reviewer time (~90s at ~$20/hr ≈ $0.50), which is the opportunity cost of that reviewer-minute, the right quantity for a capacity-constrained queue.

**One operationalization caveat, recorded for tradeoffs.md.** `is_attributed=0` is "fraud" only as a proxy — it actually means "didn't install," which includes genuine non-converters. The cost model treats label=0 as fraud because is_attributed is the only ground truth available; in production the fraud label would come from chargeback/investigation signals, not attribution. This is a portfolio-scope simplification that the cost numbers inherit.

**Confidence:** High on the mechanics and the direction (tested). The numbers are explicitly illustrative. **Revisit:** Task 5.4 sensitivity sweep on the cost ratios; the c_fp≠c_fn asymmetry if revenue data ever exists.

---

## 2026-06-10: Threshold sweep, capacity constraint, and the review-never-pays finding (Tasks 5.2/5.3)

**Selected operating point (on val, never test — §3.6):** T_block = T_review = 0.500 in fraud-score space, i.e. a clean two-way policy (block fraud_score ≥ 0.5, allow below) with **no human-review tier**. Total val cost $1,854 vs $1,116,537 for allow-everything — a **99.8% cost reduction**. Thresholds chosen on val because threshold selection is a model decision; optimizing on test would leak test into the deployed system.

**The headline finding: at the illustrative costs, human review is never cost-optimal.** The sweep drove the review band to zero width, and the reason is structural, not numerical noise: c_review ($0.50) exceeds BOTH c_fp ($0.30) and c_fn ($0.30). Reviewing an ambiguous click costs more than the worst error auto-deciding it could make, so it is always cheaper to let the model auto-decide and occasionally eat a $0.30 mistake than to pay $0.50 for a human to look. Human-in-the-loop is economically rational only when review is cheaper than the expected error it prevents — i.e. c_review < (probability the auto-decision is wrong in the ambiguous band) × error cost. With these build-guide numbers that condition never holds. This is the sharpest result of Week 5 and the one I'd lead with: the review tier isn't a free "safety" option; it has to earn its cost, and at face-value numbers it doesn't.

**Binding constraint (Task 5.3): cost, with capacity slack.** Because the unconstrained optimum reviews 0% of clicks, it trivially satisfies the SM6 reviewer cap (review ≤ 0.5% of clicks) — capacity is not binding here. I implemented the capacity constraint anyway (`best_within_capacity`) because it's the realistic mechanism and it WILL bind once the costs make review attractive (the sensitivity analysis, Task 5.4, deliberately lowers c_review to surface that regime). Documented honestly: at these costs the system is cost-bound, not capacity-bound, and the capacity machinery is exercised but slack.

**Why I kept the review tier in the design despite it being empty here.** The three-tier triage (block/review/allow) is a §G3 deliverable and the audit/router work (Week 6) targets it; the cost analysis showing review is currently uneconomic is a finding ABOUT the tier, not a reason to delete it. Whether to staff review at all becomes a live question the moment real (asymmetric, non-illustrative) costs replace these — which is exactly what tradeoffs.md (5.5) will discuss.

**Engineering note.** The sweep uses `searchsorted` over per-label sorted scores so the full 51×51 grid costs O(n log n + grid²) instead of re-scanning 3.7M val rows per cell; a test asserts it equals looping `expected_cost` cell-by-cell. The cost surface plot (`cost_surface.png`) and the comparison table (`reports/threshold_sweep.md`) are the Task 5.2 artifacts.

**Confidence:** High — the review-never-pays result is a direct consequence of c_review > max(c_fp, c_fn), verified in the numbers. **Revisit:** Task 5.4 varies the costs ±50% and into the regime where review pays; the c_fp≠c_fn asymmetry (deferred from 5.1) would also change the optimal block threshold away from the symmetric 0.5.

---

## 2026-06-10: Cost-assumption sensitivity (Task 5.4)

**Result (optimal block threshold as each cost varies ±50% on val):**

| varied | −50% | baseline | +50% |
|---|---|---|---|
| c_fp | T_block 0.30 | 0.50 | 0.58 |
| c_fn | T_block 0.68 | 0.50 | 0.38 |
| c_review | 0.50 (no change) | 0.50 | 0.50 (no change) |

Plus a c_review=$0.10 scenario where the review tier finally opens (T_block 0.68 / T_review 0.30, review load 0.13%).

**What the system is sensitive to: the FP/FN balance.** The block threshold swings from 0.30 to 0.68 across the cost variations, and the direction is exactly right — cheaper false positives (c_fp −50%) means blocking a legit click hurts less, so the optimum blocks more aggressively (threshold down to 0.30); costlier false negatives (c_fn +50%) means allowing fraud hurts more, so again block more (down to 0.38). The recommendation genuinely depends on the FP/FN ratio, which is the assumption a real deployment must pin down with revenue data (and the c_fp=c_fn simplification from Task 5.1 is exactly what this exposes).

**What it is NOT sensitive to: review cost, across the whole ±50% band.** Varying c_review from $0.25 to $0.75 changes nothing — the review tier stays empty because review is more expensive than the $0.30 error it prevents throughout that range. This confirms the Task 5.2 finding quantitatively: review-cost uncertainty is irrelevant to the recommendation *until* review drops below the error cost. The $0.10 scenario crosses that line — the review tier opens, routes 0.13% of clicks, and only then would the reviewer-capacity constraint (Task 5.3) begin to bind. So the honest summary for a deployment: don't agonize over the reviewer-cost estimate; do nail down the FP/FN balance and, separately, decide whether review can be made cheap enough to be worth staffing at all.

**Confidence:** High — the threshold movements are monotonic and directionally correct, and the review-tier crossover is exactly where the cost algebra predicts (c_review < error cost). **Revisit:** the asymmetric-cost case (c_fp ≠ c_fn) in tradeoffs.md, which the c_fp and c_fn rows above already preview.

---

## 2026-06-13: Week 5 retrospective (Task 5.6)

**What got built.** The full cost-to-decision layer: a three-tier cost model, a threshold sweep with a reviewer-capacity constraint, a ±50% sensitivity analysis, and `tradeoffs.md`. Roughly six commits, every piece tested, and — unusually for a week — a single finding that reframed how I think about the whole triage tier.

**What surprised me about the cost analysis.** That the human-review tier is uneconomic at face-value costs, and that this is structural rather than a numbers fluke. I had assumed, going in, that review was the obvious "safety valve" for ambiguous cases and that the analysis would be about *how much* to review. Instead the analysis said: don't, because a $0.50 review costs more than the $0.30 worst-case error it prevents, so auto-deciding and eating the occasional mistake is always cheaper. Once I saw it, it was obvious — review only pays when review is cheaper than the expected error — but I didn't see it coming, and it's now the thing I'd lead with in an interview. It also retroactively justified building the capacity constraint that turned out to be slack: the constraint is the mechanism that catches the regime (cheap review) where the tier actually opens.

**Did the optimum match my gut?** Partly. The block threshold landing near the middle of fraud-score space, and moving the way it did under cost changes (cheaper false positives → block more), matched intuition — that's just the cost algebra behaving. What didn't match my gut was the review tier collapsing to zero width; I expected a non-trivial review band and got none. And the 99.8% cost reduction versus doing nothing looked too good until I remembered it mostly reflects the model being a strong ranker (ROC 0.97) on a problem where the base rate makes "do nothing" enormously expensive — the model isn't performing a miracle, the no-detection baseline is just very bad.

**What I'd change.** I leaned hard on illustrative costs because I had to, and the sensitivity analysis is what keeps that honest — but if this were real, the FP/FN asymmetry is the one input worth real effort to estimate, because it's the only assumption the recommendation is actually sensitive to. I'd also flag that `tradeoffs.md` is the artifact I most need to re-read in my own voice before showing it to anyone; it's the highest-signal document in the project and the one where polish matters most.

**On track?** Yes. Week 5 was lighter than Week 4 (no full-data retraining, mostly analysis on cached predictions) and the methodology is clean. The cost-based-thresholding story is the part of this project I'd most want to be asked about, which is exactly what the week was for.

---

## 2026-06-13: Triage router (Task 6.1)

`route_case(score, thresholds, qa_sample_rate, rng) -> Action` implements PRD §8.1's four-tier logic. Three small calls: (1) `score` is the FRAUD score (high → block), reusing the cost layer's direction so the whole triage stack reasons in one score space — no second inversion. (2) QA sampling draws from a passed-in numpy Generator rather than global `random()`, so batch routing is reproducible (§3.8) and tests are deterministic; rates of 0 and 1 short-circuit without consuming the draw. (3) `TriageThresholds(block, review)` is its own small frozen type, not a reuse of the sweep's `ThresholdChoice`, so the router doesn't depend on the tuning machinery — the pipeline converts between them. Thresholds are block-inclusive and review-inclusive (a score exactly at a cutoff takes the more severe action), matching the cost model's boundary convention. 9 tests cover all four actions, both boundaries, the 0/1 rate edges, the ~rate sampling fraction, and reproducibility.

---

## 2026-06-13: Prediction + SHAP explanations (Task 6.3)

`models/predict.py` adds a `ModelBundle` (booster + feature order + optional calibrator), `predict_calibrated`, and `top_contributions` (top-5 features per case by |SHAP|). Three calls worth recording:

**SHAP units are log-odds, and the audit log says so.** For a binary LightGBM Booster, `TreeExplainer.shap_values` returns contributions in margin (log-odds) space — verified empirically: per-row `sum(shap) + base = raw_score` to 1e-3 (the raw Booster returns a single (n, n_features) array, not the list the sklearn wrapper gives). A reviewer reads a positive SHAP as "pushed the log-odds of legitimacy up," not "added X to the probability." The Task 6.4 HTML report and the audit entries label the units so no one misreads a log-odds contribution as a probability delta.

**Top-5, computed per batch.** Five features per case (the §3.9 / audit-schema convention): enough to explain a decision, few enough to keep entries manageable. SHAP is computed once over the whole batch, never per-case — per-call TreeExplainer construction is the documented slow path the build guide warns against.

**Calibrated probability via the saved isotonic map.** `predict_calibrated` composes `calibrator.predict(booster.predict(...))`, so the audit log's `raw_score` (booster probability) and `calibrated_score` (post-isotonic) are both real and distinct — the two fields the Task 1.9 schema reserved finally carry their intended values end to end.

**Confidence:** High — SHAP shape/units verified against the margin reconstruction; 7 tests cover ranking, the label-driver sanity check (the synthetic driver feature tops the ranking), calibrated-predict parity, reproducibility, and load round-trip. **Revisit:** if Task 6.5 finds SHAP on the full val batch too slow/memory-heavy in the container, compute it on the routed (blocked + reviewed + QA) subset only, which is all the audit log needs.
