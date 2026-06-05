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
