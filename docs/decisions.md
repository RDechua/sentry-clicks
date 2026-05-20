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
