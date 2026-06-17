# Click Fraud Policy — Sentry

*Author's note: this is the policy I actually implemented, not a generic
template. Where the system's behavior is shaped by a limitation of the data or
the cost model, I say so rather than papering over it. The gray-areas section
is the heart of the document — it's where the real judgment lives.*

---

## 1. Scope

This policy governs how Sentry decides whether a single mobile ad click is
fraudulent and what enforcement action follows. It covers the click-fraud
forms that the TalkingData click stream can plausibly expose: automated
clicking, paid click farms, compromised-device clicks, and competitor budget
depletion. It does not govern install fraud, attribution-window manipulation,
or post-install behavioral fraud — those need signals this dataset doesn't
carry, and I list them explicitly in §6.

The policy operates at the level of the individual click. It assigns each
click a fraud score, maps that score to a severity tier, and triggers an
action. It does not make advertiser-level or campaign-level judgments; those
would aggregate many click decisions and are out of scope.

## 2. What click fraud is (in plain language)

A click is fraudulent when it was generated with the intent to register a
click event that does not represent genuine user interest in the advertised
app. Four concrete forms:

- **Automated clicks** — bots or emulator farms generating clicks
  programmatically (P1).
- **Paid click farms** — humans paid to click ads with no intent to use the
  app (P2).
- **Compromised-device clicks** — clicks from a device whose owner did not
  consent (malware, hijacked SDKs) (P3).
- **Competitor sabotage** — clicks generated to drain a rival's ad budget (P4).

What unites them is *intent without genuine interest*. That's also what makes
fraud hard: intent isn't in the data. I only observe the click's metadata (IP,
app, device, OS, channel, timestamp) and, eventually, whether an install was
attributed to it.

**The measurement gap I have to be honest about.** Sentry's ground truth is
the `is_attributed` label — did this click lead to an app install. I treat a
non-attributed click as the fraud-suspect case and a fraud score of
`1 − P(is_attributed)`. But *most non-converting clicks are not fraud* — they
are uninterested real users who clicked and didn't install. So the label is a
proxy for fraud, not fraud itself, and it's a loose one. Every decision rule
below inherits that looseness. A production system would need a real fraud
label (confirmed bot traffic, chargeback signals, manual fraud
adjudications), not non-conversion. I designed around the proxy because it's
what the public benchmark provides, and I flag it everywhere it matters.

## 3. Severity tiers and actions

Sentry scores each click in fraud-probability space (higher = more
fraud-likely) and routes it through a three-tier policy plus a sampling tier:

| Tier | Condition | Action | What it means |
|---|---|---|---|
| **Severity 1** | score ≥ `T_block` | `AUTO_BLOCK` | High-confidence fraud. Block the click; don't charge the advertiser. |
| **Severity 2** | `T_review` ≤ score < `T_block` | `HUMAN_REVIEW` | Suspected fraud. Route to a reviewer queue with the top-5 SHAP contributors attached. |
| **Severity 3** | score < `T_review`, but drawn by QA sampling | `QA_SAMPLE` | Borderline/clean but spot-checked for drift and false-negative estimation. |
| **Clean** | score < `T_review`, not sampled | `ALLOW` | No action; the click is served and (if applicable) charged. |

The actions are implemented exactly as the four `Action` enum values
(`triage/router.py`), and every action above `ALLOW` produces a full audit
entry (§7 below, and CLAUDE.md §3.9).

## 4. Decision rules — how the system actually acts

**The thresholds are cost-based, not metric-based.** `T_block` and `T_review`
come from the cost sweep in `docs/tradeoffs.md`, which minimizes total dollar
cost over the validation set subject to a reviewer-capacity cap. They are not
tuned to maximize F1 or hit a fixed precision — real T&S systems are tuned to
cost, and so is this one.

**Two findings make the operating policy unusual, and I state them plainly:**

1. **At the illustrative costs, the cost-optimal policy has no review tier.**
   The sweep drove `T_review` up to meet `T_block` at fraud score 0.5 — a
   clean two-way split (block ≥ 0.5, allow below). The reason is structural:
   reviewing a click ($0.50) costs more than the worst error auto-deciding it
   could make ($0.30 for a false positive or false negative). Human review is
   economically rational only when it's cheaper than the expected error it
   prevents, and at face-value costs it isn't. So the *default* operating
   policy auto-decides everything. The review tier stays in the design because
   the moment real, asymmetric costs replace the illustrative ones, it can
   reopen — but I won't pretend it's active when the numbers say it shouldn't be.

2. **The enforcement demo uses quantile thresholds, not the cost-optimal
   ones.** Because the label proxy makes ~99.8% of validation traffic
   "fraud-suspect," a fixed cost-optimal cutoff blocks almost everything, and
   the review queue comes out empty. To produce a *populated* review-queue
   artifact for inspection, the `sentry enforce` demo sets thresholds at fraud-
   score quantiles (top 0.5% → block, next 2% → review). That is **artifact
   population, not policy selection** — it exists to exercise the
   routing/SHAP/audit/report machinery, and it is not a recommended operating
   point. The cost-based thresholds are the real policy.

I keep these two facts in the policy itself because hiding them would make the
review-queue screenshot look like something it isn't.

## 5. Gray areas

These are the situations where the right action genuinely isn't obvious. For
each: the scenario, why it's ambiguous, what Sentry currently does, and what
production would do differently. They're drawn from the actual structure of
this data, not invented.

**G1. A user clicks an ad and abandons the install within 30 seconds.**
Ambiguous because non-conversion is exactly what fraud looks like to my label,
but a real user changing their mind looks identical. *Current behavior:* scored
as fraud-suspect via `1 − P(is_attributed)`; the behavioral features (IP
velocity, conversion history) usually keep an isolated real user's score low.
*Production:* would use a real fraud label and session-depth signals, so a
genuine abandonment isn't counted as fraud at all.

**G2. Two devices on the same IP click the same ad within an hour.**
Ambiguous because it's either a household behind one NAT (legitimate) or one
operator running two emulators (fraud). *Current behavior:* the IP-velocity and
distinct-device features push the score up, because coordinated multi-device
activity on one IP is fraud-shaped. *Production:* would weigh the IP's
historical conversion rate more heavily before acting, since a long-lived
household IP with prior installs is different from a fresh high-velocity one.

**G3. A click matches a known click-farm pattern, but the IP's conversion rate
is normal.** Ambiguous because pattern and outcome disagree. *Current behavior:*
the F3 conversion-rate features pull the score *down* — outcome usually wins,
which is the intended design (a farm that converts normally is, for charging
purposes, behaving). *Production:* would route this to review precisely because
the disagreement is informative, and a reviewer might spot a farm that buys
installs to launder its pattern.

**G4. A high-velocity IP that is actually a corporate proxy or carrier-grade
NAT.** Ambiguous because legitimate infrastructure can produce farm-like
volume. *Current behavior:* velocity features raise the score; this IP is at
elevated false-positive risk. *Production:* would maintain an allowlist of
known infrastructure ranges and a per-IP reputation that decays, so a
well-behaved proxy isn't perpetually penalized.

**G5. One real user rotating across many IPs (mobile carrier reassignment).**
Ambiguous because per-IP history is thin for everyone, so the model can't lean
on it. *Current behavior:* with little IP history, the per-click and app-level
features dominate; the score tends toward the app's base rate. *Production:*
would need a device-stable identifier (not IP) to accumulate history — the data
dictionary already flags that IP is not a user key.

**G6. The cold-start click: an IP/app pair with no prior history in the
window.** Ambiguous because "no history" is genuinely uninformative, not
evidence either way. *Current behavior:* the historical features are `NULL`,
which LightGBM routes to a learned default direction; the click is scored on
its non-historical features alone. *Production:* same approach, but with a
warm-up reputation prior seeded from the app and channel.

**G7. An `(ip, app)` pair with 5+ clicks and zero installs.** Ambiguous in
principle but not in this data: the EDA found 99.7% of such pairs never
convert. *Current behavior:* the pair- and IP-level conversion-rate features
make this a strong fraud signal, and the score is high. *Production:* likely an
auto-block candidate, but I'd want a real fraud label to confirm the 99.7%
isn't just uninterested repeat-clickers.

**G8. A single-click IP.** Counterintuitive gray area: single-click IPs convert
at 0.86%, while multi-click IPs convert at 0.07–0.12% — frequent IPs are the
farms, and one-shot IPs are mostly legitimate. Ambiguous because the naive
intuition ("engaged users click more") is backwards here. *Current behavior:*
low velocity keeps the score low, which is correct for this data. *Production:*
fine as-is, but worth re-validating per-market, since the relationship could
invert in a different traffic mix.

**G9. Competitor sabotage that looks like organic interest (P4).** Ambiguous
because budget-depletion clicks can be crafted to mimic real users — paced,
varied IPs, plausible devices. *Current behavior:* if each click is
individually plausible and entities aren't reused, Sentry's per-entity features
are structurally blind to it; the score stays low. *Production:* needs
cross-entity correlation / graph features (descoped here, see adversary-model)
and campaign-level spend-anomaly monitoring.

**G10. A click during a legitimate traffic burst (flash sale, viral moment).**
Ambiguous because a real spike and a bot burst both raise velocity features.
*Current behavior:* velocity raises the score; legitimate bursts are at
elevated false-positive risk during the spike. *Production:* would compare the
burst's conversion rate to the app's baseline in near-real-time — a real burst
still converts, a bot burst doesn't.

**G11. A click whose calibrated fraud score lands on a flat segment of the
isotonic calibrator (mass piled at 1.0).** Ambiguous operationally: many
distinct raw scores map to the same calibrated probability, so the calibrated
score can't rank within that mass. *Current behavior:* this is why the enforce
demo routes on the raw score — the calibrated score is honest about
probability but can't separate the top mass. *Production:* would either recal
on a rolling window or use the raw margin for ranking within the block tier
while keeping the calibrated score for the charge decision.

**G12. A borderline click right at `T_block`.** Ambiguous by construction —
threshold boundaries are where the cost of a wrong call is highest. *Current
behavior:* the boundary is inclusive (score ≥ `T_block` blocks), and the QA
sample is what catches systematic errors near it. *Production:* would add a
narrow review band around the boundary specifically (a reason the review tier
earns its keep once review is cheap enough).

## 6. Out of scope

Sentry does not address: install fraud (faked installs after a real click);
attribution-window hijacking (click-injection / click-flooding that steals
attribution credit); post-install fraud (fake in-app events); account-takeover
fraud; and any advertiser- or campaign-level adjudication. It also does not act
on PII — the data is fully anonymized integer identifiers, by design.

## 7. Appeals (conceptual)

The project is not deployed, so appeals are designed, not built. The mechanism
the audit log enables: every enforcement action (block/review/QA) is logged
with the case ID, both raw and calibrated scores, the active thresholds, the
model and policy versions, and the top-5 SHAP contributors. An advertiser
contesting a block would reference the case ID; a reviewer would pull the audit
entry and see *exactly which features drove the decision and by how much* (in
log-odds). That evidence trail is the appeal's substrate — you can't fairly
adjudicate a decision you can't explain, which is why explainability is a
policy requirement, not just an ML nicety.

## 8. Versioning

This policy is versioned alongside the model and feature store. Every audit
entry stamps a `policy_version`, so a past decision can always be replayed
against the policy that was active when it was made. A policy change (new
thresholds, a new severity rule, a new gray-area ruling) increments the
version and is recorded in `docs/decisions.md` with its rationale. The
discipline mirrors the model versioning: nothing about how a decision was made
is left implicit.
