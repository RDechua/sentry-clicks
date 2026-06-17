# Adversary Model — Sentry

*Author's note: this document is me thinking like the attacker on purpose. Not
to break anything — as a discipline, because detection only improves when you
understand what you're up against and where your own system is blind. Where
Sentry can't catch a tier, I say so and name the feature that's missing.*

---

## 1. The actor model: who commits click fraud, and why

Click fraud is an economic activity, so the actors follow the money. Mobile
advertisers pay per click (or per install); an attacker who can manufacture
clicks captures that spend without delivering a real user. The actors split
into a few recognizable motivations:

- **Publishers / sub-publishers inflating their own inventory.** They get paid
  for clicks on ads they host, so they manufacture clicks to inflate payouts.
  This is the largest share by volume because the incentive is direct and
  continuous.
- **Affiliate fraudsters gaming install bounties.** Attribution pays whoever
  "delivered" the install, so they fabricate clicks that race to claim credit
  for organic installs (click-flooding / click-injection — mostly out of scope
  here, since it's an attribution-window attack, but the actor is the same).
- **Competitors depleting a rival's budget.** They burn a target's daily ad
  budget with junk clicks so the rival's real ads stop serving. Low volume,
  high intent, hard to separate from organic.
- **Botnet operators monetizing compromised devices.** They rent out
  infrastructure (residential IPs, real handsets) to any of the above.

What they share is the goal from §2 of the policy: *a click event without
genuine interest*. What separates them is how much they're willing to spend to
make each fake click look real — and that spend is exactly what sorts them into
the three tiers below.

## 2. The three tiers

I model adversaries at three sophistication tiers (consistent with PRD §4). The
tier is defined by capability and economics, not by who the actor is — the same
botnet operator can run a Tier 1 or a Tier 3 campaign depending on what they're
willing to pay.

### Tier 1 — Naïve

**Capabilities.** Unmodified emulators or simple scripts. Clicks come from a
handful of IPs in tight time windows. No effort to vary device/OS fingerprints
or to pace clicks like a human.

**Motivation.** Maximum clicks per dollar. The naive operator hasn't been
caught yet (or doesn't care) and optimizes purely for volume.

**Economics.** Almost free. A free Android emulator and a script generate
thousands of clicks an hour from one machine on one IP. The cost per fake click
is effectively zero — which is also why it's trivially detectable: zero spend
on realism means zero camouflage.

**Detection counters in Sentry.** This is what the **F2 velocity features**
are built for. `f2_clicks_per_ip_last_1hr` and `f2_clicks_per_ip_last_24hr`
spike; `f2_inter_click_time_seconds` collapses toward zero; `f2_burst_score`
and `f2_ip_click_std_inter_arrival` capture the machine-gun rhythm. A naive
campaign lights up every velocity feature at once, and the model has no trouble
with it. The EDA backs this: multi-click IPs convert at 0.07–0.12% versus 0.86%
for single-click IPs — high velocity *is* the fraud signal in this data.

**Evasion I can't detect.** Essentially none at this tier — if they stay naive,
they're caught. The only "evasion" is to stop being naive, which moves them to
Tier 2.

### Tier 2 — Adapted

**Capabilities.** Rotating residential proxies (so IPs look like real homes),
randomized device/OS/user-agent fingerprints, and clicks paced to mimic human
inter-click intervals. Operates across many apparent devices.

**Motivation.** Sustainable fraud. The Tier 2 operator has been burned by
velocity rules and now invests in looking human to keep the revenue flowing.

**Economics.** Now there's real cost. Residential proxy traffic is sold by the
gigabyte or by the IP-hour; pacing clicks like humans caps throughput, so each
fake click costs more and earns less per unit time. This is the bulk of fraud
volume (PRD estimates ~60%) precisely because it's the equilibrium between
"cheap enough to be profitable" and "realistic enough to survive velocity
rules."

**Detection counters in Sentry.** Velocity goes quiet, so the **F3 aggregate
and conversion-rate features** carry the load — and they work because they key
on *outcome*, not *rhythm*. `f3_ip_conversion_rate_24hr`,
`f3_ip_conversion_rate_alltime`, `f3_app_conversion_rate_24hr`, and the pair-
level `f3_ip_app_conversion_rate_24hr` all encode the same fact: faked traffic
doesn't install. A Tier 2 operator can pace and rotate all they like, but if
their clicks never convert, the conversion-rate features sink the score. The
EDA's 99.7%-never-convert finding for `(ip, app)` pairs with 5+ clicks is
exactly the Tier 2 signature. The distinct-entity counts
(`f3_ip_distinct_apps_24hr`, `f3_ip_distinct_devices_24hr`) also catch the
"one proxy IP, many apparent devices" pattern that fingerprint randomization
produces.

**Evasion I can't detect.** The conversion-rate counter has a hole: if the
operator *buys real installs* to raise their conversion rate (install fraud
laundering the click fraud), the outcome features are fooled. Sentry has no
view into install authenticity — it trusts `is_attributed`. That's the gray
area G3 in the policy, and it's a real blind spot at this tier.

### Tier 3 — Sophisticated

**Capabilities.** Real (physical or cloud-emulated) devices with stolen or
rented user profiles. Each device produces a *low* volume of fraud blended into
genuine use. Across the fleet the volume is large, but no single entity stands
out.

**Motivation.** High-value, low-detection fraud — targeting expensive
inventory or specific campaigns where the payoff justifies the operating cost.

**Economics.** The most expensive to run and the most damaging per case. Real
devices and real user profiles cost real money, and blending fraud into
legitimate use throttles volume hard. The PRD puts this at ~10% of volume but
disproportionate damage. The defining economic feature: the attacker has
deliberately driven each *individual entity* below every per-entity threshold.

**Detection counters in Sentry.** This is where I'm honest about a gap. Per-
entity features are *structurally blind* to Tier 3, because the fraud signal
isn't in any single IP, device, or app's history — it's in the **correlation
across entities** (many low-volume entities that share an app-set or appear in
coordinated bursts). Catching that needs graph / cluster features: bipartite
ip↔device / ip↔channel graphs, connected-component sizes, shared-neighbor
similarity. In Sentry, the **F4 graph family was descoped** to cheap degree-
style counts only (`f3_ip_distinct_*`, `f3_app_distinct_ips_24hr`). Those catch
*fan-out* (one entity touching many others) but not *coordination* (many
entities quietly touching the same targets). So Sentry detects the easy edge of
Tier 3 and misses the core of it. I'd rather state that than imply coverage I
don't have.

**Evasion I can't detect.** A patient, well-funded Tier 3 operator who keeps
every entity individually plausible and never reuses entities enough to form a
visible cluster. Competitor sabotage (policy G9) is the cleanest example: paced,
varied, individually organic-looking clicks designed only to burn budget. With
no cross-entity feature and no campaign-level spend monitoring, Sentry's score
for each such click stays low.

## 3. The arms race

Detection and evasion are a moving equilibrium, and the feature families map
onto its stages:

1. I ship velocity features (F2). Tier 1 dies; the survivors pace and rotate,
   becoming Tier 2.
2. I ship conversion-rate aggregates (F3). Tier 2's outcome signal betrays
   them; the survivors either buy installs to launder (a new attack) or
   distribute below thresholds, becoming Tier 3.
3. Tier 3 needs graph features I haven't built. Until I do, the equilibrium
   sits with the sophisticated adversary.

Two consequences I take seriously:

- **Static models decay.** As adversaries adapt, the feature distribution
  shifts and yesterday's boundary leaks. Retraining cadence is part of the
  system spec, not an afterthought — and the QA sample exists partly to keep an
  unbiased read on the false-negative rate as the mix moves.
- **Published methodology is adversary intelligence.** Anything I publish about
  *how* detection works, a sophisticated adversary reads. I've documented the
  methodology (feature families, the cost model, the tiers) without publishing
  specific decision boundaries or the trained thresholds — those are the part
  that should stay private in a real deployment. The line I've drawn: explain
  the *why* and the *shape*, not the exact cutoffs.

## 4. If I were the adversary

The most effective attack on Sentry *specifically* isn't more volume — it's
defeating the outcome features, because those are the load-bearing counter.
Two moves:

- **Buy a baseline of real installs per IP/app pair** so the conversion-rate
  features read "normal," then run fraudulent clicks under that cover. This
  directly attacks `f3_*_conversion_rate_*`, my strongest family, and Sentry
  has no install-authenticity signal to catch it.
- **Stay under every per-entity threshold and never form a cluster** — the
  pure Tier 3 play, which my missing graph features can't see.

Naming these is the point of the exercise: my best features are outcome-based,
so the highest-value attack is to forge the outcome, and my biggest structural
gap is cross-entity coordination. That tells me exactly where the next
engineering investment should go (real install-fraud signal + graph features),
which is more useful than a vague "make it more robust."

## 5. What I can't model

The honest boundary of this system:

- **No real fraud label.** `is_attributed` is a proxy (policy §2). I'm
  detecting non-conversion, not fraud, and the two diverge for every
  uninterested-but-real user.
- **No stable identity.** IP is not a user key (data dictionary): NAT collapses
  many users into one IP, carrier rotation spreads one user across many. Every
  per-IP feature inherits that noise, and I have no device-stable ID to fix it.
- **No cross-entity / graph view at the core.** The Tier 3 coordination signal
  is mostly invisible (F4 descope).
- **No cross-network signal.** Fraud that spreads thin across many ad networks
  is only catchable with shared industry signal, which a single-stream project
  can't have.
- **The feedback loop.** Once Sentry blocks a click, that click never produces
  an attribution label, so the next model trains on data the current model
  censored. The QA sample is the only unbiased window into the allowed
  population, and it's a partial fix, not a complete one (PRD §10.3, open
  question).

A senior reviewer should read this section as the system's threat-coverage map:
Tiers 1 and 2 are well covered by F2 and F3; Tier 3 and outcome-laundering are
known, named gaps with a clear path to close them. That's the honest state, and
documenting it is itself the discipline the exercise is meant to build.
