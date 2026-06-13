# Tradeoffs: turning a fraud score into a decision

This document covers how Sentry chooses its enforcement thresholds. The model
produces a calibrated probability per click; a deployed system has to turn that
number into an action — block it, send it to a human, or let it through. Where
those cutoffs sit is the decision this document is about, and I want to be
explicit that it is a *business* decision wearing a statistical costume, not a
statistical one.

I'm writing this as the artifact I'd most want a hiring manager to read, because
threshold selection is where a lot of fraud projects quietly reveal they never
thought past the AUC number. The analysis here is built on illustrative cost
figures — I don't have a real ad network's revenue data — so the numbers are
placeholders and the *method* is the point. I've tried to be honest about which
conclusions are robust to the made-up numbers and which depend on them.

## 1. The problem of threshold selection

The model scores each click with a calibrated probability that the click is
legitimate (`is_attributed = 1`, meaning it led to an app install). A score is
not an action. Production needs three actions — auto-block, route to human
review, allow — and that means two cutoffs on the fraud score: one above which
we block outright, and one above which we ask a human. Everything below both is
allowed.

One wrinkle that has to be stated before anything else, because getting it
backwards inverts the whole analysis: the label is *attribution*, not *fraud*.
A click with `is_attributed = 1` is a legitimate, converting click — the rare
0.2% positive class. Fraud is the absence of attribution, so the fraud score I
threshold on is `1 - p`, where `p` is the model's calibrated probability of
legitimacy. A high fraud score gets blocked; a low one gets allowed. I keep
this inversion in exactly one named function (`fraud_probability`) so there is a
single place to be right or wrong about it.

The naive instinct is to pick 0.5 and move on. But 0.5 is meaningless here: on a
0.2%-positive problem the threshold that matters lives wherever the cost of
mistakes is minimized, and there's no reason that's 0.5 in either score space.

## 2. Why metric-based thresholds are inadequate

The usual ways to choose a threshold are max-F1, or max-precision at a fixed
recall, or a fixed percentile of scores. I considered all three and rejected
them as the selection method, for one reason: they optimize an abstract quantity
that nobody is actually paying for.

Max-F1 treats precision and recall as equally important and both as unitless.
But in this system a false positive (blocking a real click) and a false negative
(letting fraud through) cost different things, and there's a third action —
human review — with its own cost that F1 doesn't even have a slot for. "Maximize
precision at 90% recall" is the same problem in different clothing: it fixes
recall by fiat, when recall should fall out of the economics, not be an input to
them.

These metrics are the right tools for *comparing models* — and PR-AUC is exactly
how I picked the model. They're the wrong tool for *picking an operating point*,
because an operating point is a claim about how much money each kind of mistake
costs, and a metric doesn't carry dollars. So the thresholds are chosen against
an explicit cost function instead.

## 3. The cost function

The cost model has three parameters, all per-event and all illustrative:

- **C_FP ≈ $0.30** — blocking a legitimate click. Roughly one mid-range
  cost-per-click of lost revenue, plus the harder-to-price annoyance of telling
  an advertiser their real traffic was rejected.
- **C_FN ≈ $0.30** — allowing a fraudulent click. The advertiser was charged for
  a click that will never convert; the direct cost is again about one CPC.
- **C_review ≈ $0.50** — one human review. Roughly 90 seconds of a reviewer's
  time at a $20/hour loaded rate. This is an opportunity cost: a minute spent on
  this case is a minute not spent on another.

The total cost of a triage policy on a labelled set is then: C_FP times the
legitimate clicks we blocked, plus C_FN times the fraudulent clicks we allowed,
plus C_review times everything we routed to a human. Correctly blocking fraud
and correctly allowing a real click cost nothing — they're the wins. Human
review is modeled as resolving correctly, so a reviewed case incurs only the
review fee and none of the error costs; that's a simplification (real reviewers
are a second imperfect classifier) but a reasonable one for setting thresholds.

Two assumptions are worth flagging as deliberately soft. First, I set C_FP equal
to C_FN. In reality they almost certainly differ — blocking a legitimate click
carries reputational and possibly regulatory weight that allowing one bad click
doesn't, and an ad network's revenue per allowed click is not its reputational
cost per missed fraud. Treating them as equal is a documented simplification
forced by not having revenue data, and the sensitivity analysis below is largely
about how much that simplification matters. Second, `is_attributed = 0` is
treated as "fraud," when it really means "didn't install," which includes
genuine users who simply didn't convert. The attribution label is the only
ground truth the dataset gives me; a production system would define fraud from
chargeback and investigation signals, not from non-conversion. The cost numbers
inherit that proxy.

## 4. The threshold sweep

With a cost function, choosing thresholds is just minimizing it. I sweep a grid
of `(T_block, T_review)` pairs — every pair where the review cutoff sits at or
below the block cutoff — and compute the total cost of each on the validation
set. The minimum-cost pair is the recommended operating point.

This runs on validation, never on test. Threshold selection is a model decision
in the same sense that hyperparameter selection is: if I tuned the thresholds
against the test set, the test number would no longer be an honest estimate of
deployed performance. The test set was touched exactly once, for the final
metrics, and the thresholds were already fixed by then.

The sweep is computed efficiently — fraud scores are sorted once per class and
each grid cell's counts come from `searchsorted`, so the whole surface is cheap
even on millions of validation rows — but the efficiency is a detail; the result
is the point. On validation, the minimum-cost policy is a clean two-way split at
fraud score 0.50: block everything above, allow everything below, send nobody to
review. It costs about $1,854 across the 3.7M-row validation set, against roughly
$1.12M for doing nothing (allowing every click and eating every fraud) — a 99.8%
reduction. That headline number mostly reflects that the model is a strong
ranker (test ROC-AUC 0.97); a good model makes the operating-point decision look
easy.

## 5. The reviewer capacity constraint

Unconstrained cost minimization is not realistic on its own, because it can
happily route an unaffordable fraction of traffic to human review. Real
trust-and-safety teams have a hard cap: a fixed number of reviewers, a fixed
number of cases per reviewer per day. So the selection is constrained — the
review cutoff has to be high enough that review volume stays under capacity
(the project's stated bound is 0.5% of clicks).

Here the constraint turns out to be slack, and for an interesting reason rather
than a boring one: the unconstrained optimum already sends *zero* clicks to
review, so any capacity cap above zero is satisfied trivially. The binding
constraint in this configuration is cost, not capacity. I implemented the
capacity machinery anyway, because it's the right mechanism and because it does
bind the moment the costs make review worth doing — which the sensitivity
analysis reaches. It would have been easy to skip a constraint that doesn't
bind; I'd rather have it there and document that it's slack than discover its
absence later.

## 6. Sensitivity analysis

Because the cost numbers are invented, the honest question is how much the
recommendation depends on them. I varied each cost by plus and minus 50% and
re-optimized.

The optimal block threshold is genuinely sensitive to the balance between C_FP
and C_FN, and it moves in the direction you'd hope. Make false positives cheaper
and the optimum blocks more aggressively (the block threshold drops to 0.30);
make false negatives costlier and it again blocks more (down to 0.38). Across the
variations the block threshold ranges from about 0.30 to 0.68. That is the
assumption a real deployment has to get right, and it's exactly the one I had to
fake.

The recommendation is, by contrast, completely insensitive to the cost of review
across the entire plus-or-minus-50% band. Moving C_review from $0.25 to $0.75
changes nothing, because review stays empty throughout — which points at the
single most interesting result in this analysis.

## 7. The thing worth saying out loud, and what changes in production

**Human review never pays at these costs, and that's not an accident of the
numbers — it's structural.** A review costs $0.50; the worst mistake an automated
decision can make on a case costs $0.30. So it is always cheaper to let the model
decide and occasionally be wrong than to pay a human to look. The review tier
only earns its place when review becomes cheaper than the expected error it
prevents; when I drop C_review to $0.10, the tier immediately opens, routes about
0.13% of clicks to humans, and the capacity constraint starts to bind. The lesson
I'd carry into any real system: a human-in-the-loop tier is not free insurance.
It has a price, and it has to beat the price of being wrong automatically, or it
shouldn't exist.

For an actual deployment, three things change. First, the FP/FN asymmetry gets
pinned down with real revenue and reputational data instead of being assumed
equal — that's the assumption the optimum is most sensitive to, so it's where the
modeling effort should go. Second, "fraud" stops being a proxy for non-attribution
and gets defined from chargebacks and investigations, which would shift the costs
and possibly the label itself. Third, the review tier's economics get re-examined
deliberately: either review is made cheap enough (better tooling, faster cases)
to be worth staffing, or the system commits to mostly automated enforcement and
sizes the review team to the genuinely ambiguous residual. The methodology in
this document — cost function, sweep, capacity, sensitivity — is what I'd run
again on real numbers; only the inputs would change.
