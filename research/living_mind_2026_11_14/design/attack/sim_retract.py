#!/usr/bin/env python3
"""sim_retract.py -- LEAD_DESIGN.md §10 "Float64 evidence. Deterministic. Every experience's evidence increments are
logged (moment id -> tie increments), so one experience can be retracted exactly by subtracting them."  (§9 iv:
"exact retraction of any single experience (evidence is additive)").

(a) arithmetic: is (sum of float64 increments) - (one increment) equal to the sum without it?
(b) counterfactual: is the state after retracting experience k equal to the state of a mind that never had it?
    (design rule: the negative increment is g*a_u*e, so every later increment depends on what k left behind)
(c) the same two questions for unweighted, integer-valued evidence (the fix).
"""
import json, math
import numpy as np

K = 1.0
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- (a) float64 arithmetic
print("== (a) float64: w = sum of n increments; retract increment j by subtraction; compare with the sum without j ==")
res = {}
for n in (10, 100, 1000, 10000, 100000):
    bad = 0; worst = 0.0; trials = 300
    for _ in range(trials):
        inc = rng.random(n) * rng.choice([1.0, 0.3, 0.05], n)      # g*a_u*e-like magnitudes
        j = rng.integers(n)
        full = 0.0
        for x in inc:
            full += x
        without = 0.0
        for i, x in enumerate(inc):
            if i != j:
                without += x
        r = full - inc[j]
        if r != without:
            bad += 1; worst = max(worst, abs(r - without))
    print(f"   n={n:>6}: retraction not bit-exact in {bad/trials:6.1%} of trials (largest error {worst:.1e})")
    res[n] = dict(not_exact=bad / trials, worst=worst)

# ---------------------------------------------------------------- (b) counterfactual
def life(outcomes, rule, skip=None):
    """one tie; outcomes[i] in {1: v came, 0: v predicted and did not come}; returns state and the per-experience log"""
    wp = wn = 0.0; log = []
    for i, y in enumerate(outcomes):
        if i == skip:
            log.append((0.0, 0.0)); continue
        e = (wp + K / 2) / (wp + wn + K)
        dp = 1.0 if y else 0.0
        dn = 0.0 if y else (e if rule == "design" else 1.0)
        wp += dp; wn += dn; log.append((dp, dn))
    return wp, wn, log


def expct(wp, wn):
    return (wp + K / 2) / (wp + wn + K)


print("\n== (b) retract experience k from one tie: subtract its logged increments  vs  a life without that experience ==")
print("   rule      n    k   retracted (W+, W-, e)        never-had-it (W+, W-, e)      |e difference|")
out_b = []
for rule in ("design", "unweighted"):
    for n, k in ((20, 0), (200, 0), (200, 100), (2000, 0)):
        errs = []
        for _ in range(200):
            outcomes = (rng.random(n) < 0.3).astype(int); outcomes[k] = 1
            wp, wn, log = life(outcomes, rule)
            rp, rn = wp - log[k][0], wn - log[k][1]
            cp, cn, _ = life(outcomes, rule, skip=k)
            errs.append(abs(expct(rp, rn) - expct(cp, cn)))
        print(f"   {rule:<10} {n:>4} {k:>4}   ({rp:7.2f}, {rn:7.2f}, {expct(rp, rn):.4f})   ({cp:7.2f}, {cn:7.2f}, {expct(cp, cn):.4f})   "
              f"mean {np.mean(errs):.5f}  max {np.max(errs):.5f}")
        out_b.append(dict(rule=rule, n=n, k=k, mean=float(np.mean(errs)), max=float(np.max(errs))))
print("   (with the design rule the difference is small per tie but it is not zero, and it is only the first-order effect:")
print("    the retracted experience also decided which ties and units exist, what was in every later focus, what was pruned)")

# ---------------------------------------------------------------- (c) integer evidence
print("\n== (c) evidence kept as integer counts of events (fixed point), negatives implicit (exposure - positives) ==")
bad = 0
for _ in range(300):
    n = 100000
    inc = rng.integers(0, 2 ** 20, n).astype(np.int64)        # increments in units of 2^-20
    j = rng.integers(n)
    if int(inc.sum()) - int(inc[j]) != int(np.delete(inc, j).sum()):
        bad += 1
print(f"   n=100000 integer increments: retraction not exact in {bad} of 300 trials")
json.dump(dict(a=res, b=out_b, c=bad), open(__file__.replace(".py", ".json"), "w"), indent=1)
