#!/usr/bin/env python3
"""sim_calib.py -- LEAD_DESIGN.md §1 "conviction c = W/(W+k) (how much has been lived; k = 1); expectation
e = c(f - 1/2) + 1/2 (what the mind acts on)".   Is e a usable probability for the predictions the mind issues?

Every EXPECT(lag 1) tie leaving the focus is a prediction for the next moment (§2.3).  We let the engine live
5000 moments of the toy world, then for 3000 further moments record, for every such prediction, (W+, W, base rate
of the target, did the target come?).  Scores:
   reliability: mean e versus observed frequency, per bucket of evidence W+
   skill      : log loss of the forecast e minus log loss of the forecast "base rate of the target" (negative = the
                tie makes the mind predict WORSE than if the tie did not exist)
for  (1) the design's rule and formula (prior mean 1/2, k = 1, e-weighted negatives),
     (2) the fixed counts (unweighted negatives) with prior mean = base rate and k = 1,
     (3) the same with k fitted on the first half of the recorded predictions (what a learned k would give).
"""
import json, math, sys
import numpy as np
import zsim


def collect(cfg_kw, n_live=5000, n_rec=3000):
    eng = zsim.Engine(zsim.Cfg(**cfg_kw), ncap=40000); world = zsim.World(seed=1); st = world.stream(n_live + n_rec + 10)
    for _ in range(n_live):
        eng.moment(next(st))
    rec = []
    for _ in range(n_rec):
        syms = next(st)
        nxt = set(eng.sym[x] for x in syms if x in eng.sym)
        key = tuple(eng.sym.get(x) for x in syms)
        if key in eng.tuples:
            nxt.add(eng.tuples[key])
        pf_ids, _ = eng.prev_focus
        for u in pf_ids.tolist():
            A = eng.out[1][u]
            if A is None or A.n == 0:
                continue
            n = A.n; d = A.dst[:n]; wp = A.wp[:n].copy(); W = np.array(eng._W(A, n, u, 1), dtype=float)
            b = eng.base(d); y = np.isin(d, list(nxt))
            rec.append(np.stack([wp, W, b, y.astype(float)], 1))
        eng.moment(syms)
    return np.concatenate(rec)


def ll(e, y):
    e = np.clip(e, 1e-6, 1 - 1e-6)
    return -(y * np.log(e) + (1 - y) * np.log(1 - e))


def report(name, R, efun, out):
    wp, W, b, y = R.T
    e = efun(wp, W, b)
    skill = ll(b.clip(1e-6, 1 - 1e-6), y) - ll(e, y)
    print(f"\n== {name} ==   predictions recorded: {len(y):,}   came true: {y.mean():.4f}   mean expectation issued: {e.mean():.4f}")
    print(f"   log loss per prediction: {ll(e, y).mean():.4f}   base-rate forecast: {ll(b, y).mean():.4f}   "
          f"skill (positive = tie helps): {skill.mean():+.4f}")
    print("   evidence W+      n        mean e    observed    skill")
    rows = []
    for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 5.5), (5.5, 20.5), (20.5, 1e9)):
        m = (wp >= lo) & (wp < hi)
        if m.sum() == 0:
            continue
        print(f"   [{lo:>4},{hi if hi < 1e8 else 'inf':>5})  {int(m.sum()):>8}   {e[m].mean():.4f}    {y[m].mean():.4f}    {skill[m].mean():+.4f}")
        rows.append(dict(lo=lo, n=int(m.sum()), e=float(e[m].mean()), obs=float(y[m].mean()), skill=float(skill[m].mean())))
    out[name] = dict(n=len(y), came=float(y.mean()), mean_e=float(e.mean()), logloss=float(ll(e, y).mean()),
                     base=float(ll(b, y).mean()), skill=float(skill.mean()), rows=rows)


def main():
    out = {}
    R = collect(dict(intero="none"))                                  # the design as written
    report("design: prior mean 1/2, k = 1, e-weighted negatives", R, lambda wp, W, b: (wp + 0.5) / (W + 1.0), out)
    R2 = collect(dict(intero="none", ref="base", neg="plain", clamp=True, create="events", settle="monotone", drive=False))
    report("fixed counts, prior mean = base rate, k = 1", R2, lambda wp, W, b: (wp + b) / (W + 1.0), out)
    half = len(R2) // 2; A, B = R2[:half], R2[half:]
    best = None
    for k in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128):
        wp, W, b, y = A.T
        l = ll((wp + k * b) / (W + k), y).mean()
        if best is None or l < best[1]:
            best = (k, l)
    kfit = best[0]
    report(f"fixed counts, prior mean = base rate, k fitted on the first half (k = {kfit})", B,
           lambda wp, W, b: (wp + kfit * b) / (W + kfit), out)
    report("   (same second half with k = 1, for comparison)", B, lambda wp, W, b: (wp + b) / (W + 1.0), out)
    out["k_fit"] = kfit
    json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
