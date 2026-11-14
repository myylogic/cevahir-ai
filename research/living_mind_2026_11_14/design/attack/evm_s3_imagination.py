#!/usr/bin/env python3
"""evm_s3_imagination.py -- LEAD_DESIGN §6 (imagining and measuring reality) worked by hand and in simulation.

Design text under attack:
    §6  "the mind runs §2 on its own predictions: the top prediction is fed back as an event with provenance
         'imagined' on an internal channel"            (and §2.5: LEARN (always on))
    §6  "R = e(i -> v) computed from the evidence that does NOT lie on the chain that produced it"
    §6  "It is written with provenance 'imagined'/'inferred' and weight iota*R."
    §1  "inferred/imagined conclusions later confirmed or refuted change the trust in the mind's own construction"
    §5  "Relief = -dD is felt",  D(u) = sum use(t) (1 - c_t)

Parts
  A  LEARN is always on, so every imagined event is evidence for the tie that predicted it
  B  the iota*R deposit: repeated idle cycles on the same conclusion; the sign of the deposit
  C  calibration of R for NOVEL constructions over a life (the T8 claim), design vs corrected
"""
import math
import numpy as np
from evm_common import read


def root(p):
    return 1.0 if p >= 1 else (-p + math.sqrt(p * p + 4 * p * (1 - p))) / (2 * (1 - p))


# ------------------------------------------------------------------------------------------- part A
def life(r, steps=300, p=0.6, seed=0, feedback="top", trust="sensed_only", corrected=False, tn0=0.0):
    """u is followed by v1 with probability p, else by v2 (sensed).  After every real step, r idle steps."""
    rng = np.random.default_rng(seed)
    ws_p = np.zeros(2); ws_n = np.zeros(2)          # sensed evidence of u->v1, u->v2
    wi_p = np.zeros(2); wi_n = np.zeros(2)          # imagined evidence
    tp = 0.0; tn = tn0                              # trust tie of 'imagined', sensed referee (tn0: sceptical birth prior)
    tip = tin = 0.0                                 # trust evidence whose own provenance is 'imagined'
    last_top = None

    def tau():
        if trust == "sensed_only":
            return (tp + 0.5) / (tp + tn + 1.0)
        t = 0.5                                     # C3 literal: trust evidence of provenance 'imagined' is itself
        for _ in range(60):                         # weighted by tau_imagined -> solve the fixed point
            t = (tp + t * tip + 0.5) / (tp + tn + t * (tip + tin) + 1.0)
        return t

    def expct():
        t = tau()
        Wp = ws_p + t * wi_p; Wn = ws_n + t * wi_n
        return (Wp + 0.5) / (Wp + Wn + 1.0)

    for s in range(steps):
        y = 0 if rng.random() < p else 1
        e = expct()
        if last_top is not None and not corrected:
            if y == last_top:
                tp += 1.0
            else:
                tn += tau()                         # C3 on the trust tie: predicted 'holds', did not occur
        ws_p[y] += 1.0
        ws_n[1 - y] += 1.0 if corrected else e[1 - y]
        for _ in range(r):
            e = expct()
            if feedback == "top":
                top = int(np.argmax(e))
            else:                                   # variant: sample the imagined event in proportion to e
                top = 0 if rng.random() < e[0] / (e[0] + e[1]) else 1
            last_top = top
            if not corrected:                       # design: LEARN is always on
                wi_p[top] += 1.0
                wi_n[1 - top] += e[1 - top]
                if trust == "c3_literal":           # the imagined event confirms the imagined prediction
                    tip += 1.0
    t = tau()
    Wp = ws_p + t * wi_p; Wn = ws_n + t * wi_n
    W = Wp + Wn
    return Wp / W, W / (W + 1.0), (Wp + 0.5) / (W + 1.0), t, ws_p + ws_n


def part_a():
    p = 0.6
    print("== A. u is followed by v1 with p = 0.6 and by v2 with 0.4 (sensed, 300 real moments).  After each real")
    print("      moment the mind idles r steps: top prediction fed back as an 'imagined' event, LEARN always on. ==")
    print(f"   without imagination the design's own rule gives f* = {root(0.6):.3f} / {root(0.4):.3f} (its known bias); truth 0.6 / 0.4")
    print("   40 lives per row.  'wrong' = share of lives that end expecting the 40% successor MORE than the 60% one.")
    print("   r    wrong   | lives that locked on v1: f(v1) f(v2) | lives that locked on v2: f(v1) f(v2) | tau_imagined | W read / W lived")
    for r in (0, 1, 3, 10):
        res = [life(r, seed=s) for s in range(40)]
        f = np.array([x[0] for x in res]); e = np.array([x[2] for x in res])
        t = np.mean([x[3] for x in res])
        wrong = e[:, 1] > e[:, 0]
        ratio = np.mean([(x[1][0] / (1 - x[1][0])) / x[4][0] for x in res])
        a = f[~wrong].mean(0) if (~wrong).any() else np.array([float("nan")] * 2)
        b = f[wrong].mean(0) if wrong.any() else np.array([float("nan")] * 2)
        print(f"  {r:3d}   {wrong.mean():5.1%}   |            {a[0]:5.3f} {a[1]:5.3f}            |            {b[0]:5.3f} {b[1]:5.3f}"
              f"            |    {t:5.3f}     |     {ratio:5.2f}")
    print("   -> with idle steps the mind does not converge on the world: it LOCKS onto whichever successor was ahead")
    print("      early and drives it to f ~ 0.95, the other to ~ 0.2.  In about four lives out of ten that is the WRONG")
    print("      one (the 40% successor believed at 0.92, the 60% one at 0.25), after 300 lived moments, for good.")
    print("      tau_imagined settles at 0.5-0.7 (the hit rate of the top prediction): imagined evidence is taken at more")
    print("      than half the weight of sensed evidence, and the W the mind reads is several times the W it lived.")

    r300 = [life(10, steps=300, seed=s)[2] for s in range(40)]
    r3000 = [life(10, steps=3000, seed=s)[2] for s in range(40)]
    same = sum(((a[1] > a[0]) == (b[1] > b[0])) for a, b in zip(r300, r3000))
    wrong3000 = np.mean([b[1] > b[0] for b in r3000])
    print(f"   persistence: with 3000 lived moments instead of 300 (r = 10), {same} of 40 lives keep the direction they")
    print(f"   locked at; wrong share {wrong3000:.1%}.")

    print("\n   variant: a sceptical birth prior on the trust tie (9 units of negative evidence, tau_imagined = 0.05 at birth), r = 10:")
    res = [life(10, seed=s, tn0=9.0) for s in range(40)]
    f = np.array([x[0] for x in res]); e = np.array([x[2] for x in res])
    wrong = e[:, 1] > e[:, 0]
    ratio = np.mean([(x[1][0] / (1 - x[1][0])) / x[4][0] for x in res])
    print(f"   wrong {wrong.mean():.1%}; lives locked on v1: f = {f[~wrong].mean(0)[0]:.3f} / {f[~wrong].mean(0)[1]:.3f};"
          f" tau_imagined ends at {np.mean([x[3] for x in res]):.3f}; W read / W lived = {ratio:.2f}")
    print("   (the prior delays the loop, it does not remove it: trust is learned from hits, and the top prediction hits)")

    print("\n   variant: the imagined event is SAMPLED in proportion to e instead of taken as the top (r = 10):")
    res = [life(10, seed=s, feedback="sample") for s in range(40)]
    fs = np.array([x[0][0] for x in res])
    ratio = np.mean([(x[1][0] / (1 - x[1][0])) / x[4][0] for x in res])
    print(f"   f(u->v1) over 40 lives: mean {fs.mean():.3f}, sd {fs.std():.3f}, min {fs.min():.3f}, max {fs.max():.3f};"
          f"  W read / W lived = {ratio:.2f}")
    print("   (no lock-in, but the conviction read is still a multiple of what was lived: a Polya urn adds weight, not information)")

    print("\n   variant: C3 read literally for the trust tie (an imagined event CONFIRMS the imagined prediction):")
    for r in (1, 10):
        res = [life(r, seed=s, trust="c3_literal") for s in range(20)]
        print(f"   r = {r:2d}: tau_imagined = {np.mean([x[3] for x in res]):.3f}, f(u->v1) = {np.mean([x[0][0] for x in res]):.3f},"
              f" f(u->v2) = {np.mean([x[0][1] for x in res]):.3f}")
    print("   closed form of that fixed point after n self-confirmations: tau = [(n-1) + sqrt((n-1)^2 + 2n)] / (2n):",
          ", ".join(f"n={n}: {((n-1)+math.sqrt((n-1)**2+2*n))/(2*n):.3f}" for n in (1, 3, 10, 100)))

    print("\n   corrected (imagined events write no evidence; raw counts):")
    res = [life(10, seed=s, corrected=True) for s in range(20)]
    f = np.mean([x[0] for x in res], 0)
    print(f"   r = 10: f(u->v1) = {f[0]:.3f}, f(u->v2) = {f[1]:.3f}, lived W = read W = {np.mean([x[4][0] for x in res]):.1f}")


# ------------------------------------------------------------------------------------------- part B
def part_b():
    print("\n== B. the deposit iota*R.  Conclusion i -> v reached through i -ABSTRACT-> g -EXPECT-> v; i was NEVER observed.")
    print("      'Evidence not on the chain' = the tie i -> v itself.  First cycle: none, so R = 1/2 whatever g says. ==")
    print("   The deposit is written on i -> v, which is not on the chain, so the next cycle reads it back.")
    print("   tau_im  iota |  cycles:   1       5       20      100     (c and e of a conclusion that was never observed)")
    for tau, iota in ((0.5, 1.0), (0.5, 0.3), (0.9, 1.0)):
        D = 0.0
        row = []
        for n in range(1, 101):
            R = (tau * D + 0.5) / (tau * D + 1.0)
            D += iota * R
            if n in (1, 5, 20, 100):
                f, c, e = read(tau * D, 0.0)
                row.append(f"c={c:.2f} e={e:.2f}")
        print(f"    {tau:3.1f}    {iota:3.1f} |          " + "   ".join(row))
    print("   -> conviction 0.9+ about something never met, from idle cycles alone.  Each cycle lowers the drive")
    print("      D = sum use*(1-c) of §5, so each cycle is FELT as relief: the loop is rewarded by the design's own signal.")

    print("\n   sign of the deposit.  i was SEEN twice not to be v (W- = 2 sensed).  R = e = 0.167 < 1/2, and the design")
    print("   still writes iota*R on the POSITIVE side.  tau_im = 0.5, iota = 1:")
    D = 0.0
    crossed = None
    for n in range(1, 2001):
        R = (0.5 * D + 0.5) / (0.5 * D + 2.0 + 1.0)
        D += 1.0 * R
        f, c, e = read(0.5 * D, 2.0)
        if n in (1, 5, 10, 15, 20, 30):
            print(f"     cycle {n:3d}: R = {R:.3f}, W+ = {0.5*D:.2f}, e(i->v) = {e:.3f}")
        if crossed is None and e > 0.5:
            crossed = n
    print(f"   -> after {crossed} idle cycles e(i -> v) > 1/2: the mind expects what it has twice seen to be false.")


# ------------------------------------------------------------------------------------------- part C
def one_life_c(seed, T=1500, qs=(0.95, 0.75, 0.55)):
    s_true = np.array([q * q + (1 - q) ** 2 for q in qs])
    rng = np.random.default_rng(seed)
    dom = rng.integers(0, 3, T)
    hit = rng.random(T) < s_true[dom]
    # ---- design, literal: R = e from evidence not on the chain = 1/2; self-trust per domain learned from outcomes
    tp = np.zeros(3); tn = np.zeros(3)
    R_design = np.full(T, 0.5); e_after = np.zeros(T); R_reh = np.zeros(T)
    for t in range(T):
        d = dom[t]
        tau = (tp[d] + 0.5) / (tp[d] + tn[d] + 1.0)
        e_after[t] = (tau * 1.0 * 0.5 + 0.5) / (tau * 1.0 * 0.5 + 1.0)       # e of the conclusion after the deposit iota*R
        D = 0.0
        for n in range(5):                                                   # rehearsed 5 times before it is resolved
            R = (tau * D + 0.5) / (tau * D + 1.0); D += R
        R_reh[t] = R
        if hit[t]:
            tp[d] += 1.0
        else:
            tn[d] += tau
    tau_design = (tp + 0.5) / (tp + tn + 1.0)
    # ---- corrected: R = expectation of the rule built from the observed member, with the learned horizon of the domain
    a = np.ones(3); dd = np.ones(3)                                          # agreement tie per domain, prior (1, 1)
    R_corr = np.zeros(T)
    for t in range(T):
        d = dom[t]
        A = a[d] / (a[d] + dd[d])
        rho = max(1e-9, 2 * A - 1)
        k_d = (1 - rho) / rho                                                # horizon of rules in this domain
        R_corr[t] = (1.0 + k_d * 0.5) / (1.0 + k_d)                          # S = 1 (one member, one vote), f = 1
        if hit[t]:
            a[d] += 1.0
        else:
            dd[d] += 1.0
    y = hit.astype(float)
    return dict(dom=dom, y=y, s=s_true[dom], R_design=R_design, R_reh=R_reh, e_after=e_after, R_corr=R_corr,
                tau=tau_design)


def part_c():
    print("\n== C. calibration of R for NOVEL constructions over a life (T8).  Three domains; in each, a new category is met,")
    print("      ONE member is observed (closed look), and the mind constructs the same value for a sibling it never saw.")
    print("      The sibling really has that value with probability s = q^2 + (1-q)^2:  q = 0.95 / 0.75 / 0.55. ==")
    lives = [one_life_c(seed) for seed in range(20)]
    T = len(lives[0]["y"])
    first, last = slice(0, 100), slice(T - 500, T)

    def brier(key, sl):
        return float(np.mean([((L[key][sl] - L["y"][sl]) ** 2).mean() for L in lives]))

    print("   Brier score of R against what later turned out, mean of 20 lives of 1500 constructions")
    print("   (lower is better; R = 1/2 always gives 0.250)")
    print("                                              first 100    last 500")
    print(f"   design  R (leave-path-out)                   {brier('R_design', first):.3f}       {brier('R_design', last):.3f}     (R = 1/2 for every novel construction)")
    print(f"   design  R after 5 rehearsals                 {brier('R_reh', first):.3f}       {brier('R_reh', last):.3f}")
    print(f"   design  e of the conclusion after deposit    {brier('e_after', first):.3f}       {brier('e_after', last):.3f}")
    print(f"   corrected R (learned horizon per domain)     {brier('R_corr', first):.3f}       {brier('R_corr', last):.3f}")
    print(f"   best possible (R = s of the domain)          {brier('s', first):.3f}       {brier('s', last):.3f}")
    print("   per domain, last 500 constructions:   true s | design R | design R(5 rehearsals) | design tau_self | corrected R")
    qs = (0.95, 0.75, 0.55)
    for d in range(3):
        def m(key):
            return float(np.mean([L[key][last][L["dom"][last] == d].mean() for L in lives]))
        s_d = qs[d] ** 2 + (1 - qs[d]) ** 2
        tau_d = float(np.mean([L["tau"][d] for L in lives]))
        print(f"     q = {qs[d]:.2f}                             {s_d:.3f}    {m('R_design'):.3f}        {m('R_reh'):.3f}"
              f"               {tau_d:.3f}          {m('R_corr'):.3f}")
    print("   -> the design's R cannot improve: for a construction that is new, 'the evidence not on the chain' is empty.")
    print("      Its self-trust does separate the domains (biased upward by the e-weighted rule), but §6 uses it only as")
    print("      the weight of a deposit.  The corrected R is the same quantity put where it belongs.")


if __name__ == "__main__":
    part_a(); part_b(); part_c()
