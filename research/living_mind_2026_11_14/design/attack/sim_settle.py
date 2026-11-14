#!/usr/bin/env python3
"""sim_settle.py -- counterexamples for the settling recurrence of LEAD_DESIGN.md §2.2

    a <- kWTA_F( decay*a + in + sum_u a_u (e_{u->v} - 1/2)*2 )          T <= 6 iterations

Weights are not invented: every weight below is what the design's own learning rule (§2.5) converges to for the
stated co-occurrence statistics (see sim_rule_bias.py):  f solves (1-p) f^2 + p f - p = 0,  weight = 2f - 1 (c -> 1).

(1) SPIRAL: an input x that is a rare special case of a frequent unit y  (P(y|x) = 1, P(x|y) = q small).
(2) LATCH : two units that reliably come together (P = 1 both ways) keep each other at 1 with no input at all.
(3) Linear stability region of the two-unit loop as a function of decay and q.
(4) LATCH in the full engine (zsim.Engine, design settings, interoception off): how long does a familiar pair stay
    in the focus after its last real occurrence, and how many ties does it found meanwhile?
"""
import math, json, sys
import numpy as np


def f_design(p):
    if p >= 1:
        return 1.0
    return (-p + math.sqrt(p * p + 4 * p * (1 - p))) / (2 * (1 - p))


def w_design(p):
    return 2 * f_design(p) - 1


def iterate(M, inp, a0, decay, T, F=7, clip=True):
    a = a0.copy(); hist = [a.copy()]
    for _ in range(T):
        raw = decay * a + inp + M.T @ a          # M[u, v] = weight of tie u -> v
        pos = np.where(raw > 0, raw, 0.0)
        if (pos > 0).sum() > F:
            thr = np.sort(pos)[-F]; pos = np.where(pos >= thr, pos, 0.0)
        a = np.minimum(pos, 1.0) if clip else pos
        hist.append(a.copy())
    return np.array(hist)


def main():
    out = {}
    print("== (1) spiral: x = input (rare special case), y = frequent unit;  x->y: P=1,  y->x: P = q ==")
    for q, decay in [(0.05, 0.5), (0.01, 0.5), (0.001, 0.5), (0.01, 0.2), (0.01, 0.8)]:
        wxy = w_design(1.0); wyx = w_design(q)
        M = np.array([[0.0, wxy], [wyx, 0.0]])
        h = iterate(M, np.array([1.0, 0.0]), np.zeros(2), decay, 12)
        lam = np.linalg.eigvals(decay * np.eye(2) + M.T)
        foc = ["".join(n for n, v in zip("xy", row) if v > 0) or "-" for row in h[1:]]
        print(f" q={q:<6} decay={decay}  w(y->x)={wyx:+.2f}  |eig|={abs(lam[0]):.2f}  "
              f"a_x: {' '.join('%.2f' % v for v in h[1:, 0])}")
        print(f"                          focus per iteration: {' '.join(foc)}   (T=5: {foc[4]}, T=6: {foc[5]})")
        out[f"spiral_q{q}_d{decay}"] = dict(w_yx=wyx, eig=float(abs(lam[0])), a_x=h[1:, 0].tolist(), focus=foc)

    print("\n== (1b) the same with TWO frequent associates y, z of x (e.g. a category word and a context word) ==")
    for q in [0.05, 0.01]:
        wyx = w_design(q)
        M = np.zeros((3, 3)); M[0, 1] = M[0, 2] = 1.0; M[1, 0] = M[2, 0] = wyx
        h = iterate(M, np.array([1.0, 0, 0]), np.zeros(3), 0.5, 12)
        foc = ["".join(n for n, v in zip("xyz", row) if v > 0) or "-" for row in h[1:]]
        print(f" q={q:<5} a_x: {' '.join('%.2f' % v for v in h[1:, 0])}   focus: {' '.join(foc)}")
        out[f"spiral2_q{q}"] = dict(a_x=h[1:, 0].tolist(), focus=foc)

    print("\n== (2) latch: u, v always together (weights -> +1 both ways); present ONCE (it. 1 only), then no input ==")
    for w in [0.4, 0.5, 0.6, 0.9, 1.0]:
        M = np.array([[0.0, w], [w, 0.0]])
        a = np.zeros(2); tr = []
        for it in range(40):
            inp = np.array([1.0, 1.0]) if it == 0 else np.zeros(2)
            a = iterate(M, inp, a, 0.5, 1)[-1]; tr.append(a[0])
        print(f" mutual weight {w:.1f}: a_u after 1,2,5,10,20,40 iterations with no input: "
              + " ".join("%.3f" % tr[i] for i in (1, 2, 5, 10, 20, 39))
              + ("   <- never decays" if tr[39] > 0.5 else ""))
        out[f"latch_w{w}"] = [float(tr[i]) for i in (1, 2, 5, 10, 20, 39)]
    print(" condition (linear part): decay + sqrt(w_uv*w_vu) >= 1  <=>  with decay 0.5: w >= 0.5  <=>  f >= 0.75")

    print("\n== (3) two-unit loop x<->y, stability of the linear part: |eig| = sqrt(decay^2 + |w_xy*w_yx|) for opposite signs ==")
    print("   rows: decay; columns: q = P(x|y); entry = |eig| (unstable spiral when > 1)")
    qs = [0.2, 0.1, 0.05, 0.02, 0.01, 0.001]
    print("   decay   " + "  ".join("q=%-6g" % q for q in qs))
    tab = {}
    for d in [0.2, 0.5, 0.7, 0.9]:
        row = [math.sqrt(d * d + abs(w_design(1.0) * w_design(q))) for q in qs]
        tab[d] = row
        print("   %.1f     " % d + "  ".join("%-8.2f" % r for r in row))
    out["stability"] = dict(qs=qs, table=tab)

    print("\n== (4) perseveration in the engine (interoception OFF so that the effect is not due to it) ==")
    print("   world: 3 random symbols out of 2000 per moment (no structure) + a pair (P,Q) met `reps` times, then never again")
    import zsim
    variants = [("design (ties for every focus pair)", dict()),
                ("design + activation floor 0.05", dict(floor=0.05)),
                ("fix: ties only between evented units", dict(create="events")),
                ("fix: events-only ties + base-rate reference", dict(create="events", ref="base", neg="plain"))]
    for label, kw in variants:
        for reps in [10, 100]:
            eng = zsim.Engine(zsim.Cfg(intero="none", recruit="none", **kw), ncap=20000)
            rng = np.random.default_rng(0)
            def bg():
                return tuple("B%d" % i for i in rng.choice(2000, 3, replace=False))
            for i in range(200):
                eng.moment(bg())
            for i in range(reps):               # the pair P,Q always together, interleaved with background moments
                eng.moment(("P", "Q"))
                eng.moment(bg())
            p, q = eng.sym["P"], eng.sym["Q"]
            ties_before = eng.out[0][p].n + (eng.out[1][p].n if eng.out[1][p] else 0)
            n_after = 1000; strong = 0; last = 0; anyin = 0
            for i in range(n_after):            # the pair never occurs again
                ids, vals = eng.moment(bg())
                il = ids.tolist()
                if p in il or q in il:
                    anyin += 1
                    av = max([float(vals[il.index(x)]) for x in (p, q) if x in il])
                    if av > 0.5:
                        strong += 1; last = i + 1
            ties_after = eng.out[0][p].n + (eng.out[1][p].n if eng.out[1][p] else 0)
            indeg = sum(1 for u in range(eng.n) if eng.out[0][u] is not None and p in eng.out[0][u].idx)
            print(f" {label:<46} pair met {reps:>3}x: in the focus in {anyin/n_after:5.1%} of the next {n_after} moments "
                  f"(a>0.5 in {strong/n_after:5.1%}, last at +{last}); out-ties of P {ties_before} -> {ties_after}; "
                  f"units with a tie INTO P: {indeg}")
            out[f"persev|{label}|{reps}"] = dict(any=anyin / n_after, strong=strong / n_after, last=last,
                                                 ties_before=ties_before, ties_after=ties_after, indeg=indeg)
    json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
