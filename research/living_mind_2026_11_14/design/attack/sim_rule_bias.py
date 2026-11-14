#!/usr/bin/env python3
"""sim_rule_bias.py -- what does the LEARN rule of LEAD_DESIGN.md §2.5 converge to on ONE tie?

Design text (§1, §2.5):
    f = W+/W ; c = W/(W+k), k=1 ; e = c(f-1/2)+1/2
    v occurred:                      w+ += g*a_u
    v was predicted, did not occur:  w- += g*a_u*e(u->v)
    candidate tie created with evidence g*a_u*a_v

We feed one tie u->v a Bernoulli(p) world (v follows u with probability p), a_u = g = 1, tau = 1, and read
f ("how often it held") and e ("what the mind acts on").  Then the same with the unweighted negative update.
Also: the facet experiment (a concept with m equally likely facets) and the sign of the settling weight
(e-1/2)*2 = (W+ - W-)/(W+k) used in §2.2.
"""
import numpy as np, math, json

K = 1.0


def run(p, n, rng, neg="ew", w0=1.0):
    wp, wn = w0, 0.0
    for _ in range(n):
        W = wp + wn
        e = (wp + K / 2) / (W + K)
        if rng.random() < p:
            wp += 1.0
        else:
            wn += e if neg == "ew" else 1.0
    W = wp + wn
    return wp / W, (wp + K / 2) / (W + K), W


def fixed_point(p):
    # f = p / (p + (1-p) f)  ->  (1-p) f^2 + p f - p = 0
    if p >= 1.0:
        return 1.0
    return (-p + math.sqrt(p * p + 4 * p * (1 - p))) / (2 * (1 - p))


def main():
    rng = np.random.default_rng(0)
    out = {"single_tie": [], "facets": []}
    print("== one tie, Bernoulli(p) outcomes, design rule (negative increment weighted by e) ==")
    print(" p      f_theory  f_sim(n=1e4)  e_sim   weight=(e-.5)*2   W after 1e4 exposures | plain rule: f, e")
    for p in [0.001, 0.01, 0.05, 0.1, 0.2, 0.25, 1 / 3, 0.5, 0.7, 0.9, 0.99]:
        fs, es, Ws = zip(*[run(p, 10000, rng) for _ in range(20)])
        fp, ep, _ = zip(*[run(p, 10000, rng, neg="plain") for _ in range(20)])
        f, e, W = np.mean(fs), np.mean(es), np.mean(Ws)
        print(f" {p:5.3f}  {fixed_point(p):7.4f}   {f:7.4f}      {e:6.4f}   {2*e-1:+7.3f}        {W:9.1f}"
              f"            | {np.mean(fp):6.4f} {np.mean(ep):6.4f}")
        out["single_tie"].append(dict(p=p, f_theory=fixed_point(p), f_sim=float(f), e_sim=float(e),
                                      weight=float(2 * e - 1), W=float(W), f_plain=float(np.mean(fp))))
    print("\n p at which the design's f crosses 1/2 (tie turns inhibitory in settling): p = 1/3 exactly "
          f"(check: f(1/3) = {fixed_point(1/3):.4f})")
    print(f" f at p = 1/2 (a coin): {fixed_point(0.5):.4f}  (golden-ratio conjugate; a calibrated mind would hold 0.5)")

    print("\n== one sensed co-occurrence (candidate tie, g*a_u*a_v = 1, tau_sensed = 1) ==")
    W = 1.0
    c = W / (W + K)
    e = c * (1.0 - 0.5) + 0.5
    print(f" W=1 -> c={c:.2f}, f=1, e={e:.2f}, settling weight (e-.5)*2 = {2*e-1:+.2f}")

    print("\n== a concept X with m equally likely facets (X is met n times, each time with ONE facet) ==")
    print("   design rule: tie X->facet_i ; settle weight = (W+ - W-)/(W+k);  base-rate-relative weight = (W+ - b*W)/(W+k)")
    print("   m   n     f(X->facet)  e      weight_design   weight_base(b = facet base rate 1/(10m))")
    for m in [1, 2, 3, 4, 6, 10, 30]:
        for n in [10 * m, 100 * m, 1000 * m]:
            wp = np.zeros(m)
            wn = np.zeros(m)
            born = np.zeros(m, bool)
            cnt = np.zeros(m)          # plain positive counts (for the base-rate variant)
            expo = np.zeros(m)         # exposure of X since the tie's birth
            for t in range(n):
                i = rng.integers(m)
                Wv = wp + wn
                e = (wp + K / 2) / (Wv + K)
                if not born[i]:
                    born[i] = True
                    wp[i] = 1.0
                    cnt[i] = 1.0
                    expo[i] = 1.0
                    others = born.copy(); others[i] = False
                else:
                    wp[i] += 1.0
                    cnt[i] += 1.0
                    expo[i] += 1.0
                    others = born.copy(); others[i] = False
                wn[others] += e[others]
                expo[others] += 1.0
            Wv = wp + wn
            f = (wp / np.maximum(Wv, 1e-12)).mean()
            e = ((wp + K / 2) / (Wv + K)).mean()
            wd = ((wp - wn) / (Wv + K)).mean()
            b = 1.0 / (10 * m)
            wb = ((cnt - b * expo) / (expo + K)).mean()
            print(f"  {m:3d} {n:6d}   {f:6.3f}     {e:6.3f}   {wd:+7.3f}        {wb:+7.3f}")
            out["facets"].append(dict(m=m, n=n, f=float(f), e=float(e), w_design=float(wd), w_base=float(wb)))
    json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
