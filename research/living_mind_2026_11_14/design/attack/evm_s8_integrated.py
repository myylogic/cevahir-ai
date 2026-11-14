#!/usr/bin/env python3
"""evm_s8_integrated.py -- one life, all pieces together: two tellers, a closed sense, categories, idle imagination.

The DESIGN arithmetic (LEAD_DESIGN §1, §2.5, §4, §6 read literally) and the CORRECTED arithmetic
(attack/evidence-math.md) live the identical stream.  Developmental curve: how good are the mind's expectations
about (object, property) pairs it was never allowed to look at, after N experiences?

World (random, no domain): 120 objects in 12 categories of 10; 12 properties.  A category has a rule for each
property (holds / does not hold); an object follows the rule with probability 0.9.  Two tellers: H is right 95% of the
time and makes 60% of the statements, N is right 60% of the time.  Every statement names ONE property of one object.
A quarter of the experiences are closed looks (present / absent) at pairs from a fixed lookable 20%; the other 80% of
pairs can only be known by being told or by generalising.  After every experience the mind idles r = 3 steps.

Design readings (the most favourable ones where the text is silent):
  * six class slots: the two tellers share the slot 'told' and one tau_told (§1, §10)
  * every existing tie out of a focus unit is a prediction; 'predicted, did not occur' adds g*a*e (own e) (§2.5)
  * Up sees own evidence only (reading B), cap = 1, lambda = 0.5; support is a readout, not stored (§4)
  * idle: a random object is taken up, its top prediction is fed back as an imagined event, LEARN is on (§6, §2.5)
  * trust in 'told' and 'imagined' = expectation of their trust tie, confirmed/refuted by sensed evidence (§1)
"""
import numpy as np
from evm_common import reliability

LAM = 0.5
S, T, I = 0, 1, 2


def make(seed, M=120, G=12, P=12, n_exp=6000):
    rng = np.random.default_rng(seed)
    cat = np.repeat(np.arange(G), M // G)
    rule = rng.random((G, P)) < 0.35
    flip = rng.random((M, P)) < 0.10
    truth = rule[cat] ^ flip
    lookable = rng.random((M, P)) < 0.20
    look_pairs = np.argwhere(lookable)
    ev = []
    for _ in range(n_exp):
        if rng.random() < 0.25:
            x, v = look_pairs[rng.integers(len(look_pairs))]
            ev.append(("look", int(x), int(v), -1))
        else:
            s = 0 if rng.random() < 0.6 else 1
            prec = 0.95 if s == 0 else 0.60
            x = int(rng.integers(M))
            tv = np.nonzero(truth[x])[0]; fv = np.nonzero(~truth[x])[0]
            if (rng.random() < prec and len(tv)) or len(fv) == 0:
                v = int(rng.choice(tv))
            else:
                v = int(rng.choice(fv))
            ev.append(("told", x, v, s))
    return cat, truth, lookable, ev, rng


class Design:
    """neg     'design' : every existing tie of the focus object that did not occur gets w- += e      (§2.5)
               'closed' : FIX of finding F1 only: silence writes nothing; a closed look that finds v absent
                          writes w-[sensed] += 1 on that one tie
       support 'design' : readout own + lambda * W(g -> v)                                              (§4)
               'base'   : FIX of F3/S1 only: the rule built WITHOUT the member sets the base rate of e
       r                : idle steps per experience (0 = no imagination)                                 (§6)"""

    def __init__(s, cat, P, r=3, seed=0, neg="design", support="design"):
        M = len(cat)
        s.cat = cat; s.M = M; s.P = P; s.r = r; s.neg = neg; s.support = support
        s.wp = np.zeros((3, M, P)); s.wn = np.zeros((3, M, P)); s.born = np.zeros((M, P), bool)
        s.tp = np.zeros(3); s.tn = np.zeros(3)
        s.rng = np.random.default_rng(seed)

    def tau(s):
        t = (s.tp + 0.5) / (s.tp + s.tn + 1.0)
        t[S] = 1.0
        return t

    def own(s, rows):
        t = s.tau()
        Wp = np.tensordot(t, s.wp[:, rows], axes=1); Wn = np.tensordot(t, s.wn[:, rows], axes=1)
        return Wp, Wn

    def e_own(s, x):
        Wp, Wn = s.own(x)
        return (Wp + 0.5) / (Wp + Wn + 1.0)

    def e_eff_rows(s, rows):
        Wp_all, Wn_all = s.own(slice(None))
        W = Wp_all + Wn_all
        f = np.where(W > 0, Wp_all / np.maximum(W, 1e-300), 0.0)
        u = np.minimum(W, 1.0)
        G = s.cat.max() + 1
        Sp = np.zeros((G, s.P)); Sn = np.zeros((G, s.P))
        np.add.at(Sp, s.cat, u * f); np.add.at(Sn, s.cat, u * (1 - f))
        if s.support == "design":        # §4 Down as a readout: own + lambda * W(g -> v), Up reading B, cap 1
            Wp = Wp_all[rows] + LAM * Sp[s.cat[rows]]; Wn = Wn_all[rows] + LAM * Sn[s.cat[rows]]
            return (Wp + 0.5) / (Wp + Wn + 1.0), (Wp + Wn) / (Wp + Wn + 1.0)
        Sp_i = (Sp[s.cat] - u * f)[rows]; Sn_i = (Sn[s.cat] - u * (1 - f))[rows]
        b_i = (Sp_i + 0.5) / (Sp_i + Sn_i + 1.0)
        Wp = Wp_all[rows]; Wn = Wn_all[rows]
        return (Wp + b_i) / (Wp + Wn + 1.0), (Wp + Wn) / (Wp + Wn + 1.0)

    def learn(s, x, occurred, prov, referee, absent=-1):
        """one moment with object x in the focus; `occurred` = property that occurred (or -1);
        `absent` = property a closed look found absent (or -1)."""
        e = s.e_own(x)
        miss = s.born[x].copy()
        if occurred >= 0:
            miss[occurred] = False
            if referee:
                for cl in (T, I):
                    if s.wp[cl, x, occurred] > 0:
                        s.tp[cl] += 1.0
            s.wp[prov, x, occurred] += 1.0; s.born[x, occurred] = True
        if s.neg == "design":
            if referee:
                t = s.tau()
                for cl in (T, I):
                    s.tn[cl] += t[cl] * float(((s.wp[cl, x] > 0) & miss).sum())
            s.wn[prov, x, miss] += e[miss]                   # predicted, did not occur: g*a*e
        elif absent >= 0:                                    # closed observation of absence, unweighted
            for cl in (T, I):
                if s.wp[cl, x, absent] > 0:
                    s.tn[cl] += 1.0
            s.wn[prov, x, absent] += 1.0; s.born[x, absent] = True

    def experience(s, ev, truth):
        kind, x, v, src = ev
        if kind == "told":
            s.learn(x, v, T, referee=False)
        elif truth[x, v]:
            s.learn(x, v, S, referee=True)
        else:
            s.learn(x, -1, S, referee=True, absent=v)
        for _ in range(s.r):                                 # idle: top prediction of a random object, fed back
            y = int(s.rng.integers(s.M))
            if not s.born[y].any():
                continue
            e_eff, _ = s.e_eff_rows(np.array([y]))
            cand = np.where(s.born[y], e_eff[0], -1.0)
            s.learn(y, int(np.argmax(cand)), I, referee=False)


class Corrected:
    def __init__(s, cat, P):
        M = len(cat)
        s.cat = cat; s.M = M; s.P = P
        s.n_T = np.zeros((2, M, P)); s.N_T = np.zeros((2, M))    # per teller: assertions, opportunities
        s.n_L = np.zeros((M, P)); s.N_L = np.zeros((M, P))       # closed look: present, looks

    def experience(s, ev, truth):
        kind, x, v, src = ev
        if kind == "told":
            s.n_T[src, x, v] += 1.0; s.N_T[src, x] += 1.0
        else:
            s.N_L[x, v] += 1.0
            if truth[x, v]:
                s.n_L[x, v] += 1.0
        # idle steps write nothing

    def read(s, use_chi=True, use_base=True, per_source=True, use_cat=True, chance=True):
        """every derived number is a function of the raw counters (nothing derived is stored)."""
        f_L = np.where(s.N_L > 0, s.n_L / np.maximum(s.N_L, 1e-300), 0.0)
        c_L = s.N_L / (s.N_L + 1.0)
        b = (0.5 + (c_L * f_L).sum()) / (1.0 + c_L.sum()) if use_base else 0.5   # base rate of 'holds'
        Wp = s.n_L.copy(); Wn = s.N_L - s.n_L
        info = {"b": float(b)}
        if per_source:
            tellers = [(s.n_T[0], s.N_T[0]), (s.n_T[1], s.N_T[1])]
        else:                                                                     # one slot for the class 'told'
            tellers = [(s.n_T.sum(0), s.N_T.sum(0))]
        for t, (n_T, N_T) in enumerate(tellers):
            w = c_L * f_L
            if use_chi == "self":      # no closed referee: how often does the source say AGAIN what it said once
                said = n_T > 0
                chi = ((n_T - 1.0)[said]).sum() / (1.0 + (np.broadcast_to(N_T[:, None], n_T.shape) - 1.0)[said].sum())
            elif use_chi:              # refereed by the closed sense: recall on pairs known to hold (birth: 0)
                chi = (w * n_T).sum() / (1.0 + (w * N_T[:, None]).sum())
            else:
                chi = 0.0
            E = np.maximum(n_T, chi * N_T[:, None])
            f = np.where(E > 0, n_T / np.maximum(E, 1e-300), 0.0)
            c = E / (E + 1.0)
            A = (1.3 + (c_L * c * f * f_L).sum()) / (2.0 + (c_L * c * f).sum())   # precision of consistent assertions
            rho = reliability(A, b) if chance else min(A, 1 - 1e-9)
            Kc = rho / (1 - rho) if rho < 1 else 1e12
            Ws = np.where((E > 0) & (Kc > 0), E * Kc / np.maximum(E + Kc, 1e-300), 0.0)
            Wp += Ws * f; Wn += Ws * (1 - f)
            info[f"teller{t}"] = (float(A), float(rho), float(chi))
        W = Wp + Wn
        if not use_cat:
            return (Wp + b) / (W + 1.0), W / (W + 1.0), info
        fi = np.where(W > 0, Wp / np.maximum(W, 1e-300), 0.0)
        u = np.minimum(W, 1.0)
        G = s.cat.max() + 1
        Sp = np.zeros((G, s.P)); Sn = np.zeros((G, s.P))
        np.add.at(Sp, s.cat, u * fi); np.add.at(Sn, s.cat, u * (1 - fi))
        Sp_i = Sp[s.cat] - u * fi; Sn_i = Sn[s.cat] - u * (1 - fi)             # rule WITHOUT the member
        S_i = Sp_i + Sn_i
        fg = np.where(S_i > 0, Sp_i / np.maximum(S_i, 1e-300), b)
        wgt = (W / (W + 1.0)) * (S_i / (S_i + 1.0))
        agree = 1.0 - np.abs(fi - fg)
        A_dom = (1.5 + (wgt * agree).sum()) / (2.0 + wgt.sum())               # conformity of members to their rules
        ch = b * b + (1 - b) * (1 - b)                                         # agreement of two unrelated ties
        rho_dom = min(max((A_dom - ch) / (1 - ch), 1e-3), 1 - 1e-3)
        k_g = (1 - rho_dom) / rho_dom                                         # horizon of rules
        b_i = (Sp_i + k_g * b) / (S_i + k_g)
        e = (Wp + b_i) / (W + 1.0)
        info["conformity"] = (float(A_dom), float(rho_dom), float(k_g))
        return e, W / (W + 1.0), info


def main():
    print(__doc__.split("Design readings")[0].strip().split("\n\n", 1)[1])
    print()
    marks = (250, 500, 1000, 2000, 4000, 6000)
    variants = [("design, literal", dict()),
                ("design without imagination (r = 0)", dict(r=0)),
                ("design + fix F1 (open world, unweighted closed negatives)", dict(neg="closed")),
                ("design + fix F1, without imagination", dict(neg="closed", r=0)),
                ("design + fix F1 + fix S1 (base-rate support), r = 3", dict(neg="closed", support="base")),
                ("design + fix F1 + fix S1, without imagination", dict(neg="closed", support="base", r=0))]
    rows = {m: [] for m in marks}
    abl = {name: [] for name, _ in variants}
    last = None
    for seed in range(5):
        cat, truth, lookable, ev, rng = make(seed)
        P = truth.shape[1]
        minds = [Design(cat, P, seed=seed, **kw) for _, kw in variants]
        c = Corrected(cat, P)
        ev_mask = ~lookable
        y = truth[ev_mask].astype(float)
        for i, e in enumerate(ev, 1):
            for d in minds:
                d.experience(e, truth)
            c.experience(e, truth)
            if i in marks:
                d = minds[0]
                ed, cd = d.e_eff_rows(slice(None))
                ec, cc, info = c.read()
                bd = float(((ed[ev_mask] - y) ** 2).mean()); bc = float(((ec[ev_mask] - y) ** 2).mean())
                accd = float(((ed[ev_mask] > 0.5) == (y > 0.5)).mean()); accc = float(((ec[ev_mask] > 0.5) == (y > 0.5)).mean())
                rows[i].append((bd, bc, accd, accc, float(cd[ev_mask].mean()), float(cc[ev_mask].mean()),
                                d.tau()[T], d.tau()[I]))
                last = info
        for (name, _), d in zip(variants, minds):
            ed, cd = d.e_eff_rows(slice(None))
            abl[name].append((float(((ed[ev_mask] - y) ** 2).mean()), float(((ed[ev_mask] > 0.5) == (y > 0.5)).mean()),
                              float(cd[ev_mask].mean()), d.tau()[T], d.tau()[I]))
        for name, kw in (("CORRECTED (all of evidence-math.md)", dict()),
                         ("corrected, silence not counted (chi = 0)", dict(use_chi=False)),
                         ("corrected, chi from the teller's own repetition rate", dict(use_chi="self")),
                         ("corrected, base rate fixed at 1/2", dict(use_base=False)),
                         ("corrected, one slot for both tellers", dict(per_source=False)),
                         ("corrected, reliability = hit rate (no chance correction)", dict(chance=False)),
                         ("corrected, no category base rate (own evidence only)", dict(use_cat=False))):
            ec, cc, info = c.read(**kw)
            abl.setdefault(name, []).append(
                (float(((ec[ev_mask] - y) ** 2).mean()), float(((ec[ev_mask] > 0.5) == (y > 0.5)).mean()),
                 float(cc[ev_mask].mean()), float("nan"), float("nan")))
    base = float(truth.mean())
    print("   pairs the mind may never look at; mean of 5 lives.  Brier: lower is better; 'always 1/2' = 0.250;")
    print(f"   'always the base rate' = {base*(1-base):.3f}")
    print("   experiences |  DESIGN: Brier  right side  mean c  tau_told  tau_imagined |  CORRECTED: Brier  right side  mean c (lived)")
    for m in marks:
        r = np.mean(rows[m], 0)
        print(f"     {m:6d}    |          {r[0]:.3f}    {r[2]:5.1%}    {r[4]:.3f}    {r[6]:.3f}      {r[7]:.3f}     |"
              f"             {r[1]:.3f}    {r[3]:5.1%}     {r[5]:.3f}")
    print("   corrected, learned by the end of the last life:")
    print(f"     base rate b = {last['b']:.3f};  teller H: precision A = {last['teller0'][0]:.3f}, reliability rho = {last['teller0'][1]:.3f},"
          f" completeness chi = {last['teller0'][2]:.3f}")
    print(f"     teller N: precision A = {last['teller1'][0]:.3f}, rho = {last['teller1'][1]:.3f}, chi = {last['teller1'][2]:.3f};"
          f"  category conformity A = {last['conformity'][0]:.3f}, rho = {last['conformity'][1]:.3f}, rule horizon k_g = {last['conformity'][2]:.2f}")
    print("\n   which fix buys what (after 6000 experiences, mean of 5 lives):")
    print("   variant                                                      Brier   right side   mean c   tau_told  tau_imagined")
    for name, v in abl.items():
        r = np.mean(v, 0)
        tt = "   -  " if np.isnan(r[3]) else f"{r[3]:.3f}"
        ti = "   -  " if np.isnan(r[4]) else f"{r[4]:.3f}"
        print(f"   {name:58s}   {r[0]:.3f}    {r[1]:5.1%}      {r[2]:.3f}    {tt}     {ti}")


if __name__ == "__main__":
    main()
