#!/usr/bin/env python3
"""sim_idle.py -- LEAD_DESIGN.md §6 "When no external event arrives (idle), the mind runs §2 on its own
predictions: the top prediction is fed back as an event with provenance 'imagined' on an internal channel, for up
to H steps, starting from high-drive units."  together with §2.5 "LEARN (always on) ... v occurred: w+[pi] += g*a_u
(pi = provenance of the confirming event)".

World: a Markov chain over 300 symbols (each symbol has 4 possible successors with random probabilities).
Life: 30 000 real steps (provenance sensed).  Then idle episodes of H = 20 steps.  Evidence is kept per provenance
(sensed, imagined) and combined as in §1:  W = tau_sensed*W_s + tau_imagined*W_i.
Reported: where imagination goes (distinct units, cycle), what it does to the expectations it travels on, and the
log loss of the mind's predictions on 10 000 further REAL steps, before and after idling.
"""
import json, math, sys
import numpy as np

K = 1.0
N_SYM = 300; H = 20


def make_chain(rng):
    succ = np.stack([rng.choice(N_SYM, 4, replace=False) for _ in range(N_SYM)])
    prob = rng.dirichlet(np.ones(4) * 1.0, N_SYM)
    return succ, prob


def walk(succ, prob, n, rng, start=0):
    seq = np.empty(n, np.int64); cur = start
    for i in range(n):
        cur = succ[cur, rng.choice(4, p=prob[cur])]; seq[i] = cur
    return seq


class Mind:
    def __init__(s, tau_im):
        s.wp = np.zeros((2, N_SYM, N_SYM)); s.wn = np.zeros((2, N_SYM, N_SYM))   # [provenance: 0 sensed, 1 imagined]
        s.born = np.zeros((N_SYM, N_SYM), bool); s.use = np.zeros((N_SYM, N_SYM))
        s.tau = np.array([1.0, tau_im])

    def e_row(s, u):
        Wp = s.tau[0] * s.wp[0, u] + s.tau[1] * s.wp[1, u]; Wn = s.tau[0] * s.wn[0, u] + s.tau[1] * s.wn[1, u]
        return (Wp + K / 2) / (Wp + Wn + K), Wp + Wn

    def learn(s, u, v, prov):
        """design rule §2.5 for the ties of u, when v is the event that followed"""
        e, _ = s.e_row(u); bo = s.born[u]
        inc = np.where(bo, e, 0.0); inc[v] = 0.0
        s.wn[prov, u] += inc
        s.use[u][bo] += 1
        if bo[v]:
            s.wp[prov, u, v] += 1.0
        else:
            bo[v] = True; s.wp[prov, u, v] = 1.0; s.use[u, v] = 1

    def drive(s):
        D = np.zeros(N_SYM)
        for u in range(N_SYM):
            e, W = s.e_row(u); D[u] = (s.use[u] * (K / (W + K)))[s.born[u]].sum()
        return D


def logloss(mind, seq, succ, prob):
    tot = 0.0; cur = seq[0]
    for v in seq[1:]:
        e, _ = mind.e_row(cur); bo = mind.born[cur]
        p = e[v] if bo[v] else 1.0 / N_SYM
        tot += -math.log(max(p, 1e-9))
        mask = bo.copy(); mask[v] = False
        tot += float(-np.log1p(-np.minimum(e[mask], 0.999)).sum())
        cur = v
    return tot / (len(seq) - 1)


def calib_err(mind, succ, prob, on=None):
    errs = []
    for u in (range(N_SYM) if on is None else on):
        e, _ = mind.e_row(u)
        for j in range(4):
            v = succ[u, j]
            if mind.born[u, v]:
                errs.append(abs(e[v] - prob[u, j]))
    return float(np.mean(errs)) if errs else float("nan")


def main():
    out = {}
    for tau_im, mode in [(0.3, "argmax"), (0.1, "argmax"), (0.3, "sample"), (0.3, "fix")]:
        rng = np.random.default_rng(0)
        succ, prob = make_chain(rng)
        life = walk(succ, prob, 30000, rng); test = walk(succ, prob, 10000, rng, start=int(life[-1]))
        mind = Mind(tau_im)
        for a, b in zip(life[:-1], life[1:]):
            mind.learn(int(a), int(b), 0)
        base_ll = logloss(mind, test, succ, prob); base_ce = calib_err(mind, succ, prob)
        label = {"argmax": "design: top prediction fed back, LEARN always on", "sample": "variant: prediction SAMPLED in proportion to e, LEARN always on",
                 "fix": "fix: an imagined event is not evidence for the tie that produced it"}[mode]
        print(f"\n== tau_imagined = {tau_im}; {label} ==")
        print(f"   after life (30000 real steps): calibration error mean|e-p| = {base_ce:.3f}, log loss on 10000 real steps = {base_ll:.3f}")
        print("   idle episodes  distinct units visited  cycle length  mean e on the path (true p)   calib.err on path / everywhere   log loss on real steps")
        visited = set(); done = 0; rows = []
        for target in (10, 100, 1000, 10000):
            while done < target:
                start = int(np.argmax(mind.drive())) if done % 50 == 0 else start   # "starting from high-drive units"
                cur = start; path = [cur]
                for h in range(H):
                    e, _ = mind.e_row(cur); bo = mind.born[cur]
                    ee = np.where(bo, e, -1.0)
                    if mode == "sample":
                        pp = np.where(bo, e, 0.0); nxt = int(rng.choice(N_SYM, p=pp / pp.sum()))
                    else:
                        nxt = int(np.argmax(ee))
                    if mode != "fix":
                        mind.learn(cur, nxt, 1)
                    cur = nxt; path.append(cur)
                visited.update(path); done += 1
            # cycle length of the argmax walk from the current start
            cur = start; seen = {}
            for i in range(2000):
                if cur in seen:
                    cyc = i - seen[cur]; break
                seen[cur] = i; e, _ = mind.e_row(cur); cur = int(np.argmax(np.where(mind.born[cur], e, -1.0)))
            pe = []; pt = []
            for u in set(path[:-1]):
                e, _ = mind.e_row(u); v = int(np.argmax(np.where(mind.born[u], e, -1.0)))
                j = list(succ[u]).index(v) if v in succ[u] else None
                pe.append(e[v]); pt.append(prob[u, j] if j is not None else 0.0)
            ce_path = calib_err(mind, succ, prob, on=set(path)); ce_all = calib_err(mind, succ, prob)
            ll = logloss(mind, test, succ, prob)
            print(f"   {target:>8}        {len(visited):>6}               {cyc:>4}          {np.mean(pe):.3f} ({np.mean(pt):.3f})               "
                  f"{ce_path:.3f} / {ce_all:.3f}                  {ll:.3f}")
            rows.append(dict(episodes=target, visited=len(visited), cycle=cyc, e_path=float(np.mean(pe)), p_path=float(np.mean(pt)),
                             ce_path=ce_path, ce_all=ce_all, logloss=ll))
        out[f"{mode}_tau{tau_im}"] = dict(base_ll=base_ll, base_ce=base_ce, rows=rows)
    json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
