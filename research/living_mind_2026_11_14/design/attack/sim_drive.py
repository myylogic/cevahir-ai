#!/usr/bin/env python3
"""sim_drive.py -- LEAD_DESIGN.md §5 "D(u) = sum_{ties t at u} use(t) * (1 - c_t) ... Relief = -dD is felt" and
§8 "The same ties and the same rule learn (internal state shape) + (knob act) -> relief", "pruning ... is an
internal act whose outcome is felt".

(1) one tie: its contribution use*(1-c) over its life, for the design's learning rule (analytic limit f/p).
(2) the engine (design settings; interoception off so that perception works at all): total drive D over time,
    share of moments with positive relief.
(3) knob experiment: from one common state, continue the same stream under different internal acts and measure
    the relief each act is followed by (this is exactly the outcome the one rule would attach to the act).
(4) the fix: drive = sum need_t * (1 - c_t) with need a bounded recency-weighted rate, a tie that is pruned counts
    as c = 0 for as long as it is needed, and ties born in the moment do not count in that moment's relief.
"""
import copy, json, math, sys
import numpy as np
import zsim

K = 1.0


def f_design(p):
    return 1.0 if p >= 1 else (-p + math.sqrt(p * p + 4 * p * (1 - p))) / (2 * (1 - p))


def one_tie():
    print("== (1) one tie u->v, P(v|u) = p, design rule: contribution use*(1-c) to D(u) after n uses ==")
    print("    p     n=1    n=10   n=100  n=1000  n=10000   limit f/p")
    rng = np.random.default_rng(0); res = {}
    for p in (1.0, 0.5, 0.2, 0.05, 0.01):
        row = []
        for n in (1, 10, 100, 1000, 10000):
            vals = []
            for _ in range(30):
                wp, wn = 1.0, 0.0
                for i in range(n - 1):
                    e = (wp + K / 2) / (wp + wn + K)
                    if rng.random() < p: wp += 1
                    else: wn += e
                W = wp + wn; vals.append(n * (1 - W / (W + K)))
            row.append(float(np.mean(vals)))
        print(f"   {p:4.2f}  " + "  ".join("%5.2f" % v for v in row) + f"     {f_design(p)/p:5.2f}")
        res[p] = row
    print("   -> a tie that is confirmed for ever contributes MORE drive (>= 1) than a brand-new guess (0.5): D rises with knowledge")
    return res


def total_drive(eng, recent=None):
    D = 0.0
    for lag in (0, 1):
        for u in (range(eng.n) if recent is None else recent):
            A = eng.out[lag][u]
            if A is None or A.n == 0:
                continue
            n = A.n; W = A.wp[:n] + A.wn[:n]
            D += float((A.use[:n] * (K / (W + K))).sum())
    return D


def snapshot_need(eng):
    """need of a tie = its use rate so far (bounded by 1); reference set for the fixed drive"""
    snap = {}
    for lag in (0, 1):
        for u in range(eng.n):
            A = eng.out[lag][u]
            if A is None or A.n == 0:
                continue
            n = A.n; age = np.maximum(1, eng.t - A.born[:n] + 1)
            need = np.minimum(1.0, A.use[:n] / age)
            W = A.wp[:n] + A.wn[:n]
            for v, nd, w in zip(A.dst[:n].tolist(), need.tolist(), W.tolist()):
                snap[(lag, u, v)] = (nd, w / (w + K))
    return snap


def fixed_drive_after(eng, snap):
    D0 = D1 = 0.0
    for (lag, u, v), (need, c0) in snap.items():
        D0 += need * (1 - c0)
        A = eng.out[lag][u]
        j = A.idx.get(v) if A is not None else None
        if j is None:
            c1 = 0.0                      # pruned: the question is open again
        else:
            W = A.wp[j] + A.wn[j]; c1 = W / (W + K)
        D1 += need * (1 - c1)
    return D0, D1


def score_next(eng, next_syms, scratch):
    """proper score (log loss) of the lag-1 predictions issued by the last focus, judged by the next real events:
    -log p(v) for every event v that came (p = strongest expectation of v, or v's base rate if nobody predicted it)
    + -log(1 - e) for every prediction that did not come.  Cannot be lowered by forgetting or by believing harder."""
    pf_ids, pf_vals = eng.prev_focus; touched = []
    for u in pf_ids.tolist():
        d, e = eng.expect(u, 1)
        if d is None:
            continue
        scratch[d] = np.maximum(scratch[d], e); touched.append(d)
    occ = [eng.sym.get(x) for x in next_syms]
    loss_occ = 0.0
    for v in occ:
        if v is None:
            p = 1.0 / (eng.t + 2.0)
        else:
            p = scratch[v] if scratch[v] > 0 else max(eng.occ[v] / (eng.t - eng.birth[v] + 1.0), 1.0 / (eng.t + 2.0))
        loss_occ += -math.log(min(max(p, 1e-9), 1.0))
    loss_fail = 0.0; npred = 0
    if touched:
        allv = np.unique(np.concatenate(touched)); ev = scratch[allv].copy()
        mask = ~np.isin(allv, [v for v in occ if v is not None])
        loss_fail = float(-np.log1p(-np.minimum(ev[mask], 0.999)).sum()); npred = int(mask.sum())
        scratch[allv] = 0.0
    return loss_occ, loss_fail, npred


def main():
    out = {"one_tie": one_tie()}
    print("\n== (2) engine, design settings (interoception off): total drive over life ==")
    eng = zsim.Engine(zsim.Cfg(intero="none"), ncap=60000); world = zsim.World(seed=1)
    stream = world.stream(20000); prev = 0.0; pos = 0; tot = 0; traj = []
    for i in range(1, 3001):
        eng.moment(next(stream))
        if i % 100 == 0:
            D = total_drive(eng); ties = eng.nties[0] + eng.nties[1]
            traj.append((i, D, ties)); pos += int(D < prev); tot += 1; prev = D
    for i, D, ties in traj[::5]:
        print(f"   moment {i:>5}: ties={ties:>7}  D={D:12.0f}  D/tie={D/ties:5.2f}")
    print(f"   D decreased in {pos} of {tot} windows of 100 moments (relief = -dD > 0)  [per-moment share measured in "
          f"run_stream: see stream_literal.jsonl, relief_pos_share]")
    out["traj"] = traj

    print("\n== (3) which internal act is followed by relief?  (same state, same next 300 moments of the same world) ==")
    nxt = [next(stream) for _ in range(300)]
    acts = {
        "no act (keep living)":            lambda e: None,
        "focus size F: 7 -> 3":            lambda e: setattr(e.cfg, "F", 3),
        "focus size F: 7 -> 12":           lambda e: setattr(e.cfg, "F", 12),
        "plasticity g: 1 -> 0.25":         lambda e: setattr(e.cfg, "g", 0.25),
        "plasticity g: 1 -> 4":            lambda e: setattr(e.cfg, "g", 4.0),
        "prune 10% (low c, low use, oldest first)": lambda e: e.prune_to(int(0.9 * (e.nties[0] + e.nties[1]))),
        "prune 50%":                       lambda e: e.prune_to(int(0.5 * (e.nties[0] + e.nties[1]))),
    }
    snap = snapshot_need(eng); D_before = total_drive(eng)
    print(f"   state: {eng.n} units, {eng.nties[0]+eng.nties[1]} ties, D = {D_before:.0f}")
    print("   act                                          design relief (-dD)   need-weighted, pruned=unknown   "
          "log loss per moment on what really came (on events / on failed predictions)")
    res = {}; scratch = np.zeros(eng.ncap)
    for name, act in acts.items():
        e2 = copy.deepcopy(eng); e2.cfg = copy.copy(eng.cfg)
        act(e2)
        lo = lf = 0.0
        for s in nxt:
            a, b, _ = score_next(e2, s, scratch); lo += a; lf += b
            e2.moment(s)
        D_after = total_drive(e2)
        D0, D1 = fixed_drive_after(e2, snap)
        res[name] = dict(design_relief=D_before - D_after, fixed_relief=D0 - D1, loss_events=lo / len(nxt),
                         loss_failed=lf / len(nxt))
        print(f"   {name:<44} {D_before - D_after:>+14.0f}   {D0 - D1:>+18.1f}   "
              f"{(lo + lf) / len(nxt):>14.2f}  ({lo / len(nxt):.2f} / {lf / len(nxt):.2f})")
    out["acts"] = res
    best = max(res, key=lambda k: res[k]["design_relief"]); bestf = max(res, key=lambda k: res[k]["fixed_relief"])
    bestl = min(res, key=lambda k: res[k]["loss_events"] + res[k]["loss_failed"])
    print(f"   -> act followed by the most relief under the design's D: '{best}';  under need-weighted (1-c): '{bestf}';"
          f"  lowest real prediction loss: '{bestl}'")

    print("\n== (4) the same knob experiment on the engine with the proposed fixes (base-rate reference, unweighted negatives) ==")
    engp = zsim.Engine(zsim.Cfg(intero="none", ref="base", neg="plain", clamp=True, create="events", settle="monotone",
                                drive=False), ncap=60000)
    w2 = zsim.World(seed=1); st2 = w2.stream(20000)
    for i in range(3000):
        engp.moment(next(st2))
    nxt2 = [next(st2) for _ in range(300)]; resp = {}
    actsp = dict(acts); actsp.pop("plasticity g: 1 -> 0.25")
    for name, act in actsp.items():
        e2 = copy.deepcopy(engp); e2.cfg = copy.copy(engp.cfg); e2.cfg.prune = "design"
        act(e2)
        lo = lf = 0.0
        for s in nxt2:
            a, b, _ = score_next(e2, s, scratch); lo += a; lf += b
            e2.moment(s)
        resp[name] = dict(loss_events=lo / len(nxt2), loss_failed=lf / len(nxt2))
        print(f"   {name:<44} log loss per moment {(lo + lf) / len(nxt2):8.2f}  ({lo / len(nxt2):.2f} on events / {lf / len(nxt2):.2f} on failed predictions)")
    out["acts_patched"] = resp
    json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
