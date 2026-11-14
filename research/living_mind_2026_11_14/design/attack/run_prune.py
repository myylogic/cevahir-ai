#!/usr/bin/env python3
"""run_prune.py -- "Memory grows with ties; bounded by pruning under a budget given by the body" (§10),
"pruning (dropping ties with low c and low use, oldest first)" (§8).   Does it keep the knowledge that matters?

The settling pathologies of the design are switched off here (preset 'patched'), otherwise nothing can be read
out at all and pruning could not be judged.  Same stream, same tie budget, three pruning rules:
    design : lowest class of (c, use) first, oldest first inside a class
    z      : least significant deviation from the base rate first,  |W+ - b_v W| / sqrt(W + k)
    random : control
Half-way through, a second, disjoint domain starts (new entities, relations, values): can the mind still learn?
Measures: association recall (entity -> the values it was met with) per frequency band, survival of the ties that
carry the facts, share of the budget spent on ties that carry no fact.
"""
import sys, json, numpy as np, zsim

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
OUT = "prune_results.jsonl"


def tie_census(eng, worlds):
    """classify surviving lag-0 ties; survival of fact-carrying ties E->V per band"""
    inv = {u: k for k, u in eng.sym.items()}
    facts = {}
    for w in worlds:
        for (e, r) in w.seen:
            facts[("%sE%d" % (w.prefix, e), "%sV%d" % (w.prefix, w.fact(e, r)))] = (w, e, r)
    comp = dict(fact=0, entity_value_nonfact=0, with_relation=0, with_composed=0, with_intero=0, other=0)
    for u in range(eng.n):
        A = eng.out[0][u]
        if A is None or A.n == 0:
            continue
        su = inv.get(u)
        for v in A.dst[:A.n].tolist():
            sv = inv.get(v)
            if su is None or sv is None:
                comp["with_composed"] += 1
            elif su[0] == "~" or sv[0] == "~":
                comp["with_intero"] += 1
            elif "R" in su[:2] or "R" in sv[:2]:
                comp["with_relation"] += 1
            elif (su, sv) in facts or (sv, su) in facts:
                comp["fact"] += 1
            elif {su.lstrip("b")[0], sv.lstrip("b")[0]} == {"E", "V"}:
                comp["entity_value_nonfact"] += 1
            else:
                comp["other"] += 1
    surv = {}
    for (se, sv), (w, e, r) in facts.items():
        band = ("second " if w.prefix else "") + ("head" if e < 100 else ("mid" if e < 1000 else "tail"))
        ue, uv = eng.sym.get(se), eng.sym.get(sv)
        A = eng.out[0][ue] if ue is not None else None
        ok = int(A is not None and uv in A.idx)
        for b in (band, ("second " if w.prefix else "") + "exception") if w.is_exc[e, r] else (band,):
            a = surv.setdefault(b, [0, 0]); a[0] += ok; a[1] += 1
    return comp, {b: (a[0] / a[1], a[1]) for b, a in surv.items()}


def main():
    variants = [("no budget (reference)", dict()),
                ("budget, design rule (low c, low use, oldest first)", dict(budget=BUDGET, prune="design")),
                ("budget, significance rule z", dict(budget=BUDGET, prune="z")),
                ("budget, rule z + no EXPECT ties between a composed unit and its parts", dict(budget=BUDGET, prune="z",
                                                                                         skip_part=True)),
                ]
    if len(sys.argv) > 3:
        variants = [v for v in variants if any(k in v[0] for k in sys.argv[3:])]
    rngp = np.random.default_rng(9)
    for label, kw in variants:
        cfg = dict(zsim.PRESETS["patched"]); cfg.update(kw)
        eng, world, rows = zsim.run(label, N, cfg, checkpoints=[N // 2, N], second=dict(at=N // 2), quiet=True,
                                    ncap=N + 30000)
        # the second world object is not returned by run(); rebuild its fact table deterministically
        w2 = zsim.World(seed=1 + 77, prefix="b")
        # replay which facts of world 2 were seen: a fact counts as seen if its composed unit exists
        for (key, cu) in list(eng.tuples.items()):
            names = [k for k in (next((n for n, u in eng.sym.items() if u == x), None) for x in key)]
        r = rows[-1]
        comp, surv = tie_census(eng, [world])
        ties = eng.nties[0] + eng.nties[1]
        rec = r["recall"]; rec2 = r.get("recall_second", {})
        print(f"\n[{label}]  ties={ties}  (lag0 {eng.nties[0]}, lag1 {eng.nties[1]})  pruned={eng.C['pruned']} in "
              f"{eng.C['prune_passes']} passes")
        print("   association recall, first domain  (head/mid/tail): "
              + " / ".join("%.2f" % rec[b]["assoc5"] for b in ("head", "mid", "tail")))
        if rec2:
            print("   association recall, SECOND domain (head/mid/tail): "
                  + " / ".join("%.2f" % rec2[b]["assoc5"] if b in rec2 else "-" for b in ("head", "mid", "tail")))
        print("   fact-carrying tie E->V still present (head/mid/tail/exception): "
              + " / ".join("%.2f (n=%d)" % surv[b] if b in surv else "-" for b in ("head", "mid", "tail", "exception")))
        tot0 = sum(comp.values())
        print("   lag-0 ties by content: " + ", ".join("%s %.1f%%" % (k, 100 * v / max(1, tot0)) for k, v in comp.items()))
        with open(OUT, "a") as f:
            f.write(json.dumps(dict(label=label, N=N, budget=BUDGET, ties=ties, recall=rec, recall_second=rec2,
                                    survival=surv, composition=comp, pruned=eng.C["pruned"])) + "\n")


if __name__ == "__main__":
    main()
