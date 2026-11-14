# -*- coding: utf-8 -*-
"""T3 diagnosis: is the knowledge for a held-out instance IN the substance, or is only its selection failing?

For each held-out probe: does a frame (C, dim) exist whose category C has exactly the true class as members, does
it hold the rule value as its top value among the dimension's values, and which generalisation the reliable
reading chose (rank of the class frame by e(i->g) x reliability). Scoring uses the world's truth; the mind never
sees it. Exploratory, one JSON line per life.
"""
import json, sys
from run import World, live, Probe, Event, CHANNEL, TOLD
from zihin import read as R
from zihin.evidence import frequency
from zihin.store import ABSTRACT, CATEGORY


def main(domain, seed, variant, n, night_every=0, night_length=0):
    w = World.make(seed, domain)
    mind, _, state = live(w, variant, n, record=False, night_every=night_every, night_length=night_length)
    S = mind.S
    sym = {u: mind.ch.symbol(c, si) for (c, si), u in S.atom_index.items()}
    classes = {c: {e for e in w.entities if w.cat_of[e] == c} for c in w.categories}
    out = {"domain": domain, "seed": seed, "variant": variant, "n": n, "nights": [night_every, night_length], "items": 0, "class_frame_exists": 0,
           "class_frame_right_top": 0, "class_frame_chosen": 0, "answered_right": 0}
    for ent in w.held_out:
        if not state["isa"].get(ent):
            continue
        for dim in w.ruled:
            tw = Probe(mind).twin
            for s in w.prefix(ent, dim):
                tw.step([Event(CHANNEL, s, TOLD)])
            right = w.rule[(w.cat_of[ent], dim)]
            gens = R.generalisations(tw, tw.context)
            ranked = sorted(((ea * R.reliability(tw, g), g) for g, ea in gens), reverse=True)
            cls = None
            for g, ea in gens:
                for p in tw.S.parts(g):
                    if int(tw.S.u_kind[p]) & CATEGORY:
                        members = {sym.get(int(tw.S.t_src[t])) for t in tw.S.inc.get((p, ABSTRACT), ())}
                        if members == classes[w.cat_of[ent]]:
                            cls = g
            out["items"] += 1
            ex = dict(tw.expected(CHANNEL, "specific"))
            e = {v: ex.get(v, 0.5) for v in w.values[dim]}
            out["answered_right"] += int(e[right] > 0.5 and e[right] > max(e[v] for v in e if v != right))
            if cls is None:
                continue
            out["class_frame_exists"] += 1
            f = {sym.get(t): frequency(p, q) for t, (p, q) in R.own_mass(tw, cls).items() if sym.get(t) in w.values[dim]}
            out["class_frame_right_top"] += int(bool(f) and max(f, key=f.get) == right)
            out["class_frame_chosen"] += int(bool(ranked) and ranked[0][1] == cls)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), *[int(a) for a in sys.argv[5:7]])
