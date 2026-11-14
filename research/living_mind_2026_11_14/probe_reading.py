# -*- coding: utf-8 -*-
"""F7/F8 side study: how should an instance never told a dimension be read from its generalisations?

Lives one life per (domain, seed, learning variant) and scores the held-out probes of T3 with four readings:
  default  - the variant's own reading (nested for changed/symmetric)
  sharpest - the one generalisation that pins down a sense/act value most decisively (max over targets of
             lam e(i->g) conviction f) speaks for all its targets; own evidence on top (none for held-out)
Prints one JSON line per life. Exploratory; not part of the T1-T9 records.
"""
import json, sys
from run import World, live, Probe, Event, CHANNEL, TOLD, score
from zihin import read as R
from zihin.evidence import conviction, frequency


def sharp_read(mind, unit):
    def world_t(t):
        c = int(mind.S.u_channel[t]); return c >= 0 and mind.ch.kind(c) in ("sense", "act")
    best = None
    for g, e in R.generalisations(mind, unit):
        om = {t: pn for t, pn in R.own_mass(mind, g).items() if world_t(t)}
        if not om:
            continue
        s = max(mind.cfg.lam * e * conviction(p + n, mind.cfg.k) * frequency(p, n) for p, n in om.values())
        if best is None or s > best[0]:
            best = (s, e, om)
    out = {}
    if best is not None:
        _, e, om = best
        for t, (p, n) in om.items():
            c = mind.cfg.lam * e * conviction(p + n, mind.cfg.k)
            out[t] = c * frequency(p, n) + (1 - c) * 0.5
    for t, (p, n) in R.own_mass(mind, unit).items():
        c = conviction(p + n, mind.cfg.k)
        out[t] = c * frequency(p, n) + (1 - c) * out.get(t, 0.5)
    return out


def main(domain, seed, variant):
    w = World.make(seed, domain)
    mind, _, state = live(w, variant, 600, record=False)
    res = {"domain": domain, "seed": seed, "variant": variant}
    for name in ("default", "sharpest"):
        oks = {"told": [], "induced": [], "member": []}
        items = [(e, d, w.rule[(w.cat_of[e], d)], "told" if d == w.told_rule_dim else "induced")
                 for e in w.held_out if state["isa"].get(e) for d in w.ruled]
        items += [(e, d, w.value[(e, d)], "member") for (e, d), h in state["heard"].items()
                  if d in w.ruled and not h["lie"]][:40]
        for ent, dim, right, kind in items:
            twin = Probe(mind).twin
            for s in w.prefix(ent, dim):
                twin.step([Event(CHANNEL, s, TOLD)])
            if name == "default":
                ex = dict(twin.expected(CHANNEL, "specific"))
            else:
                raw = sharp_read(twin, twin.context)
                ex = {twin.ch.symbol(int(twin.S.u_channel[t]), int(twin.S.u_symbol[t])) if hasattr(twin.ch, "symbol")
                      else t: v for t, v in raw.items()}
            oks[kind].append(score(ex, w.values[dim], right)["ok"])
        for k, v in oks.items():
            res["%s_%s" % (name, k)] = sum(v) / len(v) if v else None
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
