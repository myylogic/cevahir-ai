#!/usr/bin/env python3
"""After the red episode (stages 1-10 of trace_red.py), how much more must be lived before a
sub-category forms? Adds further colours one at a time and prints the split test.
Run: python3 split_continuation.py   (about 3 minutes)"""
import json
from bg_core import EXC, HAS, ISA, Mind
from trace_red import background, prior_life

def episode(m):
    m.perceive(ISA, "red", "colour"); m.perceive(HAS, "red", "dark"); m.perceive(HAS, "red", "light")
    for c in ("blue", "green"):
        m.perceive(ISA, c, "colour")
    for s, o in (("blue", "dark"), ("blue", "light"), ("green", "light"), ("green", "dark")):
        m.perceive(HAS, s, o)
    for name, a, b in (("black", "dark", "light"), ("white", "light", "dark")):
        m.perceive(ISA, name, "colour")
        for _ in range(6):
            m.perceive(HAS, name, a, 1, "sense", event=True)
            m.perceive(HAS, name, b, 0, "sense", event=True)
    for s in ("red", "blue", "green"):
        for _ in range(4):
            m.perceive(HAS, s, "dark", 1, "sense", event=True)
            m.perceive(HAS, s, "light", 1, "sense", event=True)

def report(m, tag):
    u = m.ix["MIX(dark|light)"]; c = m.ix["colour"]
    r = m.split_log.get((c, u), (float("nan"), float("nan")))
    subs = {m.names[k]: sorted(m.names[s] for s in v) for k, v in m.sub_members.items()}
    print(f"  [{m.name}] {tag:28s} split test on colour/MIX: ratio {r[0]:6.2f} vs threshold {r[1]:5.2f}   sub-categories: {subs}")
    return dict(tag=tag, ratio=r[0], thr=r[1], subs=subs)

out = {}
for name in ("NEWBORN", "EXPERIENCED"):
    m = Mind(name); background(m)
    if name == "EXPERIENCED":
        prior_life(m)
    episode(m)
    rows = [report(m, "after stage 10")]
    extra = [("yellow", "mixed"), ("grey", "mixed"), ("purple", "mixed"), ("ivory", "pure-light"), ("orange", "mixed"), ("jet", "pure-dark")]
    for col, kind in extra:
        m.perceive(ISA, col, "colour")
        for _ in range(4):
            if kind == "mixed":
                m.perceive(HAS, col, "dark", 1, "sense", event=True); m.perceive(HAS, col, "light", 1, "sense", event=True)
            elif kind == "pure-light":
                m.perceive(HAS, col, "light", 1, "sense", event=True); m.perceive(HAS, col, "dark", 0, "sense", event=True)
            else:
                m.perceive(HAS, col, "dark", 1, "sense", event=True); m.perceive(HAS, col, "light", 0, "sense", event=True)
        rows.append(report(m, f"+ {col} ({kind})"))
        if m.sub_members:
            g = {m.names[k[0]]: (round(v["a_rule"], 3), round(v["c_rule"], 3), round(v["k_mean"], 1)) for k, v in m.S["ctx"].items() if k[1] == m.ix["MIX(dark|light)"]}
            print(f"  [{m.name}]   rules on MIX after the split (mean, transfer confidence, coupling): {g}")
            rows[-1]["rules"] = g
    m.perceive(ISA, "teal", "colour")
    b = m.belief_view(m.ix["teal"], m.ix["MIX(dark|light)"])
    print(f"  [{m.name}] a colour never described ('teal') under MIX: e={b['e']:.3f} c={b['c']:.3f};  {m.say('teal', 'dark')}")
    out[name] = dict(rows=rows, teal=dict(e=float(b["e"]), c=float(b["c"])), allocations=m.alloc_log)
json.dump(out, open("split_continuation.json", "w"), indent=1, default=str)
