#!/usr/bin/env python3
"""Worked trace of the owner's 'red' example with the equations of bg_core.py.

Two minds hear exactly the same colour episode from a stranger:
  NEWBORN      knows only that dark and light exclude each other (8 own observations).
  EXPERIENCED  has in addition lived three unrelated domains (instruments/pitch, fruits/taste,
               animals/size) in which every member showed both poles of a contrast.
Run:  python3 trace_red.py            (about 95 s on one core)
"""
import copy
import json
import sys
import time

import numpy as np

from bg_core import EXC, HAS, ISA, GRID, Mind

OUT = {}


def background(m):
    m.set_prior_trust("sense", 19.0, 0.0)             # own perception: trust 20/21 = 0.95
    for _ in range(8):
        m.perceive(EXC, "dark", "light", 1, "sense", event=True)


def prior_life(m):
    domains = [("instrument", ["flute", "violin", "drum", "piano"], ("high", "low")),
               ("fruit", ["apple", "grape", "plum", "cherry"], ("sweet", "sour")),
               ("animal", ["dog", "horse", "fish", "bird"], ("big", "small"))]
    for cat, members, (a, b) in domains:
        for _ in range(8):
            m.perceive(EXC, a, b, 1, "sense", event=True)
        for mem in members:
            m.perceive(ISA, mem, cat, 1, "teacher")
            for _ in range(2):
                m.perceive(HAS, mem, a, 1, "sense", event=True)
                m.perceive(HAS, mem, b, 1, "sense", event=True)


def fmt(b):
    f = "  -  " if np.isnan(b["f"]) else f"{b['f']:.2f}"
    return f"f={f} e={b['e']:.3f} c={b['c']:.3f}"


def rules(m, cat):
    rows = []
    for (c, p), g in m.S["ctx"].items():
        if m.names[c].split("*")[0] == cat:
            rows.append(dict(category=m.names[c], lens=m.names[p], a=g["a_rule"], c_rule=g["c_rule"],
                             E=g["E_rule"], E0=g["E0"], A0=g["A0"], k=g["k_mean"],
                             pi=[round(float(x), 3) for x in g["pi"]],
                             members={m.names[s]: dict(w=round(float(w), 3), conforms=round(float(r), 3), K=round(float(K), 3))
                                      for s, w, r, K in zip(g["members"], g["w"], g["inl"], g["K"])}))
    return rows


def show(m, title, subjects=("red",), cat="colour", ask=True):
    print(f"\n  [{m.name}] {title}")
    rec = dict(beliefs={}, rules=[], questions=[])
    for s in subjects:
        if s not in m.ix:
            continue
        for o in ("dark", "light"):
            b = m.answer(HAS, m.ix[s], m.ix[o])
            rec["beliefs"][f"{s}/{o}"] = {k: float(b[k]) for k in ("f", "e", "c")}
            print(f"     {s:6s} is {o:5s}: {fmt(b)}    says: {m.say(s, o)}")
        if "MIX(dark|light)" in m.ix:
            b = m.belief_view(m.ix[s], m.ix["MIX(dark|light)"])
            rec["beliefs"][f"{s}/MIX"] = {k: float(b[k]) for k in ("f", "e", "c", "c_own", "w", "K")}
            print(f"     {s:6s} 'sometimes dark, sometimes light': own evidence w={b['w']:.3f} c_own={b['c_own']:.3f}"
                  f" | fed back K={b['K']:.3f} -> c={b['c']:.3f} e={b['e']:.3f}")
    for r in rules(m, cat):
        rec["rules"].append(r)
        print(f"     RULE  {r['category']} -> {r['lens']}: mean a={r['a']:.3f}  evidence E={r['E']:.3f} (schema prior E0={r['E0']:.3f},"
              f" A0={r['A0']:.2f})  coupling k~{r['k']:.1f}  transfer confidence c_rule={r['c_rule']:.3f}")
        print(f"           members {r['members']}")
    rec["realism_v"] = {k: float(v) for k, v in m.S["v"].items()}
    rec["trust"] = {q: m.trust(q) for q in m.src}
    print(f"     measured realism of own projections v = {rec['realism_v']}   trust = "
          + ", ".join(f"{q}:{v:.3f}" for q, v in rec["trust"].items()))
    if ask and cat in m.ix:
        qs, pu = m.questions(cat, delta=m.trust("stranger"))
        rec["p_unseen"] = float(pu)
        for q in qs[:3]:
            rec["questions"].append(dict(gain=float(q["gain"]), text=q["text"], uttered=bool(q["uttered"])))
            print(f"     question candidate  gain={q['gain']:.4f}  {'ARISES (>= 0.03): ' if q['uttered'] else '                   '}{q['text']}")
        print(f"     estimated share of unseen members of '{cat}': {pu:.3f}")
    OUT.setdefault(m.name, {})[title] = rec
    return rec


def tell(minds, kind, s, o, y=1.0, q="stranger", event=False, quiet=False):
    for m in minds:
        r = m.perceive(kind, s, o, y, q, event=event)
        if quiet:
            for a in r["allocated"]:
                print(f"  [{m.name}]   ALLOCATED: {a}")
            continue
        p = r["pred"]
        print(f"  [{m.name}] hears ({['ISA','HAS','EXC'][kind]} {s} {o} y={y:g}) from {q}: predicted e={p['e']:.3f} c={p['c']:.3f}"
              f" surprise={r['surprise']:.3f} nats")
        for a in r["allocated"]:
            print(f"  [{m.name}]   ALLOCATED: {a}")
        for im in r["imagined"]:
            print(f"  [{m.name}]   IMAGINED (kept out of the ledger): {im['s']} under {im['p']}: e={im['e']:.3f} c={im['c']:.3f}"
                  f" predicted realism R={im['R']:.3f}")
        for rs in r["resolved"]:
            print(f"  [{m.name}]   RESOLVED imagination about {rs['s']}: imagined e={rs['e_imagined']:.3f}, world f={rs['f_world']:.2f},"
                  f" squared error={rs['sq_err']:.3f}, realism after the fact r={rs['r_post']:.3f} (predicted R={rs['R_predicted']:.3f})")


def main():
    t0 = time.time()
    newborn, exper = Mind("NEWBORN"), Mind("EXPERIENCED")
    for m in (newborn, exper):
        background(m)
    prior_life(exper)
    minds = (newborn, exper)
    print("=" * 110)
    print("STAGE 0  before the episode")
    for m in minds:
        print(f"  [{m.name}] ledger lines={len(m.ledger)} units={len(m.names)} exclusion(dark,light)={m.S['X'][m.ix['dark'], m.ix['light']]:.3f}"
              f"  realism v={ {k: round(v, 3) for k, v in m.S['v'].items()} }  carry-over coupling K_S~{m.S['ks_mean']:.1f}")
    for (c, p), g in exper.S["ctx"].items():
        print(f"  [EXPERIENCED] lived rule {exper.names[c]} -> {exper.names[p]}: a={g['a_rule']:.3f} c_rule={g['c_rule']:.3f} k~{g['k_mean']:.1f}"
              f" pi={np.round(g['pi'], 2)}")
    OUT["EXPERIENCED"] = {"journal_prior_life": copy.deepcopy(exper.journal)}
    print(f"  [EXPERIENCED] imaginations resolved during the prior life: {len(exper.journal)};"
          f" mean squared error {np.mean([j['sq_err'] for j in exper.journal]):.4f}")

    print("=" * 110)
    print("STAGE 1  'red is a dark colour'  (first hearing, unknown source)")
    tell(minds, ISA, "red", "colour")
    tell(minds, HAS, "red", "dark")
    snap = {}
    for m in minds:
        show(m, "stage1")
        snap[m.name] = (m.answer(HAS, m.ix["red"], m.ix["dark"]), m.S["Wp"].copy(), m.S["Wn"].copy())

    print("=" * 110)
    print("STAGE 2  'red also has light shades'  (refine, do not delete)")
    tell(minds, HAS, "red", "light")
    ids = {m.name: len(m.ledger) - 1 for m in minds}
    for m in minds:
        show(m, "stage2-5")
        lines = [e for e in m.ledger if m.names[e["s"]] == "red"]
        print(f"     ledger lines about red still present: {[(['ISA','HAS','EXC'][e['kind']], m.names[e['o']], e['y']) for e in lines]}")

    print("=" * 110)
    print("CHECK  retraction of the second statement (on copies)")
    for m in minds:
        n0 = snap[m.name][1].shape[1]
        c = copy.deepcopy(m)
        c.retract_evidence(ids[m.name])
        b = c.answer(HAS, c.ix["red"], c.ix["dark"])
        dW = max(np.abs(c.S["Wp"][:, :n0, :n0] - snap[m.name][1]).max(), np.abs(c.S["Wn"][:, :n0, :n0] - snap[m.name][2]).max())
        dB = abs(b["e"] - snap[m.name][0]["e"]) + abs(b["c"] - snap[m.name][0]["c"])
        print(f"  [{m.name}] O(1) evidence retraction : red is dark {fmt(b)}; stage-1 value {fmt(snap[m.name][0])};"
              f" max |evidence difference| = {dW:.1e}; |belief difference| = {dB:.1e}"
              f" (the allocated lens stays: units {len(c.names)} vs {n0})")
        r = m.replay(skip={ids[m.name]})
        b2 = r.answer(HAS, r.ix["red"], r.ix["dark"])
        dW2 = max(np.abs(r.S["Wp"] - snap[m.name][1]).max(), np.abs(r.S["Wn"] - snap[m.name][2]).max())
        dB2 = abs(b2["e"] - snap[m.name][0]["e"]) + abs(b2["c"] - snap[m.name][0]["c"])
        print(f"  [{m.name}] replay without the line  : red is dark {fmt(b2)}; max |evidence difference| = {dW2:.1e};"
              f" |belief difference| = {dB2:.1e}; units {len(r.names)} vs {n0}")
        OUT[m.name]["retraction"] = dict(evidence_diff_o1=float(dW), belief_diff_o1=float(dB),
                                         evidence_diff_replay=float(dW2), belief_diff_replay=float(dB2))

    print("=" * 110)
    print("SENSITIVITY of the question ranking after stage 2 (gain of 'which other colours?' / gain of the best other question)")
    for m in minds:
        row = []
        for gamma in (0.5, 1.0, 2.0):
            for phi in (0.25, 0.5, 1.0):
                qs, pu = m.questions("colour", delta=m.trust("stranger"), gamma=gamma, phi=phi)
                ext = [q["gain"] for q in qs if q["text"].startswith("which")][0]
                oth = max(q["gain"] for q in qs if not q["text"].startswith("which"))
                row.append((gamma, phi, ext / oth))
        print(f"  [{m.name}] " + "  ".join(f"g={g:g},phi={p:g}: {r:.2f}" for g, p, r in row))
        OUT[m.name]["question_sensitivity"] = [dict(gamma=g, phi=p, ratio=float(r)) for g, p, r in row]

    print("=" * 110)
    print("STAGE 6  the question is answered: 'blue is a colour', 'green is a colour'  (the mind expands over the set)")
    tell(minds, ISA, "blue", "colour")
    tell(minds, ISA, "green", "colour")
    for m in minds:
        show(m, "stage6", subjects=("red", "blue"))

    print("=" * 110)
    print("STAGE 7  the world speaks about blue and green (both shades each)")
    for s, o in (("blue", "dark"), ("blue", "light"), ("green", "light"), ("green", "dark")):
        tell(minds, HAS, s, o)
    for m in minds:
        show(m, "stage7", subjects=("red", "blue"))

    print("=" * 110)
    print("STAGE 8  a wrong construction: 'black is a colour' -> imagined to have light shades -> the world refutes it")
    tell(minds, ISA, "black", "colour")
    for m in minds:
        print(f"  [{m.name}] before any evidence about black: {m.say('black', 'light')}")
    for i in range(6):
        tell(minds, HAS, "black", "dark", q="sense", event=True, quiet=True)
        tell(minds, HAS, "black", "light", y=0.0, q="sense", event=True, quiet=(i not in (1, 5)))
    for m in minds:
        show(m, "stage8", subjects=("red", "black"), ask=False)

    print("=" * 110)
    print("STAGE 9  a second exception: white -> does a new category emerge from the instances?")
    tell(minds, ISA, "white", "colour")
    for i in range(6):
        tell(minds, HAS, "white", "light", q="sense", event=True, quiet=True)
        tell(minds, HAS, "white", "dark", y=0.0, q="sense", event=True, quiet=(i not in (1, 5)))
    for m in minds:
        show(m, "stage9", subjects=("red", "black", "white"), ask=False)
    for m in minds:
        for (c, p), (ratio, thr) in m.split_log.items():
            if m.names[p].startswith("MIX(dark"):
                print(f"  [{m.name}] split test on {m.names[c]} / {m.names[p]}: log evidence ratio {ratio:.2f} against threshold {thr:.2f}")
                OUT[m.name]["split_after_stage9"] = dict(ratio=float(ratio), thr=float(thr), formed=bool(m.sub))

    print("=" * 110)
    print("STAGE 10  the mind looks at red, blue and green itself (4 dark and 4 light observations each)")
    for s_ in ("red", "blue", "green"):
        for i in range(4):
            tell(minds, HAS, s_, "dark", q="sense", event=True, quiet=True)
            tell(minds, HAS, s_, "light", q="sense", event=True, quiet=True)
    for m in minds:
        for (c, p), (ratio, thr) in m.split_log.items():
            if m.names[p].startswith("MIX(dark"):
                print(f"  [{m.name}] split test on {m.names[c]} / {m.names[p]}: log evidence ratio {ratio:.2f} against threshold {thr:.2f}")
                OUT[m.name]["split_after_stage10"] = dict(ratio=float(ratio), thr=float(thr), formed=bool(m.sub))
        show(m, "stage10", subjects=("red", "black", "white"), ask=False)
    tell(minds, ISA, "grey", "colour")
    for m in minds:
        print(f"  [{m.name}] about grey, never described: {m.say('grey', 'dark')}")
        b = m.belief_view(m.ix["grey"], m.ix["MIX(dark|light)"])
        OUT[m.name]["grey_mix"] = {k: float(b[k]) for k in ("e", "c", "K")}
        print(f"  [{m.name}] grey 'sometimes dark, sometimes light': e={b['e']:.3f} c={b['c']:.3f}")

    print("=" * 110)
    print("IMAGINATION JOURNAL of the colour episode (what was constructed, how real it turned out)")
    for m in minds:
        js = [j for j in m.journal if j["s"] in ("blue", "green", "black", "white", "grey")]
        for j in js:
            print(f"  [{m.name}] {j['s']:6s} {j['p']}: imagined e={j['e_imagined']:.3f} (c={j['c_imagined']:.3f}, predicted realism"
                  f" {j['R_predicted']:.3f}) -> world f={j['f_world']:.2f}; squared error {j['sq_err']:.3f}; realism after {j['r_post']:.3f}")
        OUT[m.name]["journal_colour"] = js
        print(f"  [{m.name}] ledger lines={len(m.ledger)} units={len(m.names)}; imagination never wrote a ledger line;"
              f" allocations: {m.alloc_log}")

    print("=" * 110)
    print("SOURCES  trust is learned only from evidence that did not come from the source itself")
    m = Mind("SOURCES")
    background(m)
    for _ in range(4):
        m.perceive(HAS, "coal", "dark", 1, "sense", event=True)
        m.perceive(HAS, "snow", "light", 1, "sense", event=True)
    for _ in range(3):
        m.perceive(HAS, "coal", "dark", 1, "friend")
        m.perceive(HAS, "snow", "light", 1, "friend")
        m.perceive(HAS, "coal", "light", 1, "liar")
        m.perceive(HAS, "snow", "dark", 1, "liar")
    print(f"  trust after 6 statements each: friend={m.trust('friend'):.3f}  liar={m.trust('liar'):.3f}  (both started at 0.500)")
    print(f"  coal is dark: {fmt(m.answer(HAS, m.ix['coal'], m.ix['dark']))}   coal is light: {fmt(m.answer(HAS, m.ix['coal'], m.ix['light']))}")
    OUT["sources"] = dict(friend=m.trust("friend"), liar=m.trust("liar"))
    print(f"\nwall time {time.time() - t0:.1f} s")
    json.dump(OUT, open("trace_red.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    sys.exit(main())
