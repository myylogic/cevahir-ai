#!/usr/bin/env python3
"""Dose-response of the red episode: how do the stage-2 readouts depend on how much was lived before?

D = number of unrelated domains lived before the colour episode (each: one category, four members,
one contrast pair, every member observed twice on each pole). Also: the same with the carry-over
coupling K_S fixed (ablation), and with the operating process frozen.
Run: python3 dose_response.py   (about 6 minutes)
"""
import json
import numpy as np
from bg_core import EXC, HAS, ISA, Mind

DOMAINS = [("instrument", ["flute", "violin", "drum", "piano"], ("high", "low")),
           ("fruit", ["apple", "grape", "plum", "cherry"], ("sweet", "sour")),
           ("animal", ["dog", "horse", "fish", "bird"], ("big", "small")),
           ("room", ["hall", "attic", "cellar", "kitchen"], ("warm", "cold")),
           ("road", ["lane", "street", "avenue", "track"], ("wide", "narrow")),
           ("river", ["brook", "creek", "stream", "canal"], ("fast", "slow"))]


def run(D, **kw):
    m = Mind(f"D{D}", **kw)
    m.set_prior_trust("sense", 19.0, 0.0)
    for _ in range(8):
        m.perceive(EXC, "dark", "light", 1, "sense", event=True)
    for cat, members, (a, b) in DOMAINS[:D]:
        for _ in range(8):
            m.perceive(EXC, a, b, 1, "sense", event=True)
        for mem in members:
            m.perceive(ISA, mem, cat, 1, "teacher")
            for _ in range(2):
                m.perceive(HAS, mem, a, 1, "sense", event=True)
                m.perceive(HAS, mem, b, 1, "sense", event=True)
    m.perceive(ISA, "red", "colour")
    r1 = m.perceive(HAS, "red", "dark")
    b1 = m.answer(HAS, m.ix["red"], m.ix["light"])
    r2 = m.perceive(HAS, "red", "light")
    u = m.ix["MIX(dark|light)"]
    b = m.belief_view(m.ix["red"], u)
    g = m.S["ctx"][(m.ix["colour"], u)]
    qs, pu = m.questions("colour", delta=m.trust("stranger"))
    ext = [q["gain"] for q in qs if q["text"].startswith("which")][0]
    oth = max(q["gain"] for q in qs if not q["text"].startswith("which"))
    return dict(D=D, f_light_before=float(b1["f"]) if not np.isnan(b1["f"]) else 0.0, surprise2=float(r2["surprise"]),
                c_red_own=float(b["c_own"]), c_red=float(b["c"]), e_red=float(b["e"]), a_rule=g["a_rule"],
                c_rule=g["c_rule"], E0=g["E0"], v=float(m.S["v"]["MIX"]), ks=m.S["ks_mean"], q_ext=float(ext), q_other=float(oth))


def main():
    out = {}
    hdr = f"{'variant':24s} {'D':>2s} | {'red c own':>9s} {'rule c':>7s} {'red c fed':>9s} | {'rule a':>6s} {'red e':>6s} | {'E0':>5s} {'K_S':>5s} {'v':>5s} | {'surprise@2':>10s} | {'gain: other colours?':>20s} {'more about red':>14s}"
    print(hdr)
    for name, kw, Ds in (("learned (full)", {}, (0, 1, 2, 3, 6)),
                         ("K_S fixed at 4", dict(k_schema=4.0), (3, 6)),
                         ("K_S fixed at 32", dict(k_schema=32.0), (3, 6)),
                         ("process frozen", dict(learn_process=False), (3,))):
        for D in Ds:
            r = run(D, **kw)
            out[f"{name}/D{D}"] = r
            print(f"{name:24s} {D:2d} | {r['c_red_own']:9.3f} {r['c_rule']:7.3f} {r['c_red']:9.3f} | {r['a_rule']:6.3f} {r['e_red']:6.3f} |"
                  f" {r['E0']:5.2f} {r['ks']:5.1f} {r['v']:5.3f} | {r['surprise2']:10.3f} | {r['q_ext']:20.4f} {r['q_other']:14.4f}")
    json.dump(out, open("dose_response.json", "w"), indent=1)


if __name__ == "__main__":
    main()
