#!/usr/bin/env python3
"""Internal pilot (NOT the decisive experiment: no Transformer baseline here).

Question: does the learned operating process transfer across subjects in prediction loss?
World: D domains in sequence. Each domain has its own category, 4 members and its own contrast pair.
  'mixed' world: every member shows either pole with probability 1/2 at each observation.
  'pure'  world: every member shows one pole only (which one is drawn per member).
Per member: one membership statement, then 4 observation events. Before each observation the mind
predicts which pole will be seen; the loss is -log p(observed pole).
Ideal observer: mixed = log 2 = 0.693 at every observation; pure = 0.693 at the first observation of a
member and ~0 afterwards.
Variants: FULL (everything learned) / FROZEN (operating process frozen: no pooling across domains,
realism fixed at 0.5) / FIXED-K1 (one global coupling k=1, the NARS-like setting, frozen) /
FLAT (no categories at all: counts + exclusion).
FULL+M / FROZEN+M: as FULL / FROZEN, but a stated membership counts as certain.
Run: python3 pilot_transfer.py   (about 8 minutes on 3 cores)
"""
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

from bg_core import EXC, HAS, ISA, Mind

D, M, NOBS, SEEDS = 5, 4, 4, 5
VARIANTS = {"FULL": {}, "FROZEN": dict(learn_process=False), "FIXED-K1": dict(learn_process=False, fixed_k=1.0),
            "FLAT": dict(learn_process=False),
            # the same two with memberships taken as certain (removes the 'told once' bottleneck m = 0.6)
            "FULL+M": dict(certain_membership=True), "FROZEN+M": dict(learn_process=False, certain_membership=True)}


def run(args):
    variant, world, seed = args
    rng = np.random.default_rng(1000 + seed)
    m = Mind(variant, **VARIANTS[variant])
    m.set_prior_trust("sense", 19.0, 0.0)
    rows = []
    for d in range(D):
        a, b, cat = f"a{d}", f"b{d}", f"cat{d}"
        for _ in range(8):
            m.perceive(EXC, a, b, 1, "sense", event=True)
        for i in range(M):
            s = f"m{d}_{i}"
            if variant != "FLAT":
                m.perceive(ISA, s, cat, 1, "teacher")
            theta = 0.5 if world == "mixed" else float(rng.random() < 0.5)
            for t in range(NOBS):
                pole = a if rng.random() < theta else b
                m.unit(s)
                if m.S is None or len(m.names) != m.S["Wp"].shape[1]:
                    m.solve()
                ea = m.answer(HAS, m.ix[s], m.ix[a])["e"]
                eb = m.answer(HAS, m.ix[s], m.ix[b])["e"]
                p = (ea if pole == a else eb) / (ea + eb)
                rows.append((d, i, t, -math.log(max(p, 1e-9))))
                m.perceive(HAS, s, pole, 1, "sense", event=True)
    return variant, world, seed, rows


def main():
    jobs = [(v, w, s) for v in VARIANTS for w in ("mixed", "pure") for s in range(SEEDS)]
    with Pool(3) as pool:
        res = pool.map(run, jobs)
    out = {}
    for world in ("mixed", "pure"):
        print(f"\nWORLD {world}: mean loss (nats) on observations 2-{NOBS} of every member, by domain index"
              f"   [ideal: {'0.693' if world == 'mixed' else '~0'}]")
        print(f"{'variant':10s} " + " ".join(f"dom{d + 1:>2d}" for d in range(D)) + "   | first member of domains 2..D, obs 2-4 | obs 1 (all)")
        for v in VARIANTS:
            rows = [(seed, r) for (vv, ww, seed, rr) in res if vv == v and ww == world for r in rr]
            by_d = []
            for d in range(D):
                per_seed = [np.mean([r[3] for sd, r in rows if sd == s and r[0] == d and r[2] >= 1]) for s in range(SEEDS)]
                by_d.append((float(np.mean(per_seed)), float(np.std(per_seed, ddof=1) / math.sqrt(SEEDS))))
            first = [np.mean([r[3] for sd, r in rows if sd == s and r[0] >= 1 and r[1] == 0 and r[2] >= 1]) for s in range(SEEDS)]
            obs1 = [np.mean([r[3] for sd, r in rows if sd == s and r[2] == 0]) for s in range(SEEDS)]
            out[f"{world}/{v}"] = dict(by_domain=by_d, first_member_later_domains=(float(np.mean(first)), float(np.std(first, ddof=1) / math.sqrt(SEEDS))),
                                       obs1=float(np.mean(obs1)))
            print(f"{v:10s} " + " ".join(f"{x:5.3f}" for x, _ in by_d) + f"   | {np.mean(first):.3f} +- {np.std(first, ddof=1) / math.sqrt(SEEDS):.3f}"
                  f"                    | {np.mean(obs1):.3f}")
    json.dump(out, open("pilot_transfer.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
