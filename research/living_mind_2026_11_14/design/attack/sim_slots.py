#!/usr/bin/env python3
"""sim_slots.py -- LEAD_DESIGN.md §3 "Slots (frames): two composed units of equal length that differ in exactly one
position give a slot-bearing unit (frame) with a variable there; the fillers seen in that position get ABSTRACT
ties to a new category unit for the slot."   + §1 "slot-bearing = a composed unit some of whose positions are
variables" (so frames are composed units and the rule applies to them again).

Toy grammar with a KNOWN number of categories (random vocabulary, nothing domain specific):
    3 noun classes A, B, C (20 words each), 2 verb classes (6 words each), 1 class of 5 link words
    m = 3 :  (A, V1, B) | (B, V2, C)
    m = 5 :  (A, V1, B, L, C) | (B, V2, C, L, A)
True categories: 6 word classes.  Sentences are sampled with Zipfian word frequencies.

Counted, for the rule exactly as written (one frame per distinct tuple-with-a-hole that has >= 2 fillers, one NEW
category per frame, applied again to frames):
    composed units, frames per order (number of variables), categories, ABSTRACT ties, categories per word,
    naive pair comparisons (the rule is stated over PAIRS of composed units),
and for the fix (a slot's filler set is first matched against existing categories with the vigilance the design
already has; a new category is recruited only when no category matches; categories whose members overlap merge).
"""
import itertools, json, sys
from collections import defaultdict
import numpy as np


def make_world(m, rng):
    cls = {"A": ["a%d" % i for i in range(20)], "B": ["b%d" % i for i in range(20)], "C": ["c%d" % i for i in range(20)],
           "V1": ["v%d" % i for i in range(6)], "V2": ["w%d" % i for i in range(6)], "L": ["l%d" % i for i in range(5)]}
    pats = [("A", "V1", "B"), ("B", "V2", "C")] if m == 3 else [("A", "V1", "B", "L", "C"), ("B", "V2", "C", "L", "A")]
    def zipf(k):
        p = 1.0 / np.arange(1, k + 1); return p / p.sum()
    def sample():
        pat = pats[rng.integers(len(pats))]
        return tuple(cls[c][rng.choice(len(cls[c]), p=zipf(len(cls[c])))] for c in pat)
    word2cls = {w: c for c, ws in cls.items() for w in ws}
    return sample, word2cls


HOLE = "_"


def literal(sentences, m, max_order):
    """frames = distinct masked signatures with >= 2 distinct filler tuples; order = number of holes"""
    comp = set(sentences)
    stats = {}
    fillers_of = {}
    for order in range(1, max_order + 1):
        sig = defaultdict(set)
        for s in comp:
            for holes in itertools.combinations(range(m), order):
                key = tuple(HOLE if i in holes else s[i] for i in range(m))
                sig[key].add(tuple(s[i] for i in holes))
        frames = {k: v for k, v in sig.items() if len(v) >= 2}
        stats[order] = len(frames)
        if order == 1:
            fillers_of = {k: {f[0] for f in v} for k, v in frames.items()}
    return comp, stats, fillers_of


def fixed_categories(fillers_of, rho=0.5):
    """vigilance matching of a slot's filler set against existing categories; overlapping categories merge"""
    cats = []          # list of sets
    for key in sorted(fillers_of):          # deterministic order
        F = fillers_of[key]
        best = None; bo = 0.0
        for i, c in enumerate(cats):
            o = len(F & c) / len(F)
            if o > bo:
                best, bo = i, o
        if best is not None and bo >= rho:
            cats[best] |= F
        else:
            cats.append(set(F))
        merged = True
        while merged:
            merged = False
            for i in range(len(cats)):
                for j in range(i + 1, len(cats)):
                    if len(cats[i] & cats[j]) / min(len(cats[i]), len(cats[j])) >= rho:
                        cats[i] |= cats[j]; del cats[j]; merged = True; break
                if merged:
                    break
    return cats


def main():
    out = {}
    for m in (3, 5):
        rng = np.random.default_rng(0)
        sample, word2cls = make_world(m, rng)
        stream = [sample() for _ in range(30000)]
        print(f"\n== tuples of length m = {m} (true word classes: 6) ==")
        print("  sentences  composed  frames by number of holes (1,2,..)        categories(=1st-order frames)  ABSTRACT ties  "
              "max categories of one word  pair comparisons if done as written  | fix: categories (purity)")
        for n in (300, 1000, 3000, 10000, 30000):
            comp, stats, fillers_of = literal(stream[:n], m, m - 1)
            ncat = len(fillers_of)
            abst = sum(len(v) for v in fillers_of.values())
            per_word = defaultdict(int)
            for k, v in fillers_of.items():
                for w in v:
                    per_word[w] += 1
            pairs = len(comp) * (len(comp) - 1) // 2
            cats = fixed_categories(fillers_of)
            pur = np.mean([max(np.bincount([list(set(word2cls.values())).index(word2cls[w]) for w in c])) / len(c) for c in cats])
            tot_frames = sum(stats.values())
            print(f"  {n:>8}  {len(comp):>8}  {str([stats[o] for o in sorted(stats)]):<42} {ncat:>10} {abst:>22} "
                  f"{max(per_word.values()) if per_word else 0:>18} {pairs:>30,}  | {len(cats)} ({pur:.2f})")
            out[f"m{m}_n{n}"] = dict(composed=len(comp), frames=stats, categories=ncat, abstract_ties=abst,
                                     max_cat_per_word=max(per_word.values()) if per_word else 0, pairs=pairs,
                                     fix_categories=len(cats), fix_purity=float(pur), total_frames=tot_frames)
    print("\n upper bound of the rule as written: every composed unit of length m yields 2^m - 2 tuples-with-holes;"
          " m = 7 (the focus size) -> 126 per composed unit")
    json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
