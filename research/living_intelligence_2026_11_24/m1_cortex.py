# -*- coding: utf-8 -*-
"""M1: the cortex living through a text (NOTES.md, section 2).

    python m1_cortex.py CORPUS [--depth 8] [--grow 1] [--escape situation] [--mixing exclude] [--update all]
                               [--out results/m1_cortex.jsonl]

One pass, in order. Each symbol is predicted before it is learned, so the bits of the whole stream are a prequential
code length. The stream is split at 90 % of its characters:

- ``living`` test bits: the store keeps learning through the test part;
- ``frozen`` test bits: the store as it was at the split, not learning; the setting of a model trained and then
  evaluated.

Also measured:

- the growth of the store along the way (nodes, ties, lexicon) and the exponent of its growth;
- the bits of words seen in the first 90 % against words never seen there;
- the bits of a word at its first, second, third ... meeting in the stream: learning a word while living.

For the author's private corpus only counts are written; no word, number or example of it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import corpora  # noqa: E402
from living.cortex import Cortex  # noqa: E402
from living.lexicon import Lexicon, words  # noqa: E402

MEETINGS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 100), (101, 10 ** 12)]


def slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of log y on log x."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def run(name: str, depth: int, grow: int, escape: str, mixing: str, update: str = "all") -> dict:
    s = corpora.stream(name)
    private = name not in corpora.PUBLIC
    split = int(len(s) * 0.9)
    cx = Cortex(depth=depth, grow=grow, escape=escape, mixing=mixing, update=update)
    bits: list[float] = []
    growth = []
    checkpoints = set()
    c = 1024
    while c < split:
        checkpoints.add(c)
        c *= 2
    checkpoints.add(split)
    t0 = time.time()
    h = ""
    pos = 0
    ordered = sorted(checkpoints)
    for c in ordered:                                   # live up to each checkpoint, then look at the store
        _, h = cx.stream(s[pos:c], h, bits=bits)
        pos = c
        sz = cx.size()
        growth.append({"chars": c, "nodes": sz["nodes"], "pending": sz["pending"], "entries": sz["entries"]})
    t_train = time.time() - t0
    frozen_bits: list[float] = []
    cx.stream(s[split:], h, learn=False, bits=frozen_bits)         # the store at the split, not learning
    t1 = time.time()
    _, h = cx.stream(s[split:], h, bits=bits)                       # the same text, living
    t_test = time.time() - t1
    sz = cx.size()
    growth.append({"chars": len(s), "nodes": sz["nodes"], "pending": sz["pending"], "entries": sz["entries"]})

    # the lexicon over the same stream, and words by their meetings
    lex = Lexicon()
    lex_growth = []
    meet_count: dict[str, int] = {}
    meet_bits = {k: [0.0, 0] for k in MEETINGS}
    seen_train: set[str] = set()
    test_words = {"seen": [0.0, 0, 0], "unseen": [0.0, 0, 0]}            # bits, chars, words (living)
    test_words_frozen = {"seen": [0.0, 0], "unseen": [0.0, 0]}
    next_cp = iter(ordered + [len(s)])
    cp = next(next_cp)
    for a, b in words(s):
        while b > cp:
            lex_growth.append({"chars": cp, **lex.size()})
            cp = next(next_cp, 10 ** 15)
        w = s[a:b]
        lex.live(w)
        k = meet_count.get(w, 0) + 1
        meet_count[w] = k
        wb = sum(bits[a:b])
        for lo, hi in MEETINGS:
            if lo <= k <= hi:
                meet_bits[(lo, hi)][0] += wb
                meet_bits[(lo, hi)][1] += b - a
                break
        if b <= split:
            seen_train.add(w)
        elif a >= split:
            kind = "seen" if w in seen_train else "unseen"
            test_words[kind][0] += wb
            test_words[kind][1] += b - a
            test_words[kind][2] += 1
            test_words_frozen[kind][0] += sum(frozen_bits[a - split:b - split])
            test_words_frozen[kind][1] += b - a
    lex_growth.append({"chars": len(s), **lex.size()})

    half = [g for g in growth if g["chars"] >= len(s) // 16]
    lhalf = [g for g in lex_growth if g["chars"] >= len(s) // 16 and g["nodes"] > 0]
    out = {
        "corpus": name, "chars": len(s), "split": split,
        "depth": depth, "grow": grow, "escape": escape, "mixing": mixing, "update": update,
        "bpc_all": round(sum(bits) / len(s), 4),
        "bpc_train_part": round(sum(bits[:split]) / split, 4),
        "bpc_test_living": round(sum(bits[split:]) / (len(s) - split), 4),
        "bpc_test_frozen": round(sum(frozen_bits) / (len(s) - split), 4),
        "store": sz,
        "growth": growth,
        "growth_exponent_nodes": round(slope([g["chars"] for g in half], [g["nodes"] for g in half]), 3),
        "growth_exponent_entries": round(slope([g["chars"] for g in half], [g["entries"] for g in half]), 3),
        "lexicon": lex.size(),
        "lexicon_growth": lex_growth,
        "growth_exponent_lexicon_nodes": round(slope([g["chars"] for g in lhalf], [g["nodes"] for g in lhalf]), 3),
        "growth_exponent_words": round(slope([g["chars"] for g in lhalf], [g["words"] for g in lhalf]), 3),
        "test_words": {k: {"bpc_living": round(v[0] / v[1], 4) if v[1] else None,
                           "bpc_frozen": round(test_words_frozen[k][0] / test_words_frozen[k][1], 4) if v[1] else None,
                           "chars": v[1], "words": v[2]} for k, v in test_words.items()},
        "word_meetings": [{"meeting": "%d" % lo if lo == hi else ("%d+" % lo if hi > 10 ** 9 else "%d-%d" % (lo, hi)),
                           "bpc": round(v[0] / v[1], 4) if v[1] else None, "chars": v[1]}
                          for (lo, hi), v in meet_bits.items()],
        "seconds_train_part": round(t_train, 1), "seconds_test_part": round(t_test, 1),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }
    if not private:
        examples = []
        for w in ("the", "evidence", "mind", "zihin", "def", "ve", "bir", "kanıt"):
            n = lex.number(w)
            if n is not None:
                examples.append({"word": w, "number": n, "spelled_back": lex.spell(n),
                                 "word_belief": round(lex.word_belief(n), 4)})
        out["lexicon_examples"] = examples
        out["sha256"] = corpora.describe(name)["sha256"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--grow", type=int, default=1)
    ap.add_argument("--escape", default="situation")
    ap.add_argument("--mixing", default="exclude")
    ap.add_argument("--update", default="all", choices=("all", "found"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m1_cortex.jsonl"))
    a = ap.parse_args()
    row = run(a.corpus, a.depth, a.grow, a.escape, a.mixing, a.update)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k not in ("growth", "lexicon_growth")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
