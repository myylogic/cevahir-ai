# -*- coding: utf-8 -*-
"""M2: the harmony living through a text (NOTES.md, section 6).

    python m2_harmony.py CORPUS [--config "dict(...)"] [--out results/m2_harmony.jsonl]

The same protocol as m1_cortex.py: one pass in order, each symbol predicted before it is learned, the stream split
at 90 %; the last tenth measured living (learning on) and frozen (the mind as it was at the split, learning nothing,
not even into its episodes or its short-term memory). The mind is living/fast.py's FastCortex: the cortex in C
(bit-identical to living/cortex.py), the episode, the line episode and line context, the word contexts, the
short-term memory and the harmony that combines them.

For the author's private corpus only counts are written.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import corpora  # noqa: E402
from living.fast import FastCortex  # noqa: E402
from living.lexicon import words  # noqa: E402

DEFAULT = dict(depth=8, episode=True, minlen=12, episode_bits=24, mix=True, lr=0.02, lr_min=0.002, words=True,
               two_inputs=True, tree_prior=True, recent=6, lines=True)
MEETINGS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 100), (101, 10 ** 12)]


def run(name: str, config: dict) -> dict:
    s = corpora.stream(name)
    split = int(len(s) * 0.9)
    mind = FastCortex(**config)
    t0 = time.time()
    a, bits_a = mind.stream(s[:split], bits=True)
    t_train = time.time() - t0
    f, bits_f = mind.stream(s[split:], s[:split], learn=False, bits=True)
    t1 = time.time()
    lv, bits_l = mind.stream(s[split:], s[:split], bits=True)
    t_test = time.time() - t1
    import numpy as np
    bits = np.concatenate([bits_a, bits_l])
    seen_train: set[str] = set()
    meet_count: dict[str, int] = {}
    meet = {k: [0.0, 0] for k in MEETINGS}
    test = {"seen": [0.0, 0.0, 0, 0], "unseen": [0.0, 0.0, 0, 0]}          # bits living, bits frozen, chars, words
    for x0, x1 in words(s):
        w = s[x0:x1]
        k = meet_count.get(w, 0) + 1
        meet_count[w] = k
        wb = float(bits[x0:x1].sum())
        for lo, hi in MEETINGS:
            if lo <= k <= hi:
                meet[(lo, hi)][0] += wb
                meet[(lo, hi)][1] += x1 - x0
                break
        if x1 <= split:
            seen_train.add(w)
        elif x0 >= split:
            kind = "seen" if w in seen_train else "unseen"
            test[kind][0] += wb
            test[kind][1] += float(bits_f[x0 - split:x1 - split].sum())
            test[kind][2] += x1 - x0
            test[kind][3] += 1
    out = {
        "corpus": name, "chars": len(s), "split": split, "config": config,
        "bpc_all": round((a + lv) / len(s), 4),
        "bpc_train_part": round(a / split, 4),
        "bpc_test_living": round(lv / (len(s) - split), 4),
        "bpc_test_frozen": round(f / (len(s) - split), 4),
        "store": mind.size(),
        "test_words": {k: {"bpc_living": round(v[0] / v[2], 4) if v[2] else None,
                           "bpc_frozen": round(v[1] / v[2], 4) if v[2] else None, "chars": v[2], "words": v[3]}
                       for k, v in test.items()},
        "word_meetings": [{"meeting": "%d" % lo if lo == hi else ("%d+" % lo if hi > 10 ** 9 else "%d-%d" % (lo, hi)),
                           "bpc": round(v[0] / v[1], 4) if v[1] else None, "chars": v[1]}
                          for (lo, hi), v in meet.items()],
        "seconds_train_part": round(t_train, 1), "seconds_test_part": round(t_test, 1),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }
    if name in corpora.PUBLIC:
        out["sha256"] = corpora.describe(name)["sha256"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--config", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m2_harmony.jsonl"))
    a = ap.parse_args()
    config = dict(DEFAULT)
    if a.config:
        config.update(eval(a.config, {"dict": dict}))
    row = run(a.corpus, config)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k not in ("word_meetings",)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
