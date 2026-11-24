# -*- coding: utf-8 -*-
"""M2: what each part of the harmony adds (NOTES.md, 6.2).

    python m2_ablation.py [CORPUS ...] [--steps ABCDEF] [--out results/m2_ablation.jsonl]

The parts are added one at a time, in this order:

- A, the cortex alone;
- B, the episode with the verdict by the rule (MODEL.md, section 5);
- C, the harmony: the cortex, every context and the episode, each a belief per bit, combined by the gates and the
  calibration by the rule;
- D, the word contexts;
- E, a second input per origin and the coarser origin as prior;
- F, the short-term memory (the full M2).

Each row has the whole stream, the last tenth living and the last tenth frozen, as in m2_harmony.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import corpora  # noqa: E402
from living.fast import FastCortex  # noqa: E402

STEPS = [
    ("A cortex", dict(depth=8)),
    ("B + episode, verdict by the rule", dict(depth=8, episode=True, minlen=12)),
    ("C harmony", dict(depth=8, episode=True, minlen=12, mix=True, words=False, two_inputs=False, tree_prior=False,
                       recent=0)),
    ("D + word contexts", dict(depth=8, episode=True, minlen=12, mix=True, words=True, two_inputs=False,
                               tree_prior=False, recent=0)),
    ("E + second input, coarser prior", dict(depth=8, episode=True, minlen=12, mix=True, words=True, two_inputs=True,
                                             tree_prior=True, recent=0)),
    ("F + short-term memory (M2)", dict(depth=8, episode=True, minlen=12, mix=True, words=True, two_inputs=True,
                                        tree_prior=True, recent=6)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="*", default=["tr-lessons", "en-pydoc", "en-repo", "tr-private", "code-pystdlib"])
    ap.add_argument("--steps", default="ABCDEF", help="which steps to run, by letter")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m2_ablation.jsonl"))
    a = ap.parse_args()
    for name in a.corpora:
        s = corpora.stream(name)
        split = int(len(s) * 0.9)
        for label, cfg in STEPS:
            if label[0] not in a.steps:
                continue
            mind = FastCortex(**cfg)
            t0 = time.time()
            tr, _ = mind.stream(s[:split])
            fr, _ = mind.stream(s[split:], s[:split], learn=False)
            lv, _ = mind.stream(s[split:], s[:split])
            row = {"corpus": name, "step": label, "config": cfg,
                   "bpc_all": round((tr + lv) / len(s), 4),
                   "bpc_test_living": round(lv / (len(s) - split), 4),
                   "bpc_test_frozen": round(fr / (len(s) - split), 4),
                   "seconds": round(time.time() - t0, 1)}
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
