# -*- coding: utf-8 -*-
"""M2: going on with a text never seen (NOTES.md, 6.7).

    python m2_say.py CORPUS [--out results/m2_say.jsonl]

The mind lives the first 90 % of the stream. Then, frozen (it learns nothing more), it is given 500 places in the
last tenth (corpora.say_positions). At each place it gets the 256 characters before it and goes on, up to 40
characters. Two ways:

- **free**: it always says its most probable symbol. Measured: how many characters it gets right before its first
  mistake;
- **speaking line**: it says a symbol only while its probability reaches e* ~ 0.583, and is silent from the first
  one that does not. Measured: how many characters it says, and how many of them are right.

m1_transformer.py --say 500 runs the same places on the Transformer trained offline on the same 90 %.
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
import m2_harmony  # noqa: E402
from living.evidence import SPEAKING_LINE  # noqa: E402
from living.fast import FastCortex  # noqa: E402


def score(pairs: list[tuple[str, str]], said_line: list[tuple[str, str]]) -> dict:
    """pairs: (free continuation, truth); said_line: (what was said under the line, truth)."""
    run = []
    for out, truth in pairs:
        k = 0
        while k < len(out) and k < len(truth) and out[k] == truth[k]:
            k += 1
        run.append(k)
    said = sum(len(o) for o, _ in said_line)
    right = 0
    for o, truth in said_line:
        k = 0
        while k < len(o) and o[k] == truth[k]:
            k += 1
        right += k
    return {"places": len(pairs),
            "free_right_before_first_mistake": round(sum(run) / len(run), 3),
            "free_ten_or_more_right": round(sum(1 for r in run if r >= 10) / len(run), 4),
            "free_all_40_right": round(sum(1 for r in run if r >= 40) / len(run), 4),
            "line_said_per_place": round(said / len(said_line), 3),
            "line_right_per_place": round(right / len(said_line), 3),
            "line_precision": round(right / said, 4) if said else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m2_say.jsonl"))
    a = ap.parse_args()
    s = corpora.stream(a.corpus)
    split = int(len(s) * 0.9)
    mind = FastCortex(**m2_harmony.DEFAULT)
    mind.stream(s[:split])
    places = corpora.say_positions(s, split)
    t0 = time.time()
    free, line = [], []
    examples = []
    for p in places:
        truth = s[p:p + 40]
        out, _ = mind.say(s[:p], most=40, line=0.0)
        said, _ = mind.say(s[:p], most=40, line=SPEAKING_LINE)
        free.append((out, truth))
        line.append((said, truth))
        if a.corpus in corpora.PUBLIC and len(examples) < 6 and p % 7 == 0:
            examples.append({"before": s[p - 40:p], "free": out, "speaking_line": said, "was": truth})
    row = {"corpus": a.corpus, "model": "harmony (M2), frozen at the split", **score(free, line),
           "seconds": round(time.time() - t0, 1)}
    if examples:
        row["examples"] = examples
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
