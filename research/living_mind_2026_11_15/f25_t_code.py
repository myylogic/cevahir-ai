# -*- coding: utf-8 -*-
"""F25 on T-code: the lives of ``t_code.py`` (imported) with the setting ``born_failures`` off and on.

    python research/living_mind_2026_11_15/f25_t_code.py [--seeds 3] [--rounds 8]

"off" must reproduce ``results/t_code.jsonl`` row for row (checked and printed); "a" sets ``born_failures``.
Written to ``results/f25_t_code.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from t_code import run  # noqa: E402

CONDITIONS = ((True, True), (False, True), (True, False))     # (consequences, explore), as in t_code.py
SETTINGS = {"off": {}, "a": {"born_failures": True}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "f25_t_code.jsonl"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    rows = []
    for setting, over in SETTINGS.items():
        for seed in range(args.seeds):
            for consequences, explore in CONDITIONS:
                row = run(seed, args.rounds, consequences, explore, **over)
                row["setting"] = setting
                rows.append(row)
                print(json.dumps(row), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    first = os.path.join(HERE, "results", "t_code.jsonl")
    if os.path.exists(first):
        old = [json.loads(line) for line in open(first, encoding="utf-8")]
        mine = [{k: v for k, v in r.items() if k != "setting"} for r in rows if r["setting"] == "off"]
        same = sum(a == b for a, b in zip(mine, old))
        print("off reproduces results/t_code.jsonl: %d/%d rows identical" % (same, min(len(mine), len(old))))
    for consequences, explore in CONDITIONS:
        for setting in SETTINGS:
            group = [r for r in rows if r["setting"] == setting and r["consequences"] == consequences
                     and r["explore"] == explore]
            per_round = [sum(r["per_round"][i] for r in group) / len(group) for i in range(args.rounds)]
            later = sum(per_round[1:]) / (args.rounds - 1)
            print("consequences %-5s explore %-5s %-3s round 1 %.2f, rounds 2-%d %.3f, last %.2f  %s" % (
                consequences, explore, setting, per_round[0], args.rounds, later, per_round[-1],
                " ".join("%.2f" % p for p in per_round)))


if __name__ == "__main__":
    main()
