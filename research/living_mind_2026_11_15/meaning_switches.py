# -*- coding: utf-8 -*-
"""T-meaning (``t_meaning.py``, imported, not changed) with reading switches of the chat configuration set for the whole
life, to measure what a switch does to greetings, paraphrases and meaning questions.

    python research/living_mind_2026_11_15/meaning_switches.py --seeds 3 --set analogy=0
    python research/living_mind_2026_11_15/meaning_switches.py --seeds 3 --set analogy=1

The world, the raising, the probes and the summary are t_meaning's; the only difference is the configuration that
``t_meaning.speaking_config`` returns (the chat's speaking configuration, ledger off), with the given switches. Prints the
summary and appends its lines, with the switches, to ``results/meaning_switches.jsonl`` (a public world: the summary
holds only this world's words and scores).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "results", "meaning_switches.jsonl")


def main() -> None:
    import torch
    import t_meaning as T
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--set", default="", help="switches, e.g. analogy=1,substitution=0")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    plain = T.speaking_config

    def configured():
        cfg = plain()
        for k, v in switches.items():
            setattr(cfg, k, v)
        return cfg
    T.speaking_config = configured
    rows = []
    for seed in range(args.seeds):
        rows += T.run(seed, args.days, False, open(os.devnull, "w"))
    lines = list(T.summary(rows))
    for line in lines:
        print(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seeds": args.seeds, "days": args.days, "switches": switches, "summary": lines},
                           ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
