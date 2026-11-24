# -*- coding: utf-8 -*-
"""M3 baseline: a character Transformer on the published SCAN files, trained on exactly the dialogues Cevahir
watched and asked exactly the commands Cevahir was asked (m3_scan.py; NOTES.md, 8).

    python m3_scan_transformer.py --scan FOLDER [--split add_jump ...] [--test 200] [--minutes 15] [--threads 2]

The life is m3_scan.py's (``life_of``: the same commands in the same order, watched twice); the stream is what
Cevahir's harmony lived of it, one dialogue a line (the command, a tab, the answer and its closing mark). The model
is m3_transformer.py's (from scratch, ``--minutes`` of training). Each test command is asked on a new line and the
answer is its greedy decoding to the line end; exact when its words, without the closing mark, are the published
answer's.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "research", "living_mind_2026_11_22"))
import m3_scan  # noqa: E402
import m3_transformer as MT  # noqa: E402


def code_version() -> str:
    """The commit of the code that wrote a row."""
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


CODE = code_version()                                   # the code that runs, read when it starts

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--split", nargs="*", default=["add_jump", "around_right", "length", "jump_around_right",
                                                   "opposite_right", "right"])
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m3_scan.jsonl"))
    args = ap.parse_args()
    from zihin.__main__ import words
    for split in args.split:
        train, lived, life, test = m3_scan.life_of(args, split)
        stream = "\n".join(" ".join(cmd) + "\t" + " ".join(ans) + " ." for cmd, ans in life)
        got = MT.train_and_answer(stream, [" ".join(cmd) for cmd, _ in test], minutes=args.minutes,
                                  threads=args.threads, seed=args.seed, most=800)
        exact = silent = 0
        for said, (_, ans) in zip(got["answers"], test):
            w = words(said)
            while w and w[-1] == ".":
                w = w[:-1]
            exact += w == words(" ".join(ans))
            silent += not w
        row = {"test": "m3_scan", "code": CODE, "split": split, "seed": args.seed, "who": "transformer",
               "train_watched": len(lived), "asked": len(test),
               "transformer": {"exact": exact, "silent": silent, "wrong": len(test) - exact - silent},
               "transformer_run": {k: got[k] for k in ("params", "steps", "train_s", "answer_s", "final_loss", "chars")}}
        print(json.dumps(row), flush=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
