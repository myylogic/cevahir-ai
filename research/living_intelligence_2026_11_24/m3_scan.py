# -*- coding: utf-8 -*-
"""M3: the published SCAN files, answered by Cevahir, by Zihin alone and by the harmony alone (NOTES.md, 8).

    python m3_scan.py --scan FOLDER [--split add_jump ...] [--test 200] [--workers 3] [--out results/m3_scan.jsonl]

The life of research/living_mind_2026_11_22/t_scan_official.py, unchanged: every training command of the published
file without a connective and ``--train`` (1,000) sampled commands with one, each watched once an epoch for two
epochs (the shorter first in the first epoch); Zihin as there (the speaking configuration with F43 and F45, window
13). Cevahir watches the same dialogues: Zihin lives each word as before, the harmony lives each dialogue as one line
(the command, a tab, the answer and its closing mark), with Zihin's expectations as its voice.

``--test`` commands of the published test file (a seeded sample; 0: all of it) are asked; each is answered by a
process that is the mind as it stands (nothing of the answers is lived), in three ways:

- zihin: Zihin's own reply, as in the earlier study;
- harmony: the harmony alone, saying the answer line under the speaking line;
- cevahir: the prefrontal decision between Zihin's reply and the harmony, token by token (living/cevahir.py).

An answer is exact when its words, without the closing mark, are the published answer's.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research", "living_mind_2026_11_22"))
import t_scan_official as T  # noqa: E402
from living.cevahir import Cevahir  # noqa: E402


def life_of(args, split):
    train_file, test_file = T.FILES[split]
    train = sorted(set(T.published(os.path.join(args.scan, train_file))))
    test = T.published(os.path.join(args.scan, test_file))
    rng = random.Random(args.seed)
    simple = [d for d in train if "and" not in d[0] and "after" not in d[0]]
    longer = [d for d in train if "and" in d[0] or "after" in d[0]]
    if args.train:
        longer = rng.sample(longer, min(args.train, len(longer)))
    if args.test:
        test = random.Random(args.seed + 7).sample(test, min(args.test, len(test)))
    lived = simple + longer
    rng = random.Random(args.seed + 1)
    life = []
    for epoch in range(2):
        block = list(lived)
        rng.shuffle(block)
        if epoch == 0:
            block.sort(key=lambda d: len(d[0]))
        life += block
    return train, lived, life, test


def own_record(c) -> dict:
    """Cevahir's record of Zihin alone: [right, all], and the claims by how far Zihin believed them (in twentieths)."""
    bins = {}
    for e, won in getattr(c, "za_claims", []):
        cell = bins.setdefault("%.2f" % (int(e * 20) / 20.0), [0, 0])
        cell[0] += won
        cell[1] += 1
    return {"za": [c.za_won, c.za_all], "za_by_belief": dict(sorted(bins.items()))}


def parse_switches(text: str) -> dict:
    """``--set`` as switches: "inherit=1,alone_now=1"."""
    return {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, text.split(",")))}


def run(args, split):
    import torch
    train, lived, life, test = life_of(args, split)
    torch.set_num_threads(1)
    switches = {"wholes": True, "told_apart": True, **parse_switches(args.set)}
    c = Cevahir(variant="speaking", switches=switches, window=13, seed=args.seed)
    t0 = time.time()
    for command, answer in life:
        c.watch(" ".join(command), " ".join(answer) + " .")
    c.session.close_turn()
    lived_s = time.time() - t0
    t0 = time.time()
    said = c.answers([" ".join(cmd) for cmd, _ in test], way="every", limit=400, workers=args.workers)
    answered_s = time.time() - t0
    truth = [c.session.words(" ".join(a)) for _, a in test]
    row = {"test": "m3_scan", "code": CODE, "split": split, "seed": args.seed, "switches": switches, **own_record(c),
           "train_watched": len(lived),
           "train_published": len(train), "asked": len(test), "zihin_moments": c.mind.t,
           "cortex": c.cortex.size(), "life_s": round(lived_s, 1), "answers_s": round(answered_s, 1),
           "workers": args.workers, "voiced_tokens": c.counts["voiced"], "unaligned": c.counts["unaligned"],
           "disagreements": {str(k): [c.dis_won[k], c.dis_all[k]] for k in sorted(c.dis_all)},
           "alone": {str(k): [c.alone_right[k], c.alone_all[k]] for k in sorted(c.alone_all)}}
    for way in ("zihin", "harmony", "cevahir"):
        got = [c.session.body(s[way]) if s[way] != "(sessiz)" else [] for s in said]
        exact = sum(g == t for g, t in zip(got, truth))
        silent = sum(not g for g in got)
        row[way] = {"exact": exact, "silent": silent, "wrong": len(test) - exact - silent}
    both = sum(c.session.body(s["cevahir"]) == t and c.session.body(s["zihin"]) != t for s, t in zip(said, truth))
    lost = sum(c.session.body(s["cevahir"]) != t and c.session.body(s["zihin"]) == t for s, t in zip(said, truth))
    row["cevahir_right_where_zihin_not"] = both
    row["cevahir_wrong_where_zihin_right"] = lost
    row["not_exact_examples"] = [{"command": " ".join(cmd), "cevahir": s["cevahir"], "zihin": s["zihin"]}
                                 for (cmd, _), s, t in zip(test, said, truth)
                                 if c.session.body(s["cevahir"]) != t][:8]           # SCAN is public
    print(json.dumps(row), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--set", default="", help="more switches, e.g. alone_now=1")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m3_scan.jsonl"))
    args = ap.parse_args()
    for split in args.split:
        run(args, split)


if __name__ == "__main__":
    main()
