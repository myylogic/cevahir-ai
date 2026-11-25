# -*- coding: utf-8 -*-
"""M3: the hippocampus alone (living/analogy.py) on SCAN and COGS (NOTES.md, 8.7 and 8.14).

    python m3_analogy.py --scan SCAN_FOLDER --cogs COGS_FOLDER/data [--test 50] [--sample 300]
    python m3_analogy.py --cogs COGS_FOLDER/data --every 7 [--shard 0 --shards 2]

SCAN: for each split, every published training pair is lived (each pair once, in sorted order), then the same seeded
sample of the test file as m3_scan.py (``random.Random(seed + 7).sample(test, n)``) is asked, by the analogy over
places and by parts. COGS: every line of train.tsv is lived in order, then ``random.Random(0).sample`` of 300 lines
of test.tsv and then of gen.tsv is asked the same way (the samples of NOTES 8.7). A plan is said when its belief
reaches the speaking line; each way is scored right / silent / wrong, and COGS's generalisation set by kind.
With ``--every K`` the whole generalisation file is asked instead, every K-th line (by parts only), split over
``--shards`` processes that each ask the lines of their own shard (lines shard*K, (shard+shards)*K, ...).
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
from living.analogy import Episodes, tokens  # noqa: E402
from living.evidence import SPEAKING_LINE  # noqa: E402


def code_version() -> str:
    """The commit of the code that wrote a row."""
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


CODE = code_version()


def ask(ep: Episodes, asked, kinds=None) -> dict:
    """Each way of answering, scored: right / silent / wrong (and, with ``kinds``, right of asked per kind)."""
    out = {}
    for way, parts in (("plain", False), ("parts", True)):
        score, by_kind = {"right": 0, "silent": 0, "wrong": 0}, {}
        for n, (said, answer) in enumerate(asked):
            plan, e, _ = ep.plan(said, parts=parts)
            tag = "silent" if plan is None or e < SPEAKING_LINE else ("right" if plan == answer else "wrong")
            score[tag] += 1
            if kinds is not None:
                row = by_kind.setdefault(kinds[n], [0, 0])
                row[0] += tag == "right"
                row[1] += 1
        out[way] = score
        if kinds is not None:
            out[way + "_kinds"] = {k: by_kind[k] for k in sorted(by_kind)}
    return out


def scan(folder: str, n: int, seed: int) -> dict:
    import t_scan_official as T
    rows = {}
    for split, (train_file, test_file) in T.FILES.items():
        t0 = time.time()
        ep = Episodes()
        for command, actions in sorted(set(T.published(os.path.join(folder, train_file)))):
            ep.live(list(command), list(actions))
        ep.ready()
        test = random.Random(seed + 7).sample(T.published(os.path.join(folder, test_file)), n)
        rows[split] = ask(ep, [(list(c), list(a)) for c, a in test])
        rows[split]["seconds"] = round(time.time() - t0, 1)
        print("scan", split, rows[split], flush=True)
    return rows


def cogs(folder: str, n: int) -> dict:
    def read(name):
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            return [line.rstrip("\n").split("\t") for line in f if line.strip()]
    t0 = time.time()
    ep = Episodes()
    for sentence, form, _ in read("train.tsv"):
        ep.live(tokens(sentence), tokens(form))
    ep.ready()
    lived_s = round(time.time() - t0, 1)
    rng = random.Random(0)
    test, gen = rng.sample(read("test.tsv"), n), rng.sample(read("gen.tsv"), n)
    rows = {"lived": len(ep), "life_s": lived_s}
    for part, sample in (("test", test), ("gen", gen)):
        t1 = time.time()
        rows[part] = ask(ep, [(tokens(s), tokens(f)) for s, f, _ in sample],
                         [k for _, _, k in sample] if part == "gen" else None)
        rows[part]["seconds"] = round(time.time() - t1, 1)
        print("cogs", part, {k: v for k, v in rows[part].items() if not k.endswith("_kinds")}, flush=True)
    return rows


def cogs_every(folder: str, every: int, shard: int, shards: int) -> dict:
    def read(name):
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            return [line.rstrip("\n").split("\t") for line in f if line.strip()]
    ep = Episodes()
    for sentence, form, _ in read("train.tsv"):
        ep.live(tokens(sentence), tokens(form))
    ep.ready()
    gen = read("gen.tsv")
    t0 = time.time()
    score, by_kind = {"right": 0, "silent": 0, "wrong": 0}, {}
    for idx in range(shard * every, len(gen), shards * every):
        sentence, form, kind = gen[idx]
        plan, e, _ = ep.plan(tokens(sentence))
        tag = "silent" if plan is None or e < SPEAKING_LINE else ("right" if plan == tokens(form) else "wrong")
        score[tag] += 1
        row = by_kind.setdefault(kind, {"right": 0, "silent": 0, "wrong": 0})
        row[tag] += 1
    return {"every": every, "shard": shard, "shards": shards, "gen": score,
            "gen_kinds": {k: by_kind[k] for k in sorted(by_kind)}, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="")
    ap.add_argument("--cogs", default="")
    ap.add_argument("--test", type=int, default=50)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m3_analogy.jsonl"))
    args = ap.parse_args()
    row = {"test": "m3_analogy", "code": CODE, "seed": args.seed}
    if args.every:
        row["cogs_gen_every"] = cogs_every(args.cogs, args.every, args.shard, args.shards)
        print(json.dumps(row["cogs_gen_every"]["gen"]), flush=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return
    if args.scan:
        row["scan"] = scan(args.scan, args.test, args.seed)
        row["scan_asked"] = args.test
    if args.cogs:
        row["cogs"] = cogs(args.cogs, args.sample)
        row["cogs_asked"] = args.sample
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
