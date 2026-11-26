# -*- coding: utf-8 -*-
"""M3: COGS (Kim and Linzen 2020), lived by Cevahir from its whole training file (NOTES.md, 8).

    python m3_cogs.py --cogs FOLDER [--test 300] [--gen 300] [--workers 3]

``FOLDER`` is the ``data`` folder of the published repository (github.com/najoungkim/COGS, MIT licence; not kept
here). Cevahir watches every line of ``train.tsv`` once, in the file's order (the sentence said, its logical form
answered): Zihin lives each word, the harmony each character, and the hippocampus keeps each dialogue. Then a seeded
sample of the in-distribution test file and of the generalisation file (``random.Random(0)``: ``--test`` lines of
test.tsv, then ``--gen`` of gen.tsv) is asked, each by a process that is the mind as it stands, in four ways: Zihin's
reply, the harmony alone, the hippocampus's analogy alone, and Cevahir (the prefrontal choice of the thought, said
token by token). Exact: the logical form's tokens, all of them, in order; counted with their case and without it
(Zihin's words are lower-cased).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from living.cevahir import Cevahir  # noqa: E402


def read(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


def code_version() -> str:
    """The commit of the code that wrote a row."""
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


CODE = code_version()                                   # the code that runs, read when it starts

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cogs", required=True)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--gen", type=int, default=300)
    ap.add_argument("--train", type=int, default=0, help="only the first N training lines (0: all)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--set", default="", help="more switches, e.g. alone_now=1")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m3_cogs.jsonl"))
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    train = read(os.path.join(a.cogs, "train.tsv"))
    if a.train:
        train = train[:a.train]
    rng = random.Random(0)
    test = rng.sample(read(os.path.join(a.cogs, "test.tsv")), a.test)
    gen = rng.sample(read(os.path.join(a.cogs, "gen.tsv")), a.gen)
    switches = {"wholes": True, "told_apart": True, **parse_switches(a.set)}
    c = Cevahir(variant="speaking", switches=switches, window=13)
    t0 = time.time()
    for sentence, form, _ in train:
        c.watch(sentence, form)
    c.session.close_turn()
    lived_s = time.time() - t0
    asked = test + gen
    t1 = time.time()
    said = c.answers([s for s, _, _ in asked], way="every", limit=120, workers=a.workers)
    row = {"test": "m3_cogs", "code": CODE, "train_lines": len(train), "zihin_moments": c.mind.t, "episodes": len(c.episodes),
           "life_s": round(lived_s, 1), "answers_s": round(time.time() - t1, 1), "workers": a.workers,
           "thought": dict(Counter(s.get("thought", "?") for s in said)), "switches": switches,
           **own_record(c)}
    for part, rows, got in (("test", test, said[:len(test)]), ("gen", gen, said[len(test):])):
        cell = {}
        kinds = Counter(k for _, _, k in rows)
        for way in ("zihin", "harmony", "analogy", "cevahir"):
            exact = loose = silent = 0
            by_kind = Counter()
            for (_, form, kind), s in zip(rows, got):
                x = s[way]
                if not x or x == "(sessiz)":
                    silent += 1
                    continue
                exact += x.split() == form.split()
                ok = x.lower().split() == form.lower().split()
                loose += ok
                by_kind[kind] += ok
            cell[way] = {"exact": exact, "exact_ignoring_case": loose, "silent": silent}
            if part == "gen":
                cell[way]["by_kind_ignoring_case"] = {k: [by_kind[k], kinds[k]] for k in sorted(kinds)}
        row[part] = cell
        row[part + "_asked"] = len(rows)
    print(json.dumps({k: v for k, v in row.items()}, ensure_ascii=False)[:3000], flush=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
