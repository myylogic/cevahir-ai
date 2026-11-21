# -*- coding: utf-8 -*-
"""T-görev: jobs made of jobs, in Turkish. A public test, generated here. It was written before F43 (Wholes) was run
on it: F43 was developed on SCAN's grammar (t_scan.py) only. This world differs from SCAN in its words, in what a
primitive is (a job of two words whose answer is two or three steps, not one word with one action), and in how
commands are built (the words that repeat a job stand before it, "iki kere çay demle"; the words that order two jobs
stand around them, "önce ... sonra ...", or between them with the order reversed, "... ama önce ...").

    python research/living_mind_2026_11_15/t_gorev.py [--seeds 3] [--train 150] [--window 13] [--set key=0|1,...]

The world. Five jobs, each answered with its steps when said alone:
    çay demle -> su_kaynat çay_koy bekle          masa kur -> örtü_ser tabak_koy
    bulaşık yıka -> ıslat köpürt durula            ekmek al -> fırına_git ekmek_seç öde
    kahve yap -> su_ısıt kahve_koy karıştır        (the new job: watched only alone)
A part P is a job, "iki kere" a job (its steps twice) or "üç kere" a job (three times). A command is a part, or
"önce P sonra Q" (the steps of P, then of Q), or "Q ama önce P" (the steps of P, then of Q). Answers have up to 18
steps and commands up to 10 words; most dialogues do not fit the mind's window.

Training: the commands without "kahve yap" (the 12 parts of the four other jobs, and ``--train`` of the 288 commands
that join two of them, drawn at random), and "kahve yap" alone; each watched once an epoch, two epochs, the shorter
commands first in the first epoch. Test: every command in which "kahve yap" stands (164; "kahve yap" alone is not one
of them), each told to a copy of the mind as it was at the end of its life. Exact match of the whole answer.

Reported: all; commands that are one part, and commands that join two; commands in which both parts hold the new job.

Appends to ``results/gorev.jsonl``.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "gorev.jsonl")
JOBS = {("çay", "demle"): ("su_kaynat", "çay_koy", "bekle"), ("masa", "kur"): ("örtü_ser", "tabak_koy"),
        ("bulaşık", "yıka"): ("ıslat", "köpürt", "durula"), ("ekmek", "al"): ("fırına_git", "ekmek_seç", "öde"),
        ("kahve", "yap"): ("su_ısıt", "kahve_koy", "karıştır")}
NEW = ("kahve", "yap")


def parts() -> List[Tuple[tuple, tuple]]:
    out = []
    for job, steps in JOBS.items():
        out += [(job, steps), (("iki", "kere") + job, steps * 2), (("üç", "kere") + job, steps * 3)]
    return out


def commands() -> List[Tuple[tuple, tuple]]:
    every = list(parts())
    for (p, a), (q, b) in itertools.product(parts(), parts()):
        every.append((("önce",) + p + ("sonra",) + q, a + b))
        every.append((q + ("ama", "önce") + p, a + b))
    return every


def holds_new(command: tuple) -> bool:
    return any(command[i:i + 2] == NEW for i in range(len(command) - 1))


def split(seed: int, n_train: int):
    rng = random.Random(seed)
    every = commands()
    plain = [d for d in every if not holds_new(d[0])]
    simple = [d for d in plain if "önce" not in d[0]]
    joined = [d for d in plain if "önce" in d[0]]
    if n_train:
        joined = rng.sample(joined, min(n_train, len(joined)))
    test = [d for d in every if holds_new(d[0]) and d[0] != NEW]
    return simple + joined + [(NEW, JOBS[NEW])], test


def life(seed: int, train, epochs: int):
    rng = random.Random(seed + 1)
    out = []
    for epoch in range(epochs):
        block = list(train)
        rng.shuffle(block)
        if epoch == 0:
            block.sort(key=lambda d: len(d[0]))          # stable: the shorter commands first, shuffled among equals
        out += block
    return out


def run(seed: int, switches: Dict[str, bool], n_train: int, window: int, epochs: int) -> dict:
    import t_relations as R
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session
    torch.manual_seed(seed)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = window
    s = Session(mind)
    train, test = split(seed, n_train)
    t0 = time.time()
    for command, answer in life(seed, train, epochs):
        s.watch(" ".join(command), " ".join(answer) + " .")
    lived = time.time() - t0
    rows = []
    for command, answer in test:
        said = Session(mind.clone()).reply(" ".join(command), limit=400)
        both = "önce" in command and sum(command[i:i + 2] == NEW for i in range(len(command) - 1)) == 2
        rows.append({"command": " ".join(command), "said": said, "right": said == " ".join(answer),
                     "joined": "önce" in command, "both": both, "fits": len(command) + 1 + len(answer) <= window - 1})
    return {"seed": seed, "train": len(train), "moments": mind.t, "lived_s": round(lived, 1), "rows": rows}


def main() -> None:
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train", type=int, default=150, help="training commands that join two parts (0: all 288)")
    ap.add_argument("--window", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--set", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    runs = [run(seed, switches, args.train, args.window, args.epochs) for seed in range(args.seeds)]
    rows = [x for r in runs for x in r["rows"]]

    def of(keep) -> str:
        kept = [x for x in rows if keep(x)]
        return "%d / %d" % (sum(x["right"] for x in kept), len(kept))
    lines = ["== T-görev: %d seeds, %d training dialogues an epoch (%d epochs, window %d), switches %s: exact %s ==" % (
        args.seeds, runs[0]["train"], args.epochs, args.window, switches or "-", of(lambda x: True)),
        "  one part: %s; two parts joined: %s (both hold the new job: %s)" % (
            of(lambda x: not x["joined"]), of(lambda x: x["joined"]), of(lambda x: x["both"])),
        "  the dialogue fits the window: %s; it does not: %s" % (of(lambda x: x["fits"]), of(lambda x: not x["fits"])),
        "  lived %s moments (seed 0) in %.0f s" % (runs[0]["moments"], runs[0]["lived_s"])]
    lines += ["  wrong (seed 0): %s -> %s" % (x["command"], x["said"][:150])
              for x in [x for x in runs[0]["rows"] if not x["right"]][:6]]
    for line in lines:
        print(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seeds": args.seeds, "train": args.train, "window": args.window, "epochs": args.epochs,
                            "switches": switches, "summary": lines, "runs": runs}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
