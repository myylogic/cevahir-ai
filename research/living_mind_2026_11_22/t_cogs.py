# -*- coding: utf-8 -*-
"""t_cogs.py: the published COGS files (Kim and Linzen 2020; github.com/najoungkim/COGS, MIT licence), as found.

    python research/living_mind_2026_11_22/t_cogs.py run --cogs FOLDER [--train 1000] [--test 100] [--gen 5]
                                                         [--workers 3] [--seed 0]

COGS asks for a sentence's logical form ("A rose was helped by a dog ." -> "rose ( x _ 1 ) AND help . theme ( x _ 3 ,
x _ 1 ) AND ..."), and tests 21 kinds of generalisation (``gen.tsv``, 1,000 sentences each) that the training file
(24,155 lines) is built to leave out: a noun seen only as a subject used as an object, deeper recursion than was
seen, an active verb in the passive, and so on. The files are not kept in this repository; ``FOLDER`` is the
``data`` folder of a copy of the published repository.

``run``: a mind lives dialogues taken from the published training file (every line that is not an ordinary
sentence: the 143 primitives and the exposure examples; and ``--train`` sampled ordinary sentences), each watched
once an epoch for two epochs, the shorter first in the first epoch (the life of ``t_scan_official.py``), and is
then asked ``--test`` sampled lines of the published in-distribution test file and ``--gen`` sampled lines of each
of the 21 generalisation cases. F43 and F45 on, window 13, as for SCAN. Exact match of the whole logical form.
The answers are given at the same time by processes that are the mind as it stands (``zihin.paralel``).

This is a first measurement of where the mind stands on a benchmark it was not built for, with a small part of
the training file: it is written down as it comes out.

Appends to ``results/cogs.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "cogs.jsonl")


def published(path: str):
    """The lines of a published file as (sentence, logical form, kind)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                out.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return out


def run(args) -> None:
    import t_relations as R
    from zihin import Mind, paralel
    from zihin.__main__ import CHANNEL, Session
    train = published(os.path.join(args.cogs, "train.tsv"))
    test = published(os.path.join(args.cogs, "test.tsv"))
    gen = published(os.path.join(args.cogs, "gen.tsv"))
    rng = random.Random(args.seed)
    special = [d for d in train if d[2] != "in_distribution"]
    ordinary = [d for d in train if d[2] == "in_distribution"]
    lived = special + rng.sample(ordinary, min(args.train, len(ordinary)))
    asked = [("in_distribution", d) for d in rng.sample(test, min(args.test, len(test)))]
    cases = sorted({d[2] for d in gen})
    for case in cases:
        of_case = [d for d in gen if d[2] == case]
        asked += [(case, d) for d in rng.sample(of_case, min(args.gen, len(of_case)))]
    rng = random.Random(args.seed + 1)
    life = []
    for epoch in range(2):
        block = list(lived)
        rng.shuffle(block)
        if epoch == 0:
            block.sort(key=lambda d: len(d[0].split()))
        life += block
    torch.manual_seed(args.seed)
    mind = Mind(R.speaking_config({"wholes": True, "told_apart": True}))
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = 13
    mind.dil = "diger"                                      # English: plain lower case
    s = Session(mind)
    t0 = time.time()
    for sentence, form, _ in life:
        s.watch(sentence, form + " .")
    s.close_turn()
    lived_s = time.time() - t0
    t0 = time.time()
    said = paralel.answers(s, [d[0] for _, d in asked], limit=400, workers=args.workers)
    answered_s = time.time() - t0
    by_case = {}
    examples = []
    for (case, (sentence, form, _)), x in zip(asked, said):
        truth = " ".join(s.words(form))
        row = by_case.setdefault(case, {"asked": 0, "exact": 0, "silent": 0})
        row["asked"] += 1
        row["exact"] += x == truth
        row["silent"] += x == "(sessiz)"
        if x != truth and len(examples) < 6:
            examples.append((case, sentence, x[:160], truth[:160]))
    gen_rows = {c: r for c, r in by_case.items() if c != "in_distribution"}
    lines = ["== the published COGS files: %d of the %d published training lines watched twice (%d that are not "
             "ordinary sentences, %d ordinary ones); exact match of the whole logical form ==" % (
                 len(lived), len(train), len(special), len(lived) - len(special)),
             "  in distribution (the published test file, a sample of %d): %d exact, %d silent" % (
                 by_case["in_distribution"]["asked"], by_case["in_distribution"]["exact"],
                 by_case["in_distribution"]["silent"]),
             "  generalisation (the published gen file, %d of each of its %d cases): %d exact of %d, %d silent" % (
                 args.gen, len(gen_rows), sum(r["exact"] for r in gen_rows.values()),
                 sum(r["asked"] for r in gen_rows.values()), sum(r["silent"] for r in gen_rows.values()))]
    for case in cases:
        r = by_case[case]
        lines.append("    %-58s %d of %d" % (case, r["exact"], r["asked"]))
    lines.append("  life %d moments in %.0f s; answers in %.0f s with %d processes" % (
        mind.t, lived_s, answered_s, args.workers))
    for case, sentence, x, truth in examples:
        lines.append("  not exact (%s): %s" % (case, sentence))
        lines.append("      said:  %s" % x)
        lines.append("      truth: %s" % truth)
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"test": "cogs_run", "seed": args.seed, "train_watched": len(lived),
                            "train_published": len(train), "by_case": by_case, "moments": mind.t,
                            "life_s": round(lived_s, 1), "answers_s": round(answered_s, 1), "workers": args.workers,
                            "summary": lines}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run"])
    ap.add_argument("--cogs", required=True, help="the data folder of a copy of the published repository")
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--gen", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
