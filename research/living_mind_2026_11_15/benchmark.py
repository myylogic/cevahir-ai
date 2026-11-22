# -*- coding: utf-8 -*-
"""benchmark.py — one scoreboard: Cevahir Zihin against a Transformer trained from scratch on the very same
dialogues, across the compositional-generalization splits used in the notes. It reuses the runners of
``t_scan`` / ``t_scan_splits`` / ``t_dans``; it adds no world and downloads nothing.

    python research/living_mind_2026_11_15/benchmark.py                    # a quick smoke run (small, fast)
    python research/living_mind_2026_11_15/benchmark.py --full             # the sizes used in the notes
    python research/living_mind_2026_11_15/benchmark.py --only add-jump,around-right [--no-transformer]

Both learners see exactly the same training dialogues. The mind watches each once an epoch (no gradient, no
weights); the Transformer (3 layers, 4 heads, width 64) is trained from scratch on the same dialogues. Reported:
exact match of the whole held-out answer. This is the fair same-data, same-compute control. A comparison against a
*pretrained* model from Hugging Face is a different setting (pretraining plus in-context learning) and needs a model
you download and approve first — see ``research/BENCHMARKING.md``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import torch  # noqa: E402

torch.set_num_threads(1)

import t_dans  # noqa: E402
import t_scan  # noqa: E402
import t_scan_splits  # noqa: E402

# t_scan_splits.run_transformer and t_dans.run_transformer both rebind t_scan.split / t_scan.life to train their
# own split's Transformer; keep the originals so the add-jump runner is never left pointing at another split.
ORIG_SPLIT, ORIG_LIFE = t_scan.split, t_scan.life

# name -> (the SCAN split key, or None for add-jump / t-dans), a one-line description
BENCHMARKS = {
    "add-jump": (None, "jump seen only alone; test: every command in which jump stands (Lake & Baroni 2018)"),
    "length": ("length", "train answers of <=22 actions; test 24..48 (longer than anything seen)"),
    "around-right": ("around_right", "around right never seen in training; test: every command in which it stands"),
    "jump-around-right": ("jump_around_right", "jump around right never seen in training"),
    "t-dans": (None, "Turkish: a direction never said alone, held out with iki/uc kez (F43 was not developed on it)"),
}


def tally(rows):
    return sum(r["right"] for r in rows), len(rows)


def run_mind(name, switches, train, test, seeds, workers, epochs, window):
    right = total = 0
    t0 = time.time()
    for seed in range(seeds):
        if name == "add-jump":
            r = t_scan.run(seed, switches, train, test, window, epochs, workers)
        elif name == "t-dans":
            r = t_dans.run(seed, switches, train, test, window, epochs)
        else:
            t_scan_splits.SPLIT = BENCHMARKS[name][0]
            r = t_scan_splits.run(seed, switches, train, test, window, epochs, workers)
        a, b = tally(r["rows"])
        right += a
        total += b
    return right, total, time.time() - t0


def run_transformer(name, train, test, epochs, steps):
    t0 = time.time()
    try:
        if name == "add-jump":
            t_scan.split, t_scan.life = ORIG_SPLIT, ORIG_LIFE
            r = t_scan.run_transformer(0, train, test, epochs, steps)
        elif name == "t-dans":
            r = t_dans.run_transformer(0, train, test, epochs, steps)
        else:
            t_scan_splits.SPLIT = BENCHMARKS[name][0]
            r = t_scan_splits.run_transformer(0, train, test, epochs, steps)
    finally:
        t_scan.split, t_scan.life = ORIG_SPLIT, ORIG_LIFE   # leave the add-jump runner as it was
    a, b = tally(r["rows"])
    return a, b, time.time() - t0


def pct(a, b):
    return "%d / %d (%.0f%%)" % (a, b, 100.0 * a / b) if b else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="the sizes used in the notes (slower)")
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ", ".join(BENCHMARKS))
    ap.add_argument("--train", type=int, default=0, help="training dialogues drawn (0: the mode's default)")
    ap.add_argument("--test", type=int, default=0, help="test commands drawn (0: the mode's default)")
    ap.add_argument("--steps", type=int, default=0, help="Transformer training steps (0: the mode's default)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--window", type=int, default=13)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--set", default="wholes=1,told_apart=1", help="the mind's switches")
    ap.add_argument("--no-transformer", action="store_true", help="the mind only (skip the slow Transformer)")
    args = ap.parse_args()

    train = args.train or (1000 if args.full else 200)
    test = args.test or (500 if args.full else 60)
    steps = args.steps or (6000 if args.full else 1200)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(BENCHMARKS)
    for n in names:
        if n not in BENCHMARKS:
            sys.exit("unknown benchmark %r; choose from: %s" % (n, ", ".join(BENCHMARKS)))
    if "add-jump" in names and train < 120:              # t_scan probes 100 watched long commands
        print("   (skipping add-jump: its watched-command probe needs >=120 training dialogues; raise --train)")
        names = [n for n in names if n != "add-jump"]

    mode = "full (notes sizes)" if args.full else "quick smoke"
    print("== Cevahir Zihin vs a from-scratch Transformer — %s ==" % mode)
    print("   same dialogues for both; mind switches %s; %d seed(s), window %d, %d workers" % (
        switches or "-", args.seeds, args.window, args.workers))
    print("   %-20s %-7s %-6s %-18s %s" % ("benchmark", "train", "test", "Zihin (no gradient)", "Transformer (scratch)"))
    for name in names:
        mr, mn, ms = run_mind(name, switches, train, test, args.seeds, args.workers, args.epochs, args.window)
        if args.no_transformer:
            tr_s = "(skipped)"
        else:
            tr, tn, ts = run_transformer(name, train, test, args.epochs, steps)
            tr_s = "%-18s (%.0fs)" % (pct(tr, tn), ts)
        print("   %-20s %-7d %-6d %-18s %s" % (name, train, test, pct(mr, mn) + " (%.0fs)" % ms, tr_s))
    if not args.full:
        print("   (quick smoke sizes; run with --full for the sizes in NOTES.md / research reports)")


if __name__ == "__main__":
    main()
