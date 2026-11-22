# -*- coding: utf-8 -*-
"""t_scan_official.py: the published SCAN files (github.com/brendenlake/SCAN), not a generation of them.

    python research/living_mind_2026_11_22/t_scan_official.py verify --scan FOLDER
    python research/living_mind_2026_11_22/t_scan_official.py run --scan FOLDER --split length [--train 1000]
                                                                  [--test 0] [--workers 4]

``verify``: every result of these notes was run on SCAN as generated here from its grammar (``t_compose.scan_commands``,
splits in ``t_scan.py`` / ``t_scan_splits.py``). This compares that generation with the published files, line by line
as sets: the 20,910 commands with their answers (``tasks.txt``), and the training and test files of five splits
(add-jump, length, and the three template splits with "around right", "jump around right", "opposite right",
"right"). The action tokens are named differently here (the published I_TURN_LEFT is this generation's token for
"turn left", and so on): the correspondence is read from the six commands of one word or two that name one action,
and must be a one-to-one renaming.

``run``: a mind lives dialogues taken from the published training file of a split (every command without a
connective that the file holds, and ``--train`` sampled commands with one; each watched once an epoch for two epochs,
the shorter first in the first epoch: the life of ``t_scan_splits.py``) and answers the published test file
(``--test 0``: all of it). F43 and F45 on, window 13. The answers are given at the same time by processes that are
the mind as it stands (``zihin.paralel``): the same answers as one after another.

Appends to ``results/scan_official.jsonl``.
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

OUT = os.path.join(HERE, "results", "scan_official.jsonl")
FILES = {"add_jump": ("add_prim_split/tasks_train_addprim_jump.txt", "add_prim_split/tasks_test_addprim_jump.txt"),
         "length": ("length_split/tasks_train_length.txt", "length_split/tasks_test_length.txt"),
         "around_right": ("template_split/tasks_train_template_around_right.txt",
                          "template_split/tasks_test_template_around_right.txt"),
         "jump_around_right": ("template_split/tasks_train_template_jump_around_right.txt",
                               "template_split/tasks_test_template_jump_around_right.txt"),
         "opposite_right": ("template_split/tasks_train_template_opposite_right.txt",
                            "template_split/tasks_test_template_opposite_right.txt"),
         "right": ("template_split/tasks_train_template_right.txt", "template_split/tasks_test_template_right.txt")}


def published(path: str):
    """The lines of a published file as (command words, action tokens)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            command, answer = line[len("IN: "):].split(" OUT: ")
            out.append((tuple(command.split()), tuple(answer.split())))
    return out


def verify(args) -> None:
    import t_compose as T
    import t_scan_splits as X
    generated = [(tuple(c), tuple(a)) for c, a in T.scan_commands()]
    every = published(os.path.join(args.scan, "tasks.txt"))
    names = {}                                            # generated token -> published token
    ours = dict(generated)
    for command, answer in every:
        if len(set(answer)) == 1 and len(answer) == 1 and command in ours and len(ours[command]) == 1:
            names.setdefault(ours[command][0], answer[0])
    one_to_one = len(set(names.values())) == len(names)
    renamed = {(c, tuple(names.get(t, t) for t in a)) for c, a in generated}
    lines = ["== the published SCAN files against the generation used in these notes ==",
             "  action tokens (generated -> published), one to one %s: %s" % (
                 one_to_one, ", ".join("%s -> %s" % kv for kv in sorted(names.items()))),
             "  tasks.txt: %d published lines, %d distinct; generated %d; the same set: %s" % (
                 len(every), len(set(every)), len(renamed), set(every) == renamed)]
    row = {"test": "scan_official_verify", "tokens": names, "one_to_one": one_to_one,
           "tasks_same": set(every) == renamed, "splits": {}}

    def rename(pairs):
        return {(c, tuple(names.get(t, t) for t in a)) for c, a in pairs}
    for split, (train_file, test_file) in FILES.items():
        train, test = published(os.path.join(args.scan, train_file)), published(os.path.join(args.scan, test_file))
        if split == "add_jump":
            ours_train = rename([d for d in generated if "jump" not in d[0] or d[0] == ("jump",)])
            ours_test = rename([d for d in generated if "jump" in d[0] and d[0] != ("jump",)])
        else:
            X.SPLIT = split
            ours_train = rename([d for d in generated if X.in_training(*d)])
            ours_test = rename([d for d in generated if not X.in_training(*d)])
        same_train, same_test = set(train) == ours_train, set(test) == ours_test
        row["splits"][split] = {"train_lines": len(train), "train_distinct": len(set(train)), "test_lines": len(test),
                                "train_same": same_train, "test_same": same_test,
                                "overlap": len(set(c for c, _ in train) & set(c for c, _ in test))}
        lines.append("  %-18s train %6d lines (%6d distinct): the same set %s; test %5d lines: the same set %s; "
                     "commands in both %d" % (split, len(train), len(set(train)), same_train, len(test), same_test,
                                              row["splits"][split]["overlap"]))
    for line in lines:
        print(line, flush=True)
    row["summary"] = lines
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args) -> None:
    import t_relations as R
    from zihin import Mind, paralel
    from zihin.__main__ import CHANNEL, Session
    train_file, test_file = FILES[args.split]
    train = sorted(set(published(os.path.join(args.scan, train_file))))
    test = published(os.path.join(args.scan, test_file))
    rng = random.Random(args.seed)
    simple = [d for d in train if "and" not in d[0] and "after" not in d[0]]
    longer = [d for d in train if "and" in d[0] or "after" in d[0]]
    if args.train:
        longer = rng.sample(longer, min(args.train, len(longer)))
    if args.test:
        test = rng.sample(test, min(args.test, len(test)))
    lived = simple + longer
    rng = random.Random(args.seed + 1)
    life = []
    for epoch in range(2):
        block = list(lived)
        rng.shuffle(block)
        if epoch == 0:
            block.sort(key=lambda d: len(d[0]))
        life += block
    torch.manual_seed(args.seed)
    mind = Mind(R.speaking_config({"wholes": True, "told_apart": True}))
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = 13
    s = Session(mind)
    t0 = time.time()
    for command, answer in life:
        s.watch(" ".join(command), " ".join(answer) + " .")
    s.close_turn()
    lived_s = time.time() - t0
    t0 = time.time()
    said = paralel.answers(s, [" ".join(c) for c, _ in test], limit=400, workers=args.workers)
    answered_s = time.time() - t0
    truth = [" ".join(s.words(" ".join(a))) for _, a in test]
    right = [x == y for x, y in zip(said, truth)]
    silent = sum(x == "(sessiz)" for x in said)
    lines = ["== the published SCAN %s split: %d of the %d published training commands watched twice (%d without a "
             "connective, %d with one); the published test file%s ==" % (
                 args.split, len(lived), len(train), len(simple), len(longer),
                 "" if not args.test else ", a sample of %d" % len(test)),
             "  exact: %d of %d (not exact: %d silent, %d wrong)" % (
                 sum(right), len(test), silent, len(test) - sum(right) - silent),
             "  life %d moments in %.0f s; answers in %.0f s with %d processes (%.2f s an answer of one processor)" % (
                 mind.t, lived_s, answered_s, args.workers, answered_s * args.workers / max(1, len(test)))]
    wrong = [(" ".join(c), x) for (c, _), x, ok in zip(test, said, right) if not ok][:5]
    for command, x in wrong:
        lines.append("  not exact: %s -> %s" % (command, x[:120]))
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"test": "scan_official_run", "split": args.split, "seed": args.seed,
                            "train_watched": len(lived), "train_published": len(train), "asked": len(test),
                            "exact": sum(right), "silent": silent, "moments": mind.t, "life_s": round(lived_s, 1),
                            "answers_s": round(answered_s, 1), "workers": args.workers, "summary": lines},
                           ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["verify", "run"])
    ap.add_argument("--scan", required=True, help="the folder of the published repository")
    ap.add_argument("--split", choices=sorted(FILES), default="length")
    ap.add_argument("--train", type=int, default=1000, help="training commands with a connective (0: all)")
    ap.add_argument("--test", type=int, default=0, help="test commands (0: the whole published test file)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    (verify if args.what == "verify" else run)(args)


if __name__ == "__main__":
    main()
