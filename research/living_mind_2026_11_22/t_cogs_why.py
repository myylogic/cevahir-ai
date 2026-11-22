# -*- coding: utf-8 -*-
"""t_cogs_why.py: why the mind answers none of COGS (``t_cogs.py``: 0 of 205). Where its answers break, and what
changes when one difficulty at a time is taken away.

    python research/living_mind_2026_11_22/t_cogs_why.py run --cogs FOLDER [--form published|pointer|short]
                                                             [--train 1000] [--test 100] [--window 13] [--workers 2]
                                                             [--ac families,known]
    python research/living_mind_2026_11_22/t_cogs_why.py distance --cogs FOLDER [--form published|short]

The life of ``t_cogs.py`` (the lines that are not ordinary sentences and ``--train`` sampled ordinary ones, each
watched twice, the shorter first), then ``--test`` sampled lines of the published in-distribution test file. For
every answer: how many of its tokens are right before the first difference, and what was due there:

* ``index``: the number of a variable (in "x _ 3" the 3: the position of a word in the sentence);
* ``variable``: the "x" or "_" of a variable;
* ``word in the sentence``: a predicate that is a word of the sentence as it stands ("boy");
* ``word not in the sentence``: a predicate that is not (the base form of an inflected verb: "break" for "broke");
* ``role``: agent, theme, recipient, ccomp, xcomp, nmod and the preposition of an nmod;
* ``connective``: and, ;, * ; ``punctuation``: ( ) , . ; ``end``: the answer was complete and the mind went on.

and whether the mind stopped there (said nothing more) or said something else, and whether the whole sentence was
still among the items it holds at once (its window) at that point.

``--form``: ``published``, the files as they are. ``pointer``: every variable "x _ k" is written as the k-th word
of the sentence itself (nothing has to be counted; the form is otherwise the same): a diagnostic form, not COGS.
``short``: the published form, only sentences whose logical form has at most 20 tokens (in the life and in the
test): what is left when the answer does not outgrow the window by much.

``distance``: no mind. For the same sampled test sentences, in how many words each differs from the nearest
sentence of the same length among the sentences the life of ``run`` holds, and among the whole training file. A
frame generalises contexts that differ in one item: a sentence two or more words away from everything lived has no
frame.

Appends to ``results/cogs_why.jsonl``.
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

import t_cogs as C  # noqa: E402

OUT = os.path.join(HERE, "results", "cogs_why.jsonl")
ROLES = {"agent", "theme", "recipient", "ccomp", "xcomp", "nmod", "in", "on", "beside"}
CONNECTIVES = {"and", ";", "*"}
PUNCTUATION = {"(", ")", ",", "."}


def pointer(sentence: str, form: str) -> str:
    """The logical form with every variable "x _ k" written as the k-th word of the sentence."""
    words = sentence.split()
    tokens = form.split()
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] == "x" and i + 2 < len(tokens) and tokens[i + 1] == "_" and tokens[i + 2].isdigit():
            out.append(words[int(tokens[i + 2])])
            i += 3
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def due(truth, i: int, sentence_words) -> str:
    """The kind of the token that was due at place ``i`` of the true form."""
    if i >= len(truth):
        return "end"
    t = truth[i]
    if t.isdigit() and i >= 2 and truth[i - 1] == "_":
        return "index"
    if t in ("x", "_"):
        return "variable"
    if t in PUNCTUATION:
        return "punctuation"
    if t in CONNECTIVES:
        return "connective"
    if t in ROLES and i >= 1 and truth[i - 1] == ".":
        return "role"
    return "word in the sentence" if t in sentence_words else "word not in the sentence"


def run(args) -> None:
    import t_relations as R
    from zihin import Mind, paralel
    from zihin.__main__ import CHANNEL, Session
    train = C.published(os.path.join(args.cogs, "train.tsv"))
    test = C.published(os.path.join(args.cogs, "test.tsv"))
    if args.form == "pointer":
        change = lambda d: (d[0], pointer(d[0], d[1]) if d[2] in ("in_distribution",) or " " in d[0] else d[1], d[2])  # noqa: E731
        train = [change(d) for d in train]
        test = [change(d) for d in test]
    if args.form == "short":
        train = [d for d in train if len(d[1].split()) <= 20]
        test = [d for d in test if len(d[1].split()) <= 20]
    rng = random.Random(args.seed)
    special = [d for d in train if d[2] != "in_distribution"]
    ordinary = [d for d in train if d[2] == "in_distribution"]
    lived = special + rng.sample(ordinary, min(args.train, len(ordinary)))
    asked = rng.sample(test, min(args.test, len(test)))
    rng = random.Random(args.seed + 1)
    life = []
    for epoch in range(2):
        block = list(lived)
        rng.shuffle(block)
        if epoch == 0:
            block.sort(key=lambda d: len(d[0].split()))
        life += block
    torch.manual_seed(args.seed)
    switches = {"wholes": True, "told_apart": True}
    switches.update({name: True for name in args.ac.split(",") if name})
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = args.window
    mind.dil = "diger"
    s = Session(mind)
    t0 = time.time()
    for sentence, form, _ in life:
        s.watch(sentence, form + " .")
    s.close_turn()
    lived_s = time.time() - t0
    t0 = time.time()
    said = paralel.answers(s, [d[0] for d in asked], limit=400, workers=args.workers)
    answered_s = time.time() - t0
    lived_sentences = {d[0] for d in lived}
    counts = {"asked": len(asked), "exact": 0, "silent": 0, "stopped": 0, "said_otherwise": 0, "went_on": 0}
    at = {}
    in_window = {"stopped": [0, 0], "said_otherwise": [0, 0]}
    right_tokens = due_tokens = 0
    by_length = {"up to 20": [0, 0], "21 to 40": [0, 0], "over 40": [0, 0]}
    examples = []
    for (sentence, form, _), x in zip(asked, said):
        truth = s.words(form)
        words = [] if x == "(sessiz)" else x.split()
        sentence_words = set(s.words(sentence))
        bucket = "up to 20" if len(truth) <= 20 else ("21 to 40" if len(truth) <= 40 else "over 40")
        by_length[bucket][1] += 1
        n = 0
        while n < len(words) and n < len(truth) and words[n] == truth[n]:
            n += 1
        right_tokens += n
        due_tokens += len(truth)
        if words == truth:
            counts["exact"] += 1
            by_length[bucket][0] += 1
            continue
        kind = due(truth, n, sentence_words)
        if not words:
            how = "silent"
        elif n == len(words):
            how = "stopped"
        elif n == len(truth):
            how = "went_on"
        else:
            how = "said_otherwise"
        counts[how] += 1
        row = at.setdefault(kind, {"silent": 0, "stopped": 0, "said_otherwise": 0, "went_on": 0})
        row[how] += 1
        if how in in_window:
            whole = len(s.words(sentence)) + 1 + n <= args.window     # the sentence, the end of the turn, what it said
            in_window[how][0 if whole else 1] += 1
        if len(examples) < 8 and how != "silent":
            examples.append((sentence, " ".join(words[max(0, n - 4):n + 3]), " ".join(truth[max(0, n - 4):n + 3]),
                             how, kind))
    lines = ["== COGS, where the answers break (%s form%s): %d training lines watched twice (%d ordinary), %d lines "
             "of the in-distribution test file; window %d ==" % (
                 args.form, "; with " + args.ac if args.ac else "", len(lived), len(lived) - len(special),
                 len(asked), args.window),
             "  exact %d ; said nothing %d ; stopped before the end %d ; said another token %d ; went on after the "
             "end %d" % (counts["exact"], counts["silent"], counts["stopped"], counts["said_otherwise"],
                         counts["went_on"]),
             "  tokens right before the first difference: %d of %d (%.0f%%)" % (
                 right_tokens, due_tokens, 100.0 * right_tokens / max(1, due_tokens)),
             "  exact by the length of the true form: %s" % " ; ".join(
                 "%s tokens %d of %d" % (name, a, b) for name, (a, b) in by_length.items()),
             "  the sentences asked that were lived as they stand: %d of %d" % (
                 sum(1 for d in asked if d[0] in lived_sentences), len(asked)),
             "  what was due where the answer broke (said nothing / stopped / said another token / went on):"]
    for kind in sorted(at, key=lambda k: -sum(at[k].values())):
        r = at[kind]
        lines.append("    %-26s %d / %d / %d / %d" % (kind, r["silent"], r["stopped"], r["said_otherwise"],
                                                       r["went_on"]))
    lines.append("  was the whole sentence still in its window where it broke: stopped %d yes, %d no ; said another "
                 "token %d yes, %d no" % (in_window["stopped"][0], in_window["stopped"][1],
                                          in_window["said_otherwise"][0], in_window["said_otherwise"][1]))
    lines.append("  life %d moments in %.0f s; answers in %.0f s with %d processes" % (
        mind.t, lived_s, answered_s, args.workers))
    for sentence, x, truth, how, kind in examples:
        lines.append("  %s at %s: %s" % (how, kind, sentence))
        lines.append("      said:  ... %s" % x)
        lines.append("      truth: ... %s" % truth)
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"test": "cogs_why", "form": args.form, "switches": args.ac, "seed": args.seed,
                            "window": args.window,
                            "train_watched": len(lived), "counts": counts, "due_where_it_broke": at,
                            "sentence_in_window": in_window, "tokens_right": right_tokens, "tokens": due_tokens,
                            "exact_by_length": by_length, "moments": mind.t, "life_s": round(lived_s, 1),
                            "answers_s": round(answered_s, 1), "summary": lines}, ensure_ascii=False) + "\n")


def distance(args) -> None:
    train = C.published(os.path.join(args.cogs, "train.tsv"))
    test = C.published(os.path.join(args.cogs, "test.tsv"))
    if args.form == "short":
        train = [d for d in train if len(d[1].split()) <= 20]
        test = [d for d in test if len(d[1].split()) <= 20]
    rng = random.Random(args.seed)
    special = [d for d in train if d[2] != "in_distribution"]
    ordinary = [d for d in train if d[2] == "in_distribution"]
    lived = special + rng.sample(ordinary, min(args.train, len(ordinary)))
    asked = rng.sample(test, min(args.test, len(test)))

    def nearest(sentence, pool):
        words = sentence.split()
        best = None
        for other in pool.get(len(words), ()):
            d = sum(1 for a, b in zip(words, other) if a != b)
            if best is None or d < best:
                best = d
                if d == 0:
                    break
        return best

    def by_length(lines):
        pool = {}
        for d in lines:
            words = d[0].split()
            pool.setdefault(len(words), []).append(words)
        return pool
    rows = {}
    for name, lines in (("the life", lived), ("the whole training file", train)):
        pool = by_length(lines)
        counts = {"0": 0, "1": 0, "2": 0, "3": 0, "4 or more": 0, "no sentence of that length": 0}
        for sentence, _, _ in asked:
            d = nearest(sentence, pool)
            counts["no sentence of that length" if d is None else ("4 or more" if d >= 4 else str(d))] += 1
        rows[name] = counts
    lines = ["== COGS, how far the asked sentences are from what was lived (%s form): %d sampled lines of the "
             "in-distribution test file; words in which a sentence differs from the nearest sentence of its length "
             "==" % (args.form, len(asked))]
    for name, counts in rows.items():
        lines.append("  among %s (%d sentences): %s" % (
            name, len(lived) if name == "the life" else len(train),
            " ; ".join("%s: %d" % kv for kv in counts.items())))
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"test": "cogs_distance", "form": args.form, "seed": args.seed, "asked": len(asked),
                            "lived": len(lived), "train_published": len(train), "words_from_the_nearest": rows,
                            "summary": lines}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "distance"])
    ap.add_argument("--cogs", required=True, help="the data folder of a copy of the published repository")
    ap.add_argument("--form", choices=["published", "pointer", "short"], default="published")
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--window", type=int, default=13)
    ap.add_argument("--ac", default="", help="studied switches to turn on beside F43 and F45 (comma separated)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    (run if args.what == "run" else distance)(args)


if __name__ == "__main__":
    main()
