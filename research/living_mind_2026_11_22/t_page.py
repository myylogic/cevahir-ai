# -*- coding: utf-8 -*-
"""t_page.py: is a page that is looked up read as well as a document that is read?

    python research/living_mind_2026_11_22/t_page.py [--seeds 5]

``Session.read_path`` (``!oku``) lives each sentence of a document, perceives its end (the end of the writer's turn,
with how it ended) and lets a moment pass; no one in particular is present. ``web.research`` (``!arastir``, and the
mind's own looks) lived each sentence and let a moment pass, without the end being perceived, and with whoever was
present still present. This measures which of the two differences shows, on pages of two sentences about one thing:
the four combinations are run. The mind was talking with one person ("ali") when it read the pages; it is asked
afterwards by someone else.

The world is T-want's (``t_want.py``): things read about and asked "X nedir ?". Four things of the library are asked
about, then looked up; each page says two things of its thing: "X <adjective> bir <class> . X <place> bulunur ."
Measured on the four things, for each way of reading a page (the same life before it, the same pages):
  answer:   "X nedir ?" answered with exactly the first sentence, and how many answers run on into the second;
  recall:   the first two words of each of the two sentences brought to mind (``dusunce.think``, as recall_once
            does): the rest of that sentence said exactly.
Appends to ``results/page.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import t_want as W  # noqa: E402

OUT = os.path.join(HERE, "results", "page.jsonl")


def sentences_of(thing: str):
    return ["%s %s ." % (thing, W.LIBRARY[thing]), "%s %s bulunur ." % (thing, W.WHERE[thing])]


def read_page(session, mind, thing: str, ends: bool, nobody: bool) -> None:
    from zihin import IMITATED
    session.close_turn()
    if nobody:
        session.company(None)
    for sentence in sentences_of(thing):
        session.listen(sentence, IMITATED)
        if ends:
            session.end_of_turn(IMITATED, tone=session.tone(sentence))
        mind.step([])


def run(seed: int, ends: bool, nobody: bool) -> dict:
    import t_relations as R
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, Session
    from zihin.dusunce import think
    torch.manual_seed(seed)
    rng = random.Random(seed)
    mind = Mind(R.speaking_config({}))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    s.partner = "ali"
    known = list(W.KNOWN)
    rng.shuffle(known)
    watched = known[:8]
    unknown = list(W.LIBRARY)
    rng.shuffle(unknown)
    asked = unknown[:4]
    for _ in range(3):
        read = [W.fact(a) for a in known]
        rng.shuffle(read)
        for sentence in read:
            s.company(None)
            s.listen(sentence, IMITATED)
            s.end_of_turn(IMITATED, tone=s.tone(sentence))
            mind.step([])
        for a in watched:
            s.watch(W.ASK % a, W.fact(a))
    for a in asked:
        s.reply(W.ASK % a, limit=40)
    for a in asked:
        read_page(s, mind, a, ends, nobody)
    s.close_turn()
    out = {"seed": seed, "answer_exact": 0, "answer_runs_on": 0, "recall_first": 0, "recall_second": 0, "said": []}
    for a in asked:
        first, second = [x.rstrip(" .") for x in sentences_of(a)]
        said = Session(mind.clone()).reply(W.ASK % a, limit=40)
        out["said"].append(said)
        out["answer_exact"] += said == first
        out["answer_runs_on"] += said.startswith(first + " ") and len(said) > len(first)
        for key, sentence in (("recall_first", first), ("recall_second", second)):
            words = sentence.split()
            thought, _ = think(Session(mind.clone()), " ".join(words[:2]), limit=40)
            out[key] += [w for w in thought if w != "."] == words[2:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    lines, rows = ["== a page of two sentences, read in four ways: %d seeds, 4 things each ==" % args.seeds], {}
    n = 4 * args.seeds
    for name, ends, nobody in (
            ("as research read it: no sentence end perceived, the partner still present", False, False),
            ("no sentence end perceived, no one in particular present", False, True),
            ("sentence ends perceived, the partner still present", True, False),
            ("as a document is read: sentence ends perceived, no one in particular present", True, True)):
        got = [run(seed, ends, nobody) for seed in range(args.seeds)]
        rows[name] = got
        total = lambda key: sum(r[key] for r in got)   # noqa: E731
        lines.append("  %s:" % name)
        lines.append("      'X nedir ?' answered with exactly the first sentence %d / %d (runs on into the second %d); "
                     "recall of the first sentence %d / %d, of the second %d / %d" % (
                         total("answer_exact"), n, total("answer_runs_on"), total("recall_first"), n,
                         total("recall_second"), n))
        lines.append("      seed 0 said: %s" % got[0]["said"])
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"test": "t_page", "seeds": args.seeds, "rows": rows, "summary": lines},
                           ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
