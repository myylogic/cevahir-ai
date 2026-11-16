# -*- coding: utf-8 -*-
"""T-deduce: conclusions never taught, from facts read apart. A public test of deduction with exceptions.

    python research/living_mind_2026_11_15/t_deduce.py [--seeds 3] [--set key=0|1,...]

The world (seeded, public, generated here). Five classes of animals, each with a property stated of the class only
("kuşlar uçar .", "balıklar yüzer .", "böcekler vızıldar .", "memeliler emzirir .", "sürüngenler sürünür ."), and six
members each, stated to be members ("kartal bir kuştur .", "hamsi bir balıktır ."). One member is an exception, stated
of itself: "penguen bir kuştur ." and "penguen yüzer .". These are read as documents, in a shuffled order, never two
facts of one chain together.

Three members of each class are asked what they do, and the answer watched is their class's property ("serçe ne yapar
?" -> "serçe uçar ."). The other three members of each class are never asked and never stated with the property: asked
"kartal ne yapar ?", the right answer ("kartal uçar") follows only by putting two facts read apart together. The
exception is asked too ("penguen ne yapar ?": its own fact, "penguen yüzer", against its class's "uçar").

Scored: right (the whole answer), wrong generalisation (another class's property, or the class's property for the
exception), silent. Appends to ``results/deduce.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "deduce.jsonl")
CLASSES = {   # class: (copula form, plural form, property, members)
    "kuş": ("kuştur", "kuşlar", "uçar", ["serçe", "güvercin", "karga", "kartal", "baykuş", "martı"]),
    "balık": ("balıktır", "balıklar", "yüzer", ["hamsi", "sazan", "palamut", "levrek", "alabalık", "istavrit"]),
    "böcek": ("böcektir", "böcekler", "vızıldar", ["arı", "sinek", "eşekarısı", "sivrisinek", "yabanarısı", "bombus"]),
    "memeli": ("memelidir", "memeliler", "emzirir", ["kedi", "köpek", "inek", "at", "aslan", "fil"]),
    "sürüngen": ("sürüngendir", "sürüngenler", "sürünür", ["yılan", "kertenkele", "timsah", "engerek", "iguana",
                                                           "bukalemun"]),
}
EXCEPTION = ("penguen", "kuş", "yüzer")
PROPERTIES = {c[2] for c in CLASSES.values()}


def run(seed: int, switches: Dict[str, bool], days: int = 3) -> dict:
    import t_relations as R
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, Session
    torch.manual_seed(seed)
    rng = random.Random(seed)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    asked, unasked = [], []
    for name, (copula, plural, prop, members) in CLASSES.items():
        order = list(members)
        rng.shuffle(order)
        asked += [(m, prop) for m in order[:3]]
        unasked += [(m, prop) for m in order[3:]]
    for _ in range(days):
        sentences = []
        for name, (copula, plural, prop, members) in CLASSES.items():
            sentences += ["%s bir %s ." % (m, copula) for m in members]
            sentences += ["%s %s ." % (plural, prop)] * 2
        sentences += ["%s bir %s ." % (EXCEPTION[0], CLASSES[EXCEPTION[1]][0]), "%s %s ." % (EXCEPTION[0], EXCEPTION[2])]
        rng.shuffle(sentences)
        dialogues = [("%s ne yapar ?" % m, "%s %s ." % (m, p)) for m, p in asked]
        rng.shuffle(dialogues)
        for i, sentence in enumerate(sentences):
            s.company(None)
            s.listen(sentence, IMITATED)
            s.end_of_turn(IMITATED, tone=s.tone(sentence))
            mind.step([])
            if i % 3 == 2 and dialogues:
                s.watch(*dialogues.pop())
        for d in dialogues:
            s.watch(*d)
    right = wrong = silent = 0
    replies = {}
    for m, p in unasked + [(EXCEPTION[0], EXCEPTION[2])]:
        said = Session(mind.clone()).reply("%s ne yapar ?" % m)
        replies[m] = said
        if said == "%s %s" % (m, p):
            right += 1
        elif said == "(sessiz)":
            silent += 1
        elif any(q in said.split() for q in PROPERTIES - {p}) or (m == EXCEPTION[0] and "uçar" in said.split()):
            wrong += 1
    exception_right = replies[EXCEPTION[0]] == "%s %s" % EXCEPTION[0::2]
    return {"seed": seed, "right": right, "wrong": wrong, "silent": silent, "n": len(unasked) + 1,
            "exception": replies[EXCEPTION[0]], "exception_right": exception_right, "replies": replies,
            "moments": mind.t, "ties": mind.S.n_ties}


def main() -> None:
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--set", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    rows = [run(seed, switches) for seed in range(args.seeds)]
    n = sum(r["n"] for r in rows)
    lines = ["== deduction from facts read apart: %d seeds, switches %s ==" % (args.seeds, switches or "-"),
             "right %d / %d, wrong generalisations %d, silent %d; the exception (penguen) right in %d of %d" % (
                 sum(r["right"] for r in rows), n, sum(r["wrong"] for r in rows), sum(r["silent"] for r in rows),
                 sum(r["exception_right"] for r in rows), len(rows)),
             "seed 0: %s" % ", ".join("%s -> %s" % (m, r) for m, r in list(rows[0]["replies"].items())[:8])]
    for line in lines:
        print(line)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seeds": args.seeds, "switches": switches, "summary": lines, "rows": rows},
                           ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
