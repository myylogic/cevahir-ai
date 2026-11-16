# -*- coding: utf-8 -*-
"""T-long: answers of several sentences about a thing never asked, from facts read apart. A public test.

    python research/living_mind_2026_11_15/t_long.py [--seeds 3] [--set key=0|1,...]

The world (seeded, public, generated here). Twelve animals, each with three facts stated in documents, one sentence at
a time and never two of one animal together: its class ("kartal bir kuştur ."), what it does ("kartal uçar ."), and what
it eats ("kartal et yer ."). Six animals are asked about, and the answer watched is all three facts, in that order:
"kartal hakkında ne biliyorsun ?" -> "kartal bir kuştur . kartal uçar . kartal et yer .". The other six are never asked.

Asked about one of those, a mind that answers with one sentence and stops has said what one recalled fact allows; a
long answer says the three facts read apart about it, one after another, and nothing it did not read about it.

``--varied``: the people who answer give the three facts in any order (shuffled for each answer).

Scored per answer: the facts said whole (of 3), sentences begun as one of its facts and left unfinished ("baykuş bir
kuştur . baykuş"), sentences that are not this animal's facts (a confabulation: another animal's fact, or a sentence
about this one that is not true of it), the number of sentences, silent. Appends to ``results/long.jsonl``.
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

OUT = os.path.join(HERE, "results", "long.jsonl")
ANIMALS = {   # animal: (class, what it does, what it eats)
    "kartal": ("kuştur", "uçar", "et"), "serçe": ("kuştur", "uçar", "tohum"), "güvercin": ("kuştur", "uçar", "tohum"),
    "baykuş": ("kuştur", "uçar", "fare"), "hamsi": ("balıktır", "yüzer", "plankton"), "levrek": ("balıktır", "yüzer",
    "balık"), "sazan": ("balıktır", "yüzer", "ot"), "inek": ("memelidir", "otlar", "ot"), "koyun": ("memelidir",
    "otlar", "ot"), "aslan": ("memelidir", "avlanır", "et"), "kurt": ("memelidir", "avlanır", "et"),
    "yılan": ("sürüngendir", "sürünür", "fare"),
}


def facts(animal: str):
    cls, does, eats = ANIMALS[animal]
    return ["%s bir %s" % (animal, cls), "%s %s" % (animal, does), "%s %s yer" % (animal, eats)]


def run(seed: int, switches: Dict[str, bool], days: int = 3, varied: bool = False) -> dict:
    import t_relations as R
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, Session
    torch.manual_seed(seed)
    rng = random.Random(seed)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    names = list(ANIMALS)
    rng.shuffle(names)
    asked, unasked = names[:6], names[6:]
    for _ in range(days):
        sentences = [f + " ." for a in names for f in facts(a)]
        rng.shuffle(sentences)
        dialogues = []
        for a in asked:                                  # the three facts in the given order, or in any order
            told = facts(a)
            if varied:
                rng.shuffle(told)
            dialogues.append(("%s hakkında ne biliyorsun ?" % a, " . ".join(told) + " ."))
        rng.shuffle(dialogues)
        for i, sentence in enumerate(sentences):
            s.company(None)
            s.listen(sentence, IMITATED)
            s.end_of_turn(IMITATED, tone=s.tone(sentence))
            mind.step([])
            if i % 6 == 5 and dialogues:
                s.watch(*dialogues.pop())
        for d in dialogues:
            s.watch(*d)
    rows = []
    for a in unasked:
        said = Session(mind.clone()).reply("%s hakkında ne biliyorsun ?" % a, limit=40)
        parts = [p.strip() for p in said.replace("(sessiz)", "").split(" . ") if p.strip()]
        parts = [p[:-2].strip() if p.endswith(" .") else p for p in parts]
        own = set(facts(a))
        others = {f for b in ANIMALS if b != a for f in facts(b)}
        right = sum(1 for f in own if f in parts)
        begun = lambda p: any(f.split()[:len(p.split())] == p.split() for f in own)
        unfinished = sum(1 for p in parts if p not in own and begun(p))
        wrong = sum(1 for p in parts if p not in own and not begun(p) and (p in others or p.split()[:1] == [a]))
        rows.append({"animal": a, "said": said, "right": right, "unfinished": unfinished, "wrong": wrong,
                     "sentences": len(parts), "silent": said == "(sessiz)"})
    return {"seed": seed, "rows": rows, "moments": mind.t, "ties": mind.S.n_ties}


def main() -> None:
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--set", default="")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--varied", action="store_true", help="the facts of each watched answer in a shuffled order")
    args = ap.parse_args()
    torch.set_num_threads(1)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    runs = [run(seed, switches, varied=args.varied) for seed in range(args.seeds)]
    rows = [r for run_ in runs for r in run_["rows"]]
    n = len(rows)
    lines = ["== long answers about things never asked: %d seeds, switches %s%s ==" % (
                 args.seeds, switches or "-", ", facts told in any order" if args.varied else ""),
             "facts said whole %d / %d, unfinished %d, confabulated %d, sentences per answer %.2f, silent %d / %d" % (
                 sum(r["right"] for r in rows), 3 * n, sum(r["unfinished"] for r in rows),
                 sum(r["wrong"] for r in rows), sum(r["sentences"] for r in rows) / max(1, n),
                 sum(r["silent"] for r in rows), n),
             "seed 0: " + " | ".join("%s -> %s" % (r["animal"], r["said"]) for r in runs[0]["rows"][:3])]
    for line in lines:
        print(line)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seeds": args.seeds, "switches": switches, "varied": args.varied, "summary": lines,
                            "runs": runs},
                           ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
