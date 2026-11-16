# -*- coding: utf-8 -*-
"""T-chain: conclusions two steps away, from facts read apart. A public test of multi-step deduction with exceptions.

    python research/living_mind_2026_11_15/t_chain.py [--seeds 3] [--set key=0|1,...]

The world (seeded, public, generated here). Two kinds of things, each with a property stated of the kind only
("hayvanlar solur .", "bitkiler filizlenir ."). Four classes, each stated to be of a kind, by its plural ("kuşlar bir
hayvandır .", "çiçekler bir bitkidir ."). Six members per class, each stated to be of its class only ("serçe bir kuştur
."). One class has a property of its own, nearer than its kind's ("kuşlar uçar ."), and one member of it an exception
of its own ("penguen bir kuştur .", "penguen yüzer ."). All are read as documents, in a shuffled order, never two facts
of a chain together.

Three members of each class are asked what they do, and the answer watched is what follows from the chain ("serçe ne
yapar ?" -> "serçe uçar .", "hamsi ne yapar ?" -> "hamsi solur ."). The other three members of each class are never asked
and never stated with any property. Asked "levrek ne yapar ?", the right answer ("levrek solur") follows only by putting
three facts read apart together: levrek is a fish, fish are animals, animals breathe. For the class with a property of
its own the nearer property is right ("kartal uçar"), and the exception its own ("penguen yüzer").

Scored: right (the whole answer), wrong generalisation (a property that does not hold for it: another kind's, the kind's
where its class has its own, the class's for the exception), silent. Two steps (the chain) and one step (the nearer
property) are counted apart. Appends to ``results/chain.jsonl``.
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

OUT = os.path.join(HERE, "results", "chain.jsonl")
KINDS = {   # kind: (copula form, plural form, property)
    "hayvan": ("hayvandır", "hayvanlar", "solur"),
    "bitki": ("bitkidir", "bitkiler", "filizlenir"),
}
CLASSES = {   # class: (copula form, plural form, kind, its own property or None, members)
    "kuş": ("kuştur", "kuşlar", "hayvan", "uçar", ["serçe", "güvercin", "karga", "kartal", "baykuş", "martı"]),
    "balık": ("balıktır", "balıklar", "hayvan", None, ["hamsi", "sazan", "palamut", "levrek", "alabalık", "istavrit"]),
    "çiçek": ("çiçektir", "çiçekler", "bitki", None, ["gül", "lale", "papatya", "menekşe", "nergis", "karanfil"]),
    "ağaç": ("ağaçtır", "ağaçlar", "bitki", None, ["meşe", "çam", "kavak", "söğüt", "çınar", "ladin"]),
}
EXCEPTION = ("penguen", "kuş", "yüzer")
PROPERTIES = {k[2] for k in KINDS.values()} | {c[3] for c in CLASSES.values() if c[3]} | {EXCEPTION[2]}


def holds(member_class: str, exception: bool = False) -> str:
    """The property that holds for a member of the class: its own (the exception), its class's, or its kind's."""
    if exception:
        return EXCEPTION[2]
    copula, plural, kind, own, members = CLASSES[member_class]
    return own or KINDS[kind][2]


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
    for name, (copula, plural, kind, own, members) in CLASSES.items():
        order = list(members)
        rng.shuffle(order)
        asked += [(m, name) for m in order[:3]]
        unasked += [(m, name) for m in order[3:]]
    for _ in range(days):
        sentences = []
        for kind, (copula, plural, prop) in KINDS.items():
            sentences += ["%s %s ." % (plural, prop)] * 2
        for name, (copula, plural, kind, own, members) in CLASSES.items():
            sentences += ["%s bir %s ." % (m, copula) for m in members]
            sentences += ["%s bir %s ." % (plural, KINDS[kind][0])] * 2
            if own:
                sentences += ["%s %s ." % (plural, own)] * 2
        sentences += ["%s bir %s ." % (EXCEPTION[0], CLASSES[EXCEPTION[1]][0]), "%s %s ." % (EXCEPTION[0], EXCEPTION[2])]
        rng.shuffle(sentences)
        dialogues = [("%s ne yapar ?" % m, "%s %s ." % (m, holds(c))) for m, c in asked]
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
    tally = {"two": [0, 0, 0, 0], "one": [0, 0, 0, 0], "exception": [0, 0, 0, 0]}   # right, wrong, silent, n
    replies = {}
    probes = [(m, c, False) for m, c in unasked] + [(EXCEPTION[0], EXCEPTION[1], True)]
    for m, c, exception in probes:
        said = Session(mind.clone()).reply("%s ne yapar ?" % m)
        replies[m] = said
        right = holds(c, exception)
        kind = "exception" if exception else ("one" if CLASSES[c][3] else "two")
        row = tally[kind]
        row[3] += 1
        if said == "%s %s" % (m, right):
            row[0] += 1
        elif said == "(sessiz)":
            row[2] += 1
        elif any(q in said.split() for q in PROPERTIES - {right}):
            row[1] += 1
    return {"seed": seed, "tally": tally, "replies": replies, "moments": mind.t, "ties": mind.S.n_ties}


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
    lines = ["== deduction two steps away: %d seeds, switches %s ==" % (args.seeds, switches or "-")]
    for kind, label in (("two", "two steps (member -> class -> kind)"), ("one", "one step (the nearer property)"),
                        ("exception", "the exception (its own)")):
        r = [sum(row["tally"][kind][i] for row in rows) for i in range(4)]
        lines.append("%-38s right %d / %d, wrong generalisations %d, silent %d" % (label, r[0], r[3], r[1], r[2]))
    lines.append("seed 0: %s" % ", ".join("%s -> %s" % (m, r) for m, r in list(rows[0]["replies"].items())[:9]))
    for line in lines:
        print(line)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seeds": args.seeds, "switches": switches, "summary": lines, "rows": rows},
                           ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
