# -*- coding: utf-8 -*-
"""T-purpose: does the mind act for what its acts lead to, and carry that to situations it never met? A public test.

    python research/living_mind_2026_11_15/t_purpose.py [--seeds 3] [--set key=0|1,...]

The world (seeded, public, generated here). The mind watches people being handed things ("al bu elma" -> "ver bakayım ."
twice as often as "teşekkür ederim ."): imitation alone would answer rudely. Then it is handed some things itself
(elma, kalem, top), and what it answers is received: "teşekkür ..." well, anything else badly (the chat's !dogru /
!yanlis: a value event at the end of the episode). Rounds of this, then it is handed things it was never handed itself
(kitap, çiçek, bardak), with no feedback.

Scored: the share of polite answers ("teşekkür" first) to the things it was handed during the rounds (has it learned
from what its own answers led to?) and to the new things (does what it learned carry over?). Imitation alone gives the
rude habit on both. Appends to ``results/purpose.jsonl``.

``--people``: two people hand it things in the rounds; ayşe receives the polite answer well and the banter badly,
mehmet (who jokes with it) the other way round. New things are then handed by each: does the mind answer each the way
their relationship has gone (F33 with F35)?
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

OUT = os.path.join(HERE, "results", "purpose.jsonl")
HANDED = ["elma", "kalem", "top"]
NEW = ["kitap", "çiçek", "bardak"]


def run(seed: int, switches: Dict[str, bool], rounds: int = 6) -> dict:
    import t_relations as R
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session
    torch.manual_seed(seed)
    rng = random.Random(seed)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    watched = [(x, "ver bakayım .") for x in HANDED + NEW for _ in range(2)] + \
              [(x, "teşekkür ederim .") for x in HANDED + NEW]
    rng.shuffle(watched)
    for x, a in watched:
        s.watch("al bu %s" % x, a)
    history = []
    for r in range(rounds):
        items = list(HANDED)
        rng.shuffle(items)
        for x in items:
            said = s.reply("al bu %s" % x)
            good = said.split()[:1] == ["teşekkür"]
            s.feedback(good)
            history.append((r, x, said, good))
    polite_known = sum(Session(mind.clone()).reply("al bu %s" % x).split()[:1] == ["teşekkür"] for x in HANDED)
    replies_new = [Session(mind.clone()).reply("al bu %s" % x) for x in NEW]
    polite_new = sum(r.split()[:1] == ["teşekkür"] for r in replies_new)
    return {"seed": seed, "polite_known": polite_known, "polite_new": polite_new, "n": len(HANDED),
            "first_round": [h[2] for h in history[:3]], "last_round": [h[2] for h in history[-3:]],
            "new": replies_new}


def run_people(seed: int, switches: Dict[str, bool], rounds: int = 6) -> dict:
    import t_relations as R
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session
    torch.manual_seed(seed)
    rng = random.Random(seed)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    watched = [(x, "ver bakayım .") for x in HANDED + NEW for _ in range(2)] + \
              [(x, "teşekkür ederim .") for x in HANDED + NEW for _ in range(2)]
    rng.shuffle(watched)
    for x, a in watched:
        s.watch("al bu %s" % x, a)
    likes = {"ayşe": "teşekkür", "mehmet": "ver"}
    for r in range(rounds):
        for person in rng.sample(list(likes), 2):
            s.partner = person
            for x in rng.sample(HANDED, len(HANDED)):
                said = s.reply("al bu %s" % x)
                s.feedback(said.split()[:1] == [likes[person]])
    out = {}
    for person, liked in likes.items():
        hits = []
        for x in NEW:
            twin = Session(mind.clone())
            twin.partner = person
            hits.append(twin.reply("al bu %s" % x))
        out[person] = {"liked": sum(h.split()[:1] == [liked] for h in hits), "replies": hits}
    return {"seed": seed, "people": out, "n": len(NEW)}


def main() -> None:
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--set", default="")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--people", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    if args.people:
        rows = [run_people(seed, switches) for seed in range(args.seeds)]
        n = sum(r["n"] for r in rows)
        lines = ["== two relationships, opposite expectations: %d seeds, switches %s ==" % (args.seeds, switches or "-")]
        for person in ("ayşe", "mehmet"):
            lines.append("new things handed by %s, answered the way %s receives well: %d / %d (seed 0: %s)" % (
                person, person, sum(r["people"][person]["liked"] for r in rows), n, rows[0]["people"][person]["replies"]))
        for line in lines:
            print(line)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"seeds": args.seeds, "switches": switches, "people": True, "summary": lines,
                                "rows": rows}, ensure_ascii=False) + "\n")
        return
    rows = [run(seed, switches) for seed in range(args.seeds)]
    n = sum(r["n"] for r in rows)
    lines = ["== acting for what acts lead to: %d seeds, switches %s ==" % (args.seeds, switches or "-"),
             "polite answers to the things it was handed: %d / %d" % (sum(r["polite_known"] for r in rows), n),
             "polite answers to things it was never handed: %d / %d" % (sum(r["polite_new"] for r in rows), n),
             "seed 0: first round %s; last round %s; new %s" % (rows[0]["first_round"], rows[0]["last_round"],
                                                                  rows[0]["new"])]
    for line in lines:
        print(line)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seeds": args.seeds, "switches": switches, "summary": lines, "rows": rows},
                           ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
