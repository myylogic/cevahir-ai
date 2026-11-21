# -*- coding: utf-8 -*-
"""T-tell: telling what is known of a thing never asked about, and answering a question about one kind of fact.
A public test, generated here. It was written before F41 (Unsaid Known) was run on it: F41 was developed on T-long
(t_long.py) only, and this world differs from it in its words, its sentence shapes, the number of facts a thing has
(two to four) and its questions (three kinds, lived by the same mind).

    python research/living_mind_2026_11_15/t_tell.py [--seeds 3] [--days 3] [--set key=0|1,...]

The world (seeded). Twelve things, each with two to four facts, read in documents one sentence at a time and never
two of one thing together: its class ("gül bir çiçektir ."), where it is ("gül bahçede bulunur ."), a property
("gül kırmızıdır .") and what it does ("gül kokar ."). Three questions, each answered by people the mind watches:
  "X nasıl bir şey ?"     -> every fact of X, in any order (shuffled for each answer)
  "X nerede bulunur ?"    -> the fact of where it is
  "X ne yapar ?"          -> the fact of what it does
The twelve things are split in three groups of four (shuffled by the seed):
  A: watched in all three kinds of dialogue;
  B: watched in the second and third kind only (never asked to be told about);
  C: never in any dialogue.

Asked on fresh copies:
  tell (B and C): "X nasıl bir şey ?". Facts said whole of the facts X has, sentences left unfinished, sentences that
        are not facts of X (confabulated), and whether the answer is exactly X's facts and nothing more ("whole").
  where (C): "X nerede bulunur ?". Right (exactly the fact), another fact of X (true of it, not what was asked),
        confabulated (not a fact of X), silent or only the name. Things with no such fact are scored apart ("unknown":
        the right answer says no fact).
  does (C): "X ne yapar ?", scored the same way. A property and a deed are both sentences of two words here, so this
        asks more of the mind than "where" does.

Appends to ``results/tell.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "tell.jsonl")
THINGS = {   # thing: (class, where, property, deed); None: it has no such fact
    "gül": ("çiçektir", "bahçede", "kırmızıdır", "kokar"), "papatya": ("çiçektir", "kırda", "beyazdır", None),
    "çam": ("ağaçtır", "ormanda", "yeşildir", None), "meşe": ("ağaçtır", "ormanda", None, None),
    "çekiç": ("alettir", "atölyede", None, "çakar"), "testere": ("alettir", "atölyede", None, "keser"),
    "bıçak": ("alettir", "mutfakta", "keskindir", "keser"), "tren": ("taşıttır", "garda", None, "taşır"),
    "gemi": ("taşıttır", "limanda", "büyüktür", "yüzer"), "uçak": ("taşıttır", None, "hızlıdır", "uçar"),
    "elma": ("meyvedir", None, "kırmızıdır", None), "limon": ("meyvedir", "manavda", "sarıdır", None),
}
ASK = {"tell": "%s nasıl bir şey ?", "where": "%s nerede bulunur ?", "does": "%s ne yapar ?"}


def fact(thing: str, kind: str) -> Optional[str]:
    cls, where, prop, deed = THINGS[thing]
    if kind == "class":
        return "%s bir %s" % (thing, cls)
    if kind == "where":
        return "%s %s bulunur" % (thing, where) if where else None
    if kind == "property":
        return "%s %s" % (thing, prop) if prop else None
    return "%s %s" % (thing, deed) if deed else None


def facts(thing: str) -> List[str]:
    return [f for f in (fact(thing, k) for k in ("class", "where", "property", "does")) if f]


def sentences(said: str) -> List[str]:
    parts = [p.strip() for p in said.replace("(sessiz)", "").split(" . ") if p.strip()]
    return [p[:-2].strip() if p.endswith(" .") else p for p in parts]


def score_tell(thing: str, said: str) -> dict:
    own, parts = facts(thing), sentences(said)
    begun = lambda p: any(f.split()[:len(p.split())] == p.split() for f in own)
    right = sum(1 for f in own if f in parts)
    unfinished = sum(1 for p in parts if p not in own and begun(p))
    wrong = sum(1 for p in parts if p not in own and not begun(p))
    return {"thing": thing, "said": said, "facts": len(own), "right": right, "unfinished": unfinished, "wrong": wrong,
            "whole": right == len(own) and len(parts) == len(own)}


def score_one(thing: str, kind: str, said: str) -> dict:
    asked, own, parts = fact(thing, kind), facts(thing), sentences(said)
    if parts == [thing] or not parts:
        verdict = "silent"
    elif asked is not None and parts == [asked]:
        verdict = "right"
    elif all(p in own for p in parts):
        verdict = "another fact"
    elif asked is not None and asked in parts:
        verdict = "right and more"
    else:
        verdict = "confabulated"
    return {"thing": thing, "said": said, "known": asked is not None, "verdict": verdict}


def run(seed: int, switches: Dict[str, bool], days: int = 3) -> dict:
    import t_relations as R
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, Session
    torch.manual_seed(seed)
    rng = random.Random(seed)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    names = list(THINGS)
    rng.shuffle(names)
    group = {"A": names[:4], "B": names[4:8], "C": names[8:]}
    for _ in range(days):
        read = [f + " ." for a in names for f in facts(a)]
        rng.shuffle(read)
        dialogues = []
        for a in group["A"]:
            told = facts(a)
            rng.shuffle(told)
            dialogues.append((ASK["tell"] % a, " . ".join(told) + " ."))
        for a in group["A"] + group["B"]:
            for kind in ("where", "does"):
                if fact(a, kind):
                    dialogues.append((ASK[kind] % a, fact(a, kind) + " ."))
        rng.shuffle(dialogues)
        every = max(1, len(read) // max(1, len(dialogues)))
        for i, sentence in enumerate(read):
            s.company(None)
            s.listen(sentence, IMITATED)
            s.end_of_turn(IMITATED, tone=s.tone(sentence))
            mind.step([])
            if i % every == every - 1 and dialogues:
                s.watch(*dialogues.pop())
        for d in dialogues:
            s.watch(*d)

    def ask(kind: str, thing: str) -> str:
        return Session(mind.clone()).reply(ASK[kind] % thing, limit=60)
    tell = {g: [score_tell(a, ask("tell", a)) for a in group[g]] for g in ("B", "C")}
    one = {kind: [score_one(a, kind, ask(kind, a)) for a in group["C"]] for kind in ("where", "does")}
    return {"seed": seed, "groups": group, "tell": tell, "one": one, "moments": mind.t, "ties": mind.S.n_ties}


def main() -> None:
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--set", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    runs = [run(seed, switches, days=args.days) for seed in range(args.seeds)]
    lines = ["== telling what is known and answering one kind of fact: %d seeds, %d days, switches %s ==" % (
        args.seeds, args.days, switches or "-")]
    for g, label in (("C", "never in a dialogue"), ("B", "asked other things before")):
        rows = [r for run_ in runs for r in run_["tell"][g]]
        lines.append("tell, %-26s facts said whole %d / %d, unfinished %d, confabulated %d, whole answers %d / %d" % (
            label + ":", sum(r["right"] for r in rows), sum(r["facts"] for r in rows),
            sum(r["unfinished"] for r in rows), sum(r["wrong"] for r in rows), sum(r["whole"] for r in rows),
            len(rows)))
    for kind in ("where", "does"):
        rows = [r for run_ in runs for r in run_["one"][kind]]
        known, unknown = [r for r in rows if r["known"]], [r for r in rows if not r["known"]]
        count = lambda rs, v: sum(1 for r in rs if r["verdict"] == v)
        lines.append("%-5s (never in a dialogue): right %d / %d, right and more %d, another fact %d, confabulated %d, "
                     "silent %d; no such fact known: silent %d / %d, another fact %d, confabulated %d" % (
                         kind, count(known, "right"), len(known), count(known, "right and more"),
                         count(known, "another fact"), count(known, "confabulated"), count(known, "silent"),
                         count(unknown, "silent"), len(unknown), count(unknown, "another fact"),
                         count(unknown, "confabulated")))
    lines.append("seed 0: " + " | ".join("%s -> %s" % (r["thing"], r["said"]) for r in runs[0]["tell"]["C"][:2]))
    for line in lines:
        print(line)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seeds": args.seeds, "days": args.days, "switches": switches, "summary": lines,
                            "runs": runs}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
