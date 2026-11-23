# -*- coding: utf-8 -*-
"""t_subject.py: F72 The Subject Carries (``Config.subject_carries``), measured (section 36).

    python research/living_mind_2026_11_22/t_subject.py [--seeds 5]

The world: 24 animals; of each the mind reads what it eats and where it lives ("X V yer .", "X P yaşar ."), twice.
It watches both questions asked of 6 of them ("X ne yer ?" -> "X V yer .", "X nerede yaşar ?" -> "X P yaşar .").
Then, on copies, of 8 animals never asked about:

* ``carried``: "X nerede yaşar ?", then "ne yer ?" (the subject dropped, as Turkish lets one drop it); and the other
  order. The second answer is scored: right (it names the animal's food), wrong (another food), silent.
* ``first``: "ne yer ?" with nothing before it: there is no thing to carry; it should say nothing it does not know.
* ``named``: "X nerede yaşar ?", then "Y ne yer ?" about another animal: the named one is answered, not the carried.
* ``direct``: "X ne yer ?" asked whole: what the mind knows when the subject is said.
* ``after silence``: a question it cannot answer ("X ne içer ?", never read) and then "ne yer ?": the answer held to
  nothing, so nothing is carried.

``--heard`` (F72 while listening, ``Config.subject_heard``): of 12 animals the mind reads both facts whole, twice,
and watches "X ne yer ?" of 6; of the 12 others it reads, once, "X P yaşar ." and then "V yer ." with the subject
dropped. Then it is asked "X ne yer ?" of those 12 (and of the 12 read whole: nothing may change there).

Every mind is the whole mind (``VARIANTS["full"]``) with F72 on or off. Appends to ``results/subject.jsonl``.
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
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "subject.jsonl")
THINGS = ["inek", "koyun", "keçi", "at", "eşek", "deve", "aslan", "kurt", "fok", "serçe", "kaz", "yunus", "tilki",
          "ayı", "balina", "kartal", "panter", "somon", "leylek", "geyik", "vaşak", "levrek", "martı", "manda"]
FOOD = ["ot", "et", "balık", "tohum"]
HOME = ["ormanda", "çölde", "denizde", "dağda"]


def heard_life(seed: int, on: bool) -> dict:
    import zihin.__main__ as chat
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, Session
    rng = random.Random(seed)
    things = list(THINGS)
    rng.shuffle(things)
    whole, dropped = things[:12], things[12:]
    food = {x: FOOD[i % len(FOOD)] for i, x in enumerate(whole)}       # (every food is read whole of 3 animals)
    food.update({x: rng.choice(FOOD) for x in dropped})
    home = {x: rng.choice(HOME) for x in things}
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = chat.VARIANTS["full"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    cfg.subject_heard = on
    torch.manual_seed(seed)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)

    def tell(sentence):
        s.company(None)
        s.listen(sentence, IMITATED)
        s.end_of_turn(IMITATED, tone=".")
        mind.step([])
    for _ in range(2):
        for x in whole:
            tell("%s %s yaşar ." % (x, home[x]))
            tell("%s %s yer ." % (x, food[x]))
        for x in whole[:6]:
            s.watch("%s ne yer ?" % x, "%s %s yer ." % (x, food[x]))
    t0 = time.process_time()
    for x in dropped:
        tell("%s %s yaşar ." % (x, home[x]))
        tell("%s yer ." % food[x])
    took = time.process_time() - t0
    s.close_turn()
    marks = {}
    for key, group in (("dropped", dropped), ("read whole", whole[6:])):
        marks[key] = "".join(mark(Session(mind.clone()).reply("%s ne yer ?" % x, limit=20), food[x], FOOD)
                             for x in group)
    return {"marks": marks, "carried_cpu_s": round(took, 2)}


def raised(seed: int, on: bool):
    import zihin.__main__ as chat
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, Session
    rng = random.Random(seed)
    things = list(THINGS)
    rng.shuffle(things)
    food = {x: rng.choice(FOOD) for x in things}
    home = {x: rng.choice(HOME) for x in things}
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = chat.VARIANTS["full"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    cfg.subject_carries = on
    torch.manual_seed(seed)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)

    def tell(sentence):
        s.company(None)
        s.listen(sentence, IMITATED)
        s.end_of_turn(IMITATED, tone=".")
        mind.step([])
    for _ in range(2):
        for x in things:
            tell("%s %s yer ." % (x, food[x]))
            tell("%s %s yaşar ." % (x, home[x]))
        for x in things[:6]:
            s.watch("%s ne yer ?" % x, "%s %s yer ." % (x, food[x]))
            s.watch("%s nerede yaşar ?" % x, "%s %s yaşar ." % (x, home[x]))
    s.close_turn()
    return s, things, food, home


def mark(said: str, right: str, values) -> str:
    words = said.split() if said != "(sessiz)" else []
    if right in words:
        return "R"
    return "W" if any(v in words for v in values) else "-"


def life(seed: int, on: bool) -> dict:
    from zihin.__main__ import Session
    s, things, food, home = raised(seed, on)

    def copy():
        c = Session(s.mind.clone())
        c.partner, c.dil = s.partner, s.dil
        return c
    marks = {"carried food": "", "carried home": "", "first": "", "named": "", "direct": "", "after silence": ""}
    took = 0.0
    fresh = things[8:16]
    for i, x in enumerate(fresh):
        y = fresh[(i + 1) % len(fresh)]
        c = copy()
        c.reply("%s nerede yaşar ?" % x, limit=20)
        t0 = time.process_time()
        marks["carried food"] += mark(c.reply("ne yer ?", limit=20), food[x], FOOD)
        took += time.process_time() - t0
        c = copy()
        c.reply("%s ne yer ?" % x, limit=20)
        marks["carried home"] += mark(c.reply("nerede yaşar ?", limit=20), home[x], HOME)
        marks["first"] += mark(copy().reply("ne yer ?", limit=20), food[x], FOOD).replace("R", "W")
        c = copy()
        c.reply("%s nerede yaşar ?" % x, limit=20)
        marks["named"] += mark(c.reply("%s ne yer ?" % y, limit=20), food[y], FOOD)
        marks["direct"] += mark(copy().reply("%s ne yer ?" % x, limit=20), food[x], FOOD)
        c = copy()
        c.reply("%s ne içer ?" % x, limit=20)
        marks["after silence"] += mark(c.reply("ne yer ?", limit=20), food[x], FOOD)
    return {"marks": marks, "carried_cpu_s": round(took, 2)}


def count(marks: str) -> dict:
    return {"right": marks.count("R"), "wrong": marks.count("W"), "silent": marks.count("-")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--heard", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.heard:
        head = ("== a sentence read with its subject dropped, after one about the thing; asked \"X ne yer ?\" "
                "(right / wrong / silent); %d seeds ==" % args.seeds)
        conditions = (("the whole mind without F72 while listening", False), ("F72 while listening", True))
    else:
        head = ("== a question whose subject is dropped, after an answer about a thing (8 animals never asked "
                "about, on copies; right / wrong / silent); %d seeds ==" % args.seeds)
        conditions = (("the whole mind without F72", False), ("F72: the subject carries", True))
    rows, lines = [], [head]
    for label, on in conditions:
        total, first, took = {}, None, 0.0
        for seed in range(args.seeds):
            got = (heard_life if args.heard else life)(seed, on)
            took += got["carried_cpu_s"]
            for key, m in got["marks"].items():
                c = total.setdefault(key, {"right": 0, "wrong": 0, "silent": 0})
                for k, n in count(m).items():
                    c[k] += n
            first = first or got["marks"]
        rows.append({"test": "subject_heard" if args.heard else "subject", "condition": label, "on": on,
                     "seeds": args.seeds, "answers": total,
                     "carried_cpu_s": round(took, 1), "first": first})
        lines.append("  %s" % label)
        for key, c in total.items():
            lines.append("    %-16s %d/%d/%d   the first seed: %s" % (key, c["right"], c["wrong"], c["silent"],
                                                                     first[key]))
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for row in rows:
            row["summary"] = lines
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
