# -*- coding: utf-8 -*-
"""T-trust: does the mind learn whose word holds? One person tells it what turns out true, another what turns out
false; then each tells it something it cannot check. A public baseline.

    python research/living_mind_2026_11_15/t_trust.py [--seeds 3] [--days 3] [--set company=1]

The world (seeded, public, generated here). Two people talk with the mind (the chat's partner, as with --kim / !kim):

* zeynep tells it facts about animals that its own experience later confirms ("kuşlar uçar ." told, then sensed);
* mehmet tells it facts that its own experience later contradicts ("balıklar uçar ." told, then "balıklar yüzer ."
  sensed);
* at the end each tells it one new fact it never checks: zeynep "kurtlar uluyor .", mehmet "atlar kişner .".

Everything is lived through the chat's own Session (live, with the partner set; what is sensed is lived with no one
in particular) with the chat's speaking configuration and ``--set`` switches for the whole life.

Read at the end (on the raised mind, nothing lived): how firmly the mind holds each new fact, the expectation of its
verb after its subject, with its teller present, with the other person present and with a stranger present; and the
trust in told speech, per person where the mind keeps it (F33) and for told speech as a whole. Prints the summary and
appends it, with the switches, to ``results/trust.jsonl`` (a public world: only this world's words and numbers).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

from zihin import SENSED, TOLD, Mind  # noqa: E402
from zihin.__main__ import CHANNEL, SOMEONE, Session  # noqa: E402

OUT = os.path.join(HERE, "results", "trust.jsonl")

FACTS = [("kuşlar", "uçar", "yüzer"), ("balıklar", "yüzer", "uçar"), ("kediler", "miyavlar", "havlar"),
         ("köpekler", "havlar", "miyavlar"), ("inekler", "möler", "kükrer"), ("aslanlar", "kükrer", "möler"),
         ("arılar", "vızıldar", "öter"), ("horozlar", "öter", "vızıldar")]
TELLERS = {"zeynep": True, "mehmet": False}      # who tells the truth
NEW = {"zeynep": ("kurtlar", "uluyor"), "mehmet": ("atlar", "kişner")}


def raise_mind(seed: int, days: int, switches: Dict[str, bool]) -> Session:
    import t_relations as R                       # the chat's speaking configuration with the switches
    torch.manual_seed(seed)
    rng = random.Random(seed)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for _ in range(days):
        facts = list(FACTS)
        rng.shuffle(facts)
        for i, (subject, true, false) in enumerate(facts):
            teller = "zeynep" if i % 2 == 0 else "mehmet"
            s.partner = teller
            s.live("%s %s ." % (subject, true if TELLERS[teller] else false), TOLD)
        for subject, true, _ in facts:              # its own experience, later, with no one in particular
            s.partner = SOMEONE
            s.live("%s %s ." % (subject, true), SENSED)
    for teller, (subject, verb) in NEW.items():
        s.partner = teller
        s.live("%s %s ." % (subject, verb), TOLD)
    s.partner = SOMEONE
    s.company(None)
    return s


def held(s: Session, present, subject: str, verb: str) -> float:
    from zihin.evidence import expectation
    from zihin.read import own_mass
    mind = s.mind
    mind.company = present if getattr(mind.cfg, "company", False) else None
    try:
        w = own_mass(mind, mind.unit_of(CHANNEL, subject)).get(mind.unit_of(CHANNEL, verb))
    finally:
        mind.company = None
    return expectation(w[0], w[1], mind.cfg.k) if w else 0.5


def run(seed: int, days: int, switches: Dict[str, bool]) -> dict:
    s = raise_mind(seed, days, switches)
    mind = s.mind
    row = {"seed": seed, "trust_told": round(float(mind.trust[TOLD]), 3)}
    if "person_trust" in mind.__dict__:
        from zihin.iliski import person_trust
        for person in TELLERS:
            row["trust_" + person] = round(float(person_trust(mind, person, None)[TOLD]), 3)
    for teller, (subject, verb) in NEW.items():
        other = next(p for p in TELLERS if p != teller)
        row[teller] = {"teller": round(held(s, teller, subject, verb), 3),
                       "other": round(held(s, other, subject, verb), 3),
                       "stranger": round(held(s, "deniz", subject, verb), 3)}
    return row


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
    rows = [run(seed, args.days, switches) for seed in range(args.seeds)]

    def mean(f):
        vals = [f(r) for r in rows if f(r) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    lines = ["== whose word holds: %d seeds, %d days, switches %s ==" % (args.seeds, args.days, switches or "-"),
             "trust in told speech, as a whole: %s" % mean(lambda r: r["trust_told"])]
    for person in TELLERS:
        lines.append("trust in %s's word: %s" % (person, mean(lambda r, p=person: r.get("trust_" + p))))
    lines.append("how firmly a new fact is held (expectation of its verb after its subject):")
    for teller, (subject, verb) in NEW.items():
        lines.append("  %-7s %-16s with the teller %s, with the other %s, with a stranger %s" % (
            teller, "%s %s" % (subject, verb), mean(lambda r, t=teller: r[t]["teller"]),
            mean(lambda r, t=teller: r[t]["other"]), mean(lambda r, t=teller: r[t]["stranger"])))
    for line in lines:
        print(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seeds": args.seeds, "days": args.days, "switches": switches, "summary": lines,
                            "rows": rows}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
