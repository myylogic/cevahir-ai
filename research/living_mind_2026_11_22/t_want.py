# -*- coding: utf-8 -*-
"""T-want: does the mind want to learn what it could not answer, look it up, and answer afterwards?

    python research/living_mind_2026_11_22/t_want.py [--seeds 3] [--days 3] [--set key=0|1,...] [--form nedir|nerede]

A public test, generated here; nothing is downloaded (the "library" stands where the web stands in the program: it
is a ``zihin.web.Source``, the same interface ``!arastir`` and ``!kendini_gelistir`` read Wikipedia through).

The world (seeded). Twelve things the mind reads about, one sentence each ("gül kırmızı bir çiçektir ."), read as
documents on three days. For eight of them it watches the question being answered ("gül nedir ?" -> "gül kırmızı bir
çiçektir ."); four are read and never asked (they measure whether the question form carries over at all). Eight more
things are in the library only: the mind has never met their words.

Then, in this order:
  1. gap:      someone asks the mind "X nedir ?" about four of the library's things. It has nothing to say (measured:
               silent, or something that is not the fact).
  2. want:     the mind names what it most wants to know, by its own evidence (``web.wanted_topics``: telling and not
               yet well known), as many topics as it was asked about. Measured: how many of them are the four things
               it could not answer (the test does not tell it which).
  3. research: it looks each of them up in the library and reads what it finds (``web.research``).
  4. answer:   the four questions are asked again, on copies. Measured: exact answers.
  5. met:      the four questions are asked once more for real (lived). Measured: whether the things are still among
               what it wants most (a want that was met should be over).
Controls: the same mind without step 3 (a copy asked again: time alone teaches nothing); a copy that reads the same
number of library pages about things it was NOT asked about (reading without wanting the right thing); and the four
library things never mentioned to it (it cannot want what it never heard of, and must stay silent about them).

Appends to ``results/want.jsonl``.
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "want.jsonl")
KNOWN = {"gül": "kırmızı bir çiçektir", "papatya": "beyaz bir çiçektir", "çam": "yeşil bir ağaçtır",
         "meşe": "yaşlı bir ağaçtır", "çekiç": "ağır bir alettir", "testere": "keskin bir alettir",
         "tren": "uzun bir taşıttır", "gemi": "büyük bir taşıttır", "elma": "tatlı bir meyvedir",
         "limon": "ekşi bir meyvedir", "serçe": "küçük bir kuştur", "kartal": "güçlü bir kuştur"}
LIBRARY = {"zambak": "mor bir çiçektir", "kayın": "ince bir ağaçtır", "matkap": "gürültülü bir alettir",
           "vapur": "yavaş bir taşıttır", "armut": "sulu bir meyvedir", "leylek": "göçmen bir kuştur",
           "bakır": "kızıl bir madendir", "pamuk": "yumuşak bir bitkidir"}
ASK = "%s nedir ?"
# --form nerede: another question and another kind of fact (where a thing is), the same things. Written after F46
# and run once, as found: F46 was developed on the first form only.
WHERE = {"gül": "bahçede", "papatya": "kırda", "çam": "ormanda", "meşe": "parkta", "çekiç": "atölyede",
         "testere": "depoda", "tren": "garda", "gemi": "limanda", "elma": "sepette", "limon": "manavda",
         "serçe": "çatıda", "kartal": "dağda", "zambak": "vazoda", "kayın": "yamaçta", "matkap": "çantada",
         "vapur": "iskelede", "armut": "tabakta", "leylek": "bacada", "bakır": "madende", "pamuk": "tarlada"}
FORM = "nedir"


def fact(thing: str) -> str:
    if FORM == "nerede":
        return "%s %s bulunur ." % (thing, WHERE[thing])
    return "%s %s ." % (thing, {**KNOWN, **LIBRARY}[thing])


class Library:
    """A ``zihin.web.Source`` over the test's own pages: a topic is found when a page is about it."""

    def __init__(self, pages: Dict[str, str]) -> None:
        self.pages = pages
        self.asked: List[str] = []

    def search(self, topic: str, limit: int = 2) -> List[str]:
        self.asked.append(topic)
        return [topic] if topic in self.pages else []

    def extract(self, identifier: str) -> str:
        return self.pages[identifier]

    def address(self, identifier: str) -> str:
        return "library:" + identifier


def run(seed: int, switches: Dict[str, bool], days: int) -> dict:
    import t_relations as R
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, Session
    from zihin.web import research, wanted_topics
    torch.manual_seed(seed)
    rng = random.Random(seed)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    known = list(KNOWN)
    rng.shuffle(known)
    watched, read_only = known[:8], known[8:]
    unknown = list(LIBRARY)
    rng.shuffle(unknown)
    asked, never = unknown[:4], unknown[4:]
    for _ in range(days):
        read = [fact(a) for a in known]
        rng.shuffle(read)
        dialogues = [(ASK % a, fact(a)) for a in watched]
        rng.shuffle(dialogues)
        every = max(1, len(read) // len(dialogues))
        for i, sentence in enumerate(read):
            s.company(None)
            s.listen(sentence, IMITATED)
            s.end_of_turn(IMITATED, tone=s.tone(sentence))
            mind.step([])
            if i % every == every - 1 and dialogues:
                s.watch(*dialogues.pop())
        for d in dialogues:
            s.watch(*d)

    def answers(m, things) -> List[str]:
        return [Session(m.clone()).reply(ASK % a, limit=40) for a in things]

    def right(said: List[str], things) -> int:
        return sum(x == fact(a).rstrip(" .").strip() or x == fact(a) for x, a in zip(said, things))

    def told(said: List[str], things) -> int:
        """The fact's own words are said and nothing else: the words of the fact that the question does not hold
        (in the second form the mind does not repeat "bulunur", a word of the question; found, not designed)."""
        n = 0
        for x, a in zip(said, things):
            words, asked_words = fact(a).rstrip(" .").split(), set((ASK % a).split())
            own = [w for w in words if w not in asked_words or w == a]
            n += x.split() == own or x == " ".join(words)
        return n

    carried = answers(mind, read_only)
    gap = [s.reply(ASK % a, limit=40) for a in asked]                 # 1. the questions are really lived
    s.close_turn()                                                    # (as the chat does before any other command)
    wanted = wanted_topics(mind, CHANNEL, len(asked))                 # 2. what it wants, by its own evidence
    idle = mind.clone()                                               # control: nothing looked up
    other = mind.clone()                                              # control: reads about what it was not asked
    library = Library({a: fact(a) for a in LIBRARY})
    reports = [research(s, topic, source=library, say=lambda line: None) for topic in wanted]   # 3.
    so = Session(other)
    for topic in never:
        research(so, topic, source=Library({a: fact(a) for a in LIBRARY}), say=lambda line: None)
    after = answers(mind, asked)                                      # 4.
    lived = mind.clone()                                              # 5. asked again for real: is the want met?
    sl = Session(lived)
    for a in asked:
        sl.reply(ASK % a, limit=40)
    still = wanted_topics(lived, CHANNEL, len(asked))
    return {"seed": seed, "watched": watched, "read_only": read_only, "asked": asked, "never": never,
            "still_wanted": len(set(still) & set(asked)), "wanted_after": still,
            "carried": right(carried, read_only), "carried_told": told(carried, read_only),
            "after_told": told(after, asked), "gap_said": gap, "gap_right": right(gap, asked),
            "wanted": wanted, "wanted_are_gaps": len(set(wanted) & set(asked)),
            "found": sum(bool(r.get("read")) for r in reports),
            "after_said": after, "after_right": right(after, asked),
            "idle_right": right(answers(idle, asked), asked),
            "other_right": right(answers(other, asked), asked),
            "never_said": answers(mind, never), "moments": mind.t, "ties": mind.S.n_ties}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--set", default="")
    ap.add_argument("--form", default="nedir", choices=["nedir", "nerede"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    global FORM, ASK
    FORM = args.form
    if FORM == "nerede":
        ASK = "%s nerede bulunur ?"
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    rows = [run(seed, switches, args.days) for seed in range(args.seeds)]
    n = 4 * len(rows)
    total = lambda key: sum(r[key] for r in rows)   # noqa: E731
    lines = ["== T-want (%s): %d seeds, %d days, switches %s ==" % (ASK % "X", len(rows), args.days, switches or "-"),
             "  read and never asked (the question form carries over):  %d / %d answered exactly, %d / %d with the "
             "fact's own words" % (total("carried"), n, total("carried_told"), n),
             "  1. asked about things it never met:                     %d / %d answered (said only the name %d)" % (
                 total("gap_right"), n, sum(x == a for r in rows for x, a in zip(r["gap_said"], r["asked"]))),
             "  2. of the %d topics it then wants most, by its own evidence: %d are the things it could not answer" % (
                 n, total("wanted_are_gaps")),
             "  3. looked up in the library: %d / %d found" % (total("found"), n),
             "  4. asked again after reading what it wanted:            %d / %d answered exactly, %d / %d with the "
             "fact's own words" % (total("after_right"), n, total("after_told"), n),
             "  control, nothing looked up:                             %d / %d" % (total("idle_right"), n),
             "  control, read as much about things it was not asked:    %d / %d" % (total("other_right"), n),
             "  5. asked once more, for real: of the things it wanted, still among what it wants most: %d / %d" % (
                 total("still_wanted"), n),
             "  things never mentioned to it: answered %d / %d (said only the name %d)" % (
                 sum(x == fact(a).rstrip(" .").strip() for r in rows for x, a in zip(r["never_said"], r["never"])), n,
                 sum(x == a for r in rows for x, a in zip(r["never_said"], r["never"])))]
    for r in rows:
        lines.append("  seed %d: asked %s; wanted %s; before %s; after %s" % (
            r["seed"], r["asked"], r["wanted"], r["gap_said"], r["after_said"]))
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"test": "t_want", "form": ASK % "X", "switches": switches, "days": args.days, "rows": rows,
                            "summary": lines}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
