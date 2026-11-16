# -*- coding: utf-8 -*-
"""T-relations: one mind, several people. Does it answer each person from their own history, keep what it lived with
one person from the others, and still answer everyone from what everyone shares? A public baseline.

    python research/living_mind_2026_11_15/t_relations.py [--seeds 3] [--days 3] [--set company=1]

The world (seeded, public, generated here). Four people talk with the mind, each in their own conversations (the chat's
partner: who speaks is perceived, as with the program's --kim / !kim), and two strangers appear only in the probes:

* each person is greeted in their own way ("merhaba" -> "merhaba canım , nasılsın ?" for one, "merhaba kardeşim ." for
  another, ...), and has their own answers to their own questions ("en sevdiğim renk ne ?" -> "en sevdiğin renk mavi .",
  "kedimin adı ne ?" -> "kedinin adı pamuk ."): what was lived with that person, nobody else's;
* one person alone has a secret ("sırrım ne ?" -> "sırrın mavi kutuda ."): what was lived with one person only, which
  a mind that does not tell people apart has no reason to keep from anyone;
* every person asks the same questions about the mind and gets the same answers ("adın ne ?" -> "adım cevahir .",
  "nasılsın ?" -> "iyiyim , teşekkür ederim ."): what everyone shares;
* between conversations the mind watches dialogues of someone it does not know ("su nedir ?" -> "su bir sıvıdır .",
  "merhaba" -> "merhaba , hoş geldin .") and reads sentences: what it learned from no one in particular.

Everything is lived through the chat's own Session (watch with the asker perceived, listen) with the chat's speaking
configuration (ledger off, which changes no evidence), and ``--set`` switches for the whole life.

Probes (each on a fresh clone of the raised mind, through Session.reply as in the chat, with the partner set): every
person and every stranger says "merhaba", asks "en sevdiğim renk ne ?", "kedimin adı ne ?", "adın ne ?" and "su nedir ?".
Scored by key words (the answer key is this world's, for scoring only; the mind is never given it):

* own: a known person gets their own greeting, colour, cat's name;
* other's: a known person gets someone else's (wrong, and another person's life told);
* leak: a stranger gets any known person's greeting word, colour or cat's name;
* shared: "adın ne ?" answered with the name, by known people and strangers;
* general: "su nedir ?" answered from what was watched of no one in particular, by known people and strangers.

Prints the summary and appends it, with the switches, to ``results/relations.jsonl`` (a public world: only this
world's words and scores).

``--change``: a relationship that changes. One person greets the mind "merhaba hocam ." for four days and then
"merhaba canım ." (two greetings a day, among two of someone unknown); her greeting is probed at the end of every day
from the fourth.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

from zihin import IMITATED, Mind  # noqa: E402
from zihin.__main__ import CHANNEL, SOMEONE, VARIANTS, Session, words  # noqa: E402

OUT = os.path.join(HERE, "results", "relations.jsonl")

PEOPLE = {   # name: (greeting reply, its key word, favourite colour, cat's name)
    "ayşe": ("merhaba canım , nasılsın ?", "canım", "mavi", "pamuk"),
    "mehmet": ("merhaba kardeşim .", "kardeşim", "kırmızı", "karabaş"),
    "zeynep": ("merhaba hocam .", "hocam", "yeşil", "tekir"),
    "can": ("merhaba dostum !", "dostum", "sarı", "boncuk"),
}
STRANGERS = ["deniz", "ece"]
SECRET = ("ayşe", "sırrım ne ?", "sırrın mavi kutuda .", "kutuda")      # one person's alone
PERSONAL = [("en sevdiğim renk ne ?", "en sevdiğin renk {colour} ."), ("kedimin adı ne ?", "kedinin adı {cat} .")]
SHARED = [("adın ne ?", "adım cevahir ."), ("nasılsın ?", "iyiyim , teşekkür ederim .")]
GENERAL = [("su nedir ?", "su bir sıvıdır ."), ("güneş nedir ?", "güneş bir yıldızdır ."),
           ("kedi nedir ?", "kedi bir hayvandır ."), ("elma nedir ?", "elma bir meyvedir ."),
           ("merhaba", "merhaba , hoş geldin ."), ("merhaba", "merhaba , hoş geldin .")]
SENTENCES = ["su bir sıvıdır .", "güneş bir yıldızdır .", "kedi bir hayvandır .", "elma bir meyvedir .",
             "kuşlar sabah öter .", "çocuklar parkta oynuyor .", "yağmur yağınca yerler ıslanır .",
             "annem çorba pişirdi .", "kitaplar rafta duruyor .", "güneş doğudan doğar ."]
PROBES = ["merhaba", "en sevdiğim renk ne ?", "kedimin adı ne ?", "sırrım ne ?", "adın ne ?", "su nedir ?"]


def world(seed: int, days: int) -> List[tuple]:
    """The life of one seed: ("watch", utterance, answer, asker) and ("read", sentence) in order."""
    rng = random.Random(seed)
    life: List[tuple] = []
    for day in range(days):
        conversations = []
        for name, (greeting, _, colour, cat) in PEOPLE.items():
            exchanges = [(q, a.format(colour=colour, cat=cat)) for q, a in PERSONAL] + list(SHARED)
            if name == SECRET[0]:
                exchanges.append(SECRET[1:3])
            rng.shuffle(exchanges)
            conversations.append([("merhaba", greeting)] + exchanges + [(name,)])
        general = list(GENERAL)
        rng.shuffle(general)
        conversations += [[exchange, (SOMEONE,)] for exchange in general]
        rng.shuffle(conversations)
        sentences = list(SENTENCES)
        rng.shuffle(sentences)
        for i, conversation in enumerate(conversations):
            asker = conversation[-1][0]
            life += [("watch", q, a, asker) for q, a in conversation[:-1]]
            if i % 3 == 2 and sentences:
                life.append(("read", sentences.pop()))
        life += [("read", s) for s in sentences]
    return life


def speaking_config(switches: Dict[str, bool]):
    import zihin.__main__ as chat
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for k, v in switches.items():
        setattr(cfg, k, v)
    return cfg


def raise_mind(seed: int, days: int, switches: Dict[str, bool]) -> Tuple[Session, dict]:
    torch.manual_seed(seed)
    mind = Mind(speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    started = time.time()
    for item in world(seed, days):
        if item[0] == "watch":
            session.watch(item[1], item[2], asker=item[3])
        else:                                            # a sentence of a document, as !oku reads it
            session.company(None)                        # (a document is no one's in particular)
            session.listen(item[1], IMITATED)
            session.end_of_turn(IMITATED, tone=session.tone(item[1]))
            mind.step([])
    took = time.time() - started
    return session, {"moments": mind.t, "ties": mind.S.n_ties, "seconds": round(took, 1)}


def probe(session: Session, partner: str, text: str) -> str:
    twin = session.mind.clone()
    asked = Session(twin, session.pieces)
    asked.partner = partner
    return asked.reply(text)


def score(partner: str, text: str, reply: str) -> Dict[str, int]:
    said = set(words(reply)) if reply != "(sessiz)" else set()
    known = partner in PEOPLE
    out: Dict[str, int] = {"silent": int(not said)}
    if text == "merhaba":
        keys = {name: PEOPLE[name][1] for name in PEOPLE}
    elif text == "en sevdiğim renk ne ?":
        keys = {name: PEOPLE[name][2] for name in PEOPLE}
    elif text == "kedimin adı ne ?":
        keys = {name: PEOPLE[name][3] for name in PEOPLE}
    elif text == SECRET[1]:
        keys = {SECRET[0]: SECRET[3]}
    else:
        keys = None
    if keys is not None:
        others = {k for name, k in keys.items() if name != partner}
        if known and partner in keys:
            out["own"] = int(keys[partner] in said)
            out["others"] = int(bool(others & said))
        elif known:                                      # a known person without one: anyone's is another's life
            out["others"] = int(bool(others & said))
        else:
            out["leak"] = int(bool(set(keys.values()) & said))
        if text == "merhaba":
            out["general"] = int("hoş" in said)
    elif text == "adın ne ?":
        out["shared"] = int("cevahir" in said)
    else:
        out["general"] = int("sıvıdır" in said)
    return out


def run(seed: int, days: int, switches: Dict[str, bool]) -> List[dict]:
    session, info = raise_mind(seed, days, switches)
    rows = []
    for partner in list(PEOPLE) + STRANGERS:
        for text in PROBES:
            reply = probe(session, partner, text)
            rows.append({"seed": seed, "partner": partner, "known": partner in PEOPLE, "text": text, "reply": reply,
                         **score(partner, text, reply), **info})
    return rows


def summary(rows: List[dict]) -> List[str]:
    lines = ["%-24s %-8s %6s %6s %6s %6s %6s %6s   %s" % ("probe", "who", "own", "other", "leak", "shared", "general",
                                                         "silent", "reply (seed 0, first of its kind)")]
    for text in PROBES:
        for known in (True, False):
            sel = [r for r in rows if r["text"] == text and r["known"] == known]
            if not sel:
                continue

            def m(key):
                vals = [r[key] for r in sel if key in r]
                return "%6.2f" % (sum(vals) / len(vals)) if vals else "     -"
            first = next((r for r in sel if r["seed"] == 0), sel[0])
            lines.append("%-24s %-8s %s %s %s %s %s %s   %s: %s" % (text, "known" if known else "stranger", m("own"),
                                                                     m("others"), m("leak"), m("shared"), m("general"),
                                                                     m("silent"), first["partner"], first["reply"]))
    return lines


def change(seed: int, switches: Dict[str, bool], days: int = 8, turn: int = 4) -> List[str]:
    """The reply to "merhaba" from the person whose greeting changed, at the end of each day from ``turn``."""
    torch.manual_seed(seed)
    mind = Mind(speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    replies = []
    for day in range(days):
        style = "merhaba hocam ." if day < turn else "merhaba canım ."
        for _ in range(2):
            s.watch("merhaba", "merhaba , hoş geldin .")
            s.watch("merhaba", style, asker="zeynep")
        if day >= turn - 1:
            replies.append("day %d: %s" % (day + 1, probe(s, "zeynep", "merhaba")))
    return replies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--set", default="", help="switches for the whole life, e.g. company=1")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--change", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    if args.change:
        lines = ["== a greeting that changed after four days: switches %s ==" % (switches or "-")]
        lines += ["seed %d: %s" % (seed, " | ".join(change(seed, switches))) for seed in range(args.seeds)]
        for line in lines:
            print(line)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"seeds": args.seeds, "switches": switches, "change": lines}, ensure_ascii=False) + "\n")
        return
    rows = []
    for seed in range(args.seeds):
        rows += run(seed, args.days, switches)
    lines = summary(rows)
    print("== one mind, four people, two strangers: %d seeds, %d days, switches %s; %d moments, %d ties (seed 0) ==" % (
        args.seeds, args.days, switches or "-", rows[0]["moments"], rows[0]["ties"]))
    for line in lines:
        print(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seeds": args.seeds, "days": args.days, "switches": switches, "summary": lines},
                           ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
