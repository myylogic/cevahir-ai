# -*- coding: utf-8 -*-
"""t_ends.py: an end is an outcome (F66, ``Config.trial_ends``): a way of reading that makes the mind go on past the
end of an answer is put away by the mind's own trials, and stays put away for the next kind of question.

    python research/living_mind_2026_11_22/t_ends.py loop [--seeds 5]

F57 tries each of the mind's ways of reading at the contexts where it decided and which then went on from outside
*with a word*. Where the other ended there (the end of a turn it watches), nothing was an outcome, so a way that
made the mind say a word more could not be found to do harm: section 27 repaired such a way by hand (F62, for F52's
exact echo). With F66 the end is what came: saying nothing there is right, and a word is another word.

The world. Four kinds of fact are read of every thing ("X V Q ."), the order of the kinds rotating. Two kinds of
question are watched twice of eight things. Then the third kind, never watched, is asked of eight other things in
the teacher's loop (whatever the mind says, the true dialogue is shown). Then, on copies: the fourth kind, never
watched and never in a loop, of eight fresh things (does what it found in the loop hold for a new kind at first
sight?), the third kind of four fresh things, and a watched kind of four fresh things.

``value``: the answers are the value alone ("V ."). Without F62 the exact echo carries the question's verb on after
the value ("dal taşır"): it does harm. ``names``: the answers name the thing and end with the verb ("X V Q ."):
without F41 and F62 the exact echo is what says that verb: it is needed, and must not be put away.

Every mind is the full configuration (``zihin.__main__.VARIANTS["full"]``) with the changes its condition names.
Appends to ``results/ends.jsonl``.
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

OUT = os.path.join(HERE, "results", "ends.jsonl")
THINGS = ["inek", "koyun", "keçi", "at", "eşek", "tavuk", "deve", "geyik", "aslan", "kurt", "fok", "serçe", "kaz",
          "arı", "tilki", "ayı", "fil", "kedi", "köpek", "yılan", "balina", "kartal", "tavşan", "sincap", "leylek",
          "karga", "yunus", "panda", "koala", "zebra", "maymun", "baykuş"]
FACTS = {"yer": ["ot", "et", "balık", "tohum"], "içer": ["su", "süt", "çay", "şerbet"],
         "taşır": ["dal", "taş", "yaprak", "kum"], "sever": ["güneş", "gölge", "yağmur", "rüzgar"]}
CONDITIONS = {
    "value": [("the whole mind of section 27", {"trial_ends": False}),
              ("without F62: the exact echo alone", {"echo_told": False, "trial_ends": False}),
              ("without F62, an end is an outcome", {"echo_told": False, "trial_ends": True}),
              ("the whole mind, an end is an outcome", {"trial_ends": True})],
    "names": [("without F41 and F62", {"known": False, "echo_told": False, "trial_ends": False}),
              ("without F41 and F62, an end is an outcome", {"known": False, "echo_told": False, "trial_ends": True}),
              ("the whole mind, an end is an outcome", {"trial_ends": True})],
}


def life(seed: int, world: str, switches: dict):
    import zihin.__main__ as chat
    from zihin import IMITATED, Mind, deney
    from zihin.__main__ import CHANNEL, Session, VARIANTS
    rng = random.Random(seed)
    things = list(THINGS)
    rng.shuffle(things)
    kinds = list(FACTS)
    rng.shuffle(kinds)
    watched, looped, new = kinds[:2], kinds[2], kinds[3]
    shift = {q: rng.randrange(4) for q in kinds}

    def fact(x: str, q: str) -> str:
        return FACTS[q][(things.index(x) * 3 + shift[q]) % 4]

    def answer(x: str, q: str) -> str:
        return "%s %s %s" % (x, fact(x, q), q) if world == "names" else fact(x, q)
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["full"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for name, value in switches.items():
        if not hasattr(cfg, name):
            raise SystemExit("no such switch: %s" % name)
        setattr(cfg, name, value)
    torch.manual_seed(seed)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for i, x in enumerate(things):
        sentences = ["%s %s %s ." % (x, fact(x, q), q) for q in kinds]
        for sentence in sentences[i % 4:] + sentences[:i % 4]:
            s.company(None)
            s.listen(sentence, IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
    for _ in range(2):
        for q in watched:
            for x in things[:8]:
                s.watch("%s ne %s ?" % (x, q), answer(x, q) + " .")
    s.close_turn()

    def mark(said: str, x: str, q: str) -> str:
        return "R" if said == answer(x, q) else ("-" if said == "(sessiz)" else "W")

    def asked(x: str, q: str) -> str:
        return mark(Session(mind.clone()).reply("%s ne %s ?" % (x, q), limit=20), x, q)
    t0 = time.time()
    loop, first = "", None
    for x in things[8:16]:
        said = s.reply("%s ne %s ?" % (x, looped), limit=20)
        s.close_turn()
        s.watch("%s ne %s ?" % (x, looped), answer(x, looped) + " .")
        s.close_turn()
        loop += mark(said, x, looped)
        first = first or "%s ne %s ? -> %s" % (x, looped, said)
    took = time.time() - t0
    fresh_new = "".join(asked(x, new) for x in things[16:24])
    fresh_loop = "".join(asked(x, looped) for x in things[24:28])
    fresh_old = "".join(asked(x, watched[0]) for x in things[28:32])
    said_new = Session(mind.clone()).reply("%s ne %s ?" % (things[16], new), limit=20)
    changes = [{"t": c["t"], "way": str(c["way"]), "on": c["on"], "better": c["better"], "worse": c["worse"]}
               for c in mind.trial_changes if not c.get("owner")]
    return {"loop": loop, "new": fresh_new, "looped_fresh": fresh_loop, "watched_fresh": fresh_old,
            "changes": changes, "loop_s": took, "first": first,
            "first_new": "%s ne %s ? -> %s" % (things[16], new, said_new)}


def count(marks: str) -> dict:
    return {"right": marks.count("R"), "wrong": marks.count("W"), "silent": marks.count("-")}


def rws(c: dict) -> str:
    return "%d/%d/%d" % (c["right"], c["wrong"], c["silent"])


def loop(args) -> None:
    rows = []
    lines = ["== a kind of question never watched, in the teacher's loop (8 things); then on copies: another kind never "
             "watched, at first sight (8 fresh things), the looped kind (4) and a watched kind (4); right / wrong / "
             "silent; %d seeds ==" % args.seeds]
    for world in ("value", "names"):
        lines.append("  %s" % ("the answer is the value alone (\"V .\")" if world == "value" else
                               "the answer names the thing and ends with the verb (\"X V Q .\")"))
        for label, switches in CONDITIONS[world]:
            totals = {key: dict(right=0, wrong=0, silent=0) for key in ("loop", "new", "looped_fresh", "watched_fresh")}
            put_away, taken, took, lives, shown = {}, {}, 0.0, [], None
            for seed in range(args.seeds):
                out = life(seed, world, switches)
                for key, total in totals.items():
                    for name, n in count(out[key]).items():
                        total[name] += n
                for c in out["changes"]:
                    side = taken if c["on"] else put_away
                    side[c["way"]] = side.get(c["way"], 0) + 1
                took += out["loop_s"]
                lives.append(out)
                shown = shown or "%s %s ; %s ; then %s" % (out["loop"], out["new"], out["first"], out["first_new"])
            rows.append({"test": "ends_loop", "world": world, "condition": label, "switches": switches,
                         "seeds": args.seeds, "totals": totals, "put_away": put_away, "taken": taken,
                         "loop_s": round(took, 1), "lives": lives})
            lines.append("    %-42s loop %-8s another new kind %-8s the looped kind %-7s a watched kind %-7s ; put "
                         "away (in lives): %s%s ; %.0f s ; the first seed: %s" % (
                             label, rws(totals["loop"]), rws(totals["new"]), rws(totals["looped_fresh"]),
                             rws(totals["watched_fresh"]),
                             ", ".join("%s %d" % kv for kv in sorted(put_away.items())) or "nothing",
                             (" ; taken: " + ", ".join("%s %d" % kv for kv in sorted(taken.items()))) if taken else "",
                             took, shown))
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for row in rows:
            row["summary"] = lines
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["loop"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    loop(args)


if __name__ == "__main__":
    main()
