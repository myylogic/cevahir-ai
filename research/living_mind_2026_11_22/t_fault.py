# -*- coding: utf-8 -*-
"""t_fault.py: F49 Fault Tracing, measured. Where the mind's own computation broke, and what it learns from it.

    python research/living_mind_2026_11_22/t_fault.py misled   [--seeds 5]
    python research/living_mind_2026_11_22/t_fault.py regular  [--seeds 5]
    python research/living_mind_2026_11_22/t_fault.py two      [--seeds 5]
    python research/living_mind_2026_11_22/t_fault.py valued   [--seeds 5]
    python research/living_mind_2026_11_22/t_fault.py many     [--seeds 5]
    python research/living_mind_2026_11_22/t_fault.py loop     [--seeds 5]
    python research/living_mind_2026_11_22/t_fault.py report

Every part is the same life in each condition (same seed, same order): F49 off, F49 on, and F49 on reading only
some of its levels (which level does what). The teacher's loop is the only way anything is corrected: the mind is
asked, it answers or stays silent, and then the true dialogue is shown to it (watched), whatever it said. Nothing
tells the mind that it was wrong, where, or why: it has its own answer's chain and what it then lives.

``misled``: frames whose members merely share a value. Five question frames about 48 things. In three of them
(V1-V3) the answer depends on the thing (one of four values, drawn at random); in two (L1, L2) it is the same for
every thing. The mind first watches the same 8 things in all five frames, and those 8 happen to share one value in
each frame: so it generalises in all five, and in V1-V3 the generalisation is right for one thing in four. Then the
teacher's loop, 8 new things a frame, in the order V1, V2, L1, V3, L2. Counted: right / wrong / silent answers in
each loop (how many errors before the mind stops asserting in a frame, and whether the next misleading frame costs
fewer), and at the end right / wrong / silent on 8 things never asked about, in every frame.

``regular``: the same life where all five frames are lawful (one value each for every thing): nothing misleads, so
nothing should be lost.

``two``: two readings that disagree. Every thing was read of once in a sentence ("X V yer ."); the 8 watched
dialogues ("X ne yer ?" -> "X V yer .") share one value. For a new thing the frame's members agree on the value
itself and on a relation (what followed the thing in what was read); the two name different words. Counted: right
answers on 8 fresh things after 0, 2, 4, 8, 16 rounds of the teacher's loop.

``valued``: no answer is ever shown; after each answer the mind is only told whether it was good or bad (the
chat's !dogru / !yanlis). Counted as in ``misled`` for one misleading and one lawful frame.

``many``: whether a principle is taken from the errors. Eight misleading frames and four lawful ones, one after
another, four new things in each (64 things). In the order "V first" the eight misleading frames come first, then
the lawful ones; in the order "L first" the lawful ones come first. Counted: wrong answers in each frame in the
order lived (does the n-th misleading frame cost fewer errors than the first?), and what the lawful frames cost.
Conditions: off; on; on reading only the frame's own record (no principle: every frame pays for itself); on with
an untested frame read by the others' vote from the start (the stricter form: not tried when frames like it failed).

``loop``: the whole loop by itself (with F46 and F47 on in both conditions): the mind read of the 8 watched things
only; a library holds one sentence about every thing, and the mind may look into it in its silent moments, by its
own decision. Each of 16 new things: asked (first answer); three silent moments (it may look the thing up); asked
again on a copy (what it can say now, before anyone shows it anything); then the true dialogue is shown. Counted:
the first answers, the looks it made, the answers after its own looking; an answer that names the right value and
stops there is counted apart (the last word of this answer shape follows a value the mind has not yet watched
anyone say there).

``report``: one life of ``misled`` (seed 0), then what the mind says about its own ways (hata.report).

Appends to ``results/fault.jsonl``.
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "fault.jsonl")
THINGS = ["inek", "koyun", "keçi", "at", "eşek", "tavuk", "deve", "geyik", "tavşan", "sincap", "aslan", "kurt",
          "kaplan", "çakal", "vaşak", "sırtlan", "ayı", "tilki", "fil", "zürafa", "maymun", "kartal", "şahin",
          "baykuş", "leylek", "serçe", "karga", "martı", "ördek", "kaz", "yılan", "kertenkele", "kaplumbağa",
          "kurbağa", "balina", "yunus", "fok", "penguen", "kanguru", "panda", "koala", "timsah", "akrep", "örümcek",
          "arı", "karınca", "kelebek", "sinek"]
FRAMES = {"V1": ("%s ne yer ?", "%s %s yer .", ["ot", "et", "balık", "tohum"]),
          "V2": ("%s nerede yaşar ?", "%s %s yaşar .", ["ormanda", "çölde", "denizde", "dağda"]),
          "V3": ("%s ne renk ?", "%s %s renktedir .", ["kahve", "sarı", "gri", "beyaz"]),
          "L1": ("%s canlı mı ?", "evet %s %s .", ["canlıdır"]),
          "L2": ("%s büyür mü ?", "evet %s %s .", ["büyür"])}
MORE_THINGS = ["ceylan", "antilop", "bizon", "manda", "lama", "porsuk", "kirpi", "köstebek", "yarasa", "sansar",
               "gelincik", "samur", "kunduz", "mors", "turna", "pelikan"]
MANY = {"V1": ("%s ne yer ?", "%s %s yer .", ["ot", "et", "balık", "tohum"]),
        "V2": ("%s nerede yaşar ?", "%s %s yaşar .", ["ormanda", "çölde", "denizde", "dağda"]),
        "V3": ("%s ne renk ?", "%s %s renktedir .", ["kahve", "sarı", "gri", "beyaz"]),
        "V4": ("%s ne içer ?", "%s %s içer .", ["su", "süt", "çay", "şerbet"]),
        "V5": ("%s neyi sever ?", "%s %s sever .", ["oyunu", "uykuyu", "güneşi", "yağmuru"]),
        "V6": ("%s nasıl yürür ?", "%s %s yürür .", ["hızlı", "yavaş", "sessiz", "ağır"]),
        "V7": ("%s ne zaman uyur ?", "%s %s uyur .", ["gece", "gündüz", "sabah", "akşam"]),
        "V8": ("%s kimden korkar ?", "%s %s korkar .", ["avcıdan", "yılandan", "ateşten", "gürültüden"]),
        "L1": ("%s canlı mı ?", "evet %s %s .", ["canlıdır"]),
        "L2": ("%s büyür mü ?", "evet %s %s .", ["büyür"]),
        "L3": ("%s taş mı ?", "hayır %s %s .", ["hayvandır"]),
        "L4": ("%s ölür mü ?", "elbette %s %s .", ["ölür"])}
CONDITIONS = [("off", False, None), ("on", True, None), ("on: the frame's own record only", True, (2,)),
              ("on: an untested frame read by the others' vote", True, "vote")]


def make_mind(seed: int, on: bool, levels=None, more=None):
    import t_relations as R
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session
    torch.manual_seed(seed)
    switches = {"faults": on}
    switches.update(more or {})
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    if levels == "vote":
        mind.__dict__["_fault_untested"] = "vote"
    elif levels is not None:
        mind.__dict__["_fault_levels"] = set(levels)
    return mind, Session(mind)


def verdict(session, said: str, answer: str) -> str:
    right = " ".join(session.words(answer)[:-1])
    if said == right:
        return "R"
    return "-" if said == "(sessiz)" or len(said.split()) < len(right.split()) else "W"


def teacher(session, question: str, answer: str) -> str:
    """The mind is asked; whatever it says, the true dialogue is then shown to it. Returns R / W / -."""
    said = session.reply(question, limit=30)
    session.close_turn()
    session.watch(question, answer)
    session.close_turn()
    return verdict(session, said, answer)


def probe(session, question: str, answer: str) -> str:
    from zihin.__main__ import Session
    copy = Session(session.mind.clone(), session.pieces)
    copy.partner, copy.dil = session.partner, session.dil
    return verdict(session, copy.reply(question, limit=30), answer)


def count(marks: str) -> dict:
    return {"right": marks.count("R"), "wrong": marks.count("W"), "silent": marks.count("-")}


def world(seed: int, lawful_all: bool):
    rng = random.Random(seed)
    things = list(THINGS)
    rng.shuffle(things)
    members = things[:8]
    truth = {}
    for f, (_, _, values) in FRAMES.items():
        for x in things:
            truth[f, x] = values[0] if x in members or len(values) == 1 or lawful_all else rng.choice(values)
    return things, members, truth


def misled(args, lawful_all: bool = False) -> None:
    name = "regular" if lawful_all else "misled"
    order = ["V1", "V2", "L1", "V3", "L2"]
    rows = []
    for label, on, levels in CONDITIONS:
        loops = {f: {"right": 0, "wrong": 0, "silent": 0} for f in order}
        fresh_end = {f: {"right": 0, "wrong": 0, "silent": 0} for f in order}
        marks_by_seed, took = [], 0.0
        for seed in range(args.seeds):
            things, members, truth = world(seed, lawful_all)
            mind, s = make_mind(seed, on, levels)

            def qa(f, x):
                q, a, _ = FRAMES[f]
                return q % x, a % (x, truth[f, x])
            for _ in range(2):
                for x in members:
                    for f in FRAMES:
                        s.watch(*qa(f, x))
            s.close_turn()
            pool, fresh = things[8:], things[40:]
            marks, k = {}, 0
            t0 = time.time()
            for f in order:
                marks[f] = "".join(teacher(s, *qa(f, x)) for x in pool[k:k + 6])
                k += 6
                for key, n in count(marks[f]).items():
                    loops[f][key] += n
            took += time.time() - t0
            for f in order:
                end = "".join(probe(s, *qa(f, x)) for x in fresh)
                for key, n in count(end).items():
                    fresh_end[f][key] += n
            marks_by_seed.append(marks)
        rows.append({"test": "fault_" + name, "condition": label, "seeds": args.seeds, "loops": loops,
                     "fresh_end": fresh_end, "marks": marks_by_seed, "loop_s": round(took, 1)})
    lines = ["== %s: %d seeds; the teacher's loop, 6 new things a frame, in the order %s; then 8 things never asked "
             "about (right / wrong / silent) ==" % (name, args.seeds, ", ".join(order))]
    for row in rows:
        lines.append("  %s" % row["condition"])
        lines.append("    in the loops:  " + "  ".join("%s %d/%d/%d" % (f, row["loops"][f]["right"], row["loops"][f]["wrong"],
                                                                  row["loops"][f]["silent"]) for f in order))
        lines.append("    fresh at the end: " + "  ".join("%s %d/%d/%d" % (
            f, row["fresh_end"][f]["right"], row["fresh_end"][f]["wrong"], row["fresh_end"][f]["silent"]) for f in order))
        vs = [f for f in order if f.startswith("V")]
        ls = [f for f in order if f.startswith("L")]
        lines.append("    wrong answers in the loops: %d in V1-V3, %d in L1-L2; right: %d and %d; the first seed: %s" % (
            sum(row["loops"][f]["wrong"] for f in vs), sum(row["loops"][f]["wrong"] for f in ls),
            sum(row["loops"][f]["right"] for f in vs), sum(row["loops"][f]["right"] for f in ls),
            " ".join("%s %s" % (f, row["marks"][0][f]) for f in order)))
    finish(args, rows, lines)


def two(args) -> None:
    from zihin import IMITATED
    rounds = [0, 2, 4, 8, 16]
    rows = []
    for label, on, levels in CONDITIONS[:2]:
        at = {r: {"right": 0, "wrong": 0, "silent": 0} for r in rounds}
        loop = {"right": 0, "wrong": 0, "silent": 0}
        for seed in range(args.seeds):
            rng = random.Random(seed)
            things = list(THINGS)
            rng.shuffle(things)
            members = things[:8]
            values = FRAMES["V1"][2]
            food = {x: values[0] if x in members else rng.choice(values) for x in things}
            mind, s = make_mind(seed, on, levels)

            def qa(x):
                return "%s ne yer ?" % x, "%s %s yer ." % (x, food[x])
            for _ in range(2):
                for x in things:                        # what was read of every thing, once a round
                    s.company(None)
                    s.listen("%s %s yer ." % (x, food[x]), IMITATED)
                    s.end_of_turn(IMITATED, tone=".")
                    mind.step([])
                for x in members:
                    s.watch(*qa(x))
            s.close_turn()
            pool, fresh = things[8:40], things[40:]
            for i in range(rounds[-1] + 1):
                if i in at:
                    for key, n in count("".join(probe(s, *qa(x)) for x in fresh)).items():
                        at[i][key] += n
                if i < rounds[-1]:
                    for key, n in count(teacher(s, *qa(pool[i]))).items():
                        loop[key] += n
        rows.append({"test": "fault_two", "condition": label, "seeds": args.seeds, "fresh_after": at, "loop": loop})
    lines = ["== two readings that disagree (what was read of the thing itself; what the watched things share): "
             "%d seeds, 8 fresh things after n rounds of the teacher's loop (right / wrong / silent) ==" % args.seeds]
    for row in rows:
        lines.append("  %-4s %s ; in the loop %d/%d/%d" % (row["condition"], "  ".join("n=%d %d/%d/%d" % (
            r, row["fresh_after"][r]["right"], row["fresh_after"][r]["wrong"], row["fresh_after"][r]["silent"])
            for r in rounds), row["loop"]["right"], row["loop"]["wrong"], row["loop"]["silent"]))
    finish(args, rows, lines)


def valued(args) -> None:
    order = ["V1", "L1"]
    rows = []
    for label, on, levels in (CONDITIONS[0], CONDITIONS[1], CONDITIONS[3]):
        loops = {f: {"right": 0, "wrong": 0, "silent": 0} for f in order}
        fresh_end = {f: {"right": 0, "wrong": 0, "silent": 0} for f in order}
        first = None
        for seed in range(args.seeds):
            things, members, truth = world(seed, False)
            mind, s = make_mind(seed, on, levels)

            def qa(f, x):
                q, a, _ = FRAMES[f]
                return q % x, a % (x, truth[f, x])
            for _ in range(2):
                for x in members:
                    for f in order:
                        s.watch(*qa(f, x))
            s.close_turn()
            pool, fresh = things[8:], things[40:]
            marks, k = {}, 0
            for f in order:
                out = ""
                for x in pool[k:k + 8]:
                    q, a = qa(f, x)
                    mark = verdict(s, s.reply(q, limit=30), a)
                    if mark != "-":
                        s.feedback(mark == "R")            # only how the answer was received; nothing is shown
                    s.close_turn()
                    out += mark
                marks[f], k = out, k + 8
                for key, n in count(out).items():
                    loops[f][key] += n
            for f in order:
                for key, n in count("".join(probe(s, *qa(f, x)) for x in fresh)).items():
                    fresh_end[f][key] += n
            first = first or marks
        rows.append({"test": "fault_valued", "condition": label, "seeds": args.seeds, "loops": loops,
                     "fresh_end": fresh_end, "first": first})
    lines = ["== told only whether its answer was good or bad (no answer is shown): %d seeds, 8 things a frame, then "
             "8 fresh things (right / wrong / silent) ==" % args.seeds]
    for row in rows:
        lines.append("  %s: in the loops: %s ; fresh at the end: %s ; the first seed: %s" % (
            row["condition"],
            "  ".join("%s %d/%d/%d" % (f, row["loops"][f]["right"], row["loops"][f]["wrong"], row["loops"][f]["silent"])
                      for f in order),
            "  ".join("%s %d/%d/%d" % (f, row["fresh_end"][f]["right"], row["fresh_end"][f]["wrong"],
                                       row["fresh_end"][f]["silent"]) for f in order),
            " ".join("%s %s" % (f, row["first"][f]) for f in order)))
    finish(args, rows, lines)


def many(args) -> None:
    vs, ls = ["V%d" % i for i in range(1, 9)], ["L%d" % i for i in range(1, 5)]
    rows, lines = [], ["== many frames (8 misleading, 4 lawful; 4 new things a frame): %d seeds; wrong answers in each "
                       "frame in the order lived (of %d), then right / wrong / silent ==" % (args.seeds, 4 * args.seeds)]
    for order_name, order in (("V first", vs + ls), ("L first", ls + vs)):
        for label, on, levels in CONDITIONS:
            loops = {f: {"right": 0, "wrong": 0, "silent": 0} for f in order}
            first = None
            for seed in range(args.seeds):
                rng = random.Random(seed)
                things = THINGS + MORE_THINGS
                rng.shuffle(things)
                members = things[:8]
                truth = {(f, x): values[0] if x in members or len(values) == 1 else rng.choice(values)
                         for f, (_, _, values) in MANY.items() for x in things}
                mind, s = make_mind(seed, on, levels)

                def qa(f, x):
                    q, a, _ = MANY[f]
                    return q % x, a % (x, truth[f, x])
                for _ in range(2):
                    for x in members:
                        for f in MANY:
                            s.watch(*qa(f, x))
                s.close_turn()
                marks, k = {}, 8
                for f in order:
                    marks[f] = "".join(teacher(s, *qa(f, x)) for x in things[k:k + 4])
                    k += 4
                    for key, n in count(marks[f]).items():
                        loops[f][key] += n
                first = first or marks
            rows.append({"test": "fault_many", "order": order_name, "condition": label, "seeds": args.seeds,
                         "loops": loops, "first": first})
            lines.append("  %s; %s" % (order_name, label))
            lines.append("    wrong:  " + "  ".join("%s %2d" % (f, loops[f]["wrong"]) for f in order))
            lines.append("    all:    " + "  ".join("%s %d/%d/%d" % (f, loops[f]["right"], loops[f]["wrong"],
                                                                     loops[f]["silent"]) for f in order))
            lines.append("    sums: V wrong %d, right %d, silent %d; L right %d, wrong %d, silent %d; the first seed: %s" % (
                sum(loops[f]["wrong"] for f in vs), sum(loops[f]["right"] for f in vs),
                sum(loops[f]["silent"] for f in vs), sum(loops[f]["right"] for f in ls),
                sum(loops[f]["wrong"] for f in ls), sum(loops[f]["silent"] for f in ls),
                " ".join("%s %s" % (f, first[f]) for f in order)))
    finish(args, rows, lines)


class Library:
    """A source that stands where the web stands: one sentence about every thing."""

    def __init__(self, facts: dict) -> None:
        self.facts, self.asked = facts, []

    def search(self, topic: str, limit: int = 2):
        self.asked.append(topic)
        return [topic] if topic in self.facts else []

    def extract(self, identifier: str) -> str:
        return self.facts[identifier]

    def address(self, identifier: str) -> str:
        return "library:" + identifier


def key_verdict(session, said: str, answer: str) -> str:
    """R: the whole answer; K: the asked word and the right value, not (yet) the rest; W: another value said;
    -: nothing after the asked word."""
    right = session.words(answer)[:-1]
    words = said.split() if said != "(sessiz)" else []
    if words == right:
        return "R"
    if len(words) < 2:
        return "-"
    return "K" if words[:2] == right[:2] else "W"


def loop(args) -> None:
    from zihin import IMITATED
    rows = []

    def tally(marks: str) -> dict:
        return {"right": marks.count("R"), "key": marks.count("K"), "wrong": marks.count("W"),
                "silent": marks.count("-")}
    for label, on, levels in CONDITIONS[:2]:
        first = {"right": 0, "key": 0, "wrong": 0, "silent": 0}
        after = {"right": 0, "key": 0, "wrong": 0, "silent": 0}
        looks, met, marks0 = 0, 0, None
        for seed in range(args.seeds):
            rng = random.Random(seed)
            things = list(THINGS)
            rng.shuffle(things)
            members = things[:8]
            values = FRAMES["V1"][2]
            food = {x: values[0] if x in members else rng.choice(values) for x in things}
            mind, s = make_mind(seed, on, levels, {"open_claims": True, "look": True})
            s.may_look, s.look_gap = True, 0.0
            s.look_source = Library({x: "%s %s yer ." % (x, food[x]) for x in things})

            def qa(x):
                return "%s ne yer ?" % x, "%s %s yer ." % (x, food[x])
            for _ in range(2):
                for x in members:
                    s.company(None)
                    s.listen("%s %s yer ." % (x, food[x]), IMITATED)
                    s.end_of_turn(IMITATED, tone=".")
                    mind.step([])
                    s.watch(*qa(x))
            s.close_turn()
            a_marks, b_marks = "", ""
            for x in things[8:24]:
                q, a = qa(x)
                a_marks += key_verdict(s, s.reply(q, limit=30), a)
                s.close_turn()
                before = len(s.looks)
                for _ in range(3):
                    s.silent_moment(say=lambda *_: None)
                looks += len(s.looks) - before
                met += sum(1 for r in s.looks[before:] if r.get("met"))
                from zihin.__main__ import Session
                copy = Session(mind.clone(), s.pieces)
                copy.partner, copy.dil = s.partner, s.dil
                b_marks += key_verdict(s, copy.reply(q, limit=30), a)
                s.watch(q, a)
                s.close_turn()
            for key, n in tally(a_marks).items():
                first[key] += n
            for key, n in tally(b_marks).items():
                after[key] += n
            marks0 = marks0 or (a_marks, b_marks)
        rows.append({"test": "fault_loop", "condition": label, "seeds": args.seeds, "first": first, "after": after,
                     "looks": looks, "met": met, "marks": marks0})
    lines = ["== the loop by itself (F46 and F47 on in both): %d seeds, 16 new things nothing was read of; asked, three "
             "silent moments of its own, asked again on a copy, then shown (whole answer right / the right value but "
             "not the rest / another value / nothing) ==" % args.seeds]
    for row in rows:
        lines.append("  %-3s first answers %d/%d/%d/%d; looks of its own %d (the claim met by what it read: %d); after "
                     "its own looking %d/%d/%d/%d; the first seed: %s | %s" % (
                         row["condition"], row["first"]["right"], row["first"]["key"], row["first"]["wrong"],
                         row["first"]["silent"], row["looks"], row["met"], row["after"]["right"], row["after"]["key"],
                         row["after"]["wrong"], row["after"]["silent"], row["marks"][0], row["marks"][1]))
    finish(args, rows, lines)


def report(args) -> None:
    from zihin import hata
    things, members, truth = world(0, False)
    mind, s = make_mind(0, True)

    def qa(f, x):
        q, a, _ = FRAMES[f]
        return q % x, a % (x, truth[f, x])
    for _ in range(2):
        for x in members:
            for f in FRAMES:
                s.watch(*qa(f, x))
    s.close_turn()
    k, pool = 0, things[8:]
    lines = ["== what the mind says about its own ways after one life of ``misled`` (seed 0) =="]
    for f in ["V1", "V2", "L1", "V3", "L2"]:
        lines.append("  %s: %s" % (f, "".join(teacher(s, *qa(f, x)) for x in pool[k:k + 6])))
        k += 6
    lines += hata.report(mind, most=40)
    finish(args, [{"test": "fault_report", "lines": lines}], lines)


def finish(args, rows, lines) -> None:
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for row in rows:
            row = dict(row)
            row.setdefault("summary", lines)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["misled", "regular", "two", "valued", "many", "loop", "report"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    {"misled": misled, "regular": lambda a: misled(a, True), "two": two, "valued": valued, "many": many,
     "loop": loop, "report": report}[args.what](args)


if __name__ == "__main__":
    main()
