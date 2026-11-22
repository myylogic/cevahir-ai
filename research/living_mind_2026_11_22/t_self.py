# -*- coding: utf-8 -*-
"""t_self.py: F57 Trials and F58 Built Relations, measured. A mind that changes its own way of learning.

    python research/living_mind_2026_11_22/t_self.py curriculum [--seeds 5]
    python research/living_mind_2026_11_22/t_self.py control    [--seeds 5]
    python research/living_mind_2026_11_22/t_self.py kept       [--seeds 5]
    python research/living_mind_2026_11_22/t_self.py worlds     [--seeds 5]
    python research/living_mind_2026_11_22/t_self.py traced     [--seed 2]

The world: 48 things. Of every thing seven sentences are read once, in an order of its own: six facts ("X V Q .":
what it eats, drinks, likes, wears, reads, draws; four values a kind, one at random) and its class
("X K sınıfındandır ."; four classes). Of every class three sentences are read ("K V doğurur .", "K V örtünür .",
and its order: "K S takımındandır ."; four orders), of every order one ("S V üretir ."). Nothing else is told. Then
one mind lives a **curriculum** in the teacher's loop (asked about a thing of its own; whatever it says, the true
dialogue is then shown), every answer the value alone ("X ne Q ?" -> "V ."):

    stage 1   what a thing eats, drinks (6 rounds each): a fact read of the thing itself, among seven
    stage 2   what it likes, wears, reads (4 rounds each): new kinds of the same question
    stage 3   what it bears ("X ne doğurur ?", 6 rounds): never read of the thing; read of its class
    stage 4   what it is covered with ("X ne örtünür ?", 4 rounds): the same structure, a second time
    stage 5   what it makes ("X ne üretir ?", 6 rounds): read neither of the thing nor of its class; of the
              class's order

40 things are asked about in the life (one question each); the other 8 are never asked about.

What is measured, all on copies of the mind (the life is not touched):

* after every stage, 8 things never asked about are asked every kind of question met so far (right / wrong /
  silent): what the mind can do then;
* after every stage, every thing taught so far is asked its own question again, by the stage it was taught in:
  whether what was learned earlier is kept; and each of them is asked one other kind of question met so far (the
  next kind in the order they were met): a thing that was asked about before, asked something else;
* at every change the mind makes to its own configuration, the same 8 fresh things are asked every kind (those met
  and those to come) by two copies taken at that moment: one as the mind is after the change, one with that one
  change undone. The same memory, one thing different: what the change bought, kind by kind;
* at the end, a kind never asked in the life ("X ne çizer ?") at first sight.

Conditions (the same life in each):

* ``as it was``: the speaking configuration with F49, every studied way of reading off (the mind before this work);
* ``F57``: that, with trials: the mind's ways of reading are tried on what it lives and it takes into use what its
  own record proves;
* ``F57 + F58``: that, and where it fails it proposes relations of its own (chains of its relations) and tries them;
* ``set by hand``: F41, F53, F54 and F55 on from the start, no trials (how its builders would set it).

``control``: what a mind that follows its own record must not do. Stage 1 as it is, then four kinds of question
that are answered otherwise than a family of frames claims (each with two words of its own, whatever the thing),
three rounds each. (a) F53 on trial (F55 in use, F49 on): whether it is taken into use. (b) A mind raised with F53
on by hand, without F49: F53 asserts a value at the first sight of each such kind; whether and when the mind puts
it away, and the first answer of each kind before and after. (c) The same with F49 on (the mind also judges every
claim by the record of its way).

``kept``: the life of ``F57 + F58``; the mind is saved and loaded; whether the loaded mind has the configuration
and the relations the life came to, and answers the fresh things as the mind that was saved.

``worlds``: the same mind and the same things in another world: every answer names the thing ("X ne Q ?" ->
"X V Q ."), five kinds in the teacher's loop (6, 6, 4, 4, 4 rounds). Which ways the mind takes into use there
(against those it takes where the answer is the value alone: ``curriculum``), and what 8 things never asked about
are answered at the end, with ``as it was`` and ``set by hand`` beside it.

``traced``: one life of ``set by hand`` (the seed in which a kind answered right for fresh things comes to be
answered wrong after a later stage); after every stage the 8 fresh things are asked that kind on copies, as the
mind is and with F50's rule "what was lived of the thing itself stands against what the others share" on in the
copies that are asked (``find_out``, nowhere to look).

Appends to ``results/self.jsonl``.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import t_fault as T  # noqa: E402
import t_grow as G  # noqa: E402

OUT = os.path.join(HERE, "results", "self.jsonl")
DIRECT = ["yer", "içer", "sever", "giyer", "okur", "çizer"]
CLASSES = ["memeli", "sürüngen", "kanatlı", "sucul"]
BY_CLASS = {"doğurur": {"memeli": "yavru", "sürüngen": "yumurta", "kanatlı": "civciv", "sucul": "larva"},
            "örtünür": {"memeli": "kıl", "sürüngen": "zırh", "kanatlı": "telek", "sucul": "mukus"}}
ORDER = {"memeli": "sıcakkanlılar", "sürüngen": "pullular", "kanatlı": "tüylüler", "sucul": "solungaçlılar"}
BY_ORDER = {"üretir": {"sıcakkanlılar": "ısı", "pullular": "zehir", "tüylüler": "ötüş", "solungaçlılar": "kabarcık"}}
STAGES = [("stage 1", [("yer", 6), ("içer", 6)]), ("stage 2", [("sever", 4), ("giyer", 4), ("okur", 4)]),
          ("stage 3", [("doğurur", 6)]), ("stage 4", [("örtünür", 4)]), ("stage 5", [("üretir", 6)])]
assert sum(rounds for _, stage in STAGES for _, rounds in stage) == 40      # the last 8 things are never asked about
KINDS = [q for _, stage in STAGES for q, _ in stage]
NEVER = "çizer"                                              # read of every thing, never asked in the life
CONDITIONS = [("as it was", {"faults": True}),
              ("F57", {"faults": True, "trials": True}),
              ("F57 + F58", {"faults": True, "trials": True, "built_relations": True}),
              ("set by hand", {"faults": True, "known": True, "families": True, "lived_apart": True,
                               "asked_place": True})]
OTHERWISE = {"sever": "hiçbir şey .", "giyer": "kim bilir .", "okur": "belli olmaz .", "çizer": "orası karışık ."}


def world(seed: int, more: dict):
    """The mind that has read everything; (session, things, the true value of (thing, kind of question))."""
    from zihin import IMITATED
    rng = random.Random(seed)
    things = list(T.THINGS)
    rng.shuffle(things)
    kind = {x: rng.choice(CLASSES) for x in things}
    truth = {(x, q): rng.choice(G.VALUES[q]) for x in things for q in DIRECT}
    for q, values in BY_CLASS.items():
        for x in things:
            truth[(x, q)] = values[kind[x]]
    for q, values in BY_ORDER.items():
        for x in things:
            truth[(x, q)] = values[ORDER[kind[x]]]
    mind, s = T.make_mind(seed, False, None, dict(more))

    def read(sentence: str) -> None:
        s.company(None)
        s.listen(sentence, IMITATED)
        s.end_of_turn(IMITATED, tone=".")
        mind.step([])
    for q, values in BY_CLASS.items():
        for k in CLASSES:
            read("%s %s %s ." % (k, values[k], q))
    for k in CLASSES:
        read("%s %s takımındandır ." % (k, ORDER[k]))
    for q, values in BY_ORDER.items():
        for order in ORDER.values():
            read("%s %s %s ." % (order, values[order], q))
    for x in things:
        sentences = ["%s %s %s ." % (x, truth[(x, q)], q) for q in DIRECT] + ["%s %s sınıfındandır ." % (x, kind[x])]
        rng.shuffle(sentences)
        for sentence in sentences:
            read(sentence)
    return s, things, truth


def mark(session, said: str, answer: str) -> str:
    """R: the answer; W: anything else said (a part of the answer, or another word); -: nothing."""
    right = " ".join(session.words(answer)[:-1])
    return "R" if said == right else ("-" if said == "(sessiz)" else "W")


def teacher(session, question: str, answer: str) -> str:
    """The mind is asked; whatever it says, the true dialogue is then shown to it. Returns R / W / -."""
    said = session.reply(question, limit=30)
    session.close_turn()
    session.watch(question, answer)
    session.close_turn()
    return mark(session, said, answer)


def probe(session, question: str, answer: str, way=None, on=None) -> str:
    """The question asked of a copy of the mind; with ``way``, of a copy in which that one way is set ``on``."""
    from zihin import deney
    from zihin.__main__ import Session
    copy = Session(session.mind.clone(), session.pieces)
    copy.partner, copy.dil = session.partner, session.dil
    if way is not None:
        deney._set(copy.mind, way, on)
    return mark(session, copy.reply(question, limit=30), answer)


def add(total: dict, marks: str) -> dict:
    for key, n in T.count(marks).items():
        total[key] = total.get(key, 0) + n
    return total


def rws(d: dict) -> str:
    return "%d/%d/%d" % (d.get("right", 0), d.get("wrong", 0), d.get("silent", 0))


def name_of(way) -> str:
    return way if isinstance(way, str) else "chain %s" % (tuple(way),)


def in_use(mind) -> dict:
    from zihin import deney
    return {"ways": [w for w in deney.WAYS if getattr(mind.cfg, w, False)],
            "relations": [list(p) for p in getattr(mind, "relations", ())]}


def life(seed: int, more: dict, measure: bool = True):
    """The curriculum lived by one mind. (session, things, truth, what was measured on the way)."""
    s, things, truth = world(seed, more)
    mind = s.mind
    out = {"fresh": {}, "taught": {}, "other": {}, "loops": {}, "bought": [], "marks": [], "loop_s": 0.0}
    used, met, taught = 0, [], []
    for stage, kinds in STAGES:
        for q, rounds in kinds:
            marks = ""
            for x in things[used:used + rounds]:
                before = len(getattr(mind, "trial_changes", ()))
                t0 = time.time()
                marks += teacher(s, "%s ne %s ?" % (x, q), "%s ." % truth[(x, q)])
                out["loop_s"] += time.time() - t0
                taught.append((stage, x, q))
                for change in (getattr(mind, "trial_changes", ())[before:] if measure else ()):
                    way, on = change["way"], change["on"]        # the same memory with and without this one change
                    row = {"t": change["t"], "way": name_of(way), "on": on, "during": q, "with": {}, "without": {}}
                    for kind in KINDS + [NEVER]:
                        for key, value in (("with", on), ("without", not on)):
                            row[key][kind] = T.count("".join(
                                probe(s, "%s ne %s ?" % (y, kind), "%s ." % truth[(y, kind)], way, value)
                                for y in things[40:48]))
                    row["met"] = [k for k in KINDS if k in met or k == q]
                    out["bought"].append(row)
            used += rounds
            met.append(q)
            out["marks"].append("%s:%s" % (q, marks))
            add(out["loops"].setdefault(q, {}), marks)
        if not measure:
            continue
        for q in met:
            add(out["fresh"].setdefault("%s|%s" % (stage, q), {}), "".join(
                probe(s, "%s ne %s ?" % (x, q), "%s ." % truth[(x, q)]) for x in things[40:48]))
        for then, x, q in taught:
            add(out["taught"].setdefault("%s|%s" % (stage, then), {}),
                probe(s, "%s ne %s ?" % (x, q), "%s ." % truth[(x, q)]))
            other = met[(met.index(q) + 1) % len(met)]           # asked about before; now another kind of question
            add(out["other"].setdefault(stage, {}), probe(s, "%s ne %s ?" % (x, other), "%s ." % truth[(x, other)]))
    if measure:
        add(out["fresh"].setdefault("never asked|%s" % NEVER, {}), "".join(
            probe(s, "%s ne %s ?" % (x, NEVER), "%s ." % truth[(x, NEVER)]) for x in things[40:48]))
    return s, things, truth, out


def curriculum(args) -> None:
    from zihin import deney
    rows = []
    for label, more in CONDITIONS:
        if args.only and label != args.only:
            continue
        total = {"fresh": {}, "taught": {}, "other": {}, "loops": {}}
        bought, changes, final, first, took = [], [], [], None, 0.0
        for seed in range(args.seeds):
            s, things, truth, out = life(seed, more)
            mind = s.mind
            for table in total:
                for key, value in out[table].items():
                    row = total[table].setdefault(key, {})
                    for name, n in value.items():
                        row[name] = row.get(name, 0) + n
            took += out["loop_s"]
            bought.append(out["bought"])
            final.append(in_use(mind))
            changes.append([{"t": c["t"], "way": name_of(c["way"]), "on": c["on"], "better": c["better"],
                             "worse": c["worse"]} for c in getattr(mind, "trial_changes", ())])
            if first is None:
                first = {"loops": " ".join(out["marks"]),
                         "report": deney.report(mind, 4) if more.get("trials") else []}
        rows.append({"test": "self_curriculum", "condition": label, "seeds": args.seeds, "loops": total["loops"],
                     "fresh": total["fresh"], "taught": total["taught"], "taught_asked_another_kind": total["other"],
                     "bought": bought,
                     "in_use_at_the_end": final, "changes": changes, "loop_s": round(took, 1), "first": first})
    lines = ["== a curriculum in the teacher's loop, one mind, %d seeds; all measured on copies (right / wrong / "
             "silent) ==" % args.seeds]
    for row in rows:
        lines.append("  %s" % row["condition"])
        for stage, _ in STAGES:
            cells = ["%s %s" % (q, rws(row["fresh"]["%s|%s" % (stage, q)])) for q in KINDS
                     if "%s|%s" % (stage, q) in row["fresh"]]
            lines.append("    8 fresh things after %s: %s" % (stage, "  ".join(cells)))
        lines.append("    a kind never asked in the life (%s), 8 fresh things at first sight: %s" % (
            NEVER, rws(row["fresh"]["never asked|%s" % NEVER])))
        for stage, _ in STAGES:
            cells = ["taught in %s %s" % (then, rws(row["taught"]["%s|%s" % (stage, then)])) for then, _ in STAGES
                     if "%s|%s" % (stage, then) in row["taught"]]
            lines.append("    the things taught, asked again after %s: %s ; asked another kind met so far: %s" % (
                stage, " ; ".join(cells), rws(row["taught_asked_another_kind"][stage])))
        lines.append("    in the loops (all rounds): %s ; the loops took %.0f s" % ("  ".join(
            "%s %s" % (q, rws(v)) for q, v in row["loops"].items()), row["loop_s"]))
        ways = {}
        for state in row["in_use_at_the_end"]:
            for way in state["ways"] + [name_of(p) for p in state["relations"]]:
                ways[way] = ways.get(way, 0) + 1
        lines.append("    in use at the end (in how many of the %d lives): %s" % (
            args.seeds, ", ".join("%s %d" % kv for kv in ways.items()) or "nothing"))
        by_way = {}
        for of_life in row["bought"]:
            for change in of_life:
                entry = by_way.setdefault((change["way"], change["on"]), {"n": 0, "with": {}, "without": {}})
                entry["n"] += 1
                for key in ("with", "without"):
                    for kind, counts in change[key].items():
                        group = "met" if kind in change["met"] else "to come"
                        cell = entry[key].setdefault(group, {})
                        for name, n in counts.items():
                            cell[name] = cell.get(name, 0) + n
        for (way, on), entry in by_way.items():
            lines.append("    %s %s (%d times): at that moment, the kinds met so far: without the change %s, with "
                         "it %s ; the kinds still to come: without %s, with %s" % (
                             way, "taken into use" if on else "put away", entry["n"],
                             rws(entry["without"].get("met", {})), rws(entry["with"].get("met", {})),
                             rws(entry["without"].get("to come", {})), rws(entry["with"].get("to come", {}))))
        lines.append("    the first seed: %s" % row["first"]["loops"])
        for change in (row["bought"][0] if row["bought"] else ()):
            lines.append("      the first seed, moment %d, %s %s during \"%s\": right of 8 by kind, without -> with: %s"
                         % (change["t"], change["way"], "taken into use" if change["on"] else "put away",
                            change["during"], "  ".join("%s %d->%d" % (
                                kind, change["without"][kind]["right"], change["with"][kind]["right"])
                                for kind in KINDS + [NEVER])))
    for row in rows:
        if row["first"]["report"]:
            lines.append("  the first seed's own record (%s):" % row["condition"])
            lines.extend("    " + line for line in row["first"]["report"])
    T.finish(args, rows, lines)


def otherwise_life(seed: int, more: dict):
    """Stage 1 as it is, then four kinds answered otherwise, three rounds each. (mind, the marks by kind, whether
    F53 was in use when each kind began)."""
    s, things, truth = world(seed, more)
    mind = s.mind
    used = 0
    for q, rounds in STAGES[0][1]:
        for x in things[used:used + rounds]:
            teacher(s, "%s ne %s ?" % (x, q), "%s ." % truth[(x, q)])
        used += rounds
    marks_of, had = [], []
    for q in OTHERWISE:
        had.append(bool(mind.cfg.families))
        marks_of.append("".join(teacher(s, "%s ne %s ?" % (x, q), OTHERWISE[q]) for x in things[used:used + 3]))
        used += 3
    return mind, marks_of, had


def control(args) -> None:
    from zihin import deney
    rows, lines = [], ["== what a mind that follows its own record must not do: %d seeds ==" % args.seeds]
    parts = [("(a) F53 on trial (F55 in use, F49 on)", {"faults": True, "trials": True, "asked_place": True}),
             ("(b) raised with F53 on by hand, without F49", {"trials": True, "asked_place": True, "families": True}),
             ("(c) raised with F53 on by hand, with F49", {"faults": True, "trials": True, "asked_place": True,
                                                         "families": True})]
    for label, more in parts:
        taken = dropped = at_end = 0
        better = worse = 0.0
        when, loops, firsts, first = [], {}, {True: {}, False: {}}, None
        for seed in range(args.seeds):
            mind, marks_of, had = otherwise_life(seed, more)
            taken += int(any(c["way"] == "families" and c["on"] for c in mind.trial_changes))
            change = next((c for c in mind.trial_changes if c["way"] == "families" and not c["on"]), None)
            dropped += int(change is not None)
            if change is not None:
                when.append(change["worse"])
            at_end += int(bool(mind.cfg.families))
            w_pos, w_neg = deney.record(mind, "families")
            better, worse = better + w_pos, worse + w_neg
            for marks, with_it in zip(marks_of, had):
                add(loops, marks)
                add(firsts[with_it], marks[0])
            first = first or " ".join("%s:%s" % (q, marks) for q, marks in zip(OTHERWISE, marks_of))
        rows.append({"test": "self_control", "part": label, "seeds": args.seeds, "taken_into_use_in": taken,
                     "put_away_in": dropped, "cases_against_when_put_away": when, "in_use_at_the_end_in": at_end,
                     "better": better, "worse": worse, "loops": loops, "first_answers_with_it": firsts[True],
                     "first_answers_without_it": firsts[False], "first": first})
        lines.append("  %s: F53 taken into use in %d of %d lives, put away in %d (cases against it then: %s), in use "
                     "at the end in %d ; its record over the lives: better %g, worse %g ; the first answer of a kind "
                     "while F53 was in use %s, while it was not %s ; all rounds %s ; the first seed %s" % (
                         label, taken, args.seeds, dropped, when, at_end, better, worse, rws(firsts[True]),
                         rws(firsts[False]), rws(loops), first))
    T.finish(args, rows, lines)


def kept(args) -> None:
    from zihin import persist
    from zihin.__main__ import Session
    same_config = same_answers = asked = 0
    states = []
    for seed in range(args.seeds):
        s, things, truth, _ = life(seed, dict(CONDITIONS[2][1]), measure=False)
        questions = [(x, q) for q in KINDS + [NEVER] for x in things[40:48]]
        before = [probe(s, "%s ne %s ?" % (x, q), "%s ." % truth[(x, q)]) for x, q in questions]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "mind.zihin")
            persist.save(s.mind, path)
            loaded = persist.load(path, ledger=False)
        twin = Session(loaded, s.pieces)
        twin.partner, twin.dil = s.partner, s.dil
        after = [probe(twin, "%s ne %s ?" % (x, q), "%s ." % truth[(x, q)]) for x, q in questions]
        same_config += int(in_use(loaded) == in_use(s.mind) and loaded.relation_paths == s.mind.relation_paths
                           and loaded.trial_changes == s.mind.trial_changes)
        same_answers += sum(a == b for a, b in zip(before, after))
        asked += len(questions)
        states.append({"in_use": in_use(loaded), "right_before": before.count("R"), "right_after": after.count("R")})
    lines = ["== the life of F57 + F58, saved and loaded: %d seeds ==" % args.seeds,
             "  the loaded mind has the configuration, the relations and the record of changes of the saved one in "
             "%d of %d lives ; of %d questions to fresh things (every kind, 8 things) %d are answered as before "
             "saving ; right before / after: %s" % (
                 same_config, args.seeds, asked, same_answers,
                 ", ".join("%d / %d" % (st["right_before"], st["right_after"]) for st in states))]
    T.finish(args, [{"test": "self_kept", "seeds": args.seeds, "same_configuration_in": same_config, "asked": asked,
                     "answered_as_before": same_answers, "lives": states}], lines)


NAMING = [("yer", 6), ("içer", 6), ("sever", 4), ("giyer", 4), ("okur", 4)]


def worlds(args) -> None:
    rows, lines = [], ["== the world where every answer names the thing (\"X V Q .\"), in the teacher's loop: %d seeds ; "
                       "8 fresh things at the end (right / wrong / silent) ==" % args.seeds]
    for label, more in (CONDITIONS[0], CONDITIONS[1], CONDITIONS[3]):
        fresh, loops, final, changes, first, took = {}, {}, [], [], None, 0.0
        for seed in range(args.seeds):
            s, things, truth = world(seed, more)
            mind = s.mind
            used, marks_of = 0, []
            for q, rounds in NAMING:
                t0 = time.time()
                marks = "".join(teacher(s, "%s ne %s ?" % (x, q), "%s %s %s ." % (x, truth[(x, q)], q))
                                for x in things[used:used + rounds])
                took += time.time() - t0
                used += rounds
                marks_of.append("%s:%s" % (q, marks))
                add(loops.setdefault(q, {}), marks)
            for q in [k for k, _ in NAMING] + [NEVER]:
                add(fresh.setdefault(q, {}), "".join(
                    probe(s, "%s ne %s ?" % (x, q), "%s %s %s ." % (x, truth[(x, q)], q)) for x in things[40:48]))
            final.append(in_use(mind))
            changes.append([{"t": c["t"], "way": name_of(c["way"]), "on": c["on"], "better": c["better"],
                             "worse": c["worse"]} for c in getattr(mind, "trial_changes", ())])
            first = first or " ".join(marks_of)
        rows.append({"test": "self_worlds", "world": "the answer names the thing", "condition": label,
                     "seeds": args.seeds, "fresh": fresh, "loops": loops, "in_use_at_the_end": final,
                     "changes": changes, "loop_s": round(took, 1), "first": first})
        ways = {}
        for state in final:
            for way in state["ways"] + [name_of(p) for p in state["relations"]]:
                ways[way] = ways.get(way, 0) + 1
        lines.append("  %-12s fresh: %s ; never asked (%s) %s" % (
            label, "  ".join("%s %s" % (q, rws(fresh[q])) for q, _ in NAMING), NEVER, rws(fresh[NEVER])))
        lines.append("               in the loops: %s ; the loops took %.0f s" % (
            "  ".join("%s %s" % (q, rws(v)) for q, v in loops.items()), took))
        lines.append("               in use at the end (in how many of the %d lives): %s ; the first seed %s" % (
            args.seeds, ", ".join("%s %d" % kv for kv in ways.items()) or "nothing", first))
    T.finish(args, rows, lines)


def traced(args) -> None:
    from zihin.__main__ import Session
    kind = args.kind
    s, things, truth = world(args.seed, dict(CONDITIONS[3][1]))
    values = sorted({truth[(x, kind)] for x in things[:40]})
    used, plain, ruled, taught = 0, [], [], []
    for stage, kinds in STAGES:
        for q, rounds in kinds:
            for x in things[used:used + rounds]:
                teacher(s, "%s ne %s ?" % (x, q), "%s ." % truth[(x, q)])
                if q == kind:
                    taught.append(truth[(x, q)])
            used += rounds
        right = [0, 0]
        for x in things[40:48]:
            for i, rule in enumerate((False, True)):
                copy = Session(s.mind.clone(), s.pieces)
                copy.partner, copy.dil = s.partner, s.dil
                if rule:
                    copy.mind.cfg.find_out = True
                    copy.mind.__dict__["can_look"] = False
                right[i] += int(copy.reply("%s ne %s ?" % (x, kind), limit=30) == truth[(x, kind)])
        plain.append(right[0])
        ruled.append(right[1])
    lines = ["== set by hand, seed %d: 8 fresh things asked \"X ne %s ?\" after each stage (right of 8) ==" % (
                 args.seed, kind),
             "  the values the things taught this kind carry: %s (of the kind's %d values)" % (taught, len(values)),
             "  as the mind is:                         %s" % "  ".join("%d" % n for n in plain),
             "  with F50's rule on in the copies asked: %s" % "  ".join("%d" % n for n in ruled)]
    T.finish(args, [{"test": "self_traced", "seed": args.seed, "kind": kind, "taught_values": taught,
                     "right_by_stage": plain, "right_by_stage_with_the_rule": ruled}], lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["curriculum", "control", "kept", "worlds", "traced"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default="", help="curriculum: one condition only (its name)")
    ap.add_argument("--seed", type=int, default=2, help="traced: the life")
    ap.add_argument("--kind", default="giyer", help="traced: the kind of question asked of the fresh things")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
    new = (set(CLASSES) | {v for values in BY_CLASS.values() for v in values.values()} | set(BY_CLASS)
           | set(ORDER.values()) | {v for values in BY_ORDER.values() for v in values.values()} | set(BY_ORDER))
    overlap = new & (set(T.THINGS) | {v for values in G.VALUES.values() for v in values} | set(G.VERBS))
    assert not overlap and len(new) == 4 + 8 + 2 + 4 + 4 + 1, (overlap, len(new))
    {"curriculum": curriculum, "control": control, "kept": kept, "worlds": worlds, "traced": traced}[args.what](args)


if __name__ == "__main__":
    main()
