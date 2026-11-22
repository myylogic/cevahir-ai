# -*- coding: utf-8 -*-
"""t_conj.py: F59 Conjectures, measured. Statements nobody told the mind and nobody asked for, formed in its own time
from what a kind of link was found to carry.

    python research/living_mind_2026_11_22/t_conj.py fields  [--seeds 5] [--rename]
    python research/living_mind_2026_11_22/t_conj.py settled [--seeds 5]
    python research/living_mind_2026_11_22/t_conj.py asked   [--seeds 5]

The world: three fields with words of their own (animals, bodies, numbers). In each field: things; classes; of
every class two statements ("K V Q ."); of every thing its class ("X K <link> ."), one fact of its own ("X V Q' .")
and one sentence that links it to another thing of the field in another way ("X Y <other> .": it chases, scratches,
follows it). All sentences are read once, in one random order over the three fields; after each the mind has one
silent moment, and at the end silent moments until it has nothing left to think over (at most 400; how many it
took is reported). What holds in the world and is told of no thing: every thing has its class's
two statements ("X V Q ."). Whether the mind comes to those by itself is the measure; nothing in it names a word of
this world.

``fields``, the conditions (the same sentences but for what is named):

* ``told in one field``: the class's two statements are also told of six animals themselves (the evidence that the
  class link carries what is told of the class); of no body and no number;
* ``nothing told of things``: no such statement of any thing (no evidence that any link carries anything);
* ``told otherwise``: the six animals are told those statements with another value (evidence that it does not);
* ``another word for the link``: as the first, but bodies and numbers are linked to their classes with words of
  their own, not the animals' word;
* ``another word, two things told``: as that, and in each of the two other fields the statements are told of two
  things of one class;
* ``one statement, one field``: one of the class's statements is told of one animal, nothing else of any thing;
* ``one statement in each of two fields``: that, and one statement told of one body: neither field alone holds
  more than one experience, the two together do (the record of a kind of link is one record over the fields);
* ``F59 off``.

Measured: the conjectures the mind made (how many hold in the world, how many do not), which of the statements
that hold and were told of no thing it came to, by field, and the silent moments in which it thought something over.
``--rename``: every word of the world replaced by a random token (the counts must not change).

``settled``: the life of ``told in one field``; then (a) the statements are told as they are of half of the things of
every field: what its conjectures came to; (b) in one field one class's statement is told *otherwise* of two of its
things, and a thing of that class nobody mentioned before is introduced ("Z K <link> ."): which of the class's two
statements the mind conjectures of it.

``asked``: the life of ``told in one field``, with a mind that can be asked ("X ne Q ?" -> "V ."; F53, F54 and F55 on,
two kinds of question watched about the six animals that were told): the things of the other two fields and the
other animals are asked what was told of none of them (on copies), with F59 and without.

Appends to ``results/conj.jsonl``.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import t_fault as T  # noqa: E402

OUT = os.path.join(HERE, "results", "conj.jsonl")
FIELDS = {
    "animals": {"things": ["aslan", "kedi", "yılan", "kertenkele", "serçe", "kartal", "levrek", "sazan", "kurt",
                           "timsah", "baykuş", "hamsi"],
                "classes": ["memeli", "sürüngen", "kanatlı", "sucul"],
                "of": {"doğurur": ["yavru", "yumurta", "civciv", "larva"],
                       "örtünür": ["kıl", "zırh", "telek", "mukus"]},
                "link": "sınıfındandır", "own_link": "sınıfındandır", "other": "kovalar",
                "own": ("yer", ["ot", "et", "balık", "tohum"]), "new": "vaşak"},
    "bodies": {"things": ["demir", "bakır", "cam", "seramik", "cıva", "alkol", "helyum", "neon"],
               "classes": ["metal", "yalıtkan", "sıvı", "gaz"],
               "of": {"iletir": ["akımı", "hiçbirini", "basıncı", "sesi"],
                      "alır": ["biçim", "çatlak", "kap", "hacim"]},
               "link": "sınıfındandır", "own_link": "türündendir", "other": "çizer",
               "own": ("tartar", ["gram", "kilo", "ton", "miligram"]), "new": "argon"},
    "numbers": {"things": ["altı", "sekiz", "yedi", "dokuz", "onbir", "oniki", "onüç", "ondört"],
                "classes": ["çift", "tek", "asal", "kare"],
                "of": {"bölünür": ["ikiye", "kendine", "birine", "köküne"],
                       "verir": ["yarım", "kalan", "hiç", "kenar"]},
                "link": "sınıfındandır", "own_link": "kümesindendir", "other": "izler",
                "own": ("eder", ["toplam", "çarpım", "fark", "bölüm"]), "new": "onbeş"},
}
CONDITIONS = [("told in one field", {}),
              ("nothing told of things", {"told": 0}),
              ("told otherwise", {"contradict": True}),
              ("another word for the link", {"own_links": True}),
              ("another word, two things told", {"own_links": True, "also": 2}),
              ("one statement, one field", {"told": 0, "single": ("animals",)}),
              ("one statement in each of two fields", {"told": 0, "single": ("animals", "bodies")}),
              ("F59 off", {"switches": {}})]


class World:
    """The sentences of one life and what holds in it."""

    def __init__(self, seed: int, told: int = 6, contradict: bool = False, own_links: bool = False, also: int = 0,
                 single=(), rename: bool = False) -> None:
        rng = random.Random(seed)
        self.name = {}
        self.rename = rename
        self.rng = random.Random(seed + 1000)
        self.sentences, self.holds, self.told, self.field_of = [], set(), set(), {}
        self.kind, self.links = {}, {}
        for field, f in FIELDS.items():
            things = list(f["things"])
            rng.shuffle(things)
            link = f["own_link"] if own_links else f["link"]
            self.links[field] = link
            kind = {x: f["classes"][i % len(f["classes"])] for i, x in enumerate(things)}
            self.kind.update(kind)
            verb, values = f["own"]
            for q, by in f["of"].items():
                for k, v in zip(f["classes"], by):
                    self.sentences.append((k, v, q))
            for i, x in enumerate(things):
                self.field_of[x] = field
                self.sentences.append((x, kind[x], link))
                self.sentences.append((x, rng.choice(values), verb))
                self.sentences.append((x, things[(i + 3) % len(things)], f["other"]))
                for q, by in f["of"].items():
                    value = by[f["classes"].index(kind[x])]
                    self.holds.add((x, value, q))
                    if field in single:                    # one statement of one thing of the field
                        named = i == 0 and q == next(iter(f["of"]))
                    elif field == "animals":
                        named = i < told
                    else:                                  # ``also`` things of one class
                        named = i % len(f["classes"]) == 0 and i < also * len(f["classes"])
                    if named:
                        wrong = rng.choice([v for v in by if v != value])
                        self.sentences.append((x, wrong if contradict else value, q))
                        self.told.add((x, value, q))
            self.things = getattr(self, "things", {})
            self.things[field] = things
        rng.shuffle(self.sentences)

    def word(self, w: str) -> str:
        if not self.rename:
            return w
        if w not in self.name:
            self.name[w] = "w%d" % self.rng.randrange(10 ** 6, 10 ** 7)
        return self.name[w]

    def text(self, words) -> str:
        return " ".join(self.word(w) for w in words)

    def untold(self):
        return self.holds - self.told


def live(world: World, switches=None, silent: int = 1, rest: int = 400):
    """One mind reads the world's sentences, a silent moment after each, and at the end silent moments until it
    has nothing left to think over (at most ``rest``). (session, [the silent moments in which it came to a
    conjecture, the silent moments at the end], seconds)."""
    from zihin import IMITATED
    mind, s = T.make_mind(world.rng.randrange(10 ** 6), False, None,
                          dict({"conjectures": True} if switches is None else switches))
    thought = [0, 0]

    def quiet(n: int, until_done: bool = False) -> int:
        lived = 0
        while lived < n and not (until_done and lived and not getattr(mind, "pondering", None)):
            before = len(getattr(mind, "conjectures", ()))
            was = getattr(mind, "t", 0)
            s.silent_moment(say=lambda *_: None)
            thought[0] += int(mind.t - was > 1 or len(getattr(mind, "conjectures", ())) > before)
            lived += 1
        return lived
    t0 = time.time()
    for words in world.sentences:
        tell(s, world.text(words) + " .")
        quiet(silent)
    thought[1] = quiet(rest, until_done=True)
    s.quiet = quiet
    return s, thought, time.time() - t0


def tell(s, sentence: str) -> None:
    from zihin import IMITATED
    s.company(None)
    s.listen(sentence, IMITATED)
    s.end_of_turn(IMITATED, tone=".")


def conjectured(world: World, mind) -> set:
    """The statements the mind conjectured, in the world's own words."""
    back = {v: k for k, v in world.name.items()}
    out = set()
    for entry in getattr(mind, "conjectures", {}).values():
        words = tuple(str(mind.symbol_of(u)[1]) for u in entry["statement"])
        out.add(tuple(back.get(w, w) for w in words))
    return out


def fields(args) -> None:
    rows, lines = [], ["== three fields, the statements of things told in one of them at most: %d seeds%s ==" % (
        args.seeds, " (every word a random token)" if args.rename else "")]
    for label, kw in CONDITIONS:
        kw = dict(kw)
        switches = kw.pop("switches", None)
        made = right = wrong = moments = rested = 0
        found = {field: [0, 0] for field in FIELDS}
        took, examples = 0.0, []
        for seed in range(args.seeds):
            world = World(seed, rename=args.rename, **kw)
            s, thought, seconds = live(world, switches)
            mine = conjectured(world, s.mind)
            made += len(mine)
            right += len(mine & world.holds)
            wrong += len(mine - world.holds)
            examples += [" ".join(c) for c in sorted(mine - world.holds)][:2]
            for fact in world.untold():
                cell = found[world.field_of[fact[0]]]
                cell[0] += int(fact in mine)
                cell[1] += 1
            moments += thought[0]
            rested += thought[1]
            took += seconds
        rows.append({"test": "conj_fields", "condition": label, "seeds": args.seeds, "renamed": bool(args.rename),
                     "made": made, "hold": right, "do_not_hold": wrong, "found_of_untold": found,
                     "moments_it_came_to_one": moments, "silent_moments_at_the_end": rested,
                     "sentences": len(world.sentences), "life_s": round(took, 1),
                     "examples_that_do_not_hold": examples[:4]})
        lines.append("  %-30s conjectures %3d (hold %3d, do not %2d) ; of what holds and was told of no thing: %s ; "
                     "silent moments in which it came to one %d, silent moments at the end %d ; %d sentences a "
                     "life, %.0f s" % (
                         label, made, right, wrong, "  ".join("%s %d/%d" % (f, a, b) for f, (a, b) in found.items()),
                         moments, rested, len(world.sentences), took))
    T.finish(args, rows, lines)


def settled(args) -> None:
    from zihin import varsayim
    held = failed = still = 0
    made_before = 0
    new_right = new_exception = new_other = 0
    worth_before, worth_after = [], []
    for seed in range(args.seeds):
        world = World(seed)
        s, _, _ = live(world)
        mind = s.mind
        made_before += len(mind.conjectures)
        worth_before.append(round(varsayim.worth(mind), 3))
        told_now = []
        for field, things in world.things.items():          # (a) half of the things of every field, as it is
            for x in things[len(things) // 2:]:
                for fact in sorted(world.holds):
                    if fact[0] == x and fact not in world.told:
                        told_now.append(fact)
        field = "numbers"                                   # (b) one class's statement told otherwise of two things
        f = FIELDS[field]
        k = f["classes"][1]
        q = "verir"
        value = f["of"][q][1]
        members = [x for x in world.things[field] if world.kind[x] == k]
        other = next(v for v in f["of"][q] if v != value)
        exceptions = [(x, other, q) for x in members[:2]]
        told_now = [fact for fact in told_now if not (fact[0] in members[:2] and fact[2] == q)]
        for words in told_now + exceptions:
            tell(s, world.text(words) + " .")
            s.quiet(1)
        s.quiet(400, until_done=True)
        tell(s, world.text((f["new"], k, world.links[field])) + " .")      # a thing of that class, new to the mind
        s.quiet(400, until_done=True)
        states = [e["state"] for e in mind.conjectures.values()]
        held += states.count("held")
        failed += states.count("failed")
        still += states.count("open")
        worth_after.append(round(varsayim.worth(mind), 3))
        mine = conjectured(world, mind)
        new_right += int((f["new"], f["of"]["bölünür"][1], "bölünür") in mine)
        new_exception += int((f["new"], value, q) in mine)
        new_other += len([c for c in mine if c[0] == f["new"]]) - int((f["new"], f["of"]["bölünür"][1], "bölünür")
                                                                       in mine) - int((f["new"], value, q) in mine)
    lines = ["== what the conjectures came to when the mind was told more: %d seeds ==" % args.seeds,
             "  %d conjectures before ; then told of half of the things as it is, and one class's statement told "
             "otherwise of two of its things: held %d, failed %d, still open %d ; what thinking things over is worth "
             "to it, before -> after: %s -> %s" % (made_before, held, failed, still, worth_before, worth_after),
             "  a thing of that class new to the mind: the class's other statement conjectured of it in %d of %d "
             "lives ; the statement that failed of two things conjectured of it in %d ; anything else in %d" % (
                 new_right, args.seeds, new_exception, new_other)]
    T.finish(args, [{"test": "conj_settled", "seeds": args.seeds, "conjectures_before": made_before, "held": held,
                     "failed": failed, "open": still, "worth_before": worth_before, "worth_after": worth_after,
                     "new_thing_other_statement": new_right, "new_thing_failed_statement": new_exception,
                     "new_thing_anything_else": new_other}], lines)


def asked(args) -> None:
    from zihin.__main__ import Session
    rows, lines = [], ["== asked what was told of no thing ('X ne Q ?' -> 'V .'), on copies: %d seeds ==" %
                       args.seeds]
    base = {"faults": True, "families": True, "lived_apart": True, "asked_place": True}
    for label, switches in (("without F59", dict(base)), ("with F59", dict(base, conjectures=True))):
        counts = {field: {"right": 0, "wrong": 0, "silent": 0} for field in FIELDS}
        for seed in range(args.seeds):
            world = World(seed)
            s, _, _ = live(world, switches)
            for _ in range(2):                              # two kinds of question, watched about the animals told
                for fact in sorted(world.told):
                    s.watch(world.text((fact[0], "ne", fact[2])) + " ?", world.text((fact[1],)) + " .")
            s.close_turn()
            if switches.get("conjectures"):
                s.quiet(400, until_done=True)
            for fact in sorted(world.untold()):
                copy = Session(s.mind.clone(), s.pieces)
                copy.partner, copy.dil = s.partner, s.dil
                said = copy.reply(world.text((fact[0], "ne", fact[2])) + " ?", limit=30)
                key = "right" if said == world.text((fact[1],)) else ("silent" if said == "(sessiz)" else "wrong")
                counts[world.field_of[fact[0]]][key] += 1
        rows.append({"test": "conj_asked", "condition": label, "seeds": args.seeds, "asked": counts})
        lines.append("  %-12s %s" % (label, "  ".join("%s %d/%d/%d" % (f, c["right"], c["wrong"], c["silent"])
                                                     for f, c in counts.items())))
    T.finish(args, rows, lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fields", "settled", "asked"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rename", action="store_true", help="fields: every word of the world a random token")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
    words = [w for f in FIELDS.values() for w in f["things"] + f["classes"] + [f["link"], f["own_link"], f["other"],
                                                                              f["own"][0], f["new"]] + f["own"][1]
             + list(f["of"]) + [v for vs in f["of"].values() for v in vs]]
    assert {w for w in words if words.count(w) > 1} == {"sınıfındandır"}     # only the shared link word repeats
    {"fields": fields, "settled": settled, "asked": asked}[args.what](args)


if __name__ == "__main__":
    main()
