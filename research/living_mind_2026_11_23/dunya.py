# -*- coding: utf-8 -*-
"""dunya.py: a small grounded world for the deepened model (MODEL.md, section 12).

Things with features, people who act on them, events with participants, and a teacher who speaks about what happens,
in Turkish or in English (the same situations in both).

What is the world's and what is the mind's
------------------------------------------
The world's language (the lexicon and the forms below) is written by hand, as the language a child hears is given to
it: the teacher speaks correctly. What a learner receives is only what perception gives and what is said:

* every thing present is a tracked index with its features, each feature an opaque symbol on an opaque dimension
  (``d3:v1``): no feature shares a string with a word, and no dimension is named for the learner;
* an event has opaque action features and, for each participant, opaque participation features (who starts it, who
  moves, who is looked at or turned to, who is touched, who changes, who goes along): what perception reports about
  motion, contact and direction (MODEL.md, section 4.1);
* the words of the teacher.

The splits (which combinations are never shown) are made from the seed alone, before any learner lives the world,
and printed by ``python dunya.py --splits``.

Situations (each a list of lines; a line is something said, perhaps with what is seen while it is said):

* naming: the teacher points at a thing and says what it is ("bu bir at" / "this is a horse");
* order and description: the teacher tells a person to act ("ali ata bak" / "ali look at the horse"); the person
  does it; while it happens the teacher describes it ("ali ata bakıyor" / "ali is looking at the horse");
* facts about kinds, told ("at bir hayvandır" / "a horse is an animal");
* questions with their answers: "X nedir ?" (what is known of a kind), "bu ne ?" (naming), "kim V-ıyor ?" (who does
  the action going on), "N ne yapıyor ?" (what a person is doing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------------------------------- the things
# kind: (features every member has, features that vary between members: dimension -> choices)
KINDS: Dict[str, Tuple[Dict[str, str], Dict[str, List[str]]]] = {
    "insan": ({"canli": "evet", "ayak": "iki", "boy": "orta", "govde": "dik"}, {"sac": ["kara", "sari", "kizil"]}),
    "at": ({"canli": "evet", "ayak": "dort", "boy": "buyuk", "govde": "yatay", "yele": "var"},
           {"renk": ["kahve", "beyaz", "kara"]}),
    "kedi": ({"canli": "evet", "ayak": "dort", "boy": "kucuk", "govde": "yatay", "biyik": "var"},
             {"renk": ["kara", "beyaz", "turuncu"]}),
    "kopek": ({"canli": "evet", "ayak": "dort", "boy": "orta", "govde": "yatay", "kuyruk": "var"},
              {"renk": ["kahve", "beyaz", "kara"]}),
    "kus": ({"canli": "evet", "ayak": "iki", "boy": "kucuk", "kanat": "var"}, {"renk": ["mavi", "sari", "kara"]}),
    "top": ({"canli": "hayir", "bicim": "yuvarlak", "boy": "kucuk"}, {"renk": ["kirmizi", "mavi", "sari"]}),
    "elma": ({"canli": "hayir", "bicim": "yuvarlak", "boy": "kucuk", "sap": "var"},
             {"renk": ["kirmizi", "yesil", "sari"]}),
    "kitap": ({"canli": "hayir", "bicim": "kutu", "boy": "kucuk", "sayfa": "var"},
              {"renk": ["kirmizi", "mavi", "yesil"]}),
    "ev": ({"canli": "hayir", "bicim": "kutu", "boy": "cokbuyuk", "kapi": "var"}, {"renk": ["beyaz", "kirmizi", "sari"]}),
    "okul": ({"canli": "hayir", "bicim": "kutu", "boy": "cokbuyuk", "kapi": "var", "bayrak": "var"},
             {"renk": ["beyaz", "sari"]}),
    "su": ({"canli": "hayir", "bicim": "akiskan"}, {}),
}
THINGS = [k for k in KINDS if k != "insan"]
OBJECT_DIMS = sorted({d for f, v in KINDS.values() for d in list(f) + list(v)})
PEOPLE = {"ali": "kara", "ayse": "sari", "can": "kizil", "zeynep": "kara", "mert": "sari", "elif": "kizil"}

# ----------------------------------------------------------------------------------------------------- the actions
# verb: (action features, the doer's participation, the other's participation, kinds it is done to, Turkish case)
VERBS = {
    "bak": ({"bakis": "var"}, {"baslatir", "bakar"}, {"yonelinir"}, THINGS, "dat"),
    "vur": ({"el": "ani", "temas": "var"}, {"baslatir", "hareket"}, {"dokunulur", "degisir"}, ["top"], "dat"),
    "kos": ({"yer": "hizli"}, {"baslatir", "hareket"}, {"yonelinir"}, ["ev", "okul", "at", "kedi", "kopek", "su"], "dat"),
    "gel": ({"yer": "yavas", "yon": "yakina"}, {"baslatir", "hareket"}, {"yonelinir"}, ["ev", "okul"], "dat"),
    "git": ({"yer": "yavas", "yon": "uzaga"}, {"baslatir", "hareket"}, {"yonelinir"}, ["ev", "okul"], "dat"),
    "gul": ({"ses": "gulme"}, {"baslatir", "ses"}, {"yonelinir"}, ["kedi", "kopek", "kus", "at"], "dat"),
    "tut": ({"el": "surekli", "temas": "var"}, {"baslatir"}, {"dokunulur"}, ["top", "kitap", "elma", "kedi", "kus"], "acc"),
    "al": ({"el": "kaldir", "temas": "var"}, {"baslatir", "hareket"}, {"dokunulur", "birlikte"}, ["top", "kitap", "elma"],
           "acc"),
}

# what the teacher does and never names: pointing at a thing (a gesture perceived like any action)
GESTURES = {"goster": ({"el": "isaret"}, {"baslatir"}, {"yonelinir"}, THINGS, None)}
ACTIONS = dict(VERBS, **GESTURES)
ACTION_DIMS = sorted({d for a, *_ in ACTIONS.values() for d in a})

# ----------------------------------------------------------------------------------------------------- the language
TR_NAME = {"ali": "ali", "ayse": "ayşe", "can": "can", "zeynep": "zeynep", "mert": "mert", "elif": "elif"}
EN_NAME = {"ali": "ali", "ayse": "ayse", "can": "can", "zeynep": "zeynep", "mert": "mert", "elif": "elif"}
# kind: (nominative, dative, accusative, plural, plural genitive)
TR_NOUN = {"at": ("at", "ata", "atı", "atlar", "atların"), "kedi": ("kedi", "kediye", "kediyi", "kediler", "kedilerin"),
           "kopek": ("köpek", "köpeğe", "köpeği", "köpekler", "köpeklerin"),
           "kus": ("kuş", "kuşa", "kuşu", "kuşlar", "kuşların"), "top": ("top", "topa", "topu", "toplar", "topların"),
           "elma": ("elma", "elmaya", "elmayı", "elmalar", "elmaların"),
           "kitap": ("kitap", "kitaba", "kitabı", "kitaplar", "kitapların"), "ev": ("ev", "eve", "evi", "evler", "evlerin"),
           "okul": ("okul", "okula", "okulu", "okullar", "okulların"), "su": ("su", "suya", "suyu", "sular", "suların")}
# verb: (imperative, progressive, past)
TR_VERB = {"bak": ("bak", "bakıyor", "baktı"), "vur": ("vur", "vuruyor", "vurdu"), "kos": ("koş", "koşuyor", "koştu"),
           "gel": ("gel", "geliyor", "geldi"), "git": ("git", "gidiyor", "gitti"), "gul": ("gül", "gülüyor", "güldü"),
           "tut": ("tut", "tutuyor", "tuttu"), "al": ("al", "alıyor", "aldı")}
TR_COLOR = {"kirmizi": "kırmızı", "mavi": "mavi", "sari": "sarı", "yesil": "yeşil", "beyaz": "beyaz", "kara": "kara",
            "kahve": "kahverengi", "turuncu": "turuncu"}
EN_NOUN = {"at": ("horse", "horses"), "kedi": ("cat", "cats"), "kopek": ("dog", "dogs"), "kus": ("bird", "birds"),
           "top": ("ball", "balls"), "elma": ("apple", "apples"), "kitap": ("book", "books"), "ev": ("house", "houses"),
           "okul": ("school", "schools"), "su": ("water", "water")}
# verb: (imperative, progressive participle, past, preposition)
EN_VERB = {"bak": ("look", "looking", "looked", "at"), "vur": ("hit", "hitting", "hit", None),
           "kos": ("run", "running", "ran", "to"), "gel": ("come", "coming", "came", "to"),
           "git": ("go", "going", "went", "to"), "gul": ("laugh", "laughing", "laughed", "at"),
           "tut": ("hold", "holding", "held", None), "al": ("take", "taking", "took", None)}
EN_COLOR = {"kirmizi": "red", "mavi": "blue", "sari": "yellow", "yesil": "green", "beyaz": "white", "kara": "black",
            "kahve": "brown", "turuncu": "orange"}
# what is known of each kind: (what it is, what it has, what it does, a further fact), in Turkish and in English
FACTS = {
    "at": (("at bir hayvandır", "atların dört ayağı vardır", "atlar hızlı koşar", "atların kırık bacağı zor iyileşir"),
           ("a horse is an animal", "horses have four legs", "horses run fast", "a broken leg of a horse heals slowly")),
    "kedi": (("kedi bir hayvandır", "kedilerin dört ayağı vardır", "kediler süt içer", "kediler karanlıkta iyi görür"),
             ("a cat is an animal", "cats have four legs", "cats drink milk", "cats see well in the dark")),
    "kopek": (("köpek bir hayvandır", "köpeklerin dört ayağı vardır", "köpekler havlar", "köpekler kokuyu iyi alır"),
              ("a dog is an animal", "dogs have four legs", "dogs bark", "dogs smell very well")),
    "kus": (("kuş bir hayvandır", "kuşların iki ayağı vardır", "kuşlar uçar", "kuşların kemikleri hafiftir"),
            ("a bird is an animal", "birds have two legs", "birds fly", "the bones of birds are light")),
    "top": (("top bir oyuncaktır", "topların biçimi yuvarlaktır", "toplar yuvarlanır", "toplar havayla dolar"),
            ("a ball is a toy", "balls are round", "balls roll", "balls are filled with air")),
    "elma": (("elma bir meyvedir", "elmaların sapı vardır", "elmalar ağaçta yetişir", "elmalar sonbaharda olgunlaşır"),
             ("an apple is a fruit", "apples have a stem", "apples grow on trees", "apples ripen in autumn")),
    "kitap": (("kitap bir eşyadır", "kitapların sayfaları vardır", "kitaplar okunur", "kitaplar kâğıttan yapılır"),
              ("a book is a thing", "books have pages", "books are read", "books are made of paper")),
    "ev": (("ev bir binadır", "evlerin kapısı vardır", "evlerde insanlar yaşar", "evler tuğladan yapılır"),
           ("a house is a building", "houses have a door", "people live in houses", "houses are made of bricks")),
    "okul": (("okul bir binadır", "okulların bayrağı vardır", "okullarda çocuklar okur", "okullar sabah açılır"),
             ("a school is a building", "schools have a flag", "children study in schools", "schools open in the morning")),
    "su": (("su bir sıvıdır", "suyun rengi yoktur", "su akar", "su yüz derecede kaynar"),
           ("water is a liquid", "water has no colour", "water flows", "water boils at one hundred degrees")),
}
EN_ARTICLE = {"at": "a horse", "kedi": "a cat", "kopek": "a dog", "kus": "a bird", "top": "a ball", "elma": "an apple",
              "kitap": "a book", "ev": "a house", "okul": "a school", "su": "water"}


class Thing:
    """A thing present in a situation: a tracked index, its kind (the world's, never shown) and its features."""

    def __init__(self, index: int, kind: str, feats: Dict[str, str], name: Optional[str] = None) -> None:
        self.index, self.kind, self.feats, self.name = index, kind, dict(feats), name

    def color(self) -> Optional[str]:
        c = self.feats.get("renk")
        return None if c in (None, "yok") else c


class World:
    """The world of one seed: its opaque code book, its splits and its situations."""

    def __init__(self, seed: int = 0, lang: str = "tr") -> None:
        assert lang in ("tr", "en")
        self.seed, self.lang = seed, lang
        rng = random.Random(seed)
        self.rng = rng
        # the opaque code book: dimensions and values become numbers in an order drawn from the seed
        dims = sorted({d for f, v in KINDS.values() for d in list(f) + list(v)}
                      | {d for a, *_ in ACTIONS.values() for d in a})
        rng.shuffle(dims)
        self.dim_code = {d: "d%d" % i for i, d in enumerate(dims)}
        values = sorted({x for f, v in KINDS.values() for x in list(f.values()) + [c for cs in v.values() for c in cs]}
                        | {x for a, *_ in ACTIONS.values() for x in a.values()} | {"yok"})
        rng.shuffle(values)
        self.val_code = {x: "v%d" % i for i, x in enumerate(values)}
        parts = sorted({p for _, d, o, *_ in ACTIONS.values() for p in d | o})
        rng.shuffle(parts)
        self.part_code = {p: "p%d" % i for i, p in enumerate(parts)}
        self._next = 0
        self.people = {n: self.make("insan", {"sac": c}, name=n) for n, c in PEOPLE.items()}
        self.teacher = self.make("insan", {"sac": "kara"}, name="ogretmen")   # present when it points; never named
        self.splits = self.make_splits()

    # ------------------------------------------------------------------------------------------- what is perceived
    def make(self, kind: str, varying: Optional[Dict[str, str]] = None, name: Optional[str] = None) -> Thing:
        fixed, var = KINDS[kind]
        feats = {d: "yok" for d in OBJECT_DIMS}          # what a thing does not have is seen too (no mane, no wings)
        feats.update(fixed)
        for d, choices in var.items():
            feats[d] = (varying or {}).get(d) or self.rng.choice(choices)
        self._next += 1
        return Thing(self._next, kind, feats, name)

    def code(self, dim: str, value: str) -> str:
        return "%s:%s" % (self.dim_code[dim], self.val_code[value])

    def thing_percept(self, t: Thing) -> dict:
        return {"index": t.index, "feats": {self.dim_code[d]: self.val_code[v] for d, v in sorted(t.feats.items())}}

    def event_percept(self, verb: str, doer: Thing, other: Thing) -> dict:
        action, mine, theirs, _, _ = ACTIONS[verb]
        every = sorted(self.part_code.values())
        seen = {d: "yok" for d in ACTION_DIMS}           # every aspect of an action is seen, "yok" where it has none
        seen.update(action)
        return {"action": {self.dim_code[d]: self.val_code[v] for d, v in sorted(seen.items())},
                "parts": [{"index": doer.index, "part": {c: int(c in {self.part_code[p] for p in mine}) for c in every}},
                          {"index": other.index,
                           "part": {c: int(c in {self.part_code[p] for p in theirs}) for c in every}}]}

    def flat(self, things: List[Thing], event: Optional[Tuple[str, Thing, Thing]] = None) -> List[str]:
        """What today's core can take of a scene: one set of facets, the features of every thing present (without
        which thing has which), and of an event its action and participation features (without who has which)."""
        out = set()
        for t in things:
            out |= {self.code(d, v) for d, v in t.feats.items()}
        if event is not None:
            verb, doer, other = event
            action, mine, theirs, _, _ = ACTIONS[verb]
            seen = {d: "yok" for d in ACTION_DIMS}
            seen.update(action)
            out |= {self.code(d, v) for d, v in seen.items()}
            out |= {self.part_code[p] for p in mine | theirs}
        return sorted(out)

    # ------------------------------------------------------------------------------------------- the teacher's words
    def noun(self, t: Thing, case: str = "nom", color: bool = False) -> str:
        if self.lang == "tr":
            if t.name:
                return TR_NAME[t.name]
            word = TR_NOUN[t.kind][{"nom": 0, "dat": 1, "acc": 2}[case]]
            return (TR_COLOR[t.color()] + " " + word) if color else word
        if t.name:
            return EN_NAME[t.name]
        word = EN_NOUN[t.kind][0]
        if t.kind == "su":
            return "the water"
        return "the " + ((EN_COLOR[t.color()] + " ") if color else "") + word

    def order(self, verb: str, doer: Thing, other: Thing, color: bool = False) -> str:
        if self.lang == "tr":
            return "%s %s %s" % (self.noun(doer), self.noun(other, VERBS[verb][4], color), TR_VERB[verb][0])
        imp, _, _, prep = EN_VERB[verb]
        return " ".join(w for w in (self.noun(doer), imp, prep, self.noun(other, color=color)) if w)

    def describe(self, verb: str, doer: Thing, other: Thing, color: bool = False) -> str:
        if self.lang == "tr":
            return "%s %s %s" % (self.noun(doer), self.noun(other, VERBS[verb][4], color), TR_VERB[verb][1])
        _, prog, _, prep = EN_VERB[verb]
        return " ".join(w for w in (self.noun(doer), "is", prog, prep, self.noun(other, color=color)) if w)

    def who(self, verb: str) -> str:
        return ("kim %s ?" % TR_VERB[verb][1]) if self.lang == "tr" else ("who is %s ?" % EN_VERB[verb][1])

    def who_answer(self, verb: str, doer: Thing) -> str:
        return ("%s %s" % (self.noun(doer), TR_VERB[verb][1])) if self.lang == "tr" else \
            ("%s is %s" % (self.noun(doer), EN_VERB[verb][1]))

    def doing(self, doer: Thing) -> str:
        return ("%s ne yapıyor ?" % self.noun(doer)) if self.lang == "tr" else ("what is %s doing ?" % self.noun(doer))

    def this_is(self, t: Thing) -> str:
        return ("bu bir %s" % TR_NOUN[t.kind][0]) if self.lang == "tr" else ("this is %s" % EN_ARTICLE[t.kind])

    def what_this(self) -> str:
        return "bu ne ?" if self.lang == "tr" else "what is this ?"

    def facts(self, kind: str) -> Tuple[str, ...]:
        return FACTS[kind][0 if self.lang == "tr" else 1]

    def what_is(self, kind: str) -> str:
        return ("%s nedir ?" % TR_NOUN[kind][0]) if self.lang == "tr" else ("what is %s ?" % EN_ARTICLE[kind])

    def what_is_answer(self, kind: str) -> str:
        return " . ".join(self.facts(kind)[:3])

    # ------------------------------------------------------------------------------------------- the splits
    def make_splits(self) -> dict:
        """Which combinations are never shown, from the seed alone. Every held-out piece is shown in other
        combinations: a held-out (verb, kind) pair's verb is ordered with other kinds and its kind is the object of
        other verbs; a held-out (person, verb) pair's person does other things and its verb is done by others."""
        rng = random.Random(self.seed * 7919 + 1)
        pairs = [(v, k) for v, spec in VERBS.items() for k in spec[3]]
        held_vk = []
        for v in sorted(VERBS):                          # one held-out kind for each verb done to three or more kinds
            kinds = sorted(VERBS[v][3])
            if len(kinds) >= 3:
                held_vk.append((v, rng.choice(kinds)))
        people = sorted(PEOPLE)
        verbs = sorted(VERBS)
        rng.shuffle(verbs)
        held_pv = [(p, verbs[i % len(verbs)]) for i, p in enumerate(people)]
        colors = [(k, c) for k in THINGS for c in KINDS[k][1].get("renk", [])]
        held_color = []
        for k in THINGS:                                 # of each kind with colours, one colour never named with it;
            cs = sorted(KINDS[k][1].get("renk", []))     # the colour stays named with some other kind
            free = [c for c in cs if any(c in KINDS[o][1].get("renk", []) and (o, c) not in held_color
                                         for o in THINGS if o != k)]
            if len(cs) >= 2 and free:
                held_color.append((k, rng.choice(free)))
        held_what = sorted(rng.sample(THINGS, len(THINGS) // 2))   # kinds never asked "X nedir ?"
        held_seen = []                                   # of each kind, a colour never seen at all (naming test); the
        for k in THINGS:                                 # colour stays named with some other kind
            cs = sorted(KINDS[k][1].get("renk", []))
            named = lambda c: any(c in KINDS[o][1].get("renk", []) and (o, c) not in held_color   # noqa: E731
                                  and (o, c) not in held_seen for o in THINGS if o != k)
            free = [c for c in cs if (k, c) not in held_color and named(c)]
            if len(cs) >= 3 and free:
                held_seen.append((k, rng.choice(free)))
        return {"verb_kind": sorted(held_vk), "person_verb": sorted(held_pv), "color_kind": sorted(held_color),
                "what_is": held_what, "unseen_color": sorted(held_seen), "pairs": len(pairs), "colors": len(colors)}

    def shown(self, person: str, verb: str, kind: str) -> bool:
        return (verb, kind) not in self.splits["verb_kind"] and (person, verb) not in self.splits["person_verb"]

    def color_shown(self, kind: str, color: Optional[str]) -> bool:
        return color is None or (kind, color) not in self.splits["color_kind"]

    def seen_color(self, kind: str, color: Optional[str]) -> bool:
        return color is None or (kind, color) not in self.splits["unseen_color"]

    # ------------------------------------------------------------------------------------------- situations
    def thing_of(self, kind: str, avoid_unseen: bool = True, color: Optional[str] = None) -> Thing:
        var = KINDS[kind][1].get("renk", [])
        if color is None and var:
            choices = [c for c in var if not avoid_unseen or self.seen_color(kind, c)]
            color = self.rng.choice(choices)
        return self.make(kind, {"renk": color} if color else None)

    def line(self, kind: str, text: str, answer: Optional[str] = None, seen: Optional[List[Thing]] = None,
             event: Optional[Tuple[str, Thing, Thing]] = None, pointed: Optional[Thing] = None,
             seen_answer: Optional[List[Thing]] = None, event_answer: Optional[Tuple[str, Thing, Thing]] = None,
             tags: Optional[dict] = None, pointed_answer: Optional[Thing] = None) -> dict:
        """One line: what is said (and the answer of a dialogue), with what is perceived while it is said."""
        def scene(things, ev, point):
            if things is None and ev is None:
                return None
            return {"things": [self.thing_percept(t) for t in (things or [])],
                    "event": self.event_percept(*ev) if ev else None,
                    "pointed": point.index if point is not None else None,
                    "flat": self.flat(things or [], ev)}
        return {"kind": kind, "text": text, "answer": answer, "scene": scene(seen, event, pointed),
                "scene_answer": scene(seen_answer, event_answer, pointed_answer), "tags": tags or {}}

    def naming(self, t: Thing, ask: bool = False) -> dict:
        """The teacher points at a thing and says what it is; or someone asks "bu ne ?" while it points, and the
        answer says what it is."""
        point = ("goster", self.teacher, t)
        present = [self.teacher, t]
        tags = {"thing": t.kind, "color": t.color()}
        if ask:                                          # the teacher goes on pointing while it is answered
            return self.line("what_this", self.what_this(), self.this_is(t), seen=present, event=point, pointed=t,
                             seen_answer=present, event_answer=point, pointed_answer=t, tags=tags)
        return self.line("name", self.this_is(t), seen=present, event=point, pointed=t, tags=tags)

    def act_lines(self, person: str, verb: str, other: Thing, beside: Optional[Thing] = None, ask: Optional[str] = None,
                  kind: str = "order") -> List[dict]:
        """An order, then the doing, described as it happens; perhaps a question about it afterwards. ``beside``: a
        second thing of the same kind present, of another colour: the teacher then says the colour (only then: a
        colour is said to tell the thing apart)."""
        doer = self.people[person]
        present = [doer, other] + ([beside] if beside is not None else [])
        color = beside is not None
        ev = (verb, doer, other)
        tags = {"person": person, "verb": verb, "thing": other.kind, "color": other.color() if color else None}
        out = [self.line(kind, self.order(verb, doer, other, color), self.describe(verb, doer, other, color),
                         seen=present, seen_answer=present, event_answer=ev, tags=tags)]
        if ask == "who":
            out.append(self.line("who", self.who(verb), self.who_answer(verb, doer), seen=present, event=ev,
                                 tags=tags))
        elif ask == "doing":
            out.append(self.line("doing", self.doing(doer), self.describe(verb, doer, other, color), seen=present,
                                 event=ev, tags=tags))
        return out

    def beside(self, t: Thing, test: bool = False) -> Optional[Thing]:
        """A thing of the same kind and of another colour, never one of the colour kept unseen for the naming test
        (None if there is none). It is seen, not named: a colour never said with a kind may stand beside."""
        cs = [c for c in KINDS[t.kind][1].get("renk", []) if c != t.color() and self.seen_color(t.kind, c)]
        return self.thing_of(t.kind, color=sorted(cs)[0] if test else self.rng.choice(cs)) if cs else None

    def training(self, rounds: int = 2) -> List[dict]:
        """The life lived before any test: naming, facts, orders and descriptions, questions, in a shuffled order of
        situations, every combination that is shown shown ``rounds`` times."""
        rng = random.Random(self.seed * 104729 + 3)
        situations: List[List[dict]] = []
        for _ in range(rounds):
            for kind in THINGS:                          # naming: every kind pointed at, in two colours
                for _ in range(2):
                    t = self.thing_of(kind)
                    situations.append([self.naming(t)])
                    if rng.random() < 0.5:
                        situations.append([self.naming(self.thing_of(kind), ask=True)])
            for kind in THINGS:                          # facts told
                for fact in self.facts(kind):
                    situations.append([self.line("fact", fact, tags={"thing": kind})])
                if kind not in self.splits["what_is"]:
                    situations.append([self.line("what_is", self.what_is(kind), self.what_is_answer(kind),
                                                 tags={"thing": kind})])
            for person in sorted(PEOPLE):                # orders and descriptions of every combination shown
                for verb in sorted(VERBS):
                    for kind in VERBS[verb][3]:
                        if not self.shown(person, verb, kind):
                            continue
                        other = self.thing_of(kind)
                        ask = rng.choice([None, None, "who", "doing"])
                        situations.append(self.act_lines(person, verb, other, ask=ask))
                        color = other.color()
                        if color and self.color_shown(kind, color) and rng.random() < 0.5:
                            situations.append(self.act_lines(person, verb, other, beside=self.beside(other)))
        rng.shuffle(situations)
        return [line for s in situations for line in s]

    def tests(self) -> Dict[str, List[dict]]:
        """The questions asked after the life, on copies (nothing is learned from them)."""
        rng = random.Random(self.seed * 15485863 + 5)
        out: Dict[str, List[dict]] = {}
        held = [(p, v, k) for p in sorted(PEOPLE) for v in sorted(VERBS) for k in VERBS[v][3] if not self.shown(p, v, k)]
        seen = [(p, v, k) for p in sorted(PEOPLE) for v in sorted(VERBS) for k in VERBS[v][3] if self.shown(p, v, k)]
        rng.shuffle(seen)
        out["T1_new"] = [self.act_lines(p, v, self.thing_of(k))[0] for p, v, k in held]
        out["T1_seen"] = [self.act_lines(p, v, self.thing_of(k))[0] for p, v, k in seen[:len(held)]]
        out["T2_what_is"] = [self.line("what_is", self.what_is(k), self.what_is_answer(k), tags={"thing": k})
                             for k in self.splits["what_is"]]
        t3 = []
        for kind, color in self.splits["color_kind"]:   # a colour never named with this kind; another one beside it
            verbs = [v for v in sorted(VERBS) if kind in VERBS[v][3]]
            for v in verbs[:2]:
                people = [p for p in sorted(PEOPLE) if self.shown(p, v, kind)]
                if not people:
                    continue
                p = rng.choice(people)
                other = self.thing_of(kind, color=color)
                t3.append(self.act_lines(p, v, other, beside=self.beside(other, test=True))[0])
        out["T3_color"] = t3
        out["T4_name"] = [self.naming(self.thing_of(kind, avoid_unseen=False, color=color), ask=True)
                          for kind, color in self.splits["unseen_color"]]
        t6 = []
        for p, v, k in held[:24]:
            other = self.thing_of(k)
            t6 += self.act_lines(p, v, other, ask=rng.choice(["who", "doing"]))[1:]
        out["T6_situation"] = t6
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lang", default="tr")
    ap.add_argument("--splits", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()
    w = World(args.seed, args.lang)
    if args.splits:
        print(json.dumps(w.splits, ensure_ascii=False, indent=1))
    train = w.training()
    tests = w.tests()
    print("training lines: %d; words: %d" % (len(train), sum(len(l["text"].split()) + len((l["answer"] or "").split())
                                                             for l in train)))
    for name, items in tests.items():
        print("%-14s %d" % (name, len(items)))
    for l in train[:args.sample]:
        print(l["kind"], "|", l["text"], "|", l["answer"] or "", "|", (l["scene"] or {}).get("flat", "")[:6])
    digest = hashlib.md5(json.dumps([train, tests], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    print("world digest:", digest)


if __name__ == "__main__":
    main()
