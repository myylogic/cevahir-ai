# -*- coding: utf-8 -*-
"""A yardstick for sentences of the mind's own, written and frozen before the mechanism (28 November 2026).

    python inherit_make.py        # writes inherit.json and inherit.sha256

How a person says a thing never told: "a sparrow is a bird" and "birds lay eggs" give "a sparrow lays eggs" (the
hierarchy of Collins and Quillian, 1969). The life: the lessons' 301 sentences; then, in a fixed shuffled order,
what kinds of things do ("kuşlar yumurtlar ."), what each thing is ("serçe bir kuştur ."), and what some kinds are
("kuş bir hayvandır ."). Dialogues are watched for two members each of three kinds: "X ne yapar ?" answered with
what its kind does, said of X. Asked: other members of those kinds; the members of four kinds of which no member
was ever asked; two members of a kind whose bare name was never told; a member told to do otherwise than its kind;
and things nothing of the sort was told of.

A key: ``must`` (what its kind is told to do, said of the thing), ``may`` (true of it by the same step: what else
was told of its kind, of its kind's kind, and what was told of the thing itself) and ``not`` (said of the member
that does otherwise). Any other sentence is wrong. None of the ``must`` sentences was ever told.
"""
import hashlib, json, os, random, re

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = {   # bare name (None: never told bare), plural, "is one", what they do, members, what the kind itself is
 "kuş": ("kuş", "kuşlar", "kuştur", ["yumurtlar", "kanat çırpar", "uçar"], ["serçe", "kartal", "leylek", "martı"], "hayvan"),
 "balık": ("balık", "balıklar", "balıktır", ["solungaçla solunum yapar"], ["sazan", "levrek", "hamsi"], "hayvan"),
 "böcek": ("böcek", "böcekler", "böcektir", ["altı bacakla yürür"], ["karınca", "çekirge", "pervane"], "hayvan"),
 "ağaç": ("ağaç", "ağaçlar", "ağaçtır", ["gölge verir", "yaprak döker"], ["çınar", "kavak", "meşe", "ıhlamur"], "bitki"),
 "çiçek": ("çiçek", "çiçekler", "çiçektir", ["baharda açar"], ["lale", "papatya", "menekşe"], "bitki"),
 "metal": ("metal", "metaller", "metaldir", ["ısıyı iletir", "dövülerek biçimlenir"], ["bakır", "nikel", "krom", "çinko"], None),
 "taşıt": ("taşıt", "taşıtlar", "taşıttır", ["yolcu taşır"], ["tren", "vapur", "otobüs"], None),
 "sürüngen": (None, "sürüngenler", "sürüngendir", ["güneşte ısınır"], ["kertenkele", "timsah"], None),
}
ABOVE = {"hayvan": ("hayvanlar", "hayvandır", ["nefes alır"]), "bitki": ("bitkiler", "bitkidir", ["topraktan su çeker"])}
BARE = {"metal": "metal sert ve parlak bir maddedir .", "taşıt": "taşıt insanları bir yerden bir yere götürür ."}
OTHERWISE = ("penguen", "kuş", "penguen uçamaz .", "uçar")
ALONE = ["deniz tuzludur .", "taş serttir ."]
SHOWN = {"kuş": ["serçe", "kartal"], "metal": ["bakır", "nikel"], "ağaç": ["çınar", "kavak"]}
NEVER_TOLD = ["fırtına", "masa", "bulut"]


def main():
    first = json.load(open(os.path.join(HERE, "speak.json"), encoding="utf-8"))
    lessons = first["lessons"]
    world = [s for s in ALONE] + list(BARE.values()) + [OTHERWISE[2], "%s bir %s ." % (OTHERWISE[0], KINDS[OTHERWISE[1]][2])]
    for kind, (bare, plural, is_one, does, members, above) in KINDS.items():
        world += ["%s %s ." % (plural, d) for d in does] + ["%s bir %s ." % (m, is_one) for m in members]
        if above:
            world.append("%s bir %s ." % (bare, ABOVE[above][1]))
    for above, (plural, _, does) in ABOVE.items():
        world += ["%s %s ." % (plural, d) for d in does]
    random.Random(28).shuffle(world)
    told = lessons + world

    def key(thing, kind):
        bare, plural, is_one, does, members, above = KINDS[kind]
        said = lambda s: thing + " " + s.split(" ", 1)[1]
        must = ["%s %s ." % (thing, d) for d in does]
        names = {plural} | ({bare} if bare else set())
        may = [said(s) for s in told if s.split()[0] in names and said(s) not in must]
        if above:
            may += [said(s) for s in told if s.split()[0] == ABOVE[above][0]]
        may += [s for s in told if thing in re.findall(r"\w+", s)]
        return must, may

    shown, asked = [], []
    for kind, (bare, plural, is_one, does, members, above) in KINDS.items():
        for m in members:
            must, may = key(m, kind)
            form = ("another of a kind that was asked" if kind in SHOWN else
                    "its kind's bare name never told" if bare is None else "of a kind never asked")
            item = {"thing": m, "kind": kind, "form": form, "q": "%s ne yapar ?" % m, "must": must, "may": may, "not": []}
            (shown if m in SHOWN.get(kind, ()) else asked).append(item)
    thing, kind, own, contrary = OTHERWISE
    must, may = key(thing, kind)
    bad = "%s %s ." % (thing, contrary)
    asked.append({"thing": thing, "kind": kind, "form": "told to do otherwise than its kind", "q": "%s ne yapar ?" % thing,
                  "must": [s for s in must if s != bad], "may": may, "not": [bad]})
    for thing in NEVER_TOLD + [s.split()[0] for s in ALONE]:
        asked.append({"thing": thing, "kind": None, "form": "nothing of the sort was told", "q": "%s ne yapar ?" % thing,
                      "must": [], "may": [s for s in told if s.split()[0] == thing], "not": []})
    never = [s for a in asked + shown for s in a["must"] if s in told]
    assert not never, never
    out = {"what": "sentences of the mind's own: what a thing does by what it is said to be", "lessons": lessons,
           "world": world, "shown": shown, "asked": asked}
    blob = json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8")
    open(os.path.join(HERE, "inherit.json"), "wb").write(blob)
    digest = hashlib.sha256(blob).hexdigest()
    open(os.path.join(HERE, "inherit.sha256"), "w").write(digest + "  inherit.json\n")
    forms = {}
    for a in asked:
        forms.setdefault(a["form"], [0, 0])
        forms[a["form"]][0] += 1
        forms[a["form"]][1] += len(a["must"])
    print(json.dumps({"world": len(world), "shown": len(shown), "asked": len(asked), "by_form": forms, "sha256": digest},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
