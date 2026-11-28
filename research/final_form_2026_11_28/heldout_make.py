# -*- coding: utf-8 -*-
"""The held-out set of this study: written once, on 28 November 2026, before the design and before any code of the
final form existed, and not looked at again until the last measurement (gate G6).

    python heldout_make.py            # writes heldout.json beside this file and prints its sha256 and item counts

Each world is a small life and questions with their keys. The worlds ask for the abilities the task names, each in
words and frames that the development batteries (SCAN, COGS, the school, the facts that change, the lesson
dialogues) do not use, drawn by a seeded generator so that nobody chose the items:

- ``compose``: a made-up command language with the shape of SCAN's grammar (other words, other action tokens, a
  modifier that repeats three times): a primitive lived only alone, a template never lived, new combinations;
- ``deduce``: members told to be of classes, a property told of each class; asked of members never asked, of a class
  none of whose members was ever asked, of an exception, and of words nothing was told about (key: silence);
- ``change``: where things are, told and told again; right is the place told last;
- ``substitute``: frames with places (code-like answers), asked with fillers never lived in them or never lived at
  all;
- ``sequence``: lived sequences and questions answered with what follows;
- ``recall``: one-off facts under interference, watched once or only told;
- ``conversation``: what a person says about themselves in conversation, asked for later (a fresh mind, and the
  mind raised on the lessons); two things never told (key: silence); everyday lines with no key (printed only);
- ``forget``: secrets lived and then forgotten (key: silence), the rest still answered;
- ``text``: a Turkish document of the repository that no corpus of the studies holds, for bits per character.

A question's ``mode`` says how it is graded: ``exact`` (the answer's words, closing marks apart, are one of the
keys), ``holds`` (the answer holds a key and none of the ``not`` tokens), ``silent`` (nothing may be said) or
``none`` (no key exists: it is printed and never counted as right).
"""
from __future__ import annotations

import hashlib
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SEED = 20261128
TEXT = "docs/ZIHIN_KILAVUZ.md"


def q(question, keys, mode="exact", tag="", no=()):
    return {"q": question, "keys": list(keys), "mode": mode, "tag": tag, "not": list(no)}


# ----------------------------------------------------------------------------------------------------- compose ---
def compose(rng):
    prims = rng.sample(["tara", "sil", "boya", "kes", "ölç", "del", "çiz", "sar", "bük", "ser"], 5)
    labels = rng.sample(["K1", "M2", "P3", "R4", "S5", "T6", "V7", "Z8", "D9", "G0", "B3", "N7"], 7)
    act = dict(zip(prims, labels[:5]))
    dirs = rng.sample(["ileri", "geri", "yukarı", "aşağı"], 2)
    turn = dict(zip(dirs, labels[5:]))
    twice, thrice = rng.sample(["çift", "ikile", "üçle", "katla"], 2)
    around = rng.choice(["tur", "dolan", "sarmal"])
    c_and = rng.choice(["ardından", "sonra"])
    c_after = rng.choice(["öncesinde", "önce"])
    known, new = prims[:4], prims[4]

    def verbs(u, with_held):
        out = [([u], [act[u]])]
        for d in dirs:
            out.append(([u, d], [turn[d], act[u]]))
        out.append(([u, around, dirs[0]], [turn[dirs[0]], act[u]] * 3))
        if with_held:
            out.append(([u, around, dirs[1]], [turn[dirs[1]], act[u]] * 3))
        return out

    def phrases(vs):
        out = []
        for c, a in vs:
            out += [(c, a), (c + [twice], a * 2), (c + [thrice], a * 3)]
        return out

    def join(x, y, how):
        return (x[0] + [how] + y[0], x[1] + y[1]) if how == c_and else (x[0] + [how] + y[0], y[1] + x[1])

    lived_s = [p for u in known for p in phrases(verbs(u, False))]
    held_s = [p for u in known for p in phrases([v for v in verbs(u, True) if v[0][-2:] == [around, dirs[1]]])]
    new_s = [p for p in phrases(verbs(new, False)) if p[0] != [new]]
    pairs = [(x, y, how) for x in lived_s for y in lived_s for how in (c_and, c_after)]
    rng.shuffle(pairs)
    lived_c = [join(*p) for p in pairs[:300]]
    rest = pairs[300:]
    lived = lived_s + [([new], [act[new]])] + lived_c
    life = []
    for epoch in range(2):
        block = list(lived)
        rng.shuffle(block)
        if epoch == 0:
            block.sort(key=lambda d: len(d[0]))
        life += [["watch", " ".join(c), " ".join(a) + " ."] for c, a in block]
    ask = []
    for c, a in rng.sample(new_s, 10):
        ask.append(q(" ".join(c), [" ".join(a)], tag="a primitive lived only alone"))
    for _ in range(5):
        x, y, how = rng.choice(new_s), rng.choice(lived_s), rng.choice([c_and, c_after])
        c, a = join(x, y, how) if rng.random() < 0.5 else join(y, x, how)
        ask.append(q(" ".join(c), [" ".join(a)], tag="a primitive lived only alone"))
    for c, a in rng.sample(held_s, 6):
        ask.append(q(" ".join(c), [" ".join(a)], tag="a template never lived"))
    for _ in range(4):
        x, y, how = rng.choice(held_s), rng.choice(lived_s), rng.choice([c_and, c_after])
        c, a = join(x, y, how)
        ask.append(q(" ".join(c), [" ".join(a)], tag="a template never lived"))
    for x, y, how in rest[:15]:
        c, a = join(x, y, how)
        ask.append(q(" ".join(c), [" ".join(a)], tag="a new combination"))
    longest = sorted(rest[15:], key=lambda p: -(len(p[0][1]) + len(p[1][1])))[:10]
    for x, y, how in longest:
        c, a = join(x, y, how)
        ask.append(q(" ".join(c), [" ".join(a)], tag="the longest new combinations"))
    return {"name": "compose", "mind": "fresh", "variant": "speaking", "window": 13,
            "switches": {"wholes": True, "told_apart": True}, "life": life, "ask": ask}


# ------------------------------------------------------------------------------------------------------ deduce ---
CLASSES = [("meyvedir", "meyveler", "olgunlaşır", ["elma", "armut", "kiraz", "şeftali", "incir"]),
           ("metaldir", "metaller", "paslanır", ["bakır", "çinko", "nikel", "kalay", "krom"]),
           ("çiçektir", "çiçekler", "açar", ["gül", "lale", "papatya", "menekşe", "zambak"]),
           ("böcektir", "böcekler", "vızıldar", ["arı", "sinek", "çekirge", "karınca", "kelebek"]),
           ("taşıttır", "taşıtlar", "ilerler", ["otobüs", "kamyon", "tramvay", "vapur", "bisiklet"]),
           ("yıldızdır", "yıldızlar", "parlar", ["sirius", "vega", "rigel", "antares", "kapella"])]


def deduce(rng):
    classes = rng.sample(CLASSES, 5)
    frame = rng.choice(["{x} ne yapar ?", "{x} nasıl davranır ?"])
    unasked = classes[4]
    exc_class = classes[rng.randrange(4)]
    exception = exc_class[3][4]
    exc_prop = rng.choice([c[2] for c in classes if c is not exc_class])
    watched, ask = [], []
    for copula, plural, prop, members in classes[:4]:
        order = [m for m in members if m != exception]
        rng.shuffle(order)
        for m in order[:2]:
            watched.append((frame.format(x=m), "%s %s ." % (m, prop)))
            ask.append(q(frame.format(x=m), ["%s %s" % (m, prop)], tag="recall"))
        for m in order[2:]:
            ask.append(q(frame.format(x=m), ["%s %s" % (m, prop)], tag="deduction: a member never asked"))
    for m in unasked[3]:
        ask.append(q(frame.format(x=m), ["%s %s" % (m, unasked[2])],
                     tag="deduction: no member of its class ever asked"))
    ask.append(q(frame.format(x=exception), ["%s %s" % (exception, exc_prop)], tag="the exception",
                 no=[exc_class[2]]))
    for w in rng.sample(["masa", "bulut", "kalem", "deniz"], 2):
        ask.append(q(frame.format(x=w), [], mode="silent", tag="control: nothing was told of it"))
    life = []
    for day in range(2):
        sentences = []
        for copula, plural, prop, members in classes:
            sentences += ["%s bir %s ." % (m, copula) for m in members]
            sentences += ["%s %s ." % (plural, prop)] * 2
        sentences.append("%s %s ." % (exception, exc_prop))
        rng.shuffle(sentences)
        dialogues = list(watched)
        rng.shuffle(dialogues)
        for i, s in enumerate(sentences):
            life.append(["tell", s])
            if i % 3 == 2 and dialogues:
                life.append(["watch", *dialogues.pop()])
        life += [["watch", *d] for d in dialogues]
    return {"name": "deduce", "mind": "fresh", "variant": "full", "life": life, "ask": ask}


# ------------------------------------------------------------------------------------------------------ change ---
def change(rng):
    things = rng.sample(["anahtar", "gözlük", "kitap", "telefon", "çanta", "şemsiye", "cüzdan", "kalem", "defter",
                         "saat", "bardak", "makas", "fener", "harita", "bilet", "ilaç", "mektup", "oyuncak", "top",
                         "terlik", "mum", "kibrit", "ip", "çekiç"], 20)
    places = ["masada", "rafta", "çekmecede", "dolapta", "sepette", "kutuda"]
    kinds = ["unchanged", "changed once", "changed after three", "changed twice after three", "changed back"]
    streams = []
    for name in things[:10]:
        steps = []
        for _ in range(3):
            place = rng.choice(places)
            steps += [["tell", "%s %s ." % (name, place)],
                      ["watch", "%s nerede ?" % name, "%s %s ." % (name, place)]]
        streams.append(steps)
    ask = []
    for i, name in enumerate(things[10:]):
        kind = kinds[i % len(kinds)]
        old, new = rng.sample(places, 2)
        told = {"unchanged": [old], "changed once": [old, new], "changed after three": [old] * 3 + [new],
                "changed twice after three": [old] * 3 + [new] * 2, "changed back": [old, new, old]}[kind]
        streams.append([["tell", "%s %s ." % (name, p)] for p in told])
        other = new if told[-1] == old else old
        ask.append(q("%s nerede ?" % name, ["%s %s" % (name, told[-1])], tag=kind,
                     no=[other] if len(set(told)) > 1 else []))
    life = []
    while any(streams):
        s = rng.choice([s for s in streams if s])
        life.append(s.pop(0))
    return {"name": "change", "mind": "fresh", "variant": "speaking", "life": life, "ask": ask}


# -------------------------------------------------------------------------------------------------- substitute ---
def substitute(rng):
    names = rng.sample(["en", "boy", "alan", "hız", "kütle", "süre", "yol", "ısı", "güç", "ağırlık", "hacim",
                        "basınç", "enerji", "sıcaklık", "derinlik", "yükseklik"], 14)
    f1, f2, unseen = names[:6], names[6:11], names[11:]
    numbers = rng.sample(["2", "3", "4", "6", "8", "9", "12", "15", "20", "25"], 8)
    ring = [(f1[i], f1[(i + 1) % len(f1)]) for i in range(len(f1))]      # every filler lived in both places
    pairs = [(a, b) for a in f1 for b in f1 if a != b and (a, b) not in ring]
    rng.shuffle(pairs)
    pairs = ring + pairs
    lived = [["watch", "%s ile %s çarp" % (a, b), "%s * %s ." % (a, b)] for a, b in pairs[:8]]
    lived += [["watch", "%s değerini göster" % v, "göster ( %s ) ." % v] for v in f2]
    lived += [["watch", "%s kere dön" % n, "tekrar %s :" % n] for n in numbers[:5]]
    life = []
    for _ in range(2):
        block = list(lived)
        rng.shuffle(block)
        life += block
    ask = []
    for a, b in pairs[8:12]:
        ask.append(q("%s ile %s çarp" % (a, b), ["%s * %s" % (a, b)], tag="a new pair of lived fillers"))
    for _ in range(3):
        a, b = rng.choice(f1), rng.choice(f2)
        a, b = (a, b) if rng.random() < 0.5 else (b, a)
        ask.append(q("%s ile %s çarp" % (a, b), ["%s * %s" % (a, b)], tag="a filler lived in another frame"))
    for w in unseen:
        a = rng.choice(f1)
        a, b = (a, w) if rng.random() < 0.5 else (w, a)
        ask.append(q("%s ile %s çarp" % (a, b), ["%s * %s" % (a, b)], tag="a word never lived"))
    for v in rng.sample(f1, 3):
        ask.append(q("%s değerini göster" % v, ["göster ( %s )" % v], tag="a filler lived in another frame"))
    for w in unseen:
        ask.append(q("%s değerini göster" % w, ["göster ( %s )" % w], tag="a word never lived"))
    for n in numbers[5:]:
        ask.append(q("%s kere dön" % n, ["tekrar %s :" % n], tag="a word never lived"))
    return {"name": "substitute", "mind": "fresh", "variant": "full", "life": life, "ask": ask}


# ---------------------------------------------------------------------------------------------------- sequence ---
def sequence(rng):
    days = ["pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar"]
    months = ["ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim", "kasım",
              "aralık"]
    life = [["tell", " ".join(days) + " ."]] * 3 + [["tell", " ".join(months) + " ."]] * 3
    ask, watched = [], []
    order = list(range(len(days) - 1))
    rng.shuffle(order)
    for k, i in enumerate(order):
        item = ("%s den sonra ne gelir ?" % days[i], days[i + 1])
        if k < 3:
            watched.append(item)
            ask.append(q(item[0], [item[1]], tag="recall"))
        else:
            ask.append(q(item[0], [item[1]], tag="relation: what follows"))
    order = list(range(len(months) - 1))
    rng.shuffle(order)
    for k, i in enumerate(order):
        item = ("%s ayından sonraki ay hangisi ?" % months[i], months[i + 1])
        if k < 5:
            watched.append(item)
            ask.append(q(item[0], [item[1]], tag="recall"))
        else:
            ask.append(q(item[0], [item[1]], tag="relation: what follows"))
    order = list(range(len(days) - 2))
    rng.shuffle(order)
    for k, i in enumerate(order):
        item = ("%s den iki gün sonrası hangi gün ?" % days[i], days[i + 2])
        if k < 3:
            watched.append(item)
            ask.append(q(item[0], [item[1]], tag="recall"))
        else:
            ask.append(q(item[0], [item[1]], tag="relation: two steps on"))
    block = [["watch", a, b + " ."] for a, b in watched] * 2
    rng.shuffle(block)
    life = life + block
    rng.shuffle(life)
    return {"name": "sequence", "mind": "fresh", "variant": "full", "life": life, "ask": ask}


# ------------------------------------------------------------------------------------------------------ recall ---
def recall(rng):
    things = rng.sample(["perde", "halı", "yastık", "battaniye", "sandalye", "koltuk", "lamba", "vazo", "tabak",
                         "kaşık", "çatal", "tencere", "fincan", "sürahi", "havlu", "sabun", "tarak", "ayna", "fırça",
                         "kova", "süpürge", "merdiven", "kapı", "pencere", "çatı", "duvar", "bahçe", "çit", "kulübe",
                         "köprü", "kule", "tekne", "yelken", "çadır", "kazak", "atkı", "eldiven", "şapka", "çorap",
                         "kemer"], 40)
    colours = ["kırmızı", "mavi", "yeşil", "sarı", "mor", "turuncu", "beyaz", "siyah", "gri", "pembe"]
    life, ask = [], []
    for i, t in enumerate(things):
        c = rng.choice(colours)
        if i < 30:
            life.append(["watch", "%s hangi renk ?" % t, "%s %s ." % (t, c)])
            ask.append(q("%s hangi renk ?" % t, ["%s %s" % (t, c)], tag="watched once"))
        else:
            life.append(["tell", "%s %s ." % (t, c)])
            ask.append(q("%s hangi renk ?" % t, ["%s %s" % (t, c)], tag="told once, never asked"))
    rng.shuffle(life)
    rng.shuffle(ask)
    return {"name": "recall", "mind": "fresh", "variant": "full", "life": life, "ask": ask}


# ------------------------------------------------------------------------------------------------ conversation ---
FACTS = [("benim adım {v}", "benim adım ne ?", ["yasin", "elif", "murat", "zehra", "kerem", "aylin", "orhan"]),
         ("benim yaşım {v}", "benim yaşım kaç ?", ["27", "34", "41", "19", "52", "63"]),
         ("en sevdiğim renk {v}", "en sevdiğim renk ne ?", ["mavi", "yeşil", "mor", "turuncu", "kırmızı", "sarı"]),
         ("kedimin adı {v}", "kedimin adı ne ?", ["pamuk", "duman", "zeytin", "boncuk", "tarçın"]),
         ("en sevdiğim yemek {v}", "en sevdiğim yemek ne ?", ["mantı", "pilav", "çorba", "börek", "makarna"]),
         ("doğduğum şehir {v}", "doğduğum şehir neresi ?", ["izmir", "bursa", "konya", "adana", "samsun"])]
SMALL_TALK = ["merhaba", "nasılsın", "bugün hava çok güzel", "biraz yorgunum", "kahve içtim", "teşekkür ederim",
              "akşam yürüyüşe çıkacağım", "dün geç yattım"]
NO_KEY = ["ne düşünüyorsun ?", "bana bir şey anlat", "bugün ne yapalım ?"]


def conversation(rng, mind):
    facts = []
    for statement, question, pool in FACTS:
        v = rng.choice(pool)
        facts.append((statement, question, pool, v))
    life = []
    order = list(facts)
    rng.shuffle(order)
    for statement, question, pool, v in order:
        life.append(["converse", statement.format(v=v)])
        for line in rng.sample(SMALL_TALK, rng.choice([1, 2])):
            life.append(["converse", line])
    order = list(facts)
    rng.shuffle(order)
    for statement, question, pool, v in order:
        life.append(["ask", q(question, [v], mode="holds", tag="told in conversation, asked later",
                              no=[p for p in pool if p != v])])
    statement, question, pool, v = facts[0]                      # the name changes
    new = rng.choice([p for p in pool if p != v])
    life.append(["converse", statement.format(v=new)])
    life.append(["converse", rng.choice(SMALL_TALK)])
    life.append(["ask", q(question, [new], mode="holds", tag="told again otherwise, asked later",
                          no=[p for p in pool if p != new])])
    for question in ["babamın adı ne ?", "arabamın rengi ne ?"]:
        life.append(["ask", q(question, [], mode="silent", tag="control: never told")])
    for line in NO_KEY:
        life.append(["ask", q(line, [], mode="none", tag="everyday line, no key")])
    return {"name": "conversation (%s mind)" % mind, "mind": mind, "variant": "full", "ledger": True,
            "life": life, "ask": []}


# ------------------------------------------------------------------------------------------------------ forget ---
def forget(rng):
    people = rng.sample(["nermin", "tarık", "sevgi", "bülent", "gönül", "ilker", "fatoş", "necati"], 8)
    syll = ["ka", "ne", "ri", "to", "lu", "me", "sa", "vi", "do", "pe"]
    codes = []
    while len(codes) < 8:
        c = "".join(rng.sample(syll, 3))
        if c not in codes:
            codes.append(c)
    life = [["watch", "%s in kodu nedir ?" % p, "%s ." % c] for p, c in zip(people, codes)]
    rng.shuffle(life)
    secrets = rng.sample(range(8), 2)
    ask = []
    for k in secrets:
        life.append(["forget", k])
    for k, item in enumerate(life[:8]):
        code = item[2][:-2]
        if k in secrets:
            ask.append(q(item[1], [], mode="silent", tag="forgotten", no=[code]))
        else:
            ask.append(q(item[1], [code], tag="kept"))
    return {"name": "forget", "mind": "fresh", "variant": "full", "ledger": True, "life": life, "ask": ask}


def main() -> None:
    rng = random.Random(SEED)
    worlds = [compose(rng), deduce(rng), change(rng), substitute(rng), sequence(rng), recall(rng),
              conversation(rng, "fresh"), conversation(rng, "raised"), forget(rng)]
    with open(os.path.join(REPO, TEXT), "rb") as f:
        raw = f.read()
    worlds.append({"name": "text", "mind": "fresh", "variant": "full", "path": TEXT,
                   "sha256": hashlib.sha256(raw).hexdigest(), "chars": len(raw.decode("utf-8")), "life": [],
                   "ask": []})
    out = {"what": "held-out set of research/final_form_2026_11_28 (gate G6)", "seed": SEED, "worlds": worlds}
    blob = json.dumps(out, ensure_ascii=False, indent=0, sort_keys=True).encode("utf-8")
    path = os.path.join(HERE, "heldout.json")
    with open(path, "wb") as f:
        f.write(blob)
    digest = hashlib.sha256(blob).hexdigest()
    with open(os.path.join(HERE, "heldout.sha256"), "w", encoding="utf-8") as f:
        f.write("%s  heldout.json\n" % digest)
    print("sha256", digest)
    for w in worlds:
        asked = len(w["ask"]) + sum(1 for item in w["life"] if item[0] == "ask")
        print("  %-28s life %4d items, asked %3d" % (w["name"], len(w["life"]), asked))


if __name__ == "__main__":
    main()
