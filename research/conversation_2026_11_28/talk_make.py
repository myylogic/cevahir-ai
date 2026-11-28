# -*- coding: utf-8 -*-
"""A yardstick of free conversation, written and frozen before any mechanism was built for it (28 November 2026).

    python talk_make.py           # writes talk.json and talk.sha256

Until today no yardstick measured what the owner does at the terminal: say something, ask about it, say something
else. This one does. It holds a lesson (what a mind is shown before the conversations: sentences told, and
dialogues watched, each form of dialogue with three different names or values) and conversations (turns said one
after another to a mind that has lived the lesson and nothing else; every conversation starts from that state).
The names and values of the conversations do not occur in the lesson.

A turn is graded by what it expects:
  say      one of the listed answers, word for word (case and the closing mark aside): right; nothing: silent;
           anything else: wrong
  silent   nothing is to be said (something was told, not asked): anything said is wrong
  unknown  the thing was never told: the lesson's sentence for not knowing is right, nothing is silent, anything
           else is wrong

Kinds of conversation: told then asked (who speaks changes: what I say of me, you say of you), the newest told
holds, not repeating the speaker (statements that share words with earlier ones), a third one's facts, a thing
never told and the same thing once it is told, a fact asked after other turns, the mind's own facts against the
speaker's, a question in a form the lesson never showed, a conclusion from two told facts, and a form whose
answer needs the inside of a word never shown (reported apart).
"""
import hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DONT = "bunu bilmiyorum"
SHOWN = {"n": ["ali", "veli", "ayşe"], "c": ["ankara", "izmir", "bursa"], "j": ["doktor", "öğretmen", "mühendis"],
         "r": ["iç anadolu", "ege", "marmara"], "p": ["tekir", "karabaş", "boncuk"],
         "t": [("zürafa", "hayvandır"), ("menekşe", "çiçektir"), ("keman", "çalgıdır")],
         "u": ["zeplin", "pusula", "kervan"]}
NEW = {"n": ["kerem", "elif", "deniz", "baran", "selin", "tolga"], "c": ["konya", "samsun", "mardin", "edirne"],
       "j": ["marangoz", "pilot", "aşçı", "terzi"], "r": ["karadeniz", "akdeniz", "doğu anadolu", "trakya"],
       "p": ["pamuk", "duman", "zeytin", "fıstık"],
       "t": [("kerevit", "kabukludur"), ("ıhlamur", "ağaçtır"), ("kanun", "çalgıdır"), ("sincap", "kemirgendir")],
       "u": ["uçurtma", "değirmen", "kandil", "pergel"]}
FORMS = {  # what is told, what is asked, what is answered
    "name": ("benim adım {0} .", "benim adım ne ?", "senin adın {0}"),
    "city": ("ben {0} da yaşıyorum .", "ben nerede yaşıyorum ?", "sen {0} da yaşıyorsun"),
    "job": ("benim mesleğim {0} .", "benim mesleğim ne ?", "senin mesleğin {0}"),
    "his job": ("{1} nin mesleği {0} .", "{1} nin mesleği ne ?", "{1} nin mesleği {0}"),
    "his city": ("{1} {0} da yaşıyor .", "{1} nerede yaşıyor ?", "{1} {0} da yaşıyor"),
    "cat": ("kedimin adı {0} .", "kedimin adı ne ?", "kedinin adı {0}"),
    "dog": ("köpeğimin adı {0} .", "köpeğimin adı ne ?", "köpeğinin adı {0}"),
    "bird": ("kuşumun adı {0} .", "kuşumun adı ne ?", "kuşunun adı {0}"),
}
KEY = {"name": "n", "city": "c", "job": "j", "his job": "j", "his city": "c", "cat": "p", "dog": "p", "bird": "p"}
OWN = [("senin adın ne ?", "benim adım cevahir"), ("sen kimsin ?", "ben cevahirim"),
       ("sen nerede yaşıyorsun ?", "ben bu bilgisayarda yaşıyorum")]


def say(*answers):
    return {"type": "say", "answers": list(answers)}


SILENT, UNKNOWN = {"type": "silent"}, {"type": "unknown"}


def main():
    lesson = []
    for form, (told, asked, answer) in FORMS.items():
        for i, v in enumerate(SHOWN[KEY[form]]):
            who = SHOWN["n"][(i + 1) % 3]
            lesson.append(["tell", told.format(v, who)])
            lesson.append(["watch", asked.format(v, who), answer.format(v, who)])
    for q, a in OWN:
        lesson.append(["watch", q, a])
    for thing, kind in SHOWN["t"]:
        lesson.append(["tell", "%s bir %s ." % (thing, kind)])
        lesson.append(["watch", "%s nedir ?" % thing, "%s bir %s" % (thing, kind)])
    for u in SHOWN["u"]:
        lesson.append(["watch", "%s nedir ?" % u, DONT])
    for i in range(3):
        n, c, r = SHOWN["n"][i], SHOWN["c"][(i + 1) % 3], SHOWN["r"][(i + 1) % 3]
        lesson.append(["tell", "%s %s da yaşıyor ." % (n, c)])
        lesson.append(["tell", "%s %s bölgesindedir ." % (c, r)])
        lesson.append(["watch", "%s hangi bölgede yaşıyor ?" % n, "%s %s bölgesinde yaşıyor" % (n, r)])

    talks = []

    def talk(kind, *turns):
        talks.append({"kind": kind, "turns": [{"said": s, "expect": e} for s, e in turns]})

    N, C, J, R, P, T, U = (NEW[k] for k in "ncjrptu")
    for i, form in enumerate(["name", "name", "city", "city", "job", "job"]):
        v = NEW[KEY[form]][i % 4]
        told, asked, answer = FORMS[form]
        talk("told then asked", (told.format(v), SILENT), (asked, say(answer.format(v))))
    for i, form in enumerate(["name", "city", "job", "name"]):
        a, b = NEW[KEY[form]][i % 4], NEW[KEY[form]][(i + 2) % 4]
        told, asked, answer = FORMS[form]
        talk("the newest told holds", (told.format(a), SILENT), (asked, say(answer.format(a))),
             (told.format(b), SILENT), (asked, say(answer.format(b))))
    for i in range(4):
        n, c, j = N[i], C[i], J[i]
        talk("not repeating the speaker", ("benim adım %s ." % n, SILENT), ("ben %s da yaşıyorum ." % c, SILENT),
             ("benim adım %s ve ben %s da yaşıyorum ." % (n, c), SILENT), ("benim adım %s ." % n, SILENT),
             ("bugün hava çok güzel ." , SILENT), ("ben %s da yaşıyorum ve mesleğim %s ." % (c, j), SILENT),
             ("benim adım ne ?", say("senin adın %s" % n)))
    for i, form in enumerate(["his job", "his city", "his job", "his city"]):
        v, who = NEW[KEY[form]][i], N[(i + 3) % 6]
        told, asked, answer = FORMS[form]
        talk("a third one's facts", (told.format(v, who), SILENT), (asked.format(v, who), say(answer.format(v, who))))
    for i in range(4):
        thing, kind = T[i]
        talk("never told, then told", ("%s nedir ?" % U[i], UNKNOWN), ("%s nedir ?" % thing, UNKNOWN),
             ("%s bir %s ." % (thing, kind), SILENT), ("%s nedir ?" % thing, say("%s bir %s" % (thing, kind))),
             ("%s nedir ?" % U[(i + 1) % 4], UNKNOWN))
    for i in range(3):
        n, c, j, p = N[i + 2], C[i + 1], J[i + 1], P[i]
        talk("asked after other turns", ("benim adım %s ." % n, SILENT), ("ben %s da yaşıyorum ." % c, SILENT),
             ("%s nedir ?" % U[i], UNKNOWN), ("benim mesleğim %s ." % j, SILENT), ("kedimin adı %s ." % p, SILENT),
             ("sen kimsin ?", say("ben cevahirim")), ("benim adım ne ?", say("senin adın %s" % n)),
             ("ben nerede yaşıyorum ?", say("sen %s da yaşıyorsun" % c)), ("kedimin adı ne ?", say("kedinin adı %s" % p)))
    for i in range(3):
        n = N[i + 3]
        talk("the mind's own against the speaker's", ("senin adın ne ?", say("benim adım cevahir")),
             ("benim adım %s ." % n, SILENT), ("senin adın ne ?", say("benim adım cevahir")),
             ("benim adım ne ?", say("senin adın %s" % n)), ("sen kimsin ?", say("ben cevahirim")))
    for i, (asked, form) in enumerate([("adım ne ?", "name"), ("benim adım neydi ?", "name"),
                                       ("nerede yaşıyorum ?", "city"), ("mesleğim ne benim ?", "job")]):
        v = NEW[KEY[form]][(i + 1) % 4]
        told, _, answer = FORMS[form]
        talk("a form never shown", (told.format(v), SILENT), (asked, say(answer.format(v), v)))
    for i in range(3):
        n, c, r = N[i], C[i + 1], R[i]
        talk("from two told facts", ("%s %s da yaşıyor ." % (n, c), SILENT), ("%s %s bölgesindedir ." % (c, r), SILENT),
             ("%s hangi bölgede yaşıyor ?" % n, say("%s %s bölgesinde yaşıyor" % (n, r))))
    for i, (mine, yours) in enumerate([("atımın", "atının"), ("balığımın", "balığının"), ("kaplumbağamın", "kaplumbağanın")]):
        talk("the inside of a word never shown", ("%s adı %s ." % (mine, P[i + 1]), SILENT),
             ("%s adı ne ?" % mine, say("%s adı %s" % (yours, P[i + 1]))))
    out = {"what": "free conversation: a lesson and conversations graded a turn at a time", "dont_know": DONT,
           "lesson": lesson, "talks": talks}
    blob = json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8")
    open(os.path.join(HERE, "talk.json"), "wb").write(blob)
    digest = hashlib.sha256(blob).hexdigest()
    open(os.path.join(HERE, "talk.sha256"), "w").write(digest + "  talk.json\n")
    kinds = {}
    for t in talks:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + len(t["turns"])
    print(json.dumps({"lesson": len(lesson), "talks": len(talks), "turns": sum(kinds.values()), "by_kind": kinds,
                      "sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
