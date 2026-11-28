# -*- coding: utf-8 -*-
"""Does showing more make the mind say more of what is asked? A yardstick written and frozen before it is run
(28 November 2026).

    python curve_make.py          # writes curve.json and curve.sha256

One library, three lives. In each the same lessons are told and the same eight texts are read (a clock, a guitar,
a bicycle, a bridge, a tent, a lantern, a windmill, a pencil), and the same sixteen requests about the windmill
and the pencil are asked (the third set's: the seven forms and one request in no shown form of each). What differs
is how many things the seven forms of request were shown of, with their answers: two (the clock and the guitar:
this life is the third set's), four (and the bicycle and the bridge), six (and the tent and the lantern). The four
dialogues about subjects of the lessons are shown in every life.

Fixed before the first run (curve_run.py):

 1. no wrong sentence in any life;
 2. kinds (parts, works, made, care, two kinds): the sentences said right do not fall from two shown to four nor
    from four to six, and are more at six than at two; no more is said beside the question for each one right
    at six than at two;
 3. requests in no shown form, at six shown: the mind speaks of both, at least 4 of the 14 right, and of what it
    says for them no more than a third is beside the question;
 4. one sentence, and the whole text, in every life.

Not a criterion, written down as what is expected: at two shown the reading of a request through its own words
has been right in three dialogues of seven and stays silent.
"""
import hashlib, json, os
import speak_make as S1
import speak2_make as S2

HERE = os.path.dirname(os.path.abspath(__file__))
MORE = [("bisiklet", S1), ("köprü", S1), ("çadır", S2), ("fener", S2)]


def dialogues(name, M):
    text, out = M.TEXTS[name], []
    for form, request, tags in S1.REQUESTS:
        must = [text[0][1]] if tags is None else [s for t, s in text if t in tags]
        out.append({"world": "texts", "thing": name, "form": form, "q": request.format(*M.FORMS[name]), "must": must,
                    "may": [s for _, s in text if s not in must]})
    return out


def main():
    third = json.load(open(os.path.join(HERE, "speak3.json"), encoding="utf-8"))
    shown = {"2": third["shown"]}
    for k in (4, 6):
        shown[str(k)] = shown[str(k - 2)] + [d for name, M in MORE[k - 4:k - 2] for d in dialogues(name, M)]
    out = {"what": "the same requests after two, four and six things were shown", "lessons": third["lessons"],
           "texts": third["texts"], "order": third["order"], "shown": shown, "asked": third["asked"]}
    blob = json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8")
    open(os.path.join(HERE, "curve.json"), "wb").write(blob)
    digest = hashlib.sha256(blob).hexdigest()
    open(os.path.join(HERE, "curve.sha256"), "w").write(digest + "  curve.json\n")
    print(json.dumps({"shown": {k: len(v) for k, v in shown.items()}, "asked": len(out["asked"]), "sha256": digest}))


if __name__ == "__main__":
    main()
