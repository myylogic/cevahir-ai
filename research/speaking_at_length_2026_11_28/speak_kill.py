# -*- coding: utf-8 -*-
"""The kill experiment for speaking at length (criteria fixed before it was run, 28 November 2026).

    python speak_kill.py [--show]

The thesis: to speak at length is to walk in what was told. A request names a thing; the mind gathers told
sentences about it and says them. How much it says is not given: each request is read in several ways (the first
sentence that names the thing; all that name it; the whole text that begins with it; of those, the sentences
that hold what answers to such a request held before), every way keeps its record over the dialogues watched,
and the way that was right and covered most speaks. Nothing but told sentences can be said.

Here the store is exact (Python dictionaries) and nothing of the package is used: only whether the idea carries
is measured. Forms of a word are found from the vocabulary alone (a word and what follows it in longer words;
an ending counts once two different words take it).

Killed if any of these fails on the requests asked (speak.json, frozen):
  1. wrong sentences (told of another thing, or never told) are at most 5 % of the sentences said;
  2. "X nedir ?" for the bicycle and for the bridge is answered with exactly one sentence, the first of its text;
  3. "X anlat" for the two is answered with at least 90 % of the text's sentences;
  4. of the lessons' subjects, at least 80 % of the sentences that name them are said.
Reported, not a criterion: how much of the kinds of sentence (parts, works, made, care, two kinds) is said, and
of the request in no shown form.
"""
import argparse, hashlib, json, os, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SHA = "3ba5745dd3db795f6bdc9528957b6b2e02d3c6d7d2747fbd7fd922e85f8c03b1"
LINE = 0.5 + 1 / 12
WAYS = ("first", "first of its text", "named", "its text", "marked among named", "marked in its text")


def rule(w_plus, w):
    return (w_plus + 0.5) / (w + 1.0)


def words(s):
    return re.findall(r"\w+", s.lower())


class Proto:
    def __init__(self):
        self.told, self.toks, self.text_of, self.texts = [], [], [], []
        self.holds = defaultdict(list)                       # word -> the sentences that hold it
        self.frames = {}                                     # frame -> ways' records and marks
        self._stem = None

    # ------------------------------------------------------------------------------------------- living ---------
    def tell(self, sentence, text=None):
        i = len(self.told)
        self.told.append(sentence)
        self.toks.append(words(sentence))
        self.text_of.append(text)
        for w in set(self.toks[-1]):
            self.holds[w].append(i)
        self._stem = None

    def read(self, sentences):
        t = len(self.texts)
        self.texts.append((len(self.told), len(self.told) + len(sentences)))
        for s in sentences:
            self.tell(s, t)

    # -------------------------------------------------------------------------------- the forms of a word -------
    def _learn_forms(self):
        V = set(self.holds)
        takes = defaultdict(set)                             # an ending -> the words that take it
        for w in V:
            for k in range(3, len(w)):
                if w[:k] in V:
                    takes[w[k:]].add(w[:k])
        known = {e for e, ws in takes.items() if len(ws) >= 2}
        memo = {"": True}

        def chain(r):                                        # what remains is made of known endings
            if r not in memo:
                memo[r] = any(r.startswith(e) and chain(r[len(e):]) for e in known)
            return memo[r]
        self._stem = {}
        for w in V:
            self._stem[w] = next((w[:k] for k in range(3, len(w)) if w[:k] in V and chain(w[k:])), w)
        self.known_endings = known
        self.of_stem = defaultdict(set)
        for w, s in self._stem.items():
            self.of_stem[s].add(w)

    def stem(self, w):
        if self._stem is None:
            self._learn_forms()
        if w in self._stem:
            return self._stem[w]
        for k in range(len(w) - 1, max(2, len(w) - 3), -1):  # a form never told of a told word: two letters at most
            if w[:k] in self._stem:
                return self._stem[w[:k]]
        return w

    def named(self, w):
        s = self.stem(w)
        return sorted({i for f in self.of_stem.get(s, ()) for i in self.holds.get(f, ())})

    def stems_of(self, i):
        return {self.stem(w) for w in self.toks[i]}

    # ----------------------------------------------------------------------------------- reading a request ------
    def ways(self, frame, filler):
        n = self.named(filler)
        t = [i for a, b in self.texts if a in n for i in range(a, b)]
        marks = {s for s, (a, b) in self.frames.get(frame, {}).get("marks", {}).items() if rule(a, a + b) >= LINE}
        marked = lambda pool: [i for i in pool if self.stems_of(i) & marks]
        return {"first": n[:1], "first of its text": t[:1], "named": n, "its text": t,
                "marked among named": marked(n), "marked in its text": marked(t)}

    def frames_of(self, request):
        toks = words(request)
        for k, w in enumerate(toks):
            if self.named(w):
                yield " ".join(toks[:k] + ["_"] + toks[k + 1:]), w

    def watch(self, request, answer):
        want = {self.told.index(s) for s in answer if s in self.told}
        for frame, filler in self.frames_of(request):
            said = self.ways(frame, filler)                  # foreseen before anything is written
            f = self.frames.get(frame)
            if f is None:                                    # born of this dialogue: it says nothing yet
                f = self.frames[frame] = {"ways": {w: [0, 0, 0, 0] for w in WAYS}, "marks": {}}
            else:
                for w, out in said.items():
                    if out:
                        rec, hit = f["ways"][w], len(want & set(out))
                        rec[0] += rule(hit, len(out)) >= LINE        # what it would have said was the answer
                        rec[1] += 1
                        rec[2] += hit
                        rec[3] += len(want)
            pool = set(said["named"]) | set(said["its text"])
            own = self.stem(filler)
            for i in pool:                                   # what the sentences of the answer hold, and the others
                for s in self.stems_of(i) - {own}:
                    f["marks"].setdefault(s, [0, 0])[0 if i in want else 1] += 1

    def answer(self, request):
        best = None
        for frame, filler in self.frames_of(request):
            f = self.frames.get(frame)
            if f is None:
                continue
            said = self.ways(frame, filler)
            for order, w in enumerate(WAYS):
                right, tried, covered, of = f["ways"][w]
                if tried and said[w] and rule(right, tried) >= LINE:
                    rank = (covered / max(1, of), rule(right, tried), -len(said[w]), -order)
                    if best is None or rank > best[0]:
                        best = (rank, w, said[w])
        return (best[1], best[2]) if best else ("", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    blob = open(os.path.join(HERE, "speak.json"), "rb").read()
    assert hashlib.sha256(blob).hexdigest() == SHA, "speak.json is not the frozen set"
    Y = json.loads(blob.decode("utf-8"))
    m = Proto()
    for s in Y["lessons"]:
        m.tell(s)
    for name in Y["order"]:
        m.read(Y["texts"][name])
    for d in Y["shown"]:
        m.watch(d["q"], d["must"])
    rows, total = [], defaultdict(int)
    for d in Y["asked"]:
        way, out = m.answer(d["q"])
        said = [m.told[i] for i in out]
        must, may = set(d["must"]), set(d["may"])
        r = {"world": d["world"], "thing": d["thing"], "form": d["form"], "q": d["q"], "way": way, "said": len(said),
             "right": sum(s in must for s in said), "beside": sum(s in may for s in said),
             "wrong": sum(s not in must and s not in may for s in said), "must": len(must), "answer": said}
        rows.append(r)
        print("%-8s %-9s %-20s %-20s said %2d: right %2d of %2d, beside %2d, wrong %2d" % (
            r["world"], r["thing"], r["form"], way or "(silent)", r["said"], r["right"], r["must"], r["beside"], r["wrong"]))
        if a.show:
            for s in said:
                print("      %s %s" % ("+" if s in must else ("~" if s in may else "!"), s))
    said = sum(r["said"] for r in rows)
    wrong = sum(r["wrong"] for r in rows)
    c1 = wrong <= 0.05 * said
    one = [r for r in rows if r["form"] == "one sentence"]
    c2 = all(r["said"] == 1 and r["right"] == 1 for r in one)
    c3 = all(r["right"] >= 0.9 * r["must"] for r in rows if r["form"] == "all")
    les = [r for r in rows if r["world"] == "lessons"]
    c4 = sum(r["right"] for r in les) >= 0.8 * sum(r["must"] for r in les)
    kinds = [r for r in rows if r["form"] in ("parts", "works", "made", "care", "two kinds")]
    result = {"what": "kill experiment: speaking at length on an exact store", "sha256": SHA,
              "said": said, "wrong": wrong, "wrong_share": round(wrong / max(1, said), 4),
              "1_wrong_at_most_5_percent": c1, "2_one_sentence": c2, "3_all_of_it": c3,
              "4_lessons": [sum(r["right"] for r in les), sum(r["must"] for r in les), c4],
              "kinds_right_of_must": [sum(r["right"] for r in kinds), sum(r["must"] for r in kinds)],
              "kinds_beside": sum(r["beside"] for r in kinds),
              "no_shown_form": [(r["right"], r["must"], r["wrong"]) for r in rows if r["form"] == "in no shown form"],
              "survives": bool(c1 and c2 and c3 and c4), "known_endings": len(m.known_endings),
              "rows": [{k: v for k, v in r.items() if k != "answer"} for r in rows]}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "kill.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
    forms = {w: sorted(m.of_stem[m.stem(w)]) for w in ("bisiklet", "köprü", "kan", "veri", "güneş", "hücre", "enerji")}
    print("forms found:", json.dumps(forms, ensure_ascii=False))


if __name__ == "__main__":
    main()
