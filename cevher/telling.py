# -*- coding: utf-8 -*-
"""The telling grain: speaking at length (docs/architecture/FINAL_FORM.md, section 18).

To speak at length is to walk in what was told. The mind keeps the sentences it was told, as they were told (the
library), and ties in the substance from every word to the sentences that hold it. A request names a thing; the
mind gathers told sentences about it and says them. How much it says is not given by a rule: a request is read in
several ways (the first sentence of the text that begins with the thing; the first that names it; its whole text;
all that name it; of those, the sentences that hold what answers to such a request held before), every way of
every form of request keeps its record in the substance over the dialogues watched, and the way that was right and
covered most of what was answered speaks. A form of request says nothing until it has been shown of two things.

Nothing but told sentences can be said. What the mind was never shown how to answer, it does not answer.

The forms of a word are found from the vocabulary alone: a told word and what follows it in longer told words; an
ending counts once two different words take it. (A language that marks its words at their beginning is not served.)

This first form keeps what was told a sentence at a time and texts shorter than the bulk threshold (they are lived
a sentence at a time); long texts are lived as before and are not told from yet.
"""
from __future__ import annotations

import re
from array import array
from typing import Dict, List, Tuple

import numpy as np

from cevher.units import LINE, body, code_of, fold, mix, rule, tokens

R_SAME, R_HOLDS, R_FORM, R_WAY, R_COVER, R_MARK, R_MARKS, R_FREE, R_SEEN, PLACE = (
    code_of("\0telling " + name) for name in ("the sentence", "holds", "form of request", "way", "covered", "mark",
                                              "marks of all forms", "through marks", "shown of", "place"))
R_LINK, BEGINS, ENDS = (code_of("\0telling " + name) for name in ("the place of what it is", "begins", "ends"))
WAYS = ("first of its text", "first", "its text", "named", "marked in its text", "marked among named",
        "what is told of what it is")
FIXED = 4                                                    # the first four are not made from what was answered
PAGE = 4                                                     # sentences a page of a word's ties
CLOSE = re.compile(r"(?<=[.!?…])\s+")


def _words(toks) -> List[str]:
    return [t for t in toks if t[:1].isalnum()]


class Telling:
    def __init__(self, mind, most_bytes: int = 32 << 20, most_words: int = 200000):
        self.mind = mind
        self.most_bytes, self.most_words = most_bytes, most_words
        self.blob = bytearray()                              # the sentences as they were told, one after another
        self.offs = array("Q", [0])
        self.text = array("I")                               # the text a sentence was read in (0: told on its own)
        self.texts = 0
        self.dead = set()                                    # sentences forgotten
        self.not_kept = 0
        self.words = set()                                   # the words told (their forms are found among them)
        self.heard = set()                                   # the words of requests watched: they attest endings too
        self._memo = None                                    # while one request is read: the sentences that name a word
        self._said: Dict[int, list] = {}                     # the words of a told sentence
        self._stem = None

    def __len__(self) -> int:
        return len(self.text)

    @property
    def lived(self):
        S = self.mind.S
        return getattr(S, "lived", S)

    # -------------------------------------------------------------------------------------------- living --------
    def new_text(self) -> int:
        self.texts += 1
        return self.texts

    def tell(self, raw: str, toks: List[str], text: int = 0) -> None:
        """A sentence told: kept as it was told, and tied to every word it holds."""
        words = _words(toks)
        data = raw.encode("utf-8", "surrogatepass")
        if not words:
            return
        if len(self.blob) + len(data) > self.most_bytes:
            self.not_kept += 1
            return
        n = len(self.text)
        self.blob += data
        self.offs.append(len(self.blob))
        self.text.append(text)
        m = self.mind
        codes = [m.code(w) for w in words]
        m._add(fold(tuple(codes), R_SAME), n + 1)
        for c in dict.fromkeys(codes):
            base = mix(R_HOLDS, c)
            m._add(mix(base, self.lived.tot(base) // PAGE + 1), n + 1)
            m._bump(base)
        if len(self.words) < self.most_words:
            fresh = [w for w in words if w not in self.words]
            if fresh:
                self.words.update(fresh)
                self._stem = None

    def kill(self, a: int, b: int) -> None:
        """Sentences a to b are forgotten (their ties are taken back by the ledger; they are never said again)."""
        self.dead.update(range(a, b))

    def sentence(self, n: int) -> str:
        return bytes(self.blob[self.offs[n]:self.offs[n + 1]]).decode("utf-8", "surrogatepass")

    # ---------------------------------------------------------------------------------- the forms of a word -----
    def _learn_forms(self) -> None:
        V = self.words
        met = V | self.heard
        takes: Dict[str, set] = {}
        for w in met:
            for k in range(3, len(w)):
                if w[:k] in met:
                    takes.setdefault(w[k:], set()).add(w[:k])
        known = {e for e, ws in takes.items() if len(ws) >= 2}
        memo = {"": True}

        def chain(r: str) -> bool:                           # what remains is made of known endings
            if r not in memo:
                memo[r] = any(r.startswith(e) and chain(r[len(e):]) for e in known)
            return memo[r]
        self._stem = {w: next((w[:k] for k in range(3, len(w)) if w[:k] in V and chain(w[k:])), w) for w in V}
        self._forms: Dict[str, list] = {}
        for w, s in self._stem.items():
            self._forms.setdefault(s, []).append(w)

    def stem(self, w: str) -> str:
        if self._stem is None:
            self._learn_forms()
        s = self._stem.get(w)
        if s is not None:
            return s
        for k in range(len(w) - 1, max(2, len(w) - 3), -1):  # a form never told of a told word: two letters at most
            if w[:k] in self._stem:
                return self._stem[w[:k]]
        return w

    def named(self, w: str) -> List[int]:
        """The told sentences that name a word, in any of its forms, in the order they were told."""
        s = self.stem(w)
        if self._memo is not None and s in self._memo:
            return self._memo[s]
        S, out = self.lived, set()
        for form in self._forms.get(s, ()):
            base = mix(R_HOLDS, code_of(form))
            for page in range(S.tot(base) // PAGE + 1):
                out.update(v - 1 for v, _, _ in S.vals(mix(base, page + 1)))
        out = sorted(n for n in out if n < len(self.text) and n not in self.dead)
        if self._memo is not None:
            self._memo[s] = out
        return out

    def said_in(self, n: int) -> list:
        """The words of a told sentence."""
        ws = self._said.get(n)
        if ws is None:
            ws = self._said[n] = _words(body(tokens(self.sentence(n)))[0])
        return ws

    # ------------------------------------------------------------------- what a thing is said to be -------------
    def _steps(self, thing: str, seen: set = frozenset()) -> list:
        """From a thing to what is told of what it is said to be. A sentence that names the thing holds other
        words; of some of them told sentences begin (in any of their forms). Each such word is a step: the place
        it stands in (the word before it and the word after), the word, and the sentences that begin with it."""
        out, own = [], self.stem(thing)
        for n in self.named(thing):
            ws = self.said_in(n)
            for i, k in enumerate(ws):
                ks = self.stem(k)
                if ks == own or ks in seen:
                    continue
                begun = [g for g in self.named(k) if g != n and self.stem(self.said_in(g)[0]) == ks
                         and own not in {self.stem(x) for x in self.said_in(g)}]
                if begun:
                    place = mix(code_of(ws[i - 1]) if i else BEGINS, code_of(ws[i + 1]) if i + 1 < len(ws) else ENDS)
                    out.append((place, ks, k, begun))
        return out

    def _inherit(self, frame: int, thing: str, depth: int = 0, seen: set = frozenset()) -> List[int]:
        """The sentences told of what a thing is said to be, and of what that is said to be: through the places
        this form of request has found to hold what a thing is."""
        out: List[int] = []
        for place, ks, k, begun in self._steps(thing, seen):
            a, t = self._record(mix(mix(R_LINK, frame), place))
            if t and rule(a, t) >= LINE:
                out += begun
                if depth < 2:
                    out += self._inherit(frame, k, depth + 1, seen | {self.stem(thing), ks})
        return out

    def _own(self, thing: str, begun: List[int]) -> list:
        """Sentences of the mind's own: each told sentence with the thing in the place of its first word, once."""
        out, had = [], set()
        for g in begun:
            words = self.words_of((g, thing))
            if words not in had:
                had.add(words)
                out.append((g, thing))
        return out

    def words_of(self, item) -> tuple:
        """The words of what would be said: a told sentence, or one with the thing in the place of its first word."""
        if isinstance(item, tuple):
            return (item[1],) + tuple(self.said_in(item[0])[1:])
        return tuple(self.said_in(item))

    def stems_of(self, n: int) -> set:
        return {self.stem(w) for w in self.said_in(n)}

    def its_text(self, numbers: List[int]) -> List[int]:
        """The texts that begin with one of these sentences, whole."""
        out = []
        for n in numbers:
            t = self.text[n]
            if t and (n == 0 or self.text[n - 1] != t):
                b = n
                while b < len(self.text) and self.text[b] == t:
                    b += 1
                out += [i for i in range(n, b) if i not in self.dead]
        return out

    # ---------------------------------------------------------------------------------- reading a request -------
    def _record(self, key: int) -> Tuple[int, int]:
        S = self.mind.S
        return S.get(key, 1), S.tot(key)

    def _judge(self, key: int, right: bool) -> None:
        if right:
            self.mind._tie(key, 1)
        else:
            self.mind._bump(key)

    def _marks(self, frame: int, n: int) -> bool:
        for s in self.stems_of(n):
            a, t = self._record(mix(mix(R_MARK, frame), code_of(s)) if frame else mix(R_MARKS, code_of(s)))
            if t and rule(a, t) >= LINE:
                return True
        return False

    def _admits(self, frame: int, n: int) -> bool:
        """A sentence that holds something the answers to this form held is admitted; so is one that holds
        nothing the form has seen. One that holds only what stood beside the answers is not: a way that takes a
        sentence for where it stands (the first, any that names the thing) does not say it."""
        against = False
        for s in self.stems_of(n):
            a, t = self._record(mix(mix(R_MARK, frame), code_of(s)))
            if t:
                e = rule(a, t)
                if e >= LINE:
                    return True
                against = against or e <= 1.0 - LINE
        return not against

    def ways(self, frame: int, filler: str) -> List[List[int]]:
        named = self.named(filler)
        text = self.its_text(named)
        return [[n for n in text[:1] if self._admits(frame, n)], [n for n in named[:1] if self._admits(frame, n)],
                [n for n in text if self._admits(frame, n)], [n for n in named if self._admits(frame, n)],
                [n for n in text if self._marks(frame, n)], [n for n in named if self._marks(frame, n)],
                self._own(filler, self._inherit(frame, filler))]

    def forms_of(self, q: List[str]):
        """The forms a request may have: the request with one word that names something left open."""
        codes = [self.mind.code(t) for t in q]
        for k, w in enumerate(q):
            if w[:1].isalnum() and self.named(w):
                yield fold(tuple(codes[:k]) + (PLACE,) + tuple(codes[k + 1:]), R_FORM), w

    def through_marks(self, q: List[str]) -> List[int]:
        """A request in no known form, read through what the known forms have learned: of the thing's text (or of
        the sentences that name it), those that hold a word of the request which marked answers before."""
        asked = {self.stem(w) for w in _words(q)}
        marks = {s for s in asked if (lambda a, t: t and rule(a, t) >= LINE)(*self._record(mix(R_MARKS, code_of(s))))}
        best: List[int] = []
        for w in _words(q):
            named = self.named(w)
            if named:
                aspects = marks - {self.stem(w)}
                out = [n for n in (self.its_text(named) or named) if self.stems_of(n) & aspects]
                if len(out) > len(best):
                    best = out
        return best

    def watch(self, q: List[str], answer: str) -> None:
        """A dialogue watched. If its answer is made of told sentences, or of what is told of what the thing is
        said to be, every way of reading the request is judged by what it would have said, before anything of the
        dialogue is written."""
        self._memo = {}
        try:
            self._watch(q, answer)
        finally:
            self._memo = None

    def _watch(self, q: List[str], answer: str) -> None:
        S, m, want, answered = self.mind.S, self.mind, set(), set()
        fresh = [w for w in _words(q) if w not in self.heard and w not in self.words]
        if fresh and len(self.heard) < self.most_words:
            self.heard.update(fresh)
            self._stem = None
        for piece in CLOSE.split(answer.strip()):
            words = _words(body(tokens(piece))[0])
            if words:
                answered.add(tuple(words))
                want.update(v - 1 for v, _, _ in S.vals(fold(tuple(m.code(w) for w in words), R_SAME)))
        want = {n for n in want if n < len(self.text) and n not in self.dead}
        forms = list(self.forms_of(q))
        steps = {filler: self._steps(filler) for _, filler in forms}
        if not want and not any(self.words_of((g, filler)) in answered
                                for filler, found in steps.items() for _, _, _, begun in found for g in begun):
            return
        out = self.through_marks(q)
        if out:
            self._judge(R_FREE, rule(len(want & set(out)), len(out)) >= LINE)
        for frame, filler in forms:
            seen = mix(mix(R_SEEN, frame), code_of(self.stem(filler)))
            if S.tot(seen):                                  # the same thing asked again shows nothing new
                continue
            m._bump(seen)
            said = self.ways(frame, filler)
            born = S.tot(frame) == 0
            m._bump(frame)
            for j, out in enumerate(said):
                if out and not (born and j >= FIXED):
                    hit = sum(self.words_of(x) in answered for x in out)
                    self._judge(mix(mix(R_WAY, frame), j + 1), rule(hit, len(out)) >= LINE)
                    cover = mix(mix(R_COVER, frame), j + 1)
                    if hit:
                        m._add(cover, 1, hit)
                    m._bump(cover, len(answered))
            for place, _, _, begun in steps[filler]:         # was what stands in this place what the thing is?
                hit = sum(self.words_of((g, filler)) in answered for g in begun)
                self._judge(mix(mix(R_LINK, frame), place), rule(hit, len(begun)) >= LINE)
            own = self.stem(filler)
            named = self.named(filler)
            for n in sorted(set(self.its_text(named)) | set(named)):     # what the answer's sentences hold, and the others
                for s in sorted(self.stems_of(n) - {own}):
                    c = code_of(s)
                    self._judge(mix(mix(R_MARK, frame), c), n in want)
                    self._judge(mix(R_MARKS, c), n in want)

    def answer(self, q: List[str]) -> Tuple[str, List[int]]:
        """The told sentences a request asks for, and the way that found them; nothing where no way is believed.
        Of the believed ways of a form, the one that covered most of what was answered speaks. Where a way that
        takes sentences for where they stand and a way that takes them for what they hold covered the answers
        alike, nothing in the life tells them apart: only what both would say is said."""
        self._memo = {}
        try:
            return self._answer(q)
        finally:
            self._memo = None

    def _answer(self, q: List[str]) -> Tuple[str, list]:
        S, best = self.mind.S, None
        for frame, filler in self.forms_of(q):
            if S.tot(frame) < 2:
                continue
            said, believed = self.ways(frame, filler), []
            for j, out in enumerate(said):
                right, tried = self._record(mix(mix(R_WAY, frame), j + 1))
                if tried and rule(right, tried) >= LINE:
                    covered, of = self._record(mix(mix(R_COVER, frame), j + 1))
                    believed.append((covered / max(1, of), rule(right, tried), -len(out), -j, j, out))
            if not believed:
                continue
            most = max(b[0] for b in believed)
            top = [b for b in believed if b[0] == most]
            fixed = [b for b in top if b[4] < FIXED and b[5]]
            marked = [b for b in top if b[4] >= FIXED]
            pick = max(fixed or [b for b in top if b[5]], default=None)
            if pick is None:
                continue
            out = pick[5]
            if fixed and marked:                             # the two kinds of way agree, or nothing is said
                held = {n for b in marked for n in b[5]}
                out = [n for n in out if n in held]
            if out and (best is None or pick[:4] > best[0]):
                best = (pick[:4], WAYS[pick[4]], out)
        if best is None:
            right, tried = self._record(R_FREE)
            if tried >= 2 and rule(right, tried) >= LINE:
                out = self.through_marks(q)
                if out:
                    return "through marks", out
            return "", []
        return best[1], best[2]

    def say(self, items: list) -> str:
        """What is said: told sentences as they were told; a sentence of the mind's own is a told sentence with
        the thing in the place of its first word."""
        out = []
        for x in items:
            if isinstance(x, tuple):
                rest = self.sentence(x[0]).split(None, 1)
                out.append(x[1] + (" " + rest[1] if len(rest) > 1 else ""))
            else:
                out.append(self.sentence(x))
        return " ".join(out)

    # ------------------------------------------------------------------------------------------- keeping --------
    def size(self) -> dict:
        return {"sentences": len(self.text) - len(self.dead), "bytes": len(self.blob), "texts": self.texts,
                "not_kept": self.not_kept}

    def state(self) -> dict:
        heard = "\n".join(sorted(self.heard)).encode("utf-8", "surrogatepass")
        return {"blob": np.frombuffer(bytes(self.blob), dtype=np.uint8), "offs": np.array(self.offs, dtype=np.uint64),
                "heard": np.frombuffer(heard, dtype=np.uint8),
                "text": np.array(self.text, dtype=np.uint32), "dead": np.array(sorted(self.dead), dtype=np.int64),
                "misc": np.array([self.texts, self.not_kept], dtype=np.int64)}

    @classmethod
    def from_state(cls, mind, st: dict, **how) -> "Telling":
        self = cls(mind, **how)
        self.blob = bytearray(st["blob"].tobytes())
        self.offs = array("Q", st["offs"].tolist())
        self.text = array("I", st["text"].tolist())
        self.dead = set(st["dead"].tolist())
        self.texts, self.not_kept = (int(x) for x in st["misc"])
        self.heard = set(st["heard"].tobytes().decode("utf-8", "surrogatepass").split("\n")) - {""}
        for n in range(len(self.text)):
            if len(self.words) < self.most_words:
                self.words.update(_words(body(tokens(self.sentence(n)))[0]))
        return self
