# -*- coding: utf-8 -*-
"""The hippocampus's analogy: an utterance never lived, answered from the lived dialogues it differs from in a few
places (research/living_intelligence_2026_11_24/MODEL.md, section 12.5).

Zihin's frames generalise one place: a lived utterance with one item replaced by a category ("X ne yapar ?"). Most
utterances a mind meets differ from every one it lived in two or three places at once (COGS: 85 % of its test
sentences against a thousand lived; research/living_mind_2026_11_22, section on COGS). This is the same reading
with several places at once, done on the episodes the hippocampus keeps.

- **Episodes.** Every dialogue lived is kept as it was said: the tokens of what was said and of the answer.
- **Neighbours.** For an utterance U, the lived utterances of the same length that differ from U in at most ``most``
  places (5), found through an index of (length, place, token).
- **The relation of a word to the answer.** For a word w, the answer token t that came with it: the one rule on the
  dialogues w was said in, e(t | w) = (W⁺ + ½) / (W + 1) with W⁺ the answers holding t; tokens held by half of all
  answers or more are the answers' own structure, not any word's. A relation is believed when e reaches the
  speaking line e*. On equal evidence the word itself is preferred, then the token sharing the longest beginning
  with it (its stem: Zihin's F34, the asked word itself, and F36, word forms).
- **The plan.** For a neighbour (U', A'), each place where U and U' differ (u in U', v in U) is filled: every token of
  A' that is u's relation becomes v's relation (v itself where u's relation is u itself and v has none yet). All
  places are filled at once, so two roles that swap places are read as a swap; one token that would have to become
  two gives nothing. A neighbour with a place that cannot be filled gives nothing. The nearest neighbours vote (the fewest places first,
  at most ``k``), and the plan is the answer most of them give.
- **Its belief.** Every dialogue lived is first foreseen by the analogy from the episodes already kept; the record
  of how often an answer foreseen over d places was right is the analogy's own (the evidence core's trust in a way of
  reading, learned from what its claims came to). A plan is believed by the rule on the record for its places, times
  the share of the neighbours that agree. An analogy over many places that was seldom right stays below the
  speaking line.

- **Words of a kind.** Two words stand for each other only if their relations play alike parts in the answers lived:
  the answers' tokens around a relation (two before it, one after it, with the relations of words read as one
  mark) make its profile, and two relations are alike when their profiles share at least half (a relation whose
  profile is not yet known is not held back). "to" (its relation "recipient", after a verb) and "on" (after "nmod")
  are not alike, though both are words with relations.
- **Wholes.** A part of an utterance (one to four tokens at a place, read with its frame: the two tokens before it,
  the one after it, a word with a relation read as one mark) is a whole of its own when taking a part of that frame
  out of the lived utterances leaves utterances that can still be answered: the rule on those examples, at the
  speaking line.
- **Answering by parts.** An utterance no lived one is near (no plan reaches the line) is answered by taking a whole
  out, answering the rest (by the analogy, or again by parts), and putting the whole's effect back. The effect is
  read from lived utterances holding a part of the same frame: what taking their part out changes in their answers.
  The change is put into the rest's answer: the part's own words and places by their place in the part, everything
  else through the alignment of the two rests' answers (read by what they say about the places near the part); a
  token of an answer that is a whole number smaller than the utterance's length is read as a place of it, and
  moves with the places. Where an answer lists runs of the same shape, a new run goes by its place (a list keeps the
  order of what was said). The examples vote; the plan's belief is the rest's belief times the share of the
  examples that agree. The most whole-like parts are tried first, then the rightmost; the search is bounded.

Nothing here names a language, a word or a format: tokens are what was said, split as Zihin splits (words and marks).
"""
from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from living.evidence import SPEAKING_LINE

TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PARTS_BUDGET = 400                                             # parts tried, at most, for one utterance
PARTS_DEPTH = 16                                               # parts taken out, at most, one after another


def tokens(text: str) -> List[str]:
    return TOKEN.findall(text)


def shared_beginning(a: str, b: str) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


class Episodes:
    """The dialogues lived, and the analogy that answers from them."""

    def __init__(self, most: int = 5, k: int = 8):
        self.most, self.k = most, k
        self.k_right: Counter = Counter()                      # its own record: answers foreseen right ...
        self.k_asked: Counter = Counter()                      # ... and foreseen, by the places of the nearest
        self.said: List[Tuple[str, ...]] = []
        self.answers: List[Tuple[str, ...]] = []
        self.index: Dict[Tuple[int, int, str], List[int]] = defaultdict(list)
        self.with_w: Counter = Counter()                       # dialogues each word was said in
        self.with_wt: Dict[str, Counter] = defaultdict(Counter)   # ... whose answers held each token
        self.t_count: Counter = Counter()                      # answers holding each token
        self.where: Dict[Tuple[str, ...], int] = {}             # each utterance lived, its latest dialogue
        self._rel: Dict[str, Optional[str]] = {}
        self._made_at = -1                                     # the kinds and the frames, read again as life doubles
        self._kinds: Dict[str, Counter] = {}
        self._frames: Dict[tuple, List[int]] = {}
        self._units: Dict[tuple, float] = {}
        self._asked: Dict[tuple, tuple] = {}
        self._forgotten = 0

    def __len__(self) -> int:
        return len(self.said)

    def live(self, said: Sequence[str], answer: Sequence[str], judge: bool = True) -> None:
        """Keep a dialogue (its tokens, as said). With ``judge``, the answer is first foreseen from the episodes
        already kept, and the record of the analogy (by the places of its nearest neighbour) says whether it was
        right: that record is how far an analogy over that many places is believed."""
        said, answer = tuple(said), tuple(answer)
        if judge and self.said:
            plan, _, _, d = self._plan(said)
            if plan is not None:
                self.k_asked[d] += 1
                self.k_right[d] += tuple(plan) == answer
        i = len(self.said)
        self.said.append(said)
        self.answers.append(answer)
        self.where[said] = i
        for j, w in enumerate(said):
            self.index[(len(said), j, w)].append(i)
        held = set(answer)
        for t in held:
            self.t_count[t] += 1
        for w in set(said):
            self.with_w[w] += 1
            row = self.with_wt[w]
            for t in held:
                row[t] += 1
        self._rel.clear()                                      # relations follow the evidence

    def forget(self, i: int) -> None:
        """Take a dialogue back exactly: its place in the index and its counts."""
        said, answer = self.said[i], self.answers[i]
        if said is None:
            return
        for j, w in enumerate(said):
            self.index[(len(said), j, w)].remove(i)
        held = set(answer)
        for t in held:
            self.t_count[t] -= 1
        for w in set(said):
            self.with_w[w] -= 1
            for t in held:
                self.with_wt[w][t] -= 1
        if self.where.get(said) == i:
            del self.where[said]
            for j in range(i - 1, -1, -1):                     # an earlier dialogue of the same utterance, if any
                if self.said[j] == said:
                    self.where[said] = j
                    break
        self.said[i] = self.answers[i] = None
        self._rel.clear()
        self._made_at = -1                                     # the kinds and frames are read again
        self._forgotten += 1

    def state(self) -> dict:
        return {"most": self.most, "k": self.k, "dialogues": [[list(s), list(a)] if s is not None else None
                                                              for s, a in zip(self.said, self.answers)],
                "k_right": {str(k): v for k, v in self.k_right.items()},
                "k_asked": {str(k): v for k, v in self.k_asked.items()}}

    @classmethod
    def from_state(cls, state: dict) -> "Episodes":
        ep = cls(state.get("most", 5), state.get("k", 8))
        for d in state.get("dialogues", []):
            if d is None:
                ep.said.append(None)
                ep.answers.append(None)
            else:
                ep.live(d[0], d[1], judge=False)
        ep.k_right = Counter({int(k): v for k, v in state.get("k_right", {}).items()})
        ep.k_asked = Counter({int(k): v for k, v in state.get("k_asked", {}).items()})
        return ep

    def relation(self, w: str) -> Optional[str]:
        """The answer token that comes with ``w``, if believed at the speaking line (see the module's text)."""
        if w in self._rel:
            return self._rel[w]
        n = self.with_w.get(w, 0)
        best, key = None, None
        if n:
            common = 0.5 * sum(1 for x in self.said if x is not None)
            for t, c in self.with_wt[w].items():
                if self.t_count[t] >= common:
                    continue
                e = (c + 0.5) / (n + 1.0)
                if e < SPEAKING_LINE:
                    continue
                kk = (e, t == w, shared_beginning(t, w))
                if key is None or kk > key:
                    best, key = t, kk
        self._rel[w] = best
        return best

    def neighbours(self, said: Sequence[str]) -> List[Tuple[int, List[int]]]:
        """The lived utterances of the same length that differ in at most ``most`` places: (episode, places)."""
        n = len(said)
        if n == 0:
            return []
        hits: Counter = Counter()
        for j, w in enumerate(said):
            for i in self.index.get((n, j, w), ()):
                hits[i] += 1
        out = []
        for i, same in hits.items():
            if n - same <= self.most:
                other = self.said[i]
                out.append((i, [j for j in range(n) if other[j] != said[j]]))
        out.sort(key=lambda x: (len(x[1]), -x[0]))             # the fewest places first, then the latest
        return out

    def plan(self, said: Sequence[str], parts: bool = True) -> Tuple[Optional[List[str]], float, int]:
        """The analogy's answer to ``said``: (tokens or None, its belief, how many neighbours could be filled). Its
        belief is the rule on the analogy's own record for that many places (how often an answer foreseen so was
        right), times the share of the neighbours that agree. Where that does not reach the speaking line and
        ``parts`` is on, the utterance is answered by parts (see the module's text); then the count is the examples
        that agreed."""
        key = (tuple(said), parts, len(self.said), self._forgotten)
        if key in self._asked:                                 # asked again before anything new was lived
            plan, e, n = self._asked[key]
            return (list(plan) if plan is not None else None), e, n
        got = self._plain(said)
        if parts and self.said and not (got[0] is not None and got[1] >= SPEAKING_LINE):
            plan, e, n = self._by_parts(tuple(said), {}, [PARTS_BUDGET], 0)
            if plan is not None:
                got = (plan, e, n)
        if len(self._asked) > 4096:
            self._asked.clear()
        self._asked[key] = got
        return (list(got[0]) if got[0] is not None else None), got[1], got[2]

    def _plain(self, said: Sequence[str]) -> Tuple[Optional[List[str]], float, int]:
        plan, agree, filled, d = self._plan(said)
        if plan is None:
            return None, 0.0, 0
        record = (self.k_right[d] + 0.5) / (self.k_asked[d] + 1.0)
        return plan, record * agree / filled, filled

    # ----- the kinds of words, and the frames of parts (read again each time life has doubled) -----

    def _make(self) -> None:
        if self._made_at >= 0 and len(self.said) < 2 * max(1, self._made_at):
            return
        self._made_at = len(self.said)
        related = set()
        for w in list(self.with_w):
            if self.with_w[w] > 0:
                r = self.relation(w)
                if r is not None:
                    related.add(r)
        kinds: Dict[str, Counter] = defaultdict(Counter)
        for a in self.answers:
            if a is None or len(a) < 2:                         # an answer of one token says nothing of a part
                continue
            n = len(a)
            def mark(k: int) -> str:
                if k < 0:
                    return "^"
                if k >= n:
                    return "$"
                return "\u25a1" if a[k] in related else a[k]
            for j, t in enumerate(a):
                if t in related:
                    kinds[t][(mark(j - 2), mark(j - 1), mark(j + 1))] += 1
        self._kinds = dict(kinds)
        frames: Dict[tuple, List[int]] = defaultdict(list)
        for i, u in enumerate(self.said):
            if u is None:
                continue
            for m in range(1, 5):
                for p in range(0, min(len(u) - m, 1024)):      # never the utterance's last token
                    frames[self._frame(u, p, m)].append(i * 1024 + p)
        self._frames = dict(frames)
        self._units = {}

    def ready(self) -> None:
        """Read the kinds and the frames again if anything was lived since they were last read (before answering)."""
        if self._made_at != len(self.said):
            self._made_at = -1
            self._make()

    def alike(self, a: str, b: str) -> bool:
        """Two relations play alike parts in the answers lived (or one of them is not yet known)."""
        if a == b:
            return True
        self._make()
        pa, pb = self._kinds.get(a), self._kinds.get(b)
        if not pa or not pb:
            return True
        na, nb = sum(pa.values()), sum(pb.values())
        return sum(min(c / na, pb[f] / nb) for f, c in pa.items()) >= 0.5

    def _shape(self, w: str) -> str:
        return "\u25a1" if self.relation(w) is not None else w

    def _frame(self, u: Sequence[str], p: int, m: int) -> tuple:
        left = tuple(self._shape(w) for w in u[max(0, p - 2):p])
        left = ("^",) * (2 - len(left)) + left
        right = tuple(self._shape(w) for w in u[p + m:p + m + 1]) or ("$",)
        return (m, left, tuple(self._shape(w) for w in u[p:p + m]), right)

    def unit(self, fr: tuple, sample: int = 12) -> float:
        """How far a part of this frame is a whole of its own: the rule on lived examples, each counted when taking
        its part out leaves an utterance that is lived or that the analogy answers at the speaking line."""
        if fr in self._units:
            return self._units[fr]
        ex = self._frames.get(fr, [])
        step = max(1, len(ex) // sample)
        ok = n = 0
        for code in ex[::step][:sample]:
            i, p2 = divmod(code, 1024)
            u = self.said[i]
            if u is None:
                continue
            rest = u[:p2] + u[p2 + fr[0]:]
            n += 1
            if rest in self.where:
                ok += 1
            else:
                plan, e, _ = self._plain(rest)
                ok += plan is not None and e >= SPEAKING_LINE
        self._units[fr] = (ok + 0.5) / (n + 1.0)
        return self._units[fr]

    # ----- answering by parts -----

    def _by_parts(self, u: Tuple[str, ...], memo: dict, budget: list, depth: int
                  ) -> Tuple[Optional[List[str]], float, int]:
        if u in memo:
            return memo[u]
        if u in self.where:
            got = (list(self.answers[self.where[u]]), 1.0, 1)
            memo[u] = got
            return got
        if depth:
            plan, e, n = self._plain(u)
            if plan is not None and e >= SPEAKING_LINE:
                memo[u] = (plan, e, n)
                return memo[u]
        memo[u] = (None, 0.0, 0)                                # no part is answered from itself
        if depth > PARTS_DEPTH or budget[0] <= 0:
            return memo[u]
        self._make()
        n = len(u)
        tries = []
        for m in range(1, 5):
            for p in range(0, n - m):
                fr = self._frame(u, p, m)
                if fr in self._frames:
                    r = self.unit(fr)
                    if r >= SPEAKING_LINE:
                        tries.append((-r, -(p + m), p, m, fr))
        tries.sort()                                           # the most whole-like parts first, then the rightmost
        for _, _, p, m, fr in tries:
            budget[0] -= 1
            if budget[0] < 0:
                break
            rest_plan, rest_e, _ = self._by_parts(u[:p] + u[p + m:], memo, budget, depth + 1)
            if rest_plan is None:
                continue
            votes: Counter = Counter()
            filled = 0
            def near(code: int) -> int:
                i, p2 = divmod(code, 1024)
                o = self.said[i]
                if o is None:
                    return 0
                return -sum(1 for r in range(-3, m + 3)
                            if 0 <= p + r < n and 0 <= p2 + r < len(o) and u[p + r] == o[p2 + r])
            for code in sorted(self._frames[fr], key=near)[:4 * self.k]:
                i, p2 = divmod(code, 1024)
                o = self.said[i]
                if o is None:
                    continue
                rest2 = o[:p2] + o[p2 + m:]
                if rest2 in self.where:
                    a2_rest = list(self.answers[self.where[rest2]])
                else:
                    a2_rest, e2, _ = self._plain(rest2)
                    if a2_rest is None or e2 < SPEAKING_LINE:
                        continue
                got = self._transplant(o, p2, m, a2_rest, list(self.answers[i]), u, p, rest_plan)
                if got is None:
                    continue
                votes[tuple(got)] += 1
                filled += 1
                if filled >= self.k:
                    break
            if votes:
                best, agree = votes.most_common(1)[0]
                e = rest_e * agree / filled
                if e >= SPEAKING_LINE:
                    memo[u] = (list(best), e, agree)
                    return memo[u]
        return memo[u]

    @staticmethod
    def _shift(a: Sequence[str], p: int, m: int, n: int) -> List[str]:
        """The places of an answer to an utterance of n tokens without a part of m tokens at p, read as places of
        the utterance with the part: those at p or after move on by m."""
        out = []
        for t in a:
            if t.isdigit() and p <= int(t) < n:
                out.append(str(int(t) + m))
            else:
                out.append(t)
        return out

    def _transplant(self, o: Tuple[str, ...], p2: int, m: int, a2_rest: List[str], a2: List[str],
                    u: Tuple[str, ...], p: int, rest_plan: List[str]) -> Optional[List[str]]:
        """What taking the part out of the example (o, its part at p2) changed in its answer, put into the answer
        of the asked utterance's rest (u, its part at p). None where the change cannot be put."""
        a2s = self._shift(a2_rest, p2, m, len(o) - m)
        a_s = self._shift(rest_plan, p, m, len(u) - m)
        def cls(k: int, pp: int) -> str:
            if pp <= k < pp + m:
                return "S%d" % (k - pp)
            if k < pp:
                return "L%d" % (pp - k) if pp - k <= 3 else "L"
            return "R%d" % (k - pp - m) if k - pp - m <= 3 else "R"
        def read(a: List[str], w: Tuple[str, ...], pp: int) -> List[str]:
            content = {self.relation(x) for x in w} - {None}
            out = []
            for t in a:
                if t.isdigit() and int(t) < len(w):
                    out.append(cls(int(t), pp))
                elif t in content:
                    out.append("\u25a1")
                else:
                    out.append(t)
            return out
        r2, r = read(a2s, o, p2), read(a_s, u, p)
        blocks = [b for b in difflib.SequenceMatcher(None, r2, r, autojunk=False).get_matching_blocks() if b[2]]
        places: Dict[str, Counter] = defaultdict(Counter)
        words: Dict[str, Counter] = defaultdict(Counter)
        for bi, bj, size in blocks:
            for d in range(size):
                t2, t = a2s[bi + d], a_s[bj + d]
                if r2[bi + d][0] in "LR" and t2.isdigit() and t.isdigit():
                    places[t2][t] += 1
                elif r2[bi + d] == "\u25a1":
                    words[t2][t] += 1
        own: Dict[str, str] = {}
        for k in range(m):
            x, y = o[p2 + k], u[p + k]
            rx = self.relation(x)
            if rx is None:
                if x != y:
                    return None
                continue
            ry = self.relation(y)
            if ry is None:
                if rx != x:
                    return None
                ry = y
            elif not self.alike(rx, ry):
                return None
            if own.get(rx, ry) != ry:
                return None
            own[rx] = ry
        theirs = {self.relation(x) for x in o} - {None}
        def put(t: str) -> Optional[str]:
            if t.isdigit() and int(t) < len(o):
                k = int(t)
                if p2 <= k < p2 + m:
                    return str(p + k - p2)
                return places[t].most_common(1)[0][0] if t in places else None
            if t in own:
                return own[t]
            if t in words:
                return words[t].most_common(1)[0][0]
            return None if t in theirs else t
        def at(i: int) -> int:
            if i == len(a2s):
                return len(a_s)
            for bi, bj, size in blocks:
                if bi <= i <= bi + size:
                    return bj + (i - bi)
            best = 0
            for bi, bj, size in blocks:
                if bi + size <= i:
                    best = bj + size
            return best
        content = {self.relation(x) for x in u} - {None}
        def shape(seq: Sequence[str]) -> tuple:
            return tuple("#" if t.isdigit() else ("\u25a1" if t in content else t) for t in seq)
        def first_place(seq: Sequence[str]) -> int:
            for t in seq:
                if t.isdigit():
                    return int(t)
            return -1
        codes = [c for c in difflib.SequenceMatcher(None, a2s, a2, autojunk=False).get_opcodes() if c[0] != "equal"]
        edits = []
        for n_op, (tag, i1, i2, j1, j2) in enumerate(codes):
            seg = list(a2[j1:j2])
            if tag == "insert":                                # one change, many ways to write it: pushed on
                bound = codes[n_op + 1][1] if n_op + 1 < len(codes) else len(a2s)
                while i1 < bound and seg and a2s[i1] == seg[0]:
                    seg = seg[1:] + [a2s[i1]]
                    i1 += 1
                i2 = i1
            where = at(i1)
            old = [put(t) for t in a2s[i1:i2]]
            new = [put(t) for t in seg]
            if None in old or None in new or a_s[where:where + len(old)] != old:
                return None
            if tag == "insert":                                # a list keeps the order of what was said
                size, here, first = len(new), shape(new), first_place(new)
                if first >= 0:
                    while (where + size <= len(a_s) and shape(a_s[where:where + size]) == here
                           and first_place(a_s[where:where + size]) < first):
                        where += size
                    while (where - size >= 0 and shape(a_s[where - size:where]) == here
                           and first_place(a_s[where - size:where]) > first):
                        where -= size
            edits.append((where, len(old), new))
        out = list(a_s)
        for where, k, new in sorted(edits, key=lambda x: -x[0]):
            out[where:where + k] = new
        return out

    def _plan(self, said: Sequence[str]) -> Tuple[Optional[List[str]], int, int, int]:
        votes: Counter = Counter()
        first: Dict[Tuple[str, ...], int] = {}
        filled = 0
        for i, places in self.neighbours(said):
            other = self.said[i]
            mapping: Dict[str, str] = {}                       # every place filled at once (two roles may swap)
            ok = True
            for j in places:
                u, v = other[j], said[j]
                ru = self.relation(u)
                if ru is None:
                    ok = False
                    break
                rv = self.relation(v)
                if rv is None:
                    if ru == u:
                        rv = v                                 # the word itself
                    else:
                        ok = False
                        break
                elif not self.alike(ru, rv):                   # words of another kind do not stand for each other
                    ok = False
                    break
                if mapping.get(ru, rv) != rv:                  # one token cannot become two
                    ok = False
                    break
                mapping[ru] = rv
            if not ok:
                continue
            answer = [mapping.get(t, t) for t in self.answers[i]]
            votes[tuple(answer)] += 1
            first.setdefault(tuple(answer), len(places))
            filled += 1
            if filled >= self.k:
                break
        if not votes:
            return None, 0, 0, 0
        best, agree = votes.most_common(1)[0]
        return list(best), agree, filled, first[best]
