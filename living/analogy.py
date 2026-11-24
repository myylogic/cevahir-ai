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

Nothing here names a language, a word or a format: tokens are what was said, split as Zihin splits (words and marks).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from living.evidence import SPEAKING_LINE

TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


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
        self._rel: Dict[str, Optional[str]] = {}

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
        self.said[i] = self.answers[i] = None
        self._rel.clear()

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

    def plan(self, said: Sequence[str]) -> Tuple[Optional[List[str]], float, int]:
        """The analogy's answer to ``said``: (tokens or None, its belief, how many neighbours could be filled). Its
        belief is the rule on the analogy's own record for that many places (how often an answer foreseen so was
        right), times the share of the neighbours that agree."""
        plan, agree, filled, d = self._plan(said)
        if plan is None:
            return None, 0.0, 0
        record = (self.k_right[d] + 0.5) / (self.k_asked[d] + 1.0)
        return plan, record * agree / filled, filled

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
