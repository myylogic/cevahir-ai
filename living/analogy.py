# -*- coding: utf-8 -*-
"""The hippocampus's analogy: an utterance never lived, answered from the lived dialogues it differs from in a few
places (research/living_intelligence_2026_11_24/MODEL.md, section 12.5).

Zihin's frames generalise one place: a lived utterance with one item replaced by a category ("X ne yapar ?"). Most
utterances a mind meets differ from every one it lived in two or three places at once (COGS: 85 % of its test
sentences against a thousand lived; research/living_mind_2026_11_22, section on COGS). This is the same reading
with several places at once, done on the episodes the hippocampus keeps.

- **Episodes.** Every dialogue lived is kept as it was said: the tokens of what was said and of the answer.
- **Neighbours.** For an utterance U, the lived utterances of the same length that differ from U in at most ``most``
  places (3), found through an index of (length, place, token).
- **The relation of a word to the answer.** For a word w, the answer token t that came with it: the one rule on the
  dialogues w was said in, e(t | w) = (W⁺ + ½) / (W + 1) with W⁺ the answers holding t; tokens held by half of all
  answers or more are the answers' own structure, not any word's. A relation is believed when e reaches the
  speaking line e*. On equal evidence the word itself is preferred, then the token sharing the longest beginning
  with it (its stem: Zihin's F34, the asked word itself, and F36, word forms).
- **The plan.** For a neighbour (U', A'), each place where U and U' differ (u in U', v in U) is filled: every token of
  A' that is u's relation becomes v's relation (v itself where u's relation is u itself and v has none yet). A
  neighbour with a place that cannot be filled gives nothing. The nearest neighbours vote (the fewest places first,
  at most ``k``), and the plan is the answer most of them give, believed with the rule on that vote:
  e = (W⁺ + ½) / (W + 1), W⁺ the neighbours agreeing, W those that could be filled.

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

    def __init__(self, most: int = 3, k: int = 8):
        self.most, self.k = most, k
        self.said: List[Tuple[str, ...]] = []
        self.answers: List[Tuple[str, ...]] = []
        self.index: Dict[Tuple[int, int, str], List[int]] = defaultdict(list)
        self.with_w: Counter = Counter()                       # dialogues each word was said in
        self.with_wt: Dict[str, Counter] = defaultdict(Counter)   # ... whose answers held each token
        self.t_count: Counter = Counter()                      # answers holding each token
        self._rel: Dict[str, Optional[str]] = {}

    def __len__(self) -> int:
        return len(self.said)

    def live(self, said: Sequence[str], answer: Sequence[str]) -> None:
        """Keep a dialogue (its tokens, as said)."""
        i = len(self.said)
        said, answer = tuple(said), tuple(answer)
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
                                                              for s, a in zip(self.said, self.answers)]}

    @classmethod
    def from_state(cls, state: dict) -> "Episodes":
        ep = cls(state.get("most", 3), state.get("k", 8))
        for d in state.get("dialogues", []):
            if d is None:
                ep.said.append(None)
                ep.answers.append(None)
            else:
                ep.live(d[0], d[1])
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
        """The analogy's answer to ``said``: (tokens or None, its belief, how many neighbours could be filled)."""
        votes: Counter = Counter()
        filled = 0
        for i, places in self.neighbours(said):
            other, answer = self.said[i], list(self.answers[i])
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
                answer = [rv if t == ru else t for t in answer]
            if not ok:
                continue
            votes[tuple(answer)] += 1
            filled += 1
            if filled >= self.k:
                break
        if not votes:
            return None, 0.0, 0
        best, agree = votes.most_common(1)[0]
        return list(best), (agree + 0.5) / (filled + 1.0), filled
