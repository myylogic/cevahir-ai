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
  answers or more are the answers' own structure, not any word's. A relation is believed when e reaches the speaking
  line e*. On equal evidence the word itself is preferred, then the token sharing the longest beginning with it (its
  stem: Zihin's F34, the asked word itself, and F36, word forms), then the token whose own dialogues held the word
  most often (the rule on them: the token most the word's own), then the one first met with it.
- **The plan.** For a neighbour (U', A'), each place where U and U' differ (u in U', v in U) is filled: every token of
  A' that is u's relation becomes v's relation (v itself where u's relation is u itself and v has none yet). All
  places are filled at once, so two roles that swap places are read as a swap; one token that would have to become two
  gives nothing, and so does a place whose word also stands unchanged elsewhere in the neighbour (its tokens cannot be
  told from the other occurrence's: "run left after run left" says nothing of "run left after jump left"). A neighbour
  with a place that cannot be filled gives nothing, and so does one whose answer, filled, gives the token of a word
  put in other tokens after it than it has always had where the word was said between neighbours of the same shapes
  (one and two tokens on, the end of an answer counted as what comes after: "smiled", said after a noun at the end,
  was always answered "smile . agent", so "a hippo smiled" is not answered from "a hippo rolled" as "smile .
  theme"); where the word was never said so, nothing holds the answer back. The nearest neighbours vote (the fewest
  places first, at most ``k``), and the plan is the answer most of them give.
- **Its belief.** Every dialogue lived is first foreseen by the analogy from the episodes already kept; the record of
  how often an answer foreseen over d places was right is the analogy's own (the evidence core's trust in a way of
  reading, learned from what its claims came to). A plan is believed by the rule on the record for its places, times
  the share of the neighbours that agree. An analogy over many places that was seldom right stays below the speaking
  line.

- **Words of a kind.** Two words stand for each other only if their relations have played a like part in the answers
  lived: the places a relation has held (the two tokens before it and the one after it, with the relations of words
  read as one mark and numbers as another) make its profile, and two relations are alike when they have held at least
  one same place (a relation not yet met in an answer of more than one token is not held back). "to" (its relation
  "recipient", after a verb) and "on" (after "nmod") are not alike, though both are words with relations.
- **Wholes.** A part of an utterance (one to four tokens at a place, read with its frame: the two tokens before it,
  the one after it, a word with a relation read as one mark) is a whole of its own as far as answering by such a part
  foresees what was lived: lived examples of the frame are each answered as if they had not been lived (the part taken
  out, the rest's answer read, the change the nearest other example with the same surroundings shows put back, as
  below), and the rule on how often that gave the lived answer is the frame's record. A frame is a whole at the
  speaking line.
- **Answering by parts.** An utterance no lived one is near (no plan reaches the line) is answered by taking a whole
  out, answering the rest (by the analogy, or again by parts), and putting the whole's effect back. The effect is read
  from lived utterances holding a part of the same frame with the same surroundings (the PARTS_AROUND tokens on each
  side, read by shape): what taking their part out changes in their answers. The change is put into the rest's answer:
  the part's own words and places by their place in the part, everything else through the alignment of the two rests'
  answers (read by what they say about the places near the part: the numbers that name them and the relations of the
  words there); a token of an answer that is a whole number smaller than the utterance's length is read as a place of
  it, and moves with the places. What the example put at the very beginning goes at the beginning; where an answer
  lists runs of the same shape, a new run goes by its place among its neighbours (a list keeps the order of what was
  said). The examples vote; a step's belief is the frame's record times the share of the examples that agree, and a
  plan's belief is that of its weakest step (the rest's own included): the steps of a chain read the same kind of
  change again and again, so their errors go together. A change no lived example carries stays below the line: SCAN's
  "around" turns four times for each turn "twice" or "thrice" asks, and an example with "twice" after the part, whose
  surroundings differ, says nothing of "thrice". The most whole-like parts are tried first, then the rightmost; the
  search is bounded.
- **Exchanging a part.** A part that cannot be taken out (what remains would not be an utterance) is exchanged for the
  part lived most often between the same neighbours; the rest is answered, and what such an exchange changes in lived
  answers is put back the same way. Only parts made of words with relations are exchanged: what such a word brings to
  an answer is its own token wherever it stands, while what a word without a relation brings depends on where it
  stands (SCAN's "right" turns once, or four times after "around"), and only for a part that holds a word with a
  relation and none of the asked part's words (an exchange puts one part's meaning where another's was; "jump" for
  "jump opposite" would add "opposite", a word that brings no token of its own). The change is read from the asked
  part's own frame with the same surroundings or, where there is none, wherever both kinds of part were lived, and
  only from examples whose part, too, shares no word with the one put in ("look left after run", exchanging "look"
  for "look opposite", keeps "look"'s token, and says nothing of what becomes of it when "run" is exchanged); each
  case is believed by its own record (lived examples exchanged as if not lived, the change read from an example with
  the same, or with other, surroundings). Taking a part out is the exchange for nothing.

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
PARTS_AROUND = 3                                               # tokens on each side of a part an example must share
PART = "\u27e8%d\u27e9"                                         # a place of a part put in for another, while it stands


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
        self._slots: Dict[tuple, list] = {}
        self._shapes: Dict[tuple, list] = {}
        self._units: Dict[tuple, float] = {}
        self._rests: Dict[tuple, Optional[List[str]]] = {}     # the answers of rests, as the parts read them
        self._windows: Dict[tuple, Dict[tuple, List[int]]] = {}  # each frame's examples, by what stands around them
        self._exchanges: Dict[tuple, float] = {}               # the record of each kind of exchange
        self._asked: Dict[tuple, tuple] = {}
        self._alike: Dict[Tuple[str, str], bool] = {}
        self._at: Dict[tuple, List[int]] = {}                  # the dialogues of each word, by its neighbours' shapes
        self._owns: Dict[tuple, Dict[int, str]] = {}           # what its token always had after it there
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
        held = list(dict.fromkeys(answer))                    # in the order first met: the same in every process
        for t in held:
            self.t_count[t] += 1
        for w in dict.fromkeys(said):
            self.with_w[w] += 1
            row = self.with_wt[w]
            for t in held:
                row[t] += 1
        self._rel.clear()                                      # relations follow the evidence
        if self._rests:
            self._rests.clear()

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
        self._rests.clear()
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
                # on equal evidence: the word itself, then its stem, then the token that comes with this word more
                # than with any other (the rule on how many of the token's dialogues held the word)
                kk = (e, t == w, shared_beginning(t, w), (c + 0.5) / (self.t_count[t] + 1.0))
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
                if a[k].isdigit():
                    return "#"
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
        slots: Dict[tuple, list] = defaultdict(list)            # the parts lived between the same neighbours
        for fr, codes in self._frames.items():
            slots[(fr[1], fr[3])].append((len(codes), fr[0], fr[2], codes[0]))
        shapes: Dict[tuple, list] = defaultdict(list)           # the frames a part of each shape was lived in
        for fr in self._frames:
            shapes[(fr[0], fr[2])].append(fr)
        for v in slots.values():
            v.sort(key=lambda x: (-x[0], x[1], x[2]))
        self._slots = dict(slots)
        self._shapes = dict(shapes)
        at: Dict[tuple, List[int]] = defaultdict(list)          # each word's dialogues, by the shapes of its neighbours
        for i, u in enumerate(self.said):
            if u is None:
                continue
            for j, w in enumerate(u):
                key = (w, self._shape(u[j - 1]) if j else "^", self._shape(u[j + 1]) if j + 1 < len(u) else "$")
                if len(at[key]) < 64:
                    at[key].append(i)
        self._at = dict(at)
        self._owns = {}
        self._units = {}
        self._rests = {}
        self._windows = {}
        self._exchanges = {}
        self._alike = {}

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
        key = (a, b) if a < b else (b, a)
        if key not in self._alike:
            pa, pb = self._kinds.get(a), self._kinds.get(b)
            if not pa or not pb:
                self._alike[key] = True
            else:
                self._alike[key] = any(f in pb for f in pa)          # seen at least once in the same place
        return self._alike[key]

    def _own(self, w: str, left: str, right: str) -> Dict[int, str]:
        """What the token of ``w`` has always had right after it, one and two tokens on, in the lived answers where w
        was said between neighbours of these shapes: {how far on: token}, a number read as one mark. The end of an
        answer counts as what comes after: where the token stood last, nothing came after it."""
        key = (w, left, right)
        got = self._owns.get(key)
        if got is None:
            r = self.relation(w)
            seen: Dict[int, set] = {1: set(), 2: set()}
            if r is not None:
                for i in self._at.get(key, ()):
                    a = self.answers[i]
                    if a is None:
                        continue
                    for q, t in enumerate(a):
                        if t == r:
                            for k in (1, 2):
                                seen[k].add(self._after(a, q + k))
            got = self._owns[key] = {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}
        return got

    def _as_lived(self, said: Sequence[str], places: Sequence[int], answer: Sequence[str]) -> bool:
        """Whether a filled answer leaves the token of each word put in among what it has always had after it where
        the word was said between neighbours of the same shapes (``_own``): an answer that puts other tokens there, or
        ends there, says of that word what was never lived. Where the word was never said so, nothing holds the answer
        back."""
        self._make()
        n = len(said)
        for j in places:
            v = said[j]
            rv = self.relation(v)
            if rv is None:
                continue
            own = self._own(v, self._shape(said[j - 1]) if j else "^", self._shape(said[j + 1]) if j + 1 < n else "$")
            if not own:
                continue
            for q, t in enumerate(answer):
                if t == rv:
                    if any(self._after(answer, q + k) != x for k, x in own.items()):
                        return False
        return True

    @staticmethod
    def _after(a: Sequence[str], k: int) -> str:
        return "$" if k >= len(a) else ("#" if a[k].isdigit() else a[k])

    def _shape(self, w: str) -> str:
        return "\u25a1" if self.relation(w) is not None else w

    def _frame(self, u: Sequence[str], p: int, m: int) -> tuple:
        left = tuple(self._shape(w) for w in u[max(0, p - 2):p])
        left = ("^",) * (2 - len(left)) + left
        right = tuple(self._shape(w) for w in u[p + m:p + m + 1]) or ("$",)
        return (m, left, tuple(self._shape(w) for w in u[p:p + m]), right)

    def _rest(self, r: Tuple[str, ...]) -> Optional[List[str]]:
        """The answer of an utterance as the parts read it: the one lived, or the analogy's own at the speaking line."""
        if r in self.where:
            return list(self.answers[self.where[r]])
        if r not in self._rests:
            plan, e, _ = self._plain(r)
            self._rests[r] = plan if plan is not None and e >= SPEAKING_LINE else None
        got = self._rests[r]
        return list(got) if got is not None else None

    @staticmethod
    def _near(u: Sequence[str], p: int, o: Sequence[str], p2: int, m: int) -> int:
        """How near an example (o, its part at p2) stands to an utterance (u, its part at p): the tokens that are the
        same around the two parts, three on each side."""
        return sum(1 for r in range(-3, m + 3)
                   if 0 <= p + r < len(u) and 0 <= p2 + r < len(o) and u[p + r] == o[p2 + r])

    def _window(self, u: Sequence[str], p: int, m: int) -> tuple:
        """What stands around a part (u, at p, m tokens): the shapes of the PARTS_AROUND tokens on each side of it."""
        w = PARTS_AROUND
        return tuple("^" if k < 0 else ("$" if k >= len(u) else self._shape(u[k]))
                     for k in list(range(p - w, p)) + list(range(p + m, p + m + w)))

    def _windows_of(self, fr: tuple) -> Dict[tuple, List[int]]:
        """The lived examples of a frame, by what stands around their part."""
        got = self._windows.get(fr)
        if got is None:
            groups: Dict[tuple, List[int]] = defaultdict(list)
            for code in self._frames.get(fr, ()):
                i, p2 = divmod(code, 1024)
                if self.said[i] is not None:
                    groups[self._window(self.said[i], p2, fr[0])].append(code)
            got = self._windows[fr] = dict(groups)
        return got

    def unit(self, fr: tuple, sample: int = 24) -> float:
        """How far a part of this frame is a whole of its own: the rule on lived examples of the frame, each
        foreseen as if it had not been lived: its part taken out, the rest's answer read (lived, or the analogy's at
        the speaking line), and the change the nearest other example with the same surroundings shows put back. An
        example counts as right when that gives the answer that was lived; one whose rest has no answer, or that no
        other example can be put into, counts as not right: a part is a whole as far as taking it out leaves what can
        be answered and putting it back gives what was lived."""
        if fr in self._units:
            return self._units[fr]
        m = fr[0]
        ex = self._frames.get(fr, [])
        groups = self._windows_of(fr)
        step = max(1, len(ex) // sample)
        right = n = 0
        for code in ex[::step][:sample]:
            i, p2 = divmod(code, 1024)
            u = self.said[i]
            if u is None:
                continue
            n += 1
            rest = self._rest(u[:p2] + u[p2 + m:])
            if rest is None:
                continue
            others = []
            for c2 in groups.get(self._window(u, p2, m), ())[:64]:
                j, q2 = divmod(c2, 1024)
                o = self.said[j]
                if o is not None and o != u:
                    others.append((-self._near(u, p2, o, q2, m), j, q2, o))
            others.sort(key=lambda x: x[0])
            for _, j, q2, o in others[:3]:
                o_rest = self._rest(o[:q2] + o[q2 + m:])
                if o_rest is None:
                    continue
                got = self._transplant(o, q2, m, o_rest, list(self.answers[j]), u, p2, rest)
                if got is not None:
                    right += got == list(self.answers[i])
                    break
        self._units[fr] = (right + 0.5) / (n + 1.0)
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
        for minus_r, _, p, m, fr in tries:
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
                return 0 if o is None else -self._near(u, p, o, p2, m)
            pool = self._windows_of(fr).get(self._window(u, p, m), ())
            for code in sorted(pool, key=near)[:4 * self.k]:
                i, p2 = divmod(code, 1024)
                o = self.said[i]
                if o is None:
                    continue
                a2_rest = self._rest(o[:p2] + o[p2 + m:])
                if a2_rest is None:
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
                e = min(rest_e, -minus_r * self._agreement(agree, filled))   # a chain is its weakest step
                if e >= SPEAKING_LINE:
                    memo[u] = (list(best), e, agree)
                    return memo[u]
        if depth == 0:                                         # the asked utterance's own parts, on their own budget
            got = self._by_exchange(u, memo, [PARTS_BUDGET // 4], depth)
            if got[0] is not None:
                memo[u] = got
        return memo[u]

    def _by_exchange(self, u: Tuple[str, ...], memo: dict, budget: list, depth: int
                     ) -> Tuple[Optional[List[str]], float, int]:
        """A part that cannot be taken out (what remains would not be an utterance) is exchanged for the part lived
        most often between the same neighbours; the rest is answered, and what that exchange changes in lived answers
        is put back, as for a part taken out: read from lived parts of the asked part's frame with the same
        surroundings, or, where there are none, wherever a part of its shape was lived where a part of the other's
        was too. Its belief is the exchange's own record for that case (``exchange_record``) times the share of the
        examples that agree."""
        slots = getattr(self, "_slots", None) or {}
        shapes = getattr(self, "_shapes", None) or {}
        n = len(u)
        held = "\u25a1"
        tries = []
        for m in range(1, 5):
            for p in range(0, n - m):
                fr = self._frame(u, p, m)
                if any(x != held for x in fr[2]) or (m, fr[2]) not in shapes:
                    continue                                   # only words with relations carry their change ...
                alts = [a for a in slots.get((fr[1], fr[3]), ())  # ... for a part that stands for something too
                        if (a[1], a[2]) != (m, fr[2]) and held in a[2]][:2]
                if alts:
                    tries.append((-alts[0][0], -(p + m), p, m, fr, alts))
        tries.sort(key=lambda x: x[:4])
        for _, _, p, m, fr, alts in tries:
            for count, m2, shape2, code in alts:
                budget[0] -= 1
                if budget[0] < 0:
                    return None, 0.0, 0
                i0, q0 = divmod(code, 1024)
                y = self.said[i0][q0:q0 + m2] if self.said[i0] is not None else None
                if not y or not self._apart(u[p:p + m], y):
                    continue
                rest_plan, rest_e, _ = self._by_parts(u[:p] + y + u[p + m:], memo, budget, depth + 1)
                if rest_plan is None:
                    continue
                around = self._window(u, p, m)
                def near(c: int) -> int:
                    i, p2 = divmod(c, 1024)
                    o = self.said[i]
                    return 0 if o is None else -self._near(u, p, o, p2, m)
                def vote(codes) -> Tuple[Counter, int]:
                    votes: Counter = Counter()
                    filled = 0
                    for c in sorted(codes, key=near)[:4 * self.k]:
                        i, p2 = divmod(c, 1024)
                        o = self.said[i]
                        if o is None or not self._apart(o[p2:p2 + m], y):
                            continue
                        a2_rest = self._rest(o[:p2] + y + o[p2 + m:])
                        if a2_rest is None:
                            continue
                        got = self._transplant(o, p2, m, a2_rest, list(self.answers[i]), u, p, rest_plan, instead=y)
                        if got is None:
                            continue
                        votes[tuple(got)] += 1
                        filled += 1
                        if filled >= self.k:
                            break
                    return votes, filled
                same = self._windows_of(fr).get(around, ()) if fr in self._frames else ()
                votes, filled = vote(same)                     # the same surroundings first ...
                elsewhere = not votes
                if elsewhere:                                  # ... else where both kinds of part were lived
                    places = []
                    for fr2 in shapes.get((m, fr[2]), ()):
                        if any((a[1], a[2]) == (m2, shape2) for a in slots.get((fr2[1], fr2[3]), ())):
                            places.append(((fr2[1] != fr[1]) + (fr2[3] != fr[3]), -len(self._frames[fr2]), fr2))
                    places.sort(key=lambda x: x[:2])
                    codes: List[int] = []
                    for _, _, fr2 in places:
                        codes.extend(c for c in self._frames[fr2][:4 * self.k]
                                     if self.said[c // 1024] is not None
                                     and self._window(self.said[c // 1024], c % 1024, m) != around)
                        if len(codes) >= 16 * self.k:
                            break
                    votes, filled = vote(codes)
                if not votes:
                    continue
                record = self.exchange_record(m, fr[2], m2, shape2, y, elsewhere)
                best, agree = votes.most_common(1)[0]
                e = min(rest_e, record * self._agreement(agree, filled))
                if e >= SPEAKING_LINE:
                    return list(best), e, agree
        return None, 0.0, 0

    def _exchange_pool(self, m: int, shape: tuple, m2: int, shape2: tuple) -> List[int]:
        """The lived parts of one shape standing where a part of another shape was lived too (between the same
        neighbours), from the frames lived most often."""
        slots = getattr(self, "_slots", None) or {}
        frames = [fr2 for fr2 in (getattr(self, "_shapes", None) or {}).get((m, shape), ())
                  if any((a[1], a[2]) == (m2, shape2) for a in slots.get((fr2[1], fr2[3]), ()))]
        frames.sort(key=lambda fr2: -len(self._frames[fr2]))
        codes: List[int] = []
        for fr2 in frames:
            codes.extend(self._frames[fr2][:4 * self.k])
            if len(codes) >= 16 * self.k:
                break
        return codes

    def exchange_record(self, m: int, shape: tuple, m2: int, shape2: tuple, y: Tuple[str, ...],
                        elsewhere: bool = False, sample: int = 24) -> float:
        """How far exchanging a part of this shape for ``y`` is believed: the rule on lived examples of such parts,
        each foreseen as if it had not been lived: its part exchanged for ``y``, that rest's answer read, and what the
        nearest other example shows the exchange changes put back: an example with the same surroundings, or, with
        ``elsewhere``, one with other surroundings (does a change read in one place carry to another?). An exchange
        is only made where the asked rest has an answer and examples can be put into it, so what counts is the
        answers so foreseen and how many were the answer lived (as for the analogy's own record); an example that
        cannot be foreseen is not counted. (A frame's record counts it as not right: there the record also says how
        far a part is a whole, and orders the search.)"""
        key = (m, shape, tuple(y), elsewhere)
        if key in self._exchanges:
            return self._exchanges[key]
        pool = self._exchange_pool(m, shape, m2, shape2)
        step = max(1, len(pool) // sample)
        right = n = 0
        for code in pool[::step][:sample]:
            i, p2 = divmod(code, 1024)
            u = self.said[i]
            if u is None or not self._apart(u[p2:p2 + m], y):
                continue                                       # foreseen as an exchange is asked
            rest = self._rest(u[:p2] + tuple(y) + u[p2 + m:])
            if rest is None:
                continue
            around = self._window(u, p2, m)
            others = []
            for c2 in pool:
                j, q2 = divmod(c2, 1024)
                o = self.said[j]
                if (o is not None and o != u and (self._window(o, q2, m) != around) == elsewhere
                        and self._apart(o[q2:q2 + m], y)):
                    others.append((-self._near(u, p2, o, q2, m), j, q2, o))
            others.sort(key=lambda x: x[0])
            for _, j, q2, o in others[:3]:
                o_rest = self._rest(o[:q2] + tuple(y) + o[q2 + m:])
                if o_rest is None:
                    continue
                got = self._transplant(o, q2, m, o_rest, list(self.answers[j]), u, p2, rest, instead=y)
                if got is not None:
                    n += 1
                    right += got == list(self.answers[i])
                    break
        self._exchanges[key] = (right + 0.5) / (n + 1.0)
        return self._exchanges[key]

    @staticmethod
    def _apart(part: Sequence[str], y: Sequence[str]) -> bool:
        """Whether a part and the part exchanged for it share no word, as an exchange asks: it puts one part's
        meaning where another's was, and where the two share a word, what becomes of that word's tokens cannot be
        seen. Asked, "jump" for "jump opposite" adds a word that brings no token of its own; in an example, "look"
        for "look opposite" keeps "look"'s token where the asked part ("run") would change it."""
        return not set(part) & set(y)

    @staticmethod
    def _agreement(agree: int, filled: int) -> float:
        """How far the examples' votes carry a plan by parts: the one rule on them, (agree + ½) / (filled + 1), so
        that one example alone carries a plan no further than three quarters."""
        return (agree + 0.5) / (filled + 1.0)

    @staticmethod
    def _shift(a: Sequence[str], p: int, m: int, n: int, y: int = 0) -> List[str]:
        """The places of an answer to an utterance of n tokens in which a part of m tokens at p was exchanged for
        one of y tokens (y = 0: taken out), read as places of the utterance with its own part: those after the
        exchanged part move on by m - y, and the exchanged part's own places are marked as its."""
        out = []
        for t in a:
            if t.isdigit() and p <= int(t) < n:
                k = int(t)
                out.append(PART % (k - p) if k < p + y else str(k - y + m))
            else:
                out.append(t)
        return out

    def _transplant(self, o: Tuple[str, ...], p2: int, m: int, a2_rest: List[str], a2: List[str],
                    u: Tuple[str, ...], p: int, rest_plan: List[str], instead: Sequence[str] = ()) -> Optional[List[str]]:
        """What taking the part out of the example (o, its part at p2), or exchanging it for ``instead``, changed in its
        answer, put into the answer of the asked utterance's rest (u, its part at p, likewise taken out or
        exchanged). None where the change cannot be put, or where it leaves anything of ``instead`` in the answer, or
        where the example's answer names no places of what was said and a word with a relation stands both in the
        example's part and outside it: which of the two its tokens belong to cannot then be told ("look opposite right
        and look opposite right twice" says nothing of taking "opposite right and look" out of a command whose first
        action is not "look"), while in an answer that names places a token is told apart by its place."""
        if not any(t.isdigit() and int(t) < len(o) for t in a2):
            inside = set(o[p2:p2 + m])
            if any(w in inside and self.relation(w) is not None for w in o[:p2] + o[p2 + m:]):
                return None
        a2s = self._shift(a2_rest, p2, m, len(o) - m + len(instead), len(instead))
        a_s = self._shift(rest_plan, p, m, len(u) - m + len(instead), len(instead))
        def cls(k: int, pp: int) -> str:
            if pp <= k < pp + m:
                return "S%d" % (k - pp)
            if k < pp:
                return "L%d" % (pp - k) if pp - k <= 3 else "L"
            return "R%d" % (k - pp - m) if k - pp - m <= 3 else "R"
        def read(a: List[str], w: Tuple[str, ...], pp: int) -> List[str]:
            near: Dict[str, int] = {}                          # each relation, by the place of its word nearest the part
            for k, x in enumerate(w):
                r = self.relation(x)
                if r is not None and (r not in near or abs(k - pp) < abs(near[r] - pp)):
                    near[r] = k
            out = []
            for t in a:
                if t.isdigit() and int(t) < len(w):
                    out.append(cls(int(t), pp))
                elif t in near:
                    out.append("\u25a1" + cls(near[t], pp))
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
                elif r2[bi + d].startswith("\u25a1"):
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
            if i == 0:                                         # put first where the example put it first
                return 0
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
        for _, (where, k, new) in sorted(enumerate(edits), key=lambda x: (-x[1][0], -x[0])):
            out[where:where + k] = new                         # from the end, so that each place still holds
        if instead and any(t.startswith("\u27e8") for t in out):
            return None                                        # the part put in for the asked one is still named
        return out

    def _plan(self, said: Sequence[str]) -> Tuple[Optional[List[str]], int, int, int]:
        votes: Counter = Counter()
        first: Dict[Tuple[str, ...], int] = {}
        filled = 0
        for i, places in self.neighbours(said):
            other = self.said[i]
            mapping: Dict[str, str] = {}                       # every place filled at once (two roles may swap)
            ok = True
            kept = {other[k] for k in range(len(other)) if k not in places}
            for j in places:
                u, v = other[j], said[j]
                if u in kept:                                  # u also stands unchanged elsewhere: its token cannot
                    ok = False                                 # be told from the other's
                    break
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
            if not self._as_lived(said, places, answer):
                continue
            votes[tuple(answer)] += 1
            first.setdefault(tuple(answer), len(places))
            filled += 1
            if filled >= self.k:
                break
        if not votes:
            return None, 0, 0, 0
        best, agree = votes.most_common(1)[0]
        return list(best), agree, filled, first[best]
