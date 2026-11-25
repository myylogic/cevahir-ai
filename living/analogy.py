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
  A' that is u's relation becomes v's relation (v itself where u's relation is u itself and v was never met: a word
  lived again and again that never brought a token of its own, as "The" in COGS, is no name to stand for one). All
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
  side, read by shape, the one right beside the part on each side by its kind: "that" and a name are one mark but
  not one kind, and what comes right after a part decides what it hangs on): what taking their part out changes in
  their answers. The change is put into the rest's answer:
  the part's own words and places by their place in the part, everything else through the alignment of the two rests'
  answers (read by what they say about the places near the part: the numbers that name them and the relations of the
  words there; a place far from the part, and its word's token, by what links it to the place right beside the part,
  the answer's items followed link by link: the verb "to a hero" hangs on governs "table" through "nmod . on" and
  "theme", and is put for the token that governs the place beside the asked part through "theme", not for the first
  verb); a token of an answer that is a whole number smaller than the utterance's length is read as a place of it,
  and moves with the places. A change made inside one of the example's items (where the answers keep lists) is made
  inside the item it stands for: the example's item, put into the asked rest's answer, is found there once, and the
  change is made in it at the same place (read token by token, the example's clause of "tolerated", whose agent is a
  thing, lines up with a far clause whose agent is a thing too, while the clause that holds the asked part is
  "Levi respected"'s). What the example put at the very beginning goes at the beginning; where an answer lists runs
  of the same shape, a new run goes by its place among its neighbours (a list keeps the order of what was said).
  The examples vote; a step's belief is the frame's record times the share of the examples that agree, and a plan's
  belief is that of its weakest step (the rest's own included): the steps of a chain read the same kind of change
  again and again, so their errors go together. A change no lived example carries stays below the line: SCAN's
  "around" turns four times for each turn "twice" or "thrice" asks, and an example with "twice" after the part, whose
  surroundings differ, says nothing of "thrice". The most whole-like parts are tried first, then the rightmost; the
  search is bounded.
- **Exchanging a part.** A part that cannot be taken out (what remains would not be an utterance) is exchanged for a
  part lived between the same neighbours: for every part the two lived most often there first, then the next eight,
  on a bound of their own. The rest is answered as the exchange's record answers the rests of its examples (the one
  lived, or the analogy's own at the speaking line), and what such an exchange changes in lived answers is put back
  the same way. Only parts made of words with relations are exchanged: what such a word brings to
  an answer is its own token wherever it stands, while what a word without a relation brings depends on where it
  stands (SCAN's "right" turns once, or four times after "around"), and only for a part that holds a word with a
  relation and none of the asked part's words (an exchange puts one part's meaning where another's was; "jump" for
  "jump opposite" would add "opposite", a word that brings no token of its own). The change is read from the asked
  part's own frame with the same surroundings or, where there is none, wherever both kinds of part were lived, and
  only from examples whose part, too, shares no word with the one put in ("look left after run", exchanging "look"
  for "look opposite", keeps "look"'s token, and says nothing of what becomes of it when "run" is exchanged); each
  case is believed by its own record (lived examples exchanged as if not lived, the change read from an example with
  the same, or with other, surroundings). Taking a part out is the exchange for nothing.
- **The order of an answer's lists.** Where the lived answers keep lists, an answer put together by parts is said in
  their order: a token that ends items and one that stands between them (every item naming a place) are read from the
  lived answers, as the pair that leaves the most of them as they were lived when their items are put in the order of
  the places each names (its numbers, and the own tokens of the words said once, a name, each read as the place it
  was said), the ended items first; it is believed by the rule on all of them, and used at the speaking line. A
  change put back where the example put it may leave a run out of that order; the answer is said in the order all
  lived answers keep, at every step of a chain, so that the next step reads a list whose items stand where their
  places put them (taking "donut beside a" out of "Skylar gave the donut beside a bench to the girl ." adds its items
  after the recipient, the last of them; put into an answer that goes on past them, they would stand last, and the
  step after would read the answer's items out of line). In COGS the pair is ";" and "AND", and every one of the lived
  answers that name places is given back by it from its items in any order; SCAN's answers name no places and have
  none.
- **A second search, with words read by their kind.** Where the search for parts is silent, a second one is made. In
  it a word with a relation is read by its kind: the mark and the token its relation has right after it in the lived
  answers, where one token does by the rule (a noun's and a preposition's "(", a verb's "."), so a phrase hung on a
  subject ("A donkey [in the room] sold ...") is no longer read as a verb with its object ("A donkey [sold the cake]
  to ..."). A part that no lived part of its frame, so read, stood in the surroundings of (the PARTS_AROUND tokens on
  each side, read so) is taken out first, from the right (it has no record of its own there), by what parts of its
  shape do elsewhere: the change is read from lived parts of that shape, nearest first, and believed by the shape's
  record elsewhere (lived examples of it, each foreseen with the change read from the nearest example with other
  surroundings, counted as a frame's record is) times the share of the examples that agree. What the first search
  answers is answered as before.
- **A change read from other surroundings.** Where its example put a run says nothing of where it goes in other
  surroundings, so such a change is put as the items the example's answer gained and lost: the lost ones must stand
  in the rest's answer and are taken away, the gained ones are put in, and the answer is said in the order of its
  lists. A token is read by where it stands near the part, where the words standing at that place in the example and
  in the asked utterance are of a kind; else through the alignment of the two rests' answers, and no place or word's
  token is put for one whose word is of another kind (a verb's place before "to the girl" in a passive is not the
  noun's place before it in an active sentence).
- **The roles an answer gives.** The answer given (not the answers of rests, which are steps on the way and are
  often utterances no one would say) is believed no further than the roles it gives its tokens were lived: an item
  that gives a token a role it never took in a lived answer is believed by the rule on the tokens that took every role
  it took, how many of them took that one too. "like" takes no recipient in any lived answer, and of the tokens that
  take all it takes none does: "Aubrey liked that Grayson gave Lina to a cat .", answered through the rest "Aubrey
  liked the ring to a cat .", gives "like" a recipient and is held back; where the first search's answer is held back
  so, the second search is made. A token that took no role is not read so (nothing lived says what roles it takes),
  and a token that took few takes a new one as the tokens that took those do (a verb met only with a theme is asked
  with an agent as most such verbs take one).

Nothing here names a language, a word or a format: tokens are what was said, split as Zihin splits (words and marks).
"""
from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from living.evidence import SPEAKING_LINE

TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PARTS_BUDGET = 4000                                            # parts tried, at most, for one utterance
PARTS_DEPTH = 16                                               # parts taken out, at most, one after another
PARTS_AROUND = 3                                               # tokens on each side of a part an example must share
RECORD_STEP = 50                                               # parts tried to answer a record's example one step on
EXCHANGE_FIRST = 2                                             # parts lived in a place, tried first for an exchange ...
EXCHANGE_MORE = 10                                             # ... and then up to this many, on a bound of their own
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
        self._units_plain: Dict[tuple, float] = {}             # ... read inside a record's own step (below)
        self._steps: Dict[tuple, Optional[List[str]]] = {}     # the rests a record read one step on
        self._in_step = False
        self._rests: Dict[tuple, Optional[List[str]]] = {}     # the answers of rests, as the parts read them
        self._windows: Dict[tuple, Dict[tuple, List[int]]] = {}  # each frame's examples, by what stands around them
        self._exchanges: Dict[tuple, float] = {}               # the record of each kind of exchange
        self._asked: Dict[tuple, tuple] = {}
        self._alike: Dict[Tuple[str, str], bool] = {}
        self._order: Optional[Tuple[str, str]] = None          # how lived answers keep their lists (read as life doubles)
        self._order_at = -1
        self._at: Dict[tuple, List[int]] = {}                  # the dialogues of each word, by its neighbours' shapes
        self._owns: Dict[tuple, Dict[int, str]] = {}           # what its token always had after it there
        self._next: Dict[str, str] = {}                        # what each relation's token has right after it
        self._mframes: Dict[tuple, List[int]] = {}             # the frames read by kind (the second search)
        self._mshapes: Dict[tuple, list] = {}
        self._elsewhere: Dict[tuple, float] = {}               # the record of a shape's parts, read elsewhere
        self._here: Dict[tuple, bool] = {}                     # whether a part of a frame stood in such surroundings
        self._roles: Dict[str, set] = {}                       # the roles each token took (read as life doubles)
        self._role_beliefs: Dict[tuple, float] = {}
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
        self._order_at = -1
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
        that agreed. The answer given is believed no further than the roles it gives its tokens were lived
        (``_roles_believed``); where the first search's answer is held back so, the second search is made."""
        key = (tuple(said), parts, len(self.said), self._forgotten)
        if key in self._asked:                                 # asked again before anything new was lived
            plan, e, n = self._asked[key]
            return (list(plan) if plan is not None else None), e, n
        got = self._plain(said)
        if got[0] is not None and got[1] >= SPEAKING_LINE:
            got = (got[0], min(got[1], self._roles_believed(got[0])), got[2])
        if parts and self.said and not (got[0] is not None and got[1] >= SPEAKING_LINE):
            plan, e, n = self._by_parts(tuple(said), {}, [PARTS_BUDGET], 0)
            if plan is not None:
                e = min(e, self._roles_believed(plan))
            if plan is None or e < SPEAKING_LINE:              # nothing lived between the same neighbours answers,
                second = self._by_parts(tuple(said), {}, [PARTS_BUDGET], 0, True)   # or not with roles lived
                if second[0] is not None:
                    second = (second[0], min(second[1], self._roles_believed(second[0])), second[2])
                    if plan is None or second[1] > e:
                        plan, e, n = second
            if plan is not None:
                got = (self._in_order(plan, said), e, n)
        if len(self._asked) > 4096:
            self._asked.clear()
        self._asked[key] = got
        return (list(got[0]) if got[0] is not None else None), got[1], got[2]

    def _heads(self, a: Sequence[str]) -> List[Tuple[str, tuple]]:
        """The tokens an answer's items give roles to, with the roles: an item "token . role ( ... )" of an answer that
        keeps lists (``_list_order``); nothing where the answers keep none."""
        order = self._list_order()
        if order is None:
            return []
        out, item = [], []
        for t in list(a) + [order[1]]:
            if t not in order:
                item.append(t)
                continue
            if "(" in item:
                j = item.index("(")
                if j >= 2 and item[1] == ".":
                    out.append((item[0], tuple(item[1:j])))
            item = []
        return out

    def _roles_believed(self, plan: Sequence[str]) -> float:
        """How far the roles an answer gives its tokens were lived: a role a token took in a lived answer is believed;
        one it never took, by the rule on the tokens that took every role it took (how many of them took that one too:
        "like" never takes a recipient, and few of the tokens that take what it takes do). A token that took no role
        is not read so: nothing lived says what roles it takes."""
        self._make()
        least = 1.0
        for tok, role in self._heads(plan):
            mine = self._roles.get(tok)
            if not mine or role in mine:
                continue
            key = (tok, role)
            if key not in self._role_beliefs:
                others = [r for t, r in self._roles.items() if t != tok and mine <= r]
                self._role_beliefs[key] = (sum(role in r for r in others) + 0.5) / (len(others) + 1.0)
            least = min(least, self._role_beliefs[key])
        return least

    def _named(self, said: Sequence[str]):
        """The places an item of an answer to ``said`` names, in its order: its numbers, and the own tokens of the
        words said once (a word whose token is the word itself, a name), each read as the place it was said."""
        once = Counter(said)
        where = {w: j for j, w in enumerate(said) if once[w] == 1 and self.relation(w) == w}
        def key(item) -> tuple:
            return tuple(int(t) if t.isdigit() else where[t] for t in item if t.isdigit() or t in where)
        return key

    def _ordered(self, a: Sequence[str], end: str, between: str, said: Sequence[str] = ()) -> Optional[List[str]]:
        """An answer's items in the order of the places each names (``_named``): the items ``end`` ends first, then
        the items ``between`` joins, each kept as its own list; None where an item names no place (not a list of
        this kind)."""
        items, cur = [], []
        for t in a:
            if t == end or t == between:
                items.append((cur, t))
                cur = []
            else:
                cur.append(t)
        items.append((cur, None))
        def first(item: List[str]) -> int:
            for t in item:
                if t.isdigit():
                    return int(t)
            return -1
        if any(first(it) < 0 for it, _ in items):
            return None
        key = self._named(said)
        ended = sorted((it for it, sep in items if sep == end), key=key)    # stable: a tie keeps its order
        joined = sorted((it for it, sep in items if sep != end), key=key)
        out: List[str] = []
        for it in ended:
            out += it + [end]
        for k, it in enumerate(joined):
            out += ([between] if k else []) + it
        return out

    def _list_order(self) -> Optional[Tuple[str, str]]:
        """How the lived answers keep their lists, if they keep them so (MODEL.md 12.6): a token that ends items and one
        that stands between them, every item naming a place, such that the lived answers holding them are said with
        their items in the order of the places each names, the ended items first (``_ordered``). The pair is the
        one whose rule on the answers it leaves as they were lived is highest, read from a sample of them and believed
        by the rule on all; None where it does not reach the speaking line. Read again as life doubles."""
        if self._order_at >= 0 and len(self.said) < 2 * max(1, self._order_at):
            return self._order
        self._order_at = len(self.said)
        related = {r for r in (self.relation(w) for w in self.with_w if self.with_w[w] > 0) if r is not None}
        lived = [(a, s) for a, s in zip(self.answers, self.said) if a is not None]
        seen = Counter(t for a, _ in lived for t in set(a))
        marks = sorted(t for t, c in seen.items() if c >= 2 and not t.isdigit() and t not in related)
        sample = [(a, set(a), s) for a, s in lived[::max(1, len(lived) // 512)][:512]]
        def rule(pairs, end: str, between: str) -> Tuple[float, int]:
            n = same = 0
            for a, held, said in pairs:
                if end in held or between in held:
                    got = self._ordered(a, end, between, said)
                    if got is not None:
                        n += 1
                        same += got == list(a)
            return (same + 0.5) / (n + 1.0), n
        best = None
        for end in marks:
            for between in marks:
                if end != between:
                    r = rule(sample, end, between)
                    if r[1] and (best is None or r > best[0]):
                        best = (r, end, between)
        self._order = None
        if best is not None and rule([(a, set(a), s) for a, s in lived], best[1], best[2])[0] >= SPEAKING_LINE:
            self._order = (best[1], best[2])
        return self._order

    def _in_order(self, plan: List[str], said: Sequence[str] = ()) -> List[str]:
        """A plan put together by parts, said in the order the lived answers keep their lists (``_list_order``)."""
        order = self._list_order()
        if order is None:
            return plan
        got = self._ordered(plan, order[0], order[1], said)
        return plan if got is None else got

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
        after: Dict[str, Counter] = defaultdict(Counter)        # what each relation's token has right after it
        for a in self.answers:
            if a is None:
                continue
            for q, t in enumerate(a):
                if t in related:
                    after[t][self._after(a, q + 1)] += 1
        self._next = {}
        for t, c in after.items():
            x, k = max(c.items(), key=lambda kv: (kv[1], kv[0]))
            if (k + 0.5) / (sum(c.values()) + 1.0) >= SPEAKING_LINE:
                self._next[t] = x
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
        mframes: Dict[tuple, List[int]] = defaultdict(list)     # the frames again, every word read by its kind
        for i, u in enumerate(self.said):
            if u is None:
                continue
            for m in range(1, 5):
                for p in range(0, min(len(u) - m, 1024)):
                    mframes[self._mframe(u, p, m)].append(i * 1024 + p)
        mshapes: Dict[tuple, list] = defaultdict(list)
        for fr in mframes:
            mshapes[(fr[0], fr[2])].append(fr)
        self._mframes = dict(mframes)
        self._mshapes = dict(mshapes)
        at: Dict[tuple, List[int]] = defaultdict(list)          # each word's dialogues, by the shapes of its neighbours
        for i, u in enumerate(self.said):
            if u is None:
                continue
            for j, w in enumerate(u):
                key = (w, self._shape(u[j - 1]) if j else "^", self._shape(u[j + 1]) if j + 1 < len(u) else "$")
                if len(at[key]) < 64:
                    at[key].append(i)
        self._at = dict(at)
        roles: Dict[str, set] = defaultdict(set)              # the roles each token took in the lived answers
        for a in self.answers:
            if a is not None:
                for tok, role in self._heads(a):
                    roles[tok].add(role)
        self._roles = dict(roles)
        self._role_beliefs = {}
        self._owns = {}
        self._elsewhere = {}
        self._here = {}
        self._units = {}
        self._units_plain = {}
        self._steps = {}
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

    def _kind(self, w: str) -> str:
        """A word read by its kind: one with a relation as the mark and the token its relation has right after it in
        the lived answers, where one token does by the rule (a noun's and a preposition's "(", a verb's ".")."""
        r = self.relation(w)
        return w if r is None else "\u25a1" + self._next.get(r, "")

    def _mframe(self, u: Sequence[str], p: int, m: int) -> tuple:
        """A part's frame with every word read by its kind (``_kind``)."""
        left = tuple(self._kind(w) for w in u[max(0, p - 2):p])
        left = ("^",) * (2 - len(left)) + left
        right = tuple(self._kind(w) for w in u[p + m:p + m + 1]) or ("$",)
        return (m, left, tuple(self._kind(w) for w in u[p:p + m]), right)

    def _mwindow(self, u: Sequence[str], p: int, m: int) -> tuple:
        """What stands around a part, every word read by its kind."""
        w = PARTS_AROUND
        return tuple("^" if k < 0 else ("$" if k >= len(u) else self._kind(u[k]))
                     for k in list(range(p - w, p)) + list(range(p + m, p + m + w)))

    def _lived_here(self, fr: tuple, around: tuple) -> bool:
        """Whether a lived part of this frame (words read by kind) stood in these surroundings."""
        key = (fr, around)
        got = self._here.get(key)
        if got is None:
            got = False
            for code in self._mframes.get(fr, ()):
                i, p2 = divmod(code, 1024)
                o = self.said[i]
                if o is not None and self._mwindow(o, p2, fr[0]) == around:
                    got = True
                    break
            self._here[key] = got
        return got

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
        """What stands around a part (u, at p, m tokens): the PARTS_AROUND tokens on each side of it, read by shape,
        and the one right beside it on each side read by its kind: what comes right before or after a part decides
        what it hangs on ("that" and a name are one mark, a word with a relation, but not one kind)."""
        w = PARTS_AROUND

        def read(k: int) -> str:
            if k < 0:
                return "^"
            if k >= len(u):
                return "$"
            return self._kind(u[k]) if k in (p - 1, p + m) else self._shape(u[k])
        return tuple(read(k) for k in list(range(p - w, p)) + list(range(p + m, p + m + w)))

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
        the speaking line, or by parts one step on: ``_rest_read``), and the change the nearest other example with
        the same surroundings shows put back. An example counts as right when that gives the answer that was lived;
        one whose rest has no answer, or that no other example can be put into, counts as not right: a part is a
        whole as far as taking it out leaves what can be answered and putting it back gives what was lived."""
        units = self._units_plain if getattr(self, "_in_step", False) else self._units
        if fr in units:
            return units[fr]
        units[fr] = 0.0                                        # while it is read, a frame is no whole
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
            rest = self._rest_read(u[:p2] + u[p2 + m:])
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
                o_rest = self._rest_read(o[:q2] + o[q2 + m:])
                if o_rest is None:
                    continue
                got = self._transplant(o, q2, m, o_rest, list(self.answers[j]), u, p2, rest)
                if got is not None:
                    right += got == list(self.answers[i])
                    break
        units[fr] = (right + 0.5) / (n + 1.0)
        return units[fr]

    # ----- answering by parts -----

    def _by_parts(self, u: Tuple[str, ...], memo: dict, budget: list, depth: int, elsewhere: bool = False
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
        if elsewhere:                                          # the second search: the parts never lived here first
            got = self._by_elsewhere(u, memo, budget, depth)
            if got[0] is not None:
                memo[u] = got
                return memo[u]
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
            rest_plan, rest_e, _ = self._by_parts(u[:p] + u[p + m:], memo, budget, depth + 1, elsewhere)
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
        if depth == 0 and not elsewhere:                       # the asked utterance's own parts, on their own budget
            got = self._by_exchange(u, memo, [PARTS_BUDGET // 4], depth)
            if got[0] is not None:
                memo[u] = got
        return memo[u]

    def _by_elsewhere(self, u: Tuple[str, ...], memo: dict, budget: list, depth: int
                      ) -> Tuple[Optional[List[str]], float, int]:
        """The second search's own step: a part that no lived part of its frame stood in the surroundings of (every
        word read by its kind, the PARTS_AROUND tokens on each side: where the changes of the first search are read
        from) is taken out by what parts of its shape do elsewhere: the change is read from lived parts of the shape,
        nearest first, and put in as the items it adds and takes away (``_by_items``); a step is believed by the
        shape's record elsewhere (``unit_elsewhere``) times the share of the examples that agree. With no record of
        their own here, such parts are taken out from the right."""
        n = len(u)
        tries = []
        for m in range(1, 5):
            for p in range(0, n - m):
                fr = self._mframe(u, p, m)
                if (m, fr[2]) not in self._mshapes or self._lived_here(fr, self._mwindow(u, p, m)):
                    continue
                r = self.unit_elsewhere(m, fr[2])
                if r >= SPEAKING_LINE:
                    tries.append((-(p + m), p, m, r, fr))
        tries.sort(key=lambda x: x[:3])
        for _, p, m, r, fr in tries:
            budget[0] -= 1
            if budget[0] < 0:
                break
            rest_plan, rest_e, _ = self._by_parts(u[:p] + u[p + m:], memo, budget, depth + 1, True)
            if rest_plan is None:
                continue
            votes: Counter = Counter()
            filled = 0
            def near(code: int) -> int:
                i, p2 = divmod(code, 1024)
                o = self.said[i]
                return 0 if o is None else -self._near(u, p, o, p2, m)
            for code in sorted(self._shape_pool(m, fr[2]), key=near)[:4 * self.k]:
                i, p2 = divmod(code, 1024)
                o = self.said[i]
                if o is None:
                    continue
                a2_rest = self._rest(o[:p2] + o[p2 + m:])
                if a2_rest is None:
                    continue
                got = self._transplant(o, p2, m, a2_rest, list(self.answers[i]), u, p, rest_plan, items=True)
                if got is None:
                    continue
                votes[tuple(got)] += 1
                filled += 1
                if filled >= self.k:
                    break
            if votes:
                best, agree = votes.most_common(1)[0]
                e = min(rest_e, r * self._agreement(agree, filled))
                if e >= SPEAKING_LINE:
                    return list(best), e, agree
        return None, 0.0, 0

    def _shape_pool(self, m: int, shape: tuple) -> List[int]:
        """The lived parts of a shape (words read by kind), from the frames it was lived in most often."""
        frames = sorted(self._mshapes.get((m, shape), ()), key=lambda fr2: (-len(self._mframes[fr2]), fr2))
        codes: List[int] = []
        for fr2 in frames:
            codes.extend(self._mframes[fr2][:4 * self.k])
            if len(codes) >= 16 * self.k:
                break
        return codes

    def unit_elsewhere(self, m: int, shape: tuple, sample: int = 24) -> float:
        """How far a part of this shape is a whole wherever it stands: up to 24 lived examples of it, each foreseen as
        if it had not been lived, with the change read from the nearest example with other surroundings put in as
        items, said in the order of the answer's lists; as for a frame (``unit``), an example whose rest has no answer
        or that nothing can be put into counts as not right."""
        key = (m, shape)
        if key in self._elsewhere:
            return self._elsewhere[key]
        pool = self._shape_pool(m, shape)
        step = max(1, len(pool) // sample)
        right = n = 0
        for code in pool[::step][:sample]:
            i, p2 = divmod(code, 1024)
            u = self.said[i]
            if u is None:
                continue
            n += 1
            rest = self._rest_read(u[:p2] + u[p2 + m:])
            if rest is None:
                continue
            around = self._window(u, p2, m)
            others = []
            for c2 in pool:
                j, q2 = divmod(c2, 1024)
                o = self.said[j]
                if o is not None and o != u and self._window(o, q2, m) != around:
                    others.append((-self._near(u, p2, o, q2, m), j, q2, o))
            others.sort(key=lambda x: x[:3])
            for _, j, q2, o in others[:3]:
                o_rest = self._rest_read(o[:q2] + o[q2 + m:])
                if o_rest is None:
                    continue
                got = self._transplant(o, q2, m, o_rest, list(self.answers[j]), u, p2, rest, items=True)
                if got is not None:
                    right += self._in_order(got, u) == list(self.answers[i])
                    break
        self._elsewhere[key] = (right + 0.5) / (n + 1.0)
        return self._elsewhere[key]

    def _by_exchange(self, u: Tuple[str, ...], memo: dict, budget: list, depth: int
                     ) -> Tuple[Optional[List[str]], float, int]:
        """A part that cannot be taken out (what remains would not be an utterance) is exchanged for a part lived
        between the same neighbours: for every part, the EXCHANGE_FIRST lived most often there first; then the next
        ones, up to EXCHANGE_MORE, on a bound of their own (the parts lived less often are tried only where those
        lived most often, for every part, answer nothing). The rest is answered as the exchange's record answers the
        rests of its examples (``_rest_believed``: the one lived, or the analogy's own at the speaking line), and what
        that exchange changes in lived answers is put back, as for a part taken out: read from lived parts of the
        asked part's frame with the same surroundings, or, where there are none, wherever a part of its shape was
        lived where a part of the other's was too. Its belief is the exchange's own record for that case
        (``exchange_record``) times the share of the examples that agree; a rest answered by parts in turn would be
        believed beyond what that record says."""
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
                        if (a[1], a[2]) != (m, fr[2]) and held in a[2]][:EXCHANGE_MORE]
                if alts:
                    tries.append((-alts[0][0], -(p + m), p, m, fr, alts))
        tries.sort(key=lambda x: x[:4])
        start = budget[0]
        for lo, hi in ((0, EXCHANGE_FIRST), (EXCHANGE_FIRST, EXCHANGE_MORE)):
            budget[0] = start                                  # each round on a bound of its own
            for _, _, p, m, fr, alts in tries:
                for count, m2, shape2, code in alts[lo:hi]:
                    budget[0] -= 1
                    if budget[0] < 0:
                        break
                    i0, q0 = divmod(code, 1024)
                    y = self.said[i0][q0:q0 + m2] if self.said[i0] is not None else None
                    if not y or not self._apart(u[p:p + m], y):
                        continue
                    rest_plan, rest_e = self._rest_believed(u[:p] + y + u[p + m:])   # as its record reads it
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
                if budget[0] < 0:
                    break
        return None, 0.0, 0

    def _governors(self, ans: Sequence[str], w: Sequence[str], k: int) -> List[Tuple[str, tuple, str]]:
        """What governs the place k of the utterance w in its answer ans, link by link: the items "token . role
        ( x _ i , arg )" whose second argument names the place (its number, or, where the answer names a word by
        its own token, that token), then the place i in turn; each link as (i, role, token)."""
        links = []
        item: List[str] = []
        for t in list(ans) + ["AND"]:
            if t not in ("AND", ";"):
                item.append(t)
                continue
            if "(" in item and item[-1] == ")":
                j = item.index("(")
                inner = item[j + 1:-1]
                if len(inner) >= 5 and inner[:2] == ["x", "_"] and inner[3] == ",":
                    arg = inner[4:]
                    named = arg[2] if len(arg) == 3 and arg[:2] == ["x", "_"] else (arg[0] if len(arg) == 1 else None)
                    if named is not None and j > 0:
                        links.append((named, inner[2], tuple(item[1:j]), item[0]))
            item = []
        def chain(start: str) -> List[Tuple[str, tuple, str]]:
            out, seen, cur = [], {start}, start
            while True:
                got = next(((g, role, tok) for arg, g, role, tok in links if arg == cur), None)
                if got is None or got[0] in seen:
                    return out
                out.append(got)
                seen.add(got[0])
                cur = got[0]
        found = chain(str(k))
        if not found and self.relation(w[k]) == w[k]:
            found = chain(w[k])
        return found

    def _rest_read(self, r: Tuple[str, ...]) -> Optional[List[str]]:
        """The answer of a record's example once its part is out, read as an asked rest is: the one lived, or the
        analogy's own at the speaking line, or, where neither answers, by parts, one part taken out (at most
        RECORD_STEP tried); within that step records read their examples' rests by the analogy alone, so that no
        record waits on another being read. Only an example whose rest none of these answers counts against its
        frame for want of an answer."""
        got = self._rest(r)
        if got is not None or getattr(self, "_in_step", False):
            return got
        if r not in self._steps:
            self._steps[r] = None
            self._in_step = True
            try:
                plan, e, _ = self._by_parts(tuple(r), {}, [RECORD_STEP], PARTS_DEPTH)
            finally:
                self._in_step = False
            self._steps[r] = list(plan) if plan is not None and e >= SPEAKING_LINE else None
        got = self._steps[r]
        return list(got) if got is not None else None

    def _rest_believed(self, r: Tuple[str, ...]) -> Tuple[Optional[List[str]], float]:
        """The answer of a rest as a record reads its examples' rests (``_rest``: the one lived, or the analogy's own
        at the speaking line), with its belief."""
        if r in self.where:
            return list(self.answers[self.where[r]]), 1.0
        plan, e, _ = self._plain(r)
        if plan is None or e < SPEAKING_LINE:
            return None, 0.0
        return plan, e

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
                    u: Tuple[str, ...], p: int, rest_plan: List[str], instead: Sequence[str] = (),
                    items: bool = False) -> Optional[List[str]]:
        """What taking the part out of the example (o, its part at p2), or exchanging it for ``instead``, changed in its
        answer, put into the answer of the asked utterance's rest (u, its part at p, likewise taken out or
        exchanged); with ``items`` (a change read from other surroundings), as the items it adds and takes away
        (``_by_items``), which needs answers that keep lists: where they keep none, where an example put a run says
        nothing of other surroundings, and nothing is put. A change made inside one of the example's items is made
        inside the item that one stands for: the example's item, put into the asked rest's answer, found there once.
        The answer is said in the order the lived answers keep their lists (``_in_order``), so that the next step of a
        chain reads it item by item. None where the change cannot be put, or where it leaves anything of ``instead`` in
        the answer, or where the example's answer names no places of what was said and a word with a relation stands
        both in the example's part and outside it: which of the two its tokens belong to cannot then be told ("look
        opposite right and look opposite right twice" says nothing of taking "opposite right and look" out of a command
        whose first action is not "look"), while in an answer that names places a token is told apart by its place."""
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
        for side2, side in ((p2 - 1, p - 1), (p2 + m, p + m)):   # far tokens: by what links them to the place
            if not (0 <= side2 < len(o) and 0 <= side < len(u)):   # right beside the part
                continue
            mine_g = self._governors(a_s, u, side)
            for g2, role, pred2 in self._governors(a2s, o, side2):
                if not g2.isdigit() or int(g2) >= len(o) or cls(int(g2), p2) not in ("L", "R"):
                    continue                                   # near places are read by their place
                g, pred = next(((g, pred) for g, role_, pred in mine_g if role_ == role), (None, None))
                if g is None or not g.isdigit() or int(g) >= len(u):
                    continue
                places[g2] = Counter({g: len(a2s) + 1})
                if self.relation(o[int(g2)]) == pred2 and self.relation(u[int(g)]) == pred:
                    words[pred2] = Counter({pred: len(a2s) + 1})
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
                if rx != x or self.with_w.get(y, 0):
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
        if items:                                              # read from other surroundings: as items, where
            order = self._list_order()                         # the answers keep lists; else nothing can be put
            if order is None:
                return None
            return self._by_items(o, p2, u, p, m, a2s, a2, a_s, r2, r, put, theirs, order)
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
        order = self._list_order()
        def spans(seq: Sequence[str]) -> List[Tuple[int, int]]:
            out, s0 = [], 0
            for k, t in enumerate(seq):
                if t in order:
                    out.append((s0, k))
                    s0 = k + 1
            out.append((s0, len(seq)))
            return out
        sp2, sp = (spans(a2s), spans(a_s)) if order is not None else ([], [])
        def anchored(tag: str, i1: int, i2: int) -> Optional[int]:
            for s0, e0 in sp2:                                 # the item the change is made inside
                if (s0 < i1 < e0) if tag == "insert" else (s0 <= i1 < i2 <= e0):
                    item = [put(t) for t in a2s[s0:e0]]
                    if None in item:
                        return None
                    hits = [s1 for s1, e1 in sp if a_s[s1:e1] == item]
                    return hits[0] + (i1 - s0) if len(hits) == 1 else None
            return None
        edits = []
        for n_op, (tag, i1, i2, j1, j2) in enumerate(codes):
            seg = list(a2[j1:j2])
            if tag == "insert":                                # one change, many ways to write it: pushed on
                bound = codes[n_op + 1][1] if n_op + 1 < len(codes) else len(a2s)
                while i1 < bound and seg and a2s[i1] == seg[0]:
                    seg = seg[1:] + [a2s[i1]]
                    i1 += 1
                i2 = i1
            here_item = anchored(tag, i1, i2)                  # made inside an item: in the item it stands for
            where = at(i1) if here_item is None else here_item
            old = [put(t) for t in a2s[i1:i2]]
            new = [put(t) for t in seg]
            if None in old or None in new or a_s[where:where + len(old)] != old:
                return None
            if tag == "insert" and here_item is None:          # a list keeps the order of what was said
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
        return self._in_order(out, u)                          # said in the order the lived answers keep their lists

    @staticmethod
    def _items(a: Sequence[str], end: str, between: str) -> Optional[List[Tuple[Tuple[str, ...], bool]]]:
        """An answer's items, each with whether ``end`` ends it; None where an item names no place."""
        items, cur = [], []
        for t in a:
            if t == end or t == between:
                items.append((tuple(cur), t == end))
                cur = []
            else:
                cur.append(t)
        items.append((tuple(cur), False))
        if any(not any(x.isdigit() for x in it) for it, _ in items):
            return None
        return items

    def _by_items(self, o, p2, u, p, m, a2s, a2, a_s, r2, r, put, theirs, order) -> Optional[List[str]]:
        """A change read from other surroundings, put as the items the example's answer gained and lost: where the
        example put a run says nothing of where it goes here, so the items lost must stand in the rest's answer and
        are taken away, the items gained are put in, and the answer is said in the order of its lists. A token is
        read by where it stands near the part (the two answers read as ``_transplant`` reads them: a place before,
        inside or after the part, a word's token by the place of its word), where the words standing at that place
        in the example and here are of a kind (``_kind``); otherwise through the alignment of the two rests' answers,
        and a place or a word's token is not put for one whose word is of another kind. None where the answers are
        not lists of this kind or the change cannot be put."""
        end, between = order
        e2s, e2, es = self._items(a2s, end, between), self._items(a2, end, between), self._items(a_s, end, between)
        if e2s is None or e2 is None or es is None:
            return None
        def stands(seq: Sequence[str], pp: int, c: str) -> Optional[str]:
            k = pp + int(c[1:]) if c[0] == "S" else (pp - int(c[1:]) if c[0] == "L" else pp + m + int(c[1:]))
            return seq[k] if 0 <= k < len(seq) else None
        fine = set()                                           # the places near the part whose words are of a kind
        for c in ["S%d" % k for k in range(m)] + ["L1", "L2", "L3", "R0", "R1", "R2", "R3"]:
            x, y = stands(o, p2, c), stands(u, p, c)
            if x is not None and y is not None and self._kind(x) == self._kind(y):
                fine.add(c)
        mine: Dict[str, set] = defaultdict(set)                # each such place, the tokens read so here
        for t, c in zip(a_s, r):
            if (c[1:] if c.startswith("\u25a1") else c) in fine:
                mine[c].add(t)
        where: Dict[str, set] = defaultdict(set)               # each token of the example, the places it is read as
        for t, c in zip(a2s, r2):
            if (c[1:] if c.startswith("\u25a1") else c) in fine:
                where[t].add(c)
        rel_u = {self.relation(x) for x in u} - {None}
        def put_item_token(t: str) -> Optional[str]:
            cs = where.get(t)
            if cs and len(cs) == 1:
                c = next(iter(cs))
                if len(mine.get(c, ())) == 1:
                    return next(iter(mine[c]))
            got = put(t)
            if got is None or got == t:
                return got
            if t.isdigit() and got.isdigit() and int(t) < len(o) and int(got) < len(u):
                if self._kind(o[int(t)]) != self._kind(u[int(got)]):
                    return None                                # a place for one whose word is of another kind
            elif t in theirs and got in rel_u:
                x = next((w for w in o if self.relation(w) == t), None)
                y = next((w for w in u if self.relation(w) == got), None)
                if x is not None and y is not None and self._kind(x) != self._kind(y):
                    return None
            return got
        def put_item(it):
            toks = [put_item_token(t) for t in it[0]]
            return None if None in toks else (tuple(toks), it[1])
        left = Counter(e2s)
        gained = []
        for it in e2:
            if left[it] > 0:
                left[it] -= 1
            else:
                gained.append(it)
        kept = Counter(e2)
        lost = []
        for it in e2s:
            if kept[it] > 0:
                kept[it] -= 1
            else:
                lost.append(it)
        out = list(es)
        for it in lost:
            got = put_item(it)
            if got is None or got not in out:
                return None
            out.remove(got)
        for it in gained:
            got = put_item(it)
            if got is None:
                return None
            out.append(got)
        key = self._named(u)
        toks: List[str] = []
        for it in sorted((it for it, k in out if k), key=key):
            toks += list(it) + [end]
        for k, it in enumerate(sorted((it for it, k in out if not k), key=key)):
            toks += ([between] if k else []) + list(it)
        return toks

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
                    if ru == u and not self.with_w.get(v, 0):
                        rv = v                                 # the word itself, never met before
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
