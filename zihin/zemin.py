# -*- coding: utf-8 -*-
"""Zemin (the ground): the levels of meaning the mind stands on. Behind ``Config.grounded`` (off by default).

The deepened model (research/living_mind_2026_11_23/MODEL.md): a mind that only computes over word sequences has no
referent, no event and nothing to say; here what is perceived and what is said meet. Nothing of a language is
given: no word, no grammar, no role, no list of meanings. What perception gives is given (``Zemin.see``): tracked
things with opaque features on dimension channels (complete: what a thing has not is seen too), events with opaque
action features and the participation features of each participant (who starts it, who moves, who is turned to, who
is touched, ...). The words are heard (``Zemin.heard``).

Everything is evidence in the one store, read by the one rule (e = c (f - 1/2) + 1/2), and written in a moment of the
mind (``live``, called at the end of ``Mind.step``), so the ledger keeps it and Exact Retraction takes it back.

What is learned, and how (MODEL.md sections 2, 4, 6, 7):

* Perception binds features to a thing, an event or a participation (SAHIP, sensed); a value on a dimension where
  another one was expected is evidence against that one (the Expected-Failure Rule, the dimension being the channel).
* A word refers to one of the things talked about, or to the event (its action features and the time relation of
  the words to it: ``once``, the words came before the event; ``icinde``, while it went on), or to nothing. What is
  talked about is the participants of the event going on or just asked for, else the thing pointed at, else everything
  in view. Which word refers to what is the alignment of the utterance (``_align_words``): how well each word's
  meaning so far predicts each of them (``likelihood``: the geometric mean over its dimensions of what the word
  expects of its value, 1/2 where it expects nothing), as a share among the words (what one word explains another
  gets little of: mutual exclusivity) and among the referents (nothing, 1/2 on every dimension, is one of them). A word
  then names the features of what it refers to, by that share (ADLANDIRIR, word -> feature); where it expected another
  value of a dimension of its referent it gets evidence against it (Expected-Failure). The individual itself is a
  feature too (its identity): a proper name names one thing, a kind word the features its things share. This is
  cross-situational learning with the referent first (Siskind 1996; Fazly et al. 2010 align features directly, which
  cannot part two meanings that always come together: here a person is in every order).
* A construction is the form of an utterance whose places are named by what they named: a participant's place (the
  word that names it best), a modifier's place (a further word that tells it apart from the others in view), the
  action's place, and fixed words (what names nothing). Its places hold the participation profile of what filled them
  (YUVA, construction -> participation feature, lag = place): a role is the profile of a place, never a name. The words
  lived in each place are kept per action word (DOLGU, island -> word, lag = place: a verb island). The time relation
  it was used for (ADLANDIRIR construction -> time) and the turn that followed it from another speaker (EXPECT, turn
  -> turn) are evidence too.
* Understanding is the construction and the referents that fit the words best; answering is the meaning first: the
  event going on (or the one asked for, when nothing goes on), the construction the question was answered with, its
  places filled from the event by their profiles, the words chosen by how well they name what fills the place.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .events import IMITATED, SENSED
from .store import EXPECT, W_NEG, W_POS

SAHIP, ADLANDIRIR, YUVA, DOLGU = 4, 5, 6, 7     # tie forms of the ground (the store's form field has room for eight)
THING, EVENT, PART, PAIR = "z_nesne", "z_olay", "z_katilim", "z_olay_zaman"
TIME, CONS, ISLAND, TURN, IDENTITY = "z_zaman", "z_yapi", "z_ada", "z_tur", "z_kimlik"
FEAT, PFEAT = "z:", "zk:"
BEFORE, DURING = "once", "icinde"
HEAD, ACTION = "_", "A"                         # kinds of places (a modifier's place is ("M", its head's place))
FLOOR = 0.02                                    # the least a likelihood is read as (a value never met)


def rule(p: float, n: float, k: float = 1.0) -> float:
    """The one rule on plain numbers: e = c (f - 1/2) + 1/2, c = W / (W + k); 1/2 with no evidence."""
    w = p + n
    return 0.5 if w <= 0.0 else (w / (w + k)) * (p / w - 0.5) + 0.5


def hat(e: float) -> float:
    """The part of a belief beyond "I do not know": 2 max(0, e - 1/2)."""
    return max(0.0, 2.0 * (e - 0.5))


class Zemin:
    """The ground's working memory (what is in view, what goes on, what waits): rebuilt by perceiving; what is learned
    is in the store."""

    def __init__(self, word_channel: str) -> None:
        self.words = word_channel
        self.pending: List[tuple] = []           # perceived and heard since the last moment, lived in the next one
        self.present: Dict[int, int] = {}        # tracked index -> thing unit (what is in view)
        self.pointed: Optional[int] = None
        self.event: Optional[dict] = None        # what goes on: key, unit, action features, parts {index: unit}
        self.waiting: Optional[tuple] = None     # an utterance heard while nothing went on (an event may follow)
        self.turn: Optional[tuple] = None        # the last utterance understood: (its turns, speaker)
        self.answerer = "cevaplayan"             # the role the mind takes in a dialogue: what that one says back
                                                 # to another is learned as an answer (not what anyone says next)
        self.serial = 0
        self.memo: dict = {}

    # the session's (or a test's) calls: queued, lived in the next moment of the mind
    def see(self, scene: Optional[dict]) -> None:
        self.pending.append(("see", scene or {"things": [], "event": None, "pointed": None}))

    def heard(self, words: List[str], speaker: str) -> None:
        self.pending.append(("heard", (tuple(words), speaker)))


# ----------------------------------------------------------------------------------------------------- reading
def _unit(mind, name: str, kind: str, symbol, create: bool = True) -> int:
    if not mind.ch.has(name):
        if not create:
            return -1
        mind.add_channel(name, kind)
    return mind._atom(mind.ch.id(name), symbol, create)


def _e(mind, tie: int) -> float:
    if tie < 0:
        return 0.5
    p, n = mind.S.mass1(tie, mind.trust)
    return rule(p, n, mind.cfg.k)


def _out(mind, unit: int, form: int, lag: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ties, targets, expectations) of a unit's living ties of a form and lag, read at once."""
    S = mind.S
    lst = S.out.get((unit, form, lag))
    if not lst:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0)
    ids = np.frombuffer(lst, dtype=np.int32).astype(np.int64)
    v = S._np()
    ids = ids[v.alive[ids]]
    if not len(ids):
        return ids, ids, np.zeros(0)
    p, n = S.mass2(torch.from_numpy(ids), mind.trust)
    p, n = p.numpy(), n.numpy()
    w = p + n
    k = mind.cfg.k
    e = np.where(w > 0, (w / (w + k)) * (np.divide(p, w, out=np.full_like(w, 0.5), where=w > 0) - 0.5) + 0.5, 0.5)
    return ids, v.dst[ids].astype(np.int64), e


def _channel(mind, unit: int) -> int:
    return int(mind.S._np().channel[unit])


def _features(mind, unit: int) -> Dict[int, float]:
    """A unit's own perceived features: {feature unit: e} (sensed ties above 1/2)."""
    _, dst, e = _out(mind, unit, SAHIP, 0)
    return {int(f): float(x) for f, x in zip(dst, e) if x > 0.5}


def _by_dimension(feats: Dict[int, float], mind) -> Dict[int, Tuple[int, float]]:
    """The value of each dimension (channel) a unit has: {channel: (feature unit, e)}."""
    out = {}
    for f, e in feats.items():
        ch = _channel(mind, f)
        if ch not in out or e > out[ch][1]:
            out[ch] = (f, e)
    return out


def _word_profile(mind, z: Zemin, w: int) -> Dict[int, Dict[int, float]]:
    """What a word names: {channel: {feature unit: e}} (its ADLANDIRIR ties, read once per state of the store)."""
    key = ("w", w, mind.S.version)
    hit = z.memo.get(key)
    if hit is not None:
        return hit
    _, dst, e = _out(mind, w, ADLANDIRIR, 0)
    out: Dict[int, Dict[int, float]] = {}
    v = mind.S._np()
    for f, x in zip(dst, e):
        out.setdefault(int(v.channel[f]), {})[int(f)] = float(x)
    if len(z.memo) > 4096:
        z.memo.clear()
    z.memo[key] = out
    return out


def likelihood(mind, z: Zemin, w: int, feats: Dict[int, float]) -> Tuple[float, int]:
    """How well word w's meaning predicts a unit with these features: (sum over the unit's dimensions of
    ln P(value | w), the number of dimensions). P(value | w) is the word's expectation of that value (the one rule)
    where it has one, else what its strongest expectation on that dimension leaves, else 1/2 (it expects nothing)."""
    prof = _word_profile(mind, z, w)
    total, n = 0.0, 0
    for ch, (f, a) in _by_dimension(feats, mind).items():
        n += 1
        values = prof.get(ch)
        if not values:
            total += math.log(0.5)
            continue
        e = values.get(f)
        if e is None:
            e = 1.0 - max(values.values())
        total += math.log(min(1.0, max(e, FLOOR)))
    return total, n


def mean_likelihood(mind, z: Zemin, w: int, feats: Dict[int, float]) -> float:
    """The geometric mean of P(value | w) over the unit's dimensions (1/2: the word says nothing of it)."""
    total, n = likelihood(mind, z, w, feats)
    return math.exp(total / n) if n else 0.5


def speaking_line(mind) -> float:
    """e* = 1/2 + theta / (2T): the expectation the mind speaks at (the race reaches the threshold in T steps)."""
    return 0.5 + mind.cfg.theta / (2.0 * mind.cfg.T)


def contradicts(mind, z: Zemin, w: int, feats: Dict[int, float]) -> bool:
    """Whether the word expects, at least as firmly as the mind speaks, another value of one of the unit's
    dimensions than the one it has: such a word does not name it ("elmaya" expects a stem; a ball has none)."""
    prof = _word_profile(mind, z, w)
    line = speaking_line(mind)
    ident = mind.ch.id(IDENTITY) if mind.ch.has(IDENTITY) else -1
    for ch, (f, a) in _by_dimension(feats, mind).items():
        values = prof.get(ch)
        if not values or ch == ident:                # (an identity is no property of a kind: a kind word heard
            continue                                 # twice with one thing expects it for a while)
        v, e = max(values.items(), key=lambda ve: ve[1])
        if e >= line and v != f and values.get(f, 0.0) < line:
            return True
    return False


def left_open(mind, z: Zemin, head: int, feats: Dict[int, float]) -> Dict[int, float]:
    """The features of a unit on the dimensions the head word does not expect a value of as firmly as the mind
    speaks: what a modifier tells apart ("ata" leaves the colour open; "kahverengi" is read there)."""
    prof = _word_profile(mind, z, head)
    line = speaking_line(mind)
    out = {}
    for f, a in feats.items():
        values = prof.get(_channel(mind, f))
        if not values or max(values.values()) < line:
            out[f] = a
    return out


def fit(mind, z: Zemin, w: int, feats: Dict[int, float]) -> float:
    """How much better than knowing nothing the word predicts the unit: the mean over its dimensions of
    ln P(value | word) - ln 1/2."""
    total, n = likelihood(mind, z, w, feats)
    return (total / n - math.log(0.5)) if n else 0.0


# ----------------------------------------------------------------------------------------------------- writing
def _add(mind, table: int, slot: int, tie: int, amount: float) -> None:
    from .learn import add_evidence1
    if amount > 0.0:
        add_evidence1(mind, table, slot, tie, amount)


def _perceive(mind, unit: int, feature: int, amount: float = 1.0) -> None:
    """Bind a perceived feature to a unit; a value expected on the same dimension (channel) is contradicted."""
    S = mind.S
    _add(mind, W_POS, SENSED, S.tie(unit, feature, SAHIP, 0), amount)
    ch = _channel(mind, feature)
    ids, dst, e = _out(mind, unit, SAHIP, 0)
    v = S._np()
    for tie, g, x in zip(ids, dst, e):
        if g != feature and x > 0.5 and v.channel[g] == ch:
            _add(mind, W_NEG, SENSED, int(tie), amount * float(x))


def live(mind) -> None:
    """Live what was perceived and heard since the last moment (called inside ``Mind.step``)."""
    z = mind.zemin
    todo, z.pending = z.pending, []
    for what, x in todo:
        if what == "see":
            _see(mind, z, x)
        elif what == "heard":
            _heard(mind, z, *x)


def _see(mind, z: Zemin, scene: dict) -> None:
    z.present = {}
    for t in scene.get("things", []):
        u = _unit(mind, THING, "sense", t["index"])
        for dim, val in sorted(t["feats"].items()):
            _perceive(mind, u, _unit(mind, FEAT + dim, "facet", val))
        _perceive(mind, u, _unit(mind, IDENTITY, "facet", t["index"]))
        z.present[t["index"]] = u
    z.pointed = scene.get("pointed")
    ev = scene.get("event")
    if ev is None:
        z.event = None
        return
    key = (tuple(sorted(ev["action"].items())), tuple((p["index"], tuple(sorted(p["part"].items())))
                                                      for p in ev["parts"]))
    if z.event is not None and z.event["key"] == key:
        return                                       # the same event goes on
    z.serial += 1
    eu = _unit(mind, EVENT, "sense", z.serial)
    action = []
    for dim, val in sorted(ev["action"].items()):
        f = _unit(mind, FEAT + dim, "facet", val)
        _perceive(mind, eu, f)
        action.append(f)
    parts = {}
    for p in ev["parts"]:
        pu = _unit(mind, PART, "sense", (z.serial, p["index"]))
        for code, on in sorted(p["part"].items()):
            _perceive(mind, pu, _unit(mind, PFEAT + code, "facet", on))
        parts[p["index"]] = pu
    z.event = {"key": key, "unit": eu, "action": action, "parts": parts, "serial": z.serial}
    if z.waiting is not None:                        # what was said just before: it asked for this, or foretold it
        words, speaker, present, pointed = z.waiting
        z.waiting = None
        _align(mind, z, words, speaker, BEFORE, present, pointed)


def _heard(mind, z: Zemin, words: Tuple[str, ...], speaker: str) -> None:
    if z.waiting is not None:                        # what waited met no event: understood with what was in view
        w, s, present, pointed = z.waiting
        z.waiting = None
        _align(mind, z, w, s, None, present, pointed)
    if z.event is not None:
        _align(mind, z, words, speaker, DURING, z.present, z.pointed)
    else:
        z.waiting = (words, speaker, dict(z.present), z.pointed)


def _pair(mind, z: Zemin, zeta: str) -> Tuple[int, Dict[int, float]]:
    """The event as talked about at a time relation (a unit with the event's action features and the time)."""
    ev = z.event
    u = _unit(mind, PAIR, "sense", (ev["serial"], zeta))
    t = _unit(mind, TIME, "context", zeta)
    if not mind.S.out.get((u, SAHIP, 0)):
        for f in ev["action"]:
            _add(mind, W_POS, SENSED, mind.S.tie(u, f, SAHIP, 0), 1.0)
        _add(mind, W_POS, SENSED, mind.S.tie(u, t, SAHIP, 0), 1.0)
    return u, _features(mind, u)


def _align(mind, z: Zemin, words, speaker, zeta, present, pointed) -> None:
    """An utterance met what it was about: its words name, its form becomes a construction, its turn follows."""
    S = mind.S
    wc = mind.ch.id(z.words) if mind.ch.has(z.words) else mind.add_channel(z.words, "act")
    wunits = [mind._atom(wc, w) for w in words]
    if not wunits:
        return
    ev = z.event if zeta is not None else None
    if ev is not None:
        about = {i: u for i, u in present.items() if i in ev["parts"]}
    elif pointed is not None and pointed in present:
        about = {pointed: present[pointed]}
    else:
        about = dict(present)
    talked: List[Tuple[int, Dict[int, float]]] = [(u, _features(mind, u)) for u in about.values()]   # then the event
    pair = None
    if ev is not None:
        pair = _pair(mind, z, zeta)
        talked.append(pair)
    share = _name(mind, z, wunits, talked)
    _construction(mind, z, wunits, about, ev, zeta, talked, share, speaker)


def _align_words(mind, z: Zemin, wunits: List[int], talked) -> List[List[float]]:
    """The share of each referent (and of nothing, the last column) each distinct word refers to: the product of
    P(referent | word) (among the referents and nothing) and P(word | referent) (among the words), each read from
    how much better than knowing nothing the word's meaning so far predicts the referent: the likelihood ratio
    over its dimensions, prod P(value | word) / (1/2) (nothing is 1: the word predicts nothing)."""
    U = list(dict.fromkeys(wunits))
    g = []
    for w in U:
        row = []
        for _, feats in talked:
            total, n = likelihood(mind, z, w, feats)
            row.append(math.exp(max(-60.0, min(60.0, total - n * math.log(0.5)))))
        g.append(row + [1.0])
    rows = [[x / sum(r) for x in r] for r in g]                          # P(referent | word)
    n = len(talked)
    cols = [sum(g[i][j] for i in range(len(U))) for j in range(n)]
    out = []
    for i in range(len(U)):
        out.append([rows[i][j] * (g[i][j] / cols[j]) for j in range(n)] + [rows[i][n]])
    for j in range(n):                                                   # each referent's share sums to one
        total = sum(out[i][j] for i in range(len(U)))
        for i in range(len(U)):
            out[i][j] = out[i][j] / total if total > 0 else 0.0
    return out


def _name(mind, z: Zemin, wunits: List[int], talked) -> List[List[float]]:
    """Each word names the features of what it refers to, by its share of it; where it expected another value of a
    dimension of its referent it gets evidence against that value. Returns the alignment."""
    S = mind.S
    share = _align_words(mind, z, wunits, talked)
    U = list(dict.fromkeys(wunits))
    v = S._np()
    for i, w in enumerate(U):
        ids, dst, e = _out(mind, w, ADLANDIRIR, 0)
        expected = [(int(t), int(g), float(x)) for t, g, x in zip(ids, dst, e) if x > 0.5]
        for j, (unit, feats) in enumerate(talked):
            a = share[i][j]
            if a <= 0.0:
                continue
            by_dim = _by_dimension(feats, mind)
            for ch, (f, b) in by_dim.items():
                _add(mind, W_POS, IMITATED, S.tie(w, f, ADLANDIRIR, 0), a * hat(b))
            for tie, g, x in expected:
                ch = int(v.channel[g])
                if ch in by_dim and by_dim[ch][0] != g:
                    _add(mind, W_NEG, IMITATED, tie, a * x * hat(by_dim[ch][1]))
    return share


def _role_profile(mind, z: Zemin, kappa: int, place: int) -> Dict[int, Dict[int, float]]:
    """The participation profile a construction's place holds: {channel: {feature: e}}."""
    key = ("r", kappa, place, mind.S.version)
    hit = z.memo.get(key)
    if hit is not None:
        return hit
    _, dst, e = _out(mind, kappa, YUVA, place)
    out: Dict[int, Dict[int, float]] = {}
    v = mind.S._np()
    for f, x in zip(dst, e):
        out.setdefault(int(v.channel[f]), {})[int(f)] = float(x)
    z.memo[key] = out
    return out


def _role_fit(mind, profile: Dict[int, Dict[int, float]], feats: Dict[int, float]) -> float:
    """How well a participation fits a place: the sum over its dimensions of ln P(value | place)."""
    total = 0.0
    for ch, (f, a) in _by_dimension(feats, mind).items():
        values = profile.get(ch)
        if not values:
            continue
        e = values.get(f)
        if e is None:
            e = 1.0 - max(values.values())
        total += math.log(max(e, FLOOR))
    return total


def _places(mind, z: Zemin, wunits, about: Dict[int, int], talked, share) -> Tuple[list, Dict[int, int]]:
    """The kind of each place of an utterance, by what its word refers to (the alignment): the head of a participant
    (the word with the greatest share of it), a modifier (a further word whose referent is that participant), the
    action (the word with the greatest share of the event), or a fixed word (it refers to nothing). Returns (places,
    {place: tracked index of the participant it heads})."""
    U = list(dict.fromkeys(wunits))
    index_of = list(about.keys())                     # talked[j] for j < len(about) are the things, then the event
    n = len(talked)
    best = {}
    for i, w in enumerate(U):
        j = max(range(n + 1), key=lambda c: share[i][c])
        best[w] = j
    places: list = list(wunits)
    heads: Dict[int, int] = {}
    head_word: Dict[int, int] = {}
    for j in range(n):
        cands = [(share[i][j], i) for i, w in enumerate(U) if best[w] == j]
        if not cands:
            continue
        w = U[max(cands)[1]]
        head_word[j] = w
    taken = set()
    for jj, w in enumerate(wunits):
        j = best[w]
        if j == n:
            continue                                  # refers to nothing: a fixed word
        if head_word.get(j) == w and w not in taken:
            taken.add(w)
            if j < len(index_of):
                places[jj] = HEAD
                heads[jj] = index_of[j]
            else:
                places[jj] = ACTION
    for jj, w in enumerate(wunits):                   # modifiers: further words referring to a participant
        j = best[w]
        if j < len(index_of) and places[jj] == w and head_word.get(j) not in (None, w):
            head_place = next((p for p, i in heads.items() if i == index_of[j]), None)
            if head_place is not None:
                places[jj] = ("M", head_place)
    return places, heads


def _key(places) -> tuple:
    return tuple(p if isinstance(p, (str, tuple)) else ("w", int(p)) for p in places)


def _construction(mind, z: Zemin, wunits, about, ev, zeta, talked, share, speaker) -> None:
    S = mind.S
    places, heads = _places(mind, z, wunits, about, talked, share)
    kappa = _unit(mind, CONS, "context", _key(places))
    if ev is not None:
        for j, i in heads.items():                   # the place holds the participation profile of what filled it
            pu = ev["parts"].get(i)
            if pu is None:
                continue
            for f, e in _features(mind, pu).items():
                _add(mind, W_POS, IMITATED, S.tie(kappa, f, YUVA, j), hat(e))
    act = next((wunits[j] for j, p in enumerate(places) if p == ACTION), None)
    island = _unit(mind, ISLAND, "context", (kappa, act if act is not None else -1))
    for j, p in enumerate(places):                   # the words lived in each place, with this action word
        if not isinstance(p, int):
            _add(mind, W_POS, IMITATED, S.tie(island, wunits[j], DOLGU, j), 1.0)
    when = zeta if zeta is not None else "-"
    t = _unit(mind, TIME, "context", when)
    _add(mind, W_POS, IMITATED, S.tie(kappa, t, ADLANDIRIR, 0), 1.0)
    turn = _unit(mind, TURN, "context", (kappa, when))            # the construction at its time relation
    island_turn = _unit(mind, TURN, "context", (island, when))    # ... with its action word (a verb island)
    said_turn = _unit(mind, TURN, "context", (tuple(wunits), when))   # ... the very words (what was lived)
    if z.turn is not None and z.turn[1] != speaker and speaker == z.answerer:   # what was said back to what
        for asked in z.turn[0]:
            _add(mind, W_POS, IMITATED, S.tie(asked, turn, EXPECT, 1), 1.0)
    z.turn = ((said_turn, island_turn, turn), speaker)


# ----------------------------------------------------------------------------------------------------- answering
def _event_dimensions(mind, z: Zemin) -> set:
    """The dimensions (channels) events are perceived on: those of the features of the events lived."""
    key = ("evdims", mind.S.n_units)
    hit = z.memo.get(key)
    if hit is not None:
        return hit
    out = set()
    if mind.ch.has(EVENT):
        c = mind.ch.id(EVENT)
        v = mind.S._np()
        for i in range(len(mind.ch.symbols[c])):
            u = mind.S.atom_index.get((c, i), -1)
            if u >= 0:
                for f in _out(mind, u, SAHIP, 0)[1]:
                    out.add(int(v.channel[f]))
                if len(out) and i > 32:              # the dimensions of events are met in the first events
                    break
    z.memo[key] = out
    return out


def _constructions(mind, n: int) -> List[Tuple[int, tuple]]:
    if not mind.ch.has(CONS):
        return []
    c = mind.ch.id(CONS)
    out = []
    for i, key in enumerate(mind.ch.symbols[c]):
        if len(key) == n:
            u = mind.S.atom_index.get((c, i), -1)
            if u >= 0:
                out.append((u, key))
    return out


def understand(mind, z: Zemin, words: List[str]) -> Optional[dict]:
    """The construction whose form the words fit and the referents in view its places name, best first: (score,
    construction, its key, {place: tracked index}, the action word or None)."""
    wc = mind.ch.id(z.words)
    wunits = [mind._atom(wc, w, create=False) for w in words]
    best = None
    feats = {i: _features(mind, u) for i, u in z.present.items()}
    for kappa, key in _constructions(mind, len(words)):
        ok, score, heads, act = True, 0.0, {}, None
        for j, p in enumerate(key):
            if isinstance(p, tuple) and p[0] == "w":
                if wunits[j] != p[1]:
                    ok = False
                    break
            elif p == ACTION:
                if wunits[j] < 0:
                    ok = False
                    break
                act = wunits[j]
        if not ok:
            continue
        used = set()
        for j, p in enumerate(key):
            if p != HEAD:
                continue
            if wunits[j] < 0:
                ok = False
                break
            mods = [m for m, q in enumerate(key) if isinstance(q, tuple) and q[0] == "M" and q[1] == j]
            def named_by(f, head=wunits[j], mods=mods):
                rest = left_open(mind, z, head, f)       # a modifier is read on what its head leaves open
                return fit(mind, z, head, f) + sum(fit(mind, z, wunits[m], rest) for m in mods if wunits[m] >= 0)

            def allowed(f, head=wunits[j], mods=mods):
                rest = left_open(mind, z, head, f)
                return not contradicts(mind, z, head, f) and \
                    not any(contradicts(mind, z, wunits[m], rest) for m in mods if wunits[m] >= 0)
            ranked = sorted(((named_by(f), i) for i, f in feats.items() if i not in used and allowed(f)),
                            reverse=True)
            if not ranked or ranked[0][0] <= 0.0:
                ok = False
                break
            score += ranked[0][0]
            heads[j] = ranked[0][1]
            used.add(ranked[0][1])
        if not ok:
            continue
        uses = sum(_e(mind, int(t)) for t in (mind.S.out.get((kappa, ADLANDIRIR, 0)) or []))
        cand = (score + math.log(1.0 + uses), kappa, key, heads, act)
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        return None
    return {"score": best[0], "kappa": best[1], "key": best[2], "heads": best[3], "action": best[4],
            "words": wunits}


def _names_it(mind, z: Zemin, w: int, thing: int, feats: Dict[int, float]) -> int:
    """1 when the word names this very thing (it expects its identity: a proper name), else 0."""
    ident = mind.ch.id(IDENTITY) if mind.ch.has(IDENTITY) else -1
    values = _word_profile(mind, z, w).get(ident, {})
    own = next((f for f in feats if _channel(mind, f) == ident), None)
    return int(own is not None and values.get(own, 0.0) > 0.5)


def _answer_turns(mind, q_turn: int) -> List[Tuple[float, int, tuple, str]]:
    """What followed this turn from another speaker, most expected first (over 1/2): (e, construction unit, its key,
    time relation)."""
    ids, dst, e = _out(mind, q_turn, EXPECT, 1)
    out = []
    v = mind.S._np()
    for turn, x in zip(dst, e):
        if x <= 0.5:
            continue
        kappa, zeta = mind.ch.symbol(int(v.channel[turn]), int(v.symbol[turn]))
        key = mind.ch.symbol(int(v.channel[kappa]), int(v.symbol[kappa]))
        out.append((float(x), int(kappa), key, zeta))
    out.sort(key=lambda t: -t[0])
    return out


def _fillers(mind, kappa: int, act: Optional[int], place: int) -> List[int]:
    """The words lived in a place of a construction: with this action word when it has any, else with any."""
    out = []
    if act is not None:
        island = mind.unit_of(ISLAND, (kappa, act))
        if island >= 0:
            out = [int(w) for w in _out(mind, island, DOLGU, place)[1]]
    if out:
        return out
    if not mind.ch.has(ISLAND):
        return []
    c = mind.ch.id(ISLAND)
    for i, (k, _) in enumerate(mind.ch.symbols[c]):
        if k == kappa:
            u = mind.S.atom_index.get((c, i), -1)
            if u >= 0:
                out += [int(w) for w in _out(mind, u, DOLGU, place)[1]]
    return list(dict.fromkeys(out))


def answer(mind, z: Zemin, words: List[str], when: Optional[str] = None) -> Optional[List[str]]:
    """What the mind says back, from the meaning: None when it does not understand or has nothing to say. ``when``:
    the time relation of the question to the event that goes on (``once``: it was heard before the event began; by
    default, during it when one goes on).

    The constructions that answered this question's turn (with its action word first, else its construction) are
    tried, most expected first; the first whose every place can be filled from what the answer tells of is said."""
    q = understand(mind, z, words)
    if q is None:
        return None
    ev = z.event
    if when is not None:                             # when the question was heard, against the event
        zeta_q = when
    else:
        zeta_q = DURING if ev is not None else (BEFORE if q["action"] is not None else "-")
    tried, seen = [], set()                          # what was said back: to these very words, else to this
    island = mind.unit_of(ISLAND, (q["kappa"], q["action"] if q["action"] is not None else -1))   # construction with
    said = tuple(q["words"]) if all(w >= 0 for w in q["words"]) else None                          # this action word,
    for key in ((said, zeta_q), (island, zeta_q), (q["kappa"], zeta_q)):                          # else to it
        if key[0] is None or key[0] == -1:
            continue
        q_turn = mind.unit_of(TURN, key)
        if q_turn < 0:
            continue
        for cand in _answer_turns(mind, q_turn):
            if (cand[1], cand[3]) not in seen:
                seen.add((cand[1], cand[3]))
                tried.append(cand)
    if not tried:
        return None
    # what the answer tells of: the event going on, else the one the question asked for
    if ev is not None:
        parts = {i: _features(mind, pu) for i, pu in ev["parts"].items()}
        action = {f: 1.0 for f in ev["action"]}
    elif q["action"] is not None:
        parts = {}
        for j, i in q["heads"].items():              # each participant with the profile of its place in the question
            parts[i] = {}
            for ch, values in _role_profile(mind, z, q["kappa"], j).items():
                f, e = max(values.items(), key=lambda fe: fe[1])
                parts[i][f] = e
        action = {}                                  # what the action word names of an event (the dimensions
        kinds = _event_dimensions(mind, z)           # events are perceived on), the time left to the answer
        time_ch = _channel(mind, _unit(mind, TIME, "context", BEFORE))
        for ch, values in _word_profile(mind, z, q["action"]).items():
            f, e = max(values.items(), key=lambda fe: fe[1])
            if e > 0.5 and ch in kinds and ch != time_ch:
                action[f] = e
    else:
        return None
    for _, kappa_a, key_a, zeta_a in tried:
        out = _fill(mind, z, q, kappa_a, key_a, zeta_a, parts, action)
        if out is not None:
            wc = mind.ch.id(z.words)
            return [str(mind.ch.symbol(wc, int(mind.S._np().symbol[w]))) for w in out]
    return None


def _fill(mind, z: Zemin, q: dict, kappa_a: int, key_a: tuple, zeta_a: str, parts, action) -> Optional[List[int]]:
    """The words of an answer construction filled from what it tells of: the action place with the word that names
    the event at the answer's time relation best, every participant's place with the participant whose participation
    fits the place's profile best and the word that names it best among those lived there (and the question's word
    for it), every modifier as the question told the thing apart (else the word that names it best). None when a
    place cannot be filled (nothing there names what it should)."""
    named = {i: q["words"][j] for j, i in q["heads"].items()}
    out: List[Optional[int]] = [None] * len(key_a)
    act_word = None
    t = _unit(mind, TIME, "context", zeta_a, create=False) if zeta_a != "-" else -1
    for j, p in enumerate(key_a):                    # the action first (the places' words are kept per action word)
        if p == ACTION:
            feats = dict(action)
            if t >= 0:
                feats[t] = 1.0
            cands = _fillers(mind, kappa_a, None, j)
            ranked = sorted(((fit(mind, z, w, feats), w) for w in cands), reverse=True)
            if not ranked or ranked[0][0] <= 0.0:
                return None
            act_word = ranked[0][1]
            out[j] = act_word
    used, head_of = set(), {}
    for j, p in enumerate(key_a):
        if isinstance(p, tuple) and p[0] == "w":
            out[j] = p[1]
        elif p == HEAD:
            prof = _role_profile(mind, z, kappa_a, j)
            ranked = sorted(((_role_fit(mind, prof, f), i) for i, f in parts.items() if i not in used), reverse=True)
            if not ranked:
                return None
            i = ranked[0][1]
            used.add(i)
            head_of[j] = i
            thing = z.present.get(i)
            if thing is None:
                return None
            tf = _features(mind, thing)
            cands = _fillers(mind, kappa_a, act_word, j)   # the words lived in this place, and the word the question
            if i in named and named[i] not in cands:       # called it by: the one that names it best ("ali" before
                cands = cands + [named[i]]                 # "kim", which names any doer)
            ranked = sorted(((_names_it(mind, z, w, thing, tf), fit(mind, z, w, tf), w) for w in cands
                             if not contradicts(mind, z, w, tf)), reverse=True)
            if not ranked or ranked[0][1] <= 0.0:
                return None
            out[j] = ranked[0][2]
    qmods = {q["heads"].get(qp[1]): q["words"][m] for m, qp in enumerate(q["key"])
             if isinstance(qp, tuple) and qp[0] == "M"}
    for j, p in enumerate(key_a):                    # a modifier tells its head's thing apart, as the question did
        if isinstance(p, tuple) and p[0] == "M":
            i = head_of.get(p[1])
            if i is None:
                return None
            if i in qmods:
                out[j] = qmods[i]
                continue
            thing = z.present.get(i)
            if thing is None:
                return None
            tf = left_open(mind, z, out[p[1]], _features(mind, thing))   # what the head's word leaves open
            cands = _fillers(mind, kappa_a, act_word, j)
            ranked = sorted(((fit(mind, z, w, tf), w) for w in cands if not contradicts(mind, z, w, tf)),
                            reverse=True)
            if not ranked or ranked[0][0] <= 0.0:
                return None
            out[j] = ranked[0][1]
    if any(w is None for w in out):
        return None
    return out
