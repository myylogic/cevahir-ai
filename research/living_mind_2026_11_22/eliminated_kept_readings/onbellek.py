# -*- coding: utf-8 -*-
"""Kept readings (Saklanan Okumalar): a reading that only reads the mind's substance is the same as long as nothing
it read has changed. Exact: the mind decides, learns and says what it would without this, bit for bit; only the
time it takes changes.

Why. With everything on, two readings are made again and again from nothing: what the classes of a word go on with
(F37, read once a run of items) and the vote of every frame of a family (F53, read once for every context). On a
mind raised on running text a family holds hundreds of frames, and one answer took thirty seconds where the
speaking configuration takes a third of one. Between two moments almost nothing those readings rest on changes.

How it is kept exact.

* **What a reading read.** While a kept reading is made, the store's indices are read through recorders
  (``_Index``): every list of ties that was looked up is noted (the ties out of a unit of one form and lag; the
  ties into a unit of one form). Evidence is only ever read of ties reached through such a list, a composed unit
  or a frame that does not exist yet would add a tie into each of its parts when it is made (noted at the part
  the fewest units hold), so a change of evidence, a new tie or a new unit is a change of a noted list.
* **What changed.** The store stamps the two lists a tie is in whenever its evidence changes or it is made, with
  its own version counter, where the list was noted (``Substance._changed``). A kept reading holds as long as no
  list it noted was stamped after it was made (``asof``). (The first version stamped units: every use of a
  frequent word then undid every reading that had looked at any list of it, and almost nothing held.)
* **Before the present run.** Many readings leave out what was born in the present run of items
  (``episode_start``). A reading made in one run is used in another only when no list it noted was stamped since
  the run it was made in began (``Substance._rev_begin``): then nothing it could have left out exists.
* **Read once a run.** F37's readings are kept for a run as they were when first read in it, whatever is written
  later in the run (``read._episode_memo``). That is kept as it is: the run's memo (``RunMemo``) is asked first;
  what it does not hold is taken from what is kept only when it still holds now, and is then the run's for the rest
  of the run. A kept reading that used such readings holds only while each of them does, and when it is used
  again they are put in the run's memo with it, as making it again would have put them there (``_replay``); if
  the run already holds another reading of one of them, the kept one is not used.
* **What is not in the store.** The configuration, the trust in each provenance, the knobs, the relations the mind
  built and the channels are part of the key (``_guard``): readings made under one are not used under another.
  The lived words that share a beginning with a word (the lexicon F37 reads) are compared as they are
  (``head``). A reading that touched anything else that can change without a stamp is not kept (``bind``), and
  nothing is kept while what is recalled depends on present company (F33).
* An undone or retracted moment drops everything kept; a copy of the mind starts with nothing kept.

``ZIHIN_KEPT_OFF=1`` makes every reading afresh (the lives are compared with and without, digest for digest).
``ZIHIN_KEPT_VERIFY=1`` makes every reading afresh *and* compares it with what is kept where that holds; a
difference is an error.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Optional

import numpy as np

OFF = bool(os.environ.get("ZIHIN_KEPT_OFF"))
VERIFY = bool(os.environ.get("ZIHIN_KEPT_VERIFY"))
LIMIT = int(os.environ.get("ZIHIN_KEPT_UNITS", "24000000"))     # noted units kept in all (8 bytes each)
PARTS = 32                                                     # the guards kept beside each other (the trials read with each way off)
FAILS: list = []                                               # (key, kept, fresh) where VERIFY found a difference
SINCE_AT = {"mates": 2, "inheritance": 2, "class_profile": 3, "said_of": 2,    # where a run reading's key names
            "class_goes": None, "forms": None, "goes_to": None}               # the run it was read in
_NONE = np.zeros(0, dtype=np.int64)


class Entry:
    """A kept reading: its value, the units it noted itself, the readings it used, and when it was made."""
    __slots__ = ("value", "units", "asof", "since", "heads", "inner", "key", "seen", "size")

    def __init__(self, value, units, asof, since, heads, inner, key) -> None:
        self.value, self.units, self.asof, self.since = value, units, asof, since
        self.heads, self.inner, self.key = heads, inner, key
        self.seen = (asof, since)
        self.size = int(units.shape[0]) + len(inner) + 8


class _Session:
    __slots__ = ("tag", "keys", "heads", "inner", "bound", "old", "check")

    def __init__(self, tag, old) -> None:
        self.tag, self.old = tag, old
        self.keys: set = set()                                 # the lists it noted (``Substance.key_out`` / ``key_in``)
        self.heads: dict = {}
        self.inner: dict = {}
        self.bound = False
        self.check = None


class _Index:
    """A recorder in front of one of the store's indices: the same answers, and the lists asked for noted."""
    __slots__ = ("real", "st", "kind")

    def __init__(self, real, st, kind: int) -> None:
        self.real, self.st, self.kind = real, st, kind

    def _note(self, key, found) -> None:
        st = self.st
        if st.paused:
            return
        top, kind = st.stack[-1], self.kind
        if kind == 0:                                          # the ties into a unit, of one form
            top.keys.add((((key[0] << 6) | key[1]) << 1) | 1)
        elif kind == 4:                                        # the ties out of a unit, of one form and lag
            top.keys.add((((key[0] << 6) | key[1]) << 6 | (key[2] + 2)) << 1)
        elif found is None:
            if kind == 3:                                      # an atom that does not exist (yet)
                top.bound = "no such atom"
                return
            parts = key if kind == 1 else [p for p in key[1] if p >= 0]   # a composed unit / a frame not made yet:
            if parts:                                          # making it adds a PART tie into each of its parts
                inc = st.saved[0]
                rare = min(parts, key=lambda p: len(inc.get((p, _PART), ())))
                top.keys.add((((rare << 6) | _PART) << 1) | 1)

    def get(self, key, default=None):
        found = self.real.get(key)
        self._note(key, found)
        return default if found is None else found

    def __getitem__(self, key):
        found = self.real[key]
        self._note(key, found)
        return found

    def __contains__(self, key) -> bool:
        found = self.real.get(key)
        self._note(key, found)
        return found is not None

    def __setitem__(self, key, value) -> None:
        self.real[key] = value

    def __delitem__(self, key) -> None:
        del self.real[key]

    def __len__(self) -> int:
        return len(self.real)

    def __iter__(self):
        if not self.st.paused:
            self.st.stack[-1].bound = "whole index"            # read as a whole: it rests on everything
        return iter(self.real)

    def __getattr__(self, name):
        if name in ("items", "keys", "values") and not self.st.paused:
            self.st.stack[-1].bound = "whole index"
        return getattr(self.real, name)


class _State:
    def __init__(self, S) -> None:
        self.S = S
        self.stack: list = []
        self.parts: Dict[tuple, dict] = {}
        self.saved = None
        self.paused = 0
        self.total = 0
        self.guard = None                                      # (the configuration's values, the guard made of them)
        self.hits = self.made = self.stale = 0
        self.why: Dict[tuple, int] = {}                        # (the kind of reading, why it was not kept) -> times


_INDICES = (("inc", 0), ("out", 4), ("tuple_index", 1), ("frame_of", 2), ("frame_first", 2), ("atom_index", 3))
_PART = 1                                                      # (store.PART)


def state(mind) -> Optional[_State]:
    """What is kept for this mind's store (made on first use), or None when nothing is to be kept."""
    if OFF or mind.__dict__.get("_kept_off"):
        return None
    if getattr(mind.cfg, "company", False) and "personal" in mind.__dict__:
        return None                                            # F33: what is recalled depends on who is present
    S = mind.S
    st = S.__dict__.get("_kept")
    if st is None:
        st = S.__dict__["_kept"] = _State(S)
    return st


def _guard(mind, st: _State) -> tuple:
    values = tuple(mind.cfg.__dict__.values())
    trust = mind.trust.numpy().tobytes()
    knobs = tuple(mind.knob.values())
    relations = getattr(mind, "relations", None)
    held = st.guard
    key = (trust, knobs, len(relations) if relations else 0, len(mind.ch.names))
    if held is not None and held[0] == values and held[1] == key and held[2] == relations:
        return held[3]
    guard = (repr(mind.cfg.__dict__), trust, knobs, repr(relations) if relations else "", len(mind.ch.names))
    st.guard = (values, key, list(relations) if relations else relations, guard)
    return guard


def _part(st: _State, guard: tuple) -> dict:
    part = st.parts.get(guard)
    if part is None:
        while len(st.parts) >= PARTS:                          # the guard least lately used goes
            old = st.parts.pop(next(iter(st.parts)))
            st.total -= sum(e.size for e in old.values())
        part = st.parts[guard] = {}
    else:
        st.parts[guard] = st.parts.pop(guard)                  # (the last used last)
    return part


def _push(st: _State, mind, tag) -> _Session:
    if not st.stack:
        S = st.S
        st.saved = tuple(getattr(S, name) for name, _ in _INDICES)
        for (name, kind), real in zip(_INDICES, st.saved):
            setattr(S, name, _Index(real, st, kind))
        find = S.tie_index.find

        def find_tie(src: int, dst: int, form: int, lag: int) -> int:
            if not st.paused:                                  # (the tie is, or would be, in this list)
                st.stack[-1].keys.add((((src << 6) | form) << 6 | ((-1 if form == _PART else lag) + 2)) << 1)
            return find(src, dst, form, lag)
        S.find_tie = find_tie
    sess = _Session(tag, mind.__dict__.get("_read_memo"))
    mind.__dict__["_read_memo"] = {}                           # nothing read before is taken unnoted
    st.stack.append(sess)
    return sess


def _pop(st: _State, mind) -> _Session:
    sess = st.stack.pop()
    mind.__dict__["_read_memo"] = sess.old
    if not st.stack:
        S = st.S
        for (name, _), real in zip(_INDICES, st.saved):
            setattr(S, name, real)
        S.__dict__.pop("find_tie", None)
        st.saved = None
    return sess


def reset(mind) -> None:
    """Nothing is being read any more (a new moment begins): no recorder stays (one may, when a reading was
    interrupted)."""
    st = mind.S.__dict__.get("_kept")
    if st is not None and st.stack:
        while st.stack:
            _pop(st, mind)
        st.paused = 0


class paused:
    """While this lasts nothing is noted (a reading guarded another way: the lexicon, by ``head``)."""

    def __init__(self, mind) -> None:
        self.st = mind.S.__dict__.get("_kept")

    def __enter__(self):
        if self.st is not None:
            self.st.paused += 1

    def __exit__(self, *exc) -> None:
        if self.st is not None:
            self.st.paused -= 1


def _entry(st: _State, sess: _Session, value, since: int, key) -> Entry:
    key_id = st.S.key_id
    units = np.fromiter((key_id(k) for k in sess.keys), dtype=np.int64, count=len(sess.keys)) if sess.keys else _NONE
    inner = tuple(sess.inner.values())
    asof = st.S.version
    for e in inner:
        if e.asof < asof:
            asof = e.asof
    return Entry(value, units, asof, since, tuple(sess.heads.items()), inner, key)


def _into(parent: _Session, sess: _Session, entry: Optional[Entry]) -> None:
    """What a reading rested on, its parent rests on too."""
    if entry is not None:
        parent.inner[id(entry)] = entry
    else:
        parent.keys |= sess.keys
        parent.inner.update(sess.inner)
        parent.heads.update(sess.heads)
    if sess.bound:
        parent.bound = sess.bound


def _holds(st: _State, mind, e: Entry, since: int) -> bool:
    """Whether a kept reading is what would be read now (see the module's text)."""
    S = st.S
    now = (S.version, since)
    if e.seen == now:
        return True
    for inner in e.inner:
        if not _holds(st, mind, inner, since):
            return False
    limit = e.asof
    if e.since != since:
        began = S.__dict__.get("_rev_begin", {}).get(e.since)
        if began is None:
            return False
        if began < limit:
            limit = began
    if e.units.shape[0] and int(S._kst[e.units].max()) > limit:
        return False
    if e.heads:
        from .read import head_units
        for (cid, head), units in e.heads:
            if head_units(mind, cid, head, since) != units:
                return False
    e.asof, e.since, e.seen = S.version, since, now            # as good as read now
    return True


def _run_key(key, since: int):
    at = SINCE_AT.get(key[0], -1)
    if at == -1:
        return None
    return key if at is None else key[:at] + key[at + 1:]


def _key_now(xkey, since: int):
    at = SINCE_AT[xkey[0]]
    return xkey if at is None else xkey[:at] + (since,) + xkey[at:]


def _replay(mind, e: Entry, since: int, check: bool = True) -> bool:
    """The run readings a kept reading used are the run's now, as making it again would have made them; False (and
    nothing done) when the run already holds another reading of one of them."""
    held = mind.__dict__.get("_episode_memo")
    memo = held[1] if held is not None and held[0] == since else None
    if memo is None:
        from .read import _episode_memo
        memo = _episode_memo(mind)
    if not isinstance(memo, RunMemo):
        return False
    if id(e) in memo.done:
        return True
    if check and not _free(memo, e, since, set()):
        return False
    _place(memo, e, since)
    return True


def _free(memo: "RunMemo", e: Entry, since: int, seen: set) -> bool:
    if id(e) in memo.done or id(e) in seen:
        return True
    seen.add(id(e))
    if e.key is not None:
        cur = memo.meta.get(_key_now(e.key, since))
        if cur is not None and cur is not e:
            return False
        if cur is e:
            return True
    return all(_free(memo, inner, since, seen) for inner in e.inner)


def _place(memo: "RunMemo", e: Entry, since: int) -> None:
    if id(e) in memo.done:
        return
    memo.done.add(id(e))
    if e.key is not None:
        key = _key_now(e.key, since)
        if key in memo.meta:
            return
        dict.__setitem__(memo, key, e.value)
        memo.meta[key] = e
    for inner in e.inner:
        _place(memo, inner, since)


def _store(st: _State, part: dict, key, e: Entry) -> None:
    old = part.pop(key, None)
    if old is not None:
        st.total -= old.size
    part[key] = e
    st.total += e.size
    while st.total > LIMIT and part:                           # what was kept first goes first
        st.total -= part.pop(next(iter(part))).size


def same(a, b) -> bool:
    """Whether two readings are the same, value for value (arrays element for element)."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and bool((a == b).all())
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(v, b[k]) for k, v in a.items())
    return a == b


def _differs(key, kept_value, fresh) -> None:
    FAILS.append((key, kept_value, fresh))
    raise AssertionError("a kept reading differs from the reading made now: %r" % (key,))


def kept(mind, key, compute: Callable):
    """``compute()``, a reading that only reads: what was kept of it while that still holds, else made now and kept."""
    st = state(mind)
    if st is None:
        return compute()
    since = getattr(mind, "episode_start", mind.t)
    st.paused += 1                                             # (asking what is kept is no part of any reading)
    try:
        part = _part(st, _guard(mind, st))
        e = part.get(key)
        usable = e is not None and _holds(st, mind, e, since)
        ready = usable and not VERIFY and _replay(mind, e, since)
    finally:
        st.paused -= 1
    if ready:
        st.hits += 1
        if st.stack:
            st.stack[-1].inner[id(e)] = e
        return e.value
    if e is not None and not usable:
        st.stale += 1
    depth = len(st.stack)
    sess = _push(st, mind, key)
    try:
        value = compute()
    finally:
        while len(st.stack) > depth + 1:                       # (a run reading that began and did not end)
            inner = _pop(st, mind)
            _into(st.stack[-1], inner, None)
        _pop(st, mind)
    st.made += 1
    fresh = None
    if not sess.bound:
        fresh = _entry(st, sess, value, since, None)
        _store(st, part, key, fresh)
    else:
        st.why[(key[0], sess.bound)] = st.why.get((key[0], sess.bound), 0) + 1
    if VERIFY and usable and not same(e.value, value):
        _differs(key, e.value, value)
    if st.stack:
        _into(st.stack[-1], sess, fresh)
    return value


def head(mind, cid: int, text: str, units: tuple) -> None:
    """A reading used the lived words of a channel that begin as ``text`` does: it holds while they are these."""
    st = mind.S.__dict__.get("_kept")
    if st is not None and st.stack:
        st.stack[-1].heads[(cid, text)] = units


def bind(mind) -> None:
    """A reading used something that can change without a trace in the store: it is not kept."""
    st = mind.S.__dict__.get("_kept")
    if st is not None and st.stack:
        st.stack[-1].bound = "bind"


class RunMemo(dict):
    """The memo of a run of items (``read._episode_memo``): what it holds is the run's, as before. What it does not
    hold is taken from what is kept when that still holds; what is read for it is kept."""

    def __init__(self, mind, since: int) -> None:
        super().__init__()
        self.mind, self.since = mind, since
        self.meta: Dict[tuple, Entry] = {}
        self.done: set = set()

    def _used(self, key) -> None:
        st = self.mind.S.__dict__.get("_kept")
        if st is not None and st.stack:
            e = self.meta.get(key)
            if e is not None:
                st.stack[-1].inner[id(e)] = e
            else:
                st.stack[-1].bound = "run reading without a record"                      # (read before anything was kept: only the run's)

    def _miss(self, key) -> bool:
        mind = self.mind
        st = state(mind)
        if st is None:
            return False
        xkey = _run_key(key, self.since)
        check = None
        if xkey is not None:
            st.paused += 1                                     # (asking what is kept is no part of any reading)
            try:
                e = _part(st, _guard(mind, st)).get(xkey)
                held = e is not None and _holds(st, mind, e, self.since)
            finally:
                st.paused -= 1
            if held:
                if VERIFY:
                    check = e
                elif _free(self, e, self.since, set()):
                    _place(self, e, self.since)
                    st.hits += 1
                    if st.stack:
                        st.stack[-1].inner[id(e)] = e
                    return True
            elif e is not None:
                st.stale += 1
        sess = _push(st, mind, key)
        sess.check = check
        if xkey is None:
            sess.bound = "not a run reading that is kept"
        return False

    def __contains__(self, key) -> bool:
        if dict.__contains__(self, key):
            self._used(key)
            return True
        return self._miss(key)

    def get(self, key, default=None):
        if dict.__contains__(self, key):
            self._used(key)
            return dict.__getitem__(self, key)
        if self._miss(key):
            return dict.__getitem__(self, key)
        return default

    def __setitem__(self, key, value) -> None:
        dict.__setitem__(self, key, value)
        mind = self.mind
        st = mind.S.__dict__.get("_kept")
        if st is None or not any(s.tag == key for s in st.stack):
            return
        while st.stack[-1].tag != key:                         # (a reading inside it that began and did not end)
            inner = _pop(st, mind)
            _into(st.stack[-1], inner, None)
        sess = _pop(st, mind)
        st.made += 1
        entry = None
        xkey = _run_key(key, self.since)
        if not sess.bound and xkey is not None:
            entry = _entry(st, sess, value, self.since, xkey)
            self.meta[key] = entry
            self.done.add(id(entry))
            _store(st, _part(st, _guard(mind, st)), xkey, entry)
        elif sess.bound:
            st.why[(key[0], sess.bound)] = st.why.get((key[0], sess.bound), 0) + 1
        if sess.check is not None and not same(sess.check.value, value):
            _differs(key, sess.check.value, value)
        if st.stack:
            _into(st.stack[-1], sess, entry)


def run_memo(mind, since: int):
    """The memo of a new run of items."""
    return {} if OFF or mind.__dict__.get("_kept_off") else RunMemo(mind, since)


def counts(mind) -> dict:
    st = mind.S.__dict__.get("_kept")
    if st is None:
        return {}
    return {"used": st.hits, "made": st.made, "no longer held": st.stale, "kept": sum(len(p) for p in st.parts.values()),
            "noted units": st.total, "not kept": dict(st.why)}
