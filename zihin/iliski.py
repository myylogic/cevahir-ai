# -*- coding: utf-8 -*-
"""F33 Relational Memory: one memory, recalled in the company of the person present.

The mind lives every experience into its one memory, as before: the evidence of a tie is the whole life. While a
person it knows is present (``mind.company``, set by the session: the partner perceived with every utterance, the one
who asks in a watched dialogue; the person as the session perceives them, an opaque identity), every increment of evidence on what followed what (EXPECT ties of lag 1 or more) and
on how a sequence began (PART ties, F26) is also noted with that person: a ledger of whom each piece of evidence was
lived with. Learning reads and writes the ties exactly as without it; nothing else is stored.

What is recalled depends on who is present (``company_view``, read wherever a unit's own evidence is read):

* what was lived with no one in particular (reading, watching people it does not know, its own thinking) is recalled
  with anyone;
* what was lived with the person present is recalled with them, all of it;
* what was lived with other people is recalled only as far as they corroborate one another. Each of them is one voice
  about the unit: their share of its positive evidence per target, as a frame's members are counted (F11). One
  person's voice alone is theirs; every further person who lived it so corroborates it, with conviction
  lam * conviction(n - 1) for n voices (F30's principle that independent evidence adds up, the first person's evidence
  not being evidence about anyone else). The corroborated frequency enters as evidence, w = k c / (1 - c) split by f.

So a mind that lived a secret with one person tells it to no one else, and what every person it lives with taught it
(its name, how it is) is known with a stranger too. No person, role or relation is named in the core: a person is a
unit perceived like any other, and how it differs to live with one or another is only what was lived with them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .events import REFERENCE, SENSED
from .evidence import conviction, expectation, frequency
from .store import EXPECT, PART, W_NEG, W_POS

LEDGER_KINDS = (EXPECT, PART)


def _ledger(mind):
    """(by tie: {person: [2 tables x P classes] raw amounts}, by (source, kind, lag): set of ties)."""
    led = mind.__dict__.get("personal")
    if led is None:
        led = mind.__dict__["personal"] = ({}, {})
    return led


def _noted(a, tie: int) -> bool:
    """Evidence on what followed what (EXPECT, lag 1 or more) and on how a sequence began (PART of a two-item unit)."""
    kind = a.form[tie]
    return (kind == EXPECT and a.lag[tie] >= 1) or (kind == PART and a.arity[a.src[tie]] == 2)


def note(mind, table: int, slot: int, ties, amounts) -> None:
    """Note increments of evidence with the person present (called by the store's writers in learn.py)."""
    person = mind.company
    by_tie, by_unit = _ledger(mind)
    a = mind.S.arrays()
    P = mind.S.P
    moment = mind.__dict__.setdefault("_personal_moment", [])
    if slot == SENSED:                                   # the world perceived directly is no one's in particular
        return
    for tie, amount in zip(ties, amounts):
        if amount == 0.0 or not _noted(a, tie):
            continue
        rows = by_tie.setdefault(tie, {})
        row = rows.get(person)
        if row is None:
            row = rows[person] = np.zeros((2, P))
            by_unit.setdefault((int(a.src[tie]), int(a.form[tie]), int(a.lag[tie])), set()).add(tie)
        row[table, slot] += amount
        moment.append((tie, person, table, slot, amount))


def end_moment(mind) -> None:
    """Keep this moment's notes for undo and retraction (the ledger is undone with the moment)."""
    moment = mind.__dict__.pop("_personal_moment", None)
    log = mind.__dict__.setdefault("personal_log", {})
    if moment:
        log[mind.t] = moment
    keep = getattr(mind.cfg, "undo_depth", 1)
    if not mind.S.keep_log:                              # without the experience ledger only undo needs them
        for t in [t for t in log if t <= mind.t - max(1, keep)]:
            del log[t]


def forget_moment(mind, m: int) -> None:
    """Remove what moment m noted (its increments were removed from the store), and what it judged of each person."""
    log = mind.__dict__.get("personal_log")
    if not log or m not in log:
        return
    by_tie, _ = _ledger(mind)
    records = mind.__dict__.get("person_trust", {})
    for entry in log.pop(m):
        if entry[0] == "trust":
            _, person, slot, agree, differ = entry
            rec = records.get(person)
            if rec is not None:
                rec[0, slot] = max(0.0, rec[0, slot] - agree)
                rec[1, slot] = max(0.0, rec[1, slot] - differ)
            continue
        tie, person, table, slot, amount = entry
        row = by_tie.get(tie, {}).get(person)
        if row is not None:
            row[table, slot] = max(0.0, row[table, slot] - amount)


def attribute(mind, slot: int, judged) -> None:
    """Trust learned per person: direct experience judged evidence of a learned class on some ties (``judged``:
    (tie, agree, differ, the class's whole evidence on the tie)); each person's share of that evidence, as noted,
    is how much of the verdict is theirs. A person's record (agree, differ per class) is read by ``person_trust``."""
    by_tie, _ = _ledger(mind)
    records = mind.__dict__.setdefault("person_trust", {})
    moment = mind.__dict__.setdefault("_personal_moment", [])
    P = mind.S.P
    for tie, agree, differ, held in judged:
        rows = by_tie.get(tie)
        if not rows or held <= 0.0 or (agree == 0.0 and differ == 0.0):
            continue
        for person, row in rows.items():
            mine = row[W_POS, slot] + row[W_NEG, slot]
            if mine <= 0.0:
                continue
            part = min(1.0, mine / held)
            rec = records.get(person)
            if rec is None:
                rec = records[person] = np.zeros((2, P))
            rec[0, slot] += agree * part
            rec[1, slot] += differ * part
            moment.append(("trust", person, slot, agree * part, differ * part))


def person_trust(mind, person, trust) -> np.ndarray:
    """The trust in what was lived with ``person``, per class: for a learned class, the person's own record of what
    held nests over the trust in the class (as own evidence comes last: e <- c_p f_p + (1 - c_p) e_class); direct
    experience stays 1. With no record it is the class trust."""
    t = (mind.trust if trust is None else trust).numpy().copy()
    rec = mind.__dict__.get("person_trust", {}).get(person)
    if rec is None:
        return t
    k = mind.cfg.k
    for c in range(len(t)):
        if c in REFERENCE:
            continue
        a, d = float(rec[0, c]), float(rec[1, c])
        if a + d > 0.0:
            c_p = conviction(a + d, k)
            t[c] = c_p * frequency(a, d) + (1.0 - c_p) * t[c]
    return t


def _trusted(row: np.ndarray, trust: np.ndarray) -> Tuple[float, float]:
    return float(row[W_POS] @ trust), float(row[W_NEG] @ trust)


def company_view(mind, unit: int, kind: int, lag: int, ties, w_pos, w_neg, trust) -> Optional[Tuple[list, list]]:
    """The unit's own evidence (``ties`` in order, trust-weighted ``w_pos``, ``w_neg``) as recalled in present company,
    or None when nothing of it was lived with anyone in particular:

    1. what was lived with no one in particular: the evidence without every person's noted share;
    2. corroborated by the others: each other person is one voice, their share of the unit's positive evidence among
       the targets of each channel (one item per channel per moment); n >= 2 voices give
       c = lam conviction(n - 1), w = k c / (1 - c), split by the mean share f;
    3. the person present last: their own evidence nests over the rest, as an instance's own evidence comes last
       (e <- c_own f_own + (1 - c_own) e), held as evidence of conviction 1 - (1 - c_own)(1 - c).
    """
    by_tie, by_unit = _ledger(mind)
    noted = by_unit.get((unit, kind, lag))
    if not noted:
        return None
    present = mind.company
    k = mind.cfg.k
    t = (mind.trust if trust is None else trust).numpy()
    held = {}                                            # each person's word, weighed by their own record
    w_pos, w_neg = [float(x) for x in w_pos], [float(x) for x in w_neg]
    own: Dict[int, Tuple[float, float]] = {}
    voices: Dict[object, Dict[int, float]] = {}
    for i, tie in enumerate(ties):
        rows = by_tie.get(tie) if tie in noted else None
        if not rows:
            continue
        for person, row in rows.items():
            p, n = _trusted(row, t)
            w_pos[i] = max(0.0, w_pos[i] - p)            # 1. no one's in particular
            w_neg[i] = max(0.0, w_neg[i] - n)
            if person not in held:
                held[person] = person_trust(mind, person, trust)
            p, n = _trusted(row, held[person])
            if person == present:
                own[i] = (p, n)
            elif p > 0.0:
                voices.setdefault(person, {})[i] = p
    if len(voices) >= 2:                                 # 2. the others corroborate one another
        a = mind.S.arrays()
        channel_of = {i: int(a.channel[a.dst[ties[i]]]) for voice in voices.values() for i in voice}
        c = mind.cfg.lam * conviction(float(len(voices) - 1), k)
        w = k * c / (1.0 - c)
        shares: Dict[int, float] = {}
        for voice in voices.values():
            totals: Dict[int, float] = {}
            for i, p in voice.items():
                totals[channel_of[i]] = totals.get(channel_of[i], 0.0) + p
            for i, p in voice.items():
                shares[i] = shares.get(i, 0.0) + p / totals[channel_of[i]]
        for i, share in shares.items():
            f = share / len(voices)
            w_pos[i] += f * w
            w_neg[i] += (1.0 - f) * w
    for i, (p, n) in own.items():                        # 3. the person present, last
        if p + n <= 0.0:
            continue
        c_rest = conviction(w_pos[i] + w_neg[i], k)
        e_rest = expectation(w_pos[i], w_neg[i], k)
        c_own = conviction(p + n, k)
        e = c_own * frequency(p, n) + (1.0 - c_own) * e_rest
        c_all = min(1.0 - 1e-12, 1.0 - (1.0 - c_own) * (1.0 - c_rest))
        f = min(1.0, max(0.0, (e - 0.5) / c_all + 0.5))
        total = k * c_all / (1.0 - c_all)
        w_pos[i], w_neg[i] = f * total, (1.0 - f) * total
    return w_pos, w_neg


def company_mass1(mind, tie: int, w_pos: float, w_neg: float, trust) -> Tuple[float, float]:
    """One tie's trust-weighted evidence without what was lived with people other than the one present (F26's
    beginnings; corroboration among people is read on expectations, ``company_view``)."""
    by_tie, _ = _ledger(mind)
    rows = by_tie.get(tie)
    if not rows:
        return w_pos, w_neg
    t = (mind.trust if trust is None else trust).numpy()
    for person, row in rows.items():
        if person != mind.company:
            p, n = _trusted(row, t)
            w_pos, w_neg = max(0.0, w_pos - p), max(0.0, w_neg - n)
    return w_pos, w_neg


def persons_with(mind, unit: int) -> List[int]:
    """The people something about ``unit`` was lived with (for introspection and tests)."""
    by_tie, by_unit = _ledger(mind)
    found = set()
    for (src, _, _), ties in by_unit.items():
        if src == unit:
            for tie in ties:
                found |= set(by_tie.get(tie, {}))
    return sorted(found)
