# -*- coding: utf-8 -*-
"""F53, Frames of Frames (``Config.families``): what the mind has generalised is itself generalised.

A frame generalises lived contexts that differ in one item (its place). Frames that differ from each other in one
fixed word, and are the same otherwise, are a **family**: "X ne yer ?", "X ne içer ?", "X ne sever ?" are one
family whose varying word is the verb. A family is a generalisation with two places, the frames' own place and
the place of the word that varies. Nothing new is stored for it: it is read from the frames that are there.

A context no frame speaks for, but which a family fits (its item at the varying place is one the family never
had there), is read by the family: every frame of it votes once, with the conviction of its members, on the *kind*
of what its members went on with:

* a relation of the frame's value to its own filler (the filler itself, its continuation, how its sequences began,
  what it was answered with, what stands where it stands), named for the context's own filler;
* the frame's own varying word (``SECOND``), named as the context's item at that place;
* ``LINK``: a thing known of the filler whose own lived sequence goes on to the frame's varying word; for the
  context, the thing known of its filler that goes on to *its* item at that place. This is the relation between
  the two places: of all that is known of the thing asked about, what the question's other word leads to;
* a word every frame's members said.

A family claims only what its frames agree on (a vote over 1/2). A word that some frames' members said and others
did not is each frame's own (the value that goes with its own word): the family says nothing for or against it.
A place holds items of one kind: the item at the varying place is of the channel the family's words there are
of, and in each frame only the members that are items of the filler's channel vote. (What the mind feels while it
speaks is an item of its context too, an inner event: it stood where a word then stood, and without this the
family read the feeling as that word. Found in F50's world: 70 of 80 answers went wrong.)

    vote(kind) = sum over frames of c_frame f_frame(kind) / sum over frames of c_frame,
    e = c vote + (1 - c) 1/2,   c = lam e_new conviction(sum of c_frame)

with e_new what one lived instance gives a filler it meets for the first time. So how firmly a new kind of
question is read depends on how many kinds of question the mind has already mastered: what it learned about
learning carries over, and the more it knows the faster it learns.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import torch

from .events import ACTED, IMAGINED
from .evidence import conviction, expectation, frequency
from .form import mass1, mv_view, parts_of
from .store import ABSTRACT, CATEGORY, COMPOSED, SLOTTED

LINK = -8      # "the thing known of the filler whose own lived sequence goes on to the item at the second place"
SECOND = -9    # "the item at the second place itself"


def _distinct(parts) -> Dict[int, Tuple[int, ...]]:
    """The values of a tuple with their positions, in order of first position."""
    where: Dict[int, List[int]] = {}
    for j, p in enumerate(parts):
        where.setdefault(p, []).append(j)
    return {p: tuple(js) for p, js in where.items()}


def _key(parts, first: Tuple[int, ...], second: Tuple[int, ...]) -> tuple:
    return first, second, tuple(-1 if j in first else (-2 if j in second else p) for j, p in enumerate(parts))


def _index(mind) -> Dict[tuple, List[Tuple[int, int]]]:
    """Every frame under the keys of the families it could belong to: (its place's positions, the positions of one
    of its fixed words, its parts with both left out) -> [(the frame, that word)]. Built as frames are made, read
    only when a family is looked for; not kept when the mind is saved or copied."""
    S = mind.S
    kept = mind.__dict__.get("_aileler")
    if kept is None:
        kept = mind.__dict__["_aileler"] = [0, {}]
    frames = S.frame_of
    if kept[0] != len(frames):
        kinds = mv_view(mind, "u_kind")
        for frame in itertools.islice(frames.values(), kept[0], None):
            parts = parts_of(mind, frame)
            first = tuple(j for j, p in enumerate(parts) if kinds[p] & CATEGORY)
            if not first:
                continue
            for word, second in _distinct(parts).items():
                if not kinds[word] & CATEGORY:
                    kept[1].setdefault(_key(parts, first, second), []).append((frame, word))
        kept[0] = len(frames)
    return kept[1]


def families(mind, unit: int) -> List[tuple]:
    """The families that fit a context: [(the filler's positions, the varying word's positions, [(frame, its word
    there), ...])], each with at least two frames that have another word there than the context has."""
    kinds = mv_view(mind, "u_kind")
    if not kinds[unit] & COMPOSED or kinds[unit] & SLOTTED:
        return []
    parts = parts_of(mind, unit)
    index = _index(mind)
    if not index:
        return []
    places = _distinct(parts)
    channels = mv_view(mind, "u_channel")
    out = []
    for filler, first in places.items():
        for word, second in places.items():
            if word == filler:
                continue
            found = index.get(_key(parts, first, second))
            if not found:
                continue                                   # the item at the varying place is of the channel the
            others = [(frame, w) for frame, w in found     # frames' words there are of
                      if w != word and channels[w] == channels[word]]
            if len({w for _, w in others}) >= 2:
                out.append((first, second, others))
    return out


def _not_self(mind, trust: Optional[torch.Tensor]) -> torch.Tensor:
    own = (mind.trust if trust is None else trust).clone()
    own[ACTED] = 0.0
    own[IMAGINED] = 0.0
    return own


def _goes_on_to(mind, sequence: Tuple[int, ...], word: int, trust: torch.Tensor) -> bool:
    """Whether a lived sequence of its own (a unit made where a run of items began) went on with ``word``."""
    from .read import _world_pos
    unit = mind.S.tuple_index.get(sequence)
    if unit is None or mv_view(mind, "u_kind")[unit] & SLOTTED:
        return False
    return word in _world_pos(mind, unit, trust)[0]


def linked_share(mind, frame: int, word: int, trust: Optional[torch.Tensor] = None,
                 channel: Optional[int] = None) -> float:
    """How far a frame's members went on with a thing known of them whose own lived sequence then goes on to
    ``word`` (the frame's word at the family's varying place): the share of their lived values for which the
    sequence (the member from its last place, the value) was lived as a sequence of its own and went on with that
    word. Each member weighs as in the frame's own vote."""
    from .read import _world_pos
    S, k = mind.S, mind.cfg.k
    kinds = mv_view(mind, "u_kind")
    parts = parts_of(mind, frame)
    first = tuple(j for j, p in enumerate(parts) if kinds[p] & CATEGORY)
    if not first:
        return 0.0
    category = parts[first[0]]
    src = mv_view(mind, "t_src")
    own = _not_self(mind, trust)
    linked = members = 0.0
    channels = mv_view(mind, "u_channel")
    for tie in S.inc.get((category, ABSTRACT), ()):
        x = src[tie]
        if channel is not None and channels[x] != channel:
            continue
        lived = tuple(x if j in first else p for j, p in enumerate(parts))
        instance = S.tuple_index.get(lived)
        if instance is None:
            continue
        targets, masses = _world_pos(mind, instance, trust)
        total = sum(masses)
        if total <= 0.0:
            continue
        e_x = expectation(*mass1(mind, tie), k)
        members += e_x
        for v, p in zip(targets, masses):
            if _goes_on_to(mind, lived[first[-1]:] + (v,), word, own):
                linked += e_x * p / total
    return linked / members if members > 0.0 else 0.0


def link(mind, unit: int, first: Tuple[int, ...], second_item: int, trust: Optional[torch.Tensor] = None):
    """``LINK`` named for a context: of the things known of its filler and not said yet (``read.known_unsaid``), the
    one whose own lived sequence goes on to the item at the second place; the one with the largest share, or None
    when there is none or two tie."""
    from .read import known_unsaid
    parts = parts_of(mind, unit)
    filler = parts[first[0]]
    known = known_unsaid(mind, filler, unit, first[-1], trust, episode=unit == mind.context)
    own = _not_self(mind, trust)
    found = sorted(((share, v) for v, share in known.items()
                    if _goes_on_to(mind, parts[first[-1]:] + (v,), second_item, own)), reverse=True)
    if not found or (len(found) > 1 and found[1][0] >= found[0][0] - 1e-12):
        return None
    return found[0][1]


def votes(mind, unit: int, trust: Optional[torch.Tensor] = None):
    """The family that speaks for a context, as (c, {kind: vote}, the filler's positions, the varying word's
    positions, the frame with the most members), or None. Of several families the one that agrees most speaks
    (conviction x its strongest vote), as among frames."""
    from .read import member_votes
    cfg = mind.cfg
    best = None
    parts = parts_of(mind, unit) if mv_view(mind, "u_kind")[unit] & COMPOSED else ()
    channels = mv_view(mind, "u_channel")
    for first, second, frames in families(mind, unit):
        num: Dict[int, float] = {}
        den = 0.0
        widest = None
        channel = channels[parts[first[0]]]                # the members of the filler's kind of item vote
        for frame, word in frames:
            mass, members, _, _, _ = member_votes(mind, frame, None, trust, channel=channel)
            if not mass:
                continue
            c_frame = conviction(members, cfg.k)
            den += c_frame
            for key, (w_pos, w_neg) in mass.items():
                kind = SECOND if key == word else key
                num[kind] = num.get(kind, 0.0) + c_frame * frequency(w_pos, w_neg)
            share = linked_share(mind, frame, word, trust, channel)
            if share > 0.0:
                num[LINK] = num.get(LINK, 0.0) + c_frame * share
            if widest is None or members > widest[0]:
                widest = (members, frame)
        if den <= 0.0:
            continue
        c_family = conviction(den, cfg.k)
        vote = {kind: w / den for kind, w in num.items()}
        score = c_family * max(vote.values())
        if best is None or score > best[0]:
            best = (score, c_family, vote, first, second, widest[1])
    if best is None:
        return None
    g = mind.knob["gain"]
    e_new = expectation(g, 0.0, cfg.k)                     # a filler met for the first time, after one instance
    return (cfg.lam * e_new * best[1],) + best[2:]


def claims(mind, unit: int, trust: Optional[torch.Tensor] = None):
    """What the family that fits a context claims there, each kind named for the context's own filler and its item
    at the second place: ({target: (e, kind)} with the firmest kind that names each target, (the filler, the widest
    frame's category, [(kind, e, whether it named nothing), ...] for the relations of the filler's own record, the
    thing known of the filler that LINK names or None, {target: [(e, kind), ...]} with every kind that names it)),
    or ({}, None). Read once a moment."""
    from .read import ALIKE, ASKED, INHERIT, KNOWN, NEXT, REPLY, SELF, START, _moment_memo, asked_place, \
        frame_category, resolve
    memo = _moment_memo(mind)
    key = ("aile", unit, id(trust) if trust is not None else None)
    if key in memo:
        return memo[key]
    found = votes(mind, unit, trust)
    if found is None:
        memo[key] = ({}, None)
        return memo[key]
    c, vote, first, second, frame = found
    parts = parts_of(mind, unit)
    filler, second_item = parts[first[0]], parts[second[0]]
    fixed = tuple(p for j, p in enumerate(parts) if j not in first)
    category = frame_category(mind, frame)
    out: Dict[int, Tuple[float, int]] = {}
    every: Dict[int, List[Tuple[float, int]]] = {}
    own = []
    known = None
    for kind, share in vote.items():
        if kind == KNOWN or share <= 0.5:                  # (KNOWN: a set, chosen by what the members said: LINK
            continue                                       # stands for it.) A family claims what its frames agree
        e = c * share + (1.0 - c) * 0.5                    # on; a word that only some of them say is each frame's
        if kind == LINK:                                   # own, and the family says nothing for or against it
            target = known = link(mind, unit, first, second_item, trust)
        elif kind == SECOND:
            target = second_item
        elif kind == SELF:
            target = filler
        elif kind == ASKED:                                # F55: read against the context's own run
            target = asked_place(mind, filler, unit, first[0], trust)
        elif kind < 0:
            target = resolve(mind, kind, filler, fixed, category)
        else:
            target = kind
        if kind in (NEXT, START, ALIKE, REPLY, INHERIT, LINK, ASKED):
            own.append((kind, e, target is None))
        if target is None:
            continue
        every.setdefault(target, []).append((e, kind))
        if target not in out or e > out[target][0]:
            out[target] = (e, kind)
    memo[key] = (out, (filler, category, own, known, every))
    return memo[key]


def speak(mind, unit: int, trust: Optional[torch.Tensor], out: Dict[int, float], src) -> None:
    """The family's claims for a context no frame speaks for, written into a reading (``read.members_map``). F46: a
    relation of the filler's own record that names nothing here is an open claim about the filler. F49: every kind
    of family claim is a way of reading with a record of its own, and a word two kinds name is claimed by both. F41:
    a thing known of the filler that is claimed (LINK, whichever other kind names it too) makes it the answer's
    topic."""
    from .read import _moment_memo, _note_claim
    found, about = claims(mind, unit, trust)
    if about is None:
        return
    cfg = mind.cfg
    filler, category, own, known, every = about
    if getattr(cfg, "find_out", False):
        _moment_memo(mind)[("filler", unit)] = filler
    if getattr(cfg, "open_claims", False) and trust is None \
            and (unit == mind.context or mind.__dict__.get("_claim_again") == unit):
        for _, e, nothing in own:
            _note_claim(mind, filler, e, nothing)
    for target, made in every.items():
        for e0, kind in made:
            write(mind, out, src, target, e0, kind)
    if known is not None and getattr(cfg, "known", False):
        _moment_memo(mind)[("known_claim", unit)] = (filler, category, known)


def write(mind, out: Dict[int, float], src, target: int, e0: float, kind: int) -> None:
    """One family claim into a reading; F49: read with what family claims of that kind have come to."""
    e = e0
    if src is not None:
        from . import hata
        ways = hata.family_ways(kind)
        e = hata.believed(mind, ways, e0)
        made = src.setdefault(target, [])
        made.append((ways, e0, e))
        made.sort(key=lambda claim: -claim[2])
    out[target] = max(out.get(target, 0.0), e)
