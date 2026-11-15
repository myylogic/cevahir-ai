# -*- coding: utf-8 -*-
"""Reading the substance (design sections 2.3, 4 "down", 5): effective evidence, predictions, drive.

Nothing here writes. Support from a generalisation to an instance is computed when the instance is read, so
it always reflects the present evidence of the generalisation and needs no retraction of its own.

Effective evidence of an instance tie i -> v (section 4):

    as written ("raw"):   W_eff = W_own + lam * e(i -> g) * W(g -> v)          summed over generalisations g
    changed   ("capped"): W_eff = W_own + lam * e(i -> g) * W(g -> v) / max(1, W(g -> v) / k)

In the changed form one generalisation counts at most as much as lam * e(i -> g) lived experiences, however
much evidence the generalisation has collected (see NOTES.md, finding on exceptions).

Generalisations of a unit: its categories (ABSTRACT ties) and, for a composed unit, the existing composed
units that have one of its parts replaced by a category of that part (frames).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from .evidence import conviction, expectation, frequency
from .form import categories_of
from .store import ABSTRACT, CATEGORY, COMPOSED, EXPECT, I64, PART, RELY, SLOTTED, W_NEG, W_POS


def generalisations(mind, unit: int) -> List[Tuple[int, float]]:
    """(generalisation unit, expectation that ``unit`` is its instance), only where that exceeds 1/2."""
    S = mind.S
    found = categories_of(mind, unit)
    kind = int(S.u_kind[unit])
    if kind & COMPOSED and not kind & SLOTTED:
        parts = S.parts(unit)
        for pos, part in enumerate(parts):
            for category, e in categories_of(mind, part):
                if getattr(S, "bind_repeats", False):
                    frame = S.tuple_index.get(tuple(category if q == part else q for q in parts))
                else:
                    frame = S.tuple_index.get(parts[:pos] + (category,) + parts[pos + 1:])
                if frame is not None:
                    found.append((frame, e))
    return [(general, e) for general, e in found if e > 0.5]


def own_mass(mind, unit: int, lag: int = 1, trust: Optional[torch.Tensor] = None) -> Dict[int, Tuple[float, float]]:
    """Trust-weighted (W+, W-) of the EXPECT ties leaving ``unit``, by target."""
    S = mind.S
    if not S.out.get((unit, EXPECT, lag)):
        return {}
    ids, dst = S.out_ids((unit, EXPECT, lag))
    trust = mind.trust if trust is None else trust
    pos, neg = S.w2(ids)
    if lag == 1 and getattr(mind.cfg, "rivals", False):
        w_pos, w_neg = rival_mass(mind, dst, (pos * trust).sum(-1), (neg * trust).sum(-1))
        return {d: (p, n) for d, p, n in zip(dst.tolist(), w_pos.tolist(), w_neg.tolist())}
    w_pos = (pos * trust).sum(-1).tolist()
    w_neg = (neg * trust).sum(-1).tolist()
    return {d: (p, n) for d, p, n in zip(dst.tolist(), w_pos, w_neg)}


def rival_mass(mind, dst: torch.Tensor, w_pos: torch.Tensor, w_neg: torch.Tensor):
    """F25 (b), the normalised reading: for a target on a sense or act channel the negative mass is the positive mass
    of the source's other targets on that channel, so f(v | x) = W+(x -> v) / sum over v' of W+(x -> v') and the
    conviction is that of everything x was followed by there (rivals share one mass instead of each holding its own).
    Targets on the mind's own channels keep their own negative evidence."""
    S = mind.S
    n = len(mind.ch.names)
    channel = S.u_channel[dst].long()
    channel = torch.where(channel >= 0, channel, torch.full_like(channel, n))   # no channel: a slot of its own
    world = torch.tensor([mind.ch.kind(c) in ("sense", "act") for c in range(n)] + [False], dtype=torch.bool)
    total = torch.zeros(n + 1, dtype=w_pos.dtype).index_add_(0, channel, w_pos)
    return w_pos, torch.where(world[channel], (total[channel] - w_pos).clamp(min=0.0), w_neg)


def effective(mind, unit: int, trust: Optional[torch.Tensor] = None, support: Optional[str] = None,
              skip: Optional[int] = None) -> Dict[int, Tuple[float, float]]:
    """Effective (W+, W-) of the lag-1 ties of ``unit``: own evidence plus support from its generalisations.

    ``skip`` leaves one generalisation out (used by the reality score of imagination: leave-path-out).
    """
    cfg = mind.cfg
    support = cfg.support if support is None else support
    if support in ("nested", "choice", "reliable", "trusted", "members"):                    # masses are only defined for the additive forms
        support = "capped"
    mass = own_mass(mind, unit, 1, trust)
    kind = int(mind.S.u_kind[unit])
    if support == "none" or kind & (CATEGORY | SLOTTED):
        return mass
    for general, e_abs in generalisations(mind, unit):
        if general == skip:
            continue
        for target, (w_pos, w_neg) in own_mass(mind, general, 1, trust).items():
            factor = cfg.lam * e_abs
            if support == "capped":
                factor /= max(1.0, (w_pos + w_neg) / cfg.k)
            old = mass.get(target, (0.0, 0.0))
            mass[target] = (old[0] + factor * w_pos, old[1] + factor * w_neg)
    return mass


def extension(mind, general: int) -> int:
    """How many instances a generalisation covers: the fillers of its category (of the category that stands
    in, for a frame). Smaller = more specific."""
    S = mind.S
    if int(S.u_kind[general]) & CATEGORY:
        return len(S.inc.get((general, ABSTRACT), ()))
    sizes = [len(S.inc.get((p, ABSTRACT), ())) for p in S.parts(general) if int(S.u_kind[p]) & CATEGORY]
    return min(sizes) if sizes else 0


def expect_map(mind, unit: int) -> Dict[int, float]:
    """Effective expectation of every lag-1 tie of ``unit``, by target.

    With support "raw" / "capped" / "none" this is the expectation of the effective masses (section 4 as
    written, or capped). With support "nested" (changed, see NOTES.md) the generalisations are read from the
    most general to the most specific, each one moving the estimate towards its own frequency in proportion
    to how much it has lived, and the instance's own evidence comes last:

        e <- 1/2;   for g general -> specific:  e <- c_g f_g + (1 - c_g) e,   c_g = lam e(i -> g) conviction_g
                    own evidence:               e <- c_own f_own + (1 - c_own) e

    Without any generalisation this is exactly e = c (f - 1/2) + 1/2. The form is idempotent for copies of
    the same category, lets the specific override the general, and lets own evidence override both.
    """
    cfg = mind.cfg
    if cfg.support == "members" and not int(mind.S.u_kind[unit]) & (CATEGORY | SLOTTED):
        return members_map(mind, unit)
    if cfg.support == "trusted" and not int(mind.S.u_kind[unit]) & (CATEGORY | SLOTTED):
        return trusted_map(mind, unit)
    if cfg.support == "reliable" and not int(mind.S.u_kind[unit]) & (CATEGORY | SLOTTED):
        return reliable_map(mind, unit)
    if cfg.support == "choice" and not int(mind.S.u_kind[unit]) & (CATEGORY | SLOTTED):
        return choice_map(mind, unit)
    if cfg.support != "nested" or int(mind.S.u_kind[unit]) & (CATEGORY | SLOTTED):
        return {v: expectation(p, n, cfg.k) for v, (p, n) in effective(mind, unit).items()}
    own = own_mass(mind, unit)
    gens = sorted(((extension(mind, g), g, e_abs) for g, e_abs in generalisations(mind, unit)),
                  key=lambda x: (-x[0], x[1]))
    out: Dict[int, float] = {}
    for _, general, e_abs in gens:
        for target, (w_pos, w_neg) in own_mass(mind, general).items():
            c = cfg.lam * e_abs * conviction(w_pos + w_neg, cfg.k)
            out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    for target, (w_pos, w_neg) in own.items():
        c = conviction(w_pos + w_neg, cfg.k)
        out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    return out


def reliability(mind, general: int) -> float:
    """Expectation of the RELY self-tie of a generalisation (F9); 1/2 when it has never been judged."""
    S = mind.S
    ties = S.out.get((general, RELY, 0), [])
    if not ties:
        return 0.5
    ids = torch.tensor(ties, dtype=I64)
    pos, neg = S.w2(ids)
    w_pos = float((pos * mind.trust).sum())
    w_neg = float((neg * mind.trust).sum())
    return expectation(w_pos, w_neg, mind.cfg.k)


def reliable_map(mind, unit: int) -> Dict[int, float]:
    """support = "reliable" (F9): the generalisation that has best predicted what its instances lived speaks.

        g* = argmax over generalisations with sense/act targets and reliability(g) > 1/2 of  e(i -> g) reliability(g)
        (none such: no generalisation speaks; what is not known stays at 1/2, "I do not know")
        e <- c f + (1 - c) 1/2 for every target of g*,  c = lam e(i -> g*) conviction(g* -> v)
        own evidence of the instance last:  e <- c_own f_own + (1 - c_own) e
    """
    cfg, S = mind.cfg, mind.S
    best = None
    for general, e_abs in generalisations(mind, unit):
        mass = {t: pn for t, pn in own_mass(mind, general).items()
                if int(S.u_channel[t]) >= 0 and mind.ch.kind(int(S.u_channel[t])) in ("sense", "act")}
        if not mass:
            continue
        r = reliability(mind, general)
        if mind.cfg.proven_only and r <= 0.5:              # F10b (measured worse): unproven ones have no voice
            continue
        score = e_abs * r
        if best is None or score > best[0]:
            best = (score, e_abs, general)
    out: Dict[int, float] = {}
    if best is not None:
        _, e_abs, general = best
        for target, (w_pos, w_neg) in own_mass(mind, general).items():
            c = cfg.lam * e_abs * conviction(w_pos + w_neg, cfg.k)
            out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * 0.5
    for target, (w_pos, w_neg) in own_mass(mind, unit).items():
        c = conviction(w_pos + w_neg, cfg.k)
        out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    return out


def _world(mind, unit: int) -> bool:
    channel = int(mind.S.u_channel[unit])
    return channel >= 0 and mind.ch.kind(channel) in ("sense", "act")


SELF, NEXT = -1, -2   # F12 relational targets of a frame: "the filler of my slot", "what the filler goes on with"


def continuations(mind, unit: int, exclude=()) -> Dict[int, float]:
    """Own lag-1 expectations of a unit on sense/act targets, without ``exclude`` (the fixed words of a frame)."""
    return {v: expectation(p, n, mind.cfg.k) for v, (p, n) in own_mass(mind, unit).items()
            if _world(mind, v) and v not in exclude and v != unit and p > 0.0}


def member_mass(mind, frame: int, exclude: int, trust: Optional[torch.Tensor] = None):
    """F11: what the members of a frame's category have lived in its slot, read now from their own evidence.

    For a frame whose category C fills one slot (one or, with bind_repeats, several positions), each member x of
    C (ABSTRACT x -> C with expectation e_x) whose instance (the frame with x in the slot) exists contributes its
    own belief: the share of its own positive evidence on sense/act targets that each value v received, s_x(v).
        W+(v) = sum_x e_x s_x(v),   W-(v) = sum_x e_x (1 - s_x(v))
    Every member counts once, with both signs; the instance being read is left out.
    F12 (bind_repeats): a value that IS the member itself counts under SELF, a value that is one of the member's
    own continuations (not a fixed word of the frame) counts under NEXT, so members that agree on a relation
    ("the answer repeats the asked word", "then comes what that word goes on with") agree although their words
    differ. Returns (masses by target or SELF/NEXT, members, slot positions, fixed words).
    """
    out = member_votes(mind, frame, exclude, trust)
    return out[:4]


def member_votes(mind, frame: int, exclude: int, trust: Optional[torch.Tensor] = None):
    """``member_mass`` and, fifth, how many members voted (the numerosity of the voice, F24)."""
    S, k = mind.S, mind.cfg.k
    parts = S.parts(frame)
    cats = {p for p in parts if int(S.u_kind[p]) & CATEGORY}
    if len(cats) != 1:
        return {}, 0.0, (), (), 0
    category = cats.pop()
    where = tuple(j for j, p in enumerate(parts) if p == category)
    fixed = tuple(p for p in parts if p != category)
    relational = getattr(S, "bind_repeats", False)
    shares: Dict[int, float] = {}
    members = voted = 0.0
    voters = 0
    looks = None                                       # F18: the read instance's filler and its facets
    if getattr(mind.cfg, "resemblance", False) and exclude is not None and int(S.u_kind[exclude]) & COMPOSED:
        from .learn import facet_set, facet_similarity
        filler = S.parts(exclude)[where[0]] if where and len(S.parts(exclude)) > where[0] else None
        looks = (filler, facet_set(mind, filler)) if filler is not None else None
        if looks is not None and not looks[1]:
            looks = None
    for tie in S.inc.get((category, ABSTRACT), ()):
        x = int(S.t_src[tie])
        instance = S.tuple_index.get(tuple(x if j in where else p for j, p in enumerate(parts)))
        if instance is None or instance == exclude:
            continue
        own = {v: p for v, (p, n) in own_mass(mind, instance, 1, trust).items() if _world(mind, v) and p > 0.0}
        total = sum(own.values())
        if total <= 0.0:
            continue
        e_x = expectation(float((S.w1(W_POS, tie) * mind.trust).sum()), float((S.w1(W_NEG, tie) * mind.trust).sum()), k)
        weight = e_x
        if looks is not None:                          # members that look like the item speak louder (GCM):
            sim = facet_similarity(mind, looks[0], x, looks[1])   # vote x exp(-sensitivity x (1 - similarity))
            if sim <= 0.0:
                continue
            weight = e_x * math.exp(-mind.knob["F"] * (1.0 - sim))
        members += e_x
        voted += weight
        voters += 1
        goes_on = continuations(mind, x, fixed) if relational else {}
        for v, p in own.items():
            key = SELF if relational and v == x else NEXT if v in goes_on else v
            shares[key] = shares.get(key, 0.0) + weight * p / total
    if voted <= 0.0:
        return {}, 0.0, where, fixed, 0
    scale = members / voted                            # frequencies from the weighted votes, conviction from how
    return ({v: (w * scale, members - w * scale) for v, w in shares.items()}, members, where, fixed,  # many members
            voters)


VOICE_CHANNEL = "sesler"   # F24: the mind's own channel for the kinds of voices (not "ses": the hearing facets use it)
VOICE_ALL = "*"         # F24: the unit of every voice (what voices in general have been worth)


def voice_kind(voters: int, top: float) -> Tuple[int, int]:
    """F24: the kind of a voice as the mind perceives it, in two magnitudes: how many members speak (numerosity in
    doubling bands: 1, 2, 3-4, 5-8, 9-16, ...) and how much they agree (the frequency of their top value in quarters,
    so that one half, where a prediction starts to hold, is a band edge; all agreeing is a band of its own). Nothing
    in it names a word, a class or a domain."""
    many = (max(1, voters) - 1).bit_length() + 1
    agree = 4 if top >= 1.0 else max(0, min(3, int(4.0 * top)))
    return many, agree


def _holds(mind, symbol) -> Tuple[float, float]:
    """Trust-weighted (W+, W-) of the tie (voice unit -> holds); zeros when never lived."""
    unit = mind.unit_of(VOICE_CHANNEL, symbol)
    tie = mind.S.find_tie(unit, mind.holds, EXPECT, 0) if unit >= 0 else -1
    if tie < 0:
        return 0.0, 0.0
    S = mind.S
    return float((S.w1(W_POS, tie) * mind.trust).sum()), float((S.w1(W_NEG, tie) * mind.trust).sum())


def voice_credibility(mind, kind: Tuple[int, int]) -> float:
    """F24: how far a voice of this kind can be believed, read general -> specific as everywhere (``expect_map``,
    nested): e <- e(every voice -> holds), then e <- c f + (1 - c) e with the kind's own (f, c). A kind the mind has
    never relied on is worth what voices in general have been worth; with no such experience at all, 1/2. Reads
    only; units and ties are made when the first outcome is lived (learn.judge_voices)."""
    if not mind.ch.has(VOICE_CHANNEL):
        return 0.5
    k = mind.cfg.k
    w_pos, w_neg = _holds(mind, VOICE_ALL)
    e = expectation(w_pos, w_neg, k)
    w_pos, w_neg = _holds(mind, kind)
    c = conviction(w_pos + w_neg, k)
    return c * frequency(w_pos, w_neg) + (1.0 - c) * e


def _claim(mind, unit: int, mass, where, fixed) -> Tuple[Optional[int], float]:
    """F24: what a voice claims for ``unit``: its top value (SELF and NEXT resolved as in ``members_map``) and its
    frequency f."""
    S = mind.S
    filler = S.parts(unit)[where[0]] if where and int(S.u_kind[unit]) & COMPOSED else None
    claim, claim_f = None, -1.0
    for target, (w_pos, w_neg) in mass.items():
        if target == SELF:
            target = filler
        elif target == NEXT:
            goes_on = continuations(mind, filler, fixed) if filler is not None else {}
            target = max(goes_on, key=goes_on.get) if goes_on else None
        f = frequency(w_pos, w_neg)
        if target is not None and f > claim_f:
            claim, claim_f = target, f
    return claim, claim_f


def members_map(mind, unit: int, trust: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """support = "members" (F11): a category speaks for an instance with what its members have lived.

    Among the frames of the instance, the one whose members agree most on a value speaks
    (score = e(i -> g) conviction(members) max_v f(v)); its expectations are e <- c f + (1 - c) 1/2 with
    c = lam e(i -> g) conviction(members); SELF resolves to the word in the instance's slot and NEXT to that word's
    own strongest continuation (F12); the instance's own evidence comes last, as everywhere.
    F24 (voices): each score is weighted by the credibility the mind has learned for voices of that kind (how many
    members speak, how much they agree; with no such experience every kind reads 1/2 and the choice is exactly
    F11's). For the context read with the full trust, the voice that spoke and every voice that could have spoken
    are noted with their claims (top value, frequency), so that what comes next judges them (learn.judge_voices).
    """
    cfg, S = mind.cfg, mind.S
    voices = getattr(cfg, "voices", False)
    best = None
    heard = []                                         # F24: every voice that could speak: (kind, claim, f)
    for general, e_abs in generalisations(mind, unit):
        mass, members, where, fixed, voters = member_votes(mind, general, unit, trust)
        if not mass:
            continue
        c_members = conviction(members, cfg.k)
        top = max(frequency(p, n) for p, n in mass.values())
        score = e_abs * c_members * top
        kind = None
        if voices:
            kind = voice_kind(voters, top)
            score *= voice_credibility(mind, kind)
            heard.append((kind,) + _claim(mind, unit, mass, where, fixed))
        if best is None or score > best[0]:
            best = (score, e_abs, c_members, mass, where, fixed, kind, len(heard) - 1)
    out: Dict[int, float] = {}
    if best is not None:
        _, e_abs, c_members, mass, where, fixed, kind, spoke = best
        c = cfg.lam * e_abs * c_members
        filler = S.parts(unit)[where[0]] if where and int(S.u_kind[unit]) & COMPOSED else None
        for target, (w_pos, w_neg) in mass.items():
            if target == SELF:
                target = filler
            elif target == NEXT:
                goes_on = continuations(mind, filler, fixed) if filler is not None else {}
                target = max(goes_on, key=goes_on.get) if goes_on else None
            if target is None:
                continue
            e = c * frequency(w_pos, w_neg) + (1.0 - c) * 0.5
            out[target] = max(out.get(target, 0.0), e)
        if voices and trust is None:
            mind.__dict__.setdefault("voice", {})[unit] = (heard[spoke], heard)
    for target, (w_pos, w_neg) in own_mass(mind, unit, 1, trust).items():
        c = conviction(w_pos + w_neg, cfg.k)
        out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    if cfg.backoff and not any(e > 0.5 for t, e in out.items() if _world(mind, t)):
        for target, e in continuation_backoff(mind, unit, trust).items():
            out[target] = max(out.get(target, 0.0), e)
    if getattr(cfg, "situations", False) and not any(e > 0.5 for t, e in out.items() if _world(mind, t)):
        for target, e in situation_backoff(mind, unit, trust).items():
            out[target] = max(out.get(target, 0.0), e)
    return out


def situation_backoff(mind, unit: int, trust: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """F23: a situation never lived as a whole is met with what followed the lived situations most like it.

    The situation is the present context: someone's utterance up to its perceived end (the turn end), and what the
    mind has said since, if anything. Lived contexts that end with the same part and hold the same perceived event at
    the same place from the end are compared with it part by part from the end. A part counts by how telling it is,
    1 / (1 + the number of those contexts that hold it), so a word nothing lived holds weighs most. The similarity s
    of a lived context is the weight of the parts both have at the same place over the weight of all parts either
    has; it must share a word of the utterance, not only punctuation and the end. Each such context votes its own
    evidence on what came next, weighted exp(-F (1 - s)) as exemplars are (GCM, as F18). The votes are held with
    conviction lam * s * conviction(voters), s the voters' weighted similarity: a weak resemblance (an unknown word
    in the utterance) cannot make the mind speak and leaves room for its curiosity."""
    S, cfg = mind.S, mind.cfg
    if not int(S.u_kind[unit]) & COMPOSED:
        return {}
    key = (unit, mind.t, id(trust) if trust is not None else None)
    cache = getattr(mind, "_situation_cache", None)
    if cache is not None and cache[0] == key:
        return cache[1]
    heard = S.parts(unit)
    n = len(heard)
    event = -1                                          # the last perceived event of the situation (the turn end)
    for j, p in enumerate(heard):
        channel = int(S.u_channel[p])
        if channel >= 0 and mind.ch.kind(channel) == "sense":
            event = j
    out: Dict[int, float] = {}
    if n >= 2 and event >= 1:
        d_event = n - 1 - event

        def held(p):                                     # how many lived contexts hold the part
            return len(S.inc.get((p, PART), ()))

        def weight(p):
            return 1.0 / (1 + held(p))

        def word(p):
            c = int(S.u_channel[p])
            return c >= 0 and mind.ch.kind(c) == "act" and str(mind.symbol_of(p)[1]).isalpha()

        # the lived situations are called up through the rarest of the parts they must share at the same place from
        # the end (the last part, or a word of the utterance): the contexts holding it there, ending with the same part
        # and holding the event at its place. The cost grows with how often that part was lived, not with the size of
        # the whole memory.
        anchors = sorted(((n - 1 - j, heard[j]) for j in range(event) if word(heard[j])), key=lambda x: held(x[1]))
        lived, seen, budget = [], {unit}, 20000
        for d_anchor, anchor in anchors:                 # each word of the utterance calls up its situations, the
            for tie in S.inc.get((anchor, PART), ()):     # rarest word first, within a bounded effort
                budget -= 1
                c = int(S.t_src[tie])
                if c in seen:
                    continue
                t = S.parts(c)
                L = len(t)
                if L > max(d_anchor, d_event) and int(S.t_lag[tie]) == L - 1 - d_anchor and t[-1] == heard[-1] \
                        and t[L - 1 - d_event] == heard[event]:
                    seen.add(c)
                    lived.append((t, c))
                if budget <= 0:
                    break
            if budget <= 0:
                break

        heard_set = set(heard)
        heard_weight = sum(weight(p) for p in heard_set)
        shares: Dict[int, float] = {}
        voted = similar = 0.0
        voters = 0
        for t, u in lived:
            L = len(t)
            same = [d for d in range(min(n, L)) if t[L - 1 - d] == heard[n - 1 - d]]
            if not any(word(heard[n - 1 - d]) for d in same if d > d_event):   # a word of the utterance itself
                continue
            own = {v: p for v, (p, q) in own_mass(mind, u, 1, trust).items() if _world(mind, v) and p > 0.0}
            total = sum(own.values())
            if total <= 0.0:
                continue
            shared = {heard[n - 1 - d] for d in same}
            union = heard_weight + sum(weight(p) for p in set(t) - heard_set)
            s = sum(weight(p) for p in shared) / union
            g = math.exp(-mind.knob["F"] * (1.0 - s))
            voted += g
            similar += g * s
            voters += 1
            for v, p in own.items():
                shares[v] = shares.get(v, 0.0) + g * p / total
        if voted > 0.0:
            c = cfg.lam * (similar / voted) * conviction(float(voters), cfg.k)
            out = {v: c * (w / voted) + (1.0 - c) * 0.5 for v, w in shares.items()}
    mind._situation_cache = (key, out)
    return out


def continuation_backoff(mind, unit: int, trust: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """F16: starting needs a reason, continuing follows the words. When nothing in the whole present situation
    expects a next symbol, the end of the sequence that is going on now (the items after the last perceived event,
    e.g. the mind's own utterance after its partner's ended) is read instead: its longest suffix that has lived
    continuations speaks, with its own evidence scaled by lam. It never reaches back across a perceived event, so
    it cannot start an utterance, only carry one on."""
    S, cfg = mind.S, mind.cfg
    if not int(S.u_kind[unit]) & COMPOSED:
        return {}
    parts = S.parts(unit)
    start = 0
    for j, p in enumerate(parts):
        channel = int(S.u_channel[p])
        if channel >= 0 and mind.ch.kind(channel) == "sense":
            start = j + 1
    tail = parts[start:]
    heard = set(parts[:start])                          # what the partner just said is an echo, not knowledge
    first = 1 if len(tail) == len(parts) else 0         # the whole tuple is the unit itself, read already
    for i in range(first, len(tail)):
        suffix = tail[i:]
        if not suffix:
            continue
        piece = suffix[0] if len(suffix) == 1 else S.tuple_index.get(suffix)
        if piece is None:
            continue
        out = {}
        for target, (w_pos, w_neg) in own_mass(mind, piece, 1, trust).items():
            if _world(mind, target) and w_pos > 0.0 and target not in heard:
                c = cfg.lam * conviction(w_pos + w_neg, cfg.k)
                out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * 0.5
        if any(e > 0.5 for e in out.values()):
            return out
    return {}


def trusted_map(mind, unit: int) -> Dict[int, float]:
    """support = "trusted" (F10): generalisations are sources, pooled by the trust they have earned.

    The same pooling as for provenance classes (W = sum_pi tau_pi w), with the trust of a generalisation
    tau_g = max(0, 2 reliability(g) - 1): one that has predicted no better than chance has no voice. Each speaks
    with at most lam e(i -> g) lived experiences (capped, as in section 4 changed):
        W+_gen(v) = sum_g tau_g lam e(i -> g) W+(g -> v) / max(1, W(g -> v) / k),   W-_gen likewise
        e <- c_gen f_gen + (1 - c_gen) 1/2;   own evidence last: e <- c_own f_own + (1 - c_own) e
    """
    cfg = mind.cfg
    pooled: Dict[int, List[float]] = {}
    for general, e_abs in generalisations(mind, unit):
        tau = max(0.0, 2.0 * reliability(mind, general) - 1.0)
        if tau <= 0.0:
            continue
        for target, (w_pos, w_neg) in own_mass(mind, general).items():
            factor = tau * cfg.lam * e_abs / max(1.0, (w_pos + w_neg) / cfg.k)
            acc = pooled.setdefault(target, [0.0, 0.0])
            acc[0] += factor * w_pos
            acc[1] += factor * w_neg
    out = {t: expectation(p, n, cfg.k) for t, (p, n) in pooled.items()}
    for target, (w_pos, w_neg) in own_mass(mind, unit).items():
        c = conviction(w_pos + w_neg, cfg.k)
        out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    return out


def choice_map(mind, unit: int) -> Dict[int, float]:
    """support = "choice" (F7): for each target, the generalisation that has lived most about it speaks.

        best g = argmax over generalisations of  lam e(i -> g) conviction(g -> v)
        e <- c_g f_g + (1 - c_g) 1/2;   own evidence:  e <- c_own f_own + (1 - c_own) e

    Competing generalisations are not summed and not ordered by size: the most experienced one is chosen (the
    choice rule of NARS between competing beliefs). Own evidence of the instance always comes last.
    """
    cfg = mind.cfg
    best: Dict[int, Tuple[float, float]] = {}
    for general, e_abs in generalisations(mind, unit):
        for target, (w_pos, w_neg) in own_mass(mind, general).items():
            c = cfg.lam * e_abs * conviction(w_pos + w_neg, cfg.k)
            if c > best.get(target, (-1.0, 0.0))[0]:
                best[target] = (c, frequency(w_pos, w_neg))
    out = {t: c * f + (1.0 - c) * 0.5 for t, (c, f) in best.items()}
    for target, (w_pos, w_neg) in own_mass(mind, unit).items():
        c = conviction(w_pos + w_neg, cfg.k)
        out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    return out


def predictions(mind, mode: Optional[str] = None) -> Dict[int, float]:
    """Expectations for the next moment (section 2.3).

    "votes" (as written): every unit of the focus votes a_u * (2 e - 1) for the targets of its lag-1 ties and
    the votes are summed, as in the settling formula. "specific" (changed): the most specific unit of the
    present context (the composed unit of the episode, or its single item) is read with its support.
    """
    cfg = mind.cfg
    mode = cfg.predict if mode is None else mode
    if mode == "own":                                  # the present context's own evidence only (no generalisation)
        if mind.context is None:
            return {}
        return {v: expectation(p, n, cfg.k) for v, (p, n) in own_mass(mind, mind.context).items()}
    if mode == "specific":
        if mind.context is None:
            return {}
        return expect_map(mind, mind.context)
    votes: Dict[int, float] = {}
    for unit, a_u, grounded in mind.focus:
        if a_u <= 0.0:
            continue
        for v, (p, n) in effective(mind, unit).items():
            votes[v] = votes.get(v, 0.0) + a_u * (2.0 * expectation(p, n, cfg.k) - 1.0)
    return {v: min(1.0, max(0.0, 0.5 + 0.5 * vote)) for v, vote in votes.items()}


def belief(mind, unit: int, target: int) -> Dict[str, float]:
    """White-box reading of one lag-1 tie: own and effective frequency, conviction, expectation."""
    cfg = mind.cfg
    own = own_mass(mind, unit).get(target, (0.0, 0.0))
    eff = effective(mind, unit).get(target, (0.0, 0.0))
    out = {}
    for name, (p, n) in (("own", own), ("eff", eff)):
        out[name + "_pos"], out[name + "_neg"] = p, n
        out[name + "_f"] = frequency(p, n)
        out[name + "_c"] = conviction(p + n, cfg.k)
        out[name + "_e"] = expectation(p, n, cfg.k)
    out["eff_e"] = expect_map(mind, unit).get(target, 0.5)
    return out


def drive_of(mind, unit: int) -> float:
    """D(u) = sum over the lag-1 ties of u of use * (1 - conviction) (section 5).

    Only ties to units of sense and act channels count: the drive is uncertainty about the world, not about
    the mind's own feelings (otherwise feeling something would itself be a relief; see NOTES.md).
    """
    S = mind.S
    ties = S.out.get((unit, EXPECT, 1), [])
    if not ties:
        return 0.0
    ids = torch.tensor(ties, dtype=I64)
    world = torch.tensor([c >= 0 and mind.ch.kind(c) in ("sense", "act")
                          for c in S.u_channel[S.t_dst[ids]].tolist()])
    pos, neg = S.w2(ids)
    total = ((pos + neg) * mind.trust).sum(-1)
    return float((S.t_use[ids] * (1.0 - conviction(total, mind.cfg.k)))[world].sum())


def total_drive(mind) -> float:
    """Drive over the recently used units: the units of the present focus."""
    return sum(drive_of(mind, unit) for unit, a, grounded in mind.focus if grounded)


def conflict_of(mind, unit: int) -> float:
    """Mass of the lag-1 ties of ``unit`` with frequency near 1/2 and high conviction (section 2.6)."""
    total = 0.0
    for p, n in own_mass(mind, unit).values():
        total += conviction(p + n, mind.cfg.k) * (1.0 - abs(2.0 * frequency(p, n) - 1.0))
    return total
