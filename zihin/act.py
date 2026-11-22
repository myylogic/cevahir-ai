# -*- coding: utf-8 -*-
"""Acting, imagining and the control knobs (design sections 2.4, 5, 6, 7, 8).

Acting. An act unit is an atomic unit on a channel of kind ``act``. The same unit is active when another
emits the symbol (provenance told / imitated) and when the mind emits it (provenance acted), so "others do X"
and "I do X" are separated by provenance, not by different units:

* others do X in situation S:  EXPECT S -> X, evidence in the class ``imitated``;
* what my doing X in S brings:  EXPECT (S, X) -> relief, evidence of the mind's own outcomes.

The expectation the mind acts on pools the two: W+ = imitated W+(S -> X) + W+((S, X) -> relief),
W- = imitated W-(S -> X) + W-((S, X) -> relief). Own repetitions of X (class ``acted`` on S -> X) are a
record of behaviour and deliberately NOT evidence for doing it again. The act accumulates 2e - 1 per settling
iteration and is emitted when the sum crosses the threshold theta within T iterations; among the acts of one
channel the first to cross wins. With nothing crossing the mind stays silent.

Imagination (flag, simple first version) and control knobs as internal acts (flag, simple first version) are
at the end of the file.
"""
from __future__ import annotations

import math
from typing import Hashable, List, Optional

import torch

from .events import ACTED, IMAGINED, IMITATED, N_PROV, Event
from .evidence import conviction, expectation
from .form import kind_masks, mass1, mass1_total, mv_view
from .learn import add_evidence1
from .read import drive_of, effective, generalisations, own_mass
from .store import ABSTRACT, CATEGORY, EXPECT, F64, SLOTTED, W_NEG, W_POS

KNOB_ACTS = (("F", 1), ("F", -1), ("theta", 1), ("theta", -1), ("gain", 1), ("gain", -1))
KNOB_RANGE = {"F": (3, 15)}


def act_expectation(mind, act_unit: int, imitated_pos: float, imitated_neg: float) -> float:
    """Pooled expectation of doing ``act_unit`` now: imitated evidence plus the mind's own outcomes."""
    S = mind.S
    outcome = act_unit if mind.context is None else S.tuple_index.get(tuple(mind.episode) + (act_unit,))
    if outcome is not None:
        tie = S.find_tie(outcome, mind.feel_unit["relief"], EXPECT, 1)
        if tie >= 0:
            w_pos, w_neg = mass1(mind, tie)
            imitated_pos += w_pos
            imitated_neg += w_neg
    return expectation(imitated_pos, imitated_neg, mind.cfg.k)


def consequence(mind, act_unit: int) -> float:
    """L5: sum over value signals of sign x (2 e - 1) for the path "this episode, then this act", read from the
    EXPECT lag-2 ties that the episode trace collected when such a path was taken before (0 when never taken
    or no value signal is defined). Signs come from the adapter (mind.value_sign). F35 (``purpose``): what a path
    never taken would lead to is read from the paths its frames hold (``general_consequences``), and the path's own
    record comes last."""
    signs = getattr(mind, "value_sign", None)
    if not signs or not mind.episode:
        return 0.0
    S = mind.S
    parts = tuple(mind.episode) + (act_unit,)
    path = S.tuple_index.get(parts)
    values = []
    for (channel, symbol), sign in signs.items():
        if not mind.ch.has(channel):
            continue
        cid = mind.ch.id(channel)
        index = mind.ch.symbol_index(cid, symbol, create=False)
        value_unit = S.atom_index.get((cid, index), -1) if index >= 0 else -1
        if value_unit >= 0:
            values.append((value_unit, sign))
    if not values:
        return 0.0
    general = general_consequences(mind, parts, values) if getattr(mind.cfg, "purpose", False) else {}
    if path is None and not general:
        return 0.0
    total = 0.0
    for value_unit, sign in values:
        e = general.get(value_unit, 0.5)
        tie = S.find_tie(path, value_unit, EXPECT, 2) if path is not None else -1
        if tie >= 0:
            w_pos, w_neg = _in_company(mind, tie, *mass1(mind, tie))
            if general:                                    # the path's own record comes last
                c = w_pos + w_neg
                c = c / (c + mind.cfg.k)
                e = c * (w_pos / (w_pos + w_neg) if w_pos + w_neg > 0.0 else 0.5) + (1.0 - c) * e
            else:
                e = expectation(w_pos, w_neg, mind.cfg.k)
        elif not general:
            continue
        total += sign * (2.0 * e - 1.0)
    return total


def general_consequences(mind, parts, values) -> dict:
    """F35 (Generalised Consequences): what a path leads to, read as its frames' members lived it. For each frame of
    the path (an item of the path replaced by a category it is a member of, as the frames of a context are found),
    each member whose path was lived votes the share of its value evidence (lag-2 ties to the value signals) that each
    value received (F11):  W+(v) = sum_x e_x s_x(v),  W-(v) = sum_x e_x (1 - s_x(v)). The frame whose members agree
    most speaks: e(v) = c f + (1 - c) 1/2, c = lam e(path -> frame) conviction(members). {value unit: e}."""
    from .form import categories_of, parts_of
    S, k = mind.S, mind.cfg.k
    value_units = [v for v, _ in values]
    valued = _valued_paths(mind, value_units)           # the paths that led to good or bad: few
    if not valued:
        return {}
    kinds = mv_view(mind, "u_kind")
    best = None
    for j, part in enumerate(parts):
        if kinds[part] & (CATEGORY | SLOTTED):
            continue
        for category, e_abs in categories_of(mind, part):
            if e_abs <= 0.5:
                continue
            where = [i for i, q in enumerate(parts) if q == part] if getattr(S, "bind_repeats", False) else [j]
            if S.tuple_index.get(tuple(category if i in where else q for i, q in enumerate(parts))) is None:
                continue
            shares = {v: 0.0 for v in value_units}
            members = 0.0
            for instance, masses in valued.items():     # the valued paths that are this frame's instances
                held = parts_of(mind, instance)
                if len(held) != len(parts) or any(held[i] != q for i, q in enumerate(parts) if i not in where):
                    continue
                x = held[where[0]]
                if x == part or any(held[i] != x for i in where):
                    continue
                tie = S.find_tie(x, category, ABSTRACT, 0)
                if tie < 0:
                    continue
                total = sum(masses.get(v, 0.0) for v in value_units)
                if total <= 0.0:
                    continue
                e_x = expectation(*mass1(mind, tie), k)
                members += e_x
                for v in value_units:
                    shares[v] += e_x * masses.get(v, 0.0) / total
            if members <= 0.0:
                continue
            top = max(shares.values()) / members
            score = e_abs * conviction(members, k) * top
            if best is None or score > best[0]:
                best = (score, e_abs, members, shares)
    if best is None:
        return {}
    _, e_abs, members, shares = best
    c = mind.cfg.lam * e_abs * conviction(members, k)
    return {v: c * (w / members) + (1.0 - c) * 0.5 for v, w in shares.items()}


def _in_company(mind, tie: int, w_pos: float, w_neg: float):
    """F33 with F35: what a path led to with other people is theirs (what pleased one person is an expectation of that
    relationship); read in present company when the mind keeps its ledger, else as it is."""
    if getattr(mind.cfg, "company", False) and "personal" in mind.__dict__ and getattr(mind.cfg, "purpose", False):
        from .iliski import company_mass1
        return company_mass1(mind, tie, w_pos, w_neg, None)
    return w_pos, w_neg


def _valued_paths(mind, value_units) -> dict:
    """F35: the composed paths that were later followed by a value signal (EXPECT lag 2), with their positive mass
    toward each value: {path: {value unit: W+}}. Read once a moment."""
    from .read import _moment_memo
    memo = _moment_memo(mind)
    key = ("valued", tuple(value_units))
    if key in memo:
        return memo[key]
    S = mind.S
    kinds, src, lag = mv_view(mind, "u_kind"), mv_view(mind, "t_src"), mv_view(mind, "t_lag")
    from .store import COMPOSED
    found: dict = {}
    for v in value_units:
        for tie in S.inc.get((v, EXPECT), ()):
            u = src[tie]
            if lag[tie] != 2 or not kinds[u] & COMPOSED or kinds[u] & SLOTTED:
                continue
            w_pos = _in_company(mind, tie, *mass1(mind, tie))[0]
            if w_pos > 0.0:
                found.setdefault(u, {})[v] = w_pos
    memo[key] = found
    return found


def _explore(mind, ranked, theta: float, T: int):
    """Hesitation ends in a try (L5 with curiosity), where a try can teach something.

    Only when some alternative is well supported (it would have been chosen alone: 2 (e - 1/2) T >= theta) is the
    silence hesitation rather than not knowing. A try is made when the mind has just been called on (its turn begins:
    the last thing in the episode is a perceived event) or when some alternative's consequences are already known
    here (it has met good or bad on these paths); in the middle of its own utterance with nothing at stake it does
    not wander. It tries, among the well supported alternatives whose known consequences are not bad, the one whose
    consequences it knows least (the evidence on its path's value ties; a path never taken knows nothing). If every
    well supported alternative has led to bad, and only then, it tries among everything else it has lived after this
    (e > 1/10). Ties: better known consequence, then the race order. Returns the chosen (score, e, unit) or None."""
    if not any(2.0 * (r[1] - 0.5) * T >= theta for r in ranked):
        return None                                        # nothing well supported: not knowing, not hesitation
    S = mind.S
    known = {r[2]: _known_consequence(mind, r[2]) for r in ranked}
    at_stake = any(k > 0.0 for k in known.values()) or (           # good or bad was met on these paths, or in
        mind.context is not None and _value_evidence(mind, mind.context) > 0.0)   # this very situation before
    last = mind.episode[-1] if mind.episode else -1
    called_on = last >= 0 and kind_masks(mind)[4][mv_view(mind, "u_channel")[last]]
    if not (called_on or at_stake):
        return None
    value = {r[2]: consequence(mind, r[2]) for r in ranked}
    supported = [r for r in ranked if 2.0 * (r[1] - 0.5) * T >= theta and value[r[2]] >= 0.0]
    pool = supported or ([r for r in ranked if r[1] > 0.1] if at_stake else [])
    if not pool:
        return None
    return min(pool, key=lambda r: (known[r[2]], -value[r[2]], -r[0], r[2]))


def _known_consequence(mind, act_unit: int) -> float:
    """How much the mind knows of what this path led to: the evidence on the value ties of "this episode, then
    this act" (0 when the path was never taken)."""
    if not getattr(mind, "value_sign", None) or not mind.episode:
        return 0.0
    path = mind.S.tuple_index.get(tuple(mind.episode) + (act_unit,))
    return 0.0 if path is None else _value_evidence(mind, path)


def _value_evidence(mind, unit: int) -> float:
    """The evidence on a unit's EXPECT lag-2 ties to the value signals (what it was later followed by: good, bad)."""
    signs = getattr(mind, "value_sign", None)
    if not signs:
        return 0.0
    S = mind.S
    known = 0.0
    for (channel, symbol) in signs:
        if not mind.ch.has(channel):
            continue
        cid = mind.ch.id(channel)
        index = mind.ch.symbol_index(cid, symbol, create=False)
        value_unit = S.atom_index.get((cid, index), -1) if index >= 0 else -1
        tie = S.find_tie(unit, value_unit, EXPECT, 2) if value_unit >= 0 else -1
        if tie >= 0:
            known += mass1_total(mind, tie)
    return known


def _race(mind, candidates, theta: float, T: int, dry: bool = False) -> List[Event]:
    """F19: one act per channel per moment, so the candidates of a channel are alternatives that exclude each
    other. Each scores its expectation plus half its consequences (L5); the leader accumulates its advantage over
    the strongest rival (over 1/2, "saying nothing", when it has none) and is emitted if that advantage reaches the
    threshold within T iterations: rate = 2 (score_leader - score_rival). With no rival this is the rule as before
    (2e - 1); with rivals that are all equally (un)supported nothing leads and the mind stays silent."""
    S = mind.S
    channels = mv_view(mind, "u_channel")
    by_channel = {}
    for unit, e in candidates.items():
        score = e + 0.5 * consequence(mind, unit)
        by_channel.setdefault(channels[unit], []).append((score, e, unit))
    acts: List[Event] = []
    for channel in sorted(by_channel):
        ranked = sorted(by_channel[channel], reverse=True)
        score, e, unit = ranked[0]
        rival = ranked[1][0] if len(ranked) > 1 else 0.5
        if getattr(mind.cfg, "known", False) and getattr(mind, "topic", None) is not None:
            driven, unsaid = _driven(mind, channel)
            if not driven:
                continue                               # F41: nothing known of the topic is left to say
            if unit in unsaid:                         # F41: the other things known of the topic and not said yet are
                rival = next((r[0] for r in ranked[1:] if r[2] not in unsaid), 0.5)   # still to say, not rivals
        if getattr(mind.cfg, "urge", False) and len(ranked) > 1 and rival > 0.5 + theta / (2.0 * T):
            u = _urge(mind, unit, ranked[1][2])        # F40: how far the two are things known about the same item
            rival = (1.0 - u) * rival + u * 0.5        # (another thing to say, not an alternative): against silence
        rate = 2.0 * (score - max(rival, 0.5 if score <= 0.5 else rival))
        if rate <= 0.0 or math.ceil(theta / rate - 1e-12) > T:
            tried = _explore(mind, ranked, theta, T) if getattr(mind.cfg, "explore", False) else None
            if tried is None:
                continue
            score, e, unit = tried
        if dry:                                        # F49 (silent claims): only which unit it would say; nothing
            acts.append(unit)                          # is set and nothing is said
            continue
        acts.append(Event(mind.ch.names[channel], mind.ch.symbol(channel, int(S.u_symbol[unit])), ACTED))
        mind.hoping = (unit, e)
        if getattr(mind.cfg, "known", False):          # F41: a thing known of a filler is said: the answer's topic
            from .read import _moment_memo
            claim = _moment_memo(mind).get(("known_claim", mind.context))
            if claim is not None and claim[2] == unit:
                mind.topic = (claim[0], claim[1], mind.episode_start)
        if mind.__dict__.get("_explain_on"):           # the chat's !neden: where this word came from
            from .neden import attribute
            mind.__dict__.setdefault("_why", []).append(attribute(mind, unit, consequence(mind, unit)))
    return acts


def _driven(mind, channel: int):
    """F41 (the urge to go on): an answer that tells what is known of its topic is driven by what is still unsaid.
    (driven, the things known of the topic that may be said now). Not driven when the topic is in the present
    context, its channel is this one, its own lived sequence has ended there (or it has just been named) and nothing
    known of it is unsaid: the answer has said what there was to say. Just after the topic is named, the things known
    of it and not said yet are all things to say: none of them is another's rival."""
    from .read import known_unsaid, topic_state, unsaid_share
    if mind.context is None or mv_view(mind, "u_channel")[mind.topic[0]] != channel:
        return True, ()
    not_self = mind.trust.clone()
    not_self[ACTED] = 0.0
    state = topic_state(mind, mind.context, not_self)
    if state is None or state[2] == "inside":
        return True, ()
    y, j, where, _ = state
    if where == "start":
        unsaid = known_unsaid(mind, y, mind.context, j, not_self, episode=True)
        if unsaid:
            return True, unsaid
    return unsaid_share(mind, y, not_self) > 0.0, ()


def _urge(mind, a: int, b: int) -> float:
    """F40 (Urge to Go On): how far two expected words are two things known about the same item rather than two
    alternatives, u in [0, 1]: over the items of the present context, the largest share of an item's lived
    continuations on the words' channel (W+) that the two hold together. "uçar" and "fare" after "… baykuş" are most of
    what "baykuş" was lived to go on with (two facts of baykuş: saying either is right, they compete only for the
    order); "hamsi" and "levrek" after "… balıktır ." are almost nothing of what any item there went on with (only one
    can be right). The rival is then weighed as (1 - u) rival + u 1/2: the race of F19 for alternatives, a race against
    silence for things to say. A rival that is not itself over the speaking line is never weighed so (the mind would
    not have said it)."""
    from .form import parts_of
    from .read import _unit_arrays
    from .store import COMPOSED
    if mind.context is None:
        return 0.0
    channels = mv_view(mind, "u_channel")
    channel = channels[a]
    items = parts_of(mind, mind.context) if mv_view(mind, "u_kind")[mind.context] & COMPOSED else (mind.context,)
    best = 0.0
    for x in set(items):
        found = _unit_arrays(mind, x, 1)
        if found is None:
            continue
        dst, pos, _ = found
        total = both = 0.0
        hit_a = hit_b = False
        for v, p in zip(dst, pos):
            p = float(p)
            if p <= 0.0 or channels[v] != channel:      # what x went on with on the words' channel
                continue
            total += p
            if v == a:
                both += p
                hit_a = True
            elif v == b:
                both += p
                hit_b = True
        if hit_a and hit_b and total > 0.0:
            best = max(best, both / total)
    return best


def decide(mind) -> List[Event]:
    """Step 4: accumulate to threshold; at most one act per act channel; silence is legal."""
    S, cfg = mind.S, mind.cfg
    theta, T = mind.knob["theta"], cfg.T
    candidates = {}
    if mind.context is not None and cfg.support == "members" and cfg.bind_repeats:
        from .read import members_map                      # F12: speak with what the frame's members lived
        not_self = mind.trust.clone()
        not_self[ACTED] = 0.0                              # own repetitions are a record, not a reason (section 7)
        channels, act = mv_view(mind, "u_channel"), kind_masks(mind)[5]
        faults = getattr(cfg, "faults", False)
        if faults:
            mind.__dict__.pop("_src_last", None)
        for unit, e in members_map(mind, mind.context, not_self).items():
            if act[channels[unit]] and (e > 0.5 or cfg.race):
                candidates[unit] = e
        if faults and cfg.race:                            # F49: what each way claimed here is kept until reality
            from . import hata                             # speaks of this context
            found = mind.__dict__.pop("_src_last", None)
            src = found[1] if found is not None and found[0] == mind.context else {}
            mind.__dict__["_src_now"] = src                # (the chat's !neden names the way of a word from it)
            acts = _race(mind, candidates, theta, T)
            if src:
                hata.note(mind, src, [mind.unit_of(event.channel, event.symbol) for event in acts])
            return acts
    elif mind.context is not None:
        only_imitated = torch.zeros(N_PROV, dtype=F64)
        only_imitated[IMITATED] = mind.trust[IMITATED]
        channels, act = mv_view(mind, "u_channel"), kind_masks(mind)[5]
        for unit, (w_pos, w_neg) in effective(mind, mind.context, trust=only_imitated).items():
            if act[channels[unit]] and w_pos > 0.0:
                candidates[unit] = act_expectation(mind, unit, w_pos, w_neg)
    elif cfg.knobs and mind.ch.has("knob"):               # knob acts: only their own outcomes speak
        channel = mind.ch.id("knob")
        for index in range(len(mind.ch.symbols[channel])):
            unit = S.atom_index.get((channel, index), -1)
            if unit >= 0 and S.find_tie(unit, mind.feel_unit["relief"], EXPECT, 1) >= 0:
                candidates[unit] = act_expectation(mind, unit, 0.0, 0.0)
    if cfg.race:
        return _race(mind, candidates, theta, T)
    best = {}
    for unit, e in candidates.items():
        rate = 2.0 * e - 1.0 + consequence(mind, unit)  # L5 will: the drive plus what this path led to
        if rate <= 0.0:
            continue
        latency = math.ceil(theta / rate - 1e-12)
        if latency > T:
            continue
        channel = int(S.u_channel[unit])
        entry = (latency, -e, unit)
        if channel not in best or entry < best[channel]:
            best[channel] = entry
    acts: List[Event] = []
    for channel in sorted(best):
        latency, neg_e, unit = best[channel]
        acts.append(Event(mind.ch.names[channel], mind.ch.symbol(channel, int(S.u_symbol[unit])), ACTED))
        mind.hoping = (unit, -neg_e)
    return acts


def think_now(mind) -> Optional[float]:
    """F48 (Thinking by Itself): in a silent moment, whether the mind decides to think about what it last took in,
    as the rate of that decision, or None.

    What it took in last, and has not thought about, left a feeling: its surprise over those moments (``Mind.heard``:
    under 1/2 when what it expected came, 1/2 when it expected nothing, over 1/2 when what it expected did not come).
    As for any act (``decide``): rate = the drive + what this led to before,
        rate = (2 surprise - 1) + (2 e_think - 1),
    e_think what its thoughts have led to (``read.think_expectation``; 1/2 before its first), acted on when
    ceil(theta / rate) <= T: the line it speaks at. So it thinks about what went against its expectations, and it
    learns from what its thoughts come to whether thinking is worth it. Reads only: the thought is the session's to
    let happen (the inner voice, ``dusunce.think``), and it is settled in ``learn.settle_thought``."""
    from .read import think_expectation
    heard = getattr(mind, "heard", None)
    if heard is None:
        return None
    rate = (2.0 * heard - 1.0) + (2.0 * think_expectation(mind) - 1.0)
    if rate <= 0.0 or math.ceil(mind.knob["theta"] / rate - 1e-12) > mind.cfg.T:
        return None
    return rate


GROUNDED = (-1, -2, -3, -5, -7)   # F50: the kinds of a frame's claim that read what was lived of the filler itself
                                  # (the asked word, its continuation, how its sequences began, what it was answered
                                  # with alone, what is known of it)


def grounded(made) -> bool:
    """F50: whether a word's claim rests on what the mind lived of the thing asked about: some way that claims it
    (more for than against) is a frame's relation that reads the filler's own record. ``made``: [(ways, the reading's
    own claim, the claim), ...] (hata's chain). The members' own word, the members of its kind, a continuation, a
    form or a similar situation rest on what others share."""
    for ways, e0, _ in made:
        kind = ways[1]
        if e0 > 0.5 and len(kind) == 2 and kind[0] == "g" and kind[1] in GROUNDED:
            return True
    return False


def lived_so(mind, context: int, unit: int) -> bool:
    """F50: whether ``unit`` is how the mind's own utterance so far went on when it lived those words as a sequence
    of their own: what a sentence it read or was told went on with from the item its utterance began with
    (``read.known_unsaid``; with F41 also from the answer's topic). It is then saying something it lived of that
    very thing, whichever way made the claim. What it only ever said itself, or imagined, is not counted, nor what
    followed the same words in the present episode (an echo)."""
    from .events import IMAGINED
    from .form import parts_of
    from .read import known_unsaid, kind_masks, mv_view
    from .store import COMPOSED
    kinds, channels = mv_view(mind, "u_kind"), mv_view(mind, "u_channel")
    parts = parts_of(mind, context) if kinds[context] & COMPOSED else (context,)
    sense = kind_masks(mind)[4]
    start = 0
    for j, p in enumerate(parts):
        if sense[channels[p]]:
            start = j + 1
    places = [start] if start < len(parts) else []
    topic = getattr(mind, "topic", None)
    if topic is not None and topic[0] in parts:
        places.append(len(parts) - 1 - parts[::-1].index(topic[0]))
    trust = mind.trust.clone()
    trust[ACTED] = 0.0
    trust[IMAGINED] = 0.0
    return any(known_unsaid(mind, parts[j], context, j, trust, episode=True).get(unit, 0.0) > 0.0 for j in places)


def rests_on_others(mind, context: int, unit: int, made) -> bool:
    """F50: whether the word the mind is about to say at ``context`` rests only on what others share: some way
    claims it (``made``: hata's chain), none of them reads the record of the thing asked about (``grounded``), the
    context has no record of its own of a word there, and it is not how a sentence the mind lived went on from
    where its own utterance began (``lived_so``)."""
    from .events import IMAGINED
    from .read import mv_view, own_mass
    if not made or grounded(made):
        return False
    channels = mv_view(mind, "u_channel")
    trust = mind.trust.clone()
    trust[ACTED] = 0.0
    trust[IMAGINED] = 0.0
    if any(w_pos > 0.0 and channels[v] == channels[unit] for v, (w_pos, _) in
           own_mass(mind, context, 1, trust).items()):
        return False                                       # this very situation was lived: memory speaks
    return not lived_so(mind, context, unit)


def look_now(mind, items, acts) -> Optional[tuple]:
    """F50 (Finding Out): in a moment in which the mind is free to speak and something was just lived for real
    (someone's turn has ended, or it is in the middle of its own answer), whether it decides to look something up
    before it says the next word, as (the item's unit, its symbol, the rate, its not-knowing u, whether the open
    claim is a plain one, whether it is about the next word only: it had a word to say), or None. Not in a moment
    that only comes to mind (a thought, a situation recalled).

    Two cases, one measure. It decided nothing where something of its own is expected (a word more for than
    against, or a claim of its frame about the thing asked about that was left open in this moment; it does not
    expect the world to go on: it is not done): u = 1 - 2 max(0, m), m its best expectation's margin over what
    stands against it (the strongest rival, or 1/2): a claim left open and hesitation between rivals are both not
    knowing. Where nothing of its own is expected it was told something, not asked: that is no reason to look (what
    is new in it the mind may ask itself about in its own time, F51). Or it is about to say a word whose claim does not
    rest on what it lived of the thing asked about (``grounded``), at a context it has no record of its own of:
    u = 1. Of the thing itself it has lived nothing there; how firmly the things of its kind agree is what it
    expects, not what it knows (and is what it goes by when it cannot look, or found nothing). As for any act:
    rate = the drive + what this led to before,
        rate = (2u - 1) + (2 e_look - 1),
    acted on when ceil(theta / rate) <= T. What to look up: the item the frame that speaks is about; with no frame,
    the item of what it was just told that it knows least (one never met before this run of items, else the best
    topic by its own evidence). Not an item whose claims its own record holds to be met (e_open under 1/2: it has
    read about it; the other words of the answer are not found by reading the same again). Reads only."""
    from .events import IMAGINED
    from .mind import INNER_CHANNEL
    from .read import _moment_memo, kind_masks, look_expectation, mv_view, own_mass
    context = mind.context
    if context is None or not mind.ch.has(INNER_CHANNEL):
        return None
    channels = mv_view(mind, "u_channel")
    inner = mind.ch.id(INNER_CHANNEL)
    if any(p == IMAGINED or channels[u] == inner for u, p in items):
        return None                                        # a moment that only comes to mind, or an inner event
    act = kind_masks(mind)[5]
    said = [mind.unit_of(e.channel, e.symbol) for e in acts if mind.ch.has(e.channel)]
    said = [u for u in said if u >= 0 and act[channels[u]]]
    if said:
        made = (mind.__dict__.get("_src_now") or {}).get(said[0])
        if not rests_on_others(mind, context, said[0], made):
            return None                                    # it rests on its own record, or on what it lived of it
        from . import hata                                 # of the thing itself it has lived nothing here: what it
        u = 1.0 - max(0.0, 2.0 * hata.proven(mind, made[0][0]) - 1.0)   # knows is how far this way is proven to hold
    else:
        sense = kind_masks(mind)[4]
        if any(e > 0.5 for v, e in mind.predicted.items() if sense[channels[v]] and channels[v] != inner):
            return None                                    # it expects the world to go on (the end of its turn): it
        ranked = sorted((e for v, e in mind.predicted.items() if act[channels[v]]), reverse=True)   # is done
        best = ranked[0] if ranked else 0.5
        rival = max(ranked[1], 0.5) if len(ranked) > 1 else 0.5
        if best <= 0.5 and not any(type(key) is tuple and len(key) == 2 and key[0] == "claim" and held[1]
                                   for key, held in _moment_memo(mind).items()):
            return None                                    # nothing of its own is expected here and no claim of its
        u = 1.0 - 2.0 * max(0.0, best - rival)             # frame was left open: it was told something, not asked
    item = _moment_memo(mind).get(("filler", context))
    plain = True
    if item is None:                                       # no frame speaks: only when it has just been called on
        sense = kind_masks(mind)[4]                        # from outside (not where it stopped in its own answer)
        if said or not any(p not in (ACTED, IMAGINED) and sense[channels[u]] and channels[u] != inner
                           for u, p in items):
            return None
        item = _least_known(mind, context)
    elif not said:
        plain = ("claim", item) not in _moment_memo(mind)   # a relation of its frame was left open: F46's own claim
    if item is None:
        return None
    v = mind.S.arrays()
    symbol = mind.ch.symbol(int(v.channel[item]), int(v.symbol[item]))
    from .read import OPEN_CHANNEL, mass1, open_claim
    held = mind.unit_of(OPEN_CHANNEL, symbol)
    tie = mind.S.find_tie(held, mind.holds, EXPECT, 0) if held >= 0 else -1
    if tie >= 0 and sum(mass1(mind, tie)) > 0.0 and open_claim(mind, symbol) <= 0.5:
        return None                                        # its claims were open and are met: it has read about it
    rate = (2.0 * u - 1.0) + (2.0 * look_expectation(mind, symbol) - 1.0)
    if rate <= 0.0 or math.ceil(mind.knob["theta"] / rate - 1e-12) > mind.cfg.T:
        return None
    return item, symbol, rate, u, plain, bool(said)


def wonder(mind) -> Optional[tuple]:
    """F51 (Asking Itself): in a silent moment in which nothing else was decided, the item the mind decides to ask
    itself about, as (its unit, its symbol, the rate), or None. Its candidates are the items it has met in one run
    of items only, in what it was told or read (``Mind.met_once``: an item lived again in another run is no longer
    new, so the words that every sentence holds never stay), the oldest first. Of such an item it has lived one
    mention: u = 1. As for any act: rate = (2u - 1) + (2 e_wonder - 1), e_wonder what asking itself has led to
    (``read.wonder_expectation``: a look that then met the unknown it found counts for, one that did not counts
    against; 1/2 before the first). It is acted on when ceil(theta / rate) <= max(T, the moments it has now been
    silent): within a moment the drive must reach the line in T, as for any act; over a silence in which nothing
    else happens it goes on adding up, so an act that has rarely paid is taken again after a longer wait and is
    never given up for good (after n askings that led nowhere, rate = 1 / (n + 1): the next after n + 1 silent
    moments). Reads only: the asking is the inner voice's (``dusunce.ask_itself``), carried out by the session."""
    from .read import wonder_expectation
    fresh = getattr(mind, "met_once", None)
    if not fresh:
        return None
    rate = 1.0 + (2.0 * wonder_expectation(mind) - 1.0)
    if rate <= 0.0 or math.ceil(mind.knob["theta"] / rate - 1e-12) > max(mind.cfg.T, getattr(mind, "idle", 0)):
        return None                                        # (patience: see below)
    v = mind.S.arrays()
    for unit in fresh:
        if unit < mind.S.n_units and v.ualive[unit]:
            return unit, mind.ch.symbol(int(v.channel[unit]), int(v.symbol[unit])), rate
    return None


def holds_back(mind, items, acts) -> bool:
    """F50: whether the word the mind is about to say is one it does not assert: it rests on what others share
    (``rests_on_others``), the way that claims it is not proven to hold by its own record (``hata.proven`` at most
    1/2: never settled, or not yet held more often than it failed by more than one experience: one lucky hit is
    not knowledge), and the mind's own record holds that it does not know this
    of the item: a relation of its frame that reads the item's own record names nothing for it in this moment
    (F46), or a claim about it was left open before and is still open (it looked and found nothing, or could not
    look). What it could not find out it does not say as if it knew, as long as what it holds back is answered
    (``hata.answered`` at least 1/2: someone shows it, or it can look later); where no one ever does, saying it is
    the only way to find out. In any moment, also one that comes to mind again (the situation it is brought back
    to after a look)."""
    from .events import IMAGINED
    from .read import OPEN_CHANNEL, _moment_memo, kind_masks, mass1, mv_view, open_claim, own_mass
    from .mind import INNER_CHANNEL
    context = mind.context
    if context is None or not acts:
        return False
    channels = mv_view(mind, "u_channel")
    if mind.ch.has(INNER_CHANNEL) and any(channels[u] == mind.ch.id(INNER_CHANNEL) for u, _ in items):
        return False                                       # what one says in an inner state (not knowing, looking)
    act = kind_masks(mind)[5]                              # is not a claim about the thing asked about
    said = [mind.unit_of(e.channel, e.symbol) for e in acts if mind.ch.has(e.channel)]
    said = [u for u in said if u >= 0 and act[channels[u]]]
    if not said:
        return False
    made = (mind.__dict__.get("_src_now") or {}).get(said[0])
    if not rests_on_others(mind, context, said[0], made):
        return False
    from . import hata
    if hata.proven(mind, made[0][0]) > 0.5 or hata.answered(mind) < 0.5:
        return False                                       # this way is proven to hold by its own record; or what
    item = _moment_memo(mind).get(("filler", context))     # it holds back is not answered (no one shows it, it has
    if item is None:                                       # nowhere to look): saying it is how it finds out
        return False
    v = mind.S.arrays()
    symbol = mind.ch.symbol(int(v.channel[item]), int(v.symbol[item]))
    now = _moment_memo(mind).get(("claim", item))         # F46: a relation of its frame left open in this very moment
    if now is None or not now[1]:
        if mind.__dict__.get("_settling"):                 # (on the copy that finds out whether a claim is still open:
            return False                                   # what was open before is the question, not the answer)
        held = mind.unit_of(OPEN_CHANNEL, symbol)
        tie = mind.S.find_tie(held, mind.holds, EXPECT, 0) if held >= 0 else -1
        if tie < 0 or sum(mass1(mind, tie)) <= 0.0 or open_claim(mind, symbol) <= 0.5:
            return False
    return True


def _least_known(mind, context: int) -> Optional[int]:
    """F50: of the items of what the mind was just told (the single items of the context, on an act channel), the
    one it knows least: the last one that was never met before the present run of items; when it met them all, the
    best topic by its own evidence (dusunce.topic_score: telling and not yet well known)."""
    from .dusunce import topic_score
    from .form import parts_of
    from .read import kind_masks, mv_view
    from .store import ATOMIC
    v = mind.S.arrays()
    act = kind_masks(mind)[5]
    channels = mv_view(mind, "u_channel")
    parts = [p for p in (parts_of(mind, context) or (context,)) if v.kind[p] & ATOMIC and act[channels[p]]]
    if not parts:
        return None
    since = getattr(mind, "episode_start", mind.t)
    new = [p for p in parts if int(v.ubirth[p]) >= since]
    if new:
        return new[-1]
    scored = [(topic_score(mind, p), -i, p) for i, p in enumerate(parts)]
    best = max(scored)
    return best[2] if best[0] > 0.0 else None


def look(mind) -> Optional[tuple]:
    """F47 (Looking Up by Itself): in a silent moment, the item the mind decides to look up, as (its unit, its
    symbol, the rate of the decision), or None.

    Its candidates are the items a claim of its own frames was left open for (F46; e_open over 1/2) at a context
    that still stands. For each, as for any act (``decide``): rate = the drive + what this led to before,
        rate = (2 e_open - 1) + (2 e_look - 1),
    e_look what looking that item up has led to (``read.look_expectation``: its own record over the record of every
    look; 1/2 when nothing was ever looked up, so the first look is driven by the want alone). It acts when the rate
    reaches the threshold in time, ceil(theta / rate) <= T: the line it speaks at. A look that did not close the
    claim counts against looking that item up again, and against looking in general; one that closed it ends the
    want. The first of equals: the shortest latency, then the firmest want, then the oldest item. Reads only: the
    act is carried out by the session (``web.look_up``: before anything is fetched the mind is brought back to the
    situation, and when it can now say it to the end there is nothing to look up) and settled in
    ``learn.settle_look``."""
    from .read import claim_again, look_expectation, open_claim
    where = getattr(mind, "open_where", None)
    if not where or not getattr(mind.cfg, "open_claims", False):
        return None
    S = mind.S
    theta, T = mind.knob["theta"], mind.cfg.T
    v = S.arrays()
    found = []
    asked = getattr(mind, "wondered", None) if getattr(mind.cfg, "wonder", False) else None
    for filler, context in where.items():
        if filler >= S.n_units or not v.ualive[filler]:
            continue
        symbol = mind.ch.symbol(int(v.channel[filler]), int(v.symbol[filler]))
        e_open = open_claim(mind, symbol)
        own = bool(asked) and filler in asked              # F51: it asked itself this, could not answer and has
        if own:                                            # not looked yet: that is not knowing for certain
            e_open = 1.0                                   # (u = 1), and its drive adds up over a silence
        if e_open <= 0.5:                                  # (``wonder``'s patience). What it looked for and did not
            continue                                       # find is looked for again by the rule as it was
        rate = (2.0 * e_open - 1.0) + (2.0 * look_expectation(mind, symbol) - 1.0)
        if rate <= 0.0:
            continue
        latency = math.ceil(theta / rate - 1e-12)
        if latency > (max(T, getattr(mind, "idle", 0)) if own else T):
            continue
        found.append((latency, -e_open, filler, symbol, rate))
    plain = getattr(mind, "open_plain", ())
    for latency, _, filler, symbol, rate in sorted(found, key=lambda entry: entry[:3]):
        if filler in plain or claim_again(mind, where[filler], filler) is not None:   # (nothing claims it there
            return filler, symbol, rate                             # any more: nothing to look for; F50: a plain
    return None                                                     # claim is settled by saying it, not by a frame)


# ---------------------------------------------------------------------- control knobs (section 8, simple)
def apply_knob(mind, symbol: Hashable) -> None:
    """Setting a knob is an internal act: the echo of the act changes the knob."""
    name, direction = symbol
    if name == "F":
        low, high = KNOB_RANGE["F"]
        mind.knob["F"] = max(low, min(high, mind.knob["F"] + int(direction)))
    elif name in ("theta", "gain"):
        mind.knob[name] = mind.knob[name] * 2.0 ** (0.25 * int(direction))


def babble(mind) -> List[Event]:
    """Exploration of the knobs: nobody shows the mind how to turn them, so no imitated evidence can start
    them. Every ``babble_every`` idle moments the next knob act of a fixed round is emitted (deterministic)."""
    cfg = mind.cfg
    if cfg.babble_every <= 0 or mind.idle == 0 or mind.idle % cfg.babble_every:
        return []
    mind.add_channel("knob", "knob")
    symbol = KNOB_ACTS[mind.babbled % len(KNOB_ACTS)]
    mind.babbled += 1
    return [Event("knob", symbol, ACTED)]


# ---------------------------------------------------------------------- imagination (section 6, simple)
def imagine(mind) -> int:
    """Idle moment: construct conclusions from support alone and record them as imagined.

    Start from the units of the last focus with the highest drive (at most H). A conclusion is a tie i -> v that the
    instance does not hold itself but one generalisation g expects (chain i -> g -> v). Its reality score R
    is the expectation of i -> v from all evidence that is NOT on that chain (other generalisations; 1/2 when
    there is none). It is written with provenance ``imagined`` and weight iota * R. Later direct experience on
    the same tie judges it and so moves the trust in the mind's own constructions (``learn.judge``).
    """
    S, cfg = mind.S, mind.cfg
    starts = sorted((-drive_of(mind, u), u) for u, _, grounded in mind.last_focus
                    if grounded and not int(S.u_kind[u]) & (CATEGORY | SLOTTED))
    made = 0
    for _, instance in starts[:cfg.H]:
        own = own_mass(mind, instance)
        for general, _ in generalisations(mind, instance):
            for target, (w_pos, w_neg) in own_mass(mind, general).items():
                if expectation(w_pos, w_neg, cfg.k) <= 0.5 or sum(own.get(target, (0.0, 0.0))) > 0.0:
                    continue
                tie = S.find_tie(instance, target, EXPECT, 1)
                if tie >= 0 and float(S.w1(W_POS, tie)[IMAGINED] + S.w1(W_NEG, tie)[IMAGINED]) > 0.0:
                    continue
                rest = effective(mind, instance, skip=general).get(target, (0.0, 0.0))
                reality = expectation(rest[0], rest[1], cfg.k) if cfg.reality else 1.0
                add_evidence1(mind, W_POS, IMAGINED, S.tie(instance, target, EXPECT, 1), cfg.iota * reality)
                made += 1
    mind.imagined += made
    return made
