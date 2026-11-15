# -*- coding: utf-8 -*-
"""The one learning rule of Zihin (design C3, section 2 step 5, section 4 "up", section 1 on trust).

Everything here is local to ties whose source was in the focus, and every change is an additive, logged
increment. Three uses of the same rule:

* ``learn_follow``  EXPECT lag 1: what was in the last focus is tied to what occurs now; what a tie expected
  on a channel that reported something else gets negative evidence weighted by the tie's own expectation.
* ``learn_together`` EXPECT lag 0: grounded units that are in the focus together.
* induction "up": a generalisation (category or frame) in the last focus acquires, with provenance
  ``inferred``, what its instance acquired, each instance counting at most ``induct_cap``.

Learned trust. When direct experience (a reference class) adds evidence to a tie, the evidence of the other
classes on that tie that has not been judged yet is judged: what agrees with the direct experience is positive
evidence on the tie (source of that class -> "holds"), what disagrees is negative. Judged mass is recorded in
the ``checked`` tables so that every claim is judged once.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .events import ACTED, INFERRED, REFERENCE, SENSED
from .evidence import conviction, expectation
from .store import ABSTRACT, ATOMIC, C_NEG, C_POS, CATEGORY, COMPOSED, EXPECT, F64, I64, RELY, SLOTTED, W_NEG, W_POS


def add_evidence(mind, table: int, slot: int, ids: torch.Tensor, amounts: torch.Tensor,
                 relief: bool = True, all_expect: bool = False) -> torch.Tensor:
    """Add evidence through the store; keep the relief bookkeeping; let direct experience judge the rest.

    ``relief`` False: the evidence is the echo of the mind's own act and says nothing about the world, so it
    does not count as a decrease of the drive (see NOTES.md, finding on self-made relief).
    ``all_expect``: the caller took ``ids`` from an EXPECT key of ``S.out`` (every one is an expectation).
    """
    S = mind.S
    if ids.numel() == 0:
        return amounts
    if ids.numel() == 1:                        # one tie: the same steps in plain numbers
        return torch.tensor([add_evidence1(mind, table, slot, int(ids[0]), float(amounts[0]), relief)], dtype=F64)
    amounts = S.add(table, slot, ids, amounts)  # snapped to the grid, aligned with ids
    if table in (W_POS, W_NEG):
        if all_expect:
            claims, claimed = ids, amounts
        else:
            expect = S.t_form[ids] == EXPECT
            if not bool(expect.any()):
                return amounts
            claims, claimed = ids[expect], amounts[expect]
        if relief:
            mind._gained.append((claims, claimed * float(mind.trust[slot])))  # for relief (section 5)
        if slot in REFERENCE and mind.cfg.learn_trust:
            judge(mind, table, claims, claimed)  # only expectations are claims about the world
    return amounts


def add_evidence1(mind, table: int, slot: int, tie: int, amount: float, relief: bool = True) -> float:
    """``add_evidence`` for one tie in plain numbers: the same snapped increment, relief entry and judging, in
    the same order. Returns the snapped amount."""
    S = mind.S
    got = S.add1(table, slot, tie, amount)
    if table <= W_NEG and S.views()[1][tie] == EXPECT:
        if relief:
            mind._gained.append((tie, got * float(mind.trust[slot])))   # one tie: (id, gain), see Mind._feel
        if slot in REFERENCE and mind.cfg.learn_trust:
            judge1(mind, table, tie, got)
    return got


def judge1(mind, table: int, tie: int, amount: float) -> None:
    """``judge`` for one tie in plain numbers (a one-element sum is that element)."""
    S = mind.S
    share = min(amount, 1.0)                                # torch.clamp(amount, max=1.0)
    w = S.views()[0]
    for slot in range(S.P):
        if slot in REFERENCE or mind.source_tie[slot] < 0:
            continue
        open_pos = (float(w[W_POS, tie, slot]) - float(w[C_POS, tie, slot])) * share
        open_neg = (float(w[W_NEG, tie, slot]) - float(w[C_NEG, tie, slot])) * share
        if open_pos == 0.0 and open_neg == 0.0:
            continue
        open_pos = S.add1(C_POS, slot, tie, open_pos)
        open_neg = S.add1(C_NEG, slot, tie, open_neg)
        agree = open_pos if table == W_POS else open_neg
        differ = open_neg if table == W_POS else open_pos
        S.add1(W_POS, SENSED, mind.source_tie[slot], agree)
        S.add1(W_NEG, SENSED, mind.source_tie[slot], differ)


def judge(mind, table: int, ids: torch.Tensor, amounts: torch.Tensor) -> None:
    """Direct experience on ties ``ids`` judges the not yet judged evidence of the learned-trust classes.

    The four tables of the ties are read once: judging a class writes only its own checked column, after reading
    it, and the source ties (which are never judged). A class with nothing open on any of the ties is passed over
    (its open amounts are all zero) unless a share or a held value is not finite."""
    S = mind.S
    share = torch.clamp(amounts, max=1.0)  # one full direct experience judges everything outstanding
    held = S.t_w.index_select(1, ids)
    open_any = ((held[W_POS] != held[C_POS]) | (held[W_NEG] != held[C_NEG])).any(0).tolist()
    finite = bool(torch.isfinite(share).all()) and bool(torch.isfinite(held).all())
    for slot in range(S.P):
        if slot in REFERENCE or mind.source_tie[slot] < 0 or (finite and not open_any[slot]):
            continue
        open_pos = (held[W_POS, :, slot] - held[C_POS, :, slot]) * share
        open_neg = (held[W_NEG, :, slot] - held[C_NEG, :, slot]) * share
        if float(open_pos.sum()) == 0.0 and float(open_neg.sum()) == 0.0:
            continue
        open_pos = S.add(C_POS, slot, ids, open_pos)
        open_neg = S.add(C_NEG, slot, ids, open_neg)
        agree = float(open_pos.sum()) if table == W_POS else float(open_neg.sum())
        differ = float(open_neg.sum()) if table == W_POS else float(open_pos.sum())
        S.add1(W_POS, SENSED, mind.source_tie[slot], agree)
        S.add1(W_NEG, SENSED, mind.source_tie[slot], differ)


def rows(S, table: int, ids: torch.Tensor) -> torch.Tensor:
    """The per-class values of one table for the ties ``ids`` ([n, P], the same values as S.t_w[table, ids])."""
    return S.t_w[table].index_select(0, ids)


def tie_mass(mind, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trust-weighted (W+, W-) of ties."""
    S = mind.S
    return (rows(S, W_POS, ids) * mind.trust).sum(-1), (rows(S, W_NEG, ids) * mind.trust).sum(-1)


def _present_of(dst: torch.Tensor, occurred: Dict[int, Tuple[float, int]]) -> torch.Tensor:
    """The intensity with which each target occurred now (0 for those that did not)."""
    if dst.numel() <= 64:
        return torch.tensor([occurred[d][0] if d in occurred else 0.0 for d in dst.tolist()], dtype=F64)
    present = torch.zeros(dst.numel(), dtype=F64)
    for unit, (intensity, _) in occurred.items():
        present[dst == unit] = intensity
    return present


def learn_follow(mind, percepts: List[Tuple[int, float, int, int]]) -> None:
    """EXPECT lag 1 between the last focus and what occurs now (percepts: unit, intensity, slot, channel)."""
    S, cfg = mind.S, mind.cfg
    g = mind.knob["gain"]
    occurred: Dict[int, Tuple[float, int]] = {}
    slot_of_channel: Dict[int, int] = {}
    for unit, intensity, slot, channel in percepts:
        if unit not in occurred or intensity > occurred[unit][0]:
            occurred[unit] = (intensity, slot)
        slot_of_channel.setdefault(channel, slot)
    if not mind.last_focus or not occurred:
        return
    # channels on which the mind hears only itself in this moment (the echo of its own act)
    echo = {ch for ch, slot in slot_of_channel.items() if slot == ACTED and mind.ch.kind(ch) in ("act", "knob")}
    changes: Dict[int, Dict[int, list]] = {}  # instance -> target -> raw masses before/after and slots
    was_present = None
    if cfg.next_only_present:               # F13: "comes next" is learned by what WAS the present a moment ago
        was_present = {u for u in (mind.context, mind.episode[-1] if mind.episode else None) if u is not None}
    # With general_miss and without reliability, induce reads only the records of targets that occurred with
    # intensity > 0 (only their positive evidence grows); the other records are not written.
    acquired = None
    if cfg.general_miss and not cfg.reliability:
        acquired = torch.tensor([v for v, (intensity, _) in occurred.items() if intensity > 0.0], dtype=I64)
    for unit, a_u, grounded in mind.last_focus:
        if not grounded or a_u <= 0.0 or int(S.u_kind[unit]) & (CATEGORY | SLOTTED):
            continue
        if was_present is not None and unit not in was_present:
            continue
        record: Dict[int, list] = {}
        if S.out.get((unit, EXPECT, 1)):
            ids, dst = S.out_ids((unit, EXPECT, 1))
            pos_rows, neg_rows = rows(S, W_POS, ids), rows(S, W_NEG, ids)
            e = expectation((pos_rows * mind.trust).sum(-1), (neg_rows * mind.trust).sum(-1), cfg.k)
            hope = mind.act_hope.get(unit)
            if hope is not None:  # an own act is an expectation of relief (changed rule, see NOTES.md)
                e = torch.where(dst == mind.feel_unit["relief"], torch.clamp(e, min=hope), e)
            raw_pos, raw_neg = pos_rows.sum(-1), neg_rows.sum(-1)
            channel = S.u_channel[dst]
            present = _present_of(dst, occurred)
            weight = e if cfg.neg_weight == "e" else torch.ones_like(e)
            miss = g * a_u * weight * (1.0 - present)
            expected = (e >= cfg.neg_floor) if cfg.neg_floor > 0.0 else torch.ones_like(present, dtype=torch.bool)
            reporting = torch.zeros(ids.numel(), dtype=torch.bool)
            for ch, slot in slot_of_channel.items():
                on = (channel == ch) & expected
                if bool(on.any()):
                    reporting |= on
                    got = add_evidence(mind, W_NEG, slot, ids[on], miss[on], relief=ch not in echo, all_expect=True)
                    d_on, rp_on, rn_on = dst[on], raw_pos[on], raw_neg[on]
                    if acquired is not None:
                        mine = torch.isin(d_on, acquired)
                        if not bool(mine.any()):
                            continue
                        d_on, rp_on, rn_on, got = d_on[mine], rp_on[mine], rn_on[mine], got[mine]
                    for d, rp, rn, m in zip(d_on.tolist(), rp_on.tolist(), rn_on.tolist(), got.tolist()):
                        record[d] = [rp, rp, rn, rn + m, slot, slot]
            S.use_ties(ids[reporting])
        for v, (intensity, slot) in occurred.items():
            if intensity <= 0.0:
                continue
            tie = S.tie(unit, v, EXPECT, 1)
            if v not in record:
                rp, rn = float(S.t_w[W_POS, tie].sum()), float(S.t_w[W_NEG, tie].sum())
                record[v] = [rp, rp, rn, rn, slot, slot]
            got = add_evidence1(mind, W_POS, slot, tie, g * a_u * intensity, relief=int(S.u_channel[v]) not in echo)
            record[v][1] += got
            record[v][4] = slot
        changes[unit] = record
    induce(mind, changes, occurred, slot_of_channel)


def induce(mind, changes: Dict[int, Dict[int, list]], occurred, slot_of_channel) -> None:
    """Section 4 "up": a generalisation in the last focus acquires what its instance acquired.

    Each instance tie counts at most ``induct_cap`` (raw evidence), discounted by the expectation of the
    ABSTRACT relation. The evidence keeps the provenance class of the event that gave it to the instance.
    """
    S, cfg, cap = mind.S, mind.cfg, mind.cfg.induct_cap
    if cfg.reliability:
        judge_generalisations(mind, changes)
    if getattr(cfg, "voices", False):
        judge_voices(mind, changes)
    activation = {unit: a for unit, a, _ in mind.last_focus}
    missed = set()
    gens = list(mind.last_gens)
    if cfg.miss_scope == "gens":         # F7: every generalisation the instance is read from is judged by it
        from .read import generalisations
        listed = {(g, i) for g, i, _ in gens}
        for instance in changes:
            for general, e_abs in generalisations(mind, instance):
                if (general, instance) not in listed:
                    listed.add((general, instance))
                    gens.append((general, instance, e_abs))
    for general, instance, e_abs in gens:
        record = changes.get(instance)
        if record is None or e_abs <= 0.5:
            continue
        for v, (pos0, pos1, neg0, neg1, slot_pos, slot_neg) in record.items():
            up = (min(pos1, cap) - min(pos0, cap)) * e_abs
            down = (min(neg1, cap) - min(neg0, cap)) * e_abs
            if up > 0.0:
                add_evidence1(mind, W_POS, slot_pos, S.tie(general, v, EXPECT, 1), up)
            if down > 0.0 and not cfg.general_miss:
                tie = S.find_tie(general, v, EXPECT, 1)
                if tie >= 0:
                    add_evidence1(mind, W_NEG, slot_neg, tie, down)
        if cfg.general_miss and general not in missed and (general in activation or cfg.miss_scope == "gens"):
            missed.add(general)
            strength = activation[general] if general in activation else e_abs
            if not S.out.get((general, EXPECT, 1)):
                continue
            ids, dst = S.out_ids((general, EXPECT, 1))
            w_pos, w_neg = tie_mass(mind, ids)
            e = expectation(w_pos, w_neg, cfg.k)
            channel = S.u_channel[dst]
            present = _present_of(dst, occurred)
            miss = mind.knob["gain"] * strength * e * (1.0 - present)
            expected = (e >= cfg.neg_floor) if cfg.neg_floor > 0.0 else torch.ones_like(present, dtype=torch.bool)
            for ch, slot in slot_of_channel.items():
                on = (channel == ch) & expected
                if bool(on.any()):
                    add_evidence(mind, W_NEG, slot, ids[on], miss[on], all_expect=True)
                    S.use_ties(ids[on])


def judge_generalisations(mind, changes: Dict[int, Dict[int, list]]) -> None:
    """F9: the same rule applied to the mind's own generalisations.

    For a composed instance that has just lived a value v on a sense/act channel, every generalisation g it is
    read from is judged by the expectation it held for v before this moment: w+ += e(i->g) e(g->v),
    w- += e(i->g) (1 - e(g->v)) on the self-tie g -> g of form RELY (provenance "inferred"). Its expectation is
    how reliably g has predicted what its instances lived. Nothing says which classification fits which
    dimension; that is learned here, per generalisation, from the mind's own life.
    """
    from .read import generalisations, own_mass
    S, k = mind.S, mind.cfg.k
    for instance, record in changes.items():
        kind = int(S.u_kind[instance])
        if not kind & COMPOSED or kind & SLOTTED:
            continue
        lived = [v for v, r in record.items() if r[1] > r[0] and r[0] == 0.0 and r[2] == 0.0  # first time only
                 and int(S.u_channel[v]) >= 0
                 and mind.ch.kind(int(S.u_channel[v])) in ("sense", "act")]
        if not lived:
            continue
        judged = _judged(mind)
        for general, e_abs in generalisations(mind, instance):
            for v in lived:
                judged.add((general, instance, v))
                verdict = _verdict(mind, general, v)
                if verdict is None:
                    continue
                tie = S.tie(general, general, RELY, 0)
                add_evidence1(mind, W_POS, INFERRED, tie, e_abs * verdict[0])
                add_evidence1(mind, W_NEG, INFERRED, tie, e_abs * verdict[1])


def judge_voices(mind, changes: Dict[int, Dict[int, list]]) -> None:
    """F24: what comes next judges the voices that could speak for a context (learning how far to believe a voice).

    When the mind read a context (F11), every generalisation whose members had lived something in its slot was a
    voice for it; each voice's claim was noted: the value it expected most, with frequency f, and the kind of the
    voice (read.voice_kind: how many members spoke, how much they agreed). When that context then lives a value for
    the first time (its tie had no evidence before, so the prediction rested on the voices and not on its own past),
    a claim held if the voice expected that value above one half and most (f > 1/2, the core's measure of a holding
    prediction); otherwise it failed. Every voice is judged, not only the one that spoke: they were all present when
    the context was read, and judging only the one relied on would let what was chosen decide what is learned. Each
    claim is one experience (gain g) on the tie (kind -> holds) and on the tie (every voice -> holds) of the mind's
    own channel ``ses``, class sensed, logged like all evidence and retractable; ``read.voice_credibility`` reads
    them general -> specific. Nothing here names a word, a class or a domain, so what one life taught about voices
    is there when the mind meets the next.
    """
    from .read import VOICE_ALL, VOICE_CHANNEL
    spoken = getattr(mind, "voice", None)
    if not spoken:
        return
    S = mind.S
    for instance, record in changes.items():
        voice = spoken.get(instance)
        if voice is None:
            continue
        lived = [v for v, r in record.items() if r[1] > r[0] and r[0] == 0.0 and r[2] == 0.0
                 and int(S.u_channel[v]) >= 0 and mind.ch.kind(int(S.u_channel[v])) in ("sense", "act")]
        if not lived:
            continue
        channel = mind.add_channel(VOICE_CHANNEL, "knob")
        for kind, claim, claim_f in voice[1]:
            held = claim in lived and claim_f > 0.5
            for symbol in (VOICE_ALL, kind):
                tie = S.tie(mind._atom(channel, symbol), mind.holds, EXPECT, 0)
                add_evidence1(mind, W_POS if held else W_NEG, SENSED, tie, mind.knob["gain"])


def _verdict(mind, general: int, v: int, leave_out: float = 0.0):
    """How generalisation g fared on a lived value v: (w+, w-) for its RELY tie, or None if g had no opinion.

    g's opinion on the channel of v is its sense/act ties on that channel; with none g is not tested.
    verdict "opinion" (F10): tested with the frequency it gave v, f(g -> v) (0 if v never followed g), weighted
    by the conviction of its strongest tie on that channel: w+ = c f, w- = c (1 - f).
    verdict "share" (F10b, measured worse, NOTES.md): tested with the share of its positive evidence on that
    channel that went to v, weighted by the conviction of that evidence.
    ``leave_out`` raw positive evidence on g -> v is removed first (what g received from this very instance).
    """
    S, k = mind.S, mind.cfg.k
    channel = int(S.u_channel[v])
    ties = [t for t in S.out.get((general, EXPECT, 1), []) if int(S.u_channel[int(S.t_dst[t])]) == channel]
    if not ties:
        return None
    best = f = total = mine = 0.0
    for t in ties:
        w_pos, w_neg = float(S.t_w[W_POS, t].sum()), float(S.t_w[W_NEG, t].sum())
        if int(S.t_dst[t]) == v:
            w_pos = max(0.0, w_pos - leave_out)
            f = w_pos / (w_pos + w_neg) if w_pos + w_neg > 0.0 else 0.0
            mine = w_pos
        total += w_pos
        best = max(best, (w_pos + w_neg) / (w_pos + w_neg + k))
    if mind.cfg.verdict == "share":
        if total <= 0.0:
            return None
        p, c = mine / total, total / (total + k)
        return c * p, c * (1.0 - p)
    if best <= 0.0:
        return None
    return best * f, best * (1.0 - f)


def _judged(mind) -> set:
    """(generalisation, instance, value) triples already judged: every lived value tests a generalisation once."""
    if not hasattr(mind, "judged"):
        mind.judged = set()
    return mind.judged


def reflect(mind, budget: int = 1) -> None:
    """F10, idle moments: re-read lived instances with the generalisations the mind has NOW.

    A generalisation that was formed after its instances were lived (a class is found only once enough members
    were met) was never tested on them. Here, while nothing comes from outside, the mind takes the next lived
    instance (round robin over composed units with own evidence on a sense/act value) and judges every present
    generalisation g of it on each value v the instance lived, once per (g, i, v), with the expectation g holds for
    v LEAVING OUT what g received from this instance (induction gives at most e(i->g) x min(W_own+, cap) of raw
    positive evidence per instance; that much is subtracted), so that g is tested as a prediction, not as an echo.
    The evidence goes to the RELY self-tie, provenance "inferred", logged in this idle moment.
    """
    from .read import generalisations
    S, cfg = mind.S, mind.cfg
    judged = _judged(mind)
    n = S.n_units
    cursor = getattr(mind, "reflect_cursor", -1)
    done = tried = 0
    while done < budget and tried < n:
        cursor = (cursor + 1) % n
        tried += 1
        kind = int(S.u_kind[cursor])
        if not kind & COMPOSED or kind & SLOTTED or not bool(S.u_alive[cursor]):
            continue
        ties = S.out.get((cursor, EXPECT, 1), [])
        lived = {}
        for tie in ties:
            v = int(S.t_dst[tie])
            channel = int(S.u_channel[v])
            if channel < 0 or mind.ch.kind(channel) not in ("sense", "act"):
                continue
            raw = float(S.t_w[W_POS, tie].sum())
            if raw > 0.0:
                lived[v] = raw
        if not lived:
            continue
        done += 1
        for general, e_abs in generalisations(mind, cursor):
            rely = None
            for v, raw_own in lived.items():
                if (general, cursor, v) in judged:
                    continue
                judged.add((general, cursor, v))
                verdict = _verdict(mind, general, v, leave_out=e_abs * min(raw_own, cfg.induct_cap))
                if verdict is None:
                    continue
                if rely is None:
                    rely = S.tie(general, general, RELY, 0)
                add_evidence1(mind, W_POS, INFERRED, rely, e_abs * verdict[0])
                add_evidence1(mind, W_NEG, INFERRED, rely, e_abs * verdict[1])
    mind.reflect_cursor = cursor


def learn_value(mind, percepts) -> None:
    """L5, consequences: a signal on a channel of kind "value" (good or bad for this mind, as its body adapter
    defines) is evidence for every step of the episode that led to it: each composed context in the episode's
    trace gets EXPECT lag 2 ("later in the same episode") to the value unit. The trace is the only memory a
    consequence needs; nothing names what is good or bad here."""
    S, g = mind.S, mind.knob["gain"]
    trace = getattr(mind, "trace", [])
    if not trace:
        return
    for unit, intensity, slot, channel in percepts:
        if intensity <= 0.0 or mind.ch.kind(channel) != "value":
            continue
        for step_unit in dict.fromkeys(trace):
            add_evidence1(mind, W_POS, slot, S.tie(step_unit, unit, EXPECT, 2), g * intensity)


def facet_set(mind, unit: int) -> set:
    """The facets an item has been perceived with (its lag-0 ties to units of facet channels, positive evidence).
    Cached per unit until its number of lag-0 ties changes (evidence on a facet tie only grows)."""
    S = mind.S
    ties = S.out.get((unit, EXPECT, 0), [])
    cache = mind.__dict__.setdefault("_facet_cache", {})
    hit = cache.get(unit)
    if hit is not None and hit[0] == len(ties):
        return hit[1]
    out = set()
    for tie in ties:
        dst = int(S.t_dst[tie])
        channel = int(S.u_channel[dst])
        if channel >= 0 and mind.ch.kind(channel) == "facet" and float(S.t_w[W_POS, tie].sum()) > 0.0:
            out.add(dst)
    cache[unit] = (len(ties), out)
    return out


def _facet_weight(mind, facet: int) -> float:
    """How telling a facet is: 1 / the number of items it was perceived with (a facet everything has tells nothing)."""
    return 1.0 / max(1, len(mind.S.out.get((facet, EXPECT, 0), [])))


def facet_similarity(mind, a: int, b: int, fa: set = None) -> float:
    """Weighted Jaccard overlap of the facet sets of two items, each facet weighted by how telling it is (1 / the
    number of items it was perceived with). A facet no other item has ever had cannot be shared by anything, so it
    says nothing about resemblance and is left out of the comparison. 0 when either has no comparable facet."""
    fa = facet_set(mind, a) if fa is None else fa
    fb = facet_set(mind, b)
    degree = {f: len(mind.S.out.get((f, EXPECT, 0), [])) for f in fa | fb}
    fa = {f for f in fa if degree[f] > 1}
    fb = {f for f in fb if degree[f] > 1}
    if not fa or not fb:
        return 0.0
    w = {f: 1.0 / degree[f] for f in fa | fb}
    union = sum(w.values())
    return sum(w[f] for f in fa & fb) / union if union > 0 else 0.0


def join_by_resemblance(mind, unit: int, slot: int, neighbours: int = 20) -> None:
    """F18: a newly perceived item joins the categories of the items it resembles. Candidates are the items that
    share facets with it; for each category of those items the similarity of the nearest member is the evidence
    (provenance inferred) that the new item is an instance of it. Nothing is said about what the facets mean."""
    from .form import categories_of
    S = mind.S
    mine = facet_set(mind, unit)
    if not mine:
        return
    shared = {}
    for f in mine:
        for tie in S.out.get((f, EXPECT, 0), []):
            x = int(S.t_dst[tie])
            if x != unit and int(S.u_kind[x]) & ATOMIC:
                shared[x] = shared.get(x, 0) + 1
    best = {}
    for x, _ in sorted(shared.items(), key=lambda kv: -kv[1])[:neighbours]:
        sim = facet_similarity(mind, unit, x, mine)
        if sim <= 0.0:
            continue
        for category, e in categories_of(mind, x):
            if e > 0.5 and sim > best.get(category, 0.0):
                best[category] = sim
    from .events import INFERRED
    for category, sim in best.items():
        if S.find_tie(unit, category, ABSTRACT, 0) < 0:
            add_evidence1(mind, W_POS, INFERRED, S.tie(unit, category, ABSTRACT, 0), sim)


def learn_facets(mind, items: List[int], facets: List[Tuple[int, float, int]], slot: int) -> None:
    """Facets (parts or features of an item on a channel of kind "facet", e.g. subword pieces of a word, later
    features of a seen object) hold together with the items of the same moment: EXPECT lag 0 in both directions,
    positive evidence g x intensity. They never enter the episode or the focus; the context learns to expect them
    in the next moment like any percept (learn_follow), so a context can expect a suffix of a word it never met."""
    g = mind.knob["gain"]
    for item in items:
        new = float(mind.S.u_use[item]) <= 1.0              # met for the first time (use counts this moment)
        for facet, intensity, _ in facets:
            add_evidence1(mind, W_POS, slot, mind.S.tie(item, facet, EXPECT, 0), g * intensity)
            add_evidence1(mind, W_POS, slot, mind.S.tie(facet, item, EXPECT, 0), g * intensity)
        if new and mind.cfg.resemblance:
            join_by_resemblance(mind, item, slot)


def learn_together(mind, focus: List[Tuple[int, float, bool]], slot: int) -> None:
    """EXPECT lag 0 among the grounded units of the focus; expected but absent partners get negative evidence."""
    S, cfg = mind.S, mind.cfg
    g = mind.knob["gain"]
    present = {unit: a for unit, a, grounded in focus if grounded and a > 0.0}
    for unit, a_u in present.items():
        if S.out.get((unit, EXPECT, 0)):
            ids, dst = S.out_ids((unit, EXPECT, 0))
            here = torch.zeros(ids.numel(), dtype=F64)     # activation of each partner if present (few are)
            for d, a_d in present.items():
                here[dst == d] = a_d
            w_pos, w_neg = tie_mass(mind, ids)
            e = expectation(w_pos, w_neg, cfg.k)
            weight = e if cfg.neg_weight == "e" else torch.ones_like(e)
            keep = ((e >= cfg.neg_floor) | (here > 0.0)) if cfg.neg_floor > 0.0 else torch.ones_like(here, dtype=torch.bool)
            add_evidence(mind, W_NEG, slot, ids[keep], (g * a_u * weight * (1.0 - here))[keep], all_expect=True)
            S.use_ties(ids[keep])
        for other, a_v in present.items():
            if other != unit:
                add_evidence1(mind, W_POS, mind.prov_of.get(other, SENSED), S.tie(unit, other, EXPECT, 0),
                              g * a_u * a_v)
