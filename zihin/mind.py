# -*- coding: utf-8 -*-
"""Zihin: one moment of experience (design section 2) on the one substance, with the one rule.

``Mind.step(events)`` lives one moment:

1. PERCEIVE  every event activates its atomic unit (created when the symbol is new); the items of the last
   focus stay active with decay.
2. RELATE    settle for at most T iterations: recognise or recruit the composed unit of the episode, activate
   categories up through ABSTRACT, let frames stand in, spread through EXPECT(lag 0), keep the F most active
   units (the focus).
3. PREDICT   expectations for the next moment through EXPECT(lag 1).
4. ACT       act units accumulate their expectation over the iterations; one that crosses the threshold is
   emitted, otherwise the mind stays silent. There is no softmax and no argmax over a vocabulary.
5. LEARN     happens at the START of the next moment for the previous focus (``learn.learn_follow``).
6. FEEL      surprise, relief, load and conflict are emitted as events of the mind's own ``feel`` channel and
   enter the next moment like any other event.

Nothing in this file knows a domain: it sees events, units, ties and evidence only.
"""
from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
import torch

from . import act as act_mod
from .events import ACTED, IMAGINED, IMITATED, N_PROV, PROVENANCE, REFERENCE, SENSED, Channels, Event
from .evidence import conviction, expectation
from .form import (READ_CACHES, SMALL, categories_of, compose, confirm_parts, expectations, kind_masks, masses,
                   masses_py, maybe_chunk, mv_view, np_view, parts_of, recognise)
from .learn import learn_facets, learn_follow, learn_together, learn_value, reflect, tie_mass
from .read import conflict_of, predictions, total_drive
from .store import ABSTRACT, ATOMIC, EXPECT, F64, I64, PART, SLOTTED, W_NEG, W_POS, Substance

FEELINGS = ("surprise", "relief", "load", "conflict")
SELF_CHANNEL, SELF_SYMBOL = "benlik", "ben"
INNER_CHANNEL, NOT_KNOWING = "ic", "bilmemek"     # L7: not knowing, perceived as an inner event


@dataclass
class Config:
    """Knobs and switches. Defaults follow the design text; every deviation is a named switch."""

    F: int = 7                      # focus size (design default)
    T: int = 6                      # settling iterations (design: T <= 6)
    k: float = 1.0                  # conviction constant (design: k = 1)
    gain: float = 1.0               # plasticity gain g
    theta: float = 1.0              # act threshold: one unit of accumulated net expectation
    decay: Optional[float] = None   # context decay per moment; None = 1 - 1/F (time constant = focus span)
    lam: float = 0.5                # support factor of section 4 (design: 0.5)
    support: str = "raw"            # "raw" = as written; "capped", "nested" = changed; "none" = ablation
    predict: str = "votes"          # "votes" = as written, "specific" = changed
    neg_floor: float = 0.0          # F14: a failed prediction is recorded only if it was expected (e >= floor);
                                    # 0 = every tie of a present unit, as written
    neg_weight: str = "e"           # failed expectation weighted by e (as written) or by 1 ("one")
    induct_cap: float = 1.0         # each instance counts at most one experience in induction
    general_miss: bool = False      # changed: a generalisation in the focus gets negative evidence for what
                                    # it expected and did not occur (the ordinary rule of step 5)
    miss_scope: str = "focus"       # which generalisations get that negative evidence: those in the focus
                                    # ("focus", changed) or every generalisation that received positive
                                    # evidence from the episode ("gens", symmetric; see NOTES.md F7)
    reliability: bool = False       # F9: every generalisation an instance is read from is judged by what the
                                    # instance then lived (evidence on its RELY self-tie); read with "reliable"
    reflect: int = 0                # F10: in each idle moment re-read this many lived instances with the
                                    # generalisations the mind has NOW (needs reliability)
    verdict: str = "opinion"        # F10 "opinion" (default) or F10b "share"; see learn._verdict
    proven_only: bool = False       # F10b: only generalisations with reliability > 1/2 speak (measured worse)
    next_only_present: bool = False # F13: EXPECT lag 1 is learned only by the previous context and the previous
                                    # item, not by older units still carried in the focus (no skip-gram smear)
    race: bool = False              # F19: the acts of one channel race on their advantage over the strongest rival
    resemblance: bool = False       # F18: a new item joins the categories of what it resembles (shared facets), and
                                    # members that resemble the item read speak louder
    backoff: bool = False           # F16: an utterance under way continues along the lived sequences of its end
    situations: bool = False        # F23: an utterance never lived as a whole is met by the lived ones most like it
    explore: bool = False           # hesitation ends in a try: the acceptable alternative whose consequences are least
                                    # known is chosen (act._explore)
    voices: bool = False            # F24: how far a kind of voice (how many members speak, how much they agree) can be
                                    # believed is learned from what then came, and weighs the F11 choice (support
                                    # "members"); the evidence is on the mind's own channel, so it carries to new domains
    born_failures: bool = False     # F25: a lag-1 tie born now starts with the failures a full record would hold: the
                                    # earlier moments its source was followed by something else on the target's
                                    # channel, per class (learn.learn_follow); off keeps every earlier result
    substitution: bool = False      # Substitution Evidence (zihin/anlam.py): y standing where x was expected, with x's
                                    # consequences after it, is inferred evidence that y belongs to x's categories
    exact_relations: bool = False   # F28 (Exact Relations): a frame's relation (SELF, NEXT, START) counts for a member
                                    # only where it names the very value the member lived; the value itself always counts
    exclusive: bool = False         # F27 (Lived Exclusion): what an instance was lived to be followed by on a channel
                                    # counts against what a generalisation proposes there that it never went on with
    corroborate: bool = False       # F30 (Corroboration): the claims of the frame that speaks are revised with what the
                                    # lived situations it did not count (F23) say about the same values (NARS revision)
    lived_situations: bool = False  # F23 reads only lived situations: a frame (a generalisation, whose own ties hold its
                                    # members' evidence as literal values) does not vote as a situation of its own
    company: bool = False           # F33 (Relational Memory): evidence is noted with the known person present, and what
                                    # is recalled depends on who is present (zihin/iliski.py)
    replies: bool = False           # F34 (Reply Relation): a frame's value may be what its filler, said on its own,
                                    # was answered with; with it several places after a prefix may be bound (F32)
    paradigmatic: bool = False      # F31 (Paradigmatic Relation): a frame's value may be what stands where the filler
                                    # stands (ALIKE: the item most interchangeable with it, not a rival filler of the
                                    # frame's place), as SELF, NEXT and START are other relations to the filler
    bind_relations: bool = False    # F32 (Relational Binding, with paradigmatic): a context whose last item is the
                                    # stand-in of its prefix's frame filler (F31) is read through each member's instance
                                    # followed by its own stand-in
    analogy: bool = False           # F29 (Analogy by Interchangeability): what a frame's members lived carries over to
                                    # the read instance as far as its filler is interchangeable with them (NARS analogy),
                                    # interchangeability read from the lived places both have filled
    starts: bool = False            # F26 (Sequence Starts): how sequences that began with a word went on (read from the
                                    # composed units of two items) is a relation of frames (START) and the reading of a
                                    # one-item utterance (F16), apart from what followed the word anywhere
    rivals: bool = False            # F25 (b): the frequency of a lag-1 tie is read as its share of the positive mass of
                                    # its source's ties on the same channel (read.own_mass); off as before
    bind_repeats: bool = False      # F12: a unit occurring at several positions of a tuple is one slot (variable)
    lag0: bool = True               # EXPECT(lag 0) learning and spreading among atomic units
    evoke: str = "activation"       # focus ranking: by activation alone (as written) or "grounded_first"
    feel_focus: bool = True         # feelings compete for the focus like any event (as written)
    select: str = "activation"      # which generalisations win a place: by "activation" (as written) or
                                    # "specific": what is present first, then the most specific ones
    body_ties: int = 1_000_000      # tie budget of the body; load = ties / budget
    slots: bool = True              # slot / frame induction
    typed_frames: bool = True       # existing categories stand in for their instances
    chunking: bool = False          # fuse reliable successions into one item
    e_chunk: float = 0.9
    m_chunk: float = 5.0
    learn_trust: bool = True        # direct experience judges the other classes
    trust_fixed: Dict[int, float] = field(default_factory=dict)       # class -> constant trust (tests)
    trust_prior: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # class -> innate (w+, w-)
    act_expect_relief: bool = True  # an own act counts as an expectation of relief (changed; see NOTES.md)
    clear_on_pause: bool = False    # changed: a silent moment releases the carried context
    imagine: bool = False           # section 6, simple first version
    H: int = 3
    iota: float = 0.5
    reality: bool = True            # reality scoring of imagined conclusions (False = ablation)
    knobs: bool = False             # section 8, simple first version
    babble_every: int = 0           # with knobs: emit one knob act every N idle moments (exploration)
    keep_log: bool = True
    log_path: Optional[str] = None  # keep the experience log on disk at this path (memory stays small)
    undo_depth: int = 1             # how many of the latest moments keep their working state for undo

    @classmethod
    def changed(cls, **over) -> "Config":
        """The design with the smallest principled changes the prototype found necessary (NOTES.md)."""
        base = dict(support="nested", predict="specific", evoke="grounded_first", feel_focus=False,
                    clear_on_pause=True, general_miss=True, select="specific")
        base.update(over)
        return cls(**base)

    @classmethod
    def reliable(cls, **over) -> "Config":
        """changed + F9: generalisations earn reliability; the most reliable one speaks for an instance."""
        base = dict(reliability=True, support="reliable")
        base.update(over)
        return cls.changed(**base)

    @classmethod
    def reflective(cls, **over) -> "Config":
        """reliable + F10: idle moments re-read lived instances with the present generalisations."""
        base = dict(reflect=1)
        base.update(over)
        return cls.reliable(**base)

    @classmethod
    def symmetric(cls, **over) -> "Config":
        """changed + F7: a generalisation is judged by every episode it learns from (NOTES.md)."""
        base = dict(miss_scope="gens")
        base.update(over)
        return cls.changed(**base)


class Mind:
    """A mind that learns while it lives. All knowledge is in ``self.S`` (units, ties, evidence)."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg if cfg is not None else Config()
        self.S = Substance(n_slots=N_PROV, keep_log=self.cfg.keep_log)
        self.S.bind_repeats = self.cfg.bind_repeats
        if self.cfg.keep_log and self.cfg.log_path:
            from .store import DiskLog
            self.S.log = DiskLog(self.cfg.log_path, keep=max(4, self.cfg.undo_depth + 2))
        self.ch = Channels()
        self.t = 0
        self.knob = {"F": self.cfg.F, "theta": self.cfg.theta, "gain": self.cfg.gain}
        S = self.S
        S.begin(0, [])
        source, feel = self.ch.add("source", "source"), self.ch.add("feel", "feel")
        self.holds = self._atom(source, "holds")
        self.source_unit = [self._atom(source, name) for name in PROVENANCE]
        self.source_tie = [-1 if p in REFERENCE else S.tie(self.source_unit[p], self.holds, EXPECT, 0)
                           for p in range(N_PROV)]
        for p, (w_pos, w_neg) in self.cfg.trust_prior.items():
            S.add1(W_POS, SENSED, self.source_tie[p], w_pos)
            S.add1(W_NEG, SENSED, self.source_tie[p], w_neg)
        self.feel_unit = {name: self._atom(feel, name) for name in FEELINGS}
        self.add_channel(SELF_CHANNEL, "context")          # the self: present whenever the mind itself acts or thinks
        self.add_channel(INNER_CHANNEL, "sense")           # inner events: the mind perceives its own not knowing
        S.end()
        # transient state (working state of the present moment; not knowledge)
        self.focus: List[Tuple[int, float, bool]] = []
        self.last_focus: List[Tuple[int, float, bool]] = []
        self.gens: List[Tuple[int, int, float]] = []
        self.last_gens: List[Tuple[int, int, float]] = []
        self.carry: Dict[int, float] = {}
        self.episode: List[int] = []
        self.trace: List[int] = []                   # composed contexts of this episode (L5)
        self.value_sign: Dict[Tuple[str, Hashable], float] = {}  # set by the adapter: which value signal is good/bad
        self.context: Optional[int] = None
        self.pending: List[Event] = []
        self.predicted: Dict[int, float] = {}
        self.act_hope: Dict[int, float] = {}
        self.hoping: Optional[Tuple[int, float]] = None
        self.chunked: List[int] = []
        self.prov_of: Dict[int, int] = {}
        self.idle = 0
        self.babbled = 0
        self.unknown = 0.0                           # L7: how much it did not know when the world last called on it
        self.witnesses = []                          # Substitution Evidence: (expected, occurred) pairs to be judged
        self.episode_start = 0                       # the moment the present run of items began (after a pause)
        self.company = None                          # F33: the known person present (set by the session)
        self.voice: Dict[int, tuple] = {}            # F24: context -> (voice that spoke, every voice heard), as
                                                     # (kind, claim, f) each; judged by what comes next
        self.imagined = 0
        self._gained: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.trust = self._trust()
        self.felt: Dict[str, float] = {name: 0.0 for name in FEELINGS}
        self.felt["drive"] = 0.0

    # ------------------------------------------------------------------ channels, units, trust
    def add_channel(self, name: str, kind: str) -> int:
        return self.ch.add(name, kind)

    def _atom(self, channel: int, symbol: Hashable, create: bool = True) -> int:
        index = self.ch.symbol_index(channel, symbol, create)
        if index < 0:
            return -1
        unit = self.S.atom_index.get((channel, index), -1)
        if unit < 0 and create:
            unit = self.S.new_unit(ATOMIC, channel, index)
        return unit

    def unit_of(self, channel: str, symbol: Hashable) -> int:
        """The atomic unit of a symbol, or -1 when the mind never met it. Never creates."""
        if not self.ch.has(channel):
            return -1
        return self._atom(self.ch.id(channel), symbol, create=False)

    def symbol_of(self, unit: int) -> Tuple[str, Hashable]:
        channel = int(self.S.u_channel[unit])
        return self.ch.names[channel], self.ch.symbol(channel, int(self.S.u_symbol[unit]))

    def _trust(self) -> torch.Tensor:
        """Trust per provenance class: 1 for direct experience, else the expectation of (source -> holds)."""
        trust = [1.0] * N_PROV                              # set as plain numbers, made a tensor once
        for p in range(N_PROV):
            if p in REFERENCE:
                continue
            if p in self.cfg.trust_fixed:
                trust[p] = float(self.cfg.trust_fixed[p])
                continue
            tie = self.source_tie[p]
            trust[p] = expectation(self.S.cell(W_POS, tie, SENSED), self.S.cell(W_NEG, tie, SENSED),
                                   self.cfg.k)
        return torch.tensor(trust, dtype=F64)

    # ------------------------------------------------------------------ one moment
    def step(self, events: Iterable[Event] = ()) -> List[Event]:
        """Live one moment with the given outside events; returns the acts the mind emits."""
        S, cfg = self.S, self.cfg
        undo = self._snapshot()
        self.__dict__.pop("_between_memo", None)           # what was read between moments is read afresh
        self.t += 1
        S.begin(self.t, self.ch.sizes())
        self.trust = self._trust()
        self._gained = []
        internal, self.pending = self.pending, []
        percepts: List[Tuple[int, float, int, int]] = []
        items: List[Tuple[int, int]] = []
        for event in list(internal) + list(events):
            channel = self.ch.add(event.channel, "sense") if not self.ch.has(event.channel) \
                else self.ch.id(event.channel)
            unit = self._atom(channel, event.symbol)
            percepts.append((unit, float(event.intensity), int(event.prov), channel))
            kind = self.ch.kind(channel)
            if kind in ("sense", "act") and event.intensity > 0:
                items.append((unit, int(event.prov)))
            if kind == "knob" and event.prov == ACTED and cfg.knobs:
                act_mod.apply_knob(self, event.symbol)
        if self.ch.has(SELF_CHANNEL) and any(prov in (ACTED, IMAGINED) for _, prov in items):
            # the mind perceives itself as the one who says or thinks this (L8): "ben" accompanies the moment
            channel = self.ch.id(SELF_CHANNEL)
            own = ACTED if any(prov == ACTED for _, prov in items) else IMAGINED   # speaking aloud, or thinking
            percepts.append((self._atom(channel, SELF_SYMBOL), 1.0, own, channel))
        learn_follow(self, percepts)                       # step 5 for the previous moment
        learn_value(self, percepts)                        # L5: a value signal reaches every step of the episode
        if cfg.substitution:
            from .anlam import substitution
            substitution(self, items)                      # meaning: y where x was expected, judged by what follows
        if getattr(self, "voice", None):
            self.voice = {}                                # F24: the voices of the last moment have been judged
        surprise = self._surprise(percepts)
        slot = items[0][1] if items else SENSED
        if items:
            self.idle = 0
            if not self.episode:
                self.episode_start = self.t                # a run of items begins after a pause (F29 reads what was
            for unit, _ in items:                          # lived before it)
                self.episode.append(unit)
                maybe_chunk(self, slot)
            self.episode = self.episode[-self.knob["F"]:]
        else:
            self.idle += 1
            self.episode = []
            if cfg.clear_on_pause:
                self.carry = {}
        self._settle(percepts, slot, bool(items))          # steps 1 and 2
        if items and self.context is not None and (not self.trace or self.trace[-1] != self.context):
            self.trace.append(self.context)                # the episode's trace (for consequences, L5)
        elif not items:
            self.trace = []
        if cfg.lag0:
            kinds = mv_view(self, "u_kind")
            learn_together(self, [f for f in self.focus if kinds[f[0]] & ATOMIC], slot)
        facets = [(u, i, c) for u, i, p, c in percepts if i > 0 and self.ch.kind(c) in ("facet", "context")]
        if facets and items:                               # facets of this moment go with its items (lag 0)
            learn_facets(self, [u for u, _ in items], facets, slot)
        listening = self.knob["theta"] >= 1e8              # spoken to: it cannot speak now
        self._read_memo = {}                               # nothing is written while predicting, deciding, feeling
        try:                                               # (read.py reads each unit once in this stretch)
            self.predicted = predictions(self, "own" if listening else None)   # step 3 (cheap while listening)
            acts = [] if listening else act_mod.decide(self)   # step 4: no decision to take while listening
            self._unknowing(items, acts, listening)
            imagine = cfg.imagine and not items
            rethink = cfg.reflect and cfg.reliability and not items
            babble = cfg.knobs and not items and not acts
            if imagine or rethink or babble:
                self._read_memo = None                     # these write (or add a channel): read afresh after
            if imagine:
                act_mod.imagine(self)
            if rethink:
                reflect(self, cfg.reflect)
            if babble:
                acts = act_mod.babble(self)
            for event in acts:
                self.pending.append(event)
            self._feel(surprise)                           # step 6
        finally:
            self._read_memo = None
        if "_personal_moment" in self.__dict__ or "personal_log" in self.__dict__:
            from .iliski import end_moment                 # F33: this moment's notes, kept for undo and retraction
            end_moment(self)
        entry = S.end()
        entry.undo = undo
        previous = S.log.get(self.t - cfg.undo_depth)
        if previous is not None:
            previous.undo = None
        self.last_focus, self.last_gens = self.focus, self.gens
        return acts

    def _unknowing(self, items, acts, listening: bool) -> None:
        """L7: not knowing, felt. When the world has just done something (an outer event, such as the end of someone's
        turn) and the mind, free to answer, decides nothing, the amount it does not know is kept:
        u = 1 - 2 max(0, best - 1/2), best = the highest expectation of any act; when the best would have been said
        had it no rival (2 (best - 1/2) T >= theta), the silence is hesitation between answers and nothing is kept. If
        the next moment passes in silence
        with nothing said, the mind perceives that state as an inner event of intensity u, alone in its moment, so
        what one says (or does) then can be lived and learned like anything else. The measurement is innate; how
        not knowing is expressed is not."""
        if not self.ch.has(INNER_CHANNEL):
            return
        if listening:
            if items:
                self.unknown = 0.0
            return
        inner = self.ch.id(INNER_CHANNEL)
        acted = any(self.ch.has(e.channel) and self.ch.kind(self.ch.id(e.channel)) == "act" for e in acts)
        channels = mv_view(self, "u_channel")
        outer = any(channels[u] != inner and self.ch.kind(channels[u]) == "sense" for u, _ in items)
        if outer:
            act = kind_masks(self)[5]
            best = max((e for u, e in self.predicted.items() if act[channels[u]]), default=0.5)
            # a best answer that would have been said were it alone is hesitation between rivals, not not knowing
            alone = 2.0 * (best - 0.5) * self.cfg.T >= self.knob["theta"]
            self.unknown = 0.0 if acted or alone else 1.0 - 2.0 * max(0.0, best - 0.5)
        elif not items and not acted and getattr(self, "unknown", 0.0) > 0.0:
            # booked as the mind's own percept (acted: certain, self-produced), not as sensed experience of the world,
            # so its ties never judge what it was told (events.JUDGES)
            self.pending.append(Event(INNER_CHANNEL, NOT_KNOWING, ACTED, self.unknown))
            self.unknown = 0.0
        else:
            self.unknown = 0.0

    def _surprise(self, percepts) -> float:
        """Mean (1 - e) of what occurred plus mean e of what was expected on a reporting channel and failed."""
        S = self.S
        occurred = {u for u, a, _, ch in percepts if a > 0 and self.ch.kind(ch) in ("sense", "act")}
        if not occurred:
            return 0.0
        unit_channel = mv_view(self, "u_channel")
        channels = {unit_channel[u] for u in occurred}
        unexpected = sum(1.0 - self.predicted.get(u, 0.5) for u in occurred) / len(occurred)
        failed = [e for u, e in self.predicted.items()
                  if e > 0.5 and u not in occurred and unit_channel[u] in channels]
        return min(1.0, unexpected + (sum(failed) / len(failed) if failed else 0.0))

    def _settle(self, percepts, slot: int, has_items: bool) -> None:
        S, cfg = self.S, self.cfg
        F = self.knob["F"]
        decay = cfg.decay if cfg.decay is not None else 1.0 - 1.0 / F
        base: Dict[int, float] = {u: decay * a for u, a in self.carry.items()}
        for unit, intensity, prov, channel in percepts:
            kind = self.ch.kind(channel)
            if kind in ("facet", "context", "value"):      # facets, context and value signals do not compete
                continue
            if intensity > 0 and (cfg.feel_focus or kind != "feel"):
                base[unit] = max(base.get(unit, 0.0), min(1.0, intensity))
                self.prov_of[unit] = prov
        context = recognise(self, slot) if has_items else None
        parts = tuple(self.episode)
        if context is not None:
            base[context] = 1.0                            # recognised now: the whole tuple is present
            self.prov_of[context] = slot
        self.context = context if context is not None else (parts[0] if len(parts) == 1 else None)
        if self.hoping is not None and has_items:          # the echo of an own act has just arrived
            act_unit, hope = self.hoping
            if parts and parts[-1] == act_unit and self.context is not None:
                self.act_hope[self.context] = hope
                if cfg.act_expect_relief:
                    S.tie(self.context, self.feel_unit["relief"], EXPECT, 1)
            self.hoping = None
        first = cfg.evoke == "grounded_first"
        fresh = {u for u, a, _, _ in percepts if a > 0}
        if context is not None:
            fresh.add(context)
        ranked = self._present_fills(base, fresh, F) if first and cfg.select == "specific" else None
        if ranked is not None:                             # nothing derived can win a place (see _present_fills)
            gens: Dict[int, Tuple[int, float]] = {}
            virtual: Dict[Tuple[int, ...], Tuple[float, int, float]] = {}
            grounded = dict(base)
        else:
            ranked, gens, virtual, grounded = self._rank(base, parts, context, fresh, F)
        focus: List[Tuple[int, float, bool]] = []
        self.gens = []
        for key, a in ranked:
            if isinstance(key, tuple):                     # a frame that won a place is recruited now
                a_frame, instance, e = virtual[key]
                key, _ = compose(self, key, SLOTTED)
                gens[key] = (instance, e)
                grounded[key] = a_frame
            if key in gens:
                self.gens.append((key, gens[key][0], gens[key][1]))
                if mv_view(self, "u_kind")[key] & SLOTTED:
                    confirm_parts(self, key, slot)
            focus.append((key, a, key in grounded))
        self.focus = focus
        self.carry = {u: a for u, a, _ in focus if u in base}
        self.act_hope = {u: h for u, h in self.act_hope.items() if u in self.carry}
        self.prov_of = {u: p for u, p in self.prov_of.items() if u in self.carry}
        S.use_units(torch.tensor([u for u, _, _ in focus], dtype=I64))

    def _present_fills(self, base, fresh, F):
        """With grounded_first and select = "specific", what is present ranks before every generalisation and
        every unit called to mind. When no unit of ``base`` can be a generalisation of this moment (none is the
        target of an ABSTRACT tie, i.e. a category, and none is composed with such a target among its parts, i.e. a
        frame) and at least F of them are active, the focus is the F first of them and the generalisations need
        not be read: returns that ranking (as ``_rank`` would give it), else None."""
        S = self.S
        inc, out = S.inc, S.out
        scored = []
        for unit, a in base.items():
            if (unit, ABSTRACT) in inc:
                return None
            if (unit, PART, -1) in out and any((p, ABSTRACT) in inc for p in parts_of(self, unit)):
                return None
            if a > 0.0:
                scored.append(((0, 0, 0, 0), -a, 0 if unit in fresh else 1, 0, (unit,), unit))
        if len(scored) < F:
            return None
        return [(key, -neg) for _, neg, _, _, _, key in heapq.nsmallest(F, scored)]

    def _rank(self, base, parts, context, fresh, F):
        """Generalisations of what is present (categories, frames), their order, and the F winners after the
        EXPECT(lag 0) spreading. Returns (ranked, gens, virtual, grounded)."""
        S, cfg = self.S, self.cfg
        # derived activation: categories of what is present, frames in which a category stands in
        derived: Dict[int, float] = {}
        gens: Dict[int, Tuple[int, float]] = {}
        virtual: Dict[Tuple[int, ...], Tuple[float, int, float]] = {}
        cats = {u: [(c, e) for c, e in categories_of(self, u) if e > 0.5] for u in base}
        bare: List[tuple] = []                             # (instance, extension size, -activation, key)
        framed: List[tuple] = []                           # (position, extension size, -activation, key)
        size: Dict[int, int] = {}                          # extension size of each category met here
        for unit, a_u in base.items():
            for category, e in cats[unit]:
                derived[category] = min(1.0, derived.get(category, 0.0) + a_u * (2.0 * e - 1.0))
                if category not in gens or e > gens[category][1]:
                    gens[category] = (unit, e)
                if category not in size:
                    size[category] = len(S.inc.get((category, ABSTRACT), ()))
                bare.append((unit, size[category], -derived[category], (0, category), category))
        if context is not None:
            for pos, part in enumerate(parts):
                part_cats = cats.get(part)
                if not part_cats:
                    continue
                if cfg.bind_repeats:
                    if part in parts[:pos]:
                        continue                           # the slot of this unit was built at its first position
                    where = [j for j, q in enumerate(parts) if q == part]
                others = sum(base.get(p, 0.0) for j, p in enumerate(parts) if j != pos)
                a_part = base.get(part, 0.0)
                for category, e in part_cats:
                    a_frame = (others + a_part * (2.0 * e - 1.0)) / len(parts)
                    if cfg.bind_repeats:
                        filled = list(parts)
                        for j in where:
                            filled[j] = category
                        frame_parts = tuple(filled)
                    else:
                        frame_parts = parts[:pos] + (category,) + parts[pos + 1:]
                    frame = S.tuple_index.get(frame_parts)
                    if frame is None:
                        if not cfg.typed_frames:
                            continue
                        virtual[frame_parts] = (a_frame, context, e)
                    else:
                        derived[frame] = max(derived.get(frame, 0.0), a_frame)
                        gens[frame] = (context, e)
                    if category not in size:
                        size[category] = len(S.inc.get((category, ABSTRACT), ()))
                    framed.append((pos, size[category], -a_frame,
                                   (1, frame_parts) if frame is None else (0, frame),
                                   frame_parts if frame is None else frame))
        order: Dict[object, tuple] = {}
        rounds: Dict[Tuple[int, int], list] = {}           # (tier, round) -> keys in the order of their group
        if cfg.select == "specific":                       # the most specific generalisation of each place first
            for tier, candidates in ((0, framed), (1, bare)):
                seen: Dict[int, int] = {}
                # sorted on the first four fields: they differ for any two candidates (the fourth names the unit),
                # so the whole tuples sort the same way
                for group, _, _, _, key in sorted(candidates):
                    seen[group] = seen.get(group, 0) + 1
                    if key not in order:
                        order[key] = (tier, seen[group], group)
                        rounds.setdefault((tier, seen[group]), []).append(key)
        grounded = dict(base)
        grounded.update(derived)
        first = cfg.evoke == "grounded_first"
        if first and cfg.select == "specific":
            present, whole = self._present_in_order(base, grounded, virtual, fresh, order, rounds, F)
            ranked = [(key, -neg) for _, neg, _, _, _, key in present[:F]]
            if not whole:                              # F or more are present: nothing called to mind gets a place
                return ranked, gens, virtual, grounded
        else:
            present = self._present(grounded, virtual, fresh, order) if first else None
            ranked = self._winners(grounded, {}, virtual, F, first, fresh, order, present)
        spread: Dict[int, Optional[tuple]] = {}
        in_play = None
        for _ in range(cfg.T if cfg.lag0 else 0):
            if first and len(present) >= F:
                break                                  # what is only called to mind ranks after all that is present
            if in_play is None:
                in_play = np.array(list(grounded), dtype=np.int64)
            evoked = self._evoked(ranked, in_play, spread)
            again = self._winners(grounded, evoked, virtual, F, first, fresh, order, present)
            if [k for k, _ in again] == [k for k, _ in ranked]:
                ranked = again
                break
            ranked = again
        return ranked, gens, virtual, grounded

    def _evoked(self, ranked, in_play: torch.Tensor, spread) -> Dict[int, float]:
        """EXPECT(lag 0) activation from the atomic units of the focus, summed per target (``in_play``: the units
        already grounded). ``spread`` keeps, per unit, its targets and 2e - 1 for the rest of this settling
        (nothing is written while settling)."""
        S, k = self.S, self.cfg.k
        kinds = mv_view(self, "u_kind")
        dsts, vals, small = [], [], True
        for key, a_u in ranked:
            if isinstance(key, tuple) or not kinds[key] & ATOMIC:
                continue
            if key not in spread:
                ties = S.out.get((key, EXPECT, 0))
                if not ties:
                    spread[key] = None
                elif len(ties) <= SMALL:                   # a few ties: targets and 2e - 1 in plain numbers
                    w_pos, w_neg = masses_py(self, ties)
                    dst = mv_view(self, "t_dst")
                    spread[key] = ([dst[t] for t in ties],
                                   [2.0 * expectation(p, n, k) - 1.0 for p, n in zip(w_pos, w_neg)])
                else:
                    ids, dst = S.out_ids((key, EXPECT, 0))
                    w = masses(self, ids)
                    spread[key] = (dst.numpy(), 2.0 * expectations(w[0], w[1], k) - 1.0)
            if spread[key] is None:
                continue
            targets, value = spread[key]
            if isinstance(targets, list):
                dsts.append(targets)
                vals.append([a_u * x for x in value])
            else:
                small = False
                dsts.append(targets)
                vals.append(a_u * value)
        if not dsts:
            return {}
        if small:                                          # summed per target in order of occurrence, from 0
            summed: Dict[int, float] = {}
            for targets, value in zip(dsts, vals):
                for d, x in zip(targets, value):
                    summed[d] = summed.get(d, 0.0) + x
            play = set(in_play.tolist())
            return {d: summed[d] for d in sorted(summed) if summed[d] > 0.0 or d in play}
        dst_all = np.concatenate([np.asarray(d, dtype=np.int64) for d in dsts])   # summed per target in one pass
        val_all = np.concatenate([np.asarray(v, dtype=np.float64) for v in vals])
        unique, inverse = np.unique(dst_all, return_inverse=True)
        summed = np.bincount(inverse, weights=val_all, minlength=unique.size)   # in order, from 0, as index_add_
        keep = (summed > 0.0) | np.isin(unique, in_play)       # a negative sum matters only for a unit in play
        return dict(zip(unique[keep].tolist(), summed[keep].tolist()))

    @staticmethod
    def _present_in_order(base, grounded, virtual, fresh, order, rounds, F) -> Tuple[List[tuple], bool]:
        """The entries of ``_present`` in ranking order, as far as the first F (select = "specific"): every
        generalisation and frame has an order rank (tier, round, group) of its own, and units without one (what
        is present) rank before them. Returns (entries, whole): whole when these are all the entries."""
        scored = sorted(((0, 0, 0, 0), -grounded[u], 0 if u in fresh else 1, 0, (u,), u)
                        for u in base if u not in order and grounded[u] > 0.0)
        if len(scored) >= F:
            return scored, False
        top: Dict[int, int] = {}
        for tier, number in rounds:
            top[tier] = max(top.get(tier, 0), number)
        for tier in (0, 1):
            for number in range(1, top.get(tier, 0) + 1):
                for key in rounds.get((tier, number), ()):
                    if isinstance(key, tuple):
                        a = virtual[key][0]
                        if a > 0.0:
                            scored.append(((1,) + order[key], -a, 1, 1, key, key))
                    else:
                        a = grounded[key]
                        if a > 0.0:
                            scored.append(((1,) + order[key], -a, 0 if key in fresh else 1, 0, (key,), key))
                    if len(scored) >= F:
                        return scored, False
        return scored, True

    @staticmethod
    def _present(grounded, virtual, fresh, order) -> List[tuple]:
        """Ranking entries of what is present and of the frames to be recruited (with ``grounded_first`` they do
        not depend on what is called to mind): (rank, -activation, not fresh, frame, (unit,) or parts, key)."""
        scored = []
        for unit, a in grounded.items():
            if a > 0.0:
                rank = (1,) + order[unit] if unit in order else (0, 0, 0, 0)
                scored.append((rank, -a, 0 if unit in fresh else 1, 0, (unit,), unit))
        for parts, (a, _, _) in virtual.items():
            if a > 0.0:
                rank = (1,) + order[parts] if parts in order else (0, 0, 0, 0)
                scored.append((rank, -a, 1, 1, parts, parts))
        return scored

    @staticmethod
    def _winners(grounded, evoked, virtual, F, grounded_first=False, fresh=(), order=None, present=None):
        """k-winners: the F candidates that win a place in the focus.

        As written: by activation alone, EXPECT(lag 0) activation added to every unit; equal activations are
        ordered: perceived or recognised in this very moment, then older units by id, then frames still to be
        recruited. Changed (see NOTES.md): with ``grounded_first`` what is present keeps the activation of its
        presence and is ranked before what is only called to mind; with ``order`` (select = "specific") what is
        present comes first and generalisations follow in rounds, the most specific one of each place first.
        Every entry ends in its own (unit,) or parts tuple, so the ranking is a total order: the F smallest entries
        are the first F of the sorted list. ``present``: the entries of ``_present`` (with ``grounded_first``).
        """
        order = order or {}
        if grounded_first:
            scored = list(present) if present is not None else Mind._present(grounded, virtual, fresh, order)
            if len(scored) < F:                        # what is only called to mind ranks after all that is present
                scored += [((2, 0, 0, 0), -min(1.0, a), 1, 0, (unit,), unit) for unit, a in evoked.items()
                           if unit not in grounded and a > 0.0]
        else:
            scored = []
            for unit, a in grounded.items():
                a = min(1.0, a + evoked.get(unit, 0.0))
                if a > 0.0:
                    rank = (1,) + order[unit] if unit in order else (0, 0, 0, 0)
                    scored.append((rank, -a, 0 if unit in fresh else 1, 0, (unit,), unit))
            for unit, a in evoked.items():
                if unit not in grounded and a > 0.0:
                    scored.append(((0, 0, 0, 0), -min(1.0, a), 1, 0, (unit,), unit))
            for parts, (a, _, _) in virtual.items():
                if a > 0.0:
                    rank = (1,) + order[parts] if parts in order else (0, 0, 0, 0)
                    scored.append((rank, -a, 1, 1, parts, parts))
        return [(key, -neg) for _, neg, _, _, _, key in heapq.nsmallest(F, scored)]

    def _feel(self, surprise: float) -> None:
        """Step 6: the mind's own state as events of the next moment (intensity carries the amount)."""
        S, k = self.S, self.cfg.k
        relief = 0.0
        if self._gained:
            ids, gain = self._gains()
            unique, inverse = np.unique(ids, return_inverse=True)
            gained = np.bincount(inverse, weights=gain, minlength=unique.size)   # in order, from 0, as index_add_
            world = kind_masks(self)[0]                         # by channel id; index -1 (no channel) -> False
            channels = np_view(self, "u_channel")[np_view(self, "t_dst")[unique]]
            expect = (np_view(self, "t_form")[unique] == EXPECT) & world[channels]
            ties = unique[expect]                               # only these enter the sum: read only them
            after = masses(self, ties, total=True)
            before = np.maximum(after - gained[expect], 0.0)   # clamp at 0 (the sign of a zero changes nothing below)
            terms = np_view(self, "t_use")[ties] * (after / (after + k) - before / (before + k))
            relief = float(torch.from_numpy(terms).sum())       # the same values summed by torch in the same order
            relief = max(0.0, relief)
        conflict = conflict_of(self, self.context) if self.context is not None else 0.0
        self.felt = {"surprise": surprise, "relief": relief / (relief + k),
                     "load": min(1.0, S.n_ties / float(self.cfg.body_ties)),
                     "conflict": conflict / (conflict + k), "relief_raw": relief,
                     "drive": self.felt.get("drive", 0.0) if self.knob["theta"] >= 1e8 else total_drive(self)}
        prov = ACTED if self.act_hope or self.hoping is not None else SENSED
        for name in FEELINGS:
            self.pending.append(Event("feel", name, prov, self.felt[name]))

    def _gains(self) -> Tuple[np.ndarray, np.ndarray]:
        """The gains of this moment in the order they were made, as int64 / float64 arrays: entries are (ids, gains)
        tensors (or arrays) or, for one tie, (id, gain) numbers (``learn.add_evidence1``)."""
        ids, gains, run_i, run_g = [], [], [], []
        for i, a in self._gained:
            if isinstance(i, (torch.Tensor, np.ndarray, list, tuple)):
                if run_i:
                    ids.append(np.array(run_i, dtype=np.int64))
                    gains.append(np.array(run_g, dtype=np.float64))
                    run_i, run_g = [], []
                ids.append(i.numpy() if isinstance(i, torch.Tensor) else np.asarray(i, dtype=np.int64))
                gains.append(a.numpy() if isinstance(a, torch.Tensor) else np.asarray(a, dtype=np.float64))
            else:
                run_i.append(i)
                run_g.append(a)
        if run_i:
            ids.append(np.array(run_i, dtype=np.int64))
            gains.append(np.array(run_g, dtype=np.float64))
        return np.concatenate(ids), np.concatenate(gains)

    # ------------------------------------------------------------------ transient state, undo, retraction
    _TRANSIENT = ("t", "focus", "last_focus", "gens", "last_gens", "carry", "episode", "trace", "context", "pending",
                  "predicted", "act_hope", "hoping", "chunked", "idle", "knob", "felt", "babbled", "imagined", "prov_of",
                  "unknown", "voice", "witnesses", "episode_start", "company")

    def _snapshot(self):
        return {name: copy.copy(getattr(self, name)) for name in self._TRANSIENT}

    def undo_last(self) -> None:
        """Undo the most recent moment completely: tensors, sizes, alphabets and working state."""
        if "personal_log" in self.__dict__:
            from .iliski import forget_moment
            forget_moment(self, self.t)
        entry = self.S.undo_last()
        for channel, size in enumerate(entry.alpha0):
            self.ch.truncate(channel, size)
        for channel in range(len(entry.alpha0), len(self.ch.names)):
            self.ch.truncate(channel, 0)
        if entry.undo is None:
            raise RuntimeError("only the most recent moment keeps its working state")
        for name, value in entry.undo.items():
            setattr(self, name, value)
        self.trust = self._trust()
        self.__dict__.pop("_between_memo", None)

    def retract(self, m: int) -> Dict[str, int]:
        """Retract moment m at the level of the ledger: exactly its increments are removed."""
        report = self.S.retract(m)
        self.trust = self._trust()
        self.__dict__.pop("_between_memo", None)
        if "personal_log" in self.__dict__:
            from .iliski import forget_moment
            forget_moment(self, m)
        return report

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_facet_cache", None)                   # caches, rebuilt on demand
        state.pop("_situation_cache", None)
        state.pop("_between_memo", None)
        for name in READ_CACHES:
            state.pop(name, None)
        return state

    def clone(self) -> "Mind":
        """A copy of the mind without the experience log (for probing without touching the life)."""
        log, self.S.log = self.S.log, {}
        try:
            twin = copy.deepcopy(self)
        finally:
            self.S.log = log
        return twin

    # ------------------------------------------------------------------ reading from outside (probes)
    def expected(self, channel: str, mode: Optional[str] = None) -> Dict[Hashable, float]:
        """Black-box reading: the expectations for the next moment on one channel, by symbol."""
        if not self.ch.has(channel):
            return {}
        cid = self.ch.id(channel)
        pred = self.predicted if mode is None else predictions(self, mode)
        return {self.ch.symbol(cid, int(self.S.u_symbol[u])): e for u, e in pred.items()
                if int(self.S.u_channel[u]) == cid}

    def size(self) -> Dict[str, int]:
        S = self.S
        out = {"units": S.n_units, "ties": S.n_ties}
        out.update(S.nbytes())
        return out
