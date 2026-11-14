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
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Optional, Tuple

import torch

from . import act as act_mod
from .events import ACTED, IMAGINED, IMITATED, N_PROV, PROVENANCE, REFERENCE, SENSED, Channels, Event
from .evidence import conviction, expectation
from .form import categories_of, compose, confirm_parts, maybe_chunk, recognise
from .learn import learn_follow, learn_together
from .read import conflict_of, predictions, total_drive
from .store import ABSTRACT, ATOMIC, EXPECT, F64, I64, SLOTTED, W_NEG, W_POS, Substance

FEELINGS = ("surprise", "relief", "load", "conflict")


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
    neg_weight: str = "e"           # failed expectation weighted by e (as written) or by 1 ("one")
    induct_cap: float = 1.0         # each instance counts at most one experience in induction
    general_miss: bool = False      # changed: a generalisation in the focus gets negative evidence for what
                                    # it expected and did not occur (the ordinary rule of step 5)
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
    undo_depth: int = 1             # how many of the latest moments keep their working state for undo

    @classmethod
    def changed(cls, **over) -> "Config":
        """The design with the smallest principled changes the prototype found necessary (NOTES.md)."""
        base = dict(support="nested", predict="specific", evoke="grounded_first", feel_focus=False,
                    clear_on_pause=True, general_miss=True, select="specific")
        base.update(over)
        return cls(**base)


class Mind:
    """A mind that learns while it lives. All knowledge is in ``self.S`` (units, ties, evidence)."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg if cfg is not None else Config()
        self.S = Substance(n_slots=N_PROV, keep_log=self.cfg.keep_log)
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
        S.end()
        # transient state (working state of the present moment; not knowledge)
        self.focus: List[Tuple[int, float, bool]] = []
        self.last_focus: List[Tuple[int, float, bool]] = []
        self.gens: List[Tuple[int, int, float]] = []
        self.last_gens: List[Tuple[int, int, float]] = []
        self.carry: Dict[int, float] = {}
        self.episode: List[int] = []
        self.context: Optional[int] = None
        self.pending: List[Event] = []
        self.predicted: Dict[int, float] = {}
        self.act_hope: Dict[int, float] = {}
        self.hoping: Optional[Tuple[int, float]] = None
        self.chunked: List[int] = []
        self.prov_of: Dict[int, int] = {}
        self.idle = 0
        self.babbled = 0
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
        trust = torch.ones(N_PROV, dtype=F64)
        for p in range(N_PROV):
            if p in REFERENCE:
                continue
            if p in self.cfg.trust_fixed:
                trust[p] = self.cfg.trust_fixed[p]
                continue
            tie = self.source_tie[p]
            trust[p] = expectation(float(self.S.t_w[W_POS, tie, SENSED]), float(self.S.t_w[W_NEG, tie, SENSED]),
                                   self.cfg.k)
        return trust

    # ------------------------------------------------------------------ one moment
    def step(self, events: Iterable[Event] = ()) -> List[Event]:
        """Live one moment with the given outside events; returns the acts the mind emits."""
        S, cfg = self.S, self.cfg
        undo = self._snapshot()
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
        learn_follow(self, percepts)                       # step 5 for the previous moment
        surprise = self._surprise(percepts)
        slot = items[0][1] if items else SENSED
        if items:
            self.idle = 0
            for unit, _ in items:
                self.episode.append(unit)
                maybe_chunk(self, slot)
            self.episode = self.episode[-self.knob["F"]:]
        else:
            self.idle += 1
            self.episode = []
            if cfg.clear_on_pause:
                self.carry = {}
        self._settle(percepts, slot, bool(items))          # steps 1 and 2
        if cfg.lag0:
            learn_together(self, [f for f in self.focus if int(S.u_kind[f[0]]) & ATOMIC], slot)
        self.predicted = predictions(self)                 # step 3
        acts = act_mod.decide(self)                        # step 4
        if cfg.imagine and not items:
            act_mod.imagine(self)
        if cfg.knobs and not items and not acts:
            acts = act_mod.babble(self)
        for event in acts:
            self.pending.append(event)
        self._feel(surprise)                               # step 6
        entry = S.end()
        entry.undo = undo
        previous = S.log.get(self.t - cfg.undo_depth)
        if previous is not None:
            previous.undo = None
        self.last_focus, self.last_gens = self.focus, self.gens
        return acts

    def _surprise(self, percepts) -> float:
        """Mean (1 - e) of what occurred plus mean e of what was expected on a reporting channel and failed."""
        S = self.S
        occurred = {u for u, a, _, ch in percepts if a > 0 and self.ch.kind(ch) in ("sense", "act")}
        if not occurred:
            return 0.0
        channels = {int(S.u_channel[u]) for u in occurred}
        unexpected = sum(1.0 - self.predicted.get(u, 0.5) for u in occurred) / len(occurred)
        failed = [e for u, e in self.predicted.items()
                  if e > 0.5 and u not in occurred and int(S.u_channel[u]) in channels]
        return min(1.0, unexpected + (sum(failed) / len(failed) if failed else 0.0))

    def _settle(self, percepts, slot: int, has_items: bool) -> None:
        S, cfg = self.S, self.cfg
        F = self.knob["F"]
        decay = cfg.decay if cfg.decay is not None else 1.0 - 1.0 / F
        base: Dict[int, float] = {u: decay * a for u, a in self.carry.items()}
        for unit, intensity, prov, channel in percepts:
            if intensity > 0 and (cfg.feel_focus or self.ch.kind(channel) != "feel"):
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
        # derived activation: categories of what is present, frames in which a category stands in
        derived: Dict[int, float] = {}
        gens: Dict[int, Tuple[int, float]] = {}
        virtual: Dict[Tuple[int, ...], Tuple[float, int, float]] = {}
        cats = {u: [(c, e) for c, e in categories_of(self, u) if e > 0.5] for u in base}
        bare: List[tuple] = []                             # (instance, extension size, -activation, key)
        framed: List[tuple] = []                           # (position, extension size, -activation, key)
        for unit, a_u in base.items():
            for category, e in cats[unit]:
                derived[category] = min(1.0, derived.get(category, 0.0) + a_u * (2.0 * e - 1.0))
                if category not in gens or e > gens[category][1]:
                    gens[category] = (unit, e)
                bare.append((unit, len(S.inc.get((category, ABSTRACT), ())), -derived[category], (0, category),
                             category))
        if context is not None:
            for pos, part in enumerate(parts):
                for category, e in cats.get(part, []):
                    a_frame = (sum(base.get(p, 0.0) for j, p in enumerate(parts) if j != pos)
                               + base.get(part, 0.0) * (2.0 * e - 1.0)) / len(parts)
                    frame_parts = parts[:pos] + (category,) + parts[pos + 1:]
                    frame = S.tuple_index.get(frame_parts)
                    if frame is None:
                        if not cfg.typed_frames:
                            continue
                        virtual[frame_parts] = (a_frame, context, e)
                    else:
                        derived[frame] = max(derived.get(frame, 0.0), a_frame)
                        gens[frame] = (context, e)
                    framed.append((pos, len(S.inc.get((category, ABSTRACT), ())), -a_frame,
                                   (1, frame_parts) if frame is None else (0, frame),
                                   frame_parts if frame is None else frame))
        order: Dict[object, tuple] = {}
        if cfg.select == "specific":                       # the most specific generalisation of each place first
            for tier, candidates in ((0, framed), (1, bare)):
                seen: Dict[int, int] = {}
                for group, size, neg_a, tie_break, key in sorted(candidates, key=lambda c: c[:4]):
                    seen[group] = seen.get(group, 0) + 1
                    order.setdefault(key, (tier, seen[group], group))
        grounded = dict(base)
        grounded.update(derived)
        first = cfg.evoke == "grounded_first"
        fresh = {u for u, a, _, _ in percepts if a > 0}
        if context is not None:
            fresh.add(context)
        ranked = self._winners(grounded, {}, virtual, F, first, fresh, order)
        for _ in range(cfg.T if cfg.lag0 else 0):
            evoked: Dict[int, float] = {}
            for key, a_u in ranked:
                if isinstance(key, tuple) or not int(S.u_kind[key]) & ATOMIC:
                    continue
                ties = S.out.get((key, EXPECT, 0), [])
                if not ties:
                    continue
                ids = torch.tensor(ties, dtype=I64)
                e = expectation((S.t_w[W_POS, ids] * self.trust).sum(-1), (S.t_w[W_NEG, ids] * self.trust).sum(-1),
                                cfg.k)
                for v, x in zip(S.t_dst[ids].tolist(), (a_u * (2.0 * e - 1.0)).tolist()):
                    evoked[v] = evoked.get(v, 0.0) + x
            again = self._winners(grounded, evoked, virtual, F, first, fresh, order)
            if [k for k, _ in again] == [k for k, _ in ranked]:
                ranked = again
                break
            ranked = again
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
                if int(S.u_kind[key]) & SLOTTED:
                    confirm_parts(self, key, slot)
            focus.append((key, a, key in grounded))
        self.focus = focus
        self.carry = {u: a for u, a, _ in focus if u in base}
        self.act_hope = {u: h for u, h in self.act_hope.items() if u in self.carry}
        self.prov_of = {u: p for u, p in self.prov_of.items() if u in self.carry}
        S.use_units(torch.tensor([u for u, _, _ in focus], dtype=I64))

    @staticmethod
    def _winners(grounded, evoked, virtual, F, grounded_first=False, fresh=(), order=None):
        """k-winners: the F candidates that win a place in the focus.

        As written: by activation alone, EXPECT(lag 0) activation added to every unit; equal activations are
        ordered: perceived or recognised in this very moment, then older units by id, then frames still to be
        recruited. Changed (see NOTES.md): with ``grounded_first`` what is present keeps the activation of its
        presence and is ranked before what is only called to mind; with ``order`` (select = "specific") what is
        present comes first and generalisations follow in rounds, the most specific one of each place first.
        """
        order = order or {}
        scored = []
        for unit, a in grounded.items():
            if not grounded_first:
                a = min(1.0, a + evoked.get(unit, 0.0))
            if a > 0.0:
                rank = (1,) + order[unit] if unit in order else (0, 0, 0, 0)
                scored.append((rank, -a, 0 if unit in fresh else 1, 0, (unit,), unit))
        for unit, a in evoked.items():
            if unit not in grounded and a > 0.0:
                scored.append(((2, 0, 0, 0) if grounded_first else (0, 0, 0, 0), -min(1.0, a), 1, 0, (unit,), unit))
        for parts, (a, _, _) in virtual.items():
            if a > 0.0:
                rank = (1,) + order[parts] if parts in order else (0, 0, 0, 0)
                scored.append((rank, -a, 1, 1, parts, parts))
        scored.sort(key=lambda s: s[:5])
        return [(key, -neg) for _, neg, _, _, _, key in scored[:F]]

    def _feel(self, surprise: float) -> None:
        """Step 6: the mind's own state as events of the next moment (intensity carries the amount)."""
        S, k = self.S, self.cfg.k
        relief, touched = 0.0, 0
        if self._gained:
            ids = torch.cat([i for i, _ in self._gained])
            gain = torch.cat([a for _, a in self._gained])
            unique, inverse = torch.unique(ids, return_inverse=True)
            gained = torch.zeros(unique.numel(), dtype=F64).index_add_(0, inverse, gain)
            touched = int(unique.numel())
            kinds = [self.ch.kind(c) if c >= 0 else "" for c in S.u_channel[S.t_dst[unique]].tolist()]
            expect = (S.t_form[unique] == EXPECT) & torch.tensor([k in ("sense", "act") for k in kinds])
            after = ((S.t_w[W_POS, unique] + S.t_w[W_NEG, unique]) * self.trust).sum(-1)
            before = torch.clamp(after - gained, min=0.0)
            relief = float((S.t_use[unique] * (conviction(after, k) - conviction(before, k)))[expect].sum())
            relief = max(0.0, relief)
        conflict = conflict_of(self, self.context) if self.context is not None else 0.0
        self.felt = {"surprise": surprise, "relief": relief / (relief + k),
                     "load": min(1.0, S.n_ties / float(self.cfg.body_ties)),
                     "conflict": conflict / (conflict + k), "relief_raw": relief, "drive": total_drive(self)}
        prov = ACTED if self.act_hope or self.hoping is not None else SENSED
        for name in FEELINGS:
            self.pending.append(Event("feel", name, prov, self.felt[name]))

    # ------------------------------------------------------------------ transient state, undo, retraction
    _TRANSIENT = ("t", "focus", "last_focus", "gens", "last_gens", "carry", "episode", "context", "pending",
                  "predicted", "act_hope", "hoping", "chunked", "idle", "knob", "felt", "babbled", "imagined", "prov_of")

    def _snapshot(self):
        return {name: copy.copy(getattr(self, name)) for name in self._TRANSIENT}

    def undo_last(self) -> None:
        """Undo the most recent moment completely: tensors, sizes, alphabets and working state."""
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

    def retract(self, m: int) -> Dict[str, int]:
        """Retract moment m at the level of the ledger: exactly its increments are removed."""
        report = self.S.retract(m)
        self.trust = self._trust()
        return report

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
