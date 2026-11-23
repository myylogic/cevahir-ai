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
from .learn import learn_facets, learn_follow, learn_together, learn_value, note_claims, reflect, tie_mass
from .read import conflict_of, predictions, total_drive
from .store import ABSTRACT, ATOMIC, EXPECT, F64, I64, PART, SLOTTED, W_NEG, W_POS, Substance

FEELINGS = ("surprise", "relief", "load", "conflict")
SELF_CHANNEL, SELF_SYMBOL = "benlik", "ben"
INNER_CHANNEL, NOT_KNOWING = "ic", "bilmemek"     # L7: not knowing, perceived as an inner event
LOOKING = "bakmak"                                # F50: having decided to look something up, perceived likewise


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
    inherit: bool = False           # F37 (Inheritance): a frame's value may be what the filler's classes go on with
                                    # (deduction through the stems that name its places; a class's own classes count,
                                    # two steps), its own record last
    forms: bool = False             # F36 (Word Forms): a word never met is read as the lived word it is a form of (a
                                    # shared stem where the lived words branch), when nothing else expects anything
    known: bool = False             # F41 (Unsaid Known): a frame's value may be how its filler's own lived sequence
                                    # goes on (a thing known of it and not said yet in this answer), the one most like
                                    # what the members said of themselves; the answer goes on while something known
                                    # of its topic is unsaid and ends when nothing is
    echo: bool = False              # F42 (Episode Echo): what already followed an item in the present episode (heard
                                    # or said) is an echo, not knowledge of what follows it now: not read as the lived
                                    # continuation of the last word (F16), nor as "how the filler goes on" (NEXT,
                                    # START); F41 holds it too
    exclusive_world: bool = False   # F44 (Lived Exclusion across channels): what an instance was lived to go on with on
                                    # any world channel counts against every world target it never went on with (its
                                    # record holds all that followed it): an instance that always went on with the end
                                    # of the turn is evidence against a word there (F27 reads one channel at a time)
    wholes: bool = False            # F43 (Wholes): a sequence said alone and answered is a whole; a watched dialogue
                                    # whose answer is made of the replies of wholes in its utterance is lived once more
                                    # in summary (wholes as their forms, replies as pointers), and an utterance never
                                    # answered whose summary was lived is heard in it and answered part by part, the
                                    # working state set aside for each part (zihin/butun.py)
    trials: bool = False            # F57 (Trials): at a context where the mind decided and which then goes on from
                                    # outside, before what came is taken in, the same context is read once more as
                                    # the mind decides it and once more with each of its ways of reading changed
                                    # (on if off, off if on); what came says which was better, each way has a record
                                    # of that, and the mind's own configuration follows the record (zihin/deney.py)
    built_relations: bool = False   # F58 (Built Relations; with trials): where the mind did not say what then came,
                                    # a chain of two of its own relations that names it for a filler of the context
                                    # ("what is known of it, then what stands at the asked place of that") is a
                                    # relation to try; tried like any way (F57), and in use it is a relation of a
                                    # frame like the others: the members vouch for it; a relation it built and
                                    # took into use is a step of the next chain (read.built_value, deney.propose)
    unseen_share: bool = False      # F60 (The Unseen Share; with families): the frame that speaks counts against a
                                    # word its members never went on with only as far as its members went on with
                                    # one another's words (aile.repeated): where every member went on with a word of
                                    # its own, a word never seen there is what is to be expected, and the family's
                                    # claim is not lowered by it
    own_race: bool = False          # F61 (One Race for Its Own Time): in a silent moment in which nothing was said
                                    # the acts of its own time (looking something up F47, asking itself F51, thinking
                                    # a thing over F59, thinking about what it took in F48) race as its words do, each
                                    # with its own rate (the drive + what it led to before): the one that reaches the
                                    # line first is decided, among equals the firmer, then the one not taken for
                                    # longest; the others wait for the next silent moment. Without it they are taken
                                    # in the order they were built, each giving way to those before it (act.own_time)
    conjectures: bool = False       # F59 (Conjectures): a thing told to go on with an item that is itself told about
                                    # is linked to it; what is told of that item, with the thing in its place, is a
                                    # candidate; where the thing was told that too the link carried it, where it was
                                    # told otherwise it did not; a candidate nobody told, whose kind of link has
                                    # been found to carry, is thought by the inner voice in a silent moment
                                    # (imagined, as every thought) and kept with its source until what is told
                                    # later settles it (zihin/varsayim.py)
    whole_kinds: bool = False       # F56 (The Kind of a Whole; with wholes): the form of a whole says which wholes
                                    # stand in it also where its answer reads as their replies in more than one
                                    # order ("walk twice and walk" answers like "walk thrice"): without it such a
                                    # lived command had no form, stood as a bare whole and took what an action
                                    # phrase takes ("[walk twice and walk] right"); zihin/butun.py (form, align)
    shares_tested: bool = False     # F64 (A Share Is Tested Before It Is Carried; with told_apart): what a word was
                                    # told apart to answer with is its reply in an utterance never answered only
                                    # where the mind's record has decided that shares carry: each time a word is
                                    # told apart in a place new for it, what it answered with before held there or
                                    # did not (butun.share_carries)
    told_apart: bool = False        # F45 (Told Apart; with wholes): when two lived dialogues differ in one word of
                                    # the utterance and their answers differ in one pair of words, each of the two
                                    # words answers with its own (Mill's method of difference): a word never said
                                    # alone becomes a whole (zihin/butun.py)
    open_claims: bool = False       # F46 (Open Claims): when the frame that speaks claims a relation of its filler's
                                    # own record (how its sequences go on, what it was answered with, what is known
                                    # of it) and the relation names nothing for this filler, the mind notes, on a
                                    # channel of its own, that something it does not know follows this item; what it
                                    # looks up first is what such claims were left open for (web.wanted_topics)
    look: bool = False              # F47 (Looking Up by Itself; with open_claims): in a silent moment the mind may
                                    # decide to look up an item a claim was left open for: the want (how firmly the
                                    # claim is open) plus what looking up led to before, against the line it speaks
                                    # at; having read, it reads the claim again where it was left open, and what that
                                    # shows is the outcome of the act (act.look, learn.settle_look). The reading is
                                    # the session's to carry out, from the sources its owner gave it
    think: bool = False             # F48 (Thinking by Itself): in a silent moment the mind may decide to think about
                                    # what it last took in: how far that went against what it expected (its surprise
                                    # over those moments) plus what its thoughts have come to before, against the line
                                    # it speaks at (act.think_now, learn.settle_thought). The inner voice is the
                                    # session's to let happen; with this on the program's clock for thinking is unused
    faults: bool = False            # F49 (Fault Tracing): every way of reading (a frame's claim by its kind, the
                                    # backoffs, forms, similar situations) has a tie of its own, (way -> holds);
                                    # what each claimed where the mind was free to speak is kept with its context
                                    # and settled when that context is lived to go on (the break is where its own
                                    # word and the lived one part); lambda, how far a reading is believed, is the
                                    # expectation of that tie read general -> specific, 1/2 with nothing lived
                                    # (zihin/hata.py)
    find_out: bool = False          # F50 (Finding Out; with open_claims, look and faults): what the mind does not
                                    # know it acts on, in the moment it is called on. Called on and with nothing to
                                    # say, or about to say something that rests on what others share and not on
                                    # what it lived of the thing asked about, it looks first when it has somewhere
                                    # to look: rate = (2u - 1) + (2 e_look - 1), u its not-knowing in that moment
                                    # (L7's measure). It does not assert what it is about to look up; that it is
                                    # looking is an inner event (how one says so is learned, never given); having
                                    # read, it is brought back to what it was asked and says what it can say now.
                                    # A thought that comes to nothing is an unknown about its topic (act.look_now,
                                    # Session.reply, Session.say_again)
    families: bool = False          # F53 (Frames of Frames): frames that differ in one fixed word and are the same
                                    # otherwise are a family, a generalisation with two places; a context no frame
                                    # speaks for but a family fits is read by the family, every frame one vote on
                                    # the kind of what its members went on with (zihin/aile.py): what was learned
                                    # about kinds of question carries over to a new kind
    asked_place: bool = False       # F55 (The Asked Place): a frame's value may be what a lived sequence of the
                                    # filler's own holds at the one place where it differs from the run of items the
                                    # filler begins in the context: a question is a statement the mind knows with one
                                    # item replaced, and the answer is that item (read.asked_places)
    lived_apart: bool = False       # F54 (Relations Lived Apart): a frame's relation "what the filler goes on with"
                                    # (NEXT) counts for a member only as far as the member lived it apart from its own
                                    # instance in the frame. Where the answer repeats the filler and goes on, the
                                    # instance's continuation is the member's continuation (one event): the relation
                                    # held for every member by the dialogue itself, whatever else was lived of it
                                    # (read.lived_apart)
    echo_exact: bool = False        # F52 (Echo of the Same Words): in the lived continuation of the last words
                                    # (F16) a continuation is an echo only where it followed these very words in
                                    # what is present; without it every word the partner just said is left out,
                                    # wherever it stood (read.continuation_backoff)
    echo_told: bool = False         # F62 (The Sentence Being Told; with echo_exact): the exact form of the echo
                                    # rule holds only where the mind's utterance, from its beginning, is how a
                                    # sequence it lived began: there a word the partner said is carried on; where
                                    # only its last words once went on so, the partner's words are echoes as before
                                    # F52 (read.continuation_backoff)
    wonder: bool = False            # F51 (Asking Itself; with open_claims, look): an item the mind has met in one
                                    # run of items only is an unknown. In a silent moment it may decide to ask
                                    # itself about it: rate = (2u - 1) + (2 e_wonder - 1), u = 1 (it has lived one
                                    # mention of it), e_wonder what asking itself has led to. It brings to mind the
                                    # question it knows best with the item in its place (imagined) and answers
                                    # inwardly; what it cannot answer there is an open claim about the item, which
                                    # it may look up (F47); what it then reads brings new items: a chain
                                    # (act.wonder, dusunce.ask_itself)
    silent_claims: bool = False     # F49, while it only listens (with faults): at a context it has no record of its
                                    # own of, the mind makes its full reading although it cannot speak, and what
                                    # its ways claim there is kept and settled by what comes next like any claim:
                                    # everything new it watches or reads is then a test of its ways. Costs the full
                                    # reading at every new context it listens to (hata.watch)
    relied_on: bool = False         # F63 (On Trial Where It Is Relied On; with silent_claims): while the mind only
                                    # listens, where something it lived of the very thing spoken of speaks in its
                                    # reading, the ways that only carry over what others share are not on trial:
                                    # agreeing with what it already knew is no prediction that held (hata.relied_on)
    trial_ends: bool = False        # F66 (An End Is an Outcome; with trials): where the mind decided and the other
                                    # then went on with no word (the end of a turn it watches), nothing is what
                                    # came: the configuration that would have said nothing did better than one
                                    # that would have said a word (deney.trial)
    split_failed: bool = False      # F67 (The Category Is Split Where It Failed; with faults): F49 reads a frame's
                                    # claim through a narrower place of its filler when the frame's own vote was
                                    # found wanting (it held less often than it claimed, on the whole). A vote that
                                    # holds for some fillers and never for others is calibrated on the whole: with
                                    # this the narrower vote is read as soon as the frame's own record holds more
                                    # than one experience against it (hata.failed)
    deduced_first: bool = False     # F68 (Deduction Stands Against the Co-members' Vote; with inherit and faults):
                                    # where F37 names a word for the filler (what was said of its classes, its
                                    # own record last) and F49's narrower vote (what its co-members went on with)
                                    # proposes another, the deduction stands against that vote
                                    # (read._deduced_against_co_members)
    own_turn: bool = False          # F69 (Its Own Turn Comes Last; with silent_claims): what a way's claims came
                                    # to while the mind only listened (what it was shown) and what they came to
                                    # when it was its turn (what it was asked) are kept apart, and its own turn's
                                    # record is read last, over the other, as the more specific evidence is
                                    # everywhere; and a moment in which it only listens is not being called on
                                    # again (what it held back is not counted unanswered there) (hata._apart)
    subject_carries: bool = False   # F72 (The Subject Carries; with known): an utterance the mind has nothing to
                                    # say to, right after an answer that told what it knew of a thing, is heard
                                    # again inwardly with that thing in the place one of its frames has there
                                    # (Session.carry_subject)
    subject_heard: bool = False     # F72 while listening (with silent claims): a sentence heard right after one in
                                    # which a thought held, that leaves open the place a frame has for that
                                    # thought's subject, is heard again inwardly with the subject in that place
                                    # (Session.carry_heard_subject)
    open_window: bool = False       # F70 (The Open Window): a family (F53) is read through the frames that come
                                    # to mind in the time the mind has: T frames (one a step of the moment),
                                    # the nearest to the context first, T more every later moment it has the
                                    # same context in mind, until the whole family is read; F37 reads the
                                    # places of an item the same way (aile.in_window, read._places_in_mind)
    unknown_outside: bool = False   # F65 (Not Knowing Is of the World; with L7): the amount the mind does not know
                                    # is kept when the *world* has just done something and it decided nothing. An
                                    # end of a turn that only came to mind (the question it asked itself, a
                                    # conjecture its inner voice thought: imagined) is not the world doing
                                    # something: without this the inner event "not knowing" follows such a thought
                                    # and is lived as the beginning of whatever the mind hears next
    urge: bool = False              # F40 (Urge to Go On): when the leading word's rival is itself expected, it races
                                    # against silence as far as the two are things known about the same item (their
                                    # share of that item's lived continuations): another thing to say, not a rival
    difference: bool = False        # F39 (Replacement Witness, Mill's method of difference): when nothing would be
                                    # said, a lived situation that differs from the present one by one replacement is a
                                    # witness of it as far as the lived pairs differing by the same replacement went on
                                    # alike (the replacement made no difference)
    purpose: bool = False           # F35 (Generalised Consequences): what a path never taken would lead to (good, bad)
                                    # is read from the paths its frames' members lived; its own record comes last
    replies: bool = False           # F34 (Reply Relation): a frame's value may be what its filler, said on its own,
                                    # was answered with; with it several places after a prefix may be bound (F32)
    witness: bool = False           # F38 (Witness Reliability): a lived suffix that leaves out a fixed item of a
                                    # frame's claim is a witness of it, as firmly as the lived contexts that end with
                                    # that item and the suffix went on as the suffix did (off: such a suffix never
                                    # counts, F34; off by default: lived with, it cost SCAN answers)
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


_RANK_SCRATCH = np.zeros(0, dtype=np.float64)   # Mind._rank_first: zero outside a call (a sum per category unit)


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
        self.topic = None                            # F41: (item, its place, episode) the present answer tells what
                                                     # it knows of (set when such a claim is acted on)
        self.open_where: Dict[int, int] = {}         # F46/F47: item -> the context a claim about it was last left open
        self.looking = None                          # F47: the item it decided, in this moment, to look up
        self.met_once: Dict[int, int] = {}           # F51: the items it has met in one run of items only (item ->
                                                     # the moment that run began), not asked itself about yet,
                                                     # the oldest first
        self.wondering = None                        # F51: the item it decided, in this moment, to ask itself about
        self.wondered: Dict[int, bool] = {}          # F51: the items it asked itself about and could not answer:
                                                     # what a look then shows settles what the asking was worth
        self.open_plain: Dict[int, bool] = {}        # F50: the items whose open claim is not a frame's relation (a
                                                     # call it had nothing to say to, a thought that came to
                                                     # nothing: False; a word it would have said on what others
                                                     # share: True, the claim is about that next word only)
        self.taking = (0.0, 0, 0.0, None)            # F48: its surprise over the moments of what it is taking in: the
                                                     # sum, how many, the greatest, and the item before that moment
        self.heard = None                            # F48: its surprise over what it last took in, not thought about yet
        self.about = None                            # F48: the item its expectation was of where it failed most
        self.thinking = None                         # F48: the rate at which it decided, in this moment, to think
        self.claimed: Dict[int, tuple] = {}          # F49: context -> what each way claimed there, not yet met by
                                                     # reality: (ways, the claimed unit, whether it was said)
        self.faults: List[dict] = []                 # F49: the last claims that failed (a record for the report)
        self.called: Dict[int, int] = {}             # F57: the contexts at which it decided, until what follows is shown
        self.trial_base: Dict[object, bool] = {}     # F57: its ways as they stood when the trials began
        self.trial_changes: List[dict] = []          # F57: every way it took into use or put away, and on what record
        self.trial_cases: List[dict] = []            # F57: the last cases in which a way made a difference
        self.trial_mark: Dict[object, tuple] = {}    # F57: a sleeping way's record when the configuration last changed
        self.pondering: Dict[int, int] = {}          # F59: the things it was told something of, to think over
        self.linked: Dict[int, tuple] = {}           # F59: item -> the things it found linked to it
        self.waiting: Dict[tuple, tuple] = {}        # F59: kind of link -> the things whose candidates wait for its record
        self.conjectures: Dict[tuple, dict] = {}     # F59: what it conjectured, with its source and what became of it
        self.supposing = None                        # F59: the thing it decided, in this moment, to think over
        self.pairs_tried: List[tuple] = []           # F57: the pairs of ways it found to do together what neither
                                                     # does alone, and tries
        self.own_taken: Dict[str, int] = {}          # F61: when it last took each act of its own time
        self.own_waited: Dict[str, int] = {}         # F61: the silent moments each act that wants has waited since
        self.told_in: Dict[object, tuple] = {}       # F64: word -> the places (the words before, after) it was told apart in
        self.owner_off: List[str] = []               # the mechanisms its owner told the program to keep off (--kapat):
                                                     # the full configuration leaves them off when the mind is loaded
        self.relations: tuple = ()                   # F58: the relations it built itself that are in use (chains)
        self.relations_tried: List[tuple] = []       # F58: those it has proposed and tries
        self.relation_paths: List[tuple] = []        # F58: every chain it ever proposed (a chain's code is its place)
        self._held_back: List[tuple] = []            # F49: (context, moment) of the claims it held back, not yet
                                                     # counted as answered or as still waiting
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
        if self.called and getattr(cfg, "trials", False):  # F57: what follows a context it decided at is shown:
            from . import deney                            # its ways are tried on it before what came is taken in
            deney.trial(self, items)
        learn_follow(self, percepts)                       # step 5 for the previous moment
        learn_value(self, percepts)                        # L5: a value signal reaches every step of the episode
        if self.claimed and getattr(cfg, "faults", False):   # F49: what its ways claimed here meets what is lived
            from . import hata
            hata.judge(self, items)
            for unit, intensity, _, channel in percepts:
                if intensity > 0.0 and self.ch.kind(channel) == "value":
                    sign = self.value_sign.get((self.ch.names[channel], self.symbol_of(unit)[1]))
                    if sign:
                        hata.valued(self, sign)
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
                if getattr(cfg, "find_out", False):        # lived before it); F50: whether it only comes to mind
                    self.__dict__["_came_to_mind"] = all(prov == IMAGINED for _, prov in items)
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
            if listening and items and getattr(cfg, "silent_claims", False) and getattr(cfg, "faults", False):
                from . import hata                         # F49: what its ways claim of what it is about to hear
                hata.watch(self)
            claims = [(key[1], held) for key, held in self._read_memo.items()   # F46: what the speaking frame
                      if type(key) is tuple and len(key) == 2 and key[0] == "claim"] \
                if getattr(cfg, "open_claims", False) else ()                    # claimed of its filler's record
            sought = None
            if not listening and items and getattr(cfg, "find_out", False) and getattr(cfg, "look", False) \
                    and self.__dict__.get("can_look"):
                sought = act_mod.look_now(self, items, acts)   # F50: called on and it does not know: it looks first
                decided = bool(acts)
                if sought is not None:
                    acts = []                              # what it is about to look up it does not assert
                    if sought[0] == self.__dict__.get("_sought"):
                        sought = None                      # (it is looking that up already: one decision, until it
                    else:                                  # is next called on from outside)
                        self.__dict__["_sought"] = sought[0]
                        self.pending.append(Event(INNER_CHANNEL, LOOKING, ACTED, min(1.0, sought[3])))
                        if not any(item == sought[0] and held[1] for item, held in claims):
                            claims = list(claims) + [(sought[0], (sought[3], True))]   # (F46 did not note it open
                        if sought[4]:                                                # in this moment: one note)
                            self.open_plain[sought[0]] = sought[5]
                    if decided and getattr(cfg, "faults", False):
                        from . import hata
                        hata.withheld(self, self.context)  # (the claim was held back, not said)
            if sought is None and acts and not listening and getattr(cfg, "find_out", False) \
                    and act_mod.holds_back(self, items, acts):
                acts = []                                  # F50: what it could not find out it does not assert
                from . import hata
                hata.withheld(self, self.context)
            if not listening and getattr(cfg, "trials", False):
                from . import deney
                deney.called(self, acts)                   # F57: it decided here; kept until what follows is shown
            self._unknowing(items, acts, listening)
            if sought is not None:
                self.unknown = 0.0                         # it is looking: that is what it feels, not not knowing
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
            free = not items and not acts and not listening   # a silent moment in which nothing was said
            race = getattr(cfg, "own_race", False)         # F61: the acts of its own time race (below)
            self.looking = act_mod.look(self) if free and getattr(cfg, "look", False) and not race else None   # F47
            if sought is not None:
                self.looking = sought[:3]                  # F50: decided in the moment it was called on
            if listening and items and "_sought" in self.__dict__ and any(
                    prov not in (ACTED, IMAGINED) for _, prov in items):
                del self.__dict__["_sought"]               # F50: called on anew from outside
            in_summary = self.__dict__.get("_in_summary")  # F43: a moment lived in summary: its items are the mind's
                                                           # own wholes and pointers, not words it was told (nothing
                                                           # of it is new to ask about, or told to think over)
            if getattr(cfg, "wonder", False):              # F51: the items met in one run of items only; and, free
                if items and listening and not in_summary:   # and with nothing else decided, whether it asks itself
                    born, act_ch = mv_view(self, "u_birth"), kind_masks(self)[5]   # about one of them
                    channel, met = mv_view(self, "u_channel"), self.met_once
                    for unit, prov in items:
                        if prov in (ACTED, IMAGINED) or not act_ch[channel[unit]]:
                            continue
                        if born[unit] == self.t:
                            met[unit] = self.episode_start
                        elif met.get(unit, self.episode_start) != self.episode_start:
                            del met[unit]                  # lived again in another run: it is no longer new
                    while len(met) > 64:
                        del met[next(iter(met))]
                self.wondering = act_mod.wonder(self) if free and not race \
                    and getattr(self, "thinking", None) is None \
                    and (self.looking is None or not self.__dict__.get("can_look", True)) else None   # (a look it
                                                                    # cannot carry out does not stand in the way)
            if getattr(cfg, "conjectures", False):         # F59: what it was told is something to think over; and,
                from . import varsayim                     # free and with nothing else decided, whether it does
                if items and listening and not in_summary:
                    varsayim.took_in(self, items)
                self.supposing = act_mod.suppose(self) if free and not race and self.looking is None \
                    and getattr(self, "wondering", None) is None else None
            if getattr(cfg, "think", False):               # F48: how surprised it was by what it is taking in (and
                if items and listening:                    # where most: the item its failed expectation was of), and
                    total, n, peak, about = self.taking    # whether it now thinks about it
                    if surprise > peak:                    # (the item before this one led it to expect otherwise)
                        peak, about = surprise, (self.episode[-2] if len(self.episode) > 1 else None)
                    self.taking = (total + surprise, n + 1, peak, about)
                elif not items and self.taking[1]:
                    self.heard, self.about = self.taking[0] / self.taking[1], self.taking[3]
                    self.taking = (0.0, 0, 0.0, None)
                self.thinking = act_mod.think_now(self) if free and not race else None
            if race:
                if sought is None:
                    self.looking = None
                self.wondering = self.supposing = self.thinking = None
                if free and not self.__dict__.get("_carrying"):   # (not in a moment of an act being carried out:
                    act_mod.own_time(self)                 # F61: the one of them it decides in this moment    its
                                                           # pauses are not moments of the mind's own time)
            self._feel(surprise)                           # step 6
        finally:
            self._read_memo = None
        if claims:
            note_claims(self, claims)
            for filler, (_, is_open) in claims:            # where the claim was left open (to be read again there)
                if is_open and self.context is not None:
                    self.open_where[filler] = self.context
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
        lived = getattr(self.cfg, "unknown_outside", False)   # F65: what only came to mind is not the world's doing
        outer = any(channels[u] != inner and self.ch.kind(channels[u]) == "sense" and not (lived and p == IMAGINED)
                    for u, p in items)
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
        EXPECT(lag 0) spreading. Returns (ranked, gens, virtual, grounded): of ``gens``, ``virtual`` and ``grounded``
        at least the entries of the ranked keys (all that the settling reads of them)."""
        cfg = self.cfg
        if cfg.select == "specific" and cfg.evoke == "grounded_first":
            found = self._rank_first(base, parts, context, fresh, F)
            if found is not None:
                return found
        return self._rank_all(base, parts, context, fresh, F)

    def _rank_first(self, base, parts, context, fresh, F):
        """``_rank`` with select = "specific" and grounded_first, when the focus is filled before every candidate
        has been listed: exactly the ranking ``_rank_all`` gives, without building the candidates that get no place.

        What is present ranks first. After it the generalisations come in rounds: in each round one frame for each
        place of the context, the most specific first (the smallest extension, then the most active, then a frame
        that exists before one to be recruited); when the frames are used up, one category for each present unit in
        the same way. A mind that has lived much has thousands of candidates for a moment and the focus has F
        places: the candidates are ordered as arrays, and only those whose turn comes are built. Returns None (then
        ``_rank_all``) when something present is itself a generalisation, or when every candidate has been listed
        and the focus is not full (then all of them are read, and what is called to mind too)."""
        S, cfg = self.S, self.cfg
        inc, out = S.inc, S.out
        for unit in base:                                  # (as _present_fills) what is present has no rank of its
            if (unit, ABSTRACT) in inc:                    # own among the generalisations and keeps its activation
                return None
            if (unit, PART, -1) in out and any((p, ABSTRACT) in inc for p in parts_of(self, unit)):
                return None
        scored = sorted(((0, 0, 0, 0), -a, 0 if unit in fresh else 1, 0, (unit,), unit)
                        for unit, a in base.items() if a > 0.0)
        ranked = [(key, -neg) for _, neg, _, _, _, key in scored[:F]]
        gens: Dict[int, Tuple[int, float]] = {}
        virtual: Dict[Tuple[int, ...], Tuple[float, int, float]] = {}
        grounded = dict(base)
        if len(scored) >= F:
            return ranked, gens, virtual, grounded
        k = cfg.k
        known: Dict[int, Optional[tuple]] = {}

        def categories(unit: int):
            """(categories, expectations) of a unit as arrays, where the expectation exceeds 1/2 (categories_of)."""
            if unit not in known:
                ties = out.get((unit, ABSTRACT, 0))
                if not ties:
                    known[unit] = None
                elif len(ties) <= SMALL:
                    pairs = [(c, e) for c, e in categories_of(self, unit) if e > 0.5]
                    known[unit] = (np.array([c for c, _ in pairs], dtype=np.int64),
                                   np.array([e for _, e in pairs], dtype=np.float64)) if pairs else None
                else:
                    ids, dst = S.out_ids((unit, ABSTRACT, 0))
                    w = masses(self, ids)
                    e = expectations(w[0], w[1], k)
                    keep = e > 0.5
                    known[unit] = (dst.numpy()[keep], e[keep]) if keep.any() else None
            return known[unit]

        extension = S.abstract_sizes()                     # len(inc[(c, ABSTRACT)]) of every unit, as one array

        def sizes_of(cats: np.ndarray) -> np.ndarray:
            return extension[cats]

        def frames(pos: int, where, cats, es, others: float, a_part: float):
            """The frames of one place in order: (key, activation, expectation, whether it exists)."""
            a_frame = (others + a_part * (2.0 * es - 1.0)) / len(parts)
            sizes, neg = sizes_of(cats), -a_frame
            turn = np.lexsort((neg, sizes)).tolist()
            i = 0
            while i < len(turn):
                j = i + 1
                while j < len(turn) and sizes[turn[j]] == sizes[turn[i]] and neg[turn[j]] == neg[turn[i]]:
                    j += 1
                tied = []                                  # equal so far: a frame that exists first (by its unit),
                for t in turn[i:j]:                        # then the frames to be recruited (by their parts)
                    category = int(cats[t])
                    if where is not None:
                        filled = list(parts)
                        for q in where:
                            filled[q] = category
                        frame_parts = tuple(filled)
                    else:
                        frame_parts = parts[:pos] + (category,) + parts[pos + 1:]
                    frame = S.tuple_index.get(frame_parts)
                    if frame is None:
                        if cfg.typed_frames:
                            tied.append(((1, frame_parts), frame_parts, float(a_frame[t]), float(es[t]), False))
                    else:
                        tied.append(((0, frame), frame, float(a_frame[t]), float(es[t]), True))
                if len(tied) > 1:
                    tied.sort(key=lambda found: found[0])
                for _, key, a, e, exists in tied:
                    yield key, a, e, exists
                i = j

        if context is not None:
            places = []
            for pos, part in enumerate(parts):
                found = categories(part) if part in base else None
                if found is None:
                    continue
                where = None
                if cfg.bind_repeats:
                    if part in parts[:pos]:
                        continue                           # the slot of this unit was built at its first position
                    where = [j for j, q in enumerate(parts) if q == part]
                others = sum(base.get(p, 0.0) for j, p in enumerate(parts) if j != pos)
                places.append(frames(pos, where, found[0], found[1], others, base.get(part, 0.0)))
            while places:                                  # one round: the next frame of every place, in order
                left = []
                for place in places:
                    entry = next(place, None)
                    if entry is None:
                        continue
                    left.append(place)
                    key, a, e, exists = entry
                    if exists:
                        a = max(0.0, a)
                    if a > 0.0:
                        ranked.append((key, a))
                        if exists:
                            gens[key] = (context, e)
                            grounded[key] = a
                        else:
                            virtual[key] = (a, context, e)
                        if len(ranked) >= F:
                            return ranked, gens, virtual, grounded
                places = left
        # the categories of what is present: their activation is summed over the units in the order of ``base``
        global _RANK_SCRATCH
        if _RANK_SCRATCH.size < S.n_units:
            _RANK_SCRATCH = np.zeros(max(S.n_units, 2 * _RANK_SCRATCH.size), dtype=np.float64)
        scratch, held = _RANK_SCRATCH, []
        try:
            for unit, a_u in base.items():
                found = categories(unit)
                if found is None:
                    continue
                cats, es = found
                running = np.minimum(1.0, scratch[cats] + a_u * (2.0 * es - 1.0))
                scratch[cats] = running
                held.append((unit, cats, es, running))
            if not held:
                return None
            every = np.concatenate([cats for _, cats, _, _ in held])
            whose = np.concatenate([np.full(len(cats), unit, dtype=np.int64) for unit, cats, _, _ in held])
            first = np.lexsort((whose, every))             # a category has its rank from the smallest unit it is of
            owned, at = np.unique(every[first], return_index=True)
            owner = whose[first][at]
            turns = []
            for unit, cats, es, running in sorted(held, key=lambda found: found[0]):
                turns.append((unit, cats, np.lexsort((cats, -running, sizes_of(cats))).tolist()))
            number = 0
            while True:
                more = False
                for unit, cats, turn in turns:
                    if number >= len(turn):
                        continue
                    more = True
                    category = int(cats[turn[number]])
                    if int(owner[np.searchsorted(owned, category)]) != unit:
                        continue
                    a = float(scratch[category])
                    if a > 0.0:
                        best = None
                        for other, c2, e2, _ in held:      # the instance it is most surely a category of
                            hit = np.flatnonzero(c2 == category)
                            if hit.size and (best is None or float(e2[hit[0]]) > best[1]):
                                best = (other, float(e2[hit[0]]))
                        ranked.append((category, a))
                        gens[category] = best
                        grounded[category] = a
                        if len(ranked) >= F:
                            return ranked, gens, virtual, grounded
                if not more:
                    return None                            # every candidate listed and the focus is not full
                number += 1
        finally:
            for _, cats, _, _ in held:
                scratch[cats] = 0.0

    def _rank_all(self, base, parts, context, fresh, F):
        """``_rank`` in full: every candidate built and ordered, and the EXPECT(lag 0) spreading."""
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
                  "unknown", "voice", "witnesses", "episode_start", "company", "topic", "open_where", "taking",
                  "heard", "about", "claimed", "faults", "_held_back", "open_plain", "met_once", "wondered",
                  "called", "trial_base", "trial_changes", "trial_cases", "trial_mark", "relations", "own_taken",
                  "told_in", "own_waited",
                  "relations_tried", "relation_paths", "pondering", "linked", "waiting", "conjectures",
                  "pairs_tried")

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
        theta = self.knob["theta"]
        for name, value in entry.undo.items():
            setattr(self, name, value)
        if getattr(self.cfg, "trials", False):
            from . import deney
            deney.sync(self)                               # F57: its ways as its record of their changes says
        if self.knob["theta"] >= 1e8:                      # the undone moment was lived while listening: whoever spoke
            self.knob["theta"] = theta                     # had raised the act threshold for that moment only and has
                                                           # set it back since; the state before the moment is the
                                                           # threshold the mind has now, not the raised one
        self.trust = self._trust()
        self.__dict__.pop("_between_memo", None)
        self.__dict__.pop("_episode_memo", None)
        self.__dict__.pop("_situation_cache", None)        # (kept by unit and moment: both come again after an undo)

    def retract(self, m: int) -> Dict[str, int]:
        """Retract moment m at the level of the ledger: exactly its increments are removed."""
        report = self.S.retract(m)
        self.trust = self._trust()
        self.__dict__.pop("_between_memo", None)
        self.__dict__.pop("_episode_memo", None)
        self.__dict__.pop("_situation_cache", None)
        if "personal_log" in self.__dict__:
            from .iliski import forget_moment
            forget_moment(self, m)
        return report

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_facet_cache", None)                   # caches, rebuilt on demand
        state.pop("_situation_cache", None)
        state.pop("_between_memo", None)
        state.pop("_episode_memo", None)
        state.pop("_lexicon", None)
        state.pop("_lexicon_index", None)
        state.pop("_why", None)
        state.pop("_explain_on", None)
        state.pop("_aileler", None)                       # F53: the index of families (read from the frames again)
        state.pop("_questions", None)                     # F51: the questions it knows (found again)
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
