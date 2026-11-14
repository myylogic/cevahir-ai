# Zihin phase 1a — running notes (decisions, findings, open points)

Branch `feat/zihin` (from main 82d4b8f). Specification: VISION.md and LEAD_DESIGN.md (design v0 "Zihin") of
2026-11-14. Code: `zihin/` (core), `tests/evolution/test_zihin_*.py` (unit tests),
`research/living_mind_2026_11_14/` (test world, runs, this log).

How to read this file: section 1 is the map of the code, section 2 the reading of the design that was built
(every place where the text left a choice), section 3 the findings (where the design did not work as written,
smallest repro, smallest principled change, both kept behind switches), section 4 the constants and why,
section 5 the measured results, section 6 what is missing and what phase 1b should do first.

## 1. Module map

| file | content |
|---|---|
| `zihin/events.py` | event (channel, symbol, provenance, intensity), channel kinds, open alphabet per channel |
| `zihin/evidence.py` | frequency, conviction, expectation; the dyadic grid that makes sums exact |
| `zihin/store.py` | unit table, tie table (COO) that grow; additive experience log; `retract`, `undo_last`; caches rebuildable from tensors |
| `zihin/learn.py` | the one rule: lag-1 and lag-0 EXPECT learning, induction up, learned trust (`judge`) |
| `zihin/form.py` | recruitment of composed units, slot / frame induction with emergent categories, chunking |
| `zihin/read.py` | support down (effective evidence), predictions, drive, conflict (read only) |
| `zihin/act.py` | act decision (accumulate to threshold), imagination (flag), control knobs (flag) |
| `zihin/mind.py` | `Config`, `Mind.step` = one moment; feelings; undo, retract, clone |

## 2. Reading of the design that was built

Events. `Event(channel, symbol, prov, intensity)`; time is the moment of `Mind.step`. An atomic unit is keyed
by (channel, symbol). Channel kinds: sense, act, feel, knob, source. The core knows only the kind.
Provenance is an attribute of the EVENT (given by the channel adapter): the same channel can carry what
another says (`told`), what another is observed doing (`imitated`), what the world shows (`sensed`) and what
the mind did itself (`acted`, the echo of its own act).

Evidence. Six classes as in the design. Stored evidence is raw; trust is applied when a tie is read. Trust of
`sensed` and `acted` is 1 by definition (direct experience; they are the judges). Trust of every other class
is the expectation of the tie (source unit of the class -> "holds"), an ordinary EXPECT(lag 0) tie whose
evidence is written by `learn.judge`. With no evidence that expectation is 1/2, so at birth told = imitated =
inferred = imagined = 0.5; this is not a chosen constant, it is the value of the rule with nothing lived.
(The design's example value 0.3 is used in one unit test through `trust_fixed`.) Per-teller trust is NOT built:
evidence has one slot per class, not per source (open point 6.x).

Judging (learned trust). When direct experience adds an amount d to a tie, the evidence of each learned-trust
class on that tie that has not been judged yet is judged in the share min(1, d): what agrees with the direct
experience is positive evidence for "source holds", what disagrees negative. The judged mass is kept in two
more additive tables per tie (`checked`), so each claim is judged once however often the fact is sensed again.

Episode. The design says "if the focus holds a tuple matched by no composed unit, recruit". Built as: the
items (events of sense / act channels) since the last silent moment form the episode tuple (at most F items);
its composed unit is recognised or recruited at every moment (so the composed units are the prefixes of
utterances). A moment without outside events ends the episode. Feelings are not items of tuples.

Activation. Perceived units get their intensity; items and recognised tuples that stay in the focus are
carried with decay d per moment; a recognised composed unit has the mean activation of its parts (bottom-up
through PART). Categories get a_instance * (2e - 1) through ABSTRACT; a frame (tuple with a category in place
of an item) gets the mean of its parts. Derived activations are recomputed every moment (not integrated),
because the literal formula "a <- kWTA(decay a + in + lateral)" iterated T times saturates everything at 1.

Learning (step 5). Sources: grounded units of the last focus. Targets of lag-1 ties: the atomic units that
occur now, with evidence g * a_u * intensity. A tie whose target did not occur gets negative evidence
g * a_u * e only if the target's CHANNEL reported something else in this moment (a silent channel is not a
failed prediction). Lag-0 ties are learned among ATOMIC units of the focus only (composed units and categories
are already tied to their members by PART / ABSTRACT).

Induction up (section 4). A generalisation that is in the last focus because of an instance acquires what the
instance tie acquired, capped at one experience per instance tie and discounted by the ABSTRACT expectation,
in the provenance class of the confirming event. (Generalisations are therefore NOT also learned by the
ordinary positive rule; doing both, as the two sections literally say, would count everything twice.)

Support down. Computed at read time (`read.effective`), never stored, so it needs no retraction of its own.

Acts. An act unit is an atomic unit on a channel of kind `act`; the same unit is active when others emit the
symbol and when the mind does. Decision: pooled expectation of imitated evidence on S -> X and own outcome
evidence on (S, X) -> relief; accumulate 2e - 1 per settling iteration; emit when the sum reaches theta
within T iterations. Own repetitions (class `acted` on S -> X) are a record, not a reason.

Feelings. surprise (as in the design, capped at 1), relief = sum over EXPECT ties that got evidence in this
moment of use * (conviction after - conviction before), squashed r / (r + k); load = ties / body budget;
conflict = sum of c * (1 - |2f - 1|) over the lag-1 ties of the context unit, squashed. Emitted as events of
the channel `feel` with the amount as intensity; provenance `acted` while an own act awaits its outcome.

Retraction. Every increment of a moment is logged. `retract(m)` subtracts exactly these increments (ledger
level); structure born in m that no other moment used is tombstoned. `undo_last()` additionally gives the rows
back and restores the working state, so that the whole state is bit-identical to the state before the moment.
Retraction is NOT a counterfactual replay: later moments computed their increments in a world that contained
m (measured in T9 as the "counterfactual gap").

## 3. Findings (design as written vs. what was needed)

(filled while building; each item: what the text says, smallest repro, smallest principled change, switch)

F1. Feelings and lateral activation capture the focus. As written, the four feelings enter every moment like
any event and EXPECT(lag 0) activation is added to every unit. Repro (`dev1` scenario, 21 statements): from
the third statement on the focus holds surprise, relief and the first four words ever heard, all at 1.0,
sustained by each other; new composed units never enter the focus, so nothing is learned about them
(prediction for a told fact: no tie at all). Change: `evoke="grounded_first"` (what is present keeps the
activation of its presence and is ranked before what is only called to mind) and `feel_focus=False`
(feelings are outcomes that ties can point to, not contents of the focus). Both kept as switches.

F2. Evidence of a general context swamps specific knowledge when expectations are summed ("votes", the
settling formula used as prediction rule). Change: `predict="specific"`: read the most specific unit of the
context (the composed unit of the episode) with its support.

F3. Support down as written lets a strong category override an exception: W_eff = W_own + lam e W(g -> v) grows
without bound with the number of confirming instances. Change: `support="capped"`: one generalisation counts
at most lam * e(i -> g) experiences.

F4. Induction as written counts confirming instances only: the tie g -> v never receives negative evidence
from the instances of g that hold something else, so every value any member has looks like a rule (f = 1).
Change: `general_miss=True`: a generalisation in the focus gets the ordinary negative evidence for what it
expected and did not occur.

F5. The context of the previous utterance (decay 6/7 per moment) outranks everything derived in the next
utterance and creates ties across unrelated utterances. Change: `clear_on_pause=True`.

F6. A copied act that never brings relief can never be given up as written: negative evidence needs an
existing expectation, and the tie (S, X) -> relief does not exist before relief has occurred once. Change:
`act_expect_relief=True`: acting is itself an expectation of relief, with the expectation the mind acted on.

F7. (found after the first lives, 2026-11-14) Learning and reading disagree about which generalisations count.
   A generalisation (frame or category) learns only while it holds a place in the focus (F = 7), but an instance
   is READ with all of its generalisations (read.generalisations). A frame that won a place once collects positive
   evidence and is never judged again, so it reports f = 1.00 for whatever it saw that once (white box, nitelik
   seed 0: frames (C71, dim) -> yafi f 1.00 c 0.33, (ent, C266) -> labomac f 1.00). Read together, these spurious
   frames push wrong values above 1/2 and an instance never told the dimension is answered at chance (T3 held-out
   0.47-0.67). Change (miss_scope = "gens", variant "symmetric"): every generalisation an instance is read from
   is judged by the same episode (positive for what occurred, negative for what it expected and did not occur).
   After it the same frames are honest (class frame -> rule value f 0.83 c 0.75; spurious frames f 0.28-0.38 for
   values). Second finding in the same probe: the "nested" combination (general -> specific by extension) lets a
   weaker but smaller-extension generalisation pull the right value back below 1/2; pooling capped evidence
   (support = "capped", variant "symmetric_pooled") is the alternative measured side by side.

F7 outcome (2026-11-14, seeds 0-1, exploratory). Negative. Judging every generalisation (symmetric) makes the
   frames honest but held-out T3 gets WORSE than changed (nitelik seed 0: told rule 0.83 -> 0.33, induced 0.67 ->
   0.00; iliski similar) and a moment costs 21-34 ms instead of 8. Readings tried on top: nested (default),
   pooled capped evidence (rejected: early recall 1.00 -> 0.10, members 0.77-0.88) and "choice" (the most lived
   generalisation speaks; rejected: it picks the frame over ALL entities, which correctly reports the base rate,
   e.g. rule value f 0.23, and every value ends below 1/2). Diagnosis: this is the reference-class problem. The
   mind holds many overlapping classes that contain the instance (8 categories for one entity, 25 for one
   dimension word, because every frame position makes its own category); which class predicts the asked
   dimension is not chosen by size (nested) nor by amount lived (choice). Candidate F8 under test
   (probe_reading.py): the generalisation that pins down a value most decisively (max lam e c f over sense/act
   targets) speaks. Symmetric runs were stopped after seeds 0-1; their records are kept as measured.

F9. (2026-11-14) Which classification predicts which dimension is learned, by the same rule. The mind holds
   many overlapping classes for one instance (F7 outcome); neither size (nested), amount lived (choice) nor
   decisiveness (sharpest, F8) is a principled way to choose. F9 applies C3 to the mind's own generalisations:
   when a composed instance lives a value v on a sense/act channel, each generalisation g it is read from is
   judged by the expectation it held for v before that moment (w+ += e(i->g) e(g->v), w- += e(i->g)(1-e(g->v))
   on a self-tie of a new form RELY, provenance "inferred", logged like all evidence). Reading ("reliable"): the
   generalisation with the highest e(i->g) x reliability speaks for an instance; own evidence comes last. Nothing
   in the core names a dimension or a class; reliability is a property each generalisation earns in the life.
   Measured on 600-experience lives (5 seeds, 2 domains) and, following the owner's criterion (the outcome at the
   end of a life matters, not the speed of learning), on 1500-experience lives against changed (results_long/).

F10. (2026-11-14) Acquisition versus meaning (owner: "the deciding consciousness making sense of what it has
   acquired is an art of its own"). oracle_t3.py shows the right knowledge for a held-out instance is in the core
   (class frame exists 100%, right on top 80%) but is selected in 30-53%. Four steps tried, all logged:
   (a) F9 judged with e(g -> v): reliabilities stay at 1/2 because most generalisations have no opinion on v when
   judged (raw w+ = w- = 1.45 on average); (b) reflection: idle moments re-read lived instances with the present
   generalisations, leaving out the instance's own contribution (reflective); (c) trusted pooling: generalisations
   pooled like provenance classes with tau_g = max(0, 2 r_g - 1) (trusted); (d) verdicts only where g has an
   opinion on the channel, w+ = c f(g -> v), w- = c (1 - f). Right answers 58-73%, no reading uniformly better.
   Remaining suspect: frames outside the focus never get negative evidence (F7), so their f is not meaningful;
   honest_reflective / honest_trusted (F7 + F10) under test.

F10b. (2026-11-14) Relative verdicts (share of g's evidence that went to the lived value) and "only generalisations
   with reliability > 1/2 speak": measured WORSE on 600-experience lives (reflective iliski 46/80 -> 26/80, nitelik
   41/60 -> 30/60; honest_reflective iliski 42 -> 20, nitelik 29 -> 25). Silencing unproven generalisations leaves
   most held-out answers at 1/2. Kept behind flags (verdict = "share", proven_only = True), default back to F10.
   Honest frames (F7) raise the right-on-top share of the class frame to 90-93% (oracle), but selection stays at
   35-43% in every reading tried. Records: results/oracle_t3.jsonl.

F10c. (2026-11-14) Sleep as re-reading (variant "sleeping": reflect 8 instances per idle moment, a night of 50 silent
   moments every 100 experiences): iliski 43/80, nitelik 40/60 right, no better than daytime reflection (46/80,
   41/60). Five readings and two judging rules have now left selection of the right generalisation at 35-45%.
   Conclusion: the reading is not the bottleneck; the STRUCTURE being read is. One category per slot position gives
   about a thousand frames in 600 experiences, many with identical members or resting on a few observations, all
   competing with the class frame. Next: consolidation of categories (sleep merges categories with the same members).

F11. (2026-11-14) A category speaks with what its members have lived, read at read time. Found from the honest
   frames (F7): reliability ranked the class frame first, but its absolute frequency for the right value had eroded
   to 0.30-0.36, because induction caps positive evidence at one experience per instance while negative evidence
   from failed predictions is not capped and grows with every repetition. Change (support = "members"): a frame
   with one category position does not use its own accumulated counts; its expectation is computed when read from
   its members' own beliefs: W+(v) = sum_x e(x -> C) s_x(v), W-(v) = sum_x e(x -> C)(1 - s_x(v)), s_x the share of
   member x's own positive evidence that went to v; every member counts once with both signs; the instance being
   read is left out. The frame whose members agree most speaks (e(i->g) x conviction(members) x max f); own evidence
   comes last. No storage, nothing to retract. Held-out T3 on 600-experience lives (5 seeds, oracle_t3.py):
   nitelik 56/60 (93%), iliski 65/80 (81%), against the best earlier reading 44/60 and 46/80.

F12. (2026-11-14) Relations, not words, can be what members share (bind_repeats, terminal variant "speaking").
   A unit occurring at several positions of a tuple is one slot (a variable), so (kedi ne yapar kedi) and
   (kopek ne yapar kopek) give the frame (C ne yapar C). In the F11 member reading a member's value that IS the member
   counts as SELF and a value that is one of the member's own continuations (not a fixed word of the frame) counts as
   NEXT; reading an instance resolves SELF to the word in its slot and NEXT to that word's strongest continuation.
   Speaking uses the same reading. Nothing names a word or a language. First chat (results/second_chat.txt): after
   watching 5 dialogues three times and being told "kus oter", "at kisner", it answers "kus ne yapar" with "kus oter"
   and "at ne yapar" with "at kisner" (never watched), and "inek ne yapar" with "inek" and silence (nothing known).
F13. (2026-11-14) "Comes next" is learned only by what was the present a moment ago (next_only_present): the previous
   context and the previous item. As written every unit still carried in the focus learned lag-1 ties, so a context
   also expected the word two steps ahead (the question expected the answer's second word; "kedi" -> "icer" in free
   text). Found while building the chat; on in the "speaking" variant only so far; the T1-T9 lives have not yet been
   rerun with it.

F14. (2026-11-14) A failed prediction is recorded only where something was expected (neg_floor, speaking variant
   1/14 = 1/(2F), a default, not tuned). Found while reading real documents: a frequent word ("ve", "bir") has
   thousands of followers, and as written every moment wrote a small negative increment to all of them (and logged
   it), so time and the ledger grew with the mind (about 180 KB of ledger per moment at 1.3 million ties, 11 GB on
   disk before the disk filled). With the floor the structure is the same (identical number of ties after 500
   report sentences: 140 213), 100 sentences take 11.5 s instead of 40.2 s, and the ledger is about 13 KB per moment.
   Cost, stated: rare followers no longer collect negatives once their expectation is below the floor, so their
   expectation settles near the floor instead of near their true small frequency; predictions above 1/2, the F11
   member reading (positive shares) and acts are unaffected. The T1-T9 worlds have not been rerun with it.
Incident (2026-11-14, recorded as it happened): two loads of the same saved mind wrote into the same live ledger
   file (a fixed name); exporting a conversation copy overwrote and then deleted the live ledger of the running
   curriculum. The substance of that mind was intact (digest verified), its ledger was not; it was discarded and
   the curriculum is lived again. Fixed: every load gets its own live ledger file; minds can be loaded without a
   ledger for conversation; a mind saved without a ledger starts a fresh one when loaded (new experiences remain
   exactly retractable, the documents of its childhood are not).

F15. (2026-11-14) Speech judged by its consequences (L5 will, T7 applied to speech; owner: "if it could think it
   would say: I cannot lie down"; "I do not want a parrot"). A channel of kind "value" carries signals the adapter
   calls good or bad (the core never names them). Every composed context of the current episode is kept in a trace;
   a value signal adds EXPECT lag 2 ("later in this episode") evidence from each step of the trace to the value unit.
   Before saying a word the act's rate is the drive plus the consequences of the path "this episode, then this
   word": rate = (2e - 1) + sum_j sign_j (2 e(path -> v_j) - 1). Own repetitions are no longer a reason to speak
   (ACTED excluded from the act's drive, design section 7). Two session fixes found with it: the mind listens while
   spoken to (it no longer completes the partner's sentence into its own context), and an answered turn stays open
   for feedback and is closed by any other input. Result (results/third_chat_feedback.txt): after one "wrong" the
   copied answer is not said again; another answer that was approved is kept; the copied knowledge is still
   expected ("on" 0.79 before, 0.87 after). Records of the chat only; not yet a multi-seed test.

## 4. Constants

F = 7, T = 6, k = 1, lam = 0.5: design defaults. g = 1 (one event = one unit of evidence). theta = 1 (one unit
of accumulated net expectation; with T = 6 an act needs e >= 0.5 + 1/12). decay = 1 - 1/F (the time constant
of the context equals the focus span; no separate constant). induct_cap = 1 (an instance counts as one
experience). Evidence grid 2**-30 (exact sums below 2**22). e_chunk = 0.9, m_chunk = 5: NOT derived, knob
defaults; chunking is off in the world runs and tested in isolation. Nothing was tuned to a target number.

## 5. Results

(see RESULTS.md and results/*.jsonl; summary copied here when the runs are done)

## 6. Missing / open

(filled at the end)
