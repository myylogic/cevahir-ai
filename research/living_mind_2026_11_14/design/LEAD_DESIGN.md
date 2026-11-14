# ZİHİN — lead design v0 (2026-11-14): one substance, one rule, events on channels

Working name "Zihin" (mind). Source of requirements: VISION.md (15 points + oversight) in this folder.
Status: design to be built as a prototype and attacked by tests; not a claim of novelty (see §9).

## 0. The three commitments

C1 ONE INTERFACE. Everything the mind meets or does is an EVENT on a CHANNEL: (time t, channel c, symbol s,
   provenance π). Channels: senses (text first; later audio, vision, odour), acts (say, ask, call, move), and
   the mind's own body and state (interoception: surprise, relief, load, conflict) and its own control knobs
   (internal acts). The core never sees "text" or "API"; an adapter per channel turns raw data into symbols of an
   open alphabet (a new symbol is allowed at any time).
C2 ONE SUBSTANCE. Knowledge is a growing structure of UNITS and TIES. There is no separate memory, no
   embedding table, no fixed vocabulary, no frozen weights. New knowledge creates units and ties and adds
   evidence; nothing is overwritten.
C3 ONE RULE. Every tie carries EVIDENCE, and evidence changes at every moment by the same local rule, for
   facts about the world, for trust in sources (including trust in the mind's own inferences and imaginations),
   for acts and their outcomes, and for control knobs and their outcomes. Learning to learn is the same rule
   applied to the internal channels. Nothing specific (browsing, asking, reading its hardware) is coded.

## 1. State

Units u = 1..n (n grows). Each unit has: kind bit-flags that are consequences, not types chosen by hand
(atomic = created from a channel symbol; composed = an ordered tuple of other units; slot-bearing = a composed
unit some of whose positions are variables), activation a_u ∈ [0,1], use count, birth time.

Ties (sparse, directed). Three innate tie FORMS (the "few innate operations"):
- EXPECT u → v (lag ℓ ∈ {0 = holds together, 1 = follows}): "when u is present, v is present / comes next".
- PART (u, i) → p: the i-th part of composed unit u is p (structure: sequences, statements, frames).
- ABSTRACT u → g: u is an instance/special case of g.

Evidence on a tie is a vector over provenance classes Π = {sensed, told, imitated(observed in others),
acted(own outcome), inferred, imagined}:  w⁺[π], w⁻[π] ≥ 0 (float64 counts).
Trust τ_π ∈ [0,1] is NOT a constant: τ for a source is the expectation of the tie (source unit → "holds"),
learned by C3 (told statements later confirmed or refuted by sensed evidence change the trust in the teller;
inferred/imagined conclusions later confirmed or refuted change the trust in the mind's own construction).

  W⁺ = Σ_π τ_π w⁺[π],  W⁻ = Σ_π τ_π w⁻[π],  W = W⁺ + W⁻
  frequency  f = W⁺ / W            (how often it held)
  conviction c = W / (W + k)       (how much has been lived; k = 1)
  expectation e = c·(f − ½) + ½    (what the mind acts on)

First hearing is weak by construction: one told piece of evidence with τ_told = 0.3 gives c = 0.3/1.3 = 0.23.
Contradiction does not delete: it adds W⁻, f moves, c rises (more has been lived).

## 2. One moment of experience

Input: the events of this moment. Then:
1. PERCEIVE. Each event symbol activates its atomic unit (created if new). Context units stay active with decay.
2. RELATE (settle, T ≤ 6 iterations): a ← kWTA_F( decay·a + in + Σ_u a_u·(e_{u→v} − ½)·2 through EXPECT(lag 0)
   + up through ABSTRACT (instance activates its categories) + down (category expectations offered to the
   instance, see §4) ). kWTA_F keeps the F most active units: the FOCUS (F is a control knob; default 7).
   Composed units whose parts are all active become active (recognition); if the focus holds a tuple matched by
   no composed unit above vigilance ρ, RECRUIT a new composed unit for it (§3).
3. PREDICT. Units reached through EXPECT(lag 1) from the focus are the predictions for the next moment, with
   their expectations.
4. ACT (no softmax, no argmax over a vocabulary). Act units accumulate their incoming expectation over the
   settling iterations; an act is emitted when its accumulated evidence crosses the threshold θ (a control
   knob: caution). Nothing crosses → no act (silence is a legal outcome).
5. LEARN (always on). When the next moment arrives, for every tie u → v with u in the last focus:
     v occurred:                      w⁺[π] += g·a_u        (π = provenance of the confirming event)
     v was predicted, did not occur:  w⁻[π] += g·a_u·e_{u→v}
   For pairs (u, v) in the focus with no tie: create EXPECT u → v with evidence g·a_u·a_v (candidate tie).
   g is the plasticity gain (a control knob). Ties are only ever created between units that were in the focus
   together: the focus is the bottleneck that keeps growth O(F²) per moment.
6. FEEL. Internal events are emitted on the interoceptive channels and enter the next moment like any event:
   surprise = mean (1 − e) of what occurred unpredicted + mean e of what was predicted and failed;
   relief = decrease of the drive D (§5) caused by this moment; load = time and memory used (the body);
   conflict = mass of ties in the focus with f near ½ and high c.

## 3. Concepts are formed (recruitment, chunking, slots)

- Recruitment: an unmatched tuple in the focus becomes a new composed unit with PART ties to its members.
- Chunking: an EXPECT(lag 1) chain u₁→u₂→…→u_m whose ties all have e ≥ e_chunk and use ≥ m_chunk is compiled
  into one composed unit (a word from characters, a phrase from words, a habit from acts). Imitation is this:
  observed sequences become chunks; replaying a chunk on an act channel is copying.
- Slots (frames): two composed units of equal length that differ in exactly one position give a slot-bearing
  unit (frame) with a variable there; the fillers seen in that position get ABSTRACT ties to a new category
  unit for the slot. Categories therefore emerge from instances that occupy the same place.
- A concept is the unit plus everything tied to it: its encounters (composed units it is part of, with their
  channels and contexts), its expectations, its categories. It has no final state: every recognition adds
  evidence and may add ties.

## 4. Generalising and feeding back

- Up (induction): when instances i₁…i_m of a category g each hold EXPECT i → v, g acquires EXPECT g → v with
  evidence = Σ_i min(W_i, cap) discounted by the ABSTRACT expectations. A rule told about the category
  ("all … are …") is evidence on g's tie directly.
- Down (support): the effective evidence of an instance tie i → v is its own plus what it inherits,
     W_eff⁺ = W_own⁺ + λ · e(i→g) · W⁺(g→v),   same for W⁻,   λ = 0.5,
  provenance "inferred". So a category rule raises the conviction of the instance belief without the instance
  being told again, and an exception stays an exception (own negative evidence outweighs inherited support).

## 5. Wanting, asking, acting

Drive of a unit: D(u) = Σ_{ties t at u} use(t) · (1 − c_t)   (uncertainty weighted by how much the tie is
needed). A category with strong ties but few known instances, or an instance with a used but weak tie, has high
drive. Total drive D = Σ_u D(u) over recently used units. Relief = −ΔD is felt (§2.6).
Acts (say, ask, look up) are units on act channels. The mind first acquires them by imitation (it has observed
others emit "X nedir?" and be answered). Which act it emits is decided by §2.4 from EXPECT ties
(state units, act) → relief, learned by C3 from its own outcomes. A question is therefore an act that the mind
has found to be followed by relief when a drive of that shape is high. No question-generation procedure exists.

## 6. Imagining and measuring reality

When no external event arrives (idle), the mind runs §2 on its own predictions: the top prediction is fed back
as an event with provenance "imagined" on an internal channel, for up to H steps, starting from high-drive
units. Each constructed conclusion (a tie instance i → v reached by a chain of ties) gets a reality score
   R = e(i → v) computed from the evidence that does NOT lie on the chain that produced it (leave-path-out).
It is written with provenance "imagined"/"inferred" and weight ι·R. Later sensed or told evidence about the same
tie confirms or refutes it, and that outcome updates (by C3) the trust τ in the source "my own construction"
(per domain: the source unit is tied to the category of the content). The mind hallucinates freely, records
what it constructed as constructed, and its self-trust tracks how often its constructions came true.

## 7. Imitation and detaching from it

Observed behaviour of others is stored with provenance "imitated": chunks of acts with EXPECT ties
"in situation S, others do X". Whether the mind itself does X is a different tie: (S, X) → relief/harm with
provenance "acted". At birth, with no evidence of its own, acts are driven by the imitated ties (copying).
As its own outcome evidence accumulates it can outweigh them: the mind stops doing X while the knowledge
"others do X" remains with its evidence untouched. Adoption, "others do, I do not", and abandonment are the
three regions of (imitated evidence, own evidence).

## 8. The operating self

Control knobs κ = (focus size F, vigilance ρ, plasticity gain g, act threshold θ, idle horizon H, chunk
thresholds). Setting a knob is an internal act; surprise, relief, load and conflict are internal senses.
The same ties and the same rule learn "(internal state shape) + (knob act) → relief". Because these units do
not mention any domain content, what the mind learns about how to run itself carries over to a new domain
unchanged; that is the testable form of "it learns how to learn". Its body is the same: load and memory are
sensed; pruning (dropping ties with low c and low use, oldest first) is an internal act whose outcome is felt.
Oversight: act channels to the outside go through one gate with owner-granted permissions, approvals for
consequential acts, an append-only log and a stop (the governance layer of this repository).

## 9. What this is close to (to be checked by the literature map before any claim)

Evidence-valued beliefs, revision, inheritance, questions: NARS (Wang). Recruitment on mismatch, vigilance:
ART (Grossberg, Carpenter). Chunking and production of skills: Soar, ACT-R. Sparse sequence memory with
continual learning: HTM (Hawkins). Learning progress as intrinsic reward: Schmidhuber, Oudeyer. Trust in
sources: Bayesian source reliability models. Candidate differences, all unproven: (i) one rule over world,
sources, self-trust, acts and control knobs through the event interface, so that the learning strategy is
acquired by the same substance and measurably transfers; (ii) provenance-resolved evidence with detachment
from imitated behaviour; (iii) reality scoring of imagination with learned self-trust; (iv) exact
retraction of any single experience (evidence is additive), and (v) developmental evaluation against
snapshot-trained networks given the identical stream.

## 10. Engine form

All state is tensors: unit table (n rows), sparse tie tables (COO: src, dst, form, lag, w⁺[6], w⁻[6], use,
last), settling = sparse matrix–vector products + top-F selection; evidence updates are scatter-adds. Float64
evidence. Deterministic. Every experience's evidence increments are logged (moment id → tie increments), so one
experience can be retracted exactly by subtracting them. Complexity per moment O(F² + F·deg). Memory grows with
ties; bounded by pruning under a budget given by the body.

## 11. Phase 1 (build now, text channel only, sandbox world)

Build §1–§5 and §7's provenance split, with imagination (§6) and control knobs as acts (§8) behind flags in
simple first versions. Developmental benchmark on generated worlds with random vocabularies (no colours in the
core or in the acceptance tests): (T1) weak first storage and refinement without deletion; (T2) category and
slot formation from instances; (T3) support from category to instance and exceptions; (T4) questions emitted
when drive is high and they reduce it; (T5) no collapse and no forgetting after hundreds of experiences, no
lookup outside the core; (T6) a second, unseen domain is learned from fewer experiences than the first (the way
of constructing transferred); (T7) copied false material is later corrected and the mind stops acting on it
while still knowing others do it; (T8) imagination: share of constructions later confirmed, and calibration of
R, improve with experience; an ablation without reality scoring drifts; (T9) exact retraction of one
experience. Baselines on the identical event stream: a small Transformer trained online one pass, the same
trained offline to convergence on the stream so far (snapshot), and snapshot + test-time fine-tuning.
Refutation: if the snapshot baselines match the mind on T1–T9 curves, or the mind fails T5, the design is wrong.
