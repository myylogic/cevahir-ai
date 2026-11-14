# Zihin — layers of one mind (owner's direction, 2026-11-14; design addendum, not yet built)

Owner: "A human's inner voice is one layer and the outer voice another; the body has its own automatic controls.
I do not think about breathing or blinking, yet models in my mind do these tasks when their triggers come (oxygen
falls in the lungs, a signal goes to the diaphragm, the breath is taken, like a machine). We are the sum of these
layers. We are turning this layering and art into mathematics."

This file states how the layers are expressed in the existing substance (units, ties, one evidence rule) WITHOUT a
separate module per layer and without writing any specific skill (vision point 13). A layer is a regime of
processing over the same structure, distinguished by channel and by speed, not a new kind of knowledge.

## L1. Deliberate layer (what is "in mind")

The focus: k-winners with F places (section 2 of LEAD_DESIGN). Slow, serial, limited. Everything new, uncertain
or surprising is handled here. Learning, recruitment and slot induction happen only for what is in the focus.

## L2. Automatic layer (habits and reflexes that run without attention)

A tie (S -> X), S a trigger (a unit of any channel, including the body), X an act or an internal act, is
AUTOMATIC when it is reliable, lived and needed:

    auto(S -> X)  <=>  e(S -> X) >= e_auto  and  use(S -> X) >= m_auto  and  r(S) >= r_auto

(r = the F9 reliability of the context, so only contexts that have proved predictive can be automated).
Automatic ties are applied every moment in full, as one sparse matrix-vector product over ALL active trigger
units, with no competition for the F focus places:

    act_auto = { X : sum_S a_S (2 e(S -> X) - 1) >= theta }     (outside the kWTA)

so many automatic processes run in parallel while the focus is free for something else. The one learning rule
still runs on them (their outcome adds evidence as for any tie). De-automation is not a separate rule: when an
automatic tie's prediction fails, the failure is surprise (section 2.6), surprise is an event that wins the focus,
and the trigger with its act are back in L1 until they are reliable again. Chunking (section 3) is the same idea
for sequences: a reliable chain is compiled into one unit; automation is a reliable trigger -> act link compiled
out of the focus.

Innate reflexes. A body adapter may supply a few ties with innate evidence (trust_prior), e.g. (body signal
"oxygen low" -> act "breathe"). They are ordinary ties: the life can strengthen, weaken or override them; they are
automatic from birth only because their evidence starts above the thresholds. This is the only place where the
designer gives behaviour, and it is the body's, not the mind's.

## L3. Inner voice and outer voice

Two channels that share the same concept units:

- inner speech: an internal channel of kind "act" whose emissions never leave the mind. Its echo comes back as
  an event with provenance "imagined" (section 6), so thinking in words is acting on the inner channel and
  hearing oneself; its conclusions are scored for reality like any construction.
- outer speech and every outward act: act channels that pass the oversight gate (owner permissions, approvals,
  audit log, stop).

The same symbol unit is active when heard, said inwardly or said aloud; the provenance and the channel tell them
apart, as "others do X" and "I do X" are told apart in section 7. Whether to say a thought aloud is itself an act
decision (pooled expectation of relief), learned from outcomes.

## L4. Body regulation

Interoceptive channels (load, memory, speed, failure; later sensors of the hardware) give body units. Regulation
is L2 applied to body triggers: (body state -> internal act such as prune, slow down, ask for resources). Acts that
change the mind's resources outside the sandbox are outward acts and pass the gate.

## L5. Drives and will (owner, 2026-11-14)

Owner: "By nature a human tends to mate and to chase pleasures; strong reinforcers such as serotonin and dopamine
wrap the mind. Why does not every mind act with the same strength or direction? These directors are models too.
But the upper consciousness can, when needed, show the will to take the more likely and the right decision."

The owner's example is an illustration and nothing in the core is about it (vision point 15, as with "red").
What is modelled is the STRUCTURE of a human mind: directors (innate value signals, each with its own gain), the
lived evidence that shapes how each director acts in this mind, and an upper layer that weighs consequences and can
outweigh a director. Any director, any value, any body.

In the substance:

- Value signals are interoceptive channels like the feelings: v_1 ... v_m, positive or negative, defined by the
  body adapter, never by the core. A body adapter may give a few innate ties with innate evidence (trust_prior),
  e.g. (sensed X -> v_j). Below, "pleasure" stands for any positive value signal and "harm" for any negative one.
  The innate GAIN of each value channel (how much one pleasure event counts) is a body constant that differs
  between minds: temperament. Same rule, different gains, different lives -> different strength and direction.
- A drive toward an act is the pooled expectation (S, X) -> pleasure at lag 1 (immediate). Consequences are ties
  at longer lags: (S, X) -> harm / relief / pleasure at lag L > 1, learned by the same rule when the outcome
  arrives later (the episode keeps a trace of its acts; the trace is the only thing a longer lag needs).
- Will is the deliberate layer weighing the act before it is emitted. The act's accumulated evidence over the
  settling iterations is

      A(X) = sum_t [ (2 e_now(S, X) - 1)  +  sum_L gamma^L (2 e_L(S, X) - 1)  +  imagined(S, X) ]

  where e_now is the immediate drive, e_L the learned later consequences discounted by gamma (a knob), and
  imagined(S, X) the reality-scored constructions of where X leads (section 6), each weighted by self-trust.
  X is emitted only if A(X) crosses theta. A strong immediate drive with a reliably bad later consequence does not
  cross: that is a veto. An automatic act (L2) is subject to the same veto as soon as the conflict (high drive and
  high harm expectation at once) is felt, because conflict is an event that wins the focus and brings the act back
  under deliberation.
- Differences between minds are therefore of three measurable kinds: innate gains (body), lived evidence (biography)
  and learned knobs (gamma, theta, self-trust: how far ahead and how cautiously this mind decides). None is written
  as a rule for a specific behaviour.

Test T12 (drives and will): two minds with different innate pleasure gains living the same stream act with
different frequency (temperament); an act with immediate pleasure and a delayed harm is first taken, then vetoed
after the delayed consequence has been lived, while the immediate pleasure expectation stays as it was (the drive is
not erased, it is outweighed); an ablation without consequence lags never stops.

## L6. Meaning of consequences: the same lesson, weighed differently in another context (owner, 2026-11-14)

Owner: "The secret of learning while living is observing the consequences of an experience, making sense of them
and evaluating them. A child loves how fire looks, holds a burning stick, its hand burns; it notices that fire
harmed it and does not take fire in its hand again. The same person as an adult may choose to enter a burning
house if someone they value is inside. Composite behaviour. The deciding consciousness making sense of what it has
acquired is an art of its own." (The example is an illustration; the structure is what is built.)

In the substance this needs nothing new beyond L5, only that its parts are kept apart:

- The lesson is evidence on a consequence tie, and it generalises like any knowledge: (holding X) -> harm, lived on
  one instance, reaches other instances through categories (section 4) and the F9 reliability of the frame.
- The lesson is never erased by a later choice. Entering the fire does not subtract from (fire, enter) -> harm;
  that evidence stays exactly as lived (like "others do X" in T7, measured bit-identical).
- A decision is the sum over the consequences that the CURRENT context (the composed situation unit and its
  generalisations) leads to, each weighted by the value it brings:

      A(X | context) = sum_t sum_j value_j * sum_L gamma^L (2 e_L(context, X -> v_j) - 1)  (+ imagined terms)

  Values are not constants either: how much an outcome matters (value_j) is itself evidence (which outcomes were
  followed by relief or by loss in this life), so what a mind is willing to bear grows with its biography.
- "Composite behaviour" is the composition of situations: the larger context (house, fire, someone valued inside)
  is a composed unit with its own consequence ties; its expectations are read before the generalisation (fire ->
  harm) as the specific is read before the general everywhere in Zihin (section 4, nested and F9 readings).

The child and the adult are the same person (owner). In Zihin terms: one substance and one biography. The adult's
decision is taken by the very structure that still holds the child's lesson; there is no reset, no retraining, no
second model. What changed in between is only what was added: more composed situations, more consequence ties,
learned values and knobs. T13 therefore runs in ONE life (the avoidance phase and the later choice are moments of
the same mind), and the continuity checks of T5/T9 (early evidence bit-identical, exact ledger) apply across it.

The central sentence (owner): "the deciding consciousness making sense of what it has acquired is an art of its
own". In Zihin: ACQUISITION and MEANING are two different things. Acquisition is evidence, lived once and kept as
lived (additive, exact, retractable). Meaning is constructed by the decider each time it reads that evidence:
which generalisation speaks (F9 reliability), what the consequences are worth (values), how far ahead it looks
(knobs), in which context. Meaning-making is itself learned and grows (reliabilities, values and knobs are evidence
too), so the same acquisition means more to an older mind. Measured gap (oracle_t3.py, 600-experience lives, 5 seeds):
the right knowledge for a held-out instance exists in the core in 100% of probes and is correct in 80%, but the
reading selects it in only 30-53%; the answers are right in 51-63%. The missing part is meaning, not acquisition.

Test T13 (consequence meaning): after one own harmful outcome the act is avoided in new instances of the same kind
(generalised avoidance); in a context whose stake has a higher learned value the act is taken although the harm
expectation is unchanged; the harm evidence is bit-identical before and after that choice.

## What would make this testable (to be added to the benchmark)

- T10 automation: after an act has been reliable in a situation for long enough, it is emitted while the focus is
  occupied by an unrelated task (dual-task), and the unrelated task does not get worse; when the situation
  changes so the act fails, the act returns to the focus (it stops being automatic) and is relearned.
- T11 inner voice: a conclusion constructed on the inner channel is used later without being said aloud, and is
  never emitted on an outward channel unless the act decision chooses to.

Status: design only. Nothing above is implemented yet; F9 and T7 come first.
