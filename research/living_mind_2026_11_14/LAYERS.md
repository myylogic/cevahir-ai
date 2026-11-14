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

## What would make this testable (to be added to the benchmark)

- T10 automation: after an act has been reliable in a situation for long enough, it is emitted while the focus is
  occupied by an unrelated task (dual-task), and the unrelated task does not get worse; when the situation
  changes so the act fails, the act returns to the focus (it stops being automatic) and is relearned.
- T11 inner voice: a conclusion constructed on the inner channel is used later without being said aloud, and is
  never emitted on an outward channel unless the act decision chooses to.

Status: design only. Nothing above is implemented yet; F9 and T7 come first.
