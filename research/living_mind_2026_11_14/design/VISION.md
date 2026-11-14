# Vision (owner's direction, 2026-11-14) — the frame for every design decision

The owner's "red" example is the visible tip. The thing wanted is not a better learning algorithm; it is a MIND
rather than a MODEL. Design briefs that reduce it to six mechanisms (P1–P6) are too narrow. The wider frame:

1. LIFE, not dataset. The system exists in time. What it meets are experiences in an order, with a before and an
   after; it has a history of its own (a biography), not a corpus it was fitted to once ("neural snapshot").
2. PROCESS, not function. It is not input → output. It runs continuously, also when nothing is asked: it
   re-thinks, connects, imagines, doubts, consolidates, forgets, wonders.
3. A WORLD IT BUILDS ITSELF. Concepts, categories, relations are its own constructions, formed when needed,
   reorganised when experience demands. No fixed embedding table, no fixed vocabulary of meanings.
   3b. A CONCEPT IS ITS LIVED ENCOUNTERS (owner, 2026-11-14). "Red" heard or read, then seen, then met in a
   tomato, in blood, on a book cover: each is a NEW experience that develops the concept, not another labelled
   sample of a fixed class. A concept is therefore not a point (vector) and not a definition. It begins as an
   almost empty handle (a word with weak conviction) and is filled by encounters across channels and contexts;
   each encounter adds a facet and ties (tomato: ripeness, taste; blood: body, danger; cover: object, design),
   and the concept keeps the traces of the encounters that made it. Recognising uses the concept AND changes
   it. Two minds with different lives have different "red". There is no final state of a concept.
4. CONVICTION, not probability of the next token. Every belief has a strength earned by experience; first
   hearing is weak; contradiction refines; generalisation feeds back. Decisions come from accumulated conviction,
   not from a softmax over a frozen snapshot.
5. IMAGINATION WITH A REALITY SENSE. It constructs (hallucinates) freely, looks at its own constructions from
   outside, measures how real they are against what it has lived, and gets better at constructing.
6. AIMS OF ITS OWN. Gaps in its world produce questions; questions direct what it attends to and seeks. It
   wants to know.
7. AN OPERATING SELF. Something observes and steers the activity (what to attend, when to imagine, when to ask,
   how much to trust itself), and THAT improves with experience and carries over to subjects it never met:
   it learns how to learn, how to think.
8. DEVELOPMENT. It starts small (few innate operations) and grows: more concepts, better construction, better
   judgement of itself. The measure of success is developmental, not a held-out accuracy: after living N
   experiences, is it a different, more capable, more coherent mind than at birth — and still the same one?

9. IMITATION FIRST (owner, confirmed 2026-11-14). A human first learns by copying what it sees and hears, one
   to one: speaking, walking, everything. The mind should do the same: at the start it reproduces what happens
   around it (observed sequences, observed acts of others, including how others ask and correct), and this
   bootstraps and speeds up everything else. Imitation is the first developmental stage, not the whole: what is
   copied is then weighed by the reality sense (5), rebuilt as own concepts (3) and steered by the operating
   self (7). (Honest note: next-token prediction is itself pure imitation of text, frozen afterwards. The new
   thing is imitation as the opening stage of a living process that goes on to judge and rebuild what it copied.)
   9b. DETACHING FROM WHAT WAS COPIED (owner, 2026-11-14). A human later evaluates what it learned by copying,
   reaches a conviction that it is right or wrong, and can abstract itself from that act or behaviour. So copied
   material enters as "taken over from others, not yet judged", separate from "what I hold". Over time each
   copied piece is judged against lived experience: adopted as its own (conviction rises), kept as "others do
   this, I do not", or abandoned. This needs (a) provenance on everything learned (copied / observed / inferred
   / imagined), (b) a self–other distinction, (c) the ability to stop acting on a copied pattern without losing
   the knowledge that others act so, i.e. retraction of the behaviour, not erasure of the memory.
10. ANY SENSE (owner, confirmed 2026-11-14). Later, sensors will be attached: cameras and lenses, microphones,
   chemical/odour receptors, anything. The core must not be built around text tokens. An experience is a
   time-stamped event from any channel; concepts bind across channels (the word "red", the seen colour, the
   heard word are one concept). Text is only the first channel used for testing.

11. ACTING IN A WORLD (owner, 2026-11-14). Later the mind gets a wide playground: APIs, remote control of
   devices, the internet. It will search, read, try things and develop by itself. So outputs are not "text
   answers" but ACTS on channels, symmetric to senses: an act is an event the mind emits on an action channel
   (say something, call an API, move a device), and what comes back is experience.
12. ITS OWN BODY (owner, 2026-11-14). It perceives its own hardware (memory, compute, speed, storage) as part of
   its world, works according to it, configures itself to it, and can reach the conclusion that it needs more.
   So the substrate is a sense channel too (interoception): slowness, fullness, failure are felt signals.
13. NOTHING OF THIS WRITTEN AS AN ALGORITHM (owner, 2026-11-14). Browsing, researching, reading its hardware,
   reconfiguring itself must not be hand-coded procedures. The core has only general capacities (perceive, act,
   predict, imitate, judge, want, attend); every specific skill is something the mind acquires by living with
   the channels it is given. Design test: adding a new sense or a new action channel requires NO new code in
   the core, only a channel adapter.

14. LEARNED INTO THE CORE, NOT INTO A MEMORY (owner, 2026-11-14). When it needs to learn something new, doing a
   piece of research should be enough, and what it learns is not an entry in a memory that is looked up; it is
   knowledge acquired in the core itself. So there is no "frozen model + external store" split (no RAG, no side
   ledger as the place where knowledge lives). The core's own substance is what grows and changes. A short-lived
   trace of an episode may exist, but its destiny is to be absorbed into the core's concepts and convictions.
   Hard problem to face honestly: writing many things into a fixed set of weights destroys what was there
   (measured in this repository: 52 naive writes into one layer collapsed the model; an external store avoided
   it). The design must get the stability of a store with the integration of weights: a core that GROWS
   (new units and ties for new knowledge) and consolidates, instead of overwriting a fixed tensor.
15. NO EXAMPLE DOMAIN IS SPECIAL (owner, 2026-11-14). "Red" was an illustration. Nothing in the core may be
   specific to colours or to any domain; acceptance tests must run on domains and kinds of concepts (objects,
   relations, acts, abstract ideas) never seen while the design was developed.

Oversight (design requirement from the lead, to be agreed with the owner). A mind that acts on the internet and
on devices, reconfigures itself and can seek resources must act through the governance layer the owner already
asked for: channels are granted by the owner; consequential acts (spending, acquiring compute, changing its
own configuration, acting on real devices) need the owner's approval; every act is in the audit log; there is
a stop that always works; the first playground is a sandbox. This is part of the architecture, not a brake
added later: the action interface is the single place where acts leave the mind.

Consequences for the design:
- Senses, acts and the body share ONE event interface; the core never sees "text" or "API", only events on
  channels with provenance and time.
- No text-specific assumption in the core: a sensory front end turns each channel into events of one common
  form; adding a channel must not require redesigning or retraining the mind.
- A birth stage of imitation: the first competence is reproducing observed sequences; tests must show that
  imitation speeds later learning and that copied-but-false material is later corrected rather than fixed.
- The unit of design is a LIFE CYCLE: what exists at birth; what one moment of experience does; what it does
  when idle; how it grows; how it stays stable and itself.
- Evaluation is longitudinal: curves over a lifetime (knowledge, construction accuracy, calibration of
  self-trust, question quality, transfer to a new domain, identity/continuity), against snapshot-trained
  baselines given the same experiences.
- It must remain an engine: tensors, equations, an implementable algorithm, language in and out eventually.
- Honesty: whether such a system "is conscious" is not something this project can test or claim. What can be
  built and tested is the functional loop: self-observation, self-directed activity, development.
