# Research release 2026.11.15 (v6)

- Git tag: `research-2026.11.15-v6` (to be created as a GitHub Release from main)
- Zenodo DOI: recorded here once the archive is published and verified
- All-versions DOI: [10.5281/zenodo.22864660](https://doi.org/10.5281/zenodo.22864660)

## The Cevahir Zihin theory: a Living Evidence Mind

This edition puts the theory of the new core first ([docs/ZIHIN_THEORY.md](../../ZIHIN_THEORY.md),
Turkish original [docs/ZIHIN_KURAM.md](../../ZIHIN_KURAM.md)).

The model in `zihin/` is not a Transformer. It has no weight matrices, embedding vectors, backpropagation or softmax
attention.

- **Ties and evidence.** It keeps every relation as a tie and keeps each tie's graded evidence separately by origin
  (sensed, told, imitated, acted, inferred, imagined). The trust in the four indirect origins (told, imitated,
  inferred, imagined) is learned from direct experience; sensed evidence and its own acts are the reference.
- **One rule.** It reads expectations from one rule, e = c(f − ½) + ½, the NARS truth-value expectation with k = 1.
- **Learning and forgetting.** It learns at every moment. While its ledger is kept, the evidence of any single
  experience can be retracted exactly; this is exact for the evidence, not a replay of a life without it.
- **Attention.** Its attention is a Seven-Place Focus: seven units win the focus each moment, so attention is local
  and readable. There is no context window; the time per moment grows only slowly with lived memory.

The theory gives names to its findings and credits related prior work (NARS for the one rule, exemplar models for the
Resemblance Vote, k-winners-take-all for the focus). Examples: Provenance Evidence, Trust Judgment, Members' Reading,
Situation Resemblance, Felt Not-Knowing, Trying in Hesitation, Dream Sorting, One-Origin Row. Before publication, the
theory, the README tops and this text were checked claim by claim against the code and the records, and a newcomer
followed the README in a fresh clone; the corrections and the program bugs found are in this edition.

## Since v5

All measurements, with negative results kept, are in
[research/living_mind_2026_11_15/NOTES.md](../../../research/living_mind_2026_11_15/NOTES.md).

- **Sleep and dreams (F22).**
  - Dreams follow the day's topics as chains of association, and an image seen with a word is seen again.
  - Each dream is kept with its claims and sorted true, false or open when life meets it.
  - The trust in its own imagination follows.
- **Hearing.** A WAV adapter: new sounds of three known synthetic voices recognised 93% / 97% / 93% (chance 33%, 0%
  without resemblance).
- **Felt not-knowing (L7) and think-then-speak (T8).** Not knowing is a measured inner state, and saying "I don't
  know" is learned. Thoughts called to mind by the heard words are said when real enough.
- **Situation resemblance (F23).** An utterance never lived as a whole is met by the most similar lived situations.
- **Online life.** During a session the mind lives in its own time: it thinks, speaks first or asks in silence, and
  sleeps.
- **Exploration and code learned from its tests.** Hesitation ends in a try of the least-known well-supported option.
  Learning only from the results of running its own code, it went from imitating wrong code to passing code on six
  of six tiny tasks by round 4. It stayed at 0% without the results and at 11% without trying.
- **Shared-stem inference.** A first form of imagination that reasons; used in dreams and inner voice only.
- **Learning to learn (F24, off by default).** Voice credibility carried into a new domain helps early learning in
  one direction only (+0.058 ± 0.025, better in 7 of 10 lives).
- **One-shot evidence (F25, off by default).** Studied: it removes the echo from thinking but makes conversation
  worse.
- **Speed and memory.**
  - The speed work made the core 2.0–3.8x faster per moment with bit-identical results.
  - Memory per tie dropped from 239 to 69 bytes (cevahir3b core 777 MB → 223 MB), again with bit-identical results;
    this layout costs 13–32% per moment, so the net gain since v5 is roughly 1.6–3.4x (about 3.9–4.4 ms per moment on
    the mind with 3.25 million ties).
  - Old mind files are converted on load.
- **Negative.**
  - Word pieces from birth multiplied the ties per moment about five times and exhausted memory, so they are now
    opt-in.
  - Memory still grows linearly with lived data.

The teacher questions, the documents used in the 2026-11-15 runs and the minds raised on them are not part of this
archive. The folder `education/ozel-dokumanlar`, public in the repository since March 2026, is part of every snapshot.
