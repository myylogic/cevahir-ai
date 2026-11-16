# Research release 2026.11.16 (v6)

- Git tag: `research-2026.11.16-v6` (a GitHub Release from main; it supersedes the v6 prepared on 2026-11-15, which was
  never published: [research-2026.11.15-v6.md](research-2026.11.15-v6.md) is kept as it was written)
- GitHub Release: https://github.com/myylogic/cevahir-ai/releases/tag/research-2026.11.16-v6
- Source commit: `b5b591ed438d24e2e4bcc83a02c1b8a5204ab804` (it also records SCAN within 12 answered 1,188 / 1,188
  with a window of 13, measured after this note was written: NOTES.md, audit section 10)
- Zenodo DOI: [10.5281/zenodo.24279940](https://doi.org/10.5281/zenodo.24279940)
- Zenodo record: https://zenodo.org/records/24279940
- Archive verification: every one of the 1987 files of the archive is identical to the tagged commit
  (`../evidence/zenodo_archive_verification_v6.json`)
- All-versions DOI: [10.5281/zenodo.22864660](https://doi.org/10.5281/zenodo.22864660)

## The Cevahir Zihin theory: a Living Evidence Mind

The theory of the new core comes first ([docs/ZIHIN_THEORY.md](../../ZIHIN_THEORY.md), Turkish original
[docs/ZIHIN_KURAM.md](../../ZIHIN_KURAM.md)).

The model in `zihin/` is not a Transformer. It has no weight matrices, embedding vectors, backpropagation or softmax
attention.

- **Ties and evidence.** Every relation is a tie, and each tie keeps its graded evidence separately by origin (sensed,
  told, imitated, acted, inferred, imagined). The trust in the four indirect origins is learned from direct
  experience; sensed evidence and its own acts are the reference.
- **One rule.** Expectations are read from one rule, e = c(f − ½) + ½, the NARS truth-value expectation with k = 1.
- **Learning and forgetting.** It learns at every moment, with no training phase. While its ledger is kept, the evidence
  of any single experience can be retracted exactly.
- **Attention.** A Seven-Place Focus: seven units win the focus each moment, so attention is local and readable.

The theory names its findings and credits prior work (NARS for the one rule and its truth functions, exemplar models
for the Resemblance Vote, the distributional hypothesis for interchangeability, Harris's successor variety for word
forms, Mill's method of difference for F39, Hull's drive and habit for F40).

## Against Transformers trained from scratch on the same data

All Transformers below were trained by the same scripts, on exactly the dialogues or documents the mind lived, and
were exact on their own training data where stated. These comparisons are small-data and small-model; larger
Transformers with much more data may close the gaps.

| test | Zihin | Transformer |
|---|---|---|
| T-compose: SCAN's "add primitive" split made small (a primitive seen only alone, then in 11 templates; 3 seeds) | **75 / 75** | 4 / 75 (training 43 / 43) |
| T-compose with "around" (window 11; 3 seeds) | **81 / 81** | 7 / 81 (training 49 / 49) |
| SCAN's own grammar, add-jump, the 844 test commands that fit a window of 11 (official run, seed 0) | **789 / 844** | **0 / 844** (training 1570 / 1574) |
| the same, every command whose words, end and answer fit the window | **596 / 596** | |
| the same 844, the mind's window 12 instead of 11 | **844 / 844** | |
| recall of the owner's documents read once, a two-word cue (exact / valid / silent, current configuration) | **90 / 146 / 31** | 0 / 0 / 1 (1.0M parameters, one pass) |
| the owner's documents, one-word questions, lived once (exact, token F1) | **0.130, 0.300** | 0, at most 0.100 (1M–4M parameters, up to 20 epochs) |
| the same questions answered by a lookup that does not learn (the first document sentence starting with the word) | 0.930, 0.946 | |
| learning throughput, one pass (items per second) | 136 | 1,338–57,281 (the Transformer and RNNs are faster) |

- The mind's 55 errors on SCAN within 11 are all of one kind: the answer's last word is decided after the command's
  first word has left the window. With a window of 12 the mind answers all 844. A working memory (the question held as
  one item) is not built; longer commands are beyond the window (844 of the split's 7,706 test commands fit 11 items,
  2,246 fit 15).
- Where the mind is slower: it learns at about 3–5 ms of CPU per moment, while small neural networks learn tens to
  hundreds of times faster per item; it needs no training run.

## New mechanisms since v5 (each measured; switches off: the lesson-free gate and 17 configurations bit-identical)

Measurements and every negative result are in
[research/living_mind_2026_11_15/NOTES.md](../../../research/living_mind_2026_11_15/NOTES.md); an audit of what the
mind can and cannot do, in Turkish, in
[DENETIM_2026_11_16.md](../../../research/living_mind_2026_11_15/DENETIM_2026_11_16.md).

- **Since the 2026-11-15 draft of v6:** sleep and dream sorting (F22), hearing, felt not-knowing (L7), think-then-speak,
  situation resemblance (F23), online life, exploration and code learned from its tests, learning to learn (F24, off),
  one-shot evidence (F25, off), a 2.0–3.8x faster bit-identical core and a 3.5x smaller tie layout. (Shared-stem
  inference was removed later: it rested on hand-written morphology.)
- **Recall of what was read once.** Sequence Starts (F26), Lived Exclusion (F27) and the perceived end of a read
  sentence: the teacher protocol from 0 (asked a bare word, as the earlier evaluations did) to 120 of 200 exact
  (F1 0.7444).
- **Exact Relations (F28)**, and the chat configuration listening in the generated worlds: held-out 0.783 / 0.812.
- **Meaning from the places words fill.** Analogy by Interchangeability (F29) and Corroboration (F30): wordings never
  watched with another greeting in front 0.17 → 1.00, synonyms 0.33 → 0.67 (the strict suffix rule of F34 later took
  the greeting in front to 0.67; F38 restores 1.00 and F39 gives 0.89, both off by default). Paradigmatic Relation (F31) and
  Relational Binding (F32): "selam ne demek ?" → "selam , merhaba demektir" (0 → 0.67).
- **Relational Memory (F33).** One mind, one memory, recalled in the company of the person present: each person's own
  greeting, colour and cat's name 0 → 1.00; a secret told to one person is told to no one else (0.75 / 1.00 → 0);
  what everyone taught is known with a stranger; per-person trust (0.956 and 0.638 against one global 0.569);
  relationships change with the encounters.
- **Compositional generalisation: Reply Relation (F34).** A frame's value may be what its filler, said on its own, was
  answered with; several places bound; corroboration by the lived suffixes that keep every fixed word of the frame;
  bound frames of every prefix (the SCAN results above).
- **Purpose: Generalised Consequences (F35).** What acts lead to, read for situations never met (polite to things never
  handed 0 → 8 of 9 against the habit it watched) and per relationship with F33 (two people with opposite tastes,
  9 of 9 each).
- **Word Forms (F36).** A word never met read as the lived word sharing its longest stem (Harris): "merhabalar",
  "nasılsınız ?" 0 → 1.00.
- **Deduction with exceptions: Inheritance (F37, off by default).** Facts read apart give conclusions never taught,
  by NARS deduction, the class named by a stem the mind found in its own words: T-deduce 2 → 48 of 48, two steps
  (T-chain) 0 → 27 of 27, the nearer property 9 of 9, exceptions 3 of 3, no wrong generalisation. Off because an
  answer on the largest mind takes about 8–11 times as long (made nine times cheaper in this round).
- **Witness Reliability (F38, off by default).** A shorter lived context as a witness of a claim, as far as the mind has
  seen it agree: the meaning world's greeting in front 0.67 → 1.00, but a SCAN life lived with it learned otherwise
  (789 → 767, then 783 with the third form); kept off, with the hypotheses each form eliminated.
- **Causal relevance: Replacement Witness (F39, off by default).** Mill's method of difference read from the mind's own
  life. With clouds the common cause of the umbrella advice and the rain, the words alone lead the mind to rain on a
  sunny day with the advice (0.556); with F39 the fine weather leads (0.577), still under the speaking line. Lived and
  answered with it: SCAN within 11 789 / 844 (the same answers), the meaning world's greeting in front 0.67 → 0.89,
  everything else measured unchanged; off because it was not measured on the owner's documents and costs about a
  tenth more time an answer.
- **Long answers, a first step: Urge to Go On (F40, off by default).** Two expected words that are both things known
  about the same item race against silence, not against each other: more facts about things never asked, no more
  confabulated (T-long 20 → 22 of 54: two sentences that stopped half way finished; confabulated 4 and 4); polite
  answers to things never handed 8 → 9 of 9.
- **`!neden`.** Where each word of the last answer came from (lived evidence, a frame and its relation, a sequence
  start, a continuation, a form, similar situations, an expected consequence), and the contradictions it overrode.

## Negative results and open problems (kept, as always)

- **"Yağmur nasıl yağar sence ?"** None of the owner's minds here gives a causal account of rain; no such text was lived.
  The mind cannot yet compose an explanation from facts read apart beyond two steps of class inheritance.
- **Long answers** remain short: the end of a read sentence is read as the end of an answer, and with the facts told in
  any order the mind says the thing's name and stops (T-long 0 of 54 with or without F40): three right continuations
  divide the evidence one would have.
- **Working memory** is not built: commands longer than the window are out of reach (at a window of 15, 71% of the
  add-jump split's test commands do not fit).
- **Active experiment** (asking to test an uncertain inference) is not built.
- **F38** costs SCAN answers when lived with in every form measured; **F37** is too slow to be on by default; **F39**
  leans to the causal answer but does not assert it.
- **Learning to learn** (F24) gave a small gain in one direction only; **one-shot evidence** (F25) made conversation
  worse.
- No claim of consciousness, general intelligence or novelty is made: the one rule and its truth functions are NARS's,
  and each mechanism names the prior work it rests on. The reports are not peer reviewed. AI assistance was used for
  code, experiments and drafting.

## Reproducing

- Public tests: `research/living_mind_2026_11_15/t_*.py` (T-compose, T-deduce, T-chain, T-cause and its corrected probe
  `t_cause_context.py`, T-long, T-purpose, T-relations, T-trust), `meaning_switches.py`, `speaking_worlds.py`; results
  in `research/living_mind_2026_11_15/results/`.
- The minds raised on the owner's private Turkish data, and the teacher and recall protocols' data, are not in this
  archive; their results are recorded as aggregates only.
- A new mind starts empty and runs with `python -m zihin` (or `.\cevahir.ps1` on Windows).
