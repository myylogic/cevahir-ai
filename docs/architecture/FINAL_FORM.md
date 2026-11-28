# Cevahir, the final form: one substance

Design and measurements of 28 November 2026 · study folder: `research/final_form_2026_11_28/` · package: `cevher/`

This document is the design of the one architecture the owner asked for on 28 November 2026, the experiment that
could have killed it, and the gates it was measured against. Sections 1 to 8 were written before any code of the
final form existed; section 9 fixes the kill criteria before the kill experiment was run. Measured numbers are in
sections 10 to 12 and are marked with the commit of the code that produced them. Negative results come first in
every section that has any.

## 0. In one paragraph

Today's Cevahir is three stores (Zihin's units and ties, the cortex's contexts, the hippocampus's episodes) and
about seventy-five mechanisms that read them. It keeps what it lives: about 1.1 kilobytes for every character read,
at 100 to 400 characters a second. The final form keeps **one substance**: tables of fixed capacity that hold
evidence counts, addressed by fixed-size codes of units. Everything lived is an additive write of counts into those
tables in the same pass that predicts; everything said is one rule read back from them, e = (W⁺ + ½)/(W + 1), said
only at the speaking line e* = 7/12 ≈ 0.583; forgetting is subtracting the same write; and what the mind learns
about its own ways of reading (how often each was confirmed) is written into the same tables and read by the same
rule. Memory per word type is a constant, memory per word lived is zero once a tie exists, and the whole is bounded
by a configured budget, beyond which the least evident ties are let go. The package is called `cevher`: the one
substance (Turkish *cevher*, whose plural is *cevahir*).

## 1. What stands today (measured again for this study)

The yardstick is `research/final_form_2026_11_28/yardstick.py`: one runner that drives today's Cevahir
(`living.cevahir.Cevahir`) and the final form through the same calls, builds every life and question with the
earlier studies' own code, and grades every answer by one rule that knows nothing of either mind (right, silent or
wrong; the words of the answer, lower-cased by Zihin's Turkish rule, closing marks apart). Its rows are in
`research/final_form_2026_11_28/results/yardstick.jsonl`; section 10 puts them beside the final form's.

Three things about the yardstick are not as the earlier notes had them, and are said here once.

- **SCAN is regenerated, not downloaded.** The published files are not in the repository and this study downloads
  no dataset. `scan_files.py` writes SCAN from its grammar in the published layout. The earlier study verified
  that this generation equals the published files as sets for all 20,910 commands and for the add-jump and length
  splits (`research/living_mind_2026_11_22/results/scan_official.jsonl`). The four template splits are written by
  a rule inferred from the published line counts, all eight of which it meets (commands holding "turn" with the
  template's tail are in neither file); that the sets equal the published ones could not be checked here. The lines
  are in the grammar's order, so the seeded sample of 50 test commands is a sample of the same set but not the
  same 50 commands as in the rows of 26 November.
- **COGS could not be run.** Its files are not in the repository (NOTES.md 8.12: "as found, not kept here") and
  were not downloaded. The recorded row of 26 November (299 of 300 on each sample, code 5f6282d) stands as the
  earlier study wrote it; it was not reproduced, and the final form was not measured on COGS. `yardstick.py cogs
  --cogs FOLDER` runs both minds as soon as the files are there.
- **The held-out set** was written by `heldout_make.py` before this design: ten worlds, 200 questions, sha256
  `854506933efb9fa274bf9d5b40745b99d20b585d63c0057dfe452e1a0c5de45d` (`heldout.json`, checked by the runner before
  every use). Its items were drawn by a seeded generator and were not looked at until the last measurement
  (section 12).

## 2. The decision

In the owner's terms: memory per word must fall to what an embedding space costs; the mind must keep learning while
it lives, in the forward pass, and learn how to learn; and all of it must be one architecture.

This reverses one line of MODEL.md section 9, "not taken: fixed-dimension embeddings". Every unit now has a code of
fixed size. The other three lines stand: no back-propagation through the store, no training phase separate from
living, no fixed context window as the only memory.

What is kept from the earlier studies, because they established it: the one rule (the Krichevsky–Trofimov
estimator), the speaking line, evidence kept apart by origin, trust in a way of reading learned from what its
claims came to, exact retraction through a ledger, learning while living, and the finding of NOTES.md 8.31 that a
mechanism is not taken before every battery has measured it.

## 3. The substance

### 3.1 Units and codes

A unit is anything the mind tells apart: a letter (a byte of UTF-8), a word or mark (a token of `\w+|[^\w\s]`,
lower-cased by the Turkish rule), a context (the last few units), a frame (a question with places left open), a
way of filling an answer (a skeleton), an origin. **Every unit has a code of 64 bits**, computed from what the unit
is made of (a word from its letters; a context, a frame or a skeleton from the codes of its parts). A code is not
learned and not stored per occurrence: it is recomputed from the unit whenever the unit is met. Two units with the
same code are one unit to the store; with 64 bits that needs about 4 × 10⁹ units before one collision is expected.

The only thing kept per word type is its spelling, so that a word the store brings back can be said: one entry of
a packed table, the code and the bytes of the word.

### 3.2 Ties: the one table

The substance is a table of **ties**. A tie is (key, value) → W: the evidence, counted as lived, that `value`
stood in the relation `key` names. A key is the code of a unit joined with a role and an origin ("what came next
after this context, in what was told"; "what the answer held where this frame left this place open"). The total of
a key, W(key) = Σ_v W(key, v), is itself a tie of the table, with a reserved value.

- **Write.** Living adds: W(key, value) += w and W(key) += w, with w = 1 for one meeting (dyadic weights are
  kept for salience). Nothing else is ever written to the substance.
- **Read.** The belief that `value` holds for `key` is the one rule on what is read back,

      e(value | key) = (W(key, value) + ½) / (W(key) + 1),

  and a thing is said only if its belief reaches e* = ½ + 1/12.
- **Retract.** Forgetting an experience subtracts the writes it made, read from its ledger entry; a tie whose
  count returns to zero is removed.

The table has a fixed number of slots. A slot holds the key's code, the value's code (32 bits; for a word, its
number in the spelling table) and the count: 16 bytes. The slots of one key sit together, so that reading a key
gives the values it holds (what followed, strongest first) without asking the table about every word it knows.

Letters use the same table in a denser layout: a context of the last 1 to 6 bytes is the key, the next byte is the
value, and a slot is 8 bytes (a 32-bit check of the key, the byte, a 24-bit count).

### 3.3 What the bound costs

Three forms of the table are candidates (section 9 chooses by measurement). They differ in what a collision does.

- **Checked slots.** A slot keeps a check of its key. Two keys never add into one count; when a key's place is
  full, the tie with the least evidence gives way. Collisions cost *ties forgotten*, never *evidence invented*.
- **Count sketch.** No check: every tie adds into d cells chosen by hashes, and is read as the least (or the
  median, with signs) of them. Nothing is ever forgotten and retraction is exact for ever, but ties add into each
  other: collisions cost *evidence invented* (a count-min sketch only overestimates), which the rule turns into
  belief in what was never lived.
- **Distributed cells** (sparse distributed memory with counters; role-filler vectors): a key activates several
  cells, each a vector of counters into which the value's code is added. Read by correlating with candidate codes.

Whatever the form, the budget is a number of bytes fixed in advance, and the bytes are:

| | bytes |
|---|---|
| per word type | its spelling and its place in the spelling table: about 24 + the word's length |
| per tie held | 16 (words), 8 (letters) |
| per word lived, once its ties exist | 0 |
| per character lived, on disk, with a ledger | the ledger entry of what was lived (the text itself, compressed); 0 without |

A word type therefore costs what its ties cost, and the table holds at most budget/16 of them. This is the cost of
an embedding space: a table of fixed size, written additively, in which a word's row is its strongest ties. What
is given up is everything the table cannot hold: once it is full, a tie met once and never again is the first to
go, and with it what one reading taught. Section 11 measures how much that is.

## 4. What living writes

One pass, forward only. A piece of life is one of: a text read, a sentence told, a dialogue watched, a turn of
conversation. The same writes are made for each, tagged with the origin (told, asked, answered).

1. **Letters.** For each byte and each context length d = 1 … D: the tie (context, next byte). This is the cortex
   of MODEL.md section 3, with its contexts in a fixed table instead of a growing tree.
2. **Words.** For each token: the ties (the last n tokens → the next token) for n = 1, 2, 3, and (a token → the
   token two places on). This is what Zihin wrote as ties between units, without one tie per pair of words in a
   sentence: only what a context was followed by.
3. **Dialogues.** For a question and its answer:
   - *the answer as lived*: (the question's code, the place) → the answer's token at that place;
   - *what a word brings*: (a word of the question) → each token of the answer, and the reverse;
   - *frames*: the answer is explained as far as it can be from the question and from what the store already
     holds, each token of it as one of: a copy of a word of the question; what a word of the question brings;
     the lived answer of a part of the question; what followed a part of the question in what was told; or a
     token of its own (a constant). The words of the question that the explanation uses become open places; the
     question with those places open is the **frame**, and the explanation with them open is its **skeleton**.
     The ties written are (frame → skeleton) and, for each constant, (frame, skeleton, place → token), once
     unconditionally and once per feature of each filler (what the filler was followed by in what was told).

Nothing of a test world is named in these rules: no word, language, mark or format. A frame is not given; it is
read from one lived dialogue and the store.

## 5. What reading does

**The next symbol** (letters): the longest context held, then shorter ones, each with the belief that what follows
is new to it (the novelty event of MODEL.md 2.2), judged by the record of the situation (MODEL.md 2.3). Read for
every byte before it is written; the bits paid are the measure of how well the store knows a text.

**An answer** (words): every way of reading the question claims an answer, and the claim believed most is said if
it reaches the speaking line.

- *Recall*: the question as lived, with no place open.
- *A frame*: the question with one or two places open matches a frame the store holds; its skeleton is filled
  from the asked question (a copy of the filler, what the filler brings, the answer of the filler asked as a
  question of its own, what followed the filler in what was told), and its constants are read back, by the
  filler's features where the constant depends on the filler.

A claim is believed as far as its weakest step: the rule on the skeleton's record, on each constant, and on each
answer it was built from.

## 6. Learning to learn: every way of reading keeps its record in the substance

Before a dialogue's answer is lived, the mind foresees it by every way it has. Each (frame, skeleton) that claims
an answer is a way of reading, and the outcome is written as a tie like any other: W(way) += 1, and
W(way, confirmed) += 1 if the claim was the answer lived. The belief in a way is the rule on that record. So:

- a frame that generalises rightly is believed more each time it is confirmed, one that fails is believed less,
  and one that was never tried is not believed at all;
- the trust per situation of the cortex (MODEL.md 2.3) and the trust between voices that the harmony's gate and
  the prefrontal records carried (MODEL.md 5.3, 12.3, 12.5) are the same thing here: a record of claims, read by
  the rule;
- the record is per way, not one global count over a voice: the lesson of NOTES.md 8.29 to 8.31 (a global record
  of what a voice came to while it was learning silenced what it had learned since) is taken as a constraint.

How strongly a way writes follows from the same record: a frame is written once when it is first read from a
dialogue, and after that only its confirmations and failures are.

## 7. Retraction, and what exactness means in a bounded store

A ledger entry holds what was lived (the text, compressed, on disk) and nothing derived. Forgetting an entry lives
it backwards: the same writes with the opposite sign, in the reverse order.

- **Below the budget** (no tie has been let go since the entry was lived): the substance after forgetting is equal
  to the substance of a mind that never lived the entry. The digest of the substance (a sum over its ties, each
  hashed with its count, independent of where a tie sits) is equal. Gate G4 checks this.
- **At the budget**: a tie that the forgotten experience pushed out does not come back, and a tie of the forgotten
  experience that was itself let go is not there to subtract. What is given up is exactly that; what it buys is
  the bound. The count sketch would keep exact retraction for ever and pay with invented evidence instead
  (section 9 measures the price).
- **What is not substance.** The record of a way of reading is evidence about the mind's own claims at the moment
  they were made; it depends on the order of life, as learned trust did in Zihin and in the cortex's situations.
  It is written through the same ledger and is taken back with the entry that wrote it.

## 8. Sleep

A night lets go of the ties met once that were not met again since the night before, as the cortex's night did
(NOTES.md 8.8). In a bounded store this is the same act as making room, done ahead of need.

## 9. The kill experiment (criteria fixed before it was run)

The thesis is that one fixed table of evidence can carry what today's three stores carry. It would be dead if a
fixed table at a small fraction of today's memory could not predict text or could not answer by composition. The
cheapest experiment that could show that:

- **K1, text.** Next-symbol bits per character, prequential over the whole of `tr-lessons`, `en-pydoc` and
  `en-repo` (the corpora of `corpora.py`), with the letter contexts in each candidate store held to **11 bytes per
  character of the text**, one hundredth of today's 1.1 kilobytes a character. Reference: the same predictor on an
  unbounded exact store, and the recorded rows of today's cortex and harmony.
- **K2, composition.** The SCAN battery of the yardstick (six splits, 50 commands each) answered by the frame
  mechanism of sections 4 to 6 over each candidate store, held to **one hundredth of what today's Cevahir holds
  after the same life** (walked by `m3_scale.py`'s walker). Reference: the same mechanism on an unbounded exact
  store.

Candidate stores: checked slots; count-min sketch; signed count sketch; distributed cells (hashed sparse
distributed memory with counters); and, for the letters' combination, the rule alone against a gate learned
online.

**Criteria.** A candidate *survives* if, at one hundredth of today's memory,

1. on text it pays at most 10 % more bits than the same predictor on the exact store, on every corpus;
2. on SCAN it gives at least 90 % of the right answers the same mechanism gives on the exact store, and no more
   wrong ones.

**The thesis is killed** if the frame mechanism on the exact store answers fewer than 270 of the 300 SCAN commands
(then no table can carry composition this way), or if no candidate survives both criteria. In that case the
study stops there and reports it.

## 10. Results

Filled in by the measurements; see the sections below as they are written.
