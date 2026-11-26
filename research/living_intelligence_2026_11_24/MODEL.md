# Living Intelligence: the model

Study of 24 November 2026 · `research/living_intelligence_2026_11_24/` · code: `living/`

This document is the design. M1, the cortex with its lexicon and channels, is built and measured; the results are in
[NOTES.md](NOTES.md). The later milestones (M2–M5) are written here as models to be built and measured, not as results.

Cevahir is the individual; Zihin is its mind. This study gives that mind a new body: brain-like modules built on one
unchanged rule of evidence, and a store of symbols that grows with what the mind lives.

## 0. In one paragraph

The mind keeps no list of words and no vector per word. It keeps a **cortex**: every context it has met (the last few
symbols) is an origin with counts of what followed it. Words are paths in a letter store that the mind numbers itself,
in the order it met them, and the path regenerates the word from its number. The alphabet is the mind's own: a symbol
never met is a novelty event, and a stream new to every channel the mind has becomes a new channel. Above the cortex,
the next symbol comes from a **harmony** of experts:

- the cortex;
- the lexicon;
- an episode memory that completes what it has seen before;
- the focus of a **prefrontal** working memory.

An **amygdala** decides what matters. A **hippocampus** holds episodes for a while and, in sleep, keeps what recurred or
mattered and lets one-offs go. Every belief in every module is the same rule of the evidence core,
e = c(f − ½) + ½, applied to the record of the right origin. The mind says only what reaches the speaking line
e* ≈ 0.583, and it keeps learning while it lives.

## 1. Why a new architecture

The Zihin of 14–24 November 2026 ([PAPER.md](../../docs/research/living_evidence_mind_2026_11_24/PAPER.md)) showed
that a mind of evidence with no network can learn by living: SCAN's six published test files 26,227 of 26,227,
real-text recall after one reading, grounded meaning. It also showed its limits.

- **Its unit is the whole word.** A word never met is an unknown symbol. The mind cannot tell "evler" from "ev" + "ler"
  except by a separate mechanism, and a word in a new script is a new symbol that shares nothing with the words it
  knows.
- **It stores what it lives.** Every sentence writes ties between its words: 12.3 ties per word of natural text, 73 % of
  them written once and never used again. Growth is linear in experience: one trillion words would need about 840 TB
  at 68 B per tie (PAPER.md, section 10). The data of the world is not something a mind can keep a copy of.
- **It has no executive and no salience.** Focus is a k-winners-take-all of seven places; nothing in the mind holds a
  goal, chooses where to look, or marks one experience as more important than another.

What it proved valuable is kept: evidence per origin and the one rule, trust learned per origin, the speaking line,
the seven-place focus, exact retraction, learning while living, and no fixed dimension anywhere.

## 2. What does not change: the one rule

### 2.1 The rule is the Krichevsky–Trofimov estimator

A claim with W⁺ confirmations and W⁻ contradictions, W = W⁺ + W⁻, is believed with

    f = W⁺ / W,    c = W / (W + k),    e = c (f − ½) + ½,    k = 1.          (zihin/evidence.py)

Expanding,

    e = (W/(W+k)) (W⁺/W − ½) + ½ = (W⁺ − W/2)/(W + k) + (W + k)/(2(W + k)) = (W⁺ + k/2) / (W + k).

With k = 1 this is (W⁺ + ½)/(W + 1). That is the **Krichevsky–Trofimov (KT) estimator**: the posterior mean of a
Bernoulli rate under the Jeffreys prior Beta(½, ½) (Krichevsky and Trofimov 1981). The theory's rule is therefore a
known estimator, and it comes with a known guarantee. For any binary sequence of length n, its sequential code length
exceeds that of the best fixed rate chosen in hindsight by at most ½ log₂ n + 1 bits; this is why context-tree
weighting uses it (Willems, Shtarkov and Tjalkens 1995). The theory already names NARS as the source of the rule
(docs/ZIHIN_THEORY.md, section 2). The identity with KT adds a guarantee from information theory, and it is stated here
so that the rule is never presented as new.

### 2.2 Many outcomes: the novelty event

A context that has met n outcomes, t of them distinct, is asked what comes next. The rule is binary; the question is
not. It is reduced to the rule by the **novelty event**: "what follows is new to this origin".

- Each distinct outcome was new exactly once, so the record of the novelty event is W⁺ = t, W = n.
- The belief ν that what follows is new is the rule on that record.
- The rest of the belief, 1 − ν, is shared among the outcomes met, in proportion to their counts.
- The novelty mass ν passes to a coarser origin: a shorter context, and at the root a symbol never met.

No new constant enters. With the context's own record, ν = (t + ½)/(n + 1).

### 2.3 Whose record: the situation

M1 measured this directly (NOTES.md, 2.2): judged by the context's own record, novelty is overestimated. On the Python
documentation the store then codes at 1.97 bits per character, worse than bzip2. The reason is that novelty is not
stationary. A context meets most of its new symbols early, so its past novelty rate t/n is higher than its future one.
A binary estimator that assumes a fixed rate cannot see the fall.

The rule is unchanged; it is applied to the record of a better origin. That origin is **the situation**: every
context that was in the same situation, meaning

- the same length d;
- whether longer contexts were excluded before it;
- how varied its record is (t: 1, 2, 3, 4–7, 8 or more);
- how much it has lived (n: 1, 2, 3, 4, 5–7, then doubling up to 64 or more).

The buckets follow powers of two. They are a partition chosen once, before any measurement, not a parameter fitted to
a text.

The situation keeps its own record of the novelty event, W⁺_σ (times what followed was new) and W_σ (times it was
asked), and

    ν(σ) = (W⁺_σ + ½) / (W_σ + 1).

This is the evidence core's learned trust per class of origin (in Zihin, the trust of a provenance class is learned
from how often its claims were confirmed), here applied to classes of contexts. It is also what the compression
literature calls secondary escape estimation (Shkarin 2002); the idea is known.

On the same text it brings the store from 1.97 to 1.59 bits per character (bzip2 1.61, xz 1.54). On the repository's
English it reaches 1.98 (bzip2 2.02, xz 2.10), and on the Turkish lessons 1.95 (bzip2 2.22, xz 2.04) (NOTES.md, 2.2).

### 2.4 The speaking line

An option racing alone must reach the act threshold θ = 1 within T = 6 settling steps, so it is said only when its
belief is at least

    e* = ½ + θ / (2T) ≈ 0.583.

Every module that acts uses this line. The cortex says a letter only if its probability reaches e*. The PFC commits to
a word only if the harmony's belief in it reaches e*. A channel is new only if the belief that a stream is new to every
channel reaches e*.

### 2.5 Exactness

Every count is an integer, or a dyadic number on the grid of the evidence core when salience weights it (section 7).
The substance of the cortex (its contexts and their counts) is therefore a function of the multiset of what was lived,
when every context is kept (growth rule m = 1, section 3.4). Reading a text and then subtracting it leaves the
substance identical to a store that never read it. M1 checks this by SHA-256 (NOTES.md, 4).

Trust records (situations) are learned from the store's own predictions, so they depend on the order of life, as
learned trust does in Zihin. Their ledger takes back a text's own events exactly, and the remaining difference is
measured.

## 3. The cortex: a letter store that numbers its own words (M1)

### 3.1 Contexts as origins

A channel delivers symbols x₁, x₂, … from its alphabet A (section 4). After history h = x₁ … x_{i−1}, the contexts of
the next symbol are its suffixes

    s_d = x_{i−d} … x_{i−1},    d = 0 … D

(s₀ is the empty context, the root). Every context the store has met is a node with counts n_s(x) of the symbols that
followed it, n_s = Σ_x n_s(x), and t_s = |{x : n_s(x) > 0}|. Nothing has a fixed size: there are as many nodes as
contexts met, and as many symbols as symbols met.

### 3.2 Prediction

Let d* be the longest context of h that is a node. The probability of the next symbol x is computed by walking from
s_{d*} down to the root, with exclusion. The walk starts with a mass M = 1 and an empty excluded set E. At each level d:

    n' = Σ_{y ∉ E} n_{s_d}(y),    t' = |{y ∉ E : n_{s_d}(y) > 0}|.

- If t' = 0, the level is skipped.
- Otherwise ν_d = ν(σ(d, [d = d*], t', n')) is the novelty belief of the level's situation.
- If x ∉ E and n_{s_d}(x) > 0, the walk stops with

      P(x | h) = M · (1 − ν_d) · n_{s_d}(x) / n'.

- If not, the walk continues with M ← M · ν_d and E ← E ∪ {y : n_{s_d}(y) > 0}.

Past the root, the remaining mass belongs to symbols never met, spread over the channel's code space. For text this is
all 1,114,112 Unicode code points: P(x) = M / (1,114,112 − |A|).

The walk with exclusion is PPM's (Cleary and Witten 1984; Moffat 1990). The confidence of each level comes from the one
rule on the situation's record. Mixing without exclusion ("blending") is kept as a measured alternative.

### 3.3 The lexicon: words as numbered paths

A word is a maximal run of letters and digits, in any script. The lexicon is a tree of letters.

- **Numbering.** Each step the mind takes for the first time becomes a node, with the next free number. The number of
  a word is the number of the node where it ends. The numbering is the mind's own computation, unique to its life: two
  minds that lived different texts number the same word differently.
- **The generator.** The path regenerates the word from its number, one letter per step back to the root. Nothing else
  is stored per word.
- **Sharing.** Words that share a beginning share nodes ("ev", "evler", "evlerimiz"), so a new word costs only its new
  part. Turkish suffixes, which share endings rather than beginnings, are learned by the cortex: the context "ler" is
  one origin for every word that ends in it.
- **Word belief.** A node counts the words that passed through it, P(v), and the words that ended there, N(v). The
  belief that the path to v is a whole word is the rule on that record, e = (N(v) + ½)/(P(v) + 1).

This is the space where words are kept. It has no dimension. Its points are paths. Nearness in it is a shared
beginning (spelling) or a shared context (use: the counts of the cortex and, from M2, the word-level ties). It grows by
exactly what is new.

### 3.4 Growth

**Materialization.** A context becomes a node once it has been met m times.

- With m = 1 every context met is kept. The substance is then exact (2.5).
- With m = 2 ("met twice") a context waits as a pending mark under its parent until its second meeting. One-off
  contexts never become nodes, but the order of life now shapes the store.

M1 measures both (NOTES.md, 2.3). On these texts, m = 2 divides the nodes by 3–4 and the ties by 2–2.6. Counting
every pending mark as a tie, the store shrinks by a factor of 1.5–1.9. The cost is 0.01–0.11 bits per character on the
larger texts and 0.38 on the 39,000 characters of the Turkish lessons.

**The growth law.** The number of nodes after N symbols is the number of distinct contexts of length ≤ D met (at
least m times). It is bounded by N·D and by |A|^D. For natural text it grows like N^β:

- the cortex with D = 8: β ≈ 0.66–0.74 on 0.5–2 million characters;
- the lexicon: β ≈ 0.40–0.67.

The exponent rises with D, because long contexts are nearly all unique.

At the scale measured, the store already holds less per character than Zihin. On the repository's English it holds
1.7 ties per character. At 12 bytes per tie and 4 per node in a compact layout, that is about 26 bytes per character,
against about 150 for Zihin (12.3 ties per word at 68 bytes, 5.6 characters per word).

A rough projection goes from the same rows (1.15 million nodes and 1.64 million ties at 0.94 million characters,
β = 0.70 and 0.73) to one trillion words, about 6 × 10¹² characters. It gives about 6 × 10¹⁰ nodes and 1.5 × 10¹¹ ties,
1–2 TB in that layout, against 840 TB for Zihin's ties.

Seven orders of magnitude of extrapolation from one small corpus is not a measurement. The point is the shape: counts
in contexts that recur grow sublinearly, while copies of sentences grow linearly.

**Directions not yet built (noted 25 November 2026).** Three ways to keep a large life within a machine, each keeping
forgetting exact:

- *Zihin's night.* The cortex's night forgets the contexts met once and keeps what recurred (NOTES.md 8.8: the store
  2–3.8 times smaller for 0.4–1.1 % more bits); Zihin's sleep dreams but forgets no tie, though 73 % of its ties are
  written once. Among them is what one reading taught (recall by a question), so a night that forgets them forgets
  facts read once. What is built (NOTES.md 8.28) forgets nothing: the night drops the caches of reading and merges the
  day's ties into the long-term index, so that a long life holds what one day touched beside its substance.
- *A store on disk.* Each moment touches a small part of the store, and most of it is touched once: the contexts in
  use can be held in memory and the rest on disk, so that a life is bounded by the disk and not by memory. The first
  part is built (NOTES.md 8.28): Zihin's ledger, five kilobytes a character of running text and most of what Cevahir
  held, is kept on disk, compressed; the long-term stores of the substance are not yet.
- *What was lived kept in the cortex's own code.* A text, or a dialogue the hippocampus keeps, can be kept coded under
  the cortex's predictions, about 1.5–2 bits a character instead of 8, and given back byte for byte (the code is
  lossless; a vector space alone would not be).

### 3.5 What is kept, and what is not

The mind does not choose which of the world's sentences to keep; it keeps none of them. It keeps three things.

- **Statistics of contexts.** A sentence lived adds counts to contexts that mostly exist already. Only its genuinely
  new contexts add nodes, and with m = 2 only those met again.
- **Recent and salient episodes** (the hippocampus, section 8). They are held for a while and released.
- **Consolidated associations** between word numbers. They are kept only for pairs that recurred or mattered
  (sections 6–8).

## 4. The mind's own alphabet, and new channels (M1)

### 4.1 Symbols and channels

A channel is an origin of streams: text, a sensor, a new script, a language no one has named. It has its own alphabet,
cortex and lexicon. Its symbols are whatever it delivers. The alphabet is the set of symbols met, numbered in the
order they were first met; it is never given in advance. A symbol never met costs the root's novelty mass and is then
part of the alphabet.

### 4.2 A new channel is born

The mind can be told which channel a stream belongs to. If it is not, it decides by the rule. For each channel c it
has, the belief that a segment X is new to c is

    η_c(X) = (|{x ∈ X : x ∉ A_c}| + ½) / (|X| + 1).

If η_c(X) ≥ e* for every channel, a new channel is born: a new origin with an empty store, named in the order of
birth. Otherwise X goes to the channel it is least new to.

### 4.3 A language it does not know, in a script it knows

A new language written in known letters is not a new channel. It is unfamiliar: the belief that its symbols are
surprising is

    u_c(X) = (|{x ∈ X : bits(x) > b̄_c + 2}| + ½) / (|X| + 1),

with b̄_c the channel's typical bits per symbol. The mind can report that it is watching something it does not know
(u ≥ e*), and it can watch that belief fall as it learns.

The margin of 2 bits (a symbol a quarter as likely as a typical one) is a constant of this report, not of the theory.
No decision of the mind depends on it: which channel a stream goes to depends on η, and speaking depends on the
structure belief and e*.

### 4.4 Watching and speaking

Before each symbol arrives, the mind holds its best guess ŷ = argmax P(· | h). It would say ŷ only if both hold:

- P(ŷ | h) ≥ e*;
- the channel has shown structure. The structure belief is the rule on the channel's record: W⁺ = symbols the context
  predicted better than the channel's store without context would have (an order-0 store that lives the same stream
  and learns its novelty the same way), W = all symbols.

Noise never passes the structure line, so the mind watches it and never speaks. A language passes as soon as its
context predicts. From then on the mind says what it is sure of and stays silent where it is not.

So, to the question of what Cevahir does when it meets a language outside the world's languages: **it watches.** It
opens a channel for it if the script is new, learns while watching, stays silent while its evidence is below the
line, and begins to speak when the evidence allows. NOTES.md section 3 measures exactly this with a made-up language
in an unused script, the same language in Latin letters, and noise.

## 5. The harmony of the next symbol (M2)

The next symbol is not recalled; it is the point where several parts of the mind agree. Each part is an origin with
a record. The harmony combines their beliefs. This section describes the harmony as built in M2 (`living/_core.c`,
`living/fast.py`) and measured in NOTES.md, section 6.

### 5.1 The origins

- **C, the cortex** (section 3), with its whole distribution after the walk with exclusion.
- **Every context of the history**, d = 0 … 8, each as an origin of its own. The cortex uses only the longest
  context and its escapes; here every length speaks.
- **W, the word contexts.** Two origins:
  - the beginning of the current word;
  - the beginning of the current word after the previous word.

  Their records are counts of the symbols that followed, keyed by 64-bit hashes of the exact strings. This is the
  word level without a closed vocabulary. A full word-level cortex over word numbers is left for M3.
- **E, the episode.** The longest earlier occurrence of the last 12 symbols, found through a hash and verified symbol
  by symbol, predicts the symbol that followed it. Its record counts how often that was right, per match length. This
  is the induction behaviour of attention heads (Olsson et al. 2022), done as a lookup.
- **L, the line episode and the line context.** The beginning of the current line, from the last line end, is a cue.
  - The line episode is the latest lived line that began exactly so, found through a hash and verified symbol by
    symbol; it predicts the symbol that followed there.
  - The line context counts what followed every lived line that began so, an origin like the word contexts.
- **S, the short-term memory.** Contexts of length 1 to 6, with records that decay by 1 − 1/F at each update, F = 7.
  This is the context decay of the evidence core, whose time constant is the span of the focus. The cortex keeps
  everything exactly; the short-term memory keeps what happened lately, and forgets.
- **P, the focus** of the prefrontal working memory (section 6) is left for M3.

### 5.2 One decision at a time

A symbol is decided in 8 binary steps of a code:

- its number in the mind's alphabet, if below 254;
- 254 for a symbol never met, followed by its code point at the root's price, log₂(1,114,112 − |A|);
- 255 for numbers from 254 up, followed by the number in 16 binary steps with an order-0 record.

At each step every origin gives its belief that the next bit is 1. Its record is restricted to the codes with the
prefix decided so far: n of them, n₁ with a 1 next. The belief is the rule with the coarser origin as prior
(section 2.2):

    e = (n₁ + p₀) / (n + 1),

with p₀ the belief of the next shorter context (½ for the root), and the word contexts leaning on the root and on
each other. With p₀ = ½ this is the rule itself. The cortex gives its belief from its distribution, summed over the
codes with each prefix. The episode gives the bit of its predicted symbol, with the belief of its record.

### 5.3 The gate (borrowed)

The gate adds the beliefs' log-odds with weights, p = σ(Σ_o w_o · logit(e_o)). It holds one weight set per bit
position, episode state and depth of the history (48 sets), and a second gate with one set per bit position, which
learns fast while the first has little record. Each set learns online by the gradient of the code length,
w ← w + λ (y − p) logit(e).

This is the one method of a network the harmony borrows: a one-layer gate trained by its own error. The learning rate
λ falls from 0.02 to 0.002 as a set is used (as 1/√(1 + u/1024) for its u updates). These are constants of the
borrowed method, not of the theory.

M2 measured the alternative that uses only the rule: the episode's verdict calibrated by the record of its situation
(step B of NOTES.md 6.2). On the four texts, the gate (step C) does better than it by 0.13–0.18 bits per character.
That is the margin the owner asked for before borrowing ("Transformer methods used because they are truly
needed").

### 5.4 The calibration by the rule

The gate's belief is judged once more by a record: the previous symbol's code and the bit prefix, times 24 buckets
of the gate's log-odds. Its belief is the rule on that record, interpolated between the two nearest buckets. A last
gate (per bit position, learned likewise) combines the two gates and the calibration. This is the calibration stage of
context-mixing compressors (Mahoney 2005), with the rule in place of an adaptive rate.

### 5.5 What stays exact

The knowledge of the mind is still the cortex's counts and the word contexts' counts: integers, written by every
symbol lived. A text can be taken back exactly from them, and the cortex alone remains bit-identical to the Python
cortex of M1 (tests/evolution/test_living_fast.py). Three things are learned from the mind's own predictions and
depend on the order of life, as learned trust does in Zihin:

- the gate's weights;
- the calibration's records;
- the short-term memory, which forgets by design.

A pass that does not learn leaves all three, and the episodes, untouched (tested).

## 6. The prefrontal cortex: focus, goal and lock (M3)

**Working memory.** Up to F = 7 units (word numbers) u_i with activations a_i ∈ (0, 1], the seven-place focus of the
evidence core.

- **Decay.** After each unit lived, a_i ← a_i (1 − 1/F): the core's context decay, whose time constant is the span of
  the focus.
- **Gate.** A unit u just lived is offered with strength g(u) = σ(u) · min(1, bits(u)/b̄), its salience (section 7)
  times how surprising it was. It enters if a place is free, or if g(u) is larger than the weakest a_i, which it then
  replaces (k-winners-take-all). This is the gating of working memory by a learned signal (O'Reilly and Frank 2006),
  with the signal taken from salience and surprise.
- **Goal.** A unit set by a question, or chosen by the mind, is held with a lock: it does not decay until the goal is
  reached (the answer said) or abandoned.

**The focus as an expert.** Word numbers u, v carry association evidence: W⁺(u → v) counts the times v came within
the next F = 7 units after u, consolidated only for pairs that recurred or mattered (section 8). With ρ(u → v) the rule on that
record,

    P_P(v) ∝ Σ_i a_i ρ(u_i → v).

This is attention over the working memory: the query is the current state, the keys are the held units, and the values
are their associations. Each held unit's weight is its activation times the evidence.

**The lock.** When the mind speaks, the harmony gives beliefs over the next word's candidates at each word boundary.

1. If the top belief reaches e*, the PFC commits to that word, and the cortex and lexicon complete it letter by letter
   (each letter at or above e*).
2. If no candidate reaches e*, the PFC raises the weight of the focus expert, which carries the goal and topic. This
   is the biasing signal of prefrontal function (Miller and Cohen 2001). The candidates are then judged again.
3. If still none reaches e*, the mind is silent.

A commitment holds until its word is complete; the mind does not switch in mid-word. The next word is the harmony the
PFC locks onto, not a sentence recalled.

**A first form, built in M2: the lock of recall.** When the goal is to recall what began as a cue begins, the mind
locks onto the lived line that began so (the line episode) and says it to its end, instead of saying what usually
follows. If no lived line began so, it speaks by the harmony. On the earlier study's recall race this is the
difference between 23 and 190 valid answers of 200 (NOTES.md 6.8).

## 7. The amygdala: what matters (M3)

Each unit u carries a salience record. W⁺_σ(u) counts the meetings of u that came with an outcome: feedback that an
answer containing u was right or wrong, an explicit mark of importance, or a surprise above the unfamiliar line.
W_σ(u) counts all meetings. The salience is the rule on that record:

    σ(u) = (W⁺_σ(u) + ½) / (W_σ(u) + 1).

Its valence (good or bad) is a second record on the outcomes alone. Salience modulates three things.

- **Learning.** An event in a salient moment is written with weight 1 + σ, on the dyadic grid of the evidence core. It
  stays exact, and the ledger records the weight.
- **Gating** into working memory (section 6).
- **Consolidation.** In sleep, episodes are replayed in proportion to their salience (section 8).

This is the role the amygdala has in the consolidation of arousing experience (McGaugh 2004). Here it is a record of
outcomes, not a feeling.

## 8. The hippocampus: episodes, completion and sleep (M4)

**Episodes.** The hippocampus holds recent segments (sentences, turns) with their unit numbers, times and salience, up
to a capacity of B symbols. When the store is full, the episode with the lowest retention σ · exp(−age/τ) is released.

**Completion.** The episode expert E (section 5) is the fast path of pattern completion: the current context is the
cue, the episode is the memory, and its continuation is the prediction. Its trust is learned per match length, by the
rule.

**Sleep.** Periodically, or when the mind is idle:

1. Salient episodes are replayed into the word-level associations of section 6, in proportion to σ.
2. Cortex contexts met fewer than m times since the last sleep are forgotten: one-off contexts do not survive. The
   ledger records what was forgotten.
3. Consolidated, old episodes are released.

This is the division of labour of complementary learning systems: a fast episodic store, and a slow store of
statistics that keeps what recurs (McClelland, McNaughton and O'Reilly 1995; Kumaran, Hassabis and McClelland 2016).
The growth law then follows the number of distinct contexts that recur, not the amount lived.

Forgetting and exactness pull against each other. Retraction is exact for everything the ledger and the episodes still
hold. Once a one-off context has been forgotten, retraction subtracts what remains of its text. M4 measures the
difference from a mind that never read the text.

## 9. What is taken from Transformers, and what is not

**Taken, because it is needed:**

- next-symbol prediction as the one task the store learns from (self-supervision);
- content-addressed retrieval (attention): the episode expert and the focus's weighting of its held units;
- several heads combined into one prediction: the harmony;
- subword units: the mind's own numbered words and chunks, not a vocabulary fixed in advance.

**Not taken:**

- fixed-dimension embeddings;
- back-propagation through the store;
- a fixed context window as the only memory;
- training separated from living.

M2 measured the learned gradient gate as needed (0.13–0.18 bits per character over the rule-only verdict) and uses it
as the harmony's gate (section 5.3).

## 10. Milestones and measures

| | what | measures | status |
|---|---|---|---|
| M1 | cortex, lexicon, channels | prequential bits per character against PPMd, xz, bzip2, gzip and a character Transformer (offline and online); growth and its exponent; seen and unseen words; bits at a word's k-th meeting; exact retraction; a new channel, a made-up language and noise | built and measured, [NOTES.md](NOTES.md) |
| M2 | the harmony: every context, the word contexts, the episode and a short-term memory, combined bit by bit by a learned gate and calibrated by the rule; the cortex in C | bits per character below PPMd on the same texts; each part's share; the rule against the gradient gate | built and measured, [NOTES.md](NOTES.md) 6 |
| M3 | PFC and amygdala; the word-level cortex over word numbers | sentences said to the end and their coherence; retention of salient facts after interference; the earlier dialogue and school tests through the new architecture | built as Cevahir, one layer (section 12): Zihin as a voice of the harmony, the prefrontal decision by records, the amygdala's record of outcomes; measured in [NOTES.md](NOTES.md) 8 (the word-level cortex is Zihin's) |
| M4 | hippocampus and sleep | memory bounded by B; the cost in bits of forgetting; growth with forgetting; retraction after consolidation | built: the hippocampus's episodes and analogy (12.5), sleep (the cortex forgets what did not recur), exact forgetting; measured in [NOTES.md](NOTES.md) 8 |
| M5 | meaning and dialogue | the ground of Zihin on the new cortex; a standard benchmark (enwik8 or text8; downloaded only with the owner's permission) | |

Every milestone keeps the earlier gates. The rule is not changed to win a measure; if a measure needs a new constant,
that is a finding to report, not a parameter to fit.

## 11. Where this stands in the literature

- **The cortex is a PPM-class model.** Its prediction is that of PPM with exclusion (Cleary and Witten 1984;
  Moffat 1990; Howard 1993), and in spirit of the hierarchical Bayesian context models (MacKay and Peto 1995;
  Teh 2006; Wood et al. 2009). Its escape belief is the one rule on a situation's record, which is secondary escape
  estimation (Shkarin 2002). Its growth question is LZ78's (Ziv and Lempel 1978) and the sequence memoizer's.
- **What is ours** is the way these parts are held together:
  - one rule, the theory's, for every belief;
  - substance that is exactly retractable;
  - words as numbered paths that regenerate themselves;
  - the mind's own alphabet, with channels born by the rule;
  - the modules around the cortex.
- **Learning statistics of letters is how infants find words** (Saffran, Aslin and Newport 1996), and predicting the
  next letter is how a network finds them (Elman 1990). The cortex is an explicit, exact version of that statistics.
- **Its ceiling.** On enwik8 (100 MB of English Wikipedia), the best context-mixing compressor, cmix, codes at about
  1.17 bits per character (Mahoney, Large Text Compression Benchmark), and a 24-layer Transformer-XL trained offline at
  0.99 (Dai et al. 2019). PPM-class models are well above both. The harmony (M2) is what has to close that gap, while
  learning online, exactly and with sublinear growth. Whether it can is the question of this study.

## 12. Cevahir: one layer (M3)

On 24 November 2026 the owner asked for everything built so far to meet in one layer, from which Cevahir emerges:
the evidence mind of 14–24 November (Zihin, with every mechanism it was built with) and the cortex of this study.
This section is the design as built (`living/cevahir.py`, `living/_core.c`); NOTES.md section 8 measures it.

### 12.1 Two bodies, one place

- **Zihin** lives each word as a moment: its units and ties with per-origin evidence, and the mechanisms F1–F76
  (frames, families, relations, wholes, inheritance, conjectures, the ground). It is the part that generalises:
  it answers the six published SCAN test files, deduces, and says who does what in scenes never described.
- **The cortex** lives each letter (sections 3–5). It is the part that predicts: below PPMd and a character
  Transformer on every corpus measured (NOTES.md 6), but alone it is a parrot. It says what usually follows what it
  has lived.
- **They meet in the harmony.** The harmony is the one layer. Every origin gives a belief for each bit of the next
  symbol; the gate combines them; the calibration judges the result by the rule. Zihin becomes one more origin of
  it, and nothing else changes in the harmony.

### 12.2 Zihin as a voice of the harmony

Before each token, Zihin's expectations on its text channel become the harmony's **voice**.

- **Candidates.** The tokens c with e(c) > ½ (believed above not knowing), at most 12, each weighted e(c) − ½.
- **How much it knows.** α = 2(e_best − ½), the complement of the evidence core's not-knowing measure u.
- **Forms.** A token is offered in the forms it was lived in: a gap lived before tokens in the same situation (in
  an answer or not, the first token of an utterance or not), times a way the token was written in the same role.
  Case and spacing therefore come from life, not from a rule. Zihin's tokens are lower-cased; the harmony's symbols
  are what was written.

In the core, at each symbol the candidates still consistent with what was lived since the voice was set give the
voice's belief that the next bit is 1, given the bits decided so far:

    p₁ = q · W₁ / W + (1 − q) · ½

- W is the weight of the candidates whose next symbol has the decided prefix, and W₁ the weight of those with a 1
  next.
- q is the one rule on the voice's own record: how often its top symbol was right, per sixth of α and per first
  symbol of a form or a later one.
- The gate gets stretch(p₁) and 2p₁ − 1 as inputs, and its weight sets are tripled by the voice's state: absent,
  below the speaking line (α < 1/6, which is 2(e* − ½)), or at it.

With the voice off, the core is bit-identical to M2 (tests/evolution/test_living_fast.py, and the recorded M2 rows).
On a stream of random words that no context predicts, a voice that knows the next word takes the harmony from
1.363 to 0.009 bits per character, and a voice that names a random word is ignored (1.355): the gate learns whom to
follow.

### 12.3 The prefrontal decision

When Cevahir answers, Zihin answers first, as it always did: its race, its speaking line, its wholes answered part by
part (F43). That is the **plan**, the thought, each word with its belief e. Then the answer is said one token at a
time:

1. **The same token.** The harmony's own next token (its most probable symbols up to the end of one token,
   believed with the product of their probabilities) and Zihin's next token are compared. If they are the same
   token, it is said in the harmony's form if either belief reaches e*.
2. **Different tokens.** The record of disagreements in this situation decides who leads. That record is kept
   while Cevahir lives: at each token heard, Zihin's top candidate and the harmony's guess are compared, and where
   they differ and exactly one of them is right, the record counts whether it was Zihin. Its belief is the rule on
   that count, per situation (in an answer or not) × (first token or later) × (Zihin at the line or below). The
   leader is said if its own belief reaches e*.
3. **Zihin has nothing to say.** When Zihin says nothing at all, the harmony speaks alone only as far as its record
   when alone allows. That record counts how often its guess was right where Zihin expected nothing, in this
   situation; the harmony speaks only if the rule on it is above ½. Once Zihin's plan is said, the harmony adds no
   word of its own: the plan is the thought, and its end is part of it.
4. **Neither.** Otherwise nothing more is said.

How Zihin's token is written (its gap, its case) is the harmony's choice: of the gaps and written forms lived in
this situation, the one it believes most here. A line end is never chosen, because an answer is one line.

**Why a record and not the gate alone.** In a situation it never lived, the cortex escapes to a shorter context and
can be confidently wrong. Having lived "zap" answered "ZAP .", it is sure that " ." follows "ZAP" also after
"zap twice". Over a life where Zihin and the cortex are mostly both right, the gate cannot learn who generalises.
That is learned where it shows: in the disagreements where one of the two was right. This is the evidence core's
trust per origin, learned from what its claims came to, and F49's record per way of reading. In a small
add-primitive world (NOTES.md 8.1), the first form, with the gate alone, answered 0 of 6 commands with the new
primitive; with the record, 6 of 6, as Zihin does. The harmony alone answers 0 of them and is silent where it does
not know.

### 12.4 What is not yet one

- **Two substances.** Zihin's units and ties and the cortex's contexts are two stores. They meet in the harmony and
  in the forms lexicon, not in one table. Moving Zihin's readings into the core, one at a time and each measured
  against the bridge, is the road to one substance.
- **The amygdala, a first form: the record of outcomes.** When an answer is judged (``Cevahir.feedback``), Zihin
  learns it as it always did (a value at the end of the episode), and every token of the answer counts for the voice
  that led it, in its situation, in the records the prefrontal decision trusts by: an answer judged right counts for
  the one that led, one judged wrong against it. The turn waits to be judged before the harmony lives it, and a wrong
  answer is not lived as something said rightly (the question alone is). No constant enters: trust stays the one
  rule on counts. A salience weight on learning, and the hippocampus's sleep that consolidates both stores, are not
  built yet.
- **Speed.** Zihin's moment is Python and costs milliseconds; the cortex's symbol costs microseconds. On text,
  Cevahir reads at Zihin's speed.
- **Facts that change.** Zihin's counts weigh every telling alike, whenever it came: three tellings of an old place
  outweigh one of a new. F76 (the newest holds, a switch of Zihin, off) reads what came last as a way of its own with
  its own record, and Zihin, and Cevahir with it, then says the place told last (NOTES.md 8.30, 8.31). It is off until
  measured on the published sets. A record that follows how far a voice can be trusted now, not how far it could be
  at first, is not built: read from the past, the record of Zihin alone holds back what Zihin has learned since.

### 12.5 The hippocampus's analogy: several places at once (`living/analogy.py`)

Zihin's frames generalise one place: a lived utterance with one item replaced by a category. Most utterances a mind
meets differ from every one it lived in two or three places at once. With a thousand lines of COGS lived, 85 % of
its test sentences are two or more words away from all of them, and Zihin answered none. With the whole training
file lived, 53 % of the in-distribution test sentences are one word away from a lived one and 89 % at most three.
Generalising several places at once is therefore what a mind that has lived enough needs, and it is the same reading
as a frame, done on the episodes the hippocampus keeps.

- **Episodes.** Every dialogue lived is kept as it was said: the tokens of the utterance and of the answer.
- **Neighbours.** For an utterance U, the lived utterances of the same length that differ from U in at most five
  places, found through an index of (length, place, token).
- **The relation of a word to the answer.** For a word w, the answer token t that came with it, believed by the one
  rule on the dialogues w was said in: e(t | w) = (W⁺ + ½) / (W + 1), with W⁺ the answers that held t. Tokens held
  by half of all answers or more are the answers' own structure, not any word's. A relation is believed when e
  reaches the speaking line. On equal evidence the word itself is preferred (Zihin's F34, the asked word itself),
  then the token that shares the longest beginning with it (F36, word forms), then the token most the word's own (the
  rule on how many of the token's own dialogues held the word), then the one first met with it. The choice must not
  depend on anything but what was lived: chosen first by the order in which a set happened to hold the candidates,
  it differed from one process to another, and with it one sample of COGS (NOTES.md 8.15).
- **The plan.** For a neighbour (U′, A′), each place where U and U′ differ (u in U′, v in U) is filled: every token of
  A′ that is u's relation becomes v's relation, or v itself where u's relation is u itself and v was never met (a
  word lived again and again that never brought a token of its own, as "The" in COGS, is no name to stand for one:
  NOTES.md 8.17). All
  places are filled at once, so two roles that swap places ("Veli'nin babası Ali" from "Ali'nin babası Veli") are
  read as a swap; one token that would have to become two gives nothing. Nor does a place whose word also stands,
  unchanged, elsewhere in the neighbour: its tokens cannot be told from the other occurrence's ("run left after run
  left" says nothing of "run left after jump left"; replacing every I_RUN would make both actions jumps). A neighbour
  with a place that cannot be filled gives nothing. The nearest neighbours vote (the fewest places first, at most
  eight), and the plan is the answer most of them give.
- **What a word's token always had after it.** A word is said between neighbours, and where it was said between
  neighbours of the same shapes (a word with a relation as one mark), the one or two tokens that its relation always
  had right after it in the answers are its own there (the end of an answer counted as what comes after: where the
  token stood last, nothing came after it). A neighbour whose answer, filled, gives the token of a word put in other
  tokens after it than those, or ends after it, says of that word what was never lived, and gives nothing. "smiled",
  said after a noun at the end of a sentence, was always answered "smile . agent", and "rolled", said there, always
  "roll . theme": "a hippo smiled" is not answered from "a hippo rolled" as "smile . theme", whatever the vote.
  Where a word was never said between such neighbours, nothing holds the answer back: a verb met only in active
  sentences stands in a passive one, a name met only as a subject stands as an object, and a verb met only without
  an object takes one. Which tokens follow a token is not named anywhere; it is read from the answers lived, where
  the word was said as it is said now. (Words of a kind, 12.6, ask only that two relations have held one same place,
  and every verb has followed a name's conjunct and come before "."; a verb that takes both an object and none has
  had "agent" and "theme" both after it, so a kind cannot tell the verbs apart where it matters: NOTES.md 8.16.)
- **Its belief: its own record.** Every dialogue lived is first foreseen by the analogy from the episodes already
  kept, and the record of how often an answer foreseen over d places was right is the analogy's own: the evidence
  core's trust in a way of reading, learned from what its claims came to (F49). A plan is believed by the rule on the
  record for its places, times the share of the neighbours that agree. Living the COGS training file, the record
  reads 98.5 % right over one place, 98.3 % over two, 96.1 % over three, 91.8 % over four and 78.1 % over five.
- **The prefrontal choice of the thought.** The analogy is a thought only if its own belief reaches the speaking
  line. Where Zihin has no plan, the analogy's is the thought. Where the two plans differ, a record decides, kept
  while Cevahir lives: before a dialogue's answer is heard, the analogy foresees it, and at each token of the answer
  where the analogy's token and Zihin's expected one differ and exactly one of them is right, the record counts
  whether it was the analogy's. Zihin's token is guessed knowing the tokens said before it; the analogy's plan is made
  once, whole, before any of them. So the plan is a guess of a token only while it has agreed with every token said
  before it: from its first slip on it says nothing of the rest (counting every later token against it made the
  record favour Zihin wherever answers are long; on COGS's generalisation set Cevahir then answered 13 fewer than the
  analogy alone, NOTES.md 8.7). The analogy leads if the rule on that record is at least one half. The thought is
  then said token by token as in 12.3. Where the analogy foresees nothing, Zihin's plan is the thought. A record of
  Zihin alone is kept: of the answers lived that the analogy foresaw nothing of (no plan, or one under the line) and
  whose first token Zihin would have said (its guess at the line), how many it would have said whole, each token its
  guess at the line, knowing those said before it, each claim with how far Zihin believed it. Holding the plan back
  where that record is under the line (``hold_alone``) is a switch, off: built for Cevahir's wrong answers on COGS,
  it holds nothing there (no claim is lived in COGS's life), while on SCAN one failed claim held back 21 of 50
  answers Zihin had right, and the record read as Zihin speaks and as far as it believes (``alone_now``) held back
  every answer of four splits (NOTES.md 8.31). Measured alone, the analogy answers 0 to 27 of 200 test commands of each
  published SCAN split (a word there changes how often an action is repeated, which one substitution cannot carry),
  where Zihin answers 199 or 200; on COGS it is the other way round. Each mechanism covers structure the other does
  not, and the record says which to follow.

Two constants enter, both bounds of a search: at most five places and eight neighbours. How far an analogy over
many places is believed is not a constant: it is its record. A dialogue forgotten (`Cevahir.forget`) leaves the
episodes exactly.

### 12.6 Words of a kind, wholes, and answering by parts (`living/analogy.py`)

The analogy of 12.5 fills places one for one. Two things it cannot do. It lets any two words with relations stand for
each other, so "to" (its relation "recipient") may stand for "on" (its relation "on") because both sit at the same
place of two utterances of the same length, and the answer it then says is wrong with a high belief. And it answers
nothing longer than what was lived: an utterance with one more phrase inside it has no neighbour of its length. Both
are mended by the same store of episodes.

- **Words of a kind.** A relation's profile is the places it has held in the lived answers: the two tokens before it
  and the one after it, every relation of a word read as one mark and every number as another. Two relations are
  alike when they have held at least one same place; a relation not yet met in an answer of more than one token is
  not held back. A place is filled only by a word whose relation is alike to the one it replaces. "recipient" has
  followed only a verb's name and "on" only "nmod", so the two never stand for each other, while "on" and "in", two
  verbs or two nouns do. (Asking for half of the places shared was tried first: it kept apart what COGS asks to put
  together, a noun met only as a subject and the nouns met as objects; NOTES.md 8.7.)
- **Wholes, and the record of a frame.** A part of an utterance is read with its frame: its own tokens, the two before
  it and the one after it, every word that has a relation read as one mark. How far a part of a frame is a whole of
  its own is the frame's own record, kept by the one rule as the analogy's is (12.5), but read from what was lived
  rather than while living: up to 24 lived examples of the frame are each foreseen as if they had not been lived (the
  part taken out, the rest's answer read, lived or the analogy's at the speaking line, and the change that the nearest
  other example with the same surroundings shows put back, as below), and the rule on how often that gave the answer
  that was lived is the record. An asked rest is answered by parts too, so a rest the analogy does not answer is read
  one step on, by taking one part out of it (within that step records read their examples' rests by the analogy
  alone, so that none waits on another being read): a clause in a long utterance ("A girl admired that a donut was
  lended to the frog by Ryan ." once "that a baby wished" is out) is foreseen as it would be asked (NOTES.md 8.23).
  An example whose rest has no answer even so, or that no other example can be put into, counts as not right: a part
  is a whole as far as taking it out leaves what can be answered and putting it back gives what was lived. (Counting only the examples that could be foreseen, as the analogy's own record counts only what it
  foresaw, raised the records of parts that are not wholes, which were then tried first and spent the search: COGS's
  pp recursion fell from 15 of 18 to 1; NOTES.md 8.14.) A frame is a whole when its record reaches the speaking line. This is Zihin's working
  memory (F43, wholes) read from the other side: there the parts of a long command are answered and put together;
  here a part is known to be one because putting it back as other examples do gives what was lived.
- **Answering by parts.** An utterance that no plan answers at the speaking line is answered by taking out a whole,
  answering the rest (by the analogy, or again by parts), and putting the whole's effect back. Its effect is read from
  the lived utterances that hold a part of the same frame: what taking their part out changes in their answers (the
  change between the answer of the rest and the answer lived), among those whose part has the same surroundings as
  the asked part: the three tokens on each side, read by shape (a word with a relation as one mark, every other word
  as itself), and the token right beside the part on each side read by its kind (below: the token its relation has
  right after it), for what comes right before or after a part decides what it hangs on: "that a cockroach gave"
  before "Oliver the cake" is not "that a prisoner liked" before "that the sandwich was slid", though "Oliver" and
  "that" are one mark (NOTES.md 8.21). What a part changes can depend on what stands near it, and the change carries
  only what the examples show: SCAN's "around" turns four times for each turn that "twice" or "thrice" asks, so an
  example with "twice" after the part says nothing of "thrice". That change is put into the answer of the asked rest.
  The part's own words map by their place in the part (each word to its relation, as in 12.5, alike by kind); the
  rest of the example maps through the alignment of the two rests' answers, read by what they say about the places
  near the part (the numbers that name them, and the relations of the words that stand there). Beyond the three
  tokens on each side every place reads alike, so a place far from the part, and the token of its word, is read by
  what links it to the place right beside the part: the items of the answer followed link by link ("table" is
  governed through "nmod . on" by "brush", and "brush" through "theme" by "lend"), and the example's far place is put
  for the asked answer's place that governs the place beside the asked part through the same role. Taking "to a
  hero" out of "Caleb lended a brush on the table to a hero" puts the recipient back on the verb that governs the
  thing given ("teleport" in "Emma imagined that a boy teleported the notebook to a hero"), not on the first verb,
  which alone stands as far from the part (NOTES.md 8.22). A token of an answer that is a whole number smaller than
  the utterance's length is read as a place of it, and moves on with the places when a part is put in. One change can be written in several ways where the tokens repeat; it is put in as far on
  as the tokens allow. Where an answer names no places of what was said, an example whose part holds a word with a
  relation that also stands outside the part lends nothing: which occurrence its tokens belong to cannot be told (an
  answer that names places tells them apart by their places). What the example put at the very beginning of its
  answer goes at the beginning, and where the answer lists runs of the same shape, a new run goes by its place among
  its neighbours (a list keeps the order of what was said). The examples nearest the asked part vote (at most
  eight). A step's belief is the frame's record
  times the share of the examples that agree; a plan's belief is that of its weakest step, the rest's own included.
  The steps of a chain read the same kind of change again and again (the phrases of a nested phrase are taken out by
  the same frames), so their errors go together rather than one by one: the chain is believed as far as its least
  believed step, and no further. (Multiplying the steps' beliefs, as if each could fail on its own, left most phrases
  nested deeper than four silent; NOTES.md 8.14.) The most whole-like parts are tried first, then the rightmost; a
  search tries at most 4,000 parts and takes out at most 16 one after another (the bound decides only where a search
  is spent: a plan found within a smaller bound is found the same within a larger one).
- **Exchanging a part.** Some parts cannot be taken out: what remains would not be an utterance ("Emma offered Lina to
  a girl" without "Lina"). Such a part is exchanged for a part lived between the same neighbours ("the cake"); the
  rest ("Emma offered the cake to a girl") is answered as the exchange's record answers the rests of its examples
  (the one lived, or the analogy's own at the speaking line, never by parts in turn: a rest put together by parts
  would be believed beyond anything the record measured; NOTES.md 8.20), and what that exchange changes in lived
  answers is put back. The change is read wherever a part of the asked part's shape was lived where a part of the other's shape
  was lived too ("Emma saw Liam" and "Emma saw the cake"), the like places first: there the answer loses the run of
  the noun phrase and names the name where it named the phrase's place. The places of the part put in are marked
  while it stands, and a change that leaves any of them named is not put. Only parts whose words have relations are
  exchanged: what such a word brings to an answer is its own token wherever it stands, while what a word without one
  brings depends on where it stands (SCAN's "right" turns once, or four times after "around"). And a part is
  exchanged only for one that holds a word with a relation and shares no word with it: an exchange puts one part's
  meaning where another's was, and a part with no relation has none of its own to give way (SCAN's "turn", whose
  turns belong to the side named after it, gives way to "walk" four times in "around" and twelve times in "around ...
  thrice"), while a part that holds the asked one ("jump opposite" for "jump") only adds a word that brings no token
  of its own. The change is read first from lived parts of the asked part's own frame with the same surroundings;
  where there are none, from the places where both kinds of part were lived; and only from examples whose part, too,
  shares no word with the part put in: exchanging "look" for "look opposite" in "look left after run" keeps "look"'s
  token, and says nothing of what becomes of it where "run" gives way. Each case has its own record, read as a
  frame's is: lived examples of the asked part's shape exchanged for the same part, foreseen as if not lived, with
  the change read from an example with the same surroundings or, for the second case, from one with other
  surroundings (does a change read in one place carry to another?). An exchange's belief is that record times the
  share of the examples that agree. Taking a part out is an exchange for nothing; exchanges are tried for the asked
  utterance's own parts only, after the parts that can be taken out, on a quarter of the search's bound: for every
  part the two parts lived most often in its place first, then the next eight, on a bound of their own (before the
  full stop after "the boy" the parts lived most often are phrases, "on the table", and a noun phrase for the name in
  "Emma gave the boy Paula" comes ninth).
- **The order of an answer's lists.** A change is put back where its example put it, and a run so put can stand out
  of the order the answer's lists keep: a definite noun's run after a conjunct, a recipient after a modifier of the
  theme. Where the lived answers keep lists, an answer put together by parts is said in their order. What the lists
  are is read from the lived answers: a token that ends items and one that stands between them, every item naming a
  place, taken as the pair that leaves the most lived answers as they were when each answer's items are put in the
  order of the places each names, the ended items first. The places an item names are its numbers and its names (the
  own token of a word said once, a word whose token is the word itself), each read as the place where it was said:
  "see . agent ( x _ 1 , Ali )" comes before "see . theme ( x _ 1 , Can )" because Ali was said before Can. The pair
  is believed by the rule on all lived answers that hold it and used at the speaking line; it is read again as life
  doubles. In COGS it is ";" (ending each definite noun's run) and "AND" (between conjuncts), and every lived answer
  that names places is given back by it from its items in any order (by the first place alone, with ties left as
  they came, 11,157 of the 24,012; by the numbers alone 17,481; NOTES.md 8.19); SCAN's answers name no places, so no
  pair is read and nothing is reordered. A plan over places is a lived answer filled and keeps its order already; only
  plans by parts are reordered (NOTES.md 8.18), and these at every step of a chain: the next step reads the answer
  item by item against its example's, and an answer out of order puts the next change inside the wrong item
  (NOTES.md 8.24).
- **A second search, with words read by their kind.** Frames read every word that has a relation as one mark, so a
  phrase hung on a subject ("A donkey [in the room] sold Ella a donut") has the frame of a verb with its object ("A
  donkey [sold the cake] to Emma"), a frame whose parts are not wholes; and phrases were lived only on objects, so no
  lived phrase stands in the surroundings of one on a subject. Where the search above is silent, a second one is made.
  In it a word with a relation is read by its kind: the mark and the token its relation has right after it in the
  lived answers, where one token does by the rule (read from COGS's training file, 408 relations are followed by "(":
  nouns, prepositions, "to" and "that"; 134 by ".": verbs; 102 by ")": names). A part that no lived part of its frame,
  so read, stood in the surroundings of (the three tokens on each side, read so: where the changes of the first search
  are read from) is taken out first, by what parts of its shape do elsewhere: the change is read from lived parts of
  that shape, nearest first. Such parts have no record of their own there, so they are taken out from the right: a
  phrase hangs on what stands before it, and taking out the left one of two first leaves the right one hung on the
  subject ("A governor on the surface beside a log": "beside a log" first). A step is believed by the shape's record
  elsewhere, read as a frame's is: up to 24 lived examples of the shape, each foreseen as if not lived with the change
  read from the nearest example with other surroundings (one whose rest has no answer, or that nothing can be put
  into, counts as not right), times the share of the examples that agree. The rest is answered by the second search
  again, so a subject with two phrases takes two such steps. What the first search answers is answered as before:
  words read by their kind in every frame, the search's and the exchanges', change what it does (NOTES.md 8.19).
- **A change read from other surroundings.** Where its example put a run says nothing of where it goes in other
  surroundings: put back token by token where its example put it, the change of a phrase in other surroundings went
  inside another item, left its "AND" at the end, or gave a definite noun's "*" to the wrong noun (the record of
  phrases like "on the table" was 0.50). So a change read from other surroundings is put as the items the example's
  answer gained and lost: the lost ones must stand in the rest's answer and are taken away, the gained ones are put
  in, and the answer is said in the order of its lists (that record is then 0.94). A token is read by where it stands
  near the part (before it, in it, after it), where the words standing at that place in the example and in the asked
  utterance are of a kind; else through the alignment of the two rests' answers, and no place or word's token is put
  for one whose word is of another kind (in "A donut was given to a teacher" the word before the phrase is a verb, in
  "The penguin lended a cake to a teacher" a noun: that example lends nothing). Where the lived answers keep no lists,
  a change read from other surroundings is not put at all: where its example put a run is all there is, and it says
  nothing of other surroundings (SCAN). Within the same surroundings a change is still put where its example put it:
  there it is where it goes, and a change inside an item ("sell . recipient ( x _ 3 , x _ 1 )" become "sell . theme
  ...") is put by its changed tokens alone where the item as a whole has no counterpart (the asked item names "Paula"
  where the example's named a place). Where the item a change is made inside has a counterpart, the change is made in
  it: the example's item, put into the asked rest's answer, is found there once, and the change is made in it at the
  same place. Read token by token, the two answers line up where their longest runs do, and a far item can win over
  the one that holds the part (a far clause whose agent is a thing, as the example's is, over the clause of a name
  that holds the part). An item said of a place is put as the item of the word at the place it is put to: a token
  is told apart by its place, and the example's clause verb may also stand in the part, where its token would be put
  as the part's own (NOTES.md 8.24).
- **The roles an answer gives.** The rests of a chain are steps on the way, often utterances no one would say ("Aubrey
  liked the ring to a cat ."), and their answers by the analogy are what makes a chain go: held to the roles their
  tokens were lived with, 95 right clause recursions fell silent (NOTES.md 8.23). But the answer given is believed no
  further than the roles it gives its tokens were lived: an item that gives a token a role it never took in a lived
  answer ("like . recipient") is believed by the rule on the tokens that took every role it took, how many of them
  took that one too, and the answer's belief is the least of its own and these. A token that took no role is not read
  so: nothing lived says what roles it takes. A token that took few takes a new one as the tokens that took those
  do, as the generalisation set asks (a verb met only without an object, asked with one, takes an agent as almost all
  verbs that take a theme do). Where the first search's answer is held back so, the second search is made; the items
  are the answer's "token . role ( ... )", read as the far places of 12.6 read them (NOTES.md 8.24). So too the roles
  said of one place: two roles the answer gives one place that no lived answer gave one place together (two phrases
  on one noun, a clause and a thing of one verb) are believed by the rule with nothing for them, over the lived places
  that took the rarer of the two (NOTES.md 8.26).
- **Tokens that take turns.** Two tokens that take turns in every lived answer that holds either (the first opens,
  the second closes, never two open, never a close with none open, none left open at the end: "(" and ")", "x" and
  "_" on COGS) take turns in the answer given, or it is believed by the rule with nothing for it, over the times a
  lived answer could have broken them (every opening after an answer's first). An item put back across the end of
  another opens one inside the other: counting the tokens does not see it, their turns do. Tokens met at most once in
  every answer could break nothing and are no pair (NOTES.md 8.29).

Nothing of COGS is named here: no word, mark or form of its answers. What the mechanism assumes is general: an answer
may name the places of what was said by their numbers or by a word's own token, and the items of an answer keep the
order of what they name. Where an answer does neither, the change of a part cannot be put (the example gives nothing)
or the examples disagree (the plan stays under the line). The analogy's own record (12.5) is kept by its plain form:
while Cevahir lives a dialogue it foresees the answer by the analogy over places, without parts. Answering by parts
is done when Cevahir is asked, and its records, the frames', the exchanges' and the shapes' elsewhere, are read then
from what was lived
(again each time life has doubled, and before answering after anything new was lived).

## References

- Cleary, J. G., and Witten, I. H. (1984). Data compression using adaptive coding and partial string matching. *IEEE
  Transactions on Communications*, 32(4), 396–402.
- Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V., and Salakhutdinov, R. (2019). Transformer-XL: attentive
  language models beyond a fixed-length context. *Proceedings of ACL*, 2978–2988.
- Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2), 179–211.
- Howard, P. G. (1993). *The design and analysis of efficient lossless data compression systems*. PhD thesis, Brown
  University.
- Krichevsky, R. E., and Trofimov, V. K. (1981). The performance of universal encoding. *IEEE Transactions on
  Information Theory*, 27(2), 199–207.
- Kumaran, D., Hassabis, D., and McClelland, J. L. (2016). What learning systems do intelligent agents need?
  Complementary learning systems theory updated. *Trends in Cognitive Sciences*, 20(7), 512–534.
- MacKay, D. J. C., and Peto, L. C. B. (1995). A hierarchical Dirichlet language model. *Natural Language
  Engineering*, 1(3), 289–308.
- Mahoney, M. *Large Text Compression Benchmark*. https://mattmahoney.net/dc/text.html (enwik8 results).
- Mahoney, M. V. (2005). *Adaptive weighing of context models for lossless data compression*. Technical Report
  CS-2005-16, Florida Institute of Technology.
- McClelland, J. L., McNaughton, B. L., and O'Reilly, R. C. (1995). Why there are complementary learning systems in
  the hippocampus and neocortex. *Psychological Review*, 102(3), 419–457.
- McGaugh, J. L. (2004). The amygdala modulates the consolidation of memories of emotionally arousing experiences.
  *Annual Review of Neuroscience*, 27, 1–28.
- Miller, E. K., and Cohen, J. D. (2001). An integrative theory of prefrontal cortex function. *Annual Review of
  Neuroscience*, 24, 167–202.
- Moffat, A. (1990). Implementing the PPM data compression scheme. *IEEE Transactions on Communications*, 38(11),
  1917–1921.
- O'Reilly, R. C., and Frank, M. J. (2006). Making working memory work: a computational model of learning in the
  prefrontal cortex and basal ganglia. *Neural Computation*, 18(2), 283–328.
- Olsson, C., et al. (2022). In-context learning and induction heads. *Transformer Circuits Thread*.
- Saffran, J. R., Aslin, R. N., and Newport, E. L. (1996). Statistical learning by 8-month-old infants. *Science*,
  274(5294), 1926–1928.
- Shkarin, D. (2002). PPM: one step to practicality. *Proceedings of the Data Compression Conference*, 202–211.
- Teh, Y. W. (2006). A hierarchical Bayesian language model based on Pitman–Yor processes. *Proceedings of
  COLING/ACL*, 985–992.
- Willems, F. M. J., Shtarkov, Y. M., and Tjalkens, T. J. (1995). The context-tree weighting method: basic
  properties. *IEEE Transactions on Information Theory*, 41(3), 653–664.
- Wood, F., Archambeau, C., Gasthaus, J., James, L., and Teh, Y. W. (2009). A stochastic memoizer for sequence data.
  *Proceedings of the 26th International Conference on Machine Learning*, 1129–1136.
- Ziv, J., and Lempel, A. (1978). Compression of individual sequences via variable-rate coding. *IEEE Transactions on
  Information Theory*, 24(5), 530–536.
