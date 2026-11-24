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
| M3 | PFC and amygdala; the word-level cortex over word numbers | sentences said to the end and their coherence; retention of salient facts after interference; the earlier dialogue and school tests through the new architecture | next |
| M4 | hippocampus and sleep | memory bounded by B; the cost in bits of forgetting; growth with forgetting; retraction after consolidation | |
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
