# Cevahir Zihin: a Living Evidence Mind

**Muhammed Yasin Yılmaz**

CEV-2026-11 · Version 2026.11.24 · Published 2026-11-24

Research report / preprint. Not peer reviewed.

## Abstract

A learning system with no neural network, weight matrix, embedding or gradient: whatever it perceives becomes a unit, every relation a tie that keeps graded evidence for and against it per origin, read by one rule (the NARS expectation, k = 1) and learned at every moment, with the evidence of any single experience exactly retractable. On this core 75 mechanisms were built and measured one at a time. Having watched about 1,090 published SCAN training commands twice, it answers all six published SCAN test files exactly (26,227 of 26,227); COGS stays at 0 with its causes measured. On 13,290 Turkish sentences it says a read sentence for 146 to 151 of 200 two-word cues after one reading, a 7.3M-parameter Transformer after an hour of training for 47. A grounded layer answers who does what in situations never described 48 and 54 of 72 (Turkish, English) against 29 and 42 for a Transformer given the scene.

## Interpretation of the findings

Evidence accounting with one reading rule gives learning without a training phase, without interference and with exact retraction, and compositional generalisation wherever an answer is made of the replies its parts were lived with; the mechanisms that examine and extend the mind itself are switches measured with their failures.

The worlds are small and generated, the Transformer baselines small and trained on one processor, a frame generalises one place (COGS), the cost of the whole mind grows with what it holds, and the grounded world gives its features as opaque codes. Not peer reviewed; no claim of general intelligence.

## Publication and evidence

This edition presents the research record below with its method, results, negative controls and source links, or
where that record was withdrawn, the note that says where its full text is archived. The original file is kept as
[docs/research/living_evidence_mind_2026_11_24/PAPER.md](../../research/living_evidence_mind_2026_11_24/PAPER.md); links are rebased to this edition's location. Dates and earlier
publication statuses inside the record belong to the historical record; this edition's date is above.

This work is a separate text in the [publication series](../README.md). The DOI identifies the release archive that
holds every text and the code; it does not mean that each paper has a DOI of its own. Cite the work's identifier
together with the release DOI. The scope of the computational checks is described in the
[validation record](../evidence/validation.json) and the [publishing principles](../PUBLISHING.md).

The project and its research direction are Muhammed Yasin Yılmaz's. AI assistance was used for code, computational
experiments, literature review and text preparation. Neither this statement, the DOI record nor the automated tests
are external scientific peer review.

---

## Research record: Cevahir Zihin: a Living Evidence Mind

*Learning by living, exact retraction, compositional generalisation and grounded meaning, without gradient training*

**Muhammed Yasin Yılmaz** · independent researcher · version 2026.11.24 (release v7)

Research report / preprint, not peer reviewed.

## Abstract

We describe Cevahir Zihin, a learning system with no neural network, no weight matrix, no embedding and no gradient:
a *living evidence mind*. Whatever it perceives becomes a unit, and every relation between units is a tie that keeps
graded evidence for and against the relation, separately for six origins (sensed, told, imitated, its own act,
inferred, imagined). One rule reads every expectation, e = c (f − ½) + ½ with c = W / (W + 1): the expectation of a
truth value in the Non-Axiomatic Reasoning System, weighted by a trust in each origin that the mind learns from its
own experience. The mind learns at every moment it lives; there is no training phase. A ledger of every increment
makes the evidence of any single experience exactly retractable.

On this core, 75 numbered mechanisms (F1 to F75) were built one at a time, each behind a switch, each measured with
its negative results; several were measured and not kept. We report four groups of results.

1. **Compositional generalisation.** Having watched about 1,090 of the published SCAN training commands twice, the
   mind answers every command of the six published SCAN test files exactly: 26,227 of 26,227 (26,223 before its last
   mechanism). On the generated splits, Transformers trained from scratch on the same dialogues answer almost none
   of the test commands (0 of 844 within the window, 1 of 7,706 at every length). COGS remains at 0, and its causes
   are measured.
2. **Recall of real text.** The author's private documents are 13,290 Turkish sentences. Cued with the first two
   words of 200 read sentences, the mind says a read sentence in 146 to 151 cases after one reading (about 320 s of
   one processor core). A 7.3-million-parameter Transformer, trained from random weights on the same machine for an
   hour, does so in 47.
3. **Grounded meaning.** A meaning layer joins perception and words in the same evidence calculus. Asked who does
   what in situations never described, it answers 48 (Turkish) and 54 (English) of 72. A Transformer given the scene
   as tokens answers 29 and 42. It stays below that Transformer in naming an object of a colour never seen (16 and 19
   of 23 against 21 and 23).
4. **Speed.** Every piece of speed work kept the mind's answers bit-identical.

We also state what is not claimed. The worlds are small and generated, the comparisons are in a small-data regime
on one processor, and nothing here measures general intelligence.

## 1. Introduction

A deployed neural language model is trained, frozen and then used. What it is told after training is not learned
unless it is trained again. What it learns later interferes with what it knew (McCloskey & Cohen, 1989; French,
1999). The contribution of a single example cannot be removed except by retraining or by methods built for that
purpose (Cao & Yang, 2015; Bourtoule et al., 2021). And nothing records where a belief came from.

This report describes a different construction, built and measured step by step from 14 to 24 November 2026. Its
design questions were the author's:

- Can a mind built from evidence accounting alone learn at every moment it lives?
- Can it keep what it knew bit for bit while learning more?
- Can it take back exactly what one experience taught it?
- Can it say where every word it says came from?
- Can it generalise to combinations it never lived?
- Can its words come to mean what it perceives?

The answers are partial, and they are reported with their failures.

**Contributions.**

1. *An architecture* (Section 2). Units and ties with per-origin evidence; one reading rule; learned trust; a
   seven-place focus in place of a context window; an advantage race that stays silent when nothing is known; a
   ledger that makes the evidence of any experience exactly retractable.
2. *A record of 75 mechanisms* (Section 3). Each was added as a switch, measured alone and in the whole, and kept,
   kept off, or eliminated on measurement. The eliminated ones are kept on record.
3. *Evaluations against Transformers trained from scratch on identical data* (Sections 5 to 8): learning while
   living and interference, SCAN and COGS, recall and a time race on real text, studied lessons and an examination.
4. *A grounded meaning layer* (Section 9) in the same calculus. Perception binds features to things, events and
   participants; a word refers to one of them; a construction's places hold participation profiles; answers are
   told from the meaning.
5. *A complete public record*: code, notes for every step, result rows, gates that require bit-identity, and a mind
   raised from nothing on public material only (Section 12).

## 2. The core

### 2.1 Units, ties and channels

The mind touches the world and itself through channels of eight kinds:

- *sense*: the end of a turn, the gist of an image or a sound, the inner state of not knowing;
- *act*: words, whoever says them, and the mind's own acts;
- *feel*: surprise, relief, conflict, load;
- *knob*: its own settings;
- *source*: where evidence comes from;
- *facet*: how a thing looks or sounds, or a piece of a word;
- *value*: a good or a bad outcome;
- *context*: who speaks, tone, the self.

Words arrive on one act channel, whoever says them. Whether a word was told, imitated or said by the mind itself is
kept as its provenance, not as a channel.

A *unit* is one of four kinds:

- an *atom*: a symbol on a channel;
- a *composed unit*: a lived sequence of up to seven items;
- a *frame*: a composed unit one of whose places is a category ("_ eats fish");
- a *category*: a class of units lived in the same places.

A *tie* relates two units. EXPECT ties carry a lag: 0 (together), 1 (right after) or 2 (a consequence later in the
same episode). PART ties hold the parts of a sequence; ABSTRACT ties hold the members of a category.

### 2.2 Provenance evidence and the one rule

Each tie keeps evidence for (W⁺) and against (W⁻) separately for six origins: sensed, told, imitated, acted,
inferred and imagined. The amounts are graded, not counts. A confirmation adds an amount scaled by how active the
expecting unit was and how strong the occurrence was. A failure adds an amount weighted by the tie's own
expectation. With the evidence weighted by the trust in each origin, every reading in the mind rests on one rule:

    f = W⁺ / W,    c = W / (W + k),  k = 1,    e = c (f − ½) + ½

This is the expectation of a truth value in NARS (Wang, 2013). With no evidence e = ½, which means "I do not know";
conviction grows with evidence. What to say, what to expect and what to be surprised by are all read by this rule.

Sensed evidence and the mind's own acts are direct experience; their trust is 1. The trust in the four indirect
origins starts at ½ and is learned. Indirect evidence is compared with the sensed evidence that later arrives on the
same tie: what holds raises the trust in that origin, and what fails lowers it. A mind that lives with several people
learns this trust per person (Section 5).

### 2.3 Attention without a context window: the seven-place focus

At each moment only F = 7 units enter the focus, chosen by a k-winners-take-all settling. The candidates are:

- what is perceived now;
- what is carried over from earlier moments (activation × (1 − 1/F) each moment);
- the recognised composed context;
- its categories and frames;
- what lived togetherness calls to mind.

Settling takes at most six rounds. There is no context window: a moment reads the ties of the units in focus and
never looks back over the past, so its cost depends on those ties, not on any context length. Attention is readable
(which seven units are in focus, and why), and there are no attention weights to train.

### 2.4 A moment

Roughly every word is one moment, in which the mind:

1. perceives what comes from outside and from inside (the echo of its own words, its feelings, its not knowing);
2. learns what comes next (lag 1), from the present of a moment ago only;
3. credits a good or bad outcome to every context in the episode (the consequence trace);
4. settles the focus;
5. learns togetherness (lag 0), so that facets bind to the item they accompany;
6. reads its expectations and, if it is its turn, races its options;
7. perceives its not knowing if it has nothing to say;
8. turns surprise, relief, conflict and load into percepts of the next moment;
9. writes every increment of the moment to the ledger.

A tie receives evidence against it only when its target's channel reported something else in that moment (the
*Expected-Failure Rule*), weighted by the tie's expectation, so a rare follower settles near the floor instead of
being refuted at full weight.

### 2.5 Speaking and not knowing

Options on one channel exclude each other. The leader is said when its advantage over the strongest rival reaches
the threshold θ = 1 within T = 6 steps. An option racing alone must therefore reach e* = ½ + θ / (2T) ≈ 0.583, the
*speaking line* used throughout. If nothing reaches it, the mind stays silent.

Not knowing is a measured inner state, u = 1 − 2 max(0, e_best − ½), perceived as an event after a silent moment.
How one says "I do not know", and how one asks, are learned from the people the mind lives with; the core holds no
sentence for it. Scripted reflexes and lessons that once spoke for the mind were removed on 15 November 2026.

### 2.6 Exact retraction

Every increment of every moment is written to a ledger. The evidence a single experience added is subtracted bit for
bit, and the increments of later moments stay untouched. This is exact for the evidence, not a replay of a life
without that experience: units and ties it created stay where later moments used them. Retracting one experience
takes 20 to 77 ms in the worlds of Section 5. The teacher's study loop (Section 7.4) uses it to take back a wrong
attempt before the right answer is shown.

### 2.7 What it is not

There is no neural network, weight matrix, embedding vector, backpropagation or softmax attention in the core. The
repository also holds a separate Transformer engine. It is used in this report only as the baseline: decoder-only
Transformers trained from random weights on exactly the data the mind lived.

## 3. The mechanisms

The core of Section 2 was built first (F1 to F19 are its design steps: the members' reading of a category, the
self/next slots, present-only learning, the expected-failure rule, the consequence trace, the continuation rule,
the resemblance vote, the advantage race). Every mechanism since was written after a measured failure had been
traced in the mind's own chain of computation. Each is a switch, measured off and on in the same lives, with the
gates of Section 4.2 bit-identical when it is off.

The program's *whole mind* (`python -m zihin`, the full configuration) turns on 36 switches: every mechanism below
except those marked "off" and those eliminated. The earlier *speaking configuration* (`--variant speaking`) is kept
for reading large texts. The table gives each later mechanism in one line, with the measurement that justified it;
the notes cited in Section 12 hold every measurement in full.

| F | name | what it does | measured (seeds as in the notes) |
|---|---|---|---|
| 20, 22 | Structural Imagination, Dream Sorting | carries a thought over to a structurally similar concept and scores each step for reality against lived evidence; dreams about the last topic, then judges which dreams came true | first forms; a dream found false is outweighed, not deleted |
| 23 | Situation Resemblance | a situation never lived is read by the lived situations most like it | part of the speaking configuration |
| 24 | Voice Credibility | learns how far a kind of "voice" (how many members, how much they agree) can be believed | +0.058 ± 0.025 early in one direction only; off |
| 26 | Sequence Starts | what followed a word where a sequence began with it is kept apart from what followed it anywhere | recall of read sentences 90 → 150 of 200 (with F27) |
| 27 | Lived Exclusion | what a situation was itself lived to be followed by counts against what a generalisation proposes there | (with F26) |
| 28 | Exact Relations | a frame's relation counts only where the member lived the value it names | with F26, F27 and listening while the world speaks: held-out answers of the chat configuration 0.57 → 0.78 and 0.04 → 0.81 |
| 29, 30 | Analogy by Interchangeability, Corroboration | words that filled the same places are interchangeable (NARS analogy); independent lived evidence adds up | greeting-fronted wordings never watched 0.17 → 1.00 answered, no wrong answer |
| 31, 32 | Paradigmatic Relation, Relational Binding | a frame's value may be what stands where its filler stands | meaning questions 0 → 0.67 |
| 33 | Relational Memory | what is lived with a person present is theirs; trust is learned per person | own greeting per person 0 → 1.00; a secret told to no one else |
| 34 | Reply Relation | a frame's value may be what its filler, said alone, was answered with | SCAN-like add-primitive 75 of 75 (Transformer 4) |
| 35 | Generalised Consequences | what an act would lead to is read from what similar paths led to | polite to new things 0 → 8 of 9 |
| 36 | Word Forms | a word never met is read as the lived word it shares its longest stem with (Harris's successor variety) | "merhabalar", "nasılsınız ?" 0 → 1.00 |
| 37 | Inheritance | deduction from facts read apart, exceptions winning (NARS deduction) | 48 of 48 (2 without); two-step chains 27 of 27 |
| 38 | Witness Reliability | a shorter context witnesses a claim as far as longer contexts ending with it went on so | in the whole mind |
| 39 | Replacement Witness | Mill's method of difference read from the mind's own life | the fine weather, not the clouds, leads |
| 40, 41 | Urge to Go On, Unsaid Known | the urge to go on is the unsaid share of what is known of the answer's topic | telling facts of things never asked 54 of 54 (from 20) |
| 42 | Episode Echo | what followed an item earlier in the episode is an echo | SCAN within 11 items 789 → 810 |
| 43 | Wholes | a working memory: a sequence said alone and answered is a whole; an utterance is heard in summary | SCAN add-jump, all lengths 7,706 of 7,706 |
| 44 | Lived Exclusion across channels | what a context went on with on any channel counts against what it never went on with | an answer that always ended does not go on |
| 45 | Told Apart | where two dialogues differ in one word, each word answers with its own share | SCAN "around right" template 0 → 500 of 500 |
| 46, 47 | Open Claims, Looking Up by Itself | a relation that names nothing for a filler is a want; the mind decides itself when to look | 20 of 20 wanted things answered after its own looks |
| 48 | Thinking by Itself | when to think is its own decision (surprise and what thinking came to) | thought about 24 of 24 surprising sentences, 0 of 24 expected |
| 49 | Fault Tracing | every way of reading keeps a record of what its claims came to, read with the core's rule | wrong answers in misleading frames 66 → 29; 8 frames in a row 120 → 14 |
| 50, 51 | Finding Out, Asking Itself | an answer resting only on others of its kind is held back and looked up first; new things are asked of oneself | 80 of 80 whole answers (9 without); 30 of 30 (5 without) |
| 52, 62 | Echo of the Same Words, The Sentence Being Told | the exact echo holds only where the utterance is a lived sentence from its beginning | (interactions of the whole mind) |
| 53 | Frames of Frames | frames that differ in one fixed word are a family; a new kind of question is read by the family | a new kind at first sight 0 → 40 of 40 |
| 54 | Relations Lived Apart | a relation counts only as far as it was lived apart from the dialogue it generalises | 0 → 40 of 40 where it bit |
| 55 | The Asked Place | a question is a known statement with one item replaced | 0 → 40 of 40 |
| 56 | The Kind of a Whole | a whole made of like parts has a form of its own | the four last SCAN errors corrected |
| 57, 58 | Trials, Built Relations | the mind tries its own ways of reading in controlled comparisons and builds relations no one wrote | 2 of 320 before; with trials 200 of 200 and a kind never asked 40 of 40; built chains 0 → 40 of 40 each (a hand setting: 8 of 120) |
| 59 | Conjectures | statements nobody said, carried by a link whose record decided for it, kept with their source | 227 conjectures, all hold; with nothing told in two fields 80 of 80 each |
| 60 | The Unseen Share | a frame whose members each went on with their own word does not count against a word it never saw | with paired trials, the first form of the three-field life 0 → 220 of 220 (the pair found in 5 of 5 lives) |
| 61 to 65 | (the whole mind's joints) | each act of its own time waits its own time; a generalisation agreeing with what was known is not tested by it; a share is tested before it is carried; the end of a self-question is not the world asking | 31 of 38 lines equal or better than each line's own configuration |
| 66 to 68 | An End Is an Outcome; The Category Is Split Where It Failed; Deduction Stands Against the Co-members' Vote | a way that makes the mind go on past an answer is found by its own trials; one experience against a calibrated vote splits it; a class's statement stands against co-members' votes | 0 → 40 of 40; 52 → 100 of 120; 0 → 3 of 3 |
| 69 | Its Own Turn Comes Last | what it learned at its own turn is booked apart from what it learned listening, and read last | misleading frames 66 → 40, never-asked things 82 → 13 wrong |
| 70 | The Open Window | a moment brings T things to mind, deepening each later moment the context stays | recall 2,746 s → 744 s, every score the same |
| 72 | The Subject Carries | a dropped subject is heard again with the thought that held | 0 → 80 of 80 in conversation, 60 of 60 in reading |
| 74 | Recall or Reason | a moment is first read as the speaking configuration reads it; recalled if that settles it within half the race | median answer 2.70 s → 0.64 s, the same words in 199 of 200 |
| 71, 73, 75 | Latching; Exemplars; The Reach of a Continuation | (Section 10) | measured and not kept |

## 4. Method

### 4.1 Worlds and data

- **Generated worlds.** Small generated worlds, each built to test one ability (named T-tests in the notes): classes
  and exceptions, deduction, long answers, wanting and looking up, a misleading frame, a half-empty library, kinds of
  questions, three fields of knowledge, a world of people, a grounded world (Section 9). Every world is generated
  from a seed; results are summed over 3 or 5 seeds.
- **SCAN** (Lake & Baroni, 2018): the generated splits, then the six published test files with the published
  training commands. **COGS** (Kim & Linzen, 2020).
- **The author's documents.** 13,290 Turkish sentences, about 186,000 words. They are private: only counts and times
  are published, and minds raised on them stay on the machine they were raised on.
- **Public lessons.** 24 lessons of public knowledge written for the study (Section 7.4); seven school subjects
  generated by a script; a few dialogue and text files of the first experiments.

### 4.2 Protocol

- **Switches.** Every mechanism is a switch and is measured off and on in the same lives with the same seeds.
- **The battery.** 37 measured lines from the worlds above are run again for every change to the whole mind, and
  every line that changes is reported.
- **Gates.** Four gates must be bit-identical whenever a new switch is off: the store digest of a fixed life, a
  plain run, a run with working memory, and 17 configurations. Speed work must be bit-identical with every switch on
  and off, in the words of every answer and in store digests.
- **Negative results** are reported in the notes beside the positive ones. A mechanism that does not earn its place
  is taken out, and its code kept on record (Section 10).

### 4.3 Baselines

The baselines are decoder-only Transformers (Vaswani et al., 2017) trained from random weights on exactly the
sequences the mind lived (each told sentence as a sequence, each dialogue as question ⏎ answer), with the same
units. Sizes range from 1 to 7.4 million parameters, with learning-rate schedules chosen for their size. Where
stated they are trained until they are exact on their own training items. Larger models, more data or a GPU would
change these numbers; the comparisons are for this regime only.

### 4.4 Hardware

Everything was run on a 4-core cloud machine, mostly one core per run, in Python with NumPy and PyTorch on the CPU
(the core uses PyTorch only as an array library).

## 5. Learning while living, interference and retraction

Two generated worlds with hidden classes, members, exceptions and a second domain, five seeds each:

- **Conviction grows with hearings.** c is 0.41 after one hearing and 0.78 after four or more. A correction weighs
  old evidence instead of deleting it (120 of 120).
- **Hidden classes are found** without being named (Jaccard 1.00).
- **Generalisation improves with age.** On instances never told the asked dimension: 93% and 81% (two domains) after
  600 experiences, 100% and 90% after 1,500.
- **No forgetting.** Facts learned early are bit-identical hundreds of experiences later, and living a second domain
  does not disturb the first (10 of 10).
- **Exact retraction.** The evidence of one experience is retracted exactly (10 of 10), in 20 to 77 ms.

On the identical stream of experiences (5 seeds × 2 worlds, identical probes), against small Transformers:

    held-out instances (never told the asked dimension)    world 1        world 2
    the mind ("members" configuration)                    90% / 97%      75% / 95%
    Transformer, online, one pass                          67% / 57%      60% / 50%
    Transformer, trained to convergence on the stream      77% / 67%      53% / 53%

A second domain lowers the one-pass Transformer's accuracy on the first from 0.85 to 0.74 and leaves the mind's at
1.00. The Transformer's exact counterpart of retraction is retraining from scratch.

The chat program's configuration first did poorly in these worlds (0.57 and 0.04). Two causes were found: the
worlds let it speak into their stream, so it lived its own words between theirs; and a relation reading took the
members' agreement on a value. Listening as in a chat, with F26 to F28, it reaches 0.78 and 0.81.

**First perception.** Image and sound files are perceived by plain pixel and signal arithmetic, with no trained
network, and recognised by resemblance. New images are recognised 73%, 83% and 67% right, new sounds 93%, 97% and
93% (chance 33%; 0% with resemblance off).

**Learning code from its consequences.** On six tiny functions, the code the mind writes is run and the test result
comes back as a good or bad signal. By round 4 it writes passing code for all six (0% without the signal, 11%
without trying in hesitation).

## 6. Compositional generalisation: SCAN and COGS

### 6.1 Within the window

F34 (the reply relation) lets a frame's value be what its filler, said alone, was answered with:

    test                                                         the mind                 Transformer from scratch
    add-primitive made small (4 primitives, 11 templates;        75 of 75                 4 of 75
      "jump" seen only alone)
    SCAN add-jump, the 844 test commands that fit 11 items       789 (F34); 810 (F42)     0 (1,570 of its 1,574
                                                                                            training commands exact)
    the same with a window of 12                                 844 of 844
    window 13, the 1,188 test commands that fit 12 items         1,188 of 1,188           not run at this size

Every command whose words, end and answer fit the window is answered exactly (596 of 596 at 11). The remaining
errors are answers whose last word is decided after the command's first word has left the window.

### 6.2 Working memory: beyond the window

F43 (wholes) gives the mind a working memory in its own terms. A sequence said alone and answered is a whole; a
watched dialogue whose answer is made of the replies of wholes in its utterance is lived again in summary. An
utterance is then heard in the summary, and each pointer is answered as if that part had been said alone. The window
bounds each level, not the answer.

On the generated splits:

- add-jump with commands of every length: 7,706 of 7,706 (1,082 of the 13,204 training dialogues watched twice; a
  Transformer on the same dialogues: 1);
- the length split: 500 of 500 (first 101; a Transformer 0);
- a Turkish task world: 492 of 492 (0 without).

F45 (told apart) gives each of two words its own share where two lived dialogues differ in that one word. The
"around right" template split goes from 0 to 500 of 500; a Turkish dance world the mechanism was not developed on
gives 600 of 600 at its first run.

### 6.3 The published SCAN files

About 1,090 of the published training commands were watched twice, and every published test file was asked whole:

    published test file           without F56        with F56
    right                         4,475 / 4,476      4,476 / 4,476
    opposite right                4,476 / 4,476      4,476 / 4,476
    around right                  4,474 / 4,476      4,476 / 4,476
    jump around right             1,173 / 1,173      1,173 / 1,173
    length (24–48 actions)        3,919 / 3,920      3,920 / 3,920
    add jump                      7,706 / 7,706      7,706 / 7,706
    all six                       26,223 / 26,227    26,227 / 26,227

The four errors without F56 had one source, traced in the mind's own list of readings. A part of the command that was
itself a lived command, whose answer is made of like parts ("walk twice and walk"), had no summary of its own, and so
took what an action phrase takes. The mind was wrong exactly where it had lived more. F56 gives such a whole its form
(which wholes stand in it, not in what order their replies come).

**Invariance.** With every SCAN word and action renamed to random letters (three bijections), the mind gives the same
answer to every test command on add-jump, around-right and length (100 of 100 each), also with shuffled lessons. Of
1,776 string constants in the mind's code, none of the mechanism holds a word of a test world.

**Standing.** SCAN has been solved before by systems built for it, for example by meta sequence-to-sequence learning
(Lake, 2019), program synthesis (Nye et al., 2020), neural-symbolic stack machines (Chen et al., 2020) and learned
analytical expressions (Liu et al., 2020). What differs here is that the same evidence core that reads documents,
chats and studies lessons answers the files, with no component written for SCAN. A claim beyond that would need the
comparison set out in the notes.

### 6.4 COGS: 0, and why

On COGS the mind answers none. The causes were measured:

- 85 of 100 asked sentences are two or more words away from every sentence the mind lived, and a frame generalises
  one place. With only short sentences, where 98 of 100 are one word away, it answers 68.
- A word that goes with another word, position indices, and answers longer than the window are not built.

They are the next items of the queue.

## 7. Real text, a race against the clock, cost, study and school

### 7.1 Recall of what was read once

A mind that lived only the author's documents (about five minutes on one core) is given the first two words of 200
random sentences:

- it says a read sentence beginning with them in 150 (90 before F26 and F27; 145 with F28);
- it says the rest of that very sentence in 89 to 90;
- it says the right one in 60 of the 61 sentences whose two words begin only that sentence.

Transformers trained on the same documents with the same units say a valid sentence in:

- 0 (1 million parameters, one pass);
- 10 (1 million parameters, 18 epochs);
- 41 (7.4 million parameters, 50 minutes).

With a one-word cue the 7.4M Transformer says a valid sentence more often (51 against 44, 47 with F28): the mind
stays silent where a word begins several sentences. Asked in the form the teacher dialogues used ("X nedir ?"), a
mind raised on them and the documents says 119 of 200 held-out answers exactly (59 before F26 to F28; 130 with F41).

### 7.2 The race

The same machine and the same documents, scored as in 7.1: exact is the rest of that sentence; valid is the rest of
some sentence beginning with the two words. The mind lives the folder once on one thread. The Transformer (7.3
million parameters, d = 256, 4 layers) is trained from random weights against the clock on one thread, with other
runs beside it on the machine (a clean four-thread run was planned and not made), and asked at each mark:

    learner                                processor s to learn    exact   valid   silent   word F1
    the mind, speaking configuration       317.5 (one reading)       90     146      31     0.519
    the whole mind (no silent claims)      326.1 (one reading)       89     151      34     0.524
    Transformer, one pass                   85.3                      0       0      70     0.001
      2 minutes                            118.4                      0       6     124     0.007
      5 minutes                            296.0                      0       8      10     0.052
      10 minutes                           586.6                      0      11       8     0.092
      20 minutes                          1180.4                      0      11       1     0.104
      40 minutes                          2093.3                      1      39       6     0.094
      60 minutes (35 epochs)              2898.7                      1      47       2     0.102

After an hour the Transformer says the rest of a read sentence in 47 of 200, the mind after one reading in 146 to
151. The Transformer answers in a few hundredths of a second (0.036 s median). The whole mind took 3.96 s at the
median in this race; F74 later brings it to 0.64 s with the same answers. A larger Transformer, a GPU or longer
training would reach further. The Transformer was given a learning-rate schedule chosen for its size in an earlier
duel, and nothing tuned to these cues.

### 7.3 Cost, growth and speed with nothing changed

Every speed change below was required to leave the words of every answer and the store digests identical:

- the core is 2 to 3.8 times faster per moment;
- memory per tie went from 239 to 69 bytes (a 3-million-tie mind: 777 → 223 MB in memory, 930 → 377 MB on disk);
- the generalisations of a moment are ranked lazily (13,479 rankings compared, 0 different), so a moment of a
  structured life no longer grows with what was lived (a life of 61,346 moments: 330 → 114 s);
- the readings of the whole mind are shared where they are provably the same (six answers 136.1 → 79.7 s);
- the open window (F70: recall of 200 cues 2,746 → 744 s, every score the same);
- recall or reason (F74: median answer 2.70 → 0.64 s; the same words in 199 of 200).

Measured growth: on natural text ties grow linearly (12.3 a moment). A tie costs 68 bytes of knowledge and about 460
bytes with Python's indices, so billions of moments do not fit in working memory in the present form. 73% of a grown
mind's ties were written only in the moment they were born, which a long-term store on disk could hold; that store
is not built.

What runs without writing can run in parallel exactly: four processes give word-for-word the same answers 4.8 times
sooner, the mind untouched. Copies that live different things and are added into one were measured and eliminated
(Section 10).

### 7.4 The study (etüt)

A study session as a teacher holds one (`etut.py`):

1. Teach a lesson: every sentence and dialogue lived once.
2. Ask its dialogues in a shuffled order. A right answer is confirmed. A wrong or missing one is taken back exactly
   (the moments of the attempt retracted) and the right answer is shown.
3. Ask again, round after round.

The examination lines of a lesson are never taught; their answers follow from what the lesson told. The 24 lessons
are public knowledge written for this: mathematics, physics, chemistry, biology, anatomy, diseases, engineering,
coding, the solar system, the stages of society, machine learning, the mind itself, and twelve advanced ones.

- **The largest mind** (raised on the author's documents and everyday dialogues: 694,512 moments, 4.8 million ties)
  answered every lesson dialogue right in the first round: 284 of 284. Asked again after all 24 lessons, 284 of 284.
- **The examination lines:** 8 of 41 right after each lesson, 11 of 41 after all of them.
- **A small mind** (25,546 moments before the lessons) did need the retraction. In coding it answered 20, 20, then 19
  of 22 over three rounds of retraction and showing: retraction leaves the mind as it was before the attempt, so a
  wrong answer that came from what it had lived before comes again.

What the mind is told, it keeps, word for word, across a large life. What it is not told it rarely reaches: the
examination lines ask for a sentence assembled from facts told apart. That diagnosis is where the grounded model of
Section 9 begins.

### 7.5 The school examination

Seven subjects are taught as a teacher gives them, and only what was never asked is examined (`t_school.py`).
Right / asked:

    subject      one whole mind, every subject, one seed       a whole mind for each subject, three seeds
    biology      recall 8/8, deduction 12/12                    recall 24/24, deduction 36/36
    physics      recall 6/6, deduction 9/9                      recall 18/18, deduction 27/27
    chemistry    recall 10/10, a new pair 6/6,                  recall 30/30, a new pair 18/18,
                 an element seen only alone 18/18               an element seen only alone 54/54
    mathematics  recall 9/9, plus one 8/10, what follows 3/10,  recall 27/27, plus one 29/30, what follows 17/30,
                 a sum never added 0/8                          a sum never added 0/24
    coding       recall 7/7, a new name 6/6, a new pair 6/6     recall 21/21, a new name 18/18, a new pair 16/18
    music        recall 3/3, three notes up 0/2                 recall 9/9, three notes up 0/6
    drawing      a new combination 2/2,                         a new combination 6/6,
                 a shape seen only alone 1/3                    a shape seen only alone 3/9

Recall, deduction from facts read apart, and composition of parts never shown together are answered. A relation
along a lived sequence is reached only in part. "Three notes up" is never reached. A shape seen only alone, drawn
after another, comes with one move too many. A sum never added is answered by neither, as expected, since nothing
lived says it. The one mind holding seven subjects answered recall, deduction and composition as the separate minds
did, on the one seed measured. Not done: the one mind over three seeds, and the Transformer on the same lessons.

## 8. The mind examines and extends itself

These mechanisms (F46 to F69) were asked for in the author's words: the mind must find out what it does not know,
trace why it was wrong, restructure what was involved, try its own ways of learning, and form hypotheses. Measured
over five seeds:

- **Fault tracing (F49).** Every way of reading (a frame's claim by its kind, a continuation, a form, similar
  situations) keeps a record of what its claims came to, read back with the core's rule.
  - In frames whose members merely share a value: wrong answers 66 → 29, and 25 of 40 → 0 on things never asked.
  - Eight misleading frames in a row cost 5, 4, 2, 2, 1, 0, 0, 0 wrong answers (120 → 14).
  - A frame found wanting is read again among the members that share another lived place with the thing asked: 40
    of 40 things never asked answered right after eight rounds, where the mind without it is wrong on all 40.
- **Finding out and asking itself (F50, F51).** A word that rests only on what others of its kind share is held
  back. The mind looks the thing up in the turn it was asked in and answers when what it sent for has come: 80 of 80
  whole answers, against 9 without. Told one sentence about the first of a chain of seven things and left alone for
  40 silent moments, it asked itself 35 questions, looked 45 times and answered all 30 things right afterwards (5 of
  30 without).
- **Frames of frames and the asked place (F53 to F55).** A mind that has mastered two kinds of question answers a new
  kind the first time it meets it: 40 of 40, from 0.
- **Trials and built relations (F57, F58).** At a context that goes on from outside, the mind reads its decision
  again once for each of its ways with that one way changed, and takes a way into use one change at a time. Where it
  fails, a chain of its own relations that would have named the answer is a relation to try.
  - In a curriculum of eight kinds of question: 2 of 320 before; with trials 200 of 200, and a kind never asked 40
    of 40.
  - With built relations, what is read only of the thing's class and of the class's order: 0 → 40 of 40 each. A
    setting its builders would have made by hand answers 8 of those 120.
- **Conjectures (F59).** A statement nobody said, carried by a kind of link whose record decided for it, is thought
  over in a silent moment and kept with its source until what is told later settles it.
  - In three fields with words of their own: 60 of 60 in the field where things were told, and 80 of 80 and 80 of
    80 in the two fields where nothing was told of any thing.
  - 227 conjectures, all holding. The counts are the same with every word replaced by a random token.
- **Together, nothing set by hand.** One mind reads three fields and is asked, in kinds of question never asked
  there, what was told of no thing: 0 of 220 as it was, 220 of 220 with fault tracing, trials and conjectures
  together.

**The whole mind (F61 to F69).** At first, all of these together did worse than their parts in eight worlds. Each
case was traced by leaving one mechanism out, for every mechanism, and five faults of putting things together were
built for (F61 to F65). Of 38 measured lines, the whole mind then equals or betters, in 31, the configuration each
line was measured with. F66 to F69 then brought the lines worse than their own configurations from seven to three.
The cost is 2 to 8 times a moment, nearly all of it in silent claims (learning about its own ways while it only
watches), which the large-text configuration keeps off.

## 9. Grounded meaning: what is perceived and what is said

### 9.1 The diagnosis

In a small grounded world, people do things to things while a teacher speaks about it, in Turkish or English:

- 10 kinds of things, each with 15 opaque features;
- 6 people;
- 8 actions, each with 6 opaque action features, and each participant with 8 participation features (who starts
  it, who moves, who is turned to, who is touched, who changes, who goes along);
- pointing.

The splits are fixed by the seed before any learner lives the world:

- orders never shown, combining a person, a verb and a kind;
- a colour never said with a kind;
- a colour never seen at all;
- "X nedir ?" never asked;
- who-does-what questions about combinations held out.

Today's mind answers orders never shown by keeping the order's words and changing the end of the verb (144 of 158 in
Turkish). It answers almost nothing that needs what is seen: naming a thing 0 of 23, who does what 1 of 72. A
Transformer given the scene's facets as tokens learns naming (21 of 23) and part of the situations (29 of 72).

### 9.2 The ground (`zihin/zemin.py`, `Config.grounded`)

Everything below is evidence in the same store, read by the same rule, written in the moments of the mind, and so
taken back exactly.

- **Perception** binds features to tracked things, events and each participant's part. A value on a dimension where
  another was expected is evidence against that one.
- **Reference.** A word refers to one of the things talked about, to the event, or to nothing:
  - what is talked about is the participants of the event going on or asked for, else what is pointed at (joint
    attention), else everything in view;
  - the assignment is a soft one-to-one balancing of likelihood ratios (Sinkhorn's balancing; Cuturi, 2013), each
    value weighted by how rare it is in talk;
  - the word then names the features of its referent by its share (cross-situational learning with the referent
    first; Siskind, 1996; Fazly et al., 2010; mutual exclusivity, Markman & Wachtel, 1988).
- **Constructions.** An utterance's form becomes a construction (Goldberg, 1995; Tomasello, 2003) whose places are
  those of participants, modifiers, the action and fixed words:
  - a place holds the participation profile of what filled it, so a role is a profile, never a name (Dowty, 1991);
  - the words lived in each place are kept per verb (verb islands);
  - the time relation of the words to the event is evidence (before it began, or while it went on; Reichenbach,
    1947);
  - what the answerer said back is learned per turn.
- **Answering from the meaning.** The answer tells the event going on, or the one an order asks for, with the
  construction that answered such a question. Its places are filled from the event by their profiles, and each word
  is chosen by how well it names what fills the place. A proper name is found by the thing's identity; a word that
  firmly expects what the thing is not is not said.

### 9.3 Measured over three lives in two languages

Each question is answered three ways: by the ground alone (silent where it does not understand), by the core of
the same mind, and by both (the ground, else the core). For orders never shown, the answer is given in two settings.
In one, the order is heard with only the things in view and answered from what it asked for; the event is never
shown. In the other, the order is heard, then done, then answered while the event goes on. Right answers, word for
word, summed over seeds 0, 1, 2:

    learner                                  orders    orders   X nedir   colour never   naming,       who does
                                             never     seen     (facts)   said with      unseen        what
                                             shown                        the kind       colour
    the ground, Turkish                      128/158   158/158  0/15      31/45          16/23         48/72
    the core of the same mind, Turkish       144/158   158/158  0/15       8/45           0/23          1/72
    both, Turkish                            151/158   158/158  0/15      31/45          16/23         48/72
    the ground, English                      152/158   158/158  0/15      25/45          19/23         54/72
    the core of the same mind, English       126/158   158/158  0/15       0/45           0/23          1/72
    Transformer with the scene, Turkish       96/158   158/158  0/15       0/45          21/23         29/72
    Transformer with the scene, English      145/158   158/158  0/15       0/45          23/23         42/72

- **Situations and colour.** The ground says who does what from what is seen (48 and 54 of 72, against 29 and 42),
  and a colour never said with a kind (31 and 25 of 45, against 0).
- **Naming.** The Transformer with the scene is better at naming a thing of a colour never seen.
- **Orders, English.** The ground says the description of an order never shown from the meaning alone in 152 of 158
  cases (the core, keeping the order's words, 126). All six misses call a house a school.
- **Orders, Turkish.** All 30 misses of the ground are orders using a word form it never heard: the two verbs taking
  that form were never shown with that kind. The core copies such a form (144), and both together answer 151.
- **The two settings** give the same number of right answers in every one of the six lives.
- **The core is undisturbed.** Its answers are as in the baseline (144 of 158 in Turkish in both), so the ground's
  ties, in the same store, do not change what it reads.
- **Cost.** Living the world takes 90 to 134 s of one core; answering every question three ways 179 to 302 s.

### 9.4 What fails, and what was tried and not kept

**What fails.**

- In some lives, where namings were parsed with the deictic word as head, the answer's place is filled with it
  ("bu bir bu", "this is a this").
- "School" is said for a house.
- A question word lived in that place of a question is kept as a participant's word.
- A ball being looked at is called "elmaya" / "the apple".
- Orders with a colour are mostly silent.
- "X nedir ?" (telling) is not addressed.

The ball case was traced. A dimension is weighted by how rare the thing's own value is, so a word that firmly expects
a rare value the thing lacks ("elmaya" expects a stem; having no stem is common) is told almost nothing against.

**Tried and not kept** (each measured on seed 0):

1. *The pointwise mutual information against what is met in talk, with the one rule's e as the prediction.* A word
   less sure than talk of the common values "expects" the rare ones more than talk does, so every word fit a horse.
2. *The same with talk as the rule's prior* (c f + (1 − c) P(value)). The alignment became sharp ("ali ata bakıyor":
   shares 0.98, 0.99, 0.98, against 0.67, 0.70, 0.46). But in "bu bir elma" the words heard with every pointed thing
   took the thing and the pointing, and the kind word was left with nothing to learn from.
3. *Mutual exclusivity value by value*, so that two words can speak of one thing: worse (the verbs went wrong).
4. *The information used only where a word is chosen*, with the form a place takes judged from the other places the
   word was lived in: mixed. The English ball was right; elsewhere a form was wrong, because the namings had been
   parsed into two constructions.

Telling which form a place takes needs the words of one meaning compared across the places they were lived in. That
is the next step.

## 10. Mechanisms measured and not kept

- **Sibling minds.** Copies of the mind that live different things at the same time and are added into one were
  measured. Adding is exact, and there is no loss after a shared beginning in generated worlds. On the author's
  documents, recall fell (85 / 148 to 75 / 95 exact / valid of 200, and still 80 / 122 when parting in 8 rounds).
  No loss is acceptable, so a life is lived in turn.
- **Kept readings.** Keeping every read-only reading across moments with its dependencies recorded is exact, and
  slower everywhere: small worlds by 30 to 55%, the big mind 59 to 115 s an answer against 30. Two thirds of what was
  kept no longer held.
- **Latching (F71).** Following the thought that holds while listening: every small world the same. On running text
  a fifth of the readings were not made, but the cost did not fall, and one exact and one valid answer were lost.
- **Exemplars (F73).** A frame's vote read through T members only: 1.86 times faster, but 8 of 200 answers changed
  (3 better, 5 worse).
- **The reach of a continuation (F75).** A continuation's record kept apart for each length of the lived match: the
  study's small mind said 10 of 41 untaught lines instead of 4. But in a small world the new ways, each one voice in
  every vote, failed and made every reading believed less, and the answer was not begun.
- **Word pieces from birth.** A mind that lived each word with its root and suffixes from the repository's
  tokenizer grew about five times as many ties per moment and ran out of memory; pieces are opt-in.
- **The ground's alternatives** of Section 9.4.

Each record and its code is kept in the repository (`eliminated_*/` folders, the notes).

## 11. Discussion

**What the evidence supports.** Keeping every belief as evidence with its origin, and reading it with one rule,
gives:

- learning at every moment without a training phase, and without damage to what was known;
- exact retraction of what one experience taught;
- an account of where every word of an answer came from (`!neden`);
- compositional generalisation wherever an answer is made of the replies its parts were lived with;
- with working memory, generalisation that is not bounded by the window.

Several abilities that are usually trained into a network appear here as mechanisms that are switched on, measured
and, where they do harm, put away. They include doubting a way of reading that misled, looking something up before
asserting it, asking oneself, trying one's own ways of reading, and conjecturing.

**Limits.**

- The worlds are small and generated, and most results are sums over 3 to 5 seeds.
- The Transformer baselines are small, trained on one processor, and the gap may close with far more data.
- A frame generalises one place, which is the first reason for COGS's 0.
- The cost of the whole mind still grows with what it holds, and memory grows linearly with lived data.
- Most mechanisms were written by the builders after tracing a failure. F57 and F58 are the first steps toward the
  mind finding its own ways.
- The grounded world gives its features as opaque codes; it is not raw perception.
- None of this is peer reviewed.

**Not claimed.** Consciousness, general intelligence or understanding in a human sense are not claimed; none of them
can be measured for any machine today. Novelty beyond what is measured here is not claimed either.

## 12. Reproducibility and availability

**Code.** Code: https://github.com/myylogic/cevahir-ai, tag `research-2026.11.24-v7`, archived on Zenodo (concept DOI
10.5281/zenodo.22864660). Run `python -m zihin` for the program, `pytest tests/evolution` for the tests.

**Records.** Every step has its notes, scripts and result rows:

- `research/living_mind_2026_11_14/` and `research/living_mind_2026_11_15/`: the core, the first worlds and duels,
  recall, relations;
- `research/living_mind_2026_11_22/`: sections 1 to 43, SCAN and COGS, the whole mind, the race, the study, the
  school;
- `research/living_mind_2026_11_23/`: the grounded model (MODEL.md), the world, the baseline and the ground.

**The open mind.** `minds/cevahir-open.zihin.gz` is the program's whole mind raised from nothing on public material
only: the 24 lessons, the seven school subjects and the first dialogue files, studied with `etut.py`. It holds
19,040 moments and 132,229 ties. It answered 433 of the 440 lesson dialogues in the first round, 440 after the rounds,
and 6 of 41 examination lines never taught. Every word it holds is a word of those files. Its recipe, sources, digest
and study record are in `minds/README.md`; `python -m zihin --load minds/cevahir-open.zihin` unpacks it on first
load.

**Data.** The author's documents are private and are not released; only counts and times are published. SCAN and
COGS are public data sets.

**Withdrawn documents.** The Turkish documents of earlier releases (the research reports, the Turkish technical book,
the Turkish theory and guide) were withdrawn from the repository before this release. Their full texts remain in
release v6 (DOI 10.5281/zenodo.24279940). The theory is in English in `docs/ZIHIN_THEORY.md`.

## References

- Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C. A., Jia, H., Travers, A., Zhang, B., Lie, D., & Papernot, N.
  (2021). Machine unlearning. *IEEE Symposium on Security and Privacy*.
- Cao, Y., & Yang, J. (2015). Towards making systems forget with machine unlearning. *IEEE Symposium on Security and
  Privacy*.
- Chen, X., Liang, C., Yu, A. W., Song, D., & Zhou, D. (2020). Compositional generalization via neural-symbolic stack
  machines. *NeurIPS*.
- Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of optimal transport. *NeurIPS*.
- Dowty, D. (1991). Thematic proto-roles and argument selection. *Language*, 67(3).
- Fazly, A., Alishahi, A., & Stevenson, S. (2010). A probabilistic computational model of cross-situational word
  learning. *Cognitive Science*, 34(6).
- French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*, 3(4).
- Goldberg, A. E. (1995). *Constructions: A Construction Grammar Approach to Argument Structure*. University of
  Chicago Press.
- Harris, Z. S. (1955). From phoneme to morpheme. *Language*, 31(2).
- Kim, N., & Linzen, T. (2020). COGS: a compositional generalization challenge based on semantic interpretation.
  *EMNLP*.
- Lake, B. M. (2019). Compositional generalization through meta sequence-to-sequence learning. *NeurIPS*.
- Lake, B. M., & Baroni, M. (2018). Generalization without systematicity: on the compositional skills of
  sequence-to-sequence recurrent networks. *ICML*.
- Liu, Q., An, S., Lou, J.-G., Chen, B., Lin, Z., Gao, Y., Zhou, B., Zheng, N., & Zhang, D. (2020). Compositional
  generalization by learning analytical expressions. *NeurIPS*.
- Markman, E. M., & Wachtel, G. F. (1988). Children's use of mutual exclusivity to constrain the meanings of words.
  *Cognitive Psychology*, 20(2).
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: the sequential learning
  problem. *Psychology of Learning and Motivation*, 24.
- Mill, J. S. (1843). *A System of Logic*.
- Nye, M., Solar-Lezama, A., Tenenbaum, J., & Lake, B. M. (2020). Learning compositional rules via neural program
  synthesis. *NeurIPS*.
- Reichenbach, H. (1947). *Elements of Symbolic Logic*. Macmillan.
- Siskind, J. M. (1996). A computational study of cross-situational techniques for learning word-to-meaning
  mappings. *Cognition*, 61(1–2).
- Tomasello, M. (2003). *Constructing a Language: A Usage-Based Theory of Language Acquisition*. Harvard University
  Press.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
  Attention is all you need. *NeurIPS*.
- Wang, P. (2013). *Non-Axiomatic Logic: A Model of Intelligent Reasoning*. World Scientific.
