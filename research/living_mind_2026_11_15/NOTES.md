# Living mind, continued (2026-11-15)

This folder continues `research/living_mind_2026_11_14/`. That folder was archived in release v5
(DOI 10.5281/zenodo.24235648, tag research-2026.11.14-v5) and is kept as evidence. New work goes here. The F-numbers
continue from that NOTES.md (F7-F21).

One process note: the F16-F21 entries were appended to the 2026_11_14 NOTES.md after the v5 release. Nothing earlier
in that file changed, and the archived v5 copy is not affected. From F22 on, notes, scripts and results go in this
folder.

## F22. Sleep: dreaming, and sorting which dreams are true (2026-11-15)

The owner asked for a mind that dreams while it sleeps, imagines in its dreams, and sorts out which of its
imaginings are actually true. In core terms this is three pieces, all built from parts that already existed.

1. **Dreams.** The mind dreams about the last topic and about the words it uses but holds with little conviction
   (the highest drive, F17). Each dream is a thought (F20 inner voice) carried over to a counterpart found by
   structural analogy, when there is one. It is lived as imagined evidence, like any thought.

2. **What a dream claims.** A dream claims what a lived sequence would teach between words: each word is followed by
   the next (the lag-1 EXPECT ties). Lag-2 word ties carry no evidence in the speaking configuration, because longer
   reach is held by composed units, so they are not counted.

   When the dream is dreamt, the real evidence on each claim is recorded. Real evidence is every class except
   imagined and the mind's own acts, weighted by learned trust. The claims nothing real had touched are the dream's
   new part, what it imagined rather than recalled. A dream with no new claim is reported as "I recalled what I
   knew".

3. **Sorting**, at every sleep. It has two parts.
   - Every imagined amount not yet judged is compared with the real evidence on the same tie. A positive imagined
     claim agrees in proportion to the real frequency f and differs in proportion to 1 − f. A negative imagined claim
     works the other way round. Both are scaled by the conviction of the real evidence: the likelihood of the
     imagined data under the real record.
   - The agreement and the difference are evidence on the mind's trust in its own imagination. This uses the same
     source tie and the same C bookkeeping as `learn.judge`, so nothing is judged twice. Imagined evidence is never
     deleted. Ties with no real evidence stay open as hypotheses.
   - Each kept dream gets a verdict by the core's own measure. A new claim holds when the expectation that real
     evidence gives it is above one half, and fails when below. The dream is true when all its new claims hold,
     false when any fails, and open otherwise. An earlier dream is reported once, when life has judged it.

**Why sorting has to happen in sleep.** `judge` runs only when reference evidence (sensed or acted) lands on a tie.
In conversation the partner's words are told, so judging imagination against what it was told happens nowhere else.

**How a dream gets refuted.** Dreams are tested through expectations. After the mind dreams "güneş ateşi söndürür",
it expects "ateşi" after "güneş" (imagined evidence, weighted by trust). F14 records a failure only where something
was expected. So when life says "güneş evi ısıtır", the expectation fails and real negative evidence lands on the
dreamt claim. The dream itself creates the expectation that life can refute.

**Demonstration** (`results/seventh_dreams.txt`; public F20 text read twice, two nights):

- On the first night it dreamt "bitkileri pişirir", a new claim that plants are cooked.
- Between the nights it was told "yağmur bitkileri sular" and three other plain sentences. On the second night:
  "yanlış çıktı: bitkileri pişirir".
- The dream "söndürür ve bitkileri sular" was reported as recalled: every claim in it had been read.
- Its trust in its own imagination went 0.58 -> 0.61 -> 0.63.

**Tests** (`tests/evolution/test_zihin_sleep.py`, 4 tests):

- a recalled sentence has no new claim;
- life backs one dream ("yağmur ateşi söndürür" told), contradicts another ("güneş evi ısıtır" after "güneş ateşi
  söndürür"), and leaves a third open;
- sorting judges each imagined amount once and deletes nothing;
- a night keeps its dreams, counts itself, reports a judged dream once, and the dreams survive saving and loading.

**Where it lives:**

- `zihin/dusunce.py`: `sleep`, `sort_dreams`, `remember_dream`, `verdict`.
- `!uyu [n]` in the session.
- The life clock sleeps once after 10 minutes of silence.
- `mind.dreams` keeps the last 500 dreams, and `mind.nights` counts the nights.

**Limits** (all open):

- Claims are word-to-next-word ties. A dream that is false only as a whole ("güneş ... söndürür" through a composed
  unit) is judged only through its word links.
- The reality score when a dream is dreamt is neutral (0.5) on links nothing real has touched. It cannot tell a
  plausible construction from a weak leap; only later life can.
- Agreement is dominated by replays of known sequences. So the trust in imagination measures how faithful the
  inner simulator is; its creative leaps alone are judged only by the per-dream verdicts.
- Trust in imagination is one number per class, not per tie. A dream judged false keeps its imagined evidence;
  real evidence outweighs it where real evidence exists.
- Dreams are text only. Seen and heard things do not enter dreams yet.

## Hearing (2026-11-15)

A WAV adapter, the twin of seeing (`zihin/isitme.py`, `!dinle <wav>`; written by a helper agent and reviewed here).

- Facets: pitch by autocorrelation in 3-semitone bands, spectral centroid and roll-off, 8 mel-band shares, the two
  strongest bands, zero-crossing rate, voiced fraction, and duration.
- Every graded measure is quantised on two grids offset by half a band.
- Only frames within 30 dB of the loudest are described.
- The gist is a 64-bit difference hash of band energies over time.
- No core change was needed.

**Test** (`t_hearing.py`, `results/t_hearing.jsonl`): three synthetic voices (low, middle and high; middle and high
overlap in pitch), three named sounds per voice watched twice, then ten new sounds asked.

| seed | resemblance on | resemblance off |
|---|---|---|
| 0 | 93% | 0% |
| 1 | 97% | 0% |
| 2 | 93% | 0% |

Chance is 33%. The core's similarity formula, computed on the facet lists alone, gives the right voice 0.80-0.83
of the vote weight. For the vision drawings the right kind gets 0.73-0.81, so this voice set is about as easy as the
drawings, slightly easier. Harder voice sets were not tested.

**Unit tests** (`tests/evolution/test_zihin_hearing.py`, 13 tests): invariance to loudness, delay, sample width,
rate and channels, and the session wiring.

**Open:**

- `learn_facets` ties every facet of a moment to every item in it. An image and a sound perceived in one moment
  therefore cross-bind their facets.
- The gist names a recording, not a sound. For a steady tone the hash bits are near-ties, so the gist can change
  with volume while the facets stay the same.

## Two fixes found on the large mind (2026-11-15)

**Closing punctuation.** Minds raised before tones existed lived "?" as a word before the turn end. The session now
hears punctuation as a turn-end tone, so every question such a mind had watched became a sequence it never lived,
and it fell silent even on questions it had watched verbatim. A mind that delivered answers the evening before had
no turn ends at all, which is why the problem showed only on the newer minds.

The session now follows the convention the mind lived. Tones are used if the mind is new or has lived tone symbols;
otherwise the punctuation stays a word. A session never switches a mind between the two
(`tests/evolution/test_zihin_session_tone.py`).

**Travelling mind files.** A mind loads when its `.zihin.gz` name is given, and when it travelled without its
manifest. In that case it warns that the file's integrity was not checked (`tests/evolution/test_zihin_persist.py`).

## F23. Meeting a situation never lived as a whole (2026-11-15)

A question never watched as a whole got silence even when its words and ending had been lived in many watched
questions. The specific context had no evidence of its own. The category reading gave only a generic prior
(the most frequent first word of answers), and F16 never starts an utterance.

**Rule** (`read.situation_backoff`, used by `members_map` when nothing else expects anything above one half).

The present situation is someone's utterance up to its perceived end, plus whatever the mind has said since. It is
compared, part by part from the end, with the lived contexts that end with the same part and hold the turn end at
the same place.

- **Weight of a part:** 1 / (1 + the number of those contexts that hold it). A word nothing lived weighs most.
- **Similarity** of a lived context: the weight of the shared aligned parts over the weight of all parts either has.
  It must share a word of the utterance, not only punctuation and the end.
- **Votes:** each such context votes its own evidence on what came next, weighted exp(−F(1 − s)) as exemplars are
  (GCM, as F18).
- **Conviction:** the votes are held with lam · s · conviction(voters), where s is the voters' weighted similarity.

So a weak resemblance cannot make the mind speak. An utterance with an unknown topic word resembles nothing
strongly, which leaves the mind silent and lets its curiosity ask back. Because the situation includes what the mind
has said since the turn end, an answer it starts is carried on along the answers lived in the same situations.

**Tests** (`tests/evolution/test_zihin_situations.py`, toy dialogues):

- "ne yapıyorsun ?", never watched, after four watched "<greeting> , ne yapıyorsun ?" → "arkadaşlarla sinemaya
  gideceğim ." (the whole answer);
- "kuantum nedir ?" → "kuantum ?";
- a watched question keeps its own answer;
- with the setting off, there is no answer.

**On the large private mind** (122k moments; no private lines quoted here):

- Unwatched paraphrases of everyday questions were answered with whole answers lived in similar situations, or with
  a single fitting word.
- Unknown topics were still met with curiosity.
- Watched questions were unchanged.

**Limits.** With only two similar situations and competing partial matches (a toy mind), the votes do not reach the
decision threshold, so the mind stays silent. The F19 race also measures a leader against its strongest rival even
when that rival is below one half; that inconsistency is noted and left as it is, because the vision results depend
on it.

## L7, first form: not knowing, felt and said (2026-11-15)

The owner asked for this: "it must know how to say it does not know; silence is not the solution". He had also said
before: "the model will not just say 'I don't know'; the mind must know what not knowing means." So the measurement
is innate and its expression is learned.

**Measurement** (`Mind._unknowing`). The world sometimes calls on the mind: an outer event is perceived (the end of
someone's turn) while the mind is free to answer. If it then decides nothing, it keeps the amount it does not know:

> u = 1 − 2 · max(0, best − 1/2), where best is the highest expectation of any act.

If the next moment passes in silence with nothing said, the mind perceives that state as an inner event (channel
"ic", symbol "bilmemek", intensity u). The event is alone in its moment, because the silence ended the episode. It
never arises while the mind listens, watches others, or answers.

**Expression.** The session presents lessons (`kendilik.not_knowing_lessons`): questions about words nobody knows,
made of Turkish syllables. Each is told at the moment the mind has itself just felt that it does not know: "bilmiyorum
." or "bilmiyorum , öğrenmek isterim .". What follows the inner event is learned like any sequence. In conversation
the mind takes the same silent moment before answering when it has nothing at once, perceives its not knowing, and
says what it learned to say then. Curiosity (T4) then asks back the unknown word.

**Tests** (`tests/evolution/test_zihin_not_knowing.py`):

- not knowing is felt only when the mind is called on and has no answer;
- before the lessons it never says "bilmiyorum";
- after the lessons, "fotosentez ne demek ?" gets "bilmiyorum , öğrenmek isterim fotosentez ?", and known questions
  keep their answers.

**Limit.** Two well-supported but rival answers make the F19 race hold back, and the mind then says it does not
know although it has two candidates. Hesitation between answers and not knowing are not told apart yet.

## Online life (2026-11-15)

The owner said this is not a request-and-answer exchange: from the moment the session starts the mind lives, inputs
and outputs flow, and it is online. The session's `Life` replaces the earlier life clock.

- Typed lines arrive as events in the mind's own time, in order.
- What the mind says comes out word by word as it says it.
- Between lines it lives silent moments, one every two seconds, so its age grows while nobody types.
- After 10 s of silence it thinks; after 45 s it may speak first if its thought was real enough; after 10 min it
  sleeps and dreams. Each happens once per silence.

Tests: `tests/evolution/test_zihin_online.py`.

## Word pieces from the repository's own tokenizer (2026-11-15)

At the owner's request the pieces come from the repository's own tokenizer infrastructure (`tokenizer_management`):

- the morphological root, when it differs from the word;
- the suffixes the morphology splits off;
- the last syllable.

They accompany each word as facets on the channel "parca", so words built alike resemble each other (F18). For
example, "yapıyorsun" and "gidiyorsun" share "ek:sun", and "yapıyorsun" and "yaptım" share "kök:yap".

The 60k BPE vocabulary itself keeps frequent words whole, so its tokens add no structure for them; the syllabifier
and the morphology of the same package do.

**Who gets pieces.** A new mind is raised with them. A mind raised without them keeps living without them, because
F18 skips members with no shared facets and switching would silence an old mind's categories. A mind raised with
them keeps them.

**Toy comparison.** With pieces, "kuantum nedir ?" got "bilmiyorum , öğrenmek isterim kuantum ?". Without pieces the
F12 relation "the answer repeats the asked word" answered just "kuantum".

Tests: `tests/evolution/test_zihin_pieces.py`.

## Scaling: what grows with lived data, measured (2026-11-15)

The owner's question: the data will grow, so what must happen as it grows? These are measurements on the private
minds, with counts only and no content. The runs shared the machine with other jobs, so absolute times are indicative.

| mind | moments | ties | max lag-0 degree | top-100 mean lag-0 degree | watching (ms/moment) |
|---|---|---|---|---|---|
| first breath | 25.5k | 98k | 1288 | 76 | 4.7 |
| cevahir2 | 122k | 1.33M | 6050 | 667 | 7.5 |
| cevahir3 | 234k | 2.54M | 9175 | 1071 | 13.6 |

**What grows** (cevahir3):

- **Ties:** 10.9 per moment.
  - EXPECT lag 1: 38%.
  - EXPECT lag 0: 17.5%.
  - PART: 43%. These are structure only; they carry no evidence but each holds a full evidence row.
  - ABSTRACT: 1.5%.
- **Composed units:** one per moment (207k); 25% were lived once.
- **Evidence size:** 72% of ties hold less than 2 units of evidence, mostly one experience.
- **Memory:** 239 bytes per tie, of which 192 are evidence rows (4 tables × 6 provenance classes, float64).

**Two bottlenecks follow from this.**

1. **Time per moment grows with memory.** The lag-0 spreading in `_settle` reads every lag-0 tie of each atomic
   focus unit in every settling iteration. `_winners` then sorts all evoked candidates to keep F = 7. Frequent words
   gain partners as data grows, so the cost of a moment grows with the vocabulary. A brain-like moment should cost
   what is local to it, not what is stored.
   - Exact fixes in progress: top-F selection without a full sort, and cached tie-id tensors (a helper agent;
     results must keep the substance digest bit-identical).
   - The algorithmic fix, to be designed: spreading only along ties whose expectation can still matter.
2. **Memory per moment is about 2.6 kB.** Two million examples are about 50M moments, which is about 130 GB at
   today's density. The plan, in order of safety:
   - (a) structure-only ties (PART) get no evidence rows, about 43% fewer rows;
   - (b) evidence stored only for the provenance classes a tie actually has, since most ties have one or two of six;
   - (c) a disk-backed evidence store for minds larger than RAM;
   - (d) consolidation in sleep, from episodic to semantic memory. Instances whose information their frames
     already carry would hand their evidence to the frame. This needs care, because the F11 reading reads members'
     own evidence through instances.

Nothing of (a)-(d) is done yet.

**Done tonight for scaling.** The F23 situation reading no longer scans the whole memory at every turn. It calls up
lived situations through the words of the utterance, rarest first, with bounded effort.

## A finding on one-shot evidence (2026-11-15; not changed)

A tie that is new receives no negative evidence from the earlier moments when its source was lived without it,
because F14 records a failure only where something was expected. So one hearing of X followed by Y gives
e(X→Y) = 0.67, which is the same as a continuation of X lived three times with one failure (f = 0.75, more
conviction).

Seen in conversation: after a question was heard once, the inner voice was torn between what the mind had read
three times and the word the question had just put after the topic. The session works around this by calling
thoughts to mind as the words are heard, before the utterance is taken in.

The principled repair belongs in the core. A new tie would take into account how often its source was lived before
it existed (a base rate from `u_use` and `t_birth`). This changes learning for every mind, so it waits for a
measured study against the F11-F21 results.

**Design for that study** (not built):

- **(a) Implicit failures at birth.** When X→Y is born at a moment when X has already been lived n times, it starts
  with W− equal to the trust-weighted n·g. Y had never followed X before, or the tie would exist. This is the same
  frequency a full record would give, and F14 still governs later failures.
- **(b) A normalised reading of alternatives.** f(Y | X) = W+(X→Y) / Σ_Y' W+(X→Y'), so rivals share one mass instead
  of each holding its own.

(a) keeps the per-tie semantics and every reading as they are; (b) changes every reading. Start with (a) behind a
flag. Measure it on the T1-T3 worlds and on held-out chat; dreams and thinking should improve with it, since both
are hurt by the echo of the last hearing.

## T8 in conversation: think, then speak (2026-11-15)

The owner asked for this: "it should be able to think and speak". As the words of an utterance are heard, the ones
the mind is least sure of call thoughts to mind (`Session.associations`): the inner voice of F20 is kept as imagined
and scored for reality from lived evidence only. If the mind has no answer at once when its turn comes, it says the
first thought that says something (at least three words) and is real enough. "Real enough" is the act threshold,
reality ≥ 1/2 + θ/2T. Only otherwise does it feel that it does not know (L7).

The thoughts are called to mind before the utterance is taken in. A thought made after hearing echoed the question:
one hearing of "X Y" makes Y as strong a continuation of X as anything read about X (see the one-shot finding above).
Thinking before taking the utterance in avoids that echo.

**On the large private mind** (no content quoted): a question about a group of people that it had read about in a
report was answered with the report's own sentence about that group, which the mind recalled by thinking about the
word. An unknown word still met curiosity.

Tests: `tests/evolution/test_zihin_think_aloud.py`.

## The morning mind, measured (2026-11-15)

The mind given to the owner was raised by continuing cevahir2 (122k moments) with:

- the teacher questions from his documents (`ogretmen.tsv`);
- the lessons about itself and about not knowing;
- his Word documents;
- capped samples of the psychology dialogues, the Turkish datasets and the subject books;
- the five public first dialogues.

It reached 295k moments, 3.25M ties and 203k composed units, and it lived with closing punctuation as words (the
older convention, kept by the session). It is a private mind and is not in the repository.

**Held-out teacher questions** (60 random questions generated from his documents and never watched; aggregate only):

| questions | exact answer | mean token F1 | "bilmiyorum" | silent |
|---|---|---|---|---|
| 60 | 17% (10/60) | 0.38 | 23% (14/60) | 0 |
| 200 (later, with the faster core and exploration) | 18% (36/200) | 0.42 | 18% (36/200) | 0 |

**Chat** (no content quoted):

- greetings, its name, what it is, where it lives and its sleep habit are answered from what it was told;
- unknown topics get "bilmiyorum , öğrenmek isterim ." followed by the unknown word asked back;
- watched questions get their answers.

**Weak points:**

- A self question can be answered with the answer others gave in the dialogues it read (a city), and in a long
  session the same question can meet hesitation.
- Dreams are still shallow constructions.

## T8, second form: imagination that reasons by a shared stem (2026-11-15)

`dusunce.infer`. The mind thinks about X. For a word of that thought whose stem other lived words share, it thinks
about such a word W and carries what follows W over to X.

- Example: "kedi bir hayvandır" and "hayvanlar nefes alır ve büyür" give "kedi nefes alır ve büyür".
- Stems come from the repository's tokenizer infrastructure: the morphological root ("kök:") and the first two
  syllables of a longer word ("gövde:", added to the word pieces). The morphology alone over-stems or does not split
  ("kedi" → "ke"; "hayvandır" unsplit). Hay-van-dır and hay-van-lar share "hayvan" by syllables.
- The construction is scored for reality from lived evidence only. A claim nothing lived supports stays near one
  half, so it is a hypothesis. It is lived as imagined and kept as a dream, so life later judges it (F22).
- Dreams now reason first and fall back to analogy; `!dusun` shows the inference too.

Test (`tests/evolution/test_zihin_sleep.py`): on the toy above, the inference carries "nefes alır" over to "kedi"
with reality between 1/2 and 3/4, and a later "kedi nefes alır" does not refute it.

**Limit, seen on the large private mind.** A two-syllable stem shared by chance makes a false mate ("bilgi-sayar" and
"bilgi-ner"), and the carried thought is then nonsense ("kuantum , öğrenmek isterim ."). So inference is used only
in imagination (dreams and `!dusun`), where life judges it, and never in what the mind says as an answer.
A better stem relation (shared contexts as well as shared form) is open.

## T-code, first test: learning to write code by its consequences (2026-11-15)

The owner wants the mind to learn to write code. The first measurable step is `t_code.py`, with results in
`results/t_code.jsonl`.

**Setup.** There are six tiny function heads ("topla a b", "kare x", ...), each with three completions, one correct.
The mind first watches others write completions. The most common one it sees is wrong (3×), the correct one once and
the other wrong one once, so imitation alone writes wrong code. Then, in rounds, the head is said to it and it writes
a completion. The function is actually run against a test, and the result comes back as the value signal "iyi" or
"kötü" (F15/L5). Nothing about code is in the core.

**Results** (share of passing completions per round, mean of 3 seeds, 8 rounds):

| condition | round 1 | rounds 2-8 |
|---|---|---|
| with consequences | 0 | 0.11 |
| without (control) | 0 | 0 |

**Diagnosis.**

- Where the wrong completion's punishment leaves one alternative leading, the mind switches to the correct
  completion and keeps it (2 of 6 tasks in a seed).
- Where the remaining alternatives are equally supported, the F19 race lets nothing lead. The mind writes only the
  shared first token and then stays silent.
- Silence is punished as well, so it never tries the other completions: there is no exploration.

**Next, designed but not built.** Hesitation should end in a try, preferring the alternative whose consequences are
least known (information-seeking), with the tie broken deterministically. The same mechanism would let hesitation in
conversation end in an answer.

## Exploration: hesitation ends in a try, and code is learned from its tests (2026-11-15)

`act._explore`, `Config.explore` (on for speaking minds).

**When the silence is hesitation.** It is hesitation when the F19 race lets nothing lead but some alternative is well
supported, meaning it would have been chosen alone: 2(e − 1/2)T ≥ θ. Nothing well supported is not hesitation; it is
not knowing (L7).

**When a try is made.** Only where it can teach something:

- the mind has just been called on (the last thing in the episode is a perceived event, the turn end); or
- good or bad was already met on these paths, or in this very situation.

In the middle of its own utterance with nothing at stake it does not wander. An ungated first version did wander:
it babbled on in conversation, and held-out exact answers fell from 10 to 5 of 60.

**What is tried.**

- Among the well supported alternatives whose known consequences are not bad, the one whose consequences it knows
  least: the evidence on its path's value ties, where a path never taken knows nothing.
- If every well supported alternative has led to bad, and only then, anything else it has lived after this (e > 1/10).
- The try is lived like any act, so its consequence teaches.

**T-code with exploration** (`results/t_code.jsonl`; passing completions per round, mean of 3 seeds):

| condition | r1 | r2 | r3 | r4-r8 |
|---|---|---|---|---|
| consequences + exploration | 0 | 0.39 | 0.72 | **1.0** |
| exploration, no consequences | 0 | 0 | 0 | 0 |
| consequences, no exploration | 0 | 0.11 | 0.11 | 0.11 |

From imitating wrong code, the mind reaches passing code for all six tasks by round 4, learning only from the
result of running its own code. Each task follows the same course: wrong, the other wrong, correct, kept. Neither
the consequences nor the exploration does this alone.

**In conversation.** A self question that met hesitation at the start of the turn ("nerede yaşıyorsun ?") is now
answered from what it was told ("bir bilgisayarda yaşıyorum ."). Held-out teacher questions are unchanged: 17% exact,
F1 0.38, 23% not knowing.

Tests: `tests/evolution/test_zihin_explore.py`, including the code test at 4 rounds.

**Limit.** The tasks are tiny: three alternatives and a decisive second token. Writing new code means composing
sequences nobody showed it. That needs imagination and analogy (T8) working with consequences, and it is open.

## F24. Learning to learn: how far a voice can be believed, learned in one domain and carried to the next (2026-11-15)

**The question (T6 again).** The first T6 (`research/living_mind_2026_11_14/RESULTS.md`, section T6) was negative. A
mind that had lived nitelik learned iliski no faster than a fresh mind, and the same held the other way round. Only
the learned trust in the teller carried over. The question here: can something the mind learns about learning itself
carry over? The owner's rule applies: no hand-coded skill, the mind's own measured evidence decides, everything is
lived as experience and stays retractable.

**Where early learning fails** (diagnosis with the setting off; fresh minds, seeds 0 and 4, both domains; held-out
probes of `run.py`, white box on a clone):

| experiences | probes | answered right | class frame exists | class frame has the right value on top | class frame chosen |
|---|---|---|---|---|---|
| 30 | 12 | 0.25 | 1.00 | 0.33 | 0.25 |
| 60 | 22 | 0.36 | 1.00 | 0.59 | 0.23 |
| 90 | 32 | 0.38 | 1.00 | 0.59 | 0.31 |
| 120 | 42 | 0.57 | 1.00 | 0.79 | 0.43 |
| 180 | 48 | 0.69 | 1.00 | 0.88 | 0.58 |
| 300 | 56 | 0.71 | 1.00 | 0.93 | 0.70 |

The knowledge is there early: the frame of the true class exists in every probe. The F11 reading chooses it far less
often than it is right. In the early probes (up to 180 experiences) where another voice spoke, 53 of 92 times that
voice was one member speaking alone. Its "agreement" is 1.00 by construction, because one member always agrees with
itself. F11's score e(i→g) · conviction(members) · max f prefers such a voice to a class whose three members agree two
to one.

So early learning in a new domain is a problem of meaning: which voice to believe. That kind of knowledge does not
depend on any word, so it could be learned once and carried.

**Mechanism** (`Config.voices`, default off; acts on support "members"):

1. **The kind of a voice.** When the mind reads a context, every generalisation whose members have lived something in
   its slot is a voice (F11). The mind perceives two magnitudes of each voice:
   - n, how many members speak, in doubling bands 1, 2, 3–4, 5–8, 9–16, 17+;
   - f\*, how much they agree: the frequency of their top value, in quarters. One half, where a prediction starts to
     hold, is a band edge, and all agreeing is a band of its own.

   kind = (band(n), band(f\*)). A kind names no word, class, dimension or domain (`read.voice_kind`).
2. **Self-test by what comes next.** For the context read with the full trust, the mind notes the voice that spoke
   and every voice that could have spoken, each with its claim (top value v\*, frequency f\*). Then the context may
   live a value v for the first time: its tie to v had no evidence, so the prediction rested on the voices and not on
   the context's own past. Each claim is then judged by the core's measure of a holding prediction:

   > held ⇔ v\* = v and f\* > 1/2

   Each judged claim is one experience (gain g) on two ties of the mind's own channel `ses`, class sensed:
   (kind → holds) and (every voice → holds). The evidence goes through the ordinary logged path, so it can be
   retracted exactly (`learn.judge_voices`).
3. **Credibility,** read general → specific, the nested form of `expect_map`:

   > e ← e(every voice → holds);  e ← c_kind · f_kind + (1 − c_kind) · e

   A kind never lived is worth what voices in general have been worth. With nothing lived about voices it is 1/2
   (`read.voice_credibility`).
4. **Choice.** F11's score of each voice is weighted by its credibility:

   > score = e(i→g) · conviction(members) · max f · credibility(kind)

   With no experience of voices every credibility is 1/2, and the choice is exactly F11's. Only the choice changes;
   the expectations of the chosen voice are F11's.
5. **Carrying.** The ties live on the mind's own channel, not on any domain's words. When the mind meets a new
   vocabulary, its first readings already weigh voices by what the first domain taught, and the new domain's outcomes
   add to the same ties.

**Two earlier versions, measured and discarded:**

- **v1: only the voice that spoke was judged; a kind never lived counted 1/2; agreement in tenths.** One life per
  domain (seed 0), mixed to worse: nitelik told-rule held-out 0.50 and 0.83 at 180 and 240 experiences, against 1.00
  and 1.00 with the setting off. In the learned tables voices failed 80–90% of the time, because most first-time facts
  are of free dimensions that nothing can predict. A kind never relied on (1/2) therefore beat every kind that had
  been tried.
- **v2: v1 with nested credibility and agreement in quarters,** still judging only the voice that spoke. Pilot, seeds
  0–1, both orders: fresh single-domain lives fell sharply (iliski seed 1: 0.17–0.50 at 60–300 experiences, against
  0.75–0.90 off; nitelik seed 1: 0.17 at 60 and 120, against 0.83 and 0.67). The remaining cause is selection bias:
  judging only the voice relied on lets the choice decide what is learned.

The version above (v3) judges every voice. All voices were present when the context was read, so outcomes that
nothing could have predicted burden every kind alike.

**Runs** (`t6_learning_to_learn.py`, records in `results/t6_learning_to_learn.jsonl`). The protocol of the first T6:
same worlds, schedules and probes, with checkpoints every 30 experiences instead of 60 (probes run on clones, so the
lives are the same). 10 seeds × 2 orders, so 20 lives per condition. Seeds 0–4 are the first T6's seeds; seeds 5–9
were added because a life has only 6–8 held-out probes.

- **Setting on:** own runs, all seeds.
- **Setting off:** own runs for seeds 5–9. For seeds 0–4 the first T6 records serve, since the off path is the same
  code. An own off run of seed 0 (nitelik → iliski) reproduces them: every T1, T2 and T3 row and every substance field
  of T5 is identical. Only the ledger's in-memory size differs, because log compaction was added (2026-11-14 21:10)
  after those records were made (20:26).
- At the coordinator's request the runs used at most two worker processes; a curriculum run shared the machine.

**Held-out in B** (mean of told-rule and induced-rule probes), mean over 10 seeds:

| order | setting | mind | 60 | 120 | 180 | 300 | 600 |
|---|---|---|---|---|---|---|---|
| nitelik → iliski | off | fresh | 0.40 | 0.65 | 0.68 | 0.74 | 0.86 |
| nitelik → iliski | off | experienced | 0.38 | 0.65 | 0.68 | 0.74 | 0.86 |
| nitelik → iliski | on | fresh | 0.33 | 0.69 | 0.70 | 0.76 | 0.82 |
| nitelik → iliski | on | experienced | 0.42 | 0.73 | 0.74 | 0.71 | 0.86 |
| iliski → nitelik | off | fresh | 0.53 | 0.72 | 0.80 | 0.89 | 0.97 |
| iliski → nitelik | off | experienced | 0.53 | 0.73 | 0.81 | 0.89 | 0.97 |
| iliski → nitelik | on | fresh | 0.51 | 0.73 | 0.72 | 0.89 | 0.93 |
| iliski → nitelik | on | experienced | 0.46 | 0.73 | 0.73 | 0.89 | 0.93 |

**Per seed**, early (mean of 60, 120, 180) / late (mean of 300, 600):

| order | seed | off fresh | off experienced | on fresh | on experienced | on: experienced − fresh, early |
|---|---|---|---|---|---|---|
| nitelik → iliski | 0 | 0.49 / 0.75 | 0.40 / 0.75 | 0.54 / 0.75 | 0.57 / 0.81 | +0.02 |
| nitelik → iliski | 1 | 0.67 / 0.81 | 0.67 / 0.81 | 0.79 / 0.91 | 0.74 / 0.81 | −0.05 |
| nitelik → iliski | 2 | 0.77 / 0.87 | 0.77 / 0.87 | 0.71 / 0.90 | 0.77 / 0.87 | +0.07 |
| nitelik → iliski | 3 | 0.57 / 0.72 | 0.57 / 0.72 | 0.44 / 0.72 | 0.53 / 0.75 | +0.09 |
| nitelik → iliski | 4 | 0.44 / 0.59 | 0.44 / 0.59 | 0.37 / 0.75 | 0.59 / 0.72 | +0.23 |
| nitelik → iliski | 5 | 0.54 / 0.88 | 0.54 / 0.88 | 0.62 / 0.72 | 0.62 / 0.56 | +0.00 |
| nitelik → iliski | 6 | 0.45 / 0.84 | 0.47 / 0.84 | 0.42 / 0.88 | 0.49 / 0.84 | +0.07 |
| nitelik → iliski | 7 | 0.54 / 0.88 | 0.54 / 0.88 | 0.60 / 0.78 | 0.63 / 0.78 | +0.03 |
| nitelik → iliski | 8 | 0.69 / 0.84 | 0.69 / 0.84 | 0.63 / 0.66 | 0.76 / 0.88 | +0.13 |
| nitelik → iliski | 9 | 0.58 / 0.78 | 0.58 / 0.78 | 0.58 / 0.84 | 0.58 / 0.84 | +0.00 |
| iliski → nitelik | 0 | 0.76 / 1.00 | 0.76 / 1.00 | 0.62 / 0.92 | 0.70 / 0.88 | +0.08 |
| iliski → nitelik | 1 | 0.75 / 0.86 | 0.75 / 0.86 | 0.69 / 0.90 | 0.75 / 0.90 | +0.06 |
| iliski → nitelik | 2 | 0.65 / 0.92 | 0.68 / 0.92 | 0.63 / 0.92 | 0.55 / 0.88 | −0.08 |
| iliski → nitelik | 3 | 0.72 / 0.87 | 0.72 / 0.87 | 0.56 / 0.87 | 0.56 / 0.87 | +0.00 |
| iliski → nitelik | 4 | 0.40 / 0.88 | 0.40 / 0.88 | 0.48 / 0.92 | 0.41 / 0.92 | −0.07 |
| iliski → nitelik | 5 | 0.79 / 0.96 | 0.79 / 0.96 | 0.67 / 0.96 | 0.64 / 0.96 | −0.03 |
| iliski → nitelik | 6 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | +0.00 |
| iliski → nitelik | 7 | 0.81 / 0.96 | 0.81 / 0.96 | 0.75 / 0.96 | 0.74 / 0.96 | −0.01 |
| iliski → nitelik | 8 | 0.53 / 0.96 | 0.57 / 0.96 | 0.61 / 0.96 | 0.58 / 0.96 | −0.03 |
| iliski → nitelik | 9 | 0.43 / 0.92 | 0.43 / 0.92 | 0.51 / 0.75 | 0.51 / 0.75 | +0.00 |

**Paired differences:** mean over lives ± standard error, (lives better / equal / worse). Early = checkpoints 60, 120,
180; late = 300, 600.

| comparison | B = iliski, early | B = nitelik, early | both, early | both, late |
|---|---|---|---|---|
| off: experienced − fresh (the first T6) | −0.006 ± 0.009 (1/8/1) | +0.007 ± 0.004 (2/8/0) | +0.000 ± 0.005 (3/16/1) | +0.000 ± 0.000 |
| on: experienced − fresh | **+0.058 ± 0.025 (7/2/1)** | −0.009 ± 0.016 (2/3/5) | +0.025 ± 0.016 (9/5/6) | −0.006 ± 0.016 (3/10/7) |
| fresh minds: on − off (one domain) | −0.005 ± 0.025 (4/1/5) | −0.031 ± 0.029 (3/1/6) | −0.018 ± 0.019 (7/2/11) | −0.011 ± 0.019 (7/8/5) |
| experienced with the setting − fresh without it | **+0.054 ± 0.017 (7/2/1)** | −0.041 ± 0.026 (3/2/5) | +0.006 ± 0.019 (10/4/6) | −0.017 ± 0.021 (7/8/5) |

Reading:

- **With the setting off, experience gives nothing,** as in the first T6.
- **With it on, the mind that lived nitelik learns iliski faster early:** +0.058 ± 0.025 held-out over a fresh mind
  with the same setting (7 lives better, 1 worse), and +0.054 ± 0.017 over the fresh mind of the first T6. The largest
  gains fall in the first 60–90 experiences (at 60: +0.33 in seed 4, +0.29 in seed 6, +0.20 in seed 3). What moved is a
  lesson about which voices to believe, not content. It is the first transfer beyond trust in the teller measured in
  these worlds.
- **The other order shows none.** A mind that lived iliski learns nitelik no faster (−0.009 ± 0.016). The setting also
  costs fresh nitelik minds a little early (−0.031 ± 0.029), mostly at 180 experiences.
- **Late in life nothing carries** (−0.006 ± 0.016): by then a fresh mind with the setting has learned the same table
  from its own outcomes.

**Single-domain check.** The fresh minds are single-domain lives; 10 seeds per domain.

| domain | setting | held-out 60 | 120 | 180 | 300 | 600 | members 600 | exceptions 600 | T1 c (1 hearing) | T1 c (≥ 4) | T2 classes | T5 early facts recalled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nitelik | off | 0.53 | 0.72 | 0.80 | 0.89 | 0.97 | 1.00 | 1.00 | 0.401 | 0.757 | 1.00 | 1.00 |
| nitelik | on | 0.51 | 0.73 | 0.72 | 0.89 | 0.93 | 1.00 | 1.00 | 0.401 | 0.757 | 1.00 | 1.00 |
| iliski | off | 0.40 | 0.65 | 0.68 | 0.74 | 0.86 | 1.00 | 1.00 | 0.409 | 0.800 | 1.00 | 1.00 |
| iliski | on | 0.33 | 0.69 | 0.70 | 0.76 | 0.82 | 1.00 | 1.00 | 0.409 | 0.800 | 1.00 | 1.00 |

- T1, T2, members, exceptions and T5 (early facts recalled, their evidence bit-identical, the ledger exact) are the
  same with and without the setting. It changes only which voice speaks, never what is learned from the world.
- Held-out, which is the reading, is a little lower with the setting at the end of a life (nitelik 0.97 → 0.93, iliski
  0.86 → 0.82; paired late −0.011 ± 0.019) and in nitelik around 180 experiences (0.80 → 0.72).
- These differences are within the noise of 6–8 probes per life, but the sign at 600 is the same in both domains. So
  the setting does not improve single-domain learning, and it stays off by default; earlier results are unchanged.

**Is A kept?** A's evidence is untouched by B with the setting on, as before: early facts bit-identical 20/20,
members and exceptions unchanged 20/20. A's held-out answers, however, are no longer fixed: they changed in 7 of 20
lives (4 up, 3 down; mean 0.876 → 0.887). B's outcomes add to the same kind ties, and that changes which voice speaks
for A's instances. So learning about learning in B changes how A is read. Nothing in A is lost: retracting B's moments
would restore the old reading exactly.

**What was learned about voices** (pooled over the 10 A lives of each domain; a selection of the kinds with at least
100 outcomes in one domain; every kind with agreement below one half, at any numerosity, held 0 times):

| numerosity | agreement | held / judged, A = nitelik | share | held / judged, A = iliski | share |
|---|---|---|---|---|---|
| every voice | — | 1076 / 11120 | 0.10 | 484 / 5568 | 0.09 |
| 1 | all | 151 / 2092 | 0.07 | 154 / 1721 | 0.09 |
| 2 | 1/2–3/4 | 42 / 1858 | 0.02 | 29 / 713 | 0.04 |
| 2 | all | 68 / 151 | 0.45 | 35 / 104 | 0.34 |
| 3–4 | 1/4–1/2 | 0 / 2440 | 0.00 | 0 / 875 | 0.00 |
| 3–4 | 1/2–3/4 | 133 / 454 | 0.29 | 95 / 323 | 0.29 |
| 5–8 | 1/2–3/4 | 226 / 537 | 0.42 | 47 / 182 | 0.26 |
| 5–8 | 3/4–1 | 87 / 158 | 0.55 | 3 / 23 | 0.13 |
| 9–16 | 1/2–3/4 | 116 / 367 | 0.32 | 58 / 158 | 0.37 |
| 17+ | 1/2–3/4 | 121 / 229 | 0.53 | — | — |

The lesson holds no word, and much of it is the same in both domains:

- one member speaking alone holds 7–9% of the time;
- two members who all agree hold 34–45% of the time;
- a voice whose majority is below one half never holds, by the core's measure.

Where the two domains differ is the voices of many members with a weak majority.

**Diagnosis.**

1. **Why the gain is early and small.** A fresh mind with the setting learns the same table quickly from its own
   outcomes, because every voice is judged on every first-time fact: 1–2 judged claims per experience over a life
   (11120 in 6000 nitelik experiences, 5568 in 6000 iliski experiences), more early on, when most facts are new. By
   60–120 experiences it has caught up. A head start can only show in the first few dozen experiences, where a life has
   few held-out probes (the held-out entity must already have been told its class).
2. **Why it is asymmetric.** The kinds are too coarse to tell a class from a voice of the majority of everything. How
   often such a voice is right depends on the world:
   - In nitelik (3 classes, 4 values, so classes often share a rule value) the majority of all entities is right about
     half the time (17+ members, agreement 1/2–3/4: 0.53).
   - In iliski (4 classes, 5 values) it is right less often (5–8 members: 0.26).

   Credibility learned in nitelik prefers the class's kind of voice in iliski, and that helps. In nitelik the opposite
   happens around 180 experiences: credibility learned from iliski, and nitelik's own early tables, let a many-member
   majority voice beat the class voice. What tells a class from the majority is which category it is. That is tied to
   the domain's own units and cannot be carried.
3. **What would go further (open).** Domain-free descriptions of a voice that separate a class from the majority:
   - how much more its members agree than all entities do (agreement beyond the base rate);
   - whether its category has been homogeneous on other dimensions too (category coherence). This is computed from the
     category's own members and names no word.

**Verdict: mixed.** The mind learns something about learning from its own outcomes; the lesson holds no word, and it
carries.

- In one order (nitelik → iliski) it makes the new domain faster early: +0.05 to +0.06 held-out, about two to three
  standard errors, 7 of 10 lives better and 1 worse.
- In the other order it does not.
- It does not improve single-domain learning (slightly lower held-out at the end of a life).

It stays behind `Config.voices`, off by default.

**Tests** (`tests/evolution/test_zihin_t6.py`, 8 tests):

- the kind bands;
- credibility is read general → specific;
- with no experience of voices the choice is exactly F11's;
- every voice that could speak is judged once, on the first lived value only;
- the voice that speaks maximises the F11 score times credibility;
- an outcome retracts exactly;
- the lesson holds no word: it carries into a new vocabulary (a fresh mind reads 1/2) and grows there;
- off by default.

## Speed, bit-identical (2026-11-15)

A helper agent's work, reviewed and merged here. Commits on the `speed` branch: 3d9f7ba, 79361e4, 2bf1a89, 06cbad8.

**What changed, at a glance:**

- The focus is not re-ranked when what is present already fills it, and the F winners are picked without sorting
  every candidate (`heapq.nsmallest`, exact because every key ends in a unique tuple).
- Lag-0 spreading reads each unit once per settling.
- Tie-id tensors are kept between moments, invalidated on undo.
- One-tie evidence and judging run in plain numbers with the same rounding.
- `judge` reads the four tables once and passes over classes with nothing open.
- Records are written only for what was acquired, where nothing else reads them.

**Exactness.**

- The substance digest is unchanged on the small public run, on the two large private minds after 60 dialogues, and
  in 17 configurations, including mid-run undo and retraction.
- After merging into this branch, the digest of the small run is also identical before and after the merge (548
  moments, d8ba0375…).

**Speed** (ms per moment, watching, no profiler):

| run | before | after |
|---|---|---|
| small run | 2.9 | 1.4-1.5 |
| cevahir2 (1.33M ties) | 6.7-7.7 | 2.6-2.7 |
| cevahir3 (2.54M ties) | 12.6-13.1 | 3.2-3.5 |

Growth with memory is now small: 2.6 → 3.3 ms between 1.3M and 2.5M ties, where it was 7 → 12.6. The time side of
the scaling bottleneck above is largely solved. The memory side (bytes per tie, structure-only rows) is not.

What remains per moment on the large mind: `learn_follow` (~0.8 ms) and `learn_together` (~0.7 ms), mostly gathers
over long lag-0 lists that are bound by memory speed; `_feel` (~0.25 ms).

## Word pieces from birth: a memory explosion (2026-11-15)

To compare with the morning mind, a mind was raised from scratch on the same curriculum with today's conventions:
tones, and the repository tokenizer's pieces as facets from birth.

**What happened.**

- It ran at 2.2-4.8 ms per moment.
- It grew about five times as many ties per moment: 6.45M ties at 116k moments, about 55 per moment, against 11 for
  the mind raised without pieces. Its core tensors were already 1.5 GB at that point.
- At 219k moments its process held 13.4 GB, and it was killed for lack of memory while saving.

**Consequence.** Pieces are now opt-in for a new mind (`--parca cevahir`). A mind raised with pieces keeps them, and
a mind raised without them keeps living without them. Where the ties come from (facet binding for every item, joining
categories by resemblance through very common pieces such as a frequent suffix or last syllable, or both) is open.
Pieces become a default again only once their tie cost is measured and bounded.

## F25. One-shot evidence: a tie born with the failures a full record would hold (2026-11-15)

**The finding** (section "A finding on one-shot evidence" above). F14 records a failure only on a tie that exists, so
a tie X→Y that is new has no negative evidence from the earlier moments when X was lived without Y. One hearing of
"X Y" gives e(X→Y) = 0.67 (trust 1/2), as strong as a continuation of X lived three times with one failure. Thinking
about X then echoes what was heard last.

**Mechanism (a)** (`Config.born_failures`, default off; `learn.born_failure`, called from `learn_follow` where
`S.tie(unit, v, EXPECT, 1)` creates the tie):

- **The count.** Before this moment v never followed X while X learned what comes next, or the tie would exist. Every
  earlier moment in which X learned and v's channel reported something else gave that something positive evidence on
  X's tie to it, g · a_X · intensity, in the class of that moment. So X's other lag-1 ties to targets on v's channel
  hold, in their positive evidence, exactly those moments, per class. They are read before this moment's positive
  evidence is added (a second target on the same channel in the same moment does not count).
  - Not `u_use`: it counts the moments X was in the focus, and a word stays in the focus for several moments (the
    carry decays by 1 − 1/F) while with F13 only what was the present learns. Test: in "a x y", "b x y", "c x z"
    with F13 the new x→z holds exactly 2 failures while u_use(x) > 3.
  - Not `t_birth`: the birth is now; what is needed is how often X learned before it, which the rivals hold.
  - Not the rivals' `t_use`: a tie is used only while it is expected (e ≥ the F14 floor, 1/14 in the speaking
    configuration), so the count would stop for a word with many successors.
- **The amount and the class.** Per class p: W−[p](X→v) = Σ over X's lag-1 ties on v's channel of W+[p]. This is the
  full record with each failure of weight g · a_X, which is the design's "trust-weighted n·g": the reading weighs
  each class by its trust, and the frequency of the new tie is the frequency v has had after X. F14 still governs
  every later failure (weighted by e, from the floor).
- **Marked as judged** in the classes whose trust is learned (C−[p] += the same amount; sensed and acted are direct
  experience and never judged). They claim nothing new: each such moment is already an open claim of its source on
  the rival's tie and is judged there. Unmarked, the next direct experience on the new tie (an own act's echo, a
  sensed event) would judge the whole mass a second time as a disagreement with the teller. That mass grows with how
  often X was lived, so one act on a frequent word could cost the trust in told or imitated thousands of units. Test:
  a later sensed "x z" leaves the told source tie and the trust bit-identical to the setting off.
- **Only targets on sense and act channels** (the world, one event per moment); feelings and context channels keep
  F14 alone.
- **The induce record** of the newborn is [0, gained, 0, failures, class, class holding most failures]. Nothing
  before the moment, so F9 and F24 still see a first time; with `general_miss` off (as written) the generalisation
  receives the capped failures like any other.
- **Exactly retractable.** The amounts go through `S.add1`, logged with the moment. Retracting the birth moment removes
  the positive evidence, the failures and their marks, and the tie is removed when nothing else supports it; every
  tensor then equals the sum of the other moments' increments. Undo restores the digest (tests).
- **Off is the old code path.** The small run's digest is unchanged (548 moments, d8ba0375…). All 10 archived
  members lives of `run.py` are reproduced row for row (65/65 rows each; the ledger's byte size is left out, since
  the speed work writes fewer ledger records). T-code reproduces `results/t_code.jsonl` 9/9.

**Variant (b)** (`Config.rivals`, default off; `read.rival_mass`, reading only). For a target on a sense or act channel
the negative mass is the positive mass of the source's other targets on that channel, so f(v | X) = W+(X→v) / Σ_v'
W+(X→v') and the conviction is that of everything X was followed by there. Learning is unchanged; every reading of
lag-1 own evidence changes.

Tests: `tests/evolution/test_zihin_born_failures.py` (10).

**1. The echo toy** (`f25_echo.py`, `results/f25_echo.jsonl`). A fresh mind lives "kedi uyur" three times, then
"kedi koşar" once; then it thinks about "kedi" (`dusunce.think`). In the question toy it lives "kedi süt içer ." three
times, then hears "kedi ne yer ?" once. Expectations are read from the atom's own ties (trust 1/2 in told and
imitated):

| toy | configuration | setting | e(kedi→Y) | e(kedi→Z) | the inner voice after "kedi" |
|---|---|---|---|---|---|
| told: uyur ×3, koşar ×1 | speaking | off | 0.690 | 0.667 | nothing (hesitation) |
| | speaking | a | 0.690 | **0.333** | **uyur** (reality 0.69) |
| | speaking | b | 0.667 | 0.333 | uyur (0.67) |
| question: süt ×3, "ne" ×1 | speaking | off | 0.690 | 0.667 | nothing |
| | speaking | a | 0.690 | **0.333** | **süt içer** (0.76) |
| | speaking | b | 0.667 | 0.333 | süt içer (0.75) |
| read: uyur ×3, koşar ×1 | members | off | 0.690 | 0.667 | **koşar**: the echo (0.67) |
| | members | a | 0.690 | **0.333** | **uyur** (0.69) |
| | members | b | 0.667 | 0.333 | uyur (0.67) |

The old tie is untouched by (a); only the one-shot tie falls, to the frequency 1/4 it has had. In the members
configuration acts speak only from imitated evidence, so the told toys give no thought there in any setting (and the
question toy reads 0.52 against 0.56, off, without F13). (a)+(b) gives the same rows as (b).

**2. The worlds of the first lives** (`f25_born_failures.py`, `results/f25_worlds.jsonl`; `run.py` imported
unchanged, variant members, 600 experiences, seeds 0–4, both domains, T9 included). Held-out (mean of told-rule and
induced-rule probes):

| domain | setting | 60 | 120 | 180 | 300 | 600 |
|---|---|---|---|---|---|---|
| nitelik | off | 0.533 | 0.653 | 0.780 | 0.873 | 0.933 |
| nitelik | a | 0.533 | 0.653 | 0.780 | 0.873 | 0.933 |
| iliski | off | 0.413 | 0.676 | 0.680 | 0.684 | 0.812 |
| iliski | a | 0.413 | 0.676 | 0.680 | 0.684 | 0.812 |

Paired a − off: 0.000 at every checkpoint in 10 of 10 lives (identical, not just equal on average). Also unchanged
at 600: members and exceptions 1.00, early facts 1.00 (their evidence bit-identical), categories 1.00, exact, T9 exact
retraction in 10/10 lives (own evidence 0, untouched ties identical, state = ledger of the other moments, own e =
the counterfactual life's). T1 by hearings: c at one hearing 0.399 in both settings (nitelik), 0.410 → 0.407
(iliski); trust in told 0.665 → 0.664 (nitelik), 0.695 → 0.687 (iliski).

Why the worlds barely see it: F11 reads the members' positive evidence only, the held-out instance has no own tie,
and a fact's context ("entity dimension") is new at its first hearing, so it has no rivals. What does change is a
context lived again with another value, which is a lie and its corrections:

| correction (40 per row, 5 seeds × 2 domains) | off: true e | off: false e | a: true e | a: false e | a: true > false |
|---|---|---|---|---|---|
| sensed, first | 0.750 | 0.491 | 0.565 | 0.490 | 1.00 |
| sensed, second | 0.833 | 0.409 | 0.682 | 0.408 | 1.00 |
| told, first | 0.697 | 0.547 | **0.500** | 0.547 | **0.00** |
| told, second | 0.787 | 0.467 | 0.611 | 0.467 | 1.00 |

With (a) one told correction against one told lie leaves the truth at exactly 1/2: the context was lived twice, once
with each value. The lie stays at 0.547, because F14 weights its later failure by its expectation while the
correction was born with the full count. So the first-born value is favoured, where the old rule favoured the
newest. The positive evidence of the lie stays bit-identical in every case (refinement without deletion, 160/160).

**2b. The same worlds with the speaking configuration** (`--speaking`: F13, the F14 floor, F16, F18, F19, F23 and
exploration; off/a only, since these lives differ from the archived ones). The speaking mind does poorly in these
worlds in both settings. It also speaks into the stream, which these lives were not made for; that was not analysed
further. Held-out:

| domain | setting | 60 | 120 | 180 | 300 | 600 |
|---|---|---|---|---|---|---|
| nitelik | off | 0.500 | 0.458 | 0.525 | 0.480 | 0.550 |
| nitelik | a | 0.433 | 0.533 | 0.595 | 0.577 | 0.583 |
| iliski | off | 0.058 | 0.065 | 0.056 | 0.037 | 0.037 |
| iliski | a | 0.033 | 0.033 | 0.043 | 0.037 | 0.037 |

Paired a − off: nitelik −0.067 ± 0.163 at 60, then +0.075 ± 0.050, +0.070 ± 0.066, +0.097 ± 0.086 and +0.033 ± 0.020
(2 lives better, the others equal or one worse). iliski −0.025 to −0.033 up to 180, then 0.

At 600, a against off:

- members 0.83 vs 0.78 (nitelik) and 0.86 vs 0.79 (iliski);
- early facts recalled 0.58 vs 0.48 and 0.38 vs 0.30;
- exceptions kept 0.62 vs 0.66 and 0.87 vs 1.00;
- categories 284 vs 342 and 194 vs 231;
- T9 exact in 10/10 lives.

Mixed and small against the variation between seeds.

**3. T-code** (`f25_t_code.py`, `results/f25_t_code.jsonl`, `t_code.run` with its configuration plus the setting;
3 seeds, 8 rounds; off reproduces `results/t_code.jsonl` 9/9). Share of passing completions:

| condition | setting | round 1 | rounds 2–8 | per round |
|---|---|---|---|---|
| consequences, exploration | off | 0 | 0.873 | 0 .39 .72 1 1 1 1 1 |
| consequences, exploration | a | 0 | **0.952** | 0 .67 1 1 1 1 1 1 |
| no consequences (control) | off / a | 0 | 0 / 0 | |
| consequences, no exploration | off | 0 | 0.111 | 0 .11 .11 … |
| consequences, no exploration | a | 0 | **0** | 0 0 0 … |

Better with exploration: round 2 rises from 0.39 to 0.67, and from round 3 on every task is solved in every seed.
Worse without exploration: the two tasks one seed had solved (where the punished wrong completion left one
alternative leading) are lost. The likely reason, not traced here, is that the completions seen once now start low,
so none leads after the punishment and the mind stays silent.

**4. Held-out teacher questions, fresh mind** (`f25_teacher.py`, `results/f25_teacher.jsonl`; aggregates only). A
fresh mind with the speaking configuration watches the 1358 pairs of `ogretmen.tsv` and the lessons about itself
(22165 moments), then answers 200 questions of `ogretmen.tsv.heldout` (fixed random order, seed 0) with
`Session.reply`:

| setting | exact | token F1 | "bilmiyorum" | silent | asks a word back | other answers |
|---|---|---|---|---|---|---|
| off | 0.005 (1) | 0.229 | 0.150 | 0.120 | 0.670 | 12 |
| a | 0.000 | 0.151 | **0.995** | 0 | 0 | 1 |
| b | 0.000 | 0.151 | 1.000 | 0 | 0 | 0 |
| a+b | 0.000 | 0.151 | 1.000 | 0 | 0 | 0 |

Paired over the questions (a − off): F1 −0.077 ± 0.008 (lower in 142 of 200), "bilmiyorum" +0.845 ± 0.026. A fresh
mind knows almost none of these answers, and the settings change how it says so. A white-box look at the end of each
question's turn (read only, recorded in the same runs as `turn_end`) shows why:

- **Off: hesitation.** The best act has e 0.609 on average. In 83% of the questions it is above the line at which it
  would be said alone (1/2 + θ/2T = 0.583), but a rival close behind keeps its advantage too small for the race
  (1.9 candidates on average, never two above the line). L7 calls that hesitation, not not knowing: not knowing is
  kept in only 17% of the questions. The curiosity reflex then asks a word of the question back (2 words).
- **(a): not knowing.** The first answer words lived after the same question endings carry their true frequency
  (the best act is at 0.572), so no act reaches the line in any of the 200 questions. Not knowing is felt in 200 of
  200, and the mind says "bilmiyorum , öğrenmek isterim ." and then asks the word back.
- **F1.** Most of off's F1 is the asked-back word (85% of its F1 mass; those answers score 0.29). The same word at the
  end of the longer "bilmiyorum …" answer scores 0.15, so the F1 drop is largely the length of the non-answer.
- **Lost attempts.** What is really lost is the 12 attempted answers (1 exact, F1 0.20), which fall to 1.

**4b. The same, after also reading the documents the questions were made from** (`--documents`, the owner's 11 private
documents, 13290 sentences; 195332 moments lived before answering, 2.34M ties; the order of the morning mind: teacher
pairs, lessons, documents):

| setting | exact | token F1 | "bilmiyorum" | silent | attempted answers (F1, words) |
|---|---|---|---|---|---|
| off | 0.130 (26) | 0.300 | 0.170 | 0.010 | 164 (0.366, 14.8) |
| a | 0.140 (28) | 0.257 | 0.025 | **0.210** | 153 (0.336, **5.6**) |
| b | 0.120 (24) | 0.232 | 0.015 | 0.245 | 148 (0.314, 5.9) |

Paired (a − off): exact +0.010 ± 0.019 (better in 8 questions, worse in 6), F1 −0.043 ± 0.021, "bilmiyorum" −0.145 ±
0.028, silent +0.200 ± 0.029, length −8.4 ± 0.7 words. With the documents the mind attempts most questions in both
settings. Off it talks on along one-shot continuations (14.8 words; F1 rewards the extra words with recall). With (a)
it stops where a continuation was lived only once against a common rival, so answers are shorter, about as often
exact, and lower in F1. One question in five gets silence instead of "bilmiyorum" or an attempt. (b): exact −0.010 ±
0.016, F1 −0.068 ± 0.018, silent +0.235.

**5. Variant (b) in short.**

- Echo toy: the same inner voice as (a).
- Fresh teacher mind: the same as (a) (all "bilmiyorum").
- With the documents: exact 0.12 against off 0.13 and (a) 0.14; F1 0.232, lower than (a); silent 24.5%.
- Worlds (members, 5 seeds × 2 domains): held-out identical to off at every checkpoint in 10 of 10 lives, and T1, T2,
  T3, T5 and T9 unchanged. After corrections the truth reads like (a) (0.591 and 0.588; one told correction leaves it
  at 1/2), while the lie reads lower than with (a) (0.409 and 0.412 against 0.477 and 0.479), because its rivals'
  evidence counts against it at once.

(b) changes every reading and costs time on every read (41–45 s per nitelik life against 30–33 s). Where it matters
for speaking (the teacher questions after the documents) it is below (a).

**Reading.**

- (a) does what it was built to do. A continuation seen once after a source lived many times starts at the frequency
  it has had. The old ties are untouched, and retraction stays exact in the ledger sense.
- The worlds' held-out generalisation is identical life for life, but the lives are not unchanged. The independent
  verifier found differences in:
  - tie counts;
  - trust in told, lower in 6 of 6 lives;
  - T1 c1;
  - the T9 increments;
  - the told first correction, which now loses (true > false 1.00 → 0.75).
- The trust claim holds only for the born amounts themselves. In lived minds trust moves through the changed
  behaviour (told 0.823 → 0.771 in the fresh teacher run), so the teacher comparison measures a changed trust as well.
- The echo is gone. Thinking about X now says what followed X three times, not what followed it last. In the members
  configuration the old rule said the echo outright.
- In conversation the effect depends on what the mind has lived.
  - A mind that knows little turns hesitation into "I do not know". That is honest, but it ends its few attempts.
  - A mind that has read the sources answers exactly about as often, in shorter answers, with lower F1, and is
    silent in one question in five.
- Two costs are structural:
  - **Two laws.** The born failures count every earlier moment fully, while F14 weights every later failure by the
    tie's expectation. So the value born first is favoured: a lie told once beats one told correction.
  - **A reading tuned on inflated ties.** The speaking reading's margins were set while one-shot ties stood at
    0.6–0.7: the race's lead, L7's line at which an answer would be said alone, and F16's e > 1/2 gate for going on.

**Recommendation: not the speaking default yet.** Keep the setting.

The echo fix is real and the worlds lose nothing. But in conversation it trades long answers for silence, and it
reverses which value wins after a single correction. Before it can be the default:

1. **One law for born and later failures.** Either the born count is weighted the way F14 weights failures (each
   earlier moment by the expectation the tie would have had then, from 1/2, down to the floor), or later failures
   count fully. An estimate, not measured: with F14's weighting
   - the born failures are bounded near 1/(2·floor) − 1 ≈ 6 (trust-weighted) at the speaking floor;
   - the toy's e(kedi→koşar) would be about 0.47 instead of 0.33, still below uyur's 0.69;
   - one told correction would end slightly above the lie (about 0.58 against 0.55) instead of at 1/2.
2. **Re-tune the speaking reading with the setting on** (the race margin, L7's alone line, F16's gate), then repeat 4
   and 4b.
3. **Measure on a mind raised with it.** The setting changes learning, so the morning mind cannot simply be switched
   over.
4. **A cheaper partial path, not built:** use (b), which is a reading only, for the inner voice alone (thinking and
   dreams). It gives (a)'s inner voice in every toy without changing learning or what the mind says aloud.

## Memory: a tie's evidence in two numbers, bit-identical (2026-11-15)

The memory side of the scaling bottleneck above, done exactly. Counts only, measured on cevahir3b (295k moments,
3.25M ties).

**A correction first.** The plan said PART ties carry no evidence. They do: every recognition of a composed unit
gives its PART ties evidence (form.py), and all 1.45M PART ties of cevahir3b hold some. So plan (a), no rows for
structure-only ties, would save nothing here. What the measurement shows instead:

- 99.6% of ties hold W evidence of exactly one provenance class (3,239,447 of 3,251,213; 11,762 hold two, one
  holds three).
- Judged amounts (the C tables) are on 10 ties.

**What changed** (`zihin/store.py`; every reader goes through it):

- A tie keeps its class (`t_cls`, one byte) and two values (`t_ev`: w+, w- of that class).
- A tie that receives a second class or a judged amount moves, once, to a full row of the four tables (`ex`,
  through `t_row`).
- Readings assemble exactly the values the four tables hold: `w`, `w2`, `w4`, `w1`, `cell` and `column`, plus
  `t_w`, which is now an assembled read-only copy for old scripts and tests. Assembly selects values and does no
  arithmetic, so every downstream operation sees the same bits.
- The digest is computed over the same values in tie order, streamed, so it is the old digest exactly.
- `undo_last` gives the moment's rows back and restores every value. Correction by the independent verifier: one
  stored array is not always as before. A tie born earlier with no evidence that gets its first class in the undone
  moment keeps that class byte (`t_cls`) after the undo, and retracting all of a tie's evidence also leaves its class
  set. Values, rows and the digest are restored. The effect is memory only: later evidence of another class gives
  such a tie a full row (197 B) it did not need.
- A mind saved with the full tables is converted on load, and the manifest's digest still matches.

**Exactness.**

| check | old | new |
|---|---|---|
| small run (548 moments) | d8ba0375… | d8ba0375… |
| 17 configurations with mid-run undo and retraction | 17 digests | the same 17 |
| fixed run, 999 moments with dreams, sorting, undo, retraction: per-moment acts, predictions, focus, feelings | file | identical file |
| cevahir3b, 60 watched dialogues + 3 replies: per-moment outputs, answers, digest | 1570adbc…, 5f34bbdc…, f29fea1a… | the same |

Persistence was also checked:

- A mind saved by the old code loads with the new code, passes the manifest check and lives on identically,
  including undo and the retraction of moments it had logged before the save.
- A mind saved by the new code loads again with the same digest.

**Memory** (cevahir3b, core tensors):

| | before | after |
|---|---|---|
| core | 776.7 MB | 223.1 MB |
| bytes per tie | 238.9 | 68.6 |
| process peak, 60 dialogues | 4.91 GB | 4.01 GB |

The two million examples of the owner's question would be about 37 GB of core at this density, where they were
about 130 GB.

**Speed.** Watching cevahir3b took 4.39 ms per moment, against 4.09 ms for the old code in the same session (+7%,
single runs on a shared machine). The independent verifier measured more: watching 2.94 → 3.87 ms per moment
(+32%) in sequential runs and +13% when run side by side; replies +16% to +28%. Take 13-32% slower as the cost. The small run takes 1.0 s, against 0.8-0.9 s. Reading a tie now assembles its
row instead of indexing it.

**What remains, by size per tie (68.6 B):**

- The other tie columns, 44 B: t_src and t_dst are int64, as are t_last and t_birth. int32 would save 16 B, with
  a guard at 2^31.
- Python caches beside the tensors: `tie_index` (a dict of 4-tuples) and the `out`/`inc` lists. These are
  probably the next memory problem, but their size was not measured. The 4.0 GB process peak also includes
  converting the old file on load.
- A disk-backed store and consolidation in sleep, (c) and (d) above. Not started.

**Verification notes on F25** (independent verifier; every re-run measurement reproduced exactly):

- **Speaking worlds.** The sentence on them is wrong. At 120, 180 and 300 the standard error is smaller than the
  difference (+0.075±0.050, +0.070±0.066, +0.097±0.086), and 600 (+0.033±0.020) is 1.6 SE, not an exception. No
  checkpoint reaches 2 SE, so the conclusion stands.
- **A cross-tie path that T9 does not see.** Born failures are computed from a rival's W+ and logged in the later
  birth moment. Retracting the moment that taught the rival therefore leaves its mass on ties born after it (e 0.500
  against 0.667 counterfactual in the toy). In the members worlds the gap is small.
- **Own speech.** In conversation about 80% of born-failure mass comes from the mind's own speech (class acted,
  trust 1), yet the core treats its own echo as saying nothing about the world when computing relief. This may lock
  in a first habit. Before (a) is tried again as a default: leave own acts out of born failures (or weigh them down),
  use one law for born and later failures, and repeat the teacher tests over more than one life or data order. The
  ± values there are over questions within one deterministic life, not over lives.

**Merged** into feat/zihin with both settings off, as recommended. Off is bit-identical: the digest, the run.py rows
and the full-state digest are unchanged.

**Also fixed while merging.** The F24 voices channel was named "ses", the same name as the hearing facet channel, so
`Config(voices=True)` crashed in a session. It is now "sesler", with a session test.

## Scripted lessons removed from the core (2026-11-15)

**What was there.** `zihin/kendilik.py` held 14 question/answer lessons about the mind (its name, what it is, where it
lives, that it does not eat or sleep, ...) and a generator of "not knowing" lessons (questions about made-up words
built from a list of syllables and six question templates, answered "bilmiyorum ." or "bilmiyorum , öğrenmek
isterim ."). `Session.lessons_about_itself` and `Session.not_knowing_lesson` lived them as told experience, the second
right after the mind had felt that it did not know, and the command `!kendini_tani` called them. They were written by
the developer, not by the owner and not by the mind's own life.

**Decision.** The owner rejected them: the core must not hold sentences for the mind to say, and his own first data
already tells the mind about itself. They are removed: `zihin/kendilik.py` keeps only introspection from the mind's
own evidence (`self_report`, the ties of the self unit); the two Session methods and the command are gone; a test
pins that none of them comes back. The felt not-knowing state (L7) is a measured inner state and stays; what the mind
says when it feels it is now learned only from the people it lives with and from its data.

**What this touches.**

- Minds raised before lived these lessons several times in their curricula (`!kendini_tani` in the curricula of
  cevahir2 and of the morning mind cevahir3b), and the morning mind also watched the five toy dialogues of
  `research/living_mind_2026_11_14/data/first_dialogues.tsv` (written by the developer for the first chat test,
  including "adın ne / adım cevahir") three times in each of two curricula. Their self answers and their "bilmiyorum"
  come largely from these lessons. Without a ledger the lessons cannot be retracted from those minds; a mind raised
  without them is needed.
- Measurements that used the lessons keep their numbers as records of that configuration: the morning mind's
  held-out teacher questions (17% / 18% exact, "bilmiyorum" 23% / 18%), L7's "after lessons" result, and F25's
  teacher-question protocol (`f25_teacher.py` calls `lessons_about_itself(1)`, so it runs only at the commits before
  this change).
- The substance-digest gate used for every bit-identical change so far (`digest_run.py`, 548 moments,
  d8ba0375…) lived the lessons in 419 of its 548 moments. Its exactness verdicts stand (old and new code lived the
  same lessons), but the gate is replaced by one without any scripted content: public research texts and dialogues
  and two byte-protected Turkish reports (`docs/research/living_learning_2026_09_20/REPORT_TR.md`,
  `docs/research/living_learning_library_2026_09_21/REPORT_TR.md`, 600 sentences each), 6695 moments, digest
  cb3b9937… on both main 6e15b57 and this change.

**Next.** An audit of the whole chat program and core for any other scripted sentence, template, word list or coded
reflex (for example the curiosity reflex that asks back an unknown word with "?"), each to be removed or replaced by
a learned or mathematical mechanism; then a mind raised again on the owner's data only.

## Memory map and locality of a living mind (2026-11-15)

Measured without changing `zihin/` (`locality_profile.py`, `results/locality_profile.jsonl`; the memory map's scripts
stayed in a scratch area). A recorder wrapped the store's accessors; a plain run gives the same per-moment hash of
acts, focus and predictions and the same final digest on all three minds.

**Where the RAM goes.** On cevahir2 (1.33M ties) a loaded mind adds 756 MB: 570 B per tie. Python index objects are
87% of it (tie_index 259 MB, frame_first 195 MB, out 105 MB, inc 60 MB, tuple_index 32 MB); the core tensors are
90 MB (65 B per tie). On cevahir3b the indices are about 1.2–1.6 GB of about 2 GB. Importing torch alone costs
484 MB, more than a small mind. All seven saved indices are fully determined by the tensors (a rebuild reproduces
them exactly), and every out/inc list equals its segment of a numpy CSR in ascending tie-id order (0 mismatches over
251k lists; built in 0.27 s). **Correction:** the estimate above that two million examples would need about 37 GB
counted the core tensors only; at today's 570–650 B per tie in RAM it is about 300 GB.

**How much of the mind a moment uses** (60 watched dialogues, then 10 weather and 10 food questions):

| | cevahir_ilk_nefes (97k ties) | cevahir2 (1.33M) | cevahir3b (3.25M) |
|---|---|---|---|
| adjacency keys touched / lists read per moment | 28 / 11 | 29 / 11 | 29 / 11 |
| ties in the working set of a moment | 0.84% | 0.094% | 0.17% |
| of a dialogue | 3.8% | 1.0% | 1.1% |
| of the whole run (1276–1383 moments) | 23% | 5.0% | 5.5% |
| share of tie touches on the 1000 largest out lists | | 97.4% | 98.0% |

- A moment works in a small, fixed region whatever the size of the mind: the adjacency of its focus and context.
- The volume is in a few hub lists: the 1000 largest out lists hold 9–12% of the ties and receive 97–98% of tie
  touches.
- **Evidence predicts need, age does not.** 70% of ties carry one experience or less (W ≤ 1) and only 0.1–1.1% of
  them were touched in a run; ties with W > 10 were touched in about 30%. ln W predicts which ties a stretch of life
  needs (AUC 0.87–0.94); age does not (0.57–0.71), and beyond about 1000 moments recency is flat, so no single
  Anderson–Schooler power law holds here.
- In today's creation-order layout a moment's ties fall on 870 of 2,594 pages of cevahir2 and a session on 99.9% of
  the pages, so memory-mapping the present arrays would gain nothing. Grouped by source unit (CSR order, logical ids
  unchanged) it would be 11.3 pages per moment and 14.5% of the pages over a session. Static graph communities did
  not beat creation order; a sleep-time relayout by the day's first use gained 8–19% fewer page misses out of sample.
- With the 100 largest out lists and the 10 largest in lists pinned (about 19% of entries), an LRU cache of 1% of the
  rest misses 0.4% (cevahir2) and 1.7% (cevahir3b) of the bytes read.
- `learn_together`'s scan of lag-0 lists is about 90% of all adjacency reads, and 74–88% of the reads of a moment
  cannot change anything: for a one-class tie with frequency f below the failure floor θ = 1/14, the expectation stays
  below θ exactly while the trust in its class exceeds κ / ((1 − κ) W) with κ = (½ − θ) / (½ − f), and such a tie
  receives nothing unless its partner is present. This threshold, read from the one rule, is the exact counterpart of
  "only the part of the mind that can matter works". Not built yet.
- The reply path scans the whole unit table for the most frequent words (`dusunce.frequent_words`, 22–27% of reply
  time); sleep touches 47% of the mind. Both can be made local.

## T-meaning: no relation of meaning equivalence yet (2026-11-15)

A public, seeded world (`t_meaning.py`, `results/t_meaning_baseline.jsonl`; 6 seeds, 2 and 4 days, with and without
word pieces) lives greetings, paraphrased questions and content synonyms through the chat's Session in the speaking
configuration. Greetings: "merhaba" 16 a day, "selam" 6, "selamlar" 1–3 in a life, "selamün aleyküm" with its own reply
"aleyküm selam", "hello" in one bilingual exchange, "hi" only told once ("hi , merhaba demektir ."), "merhabalar" never.

- Every variant lived at least once is answered from its own lived context (merhaba, selam, selamlar, selamün aleyküm
  with "aleyküm selam", hello: 1.00). The variant only told as equivalent ("hi") gets 0/24; the never-met variant 0/12
  (1/1 only through spelling pieces).
- "selam / hello / merhaba ne demek ?": 0/24 each. "hi ne demek ?" is answered only by recalling the told sentence.
- Never-watched wordings of watched questions are answered 20% of the time (19% after twice the experience; 49–51%
  with word pieces). In 51% of the probes the right first word led at the end of the turn but stayed below the line
  at which it would be said (0.564–0.590 against 7/12 = 0.583), while the watched wording gives it 0.83–0.96.
- The slot machinery sees some synonymy (adın / ismin share 5.4–8.8 categories) but as one two-member category per
  sentence frame, and the members' reading counts members, so a one-member voice stays near 0.58–0.63.
- No mechanism reads what follows two expressions across a turn. Read relationally (a reply that repeats the item or
  an equivalent of it counts as the same reply), that signal separates greetings from other one-word utterances
  (merhaba / selam 0.80, merhaba / evet 0.00), where next-word overlap cannot (0.99 and 0.98).
- In the private minds greetings are barely lived (cevahir2: merhaba 6, selam 3; hello, hi, selamlar and selamün
  aleyküm never), and opposites look at least as alike as synonyms by every distributional measure (cevahir3b büyük /
  küçük: 22 shared categories, 36 shared slots, across-turn overlap 0.76; the synonym pairs 0 and 0). Substitutability
  alone cannot mean sameness; only consequences that differ can separate same from opposite.

What is missing: a graded equivalence relation with evidence from substitution and from relational consequences,
counted by the one rule; a reading that brings an equivalent's own evidence in; and an answer relation for "X ne
demek ?". The design is the next step.

## Speed race: the core against small neural networks (2026-11-15)

The owner asked that the core compete with neural networks, at least on speed. Before anything is changed for speed,
this race says where the core stands and what it has to beat. Script: `speed_race.py`. It imports
`baseline_transformer.py` and `run.py` of 2026_11_14 unchanged. Every row is in `results/speed_race.jsonl`.

- CPU only. The 4-core machine was shared with other agents; the 1-minute load average (0.8-4.2) is stored in each
  row. Systems are timed back to back in one process, 3 repetitions, medians.
- The stream is the baseline's: `World.make(seed, domain).schedule(600)`, 600 experiences of three symbols.
- Zihin lives an experience as 4 moments (3 symbols and a silent moment). A network is given `<s> a b c </s>` and
  learns its 4 next-token targets. So an item is a moment for Zihin and a predicted token for a network, 4 per
  experience for both.
- Zihin is the baseline's variant "members" (F11).
- The networks: the baseline's TinyGPT (2 layers, width 64, 4 heads, about 106k parameters here), and a GRU and an
  LSTM of width 64 (29-41k parameters). All use Adam 3e-3 and are untuned.
- The quality rows reproduce the 2026_11_14 baseline exactly. All 30 final results (10 lives × Zihin, online and
  offline TinyGPT, and the offline epoch counts) equal `baseline_transformer.jsonl`.

**1. Learning throughput, one pass over the stream** (medians over 3 repetitions × 2 domains):

| learner | items/s, 1 thread | items/s, 4 threads | ms per item, 1 thread | against Zihin |
|---|---|---|---|---|
| Zihin, online, one pass | 136 | 135 | 7.34 | 1 |
| Zihin without its ledger | 137 | | 7.28 | 1.0× |
| TinyGPT online (one Adam step per utterance) | 1,338 | 572 | 0.75 | 9.8× |
| GRU online | 2,563 | 2,169 | 0.39 | 19× |
| LSTM online | 2,712 | 2,209 | 0.37 | 20× |
| TinyGPT, batches of 32 (one epoch) | 24,804 | 24,601 | 0.040 | 182× |
| GRU, batches of 32 | 52,687 | 41,417 | 0.019 | 387× |
| LSTM, batches of 32 | 57,281 | 48,803 | 0.017 | 421× |

- On this stream Zihin learns 136 moments a second.
- A network that learns as Zihin does, one utterance at a time and once, is 10-20 times faster.
- Trained the usual way, in batches, a network is 180-420 times faster per pass.
- Four threads help nobody here. Zihin is one sequential life, and these networks are too small: four threads made
  the batch-1 TinyGPT slower (and that row is noisy).
- The ledger that makes retraction exact costs about 1% of the time.

**2. Answer latency.** Zihin was measured with `Session.reply` on 10 everyday questions, loaded without its ledger.
The answers were identical across repetitions (same hash; their content was never printed). The networks were
measured on next-token inference at the mind's vocabulary size, with random weights, since latency does not depend
on the weights.

| | first breath (97k ties, 1,711 words) | cevahir2 (1.33M ties, 10,360 words) |
|---|---|---|
| Zihin, per produced word, end to end (hearing, thinking, speaking) | 24.5 ms | 82.7 ms |
| Zihin, a moment that produces a word | 10.2 ms | 9.4 ms |
| Zihin, first word after the question is given | 41.6 ms | 151 ms |
| TinyGPT without a cache (its standard form), 1 thread | 0.46 ms | 1.02 ms |
| decoder of the same size with a key/value cache, 1 thread | 0.24 ms | 0.57 ms |
| GRU / LSTM, 1 thread | 0.07 / 0.14 ms | 0.32 / 0.57 ms |
| GPT-2-small layout (d 768, 12 layers), 1 thread / 4 threads | 40.2 / 15.1 ms | 40.8 / 15.5 ms |
| the same with GPT-2's 50,257 tokens (163M parameters, untied) | | 52.6 / 18.1 ms |

- Measured per word-producing moment, Zihin (about 10 ms) is 4 times faster than a decoder of GPT-2-small size on one
  core and 1.5-1.6 times faster than it on four. It is 9-160 times slower than the small networks.
- End to end it is slower: on cevahir2 a reply took 331 ms for about 4 words. Zihin hears and learns the question
  and thinks before it speaks: 192 moments for 40 words. Some moments that produce a word are thoughts, not speech.
- Learning at the minds' vocabulary was also timed: Zihin watching 20 dialogues against one Adam step per 11-token
  utterance. On first breath Zihin learns 264 moments/s, against 2,202-3,796 items/s for the networks on 1 thread.
  On cevahir2 Zihin learns 247 moments/s, against 976-1,130 on 1 thread and 1,625-2,429 on 4 threads.
- The gap narrows as the vocabulary grows: 10-20× on the stream, 4× at 10k words on one core. A network's step
  grows with its vocabulary (a dense output layer and its Adam update); a Zihin moment barely does.

**3. Memory per learned item.**

- **On the stream:**
  - Zihin's core is 332-500 B per moment; the whole mind pickled without its ledger is 411-619 B per moment.
  - The ledger is 3.1-3.9 kB per moment, 6-12 times the core.
  - The networks' parameters are fixed: 117-432 kB, with 2× that for Adam's state while training. That is 49-180
    B per item after one pass, and the share falls as more items come; Zihin's memory grows with every moment.
- **On the saved minds:**
  - First breath: core 263 B per moment, process 3.3 kB per moment (RSS growth on load).
  - cevahir2: core 736 B per moment (90 MB), process 6.8 kB per moment (0.83 GB).
  - Process memory is 9-13 times the core tensors: the Python indices beside the tensors, not the tensors, are the
    memory to beat.
  - The small networks at 10,360 words hold 5.4-6.0 MB, whatever they have learned.

**4. Quality at equal learning time** (means over 5 lives per domain; the clock is paused while probing; accuracy =
the right value on top). "Offline" is the baseline's training: batches of 32, epochs until the loss stops falling.
"Within TinyGPT's pass" is Zihin's last checkpoint before TinyGPT's one pass ended (about 2 s): 60 experiences lived
in 9 of 10 lives.

| nitelik | learning s | held-out, told rule | held-out, induced | members | exceptions | early facts |
|---|---|---|---|---|---|---|
| Zihin, one pass | 20.5 | 0.90 | 0.97 | 1.00 | 1.00 | 1.00 |
| Zihin within TinyGPT's pass | 1.3 | 0.60 | 0.60 | 0.71 | 0.41 | 0.86 |
| TinyGPT / GRU / LSTM online, one pass | 2.1 / 1.2 / 1.1 | 0.67 / 0.77 / 0.80 | 0.57 / 0.60 / 0.53 | 0.82 / 0.91 / 0.85 | 0.66 / 0.44 / 0.44 | 0.44 / 0.34 / 0.38 |
| TinyGPT offline (41-70 epochs) | 6.2 | 0.77 | 0.67 | 1.00 | 1.00 | 1.00 |
| GRU offline (48-78 epochs) | 3.2 | 0.90 | 0.67 | 1.00 | 1.00 | 1.00 |
| LSTM offline (57-100 epochs) | 3.7 | 0.97 | 0.83 | 1.00 | 1.00 | 1.00 |

| iliski | learning s | held-out, told rule | held-out, induced | members | exceptions | early facts |
|---|---|---|---|---|---|---|
| Zihin, one pass | 18.4 | 0.75 | 0.95 | 1.00 | 1.00 | 1.00 |
| Zihin within TinyGPT's pass | 1.2 | 0.50 | 0.45 | 0.73 | 0.81 | 0.56 |
| TinyGPT / GRU / LSTM online, one pass | 2.0 / 1.1 / 1.0 | 0.60 / 0.45 / 0.50 | 0.50 / 0.47 / 0.57 | 0.78 / 0.87 / 0.85 | 0.80 / 1.00 / 1.00 | 0.46 / 0.40 / 0.34 |
| TinyGPT offline (40-81 epochs) | 7.2 | 0.53 | 0.53 | 1.00 | 1.00 | 1.00 |
| GRU offline (43-74 epochs) | 3.0 | 0.70 | 0.60 | 1.00 | 1.00 | 1.00 |
| LSTM offline (47-89 epochs) | 3.0 | 0.72 | 0.60 | 1.00 | 1.00 | 1.00 |

- Every network had finished learning before Zihin's life was over, in 1-9 s against 17-23 s. Zihin's time is not
  shorter than any network's time to its best answer.
- At the same short time Zihin has lived only about 60 experiences and is behind on held-out and members. It is
  ahead only on early facts.
- With the full stream, Zihin keeps the clear lead on iliski (held-out mean 0.85 against 0.65-0.66 for the
  recurrent networks and 0.53 for TinyGPT).
- **A new and honest result:** on nitelik an LSTM trained offline nearly ties Zihin.
  - Held-out mean 0.90 ± 0.04 against 0.93 ± 0.03 for Zihin.
  - The LSTM is better on the told rule (0.97 against 0.90) and worse on the induced rule (0.83 ± 0.09 against
    0.97 ± 0.03).
  - It gets there in 3.7 s.
  - The 2026_11_14 comparison had only Transformers; the recurrent networks are the stronger baseline in this
    regime.
- Online, one pass, the condition Zihin lives in, no network comes close. Held-out is 0.45-0.80, and the facts told
  once early are 0.34-0.46 against Zihin's 1.00 (forgetting).
- There are only 6-8 held-out probes per life, so standard errors are about 0.03-0.10.

**Where Zihin's time goes.** These are wall-clock shares of `Mind.step` on the stream, without a profiler, at 7.4-7.6
ms per moment:

- reading predictions every moment (`predictions`, the F11 members reading): 28-34%;
- learning (`learn_follow`; induction is about half of it under cProfile): 32%;
- feelings: 10-11%;
- deciding acts: 7-10%;
- lag-0 learning: 8%;
- settling: 7%.

Under cProfile a moment makes about 4,500 Python calls and about 600 small tensor operations: per moment
`index_select` 209, `sum` 167, `zeros` 100, `where` 81, `tensor` 74. Assembling evidence rows in the store (`w`, `w2`)
is about a quarter of all time. Each such operation costs microseconds of dispatch whatever its size. The core is
bound by interpreter and dispatch overhead, not by arithmetic.

**Unfair to Zihin:**

- **Implementation.** Zihin is Python driving many tiny tensor operations. The networks run a few fused, vectorised
  kernels (MKL, oneDNN). The race measures this implementation, not the floor of the method.
- **Work per moment.** A moment does more than predict and update. It builds structure (units, categories, frames),
  reads with F11, feels, decides acts, and keeps an exact ledger. A network step only predicts and updates fixed
  weights.
- **Retraction.** Zihin learns each item once, and the item stays exactly retractable (20-77 ms per experience in
  2026_11_14). The networks reach their best only by seeing everything 40-100 times; their exact retraction is
  retraining.
- **Vocabulary.** The networks are given the whole vocabulary in advance. Zihin grows its alphabet as it lives.
- **Batching.** Networks batch, 32 utterances per step. A life's moments depend on each other and cannot be batched.
  Separate minds can run side by side: two quality processes ran at once, and their Zihin lives took about 10%
  longer than alone (confounded with the other agents' load).
- **Replies.** A Zihin reply also hears and learns the question and its own words, and thinks first. A network's
  inference learns nothing.
- **Weights.** Network latency was measured with random weights, Zihin's on real minds.

**Unfair to the networks:**

- **Size and tuning.** They are tiny and untuned. A GPU would make them far faster, batched most of all; this race
  is CPU only.
- **Batch 1.** Batch-1 online training is their least efficient regime; it is used only because it is Zihin's.
- **No cache.** TinyGPT has no cache in its standard form, so the cached decoder row is the fair one.
- **Cost growth.** Zihin's cost per moment grows with what it has lived (more ties per unit). The networks' cost per
  token is fixed by their size.
- **Equal time.** At equal time a network could also have used Zihin's whole 18-20 s, for example several trainings
  or an ensemble. That was not done.

**What to beat:**

- One pass, online, one core: 0.75 ms per item (TinyGPT) and 0.37-0.39 ms (GRU, LSTM), against Zihin's 7.3 ms.
  That is a 10-20× gap.
- The profile places that gap in the number of small operations per moment, not in what they compute.
- Matching batched training (0.02-0.04 ms per item per pass) needs a different kind of implementation, vectorised
  over many ties at once or compiled, not tuning.
- Already won: a word-producing moment (about 10 ms) beats a GPT-2-sized decoder on CPU; quality after one online
  pass beats every network; generalisation beats them on iliski; and the cost per moment barely grows with
  vocabulary.

## Low-end machines: Zihin against Transformers on one core (2026-11-15)

The owner believes that Cevahir runs on low-end machines where Transformers could not. Measured on one core with no
GPU (`lowend_profile.py`, `results/lowend_profile.jsonl`; an independent verifier re-measured RAM to within about
2 MB and corrected the points below). Every figure includes the same baseline of about 515 MB for Python and torch.

| | RAM to answer | RAM to train | answer time on one core |
|---|---|---|---|
| Zihin, cevahir_ilk_nefes (98k ties) | 0.64 GB | (it lives; no training) | 42 ms per reply |
| Zihin, cevahir2 (1.33M ties, today's layout) | 1.36 GB | | 114 ms |
| Zihin, cevahir3b (3.25M ties, today's layout) | 2.58 GB | | 223 ms |
| Transformer 1M / 4M / 20M / 100M parameters | 0.53 / 0.54 / 0.61 / 0.92 GB | about 0.9 / 1.0 / 1.2–1.7 / 2.4–3.3 GB, by batch size | 15 words: 21 ms / 43 ms / 0.2 s / 1.1 s |

- **Belief (2) is not supported at these sizes.** Transformers of 1–100M parameters answer on a 2 GB machine in less
  RAM than the larger Zihin minds; with small batches a 20M model even trains in 2 GB and a 100M model in 4 GB (the
  first summary said otherwise because it fixed the batch size). It holds only against billion-parameter models.
- **What holds is that Zihin needs no training run.** It learns while living at about 3–5 ms of CPU per moment; a
  Transformer must first be trained: 20 epochs over 260k tokens take 9 minutes (1M) to 7 hours (100M) on one core
  (the first summary's 13 minutes to 9.8 hours were measured on a busier machine).
- **Zihin's RAM grows with what it lived:** about 650 B per tie (504 B of Python index objects, 138 B of tensors),
  about 7 kB per lived moment; the file is 116.5 B per tie. The Python indices are 78% of the RAM of cevahir3b.
- **Projection** (linear in lived moments at cevahir3b's density, 11 ties per moment): two million teacher examples
  (16 moments each, 32M moments) would need about 190–260 GB of RAM and a 38 GB file in today's design, and 47–79
  hours on one core to live once, about as much as 20 epochs of a 4M Transformer over the same data. Time is not the
  barrier; memory is. The time-per-moment exponent (ties^0.13) is not robust (its confidence interval includes 0),
  so only "grows slowly" holds.
- **The three changes that would change this most:** compact integer tables instead of Python objects (about 500 B
  to 30–50 B per tie); memory-mapped tie arrays on disk so that RAM holds the working set (see "Memory map and
  locality"); consolidating rarely used ties in sleep (this changes learning and must be measured).
- `lowend_profile.py` builds its fresh teacher mind with the lessons that were removed later the same day
  (`lessons_about_itself`), so that step runs only at the commits before d628100.

## Same data, same questions: Zihin against Transformers trained from scratch (2026-11-15)

The owner believes that a Transformer trained on this little data could not even give these answers. Tested with the
repository's own Transformer (`CevahirNeuralNetwork` from `src/neural_network.py`, standalone on CPU), trained from
scratch on exactly what the fresh Zihin mind of the F25 teacher protocol lived (recorded from a real session:
1358 teacher pairs, the lessons of that time, 13,290 sentences of the owner's documents; 195,332 moments), and asked
the same 200 held-out questions, scored by the same function (`same_data_duel.py`, `results/same_data_duel.jsonl`;
numbers only; an independent verifier re-scored every answer and retrained two models bit for bit).

| | exact | token F1 | "bilmiyorum" |
|---|---|---|---|
| Zihin, fresh, lived once | 0.130 (26) | 0.300 | 0.170 |
| Transformer 1M / 4M, BPE, one pass in lived order | 0 / 0 | 0.007 / 0.063 | 0 / 0 |
| Transformer 1M, Zihin's own units, one pass | 0 | 0.000 (silent) | 0 |
| Transformer 1M / 4M, BPE, 20 / 10 epochs | 0 / 0 | 0.045 / 0.056 | 0.005 / 0.030 |
| Transformer 1M, Zihin's own units, 20 epochs | 0 | 0.100 | 0.255 |
| a lookup that does not learn: the first document sentence that starts with the asked word | 0.930 (186) | 0.946 | |

- At the mind's own exposure (one pass) no Transformer gave a usable answer; with 15–30 times the exposure (10–20
  epochs, 14–34 minutes on two threads) none answered exactly and the best F1 was 0.100 against Zihin's 0.300.
- The tokenizer does not explain the gap: with exactly Zihin's units the Transformer did better than with BPE, still
  far below.
- **A second attempt to make the Transformer as strong as possible** (an independent verifier, about 1.5 hours: a
  larger learning rate, 27 epochs, fine-tuning on the question form, beam search and sampling) reached at best 3 of
  200 exact and F1 0.135 under the protocol, against Zihin's 26 and 0.300. It was stopped before it finished. Given
  the first words of the answer instead of the question (a diagnostic, not the protocol), the same model completed
  34% exactly: it holds much of the documents' text but cannot reach it from a one-word question.
- **The limits are large.** The first many-epoch runs stopped at an epoch cap (best epochs 14 of 20, 18 of 20, 10 of
  10), only 1M–4M models were run, one seed each. The BPE models also met a case mismatch (they were trained on raw
  case and asked lower-cased questions), and the word-level 1M model has only 111,720 parameters outside its
  embeddings. A larger or longer-trained Transformer may do better; this shows that one to two hours of training per
  model on this machine got nowhere near.
- **The task itself is weak.** The teacher questions were generated from the documents by a template ("<X> nedir ?",
  answered by the first sentence that starts with X; `research/living_mind_2026_11_14/teacher_qa.py`), so a lookup
  that learns nothing answers 186 of 200. Belief (1) holds for this small budget on this task; it is not shown in
  principle, and Zihin's 13% is far from what the data allows. The audit of 2026-11-15 also flags these generated
  dialogues as canned content: they will not be used to raise minds again, and the next evaluations will use
  questions the owner writes or template-free cloze probes.
- `same_data_duel.py` recorded the stream through the lessons of that time (`lessons_about_itself`,
  `not_knowing_lesson`), so it runs only at the commits before d628100; between runs the script was edited and the
  epoch cap was lowered from 24 to 20 after a restart.
- Memory and time on the same machine: Zihin peaked at 1.85 GB living and 1.91 GB answering (753 ms per answer on a
  loaded machine); the Transformers at 0.94–1.83 GB training and 0.54–0.57 GB answering (39–255 ms).

## Memory: compact indices and a day store merged in sleep (2026-11-15)

The memory map above found that 78–87% of a loaded mind's RAM was Python index objects. Four changes to the store
replace them with compact forms. Every one is exact: the same lookups return the same ids in the same order, so
every result is bit-identical (17 configurations of the lesson-free harness, and cevahir_ilk_nefes, cevahir2 and
cevahir3b compared moment by moment over 60 watched dialogues and 3 replies; tests for each step).

1. **A tie's key is one packed integer** (`tie_key`) instead of a 4-tuple, and **an adjacency list is an array of
   32-bit ids** instead of a list of Python ints (4 bytes per id against about 40).
2. **Frame keys are one bytes object** (`frame_key`: the slot's positions as a bit mask, the masked parts as 32-bit
   ints) instead of nested tuples of Python ints.
3. **Arrays grow by an eighth** of their capacity instead of doubling, so up to half of every array is no longer left
   empty, and a loaded mind's tie arrays no longer double at its first new tie.
4. **The tie lookup is a long-term store and a day store.** Per form and lag, the long-term store keeps the keys
   src·2^31 + dst sorted, with their ids, in two numpy arrays (12 bytes a tie, against about 100 for a dict entry);
   the ties made since the last merge are in a small dict. A lookup asks the day store, then bisects the long-term
   keys. The day is merged in sleep, when the mind is saved, or when it grows past an eighth of the long-term store.
   This is the counterpart, in memory, of consolidating the day in sleep: what was learned today is found at once, and
   sleep sorts it into the long-term store.

Minds saved before are converted on load. RAM of a loaded mind over the Python and torch base (about 514 MB), after a
fresh load of a file saved in each form:

| | this morning | 1 | + 2 | + 3, 4 |
|---|---|---|---|---|
| cevahir2 (1.33M ties) | 717 MB | 554 MB | 441 MB | 319 MB |
| cevahir3b (3.25M ties) | 1784 MB | 1391 MB | 1127 MB | 762 MB |
| loading cevahir3b | 6.7 s | 4.8 s | 4.2 s | 2.3 s |

- cevahir3b now needs about 234 bytes of RAM per tie, against 549 this morning. A whole 60-dialogue session with it
  peaks at about 1.5 GB of process memory with a file saved in the new form (4.0 GB this morning).
- What remains: the tensors (about 220 MB of evidence and columns in cevahir3b, plus growth room), the frame table's
  bytes keys and values, the composed units' tuple index, the per-list array headers, and torch's own 484 MB on import.
- Not yet built: the dormancy certificate (reading only the ties that can still change, from the one rule), and a
  memory-mapped file in which the long-term store stays on disk and only the region a moment reads comes into RAM.

## Substitution Evidence and the Substitute Reading: a first measurement (2026-11-15)

`zihin/anlam.py` and `read.substitute_backoff`, behind `Config.substitution` (off by default). A witness is a moment
where the mind expected x (e > 1/2) and y occurred instead; at the next moment, what follows y is read against x's
own lived continuations: if x led to it, inferred evidence is added to y's membership of x's categories, if x led
elsewhere, evidence against (amount 2 |e - 1/2|). The Substitute Reading then lets a context never lived as a whole be
read through the lived context that differs from it in one word sharing a category, held with conviction
lam · s · conviction, s = (2 e(p→g) − 1)(2 e(q→g) − 1). No word list; every increment is logged.

On the T-meaning world (3 seeds, 2 days, speaking configuration):

- **It does what it is for at the level of memberships.** Shared categories rose for merhaba / selam 9.7 → 13.7,
  merhaba / hello 1.0 → 4.3, merhaba / selamlar 1.0 → 2.7, doktor / hekim 3.3 → 5.0, cevap / yanıt 4.0 → 5.7.
- **It does not yet change what the mind says.** The right first word's expectation rose for some never-watched
  wordings (merhabalar 0.442 → 0.574, nasılsınız 0.442 → 0.574) but stayed under the line at which it would be said
  (0.583); "ismin ne ?" stayed at 0.576. Greetings, paraphrase answers and the meaning questions are unchanged.
- Lowering the speaking line would make the mind speak on weak evidence everywhere, so it was not done. The flag stays
  off; what would carry a substitution over the line is stronger evidence of equivalence (more lived substitutions,
  consequences across the turn), not a smaller threshold.

## Recall of what was read once: Sequence Starts (F26), Lived Exclusion (F27) and the end of a read sentence (2026-11-15)

**First, a mistake in the earlier evaluations.** The teacher protocol (`research/living_mind_2026_11_14/teacher_qa.py`)
taught "<X> nedir ?" answered by the first document sentence that starts with X. But `ogretmen.tsv.heldout` holds the
held-out subject alone, and every evaluation that read it asked that bare word: `f25_teacher.py`, `same_data_duel.py`
and a white-box probe of this afternoon. A mind that never lived a one-word question has no frame for it. On a mind
raised today on the 1358 pairs and the documents (no lessons), 0 of 200 answers began with the asked word, and 138 were
silent; the asked word's expectation at the turn end was exactly 1/2 in all 200. Asked in the taught form, the same
mind began 200 of 200 answers with the asked word and said 59 exactly. The duel's Transformers were prompted with the
bare word too ("<soru> X <cevap>"). So the duel measured both sides on a question neither had learned. It is not rerun
in the taught form here; the template-free test below replaces it.

**Where the answers failed, decision by decision** (a white box at every moment of every answer):

- **Most failures were at the second word.** The frame (C, nedir, ⏎, C) carries NEXT: "the answer goes on with what
  the asked word goes on with". NEXT resolved to the asked word's strongest continuation anywhere, often a word that
  followed it inside other sentences.
- **The memory already held what was needed.** For all 200 held-out subjects the composed unit (X, w2) exists, because
  reading the sentence began an episode X, w2. For 176 the right w2 is the most frequent of those beginnings, and 137
  subjects begin only one sentence.
- **Many answers said the whole sentence and went on** (21 of 200 with the switches off, 34 with Sequence Starts).
  Reading never perceived the end of a sentence. `Session.body` strips the closing punctuation so that it is
  perceived with the end of an utterance. But `read_path`, unlike `live`, `watch` and the lines of `!ogret`, never
  perceived that end, so a recalled sentence had no lived end.
- **Many answers stopped in the middle.** A generalisation proposed another word just under the speaking line and raced
  the lived continuation. In the template-free test, a context lived once with "," had a frame proposing ":" at 0.657
  against 0.667, and the mind was silent.

**Three changes.**

1. **F26, Sequence Starts** (`Config.starts`; `read.start_mass`, reading only).
   - **What it counts.** An episode begins after a pause, and its context after two items is the composed unit
     (x, v). That unit is recognised once in every episode that began x, v, and each recognition is evidence on its
     PART ties. Their trust-weighted mass B(v) counts those beginnings. What began with x and went on otherwise counts
     against v: W+(v) = B(v), W−(v) = Σ_{v'≠v} B(v').
   - **Nothing new is stored.** The atom's own lag-1 ties keep what followed x anywhere.
   - **Frames.** A frame's members agree on START when the value each lived is how a sequence that began with it went
     on. The priority is SELF, then START, then NEXT, then the value itself. START resolves to the strongest beginning
     of the filler, never to a fixed word of the frame.
   - **F16.** An utterance of one item so far is read as a sequence that began with it, before the item anywhere. A
     composed suffix shorter than F is already such a beginning.
   - **A word brought to mind.** A moment that is one item after a pause reads its beginnings last, as own evidence.
     What never went on a sequence that began so counts against.
2. **F27, Lived Exclusion** (`Config.exclusive`; `read.excluded`, reading only).
   - **The principle.** A channel carries one item per moment. So what an instance was itself lived to be followed by
     on a channel is evidence against every target there that it was never followed by.
   - **The reading.** Some targets are proposed by a generalisation but were never lived after the instance, on a world
     channel where the instance went on with something. Each is read with W+ = 0 and W− = the instance's own positive
     mass on that channel, nested last as own evidence: e ← (1 − c) e, c = conviction(W−).
   - **What is untouched.** Targets with the instance's own evidence keep it (F14 records their failures), and feelings
     are never touched.
   - **Relation to F25.** This is the full-record reading F25 described for ties not yet born, applied where a
     generalisation proposes them. It is a reading only, whereas F25 (a) changed learning.
3. **The end of a read sentence is perceived** (`Session.read_path`, program behaviour). The mind hears the sentence,
   then perceives the end of the utterance with its tone (imitated), then a pause, as for every other utterance it hears.

**Teacher protocol, taught form** (`recall_once.py teacher`; 200 held-out subjects; minds raised on the 1358 pairs and
then the 11 documents, no lessons; F1 and prefix against the answer without its closing punctuation):

| mind / reading | exact | starts with the subject | silent | F1 | prefix | went on past the end |
|---|---|---|---|---|---|---|
| bare word, as the earlier evaluations asked | 0 | 0 | 138 | 0.000 | 0.000 | 0 |
| sentence ends not perceived, switches off | 59 | 200 | 0 | 0.534 | 0.492 | 21 |
| sentence ends not perceived, starts | 89 | 200 | 0 | 0.722 | 0.695 | 34 |
| sentence ends perceived, switches off | 78 | 200 | 0 | 0.547 | 0.492 | 2 |
| sentence ends perceived, starts | 117 | 200 | 0 | 0.737 | 0.695 | 6 |
| sentence ends perceived, starts + exclusive | 119 | 200 | 0 | 0.741 | 0.700 | 6 |

The lookup that learns nothing (the first sentence that starts with the subject) answers 186 by the construction of the
protocol. Of the 83 left with starts and sentence ends (white box), 62 are cases where reading order would be needed:

- 23 begin with a subject that begins several sentences. The mind says the most frequent beginning and goes on with
  that sentence; the reference is the first such sentence in the documents.
- 31 stop where the sentence's first words begin several sentences and the lived continuations fork.
- 8 stop where the members predict the end: bibliographic answers such as "X , r ." end there in the pairs.

The other 21 are small classes.

**Template-free recall** (`recall_once.py mind` and `transformer`).

- **The mind.** It lived only the documents: 186,457 moments, 2.30 million ties, about five minutes on one core.
- **The sample.** 200 sentences of the same documents, sampled with `random.Random(0)`.
- **The cue.** The first one or two words of each sentence are brought to mind with the inner voice.
- **Scores.** Exact means the rest of that very sentence. Valid means the rest of some read sentence that begins with
  the same words. Unique start means the cue words begin only that sentence.

| | exposure | 2 words: exact / valid / silent | unique start | 1 word: exact / valid / silent | unique start |
|---|---|---|---|---|---|
| Zihin, switches off | one reading | 59 / 90 / 57 | 40/61 | 11 / 30 / 63 | 8/21 |
| Zihin, starts | one reading | 66 / 116 / 51 | 42/61 | 22 / 42 / 109 | 19/21 |
| Zihin, starts + exclusive | one reading | **89 / 150 / 27** | **60/61** | 23 / 44 / 109 | 19/21 |
| Transformer, repository model, 1.0M parameters, one pass | 58 s | 0 / 0 / 1 | 0/61 | 0 / 0 / 5 | 0/21 |
| same, best of 18 epochs | 34 min | 0 / 10 / 3 | 0/61 | 0 / 4 / 10 | 0/21 |
| GPT, the duel verifier's recipe, 7.35M parameters, best epoch 16 of 27 | 50 min | 17 / 41 / 1 | 11/61 | 16 / 51 / 0 | 6/21 |

- **Two words that begin only one read sentence:** the mind says the rest of it exactly in 60 of 61.
- **What the Transformers saw.** They were trained for the duel, on the pairs and the documents, with the mind's units.
  The mind they are compared with here lived only the documents.
- **A mind raised on the duel's stream** (pairs, then documents) gives the same 89 / 150 with two words, but 13 / 32 with
  one. The subjects of the pairs also began a question ("X nedir"), so their beginnings are split.
- **One word, valid sentences:** the GPT says one more often (51 against 44). When the word begins several sentences it
  always says one of them, while the mind stays silent (109 silences) unless one beginning leads.
- **Limits.** One data set and one sample of 200, with no variation of the sampling seed. The Transformers are the
  duel's: small, trained for its protocol, one seed each. A larger or longer-trained model may do better.

**Worlds** (the `f25_born_failures.py --speaking` protocol, 5 seeds × 2 domains, 600 experiences; scratch runners
that import it unchanged):

- **Starts alone.** The lives are identical, except that one nitelik seed differs at 300 experiences (−0.08).
- **Starts + exclusive, separate lives.** In these worlds the mind is not listening while the world speaks, so it
  speaks into the stream and lives its own echoes. With exclusive it speaks less there, and the lives differ:
  - ties 25,939 → 20,366 and categories 338 → 220 (nitelik);
  - T1 c1 0.208 → 0.092, because the white-box conviction counts the mind's own echoes with trust 1;
  - T3 exception_ok 0.707 → 0.635 and held_induced_ok 0.567 → 0.533.
- **The same lives, read both ways** (switches off while living; probes read off and on at each checkpoint):
  - nitelik exception_ok 0.707 → 0.780, exception_kept 0.743 → 0.835, held_induced_ok 0.567 → 0.533, member_ok
    0.796 → 0.791;
  - iliski member_ok 0.757 → 0.820, early_recall 0.24 → 0.28;
  - the rest equal.

  What the readings do is right. What changes the separate lives is how much the mind talks over the world, which
  this protocol was not made for.
- **T-meaning** (3 seeds, 2 days): identical tables with and without.

**Exactness.** With the switches off and the old `read_path`, the code gives the committed code's digests in all 17
configurations, and the lesson-free gate is unchanged (6626 moments, 314bdd47…). With the end of a read sentence
perceived it gives 7039 moments, 162de2fe….

**Tests.**

- `test_zihin_starts.py` (6) and `test_zihin_exclusion.py` (4). The exclusion tests are hand-checkable: the frame's
  ":" goes from 25/42 to 25/63, and "," stays at 2/3.
- Two tests changed:
  - a sentence's last word now has the perceived end after it, so it can be a topic;
  - a dream may end with a topic brought to mind after which its race was silent.

**Defaults.** The speaking configuration (new chat minds) now reads both. A mind raised before keeps its own
configuration, and the program names them among the mechanisms it lives without (`--ac starts,exclusive`).

## Exact Relations (F28), and the chat configuration listening in the worlds (2026-11-15)

The chat configuration did poorly in the generated worlds of the first lives: held-out 0.567 and 0.037 after 600
experiences, against the "members" configuration's 93% and 81% (RESULTS.md). Two causes were found today, one in the
protocol and one in the core (`speaking_worlds.py`; 5 seeds × 2 domains; numbers at 600 experiences).

**1. It spoke into the world's stream.** `run.py` lets the mind act in every moment, and the chat configuration does:
it lives its own words between the world's. In a chat it listens while the partner speaks (`Session.listen`). With
that one change of protocol, and the same switches as in `f25_born_failures.py`:

| | nitelik, speaking into the stream | nitelik, listening | iliski, speaking into the stream | iliski, listening |
|---|---|---|---|---|
| held-out (told rule / induced) | 0.567 (0.567 / 0.567) | 0.633 (0.767 / 0.500) | 0.037 (0.050 / 0.025) | 0.025 (0.050 / 0.000) |
| members | 0.796 | 0.879 | 0.757 | 0.983 |
| exceptions right / kept | 0.707 / 0.743 | 0.765 / 0.765 | 1.000 / 1.000 | 1.000 / 1.000 |
| early facts recalled | 0.520 | 0.720 | 0.240 | 0.960 |
| hidden classes (Jaccard) | 0.566 | 1.000 | 0.875 | 1.000 |
| categories / ties | 338 / 25,939 | 61 / 9,832 | 229 / 22,910 | 56 / 6,621 |
| T1 c1 | 0.208 | 0.400 | 0.106 | 0.409 |

- Speaking into the stream it learned five times as many categories, mostly from its own words, and found the hidden
  classes worse.
- Held-out answers in the relational world stayed near zero even when it listened.

**2. F12's NEXT took the members' agreement on a value.** In the relational world an utterance is "<relation> <entity>
<value>", so the value follows the entity directly. Every member's value is then also one of the words the member goes
on with. F12 counted such a value as NEXT ("what the member goes on with") and never as the value itself. So the members
of a category, who had all lived the same rule value, were read as agreeing on NEXT. NEXT resolves to the read entity's
own strongest continuation: a value of another relation, or its category.

**F28, Exact Relations** (`Config.exact_relations`; `read.member_votes`, reading only). A relation counts for a member
only where it names the very value the member lived, resolved as it would be for the instance being read:

- SELF names the member itself;
- START names its strongest beginning (F26);
- NEXT names its strongest continuation;
- the fixed words of the frame are never named.

Of the relations that name the same value, only the most specific counts (SELF, then START, then NEXT). The value
itself always counts. So members that agree on a value now agree on it, and members that agree on a relation still do.
In the toy of `test_zihin_exact_relations.py`, the members kedi and köpek both lived "siyah" after "renk", but each went
on most with another word. Off, the frame says NEXT, which points at kuş's own continuation: e("kısa") = 25/42,
e("siyah") = 1/2. With F28 it is the other way round.

**Measured.**

- **Listening, the three readings of today (F26, F27, F28) off and on.** The learned structure is the same in every
  life (ties, categories, T1).

  | | nitelik off | nitelik on | iliski off | iliski on |
  |---|---|---|---|---|
  | held-out (told rule / induced) | 0.633 (0.767 / 0.500) | **0.783** (0.833 / 0.733) | 0.025 (0.050 / 0.000) | **0.812** (0.725 / 0.900) |
  | members | 0.879 | 1.000 | 0.983 | 1.000 |
  | exceptions right / kept | 0.765 / 0.765 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
  | early facts recalled | 0.720 | 0.920 | 0.960 | 1.000 |

  Held-out in the relational world went from 0.025 to 0.812, where the "members" configuration was (0.81). F28 alone
  gave 0.700 and 0.812 (scratch run); F26 and F27 add the rest in nitelik.
- **The lives of the old protocol (speaking into the stream), read both ways at each checkpoint.**
  - nitelik: told rule 0.567 → 0.533, induced 0.567 → 0.500, members 0.796 → 0.798, exceptions right 0.707 → 0.780
    and kept 0.743 → 0.835, early facts 0.520 → 0.540.
  - iliski: told rule 0.050 → 0.325, induced 0.025 → 0.150, members 0.757 → 0.838, early facts 0.240 → 0.340.

  In these lives the held-out probes of nitelik lose a little; the lives are the ones its own words filled.
- **Separate lives of the old protocol, off against on** (scratch run of `f25_born_failures.py --speaking`):
  - nitelik: held-out 0.567 → 0.517 (told 0.567 → 0.667, induced 0.567 → 0.367);
  - iliski: held-out 0.037 → 0.163;
  - categories 338 → 216 and 229 → 184 (it speaks less into the stream).
- **Teacher protocol, taught form:** 119 → 120 exact, F1 0.741 → 0.745.
- **Template-free recall:**
  - Two words: exact / valid / silent 89 / 150 / 27 → 90 / 145 / 32. Seven items stopped being valid and two became
    valid. All nine are forks, where several read sentences go on from the same words and the loose NEXT used to break
    the tie by accident.
  - One word: 23 / 44 → 23 / 47.
- **T-meaning** (3 seeds, 2 days): the replies are the same.
  - hello: best e 0.592 → 0.625.
  - "hi" told once: answered from its own told meaning in 0 → 1 of 3 seeds.
  - selamlar: best e 0.700 → 0.645, still leading.
- **Cost.** Answering moments that read a large category: 31 → 45 ms a moment in the teacher protocol (1358 members;
  each member's strongest continuation and beginning is found, once per moment).

**Defaults.** The speaking configuration reads F28 too. Minds raised before keep their configuration (`--ac
exact_relations`).

**What this changes in what was said before.** The README and the theory said that the chat configuration does poorly
in these worlds. Most of that was its speaking into the stream (a protocol the worlds were not made for) and NEXT taking
the members' agreement. Listening, with F26–F28, it reaches 0.78 and 0.81, with members, exceptions and the hidden classes
all at 1.00. The "members" configuration's 0.93 in nitelik is not reached.

## Analogy by Interchangeability (F29): what two words have in common, read from the places they filled (2026-11-15)

**The failure, decision by decision.** In the meaning world, 9 of the 17 wordings never watched failed just under the
speaking line, with the right first word leading at e = 0.562–0.580 (white box at the turn end of every probe). In
most of them, the frame that spoke was born at the probe moment from one lived instance. For example, `[X] , adın ne
<turn end>` had the single member merhaba, whose instance was lived four times and answered "adım". It spoke with
c = 1/2 · 0.667 · conviction(0.667) = 0.133, so e = 0.567. The mind's own knowledge that merhaba and selam (or adın
and ismin) stand in the same places in many other lived frames was not used: the new category's memberships start
from one recognition.

**Interchangeability.** A category is a lived place: the fillers of one slot of contexts that differ only there. Two
words y and x are interchangeable to the extent that they filled the same places:

- **For.** Each shared place g counts m_y m_x / (n_g − 1). Here m = 2e − 1 is a membership's evidence above
  ignorance, and the place's evidence is shared among its n_g members. A place of two is a whole piece; the place of
  everything that ends an utterance is almost none.
- **Against.** A place where one is a member and the other is held not to be (e < 1/2).
- **Only what was lived before the present run of items counts** (`memberships_before`, the ABSTRACT tie's birth).
  The utterance being read cannot vouch for itself: its own frames would otherwise make every new word similar to the
  words it replaces. `Mind.episode_start` keeps the moment the run began.

| pair (seed 0 / seed 1) | shared places | W+ | conviction |
|---|---|---|---|
| merhaba / selam | 8 / 8, mostly of two | 3.31 / 3.78 | 0.77 / 0.79 |
| adın / ismin | 7 / 2 | 0.78 / 0.22 | 0.44 / 0.18 |
| hızlı / çabuk, doktor / hekim, cevap / yanıt | 2–7 | 0.17–0.72 | 0.14–0.42 |
| merhaba / hello, merhaba / selamlar | 1, of ten | 0.04–0.05 | 0.03–0.05 |
| merhaba / evet, merhaba / teşekkürler, evet / hayır | 1, of ten | 0.04–0.06 | 0.04–0.06 |
| araba / doktor, cevap / kedi, adın / nerede | 0 | 0 | 0 |

**The reading** (`read.member_analogy`, `Config.analogy`, on in the speaking configuration):

- The members of the frame the members' reading chose carry what they lived over to the read instance by the analogy
  of NARS. From "x's instance went on with v" (f1, c1) and "y and x are interchangeable" (f2, c2): f = f1 f2,
  c = c1 c2 f2.
- The members' conclusions add as evidence: each member's instance is one lived source.
- The members keep the frequency of every claim. The analogy only decides how firmly a claim holds for this instance:
  the conviction is the stronger of the members' breadth (F11) and the depth carried over.

**Two forms that failed first** (measured, kept here):

1. **Lived contexts one interchangeable word away, pooled by revision and joined by the choice rule.** The meaning
   world gained as much, and "hi" told once was answered as merhaba in 3 of 3 seeds. But the listening worlds fell:
   nitelik 0.783 → 0.650, iliski 0.812 → 0.600 (though iliski at 60 experiences rose 0.393 → 0.628). In those worlds
   every entity fills the same places as every other, so the pooled neighbours' values competed with the class frame.
2. **Inside the chosen frame, with the analogy's own frequencies.** The worlds were fine (iliski 0.812 → 0.850), but
   the teacher protocol fell 120 → 58. The literal second word of one interchangeable subject was carried over
   against the 1357 other members, who all agree on START, not on that word.

The commit message of 77c8b69 credited the final form with the 3 of 3 for "hi"; that came from the first form. With
the final form, "hi" is answered from its told meaning in 1 of 3 seeds, as before.

**Measured, final form** (`meaning_switches.py`, `speaking_worlds.py --listen`, `recall_once.py`):

| | off | on |
|---|---|---|
| meaning world (3 seeds), paraphrases with a greeting in front | 0.17 | **0.67** |
| meaning world, synonym paraphrases | 0.33 | **0.56** |
| meaning world, extra word / form variant / restructured | 0.33 / 0 / 0 | 0.33 / 0 / 0 |
| greetings and meaning questions | | same replies |
| listening worlds, held-out at 600: nitelik / iliski | 0.783 / 0.812 | 0.783 / 0.812 |
| members, exceptions, hidden classes | 1.00 | 1.00 |
| template-free recall, two words: exact / valid / silent | 90 / 145 / 32 | 90 / 146 / 31 |
| template-free recall, one word | 23 / 47 / 109 | 23 / 47 / 109 |
| teacher protocol, taught form | 120 (F1 0.745) | 120 (F1 0.744) |

- **Now answered** (seed 0): "selam , adın ne ?", "selam , nasılsın ?", "hi , nasılsın ?", "ismin ne ?" and "senin
  ismin ne ?". In seeds where adın and ismin shared fewer read places, the two "ismin" wordings stay just under the
  line.
- **The five hard wordings stay silent:** "adını söyler misin ?", "nasılsınız ?" (a form never met), "keyfin nasıl ?",
  "ne haber ?", "evin nerede ?". No wrong answer appeared anywhere.
- **Still out of reach:**
  - an extra word in front ("peki adın ne ?": nothing says that "peki" changes nothing);
  - a word form never met;
  - a restructured question;
  - greetings lived once or twice (hello, selamlar), whose interchangeability evidence is too small to carry anything.
- **Cost.** Answering moments that read large categories are about 18% slower (teacher protocol 112 → 132 s for 200
  answers).
- **Exactness.** With the switch off, all 17 configurations are bit-identical to the committed code, and the
  lesson-free gate is unchanged. Tests: `test_zihin_analogy.py` (7), including the hand-checkable 2/9 of two places of
  two members and the exclusion of the present run.
- **Prior work.** Interchangeability from shared contexts is the distributional hypothesis (Harris), weighted as
  telling places are, and the carrying over is NARS's analogy rule. What is ours is reading both from the mind's own
  induced places, at the moment of use, inside the members' reading.

## Corroboration (F30): independent lived evidence about the same claim adds up (2026-11-15)

**The gap.** The members' reading lets one frame speak and consults the situations most like the present one (F23)
only when nothing else expects anything. Two independent readings that agree never add up. For "peki adın ne ?", the
frame `[X] adın ne <end>` (its one member: "senin adın ne ?", lived four times) said "adım" at 0.567, and the bare
lived question "adın ne ?" (twelve times), found by F23, said it at 0.547. Each stayed under the speaking line alone.

**F30** (`read.corroborated`, `Config.corroborate`, on in the speaking configuration):

- **The revision.** The claims of the frame that speaks are revised with what the lived situations most like the
  present one say about the same values, by the revision rule of NARS. Each reading's (f, c) is turned into evidence
  w = k c / (1 - c), and the evidence adds: W+ = f1 w1 + f2 w2, W- = (1 - f1) w1 + (1 - f2) w2.
- **Disagreement counts.** A value's share of the situations' votes is f2, so situations that went on otherwise count
  against it.
- **No lived context is counted twice.** The situations leave out the frame's members' instances, and every frame:
  a frame is a generalisation, not a lived situation, and its own ties hold its members' evidence as literal values.
- **Values only the situations claim** are left to F23's own reading, which speaks when nothing else does.

**A latent fault found on the way.** F23 itself counts frames among the lived situations, because a frame holds the
same words at the same places. In its fallback use this lets a frame's literal values vote. In the toy of
`test_zihin_exact_relations.py` with loose relations, it voted "armut" and "kiraz" against the SELF relation, and the
first corroboration silenced "muz nedir ?". Corroboration now reads only lived situations. F23's own fallback is left
as it was for now (changing it changes every F23 answer and needs its own measurement).

**Measured** (`meaning_switches.py --set corroborate=0|1` with F26–F29 on; `speaking_worlds.py --listen`;
`recall_once.py`):

| | F29 | F29 + F30 |
|---|---|---|
| meaning world, a greeting in front (selam, selamlar, hello, hi) | 0.67 | **1.00** |
| meaning world, an extra word ("peki adın ne ?") | 0.33 | **0.67** |
| meaning world, synonyms | 0.56 | **0.67** |
| meaning world, form variant / restructured | 0 / 0 | 0 / 0, all silent |
| greetings and meaning questions | | same replies |
| listening worlds, held-out at 600 | 0.783 / 0.812 | 0.783 / 0.812 |
| teacher protocol, taught form | 120 | 120 |
| template-free recall, two words / one word (valid) | 146 / 47 | 146 / 47 |

- **The meaning world over the day.**
  - A greeting in front: 0.17 → 0.67 (F29) → 1.00 (F30).
  - Synonyms: 0.33 → 0.56 → 0.67.
  - An extra word: 0.33 → 0.67.
  - The five hard wordings stayed silent throughout ("adını söyler misin ?", "nasılsınız ?", "keyfin nasıl ?", "ne
    haber ?", "evin nerede ?").
- **The worlds of the first lives do not change.** They hold no perceived event inside an utterance, so F23 finds no
  situation there. Recall from one reading does not change either (the inner voice brings words to mind without one).
- **A narrow life generalises what it lived.** In a toy where every "... ne ?" question was the name question, "keyfin
  ne ?" is answered with the name: the induction from what it lived, pinned in `test_zihin_corroboration.py`. Where
  other "... ne ?" questions had other answers, it stays silent.
- **Cost.** About 13% slower answers in the teacher protocol (F23 is now read whenever a frame speaks after a
  perceived event).
- **Exactness.** With the switch off, all 17 configurations are bit-identical and the lesson-free gate is unchanged.

## Paradigmatic Relation (F31) and Relational Binding (F32): the answer stands where the asked word stands (2026-11-15)

**The gap.** A frame's members agree on a value or on a relation of the value to the filler:
- SELF: the value is the filler itself;
- NEXT: what the filler goes on with;
- START: how a sequence that began with the filler went on.

All three relate the value to the filler by identity or along the line of speech (the syntagmatic axis). In "otomobil
ne demek ?" → "otomobil , araba demektir .", the word after the comma follows otomobil nowhere. It is a word that
stands where otomobil stands: araba and otomobil fill the same places in what was read. The four members of the
definition frame each lived their own word there (araba, doktor, cevap, hızlı), so the frame had nothing to agree on.
"selam ne demek ?" was answered "selam ," and then silence.

**F31** (`read.stand_in`, `Config.paradigmatic`). The other axis, the paradigmatic one (Saussure; Jakobson's axis of
selection; the vertical axis of the owner's "dikey düşünme").
- **The stand-in of y.** The item most interchangeable with y, read from the lived places both filled before the
  present run of items (F29's interchangeability, all candidates in one pass over y's places). It must be held above ½
  and held more than any other item. It is never y, a fixed word of the frame, or a rival filler of the frame's own
  place: the place's fillers are what the frame tells apart, not what it relates them to.
- **A new relation, ALIKE.** A value a member lived counts under ALIKE when it is that member's stand-in, exactly the
  value (F28).
  - The relation is sought only where it can hold for the members: where the share of their evidence that could name
    it is above ½. This bounds the effort and changes no held reading.
  - For the read instance, ALIKE resolves to the filler's stand-in: selam → merhaba.

**F32** (`read.bound_frames`, `Config.bind_relations`). The word after the stand-in.
- "selam ne demek ? ⏎ selam , merhaba" was never lived, and a frame for it would need two places.
- A place already binds its repeats (bind_repeats: the filler itself wherever it recurs). F32 binds the filler's
  stand-in the same way. A context whose last item is the stand-in of its prefix's filler is read through each
  member's instance followed by that member's own stand-in: "otomobil ne demek ? ⏎ otomobil , araba" goes on with
  "demektir".
- **What failed first.** Binding continuations as well (NEXT, START) lowered the teacher protocol: 120 → 103 exact of
  200, and 52 → 48 of 80 in the comparison of forms. A continuation is what follows the filler, read by the
  context's own evidence, not a second place.

**Measured** (`meaning_switches.py --set paradigmatic=0|1,bind_relations=0|1`; `recall_once.py` and
`speaking_worlds.py --listen` with all switches of the speaking configuration):

| | F30 | + F31 | + F31 + F32 |
|---|---|---|---|
| meaning world, "selam / hello / merhaba ne demek ?" (3 seeds) | 0 / 0 / 0 | 0.67 each | 0.67 each |
| reply (seed 0) | "selam ," | "selam , merhaba ," | "selam , merhaba demektir" |
| meaning world, paraphrases by wording | | unchanged | unchanged |
| teacher protocol, taught form (200) | 120, F1 0.7444 | 120 | 120, F1 0.7444 |
| template-free recall, two words / one word (exact, valid, silent) | 90/146/31, 23/47/109 | | same |
| listening worlds, held-out at 600 | 0.783 / 0.812 | 0.783 / 0.812 | 0.783 / 0.812 |

- **The answers.** "selam , merhaba demektir", "hello , selam demektir", "merhaba , selam demektir". Nothing in the
  definitions names a greeting; the relation was learned from four definitions and from where the words stood.
- **Seed 2 stays "selam ,".** In that life "otomobil" never filled a lived place with "araba" (the two were read in
  different sentences), and "hekim" stood more where "çorba" stood ("annem X") than where "doktor" did. So only two
  of the four members carry the relation (f = ½). That is the evidence of that life, not a slip of the reading.
- **Cost.**
  - Answers in the teacher protocol: about 10% slower (165 s against 150 s for 200 answers).
  - Recall from one reading, where the inner voice reads between moments: about 20% slower with one word and 45%
    with two (11.5 s against 9.5 s, 48 s against 33 s, all runs four at a time).
  - What brought it there: the candidates of a stand-in in one array pass; the relation sought only where it can
    hold; the reading kept while the moment does not change. That reading is dropped at every step, undo and
    retraction, the way F23 keeps its reading.
- **Exactness.** With the switches off, the lesson-free gate and the 17 configurations are bit-identical.
- **Both are on in the speaking configuration.** A mind raised before them lives without them unless the owner
  turns them on (`--ac paradigmatic bind_relations`).

**F23 reading only lived situations** (`Config.lived_situations`, off; the fault noted with F30). With it on, the
listening worlds (0.783 / 0.812) and every answer of the meaning world are unchanged; two "best e" values moved by
about 0.02. Off until the teacher protocol and recall are measured with it.

## Relational Memory (F33): one mind, one memory, recalled in the company of the person present (2026-11-15)

**The owner's direction.** One mind develops different relational histories with different people, as a person
does with their mother, a friend or a stranger. It is not a mind, a personality or a behaviour template per person.
- It recognises the people it meets and ties what it lives with each of them into its one memory.
- It learns the character of each relationship from that experience, and recalls the relevant history the next
  time.
- General knowledge is the whole mind's; the privacy of personal experience, and the contexts in which it may be
  recalled, are kept.
- Relationships may change over time.

**What existed.** The chat perceived who speaks: the partner given with `--kim` or `!kim`, with every word, as a
context facet. Nothing read it: what followed what was one evidence for everyone.

**The world, T-relations** (`t_relations.py`, public, 3 seeds × 3 days, the chat's Session and speaking
configuration).
- Four people talk with the mind, and two strangers appear only in the probes.
- Each person is greeted their own way ("merhaba canım , nasılsın ?", "merhaba kardeşim .", ...) and has their own
  answers to their own questions ("en sevdiğim renk ne ?", "kedimin adı ne ?").
- One of them alone has a secret ("sırrım ne ?" → "sırrın mavi kutuda .").
- All four ask the mind its name and how it is, and get the same answers.
- Between conversations it watches dialogues of someone unknown ("su nedir ?", "merhaba" → "merhaba , hoş geldin .")
  and reads sentences.

**Without F33 (the person-blind mind).**
- Every person got the general greeting (own 0).
- Every personal question stopped after its form ("en sevdiğin renk"): the four answers disagree.
- The secret, lived with one person only, was told to everyone: 3 of the other 4 known people and every stranger.

**F33** (`zihin/iliski.py`, `Config.company`, on in the speaking configuration).
- **One memory and a ledger.** Every experience is lived into the one memory as before. While a known person is
  present (`mind.company`, set by the session: the partner, or the one who asks in a watched dialogue; a document or
  a taught file is no one's in particular), every increment on what followed what (EXPECT, lag 1 or more) and on how
  a sequence began (PART of a two-item unit, F26) is also noted with that person. Learning reads and writes exactly
  as before.
- **Recall depends on who is present** (`company_view`, read wherever a unit's own evidence is read):
  1. what was lived with no one in particular: with anyone;
  2. what was lived with other people: only as far as they corroborate one another. Each person is one voice, their
     share of the unit's positive evidence among the targets of each channel. n ≥ 2 voices give conviction
     lam · conviction(n − 1): one person's voice alone is theirs (F30's principle: independent evidence adds up, and
     the first person's evidence is not evidence about anyone else);
  3. the person present comes last: their own evidence nests over the rest, as an instance's own evidence does.
- **That a question went on otherwise is known from the whole life** (F27 reads the whole evidence; what it went on
  with is recalled in company). So a stranger asking the secret question is left silent, rather than answered with
  something a frame learned elsewhere: with recall in company alone, the frame "[X] ne ?" answered it with the name.
- **The ledger is undone with its moment** (undo, `!unut`), and saved with the mind.

**Measured** (`t_relations.py --set company=0|1`):

| | without F33 | with F33 |
|---|---|---|
| a known person gets their own greeting / colour / cat's name | 0 / 0 / 0 | 1.00 / 1.00 / 1.00 |
| the secret, to its person | 1.00 | 1.00 |
| the secret, to the other known people / to strangers | 0.75 / 1.00 | 0 / 0 (silent) |
| someone else's greeting, colour or cat's name to a known person or a stranger | 0 | 0 |
| the name, which all four taught, to a stranger | 1.00 | 1.00 |
| watched of no one in particular ("su nedir ?"), to anyone | 1.00 | 1.00 |

- **The replies.**
  - "merhaba canım , nasılsın" to the person greeted so, "merhaba , hoş geldin" to a stranger.
  - "en sevdiğin renk kırmızı" to the person whose colour it is, "en sevdiğin renk" and silence to a stranger: it
    knows how such an answer goes, not this person's.
- **What is not right yet.** After "merhaba kardeşim" the mind says "hoş", from the general habit, where the person's
  own greeting ended.
- **Nothing else moves.** The other protocols have no named person, so they are unchanged with the switch on: teacher
  protocol, recall, listening worlds and the meaning world (no note is made; reading is as before).
- **Cost.** Raising a mind in this world takes about 20% longer (2.0 s against 1.7 s for 1638 moments). The ledger
  noted 643 ties. It is a dictionary of rows (a few hundred bytes a tie), to be made compact before long lives with
  people.
- **Exactness.** With the switch off, the lesson-free gate and the 17 configurations are bit-identical.

**Not yet** (next):
- trust learned per person (whose word has held);
- relationships changing over time (recent experience against old);
- recognising a person from what is perceived of them (face, voice, way of speaking) rather than from the name the
  session is given.

### F33, second step: whose word holds, and a relationship that changes (2026-11-16)

**Trust learned per person** (`iliski.attribute`, `iliski.person_trust`).
- **How the record is kept.** Direct experience judges the evidence of the other classes on the same tie, as
  before, and that verdict also feeds the class's trust. With F33, each person's noted share of the judged evidence
  is how much of the verdict is theirs: a record per person and class of what held and what did not.
- **How it is read.** The person's record nests over the class's trust (their own record comes last). Their word is
  weighed by it wherever their evidence is recalled.
- **The world seen directly** (sensed) is no one's in particular: only what was said, watched or done with someone
  is noted with them.

**T-trust** (`t_trust.py`, public, 3 seeds × 3 days). zeynep tells facts about animals that the mind's own
experience later confirms ("kuşlar uçar ."); mehmet tells facts it contradicts ("balıklar uçar ." against a sensed
"balıklar yüzer ."). At the end each tells one fact it never checks.

| | without F33 | with F33 |
|---|---|---|
| trust in told speech as a whole | 0.569 | 0.569 |
| trust in zeynep's word / mehmet's word | (none: one trust for all) | 0.956 / 0.638 |
| zeynep's unchecked fact, held with her / with mehmet / with a stranger | 0.681 / 0.681 / 0.681 | 0.744 / 0.5 / 0.5 |
| mehmet's unchecked fact, held with him / with zeynep / with a stranger | 0.681 / 0.681 / 0.681 | 0.695 / 0.5 / 0.5 |

- **The records separate the two people,** while the person-blind mind holds one trust for everyone. Each
  unchecked fact is that person's: it is not held with anyone else.
- **mehmet keeps 0.64, not near 0.** Every transition of an utterance is a claim, and his verbs end sentences as
  they do in what the mind sees (the sensed "köpekler havlar ." confirms "havlar" then the end, which he also said
  in "kediler havlar ."). Only his subject–verb pairs are false. A finer notion of what an utterance claims (the
  informative part, not what everyone's sentences share) is open.

**A relationship that changes** (`t_relations.py --change`). One person greets the mind "merhaba hocam ." for four
days and then "merhaba canım ." (two greetings a day, among two of someone unknown).

| | day 4 | day 5 | day 6 | day 7 | day 8 |
|---|---|---|---|---|---|
| without F33 | "merhaba" | "merhaba" | "merhaba" | "merhaba" | "merhaba" |
| with F33 (all 3 seeds) | "merhaba hocam hoş geldin" | "merhaba" | "merhaba" | "merhaba canım" | "merhaba canım" |

- **Change comes from evidence, not from a forgetting rule.** The new habit takes over after 6 encounters, against
  the 8 that made the old one. Each new encounter is also a failure of the old expectation, recorded as such.
- **No recency weighting was added.** How much faster a recent relationship should outweigh an old one is open.
- **Without F33** the habits of everyone mix and the reply stops after "merhaba".

## Reply Relation (F34): compositional generalisation, a primitive seen alone used in every template (2026-11-16)

**The test, T-compose** (`t_compose.py`, public). It is SCAN's "add primitive" split (Lake & Baroni 2018), made small
enough for the mind's window of seven items.
- **Commands.** Four primitives (walk, run, look, jump → i_walk, i_run, i_look, i_jump) and two directions (left,
  right → i_lturn, i_rturn) in eleven templates, with answers of at most four actions: X, X twice, X thrice, X left,
  X right, X opposite left / right, X and Y, X after Y, X left twice, X twice and Y.
- **Training.** Every command of walk, run and look, and "jump" only alone, watched as dialogues three times in a
  shuffled order.
- **Test.** Every command where jump stands in a template: 25 per seed.

**Without F34 the mind answered none** (0/75). "walk and jump" came out "i_jump", and "jump twice" silence.
- The frame "[X] twice ⏎" took jump in as a member while the question was heard, but its members' answers (i_walk,
  i_run, i_look) agree on nothing the mind could name.
- Corroboration (F30) found "jump ⏎" the lived situation most like "walk and jump ⏎", because "jump" is rare, and
  pulled the frame's right claim down.

**F34** (`read.reply_of`, `read.bound_frames`, `Config.replies`, on in the speaking configuration):
- **REPLY.** What the filler, said on its own, was answered with: the strongest world continuation of the lived
  context (filler, the frame's perceived event), the end of a turn. NEXT is what a word goes on with; REPLY is what
  it as a whole utterance was followed by. A member's value counts under REPLY when it is exactly that (F28). The
  members of "[X] twice ⏎" agree on it (f = 1), and for jump it resolves to i_jump.
- **Several places bound** (F32 generalised). The items after a prefix of the context may each be the filler's
  stand-in (ALIKE), its reply (REPLY), or an item every member has there. The context is then read through each
  member's instance with its own related items. So "jump twice ⏎ i_jump" is read through "walk twice ⏎ i_walk", and
  "jump left twice ⏎ i_lturn i_jump i_lturn" through "walk left twice ⏎ i_lturn i_walk i_lturn".
- **Corroboration reads the lived suffixes of the context.** Evidence about a frame's claim is what followed the
  claim's own context however much of it was lived: the context without its front ("adın ne ⏎" for "peki adın ne
  ⏎"). A suffix counts by how much of the frame's fixed items it holds. A situation that holds another item where
  the claim holds one is evidence about that item, not about this claim. Two attempts failed before this:
  - comparing only the frame's fixed items let "walk left twice ⏎" vote against "jump twice ⏎" (5/25);
  - leaving out a different filler only, but not a different fixed item, let "run and walk ⏎" vote against "jump
    twice and walk ⏎" (20/25).

**Measured**:

| T-compose, add-primitive split (3 seeds, 75 commands) | exact |
|---|---|
| the mind without F34 | 0 / 75 |
| **the mind with F34** | **75 / 75** (every template 100%) |
| a decoder-only Transformer trained from scratch on the same dialogues (3 layers, d = 64, 6000 steps; its own 43 training commands all exact) | 4 / 75 |

- **The Transformer** learns its training set completely and answers almost nothing new: "walk and jump" → "i_walk
  i_walk", "run twice and jump" → "i_run i_run ⏎ i_run". This is the result Lake & Baroni report for sequence models
  on this split.
- **The mind** learns what "jump" means from "jump" alone, and the templates from the other three primitives. It
  composes the two the first time it is asked.
- **Nothing else moves.** Teacher protocol 120 exact (149 s), template-free recall 90/146/31 and 23/47/109, listening
  worlds 0.783 / 0.812, relations and trust worlds unchanged.
  - The meaning world: the greeting told once ("hi , merhaba demektir .") is now answered as a greeting in 3 of 3
    seeds (was 1).
  - One synonym wording ("ismin ne ?") was lost in one seed: the old corroboration found "adın ne ⏎" by resemblance,
    and the suffix rule does not.
- **A narrow life no longer over-generalises.** One where every "... ne ?" was the name question answered "keyfin ne
  ?" with the name; with F34 it stays silent (a word never met, and no lived suffix).
- **Limits.** The mind's window is seven items, so SCAN's longer commands ("jump around left twice", up to 48
  actions) are out of reach. The hierarchy that would carry them (chunks of chunks) is a next layer.
- **Exactness.** With the switch off, the lesson-free gate and the 17 configurations are bit-identical.

## Generalised Consequences (F35): acting for what acts lead to, in situations never met, and per relationship (2026-11-16)

**What existed (L5).** Before acting, the mind weighs what the path "this episode, then this act" led to: the value
signals (good, bad) its lag-2 ties hold, collected when that exact path was taken and received (`!dogru`,
`!yanlis`). A path never taken had no consequence at all. So what the mind learned from how its acts were received
stayed with the exact situations in which it was received.

**The world, T-purpose** (`t_purpose.py`, public, 3 seeds).
- The mind watches people being handed things ("al bu elma"), answered "ver bakayım ." twice as often as
  "teşekkür ederim .": imitation alone answers rudely.
- Then it is handed three things itself over six rounds, and its answers are received: "teşekkür" well, anything
  else badly.
- Then it is handed three things it was never handed itself, with no feedback.

**Without F35** it learned the three things it was handed (9/9 polite after the rounds) and answered every new thing
with the habit (0/9).

**F35** (`act.general_consequences`, `Config.purpose`, on in the speaking configuration).
- **What a path would lead to is read from the paths its frames hold,** as the frames of a context are found: an
  item of the path replaced by a category it is a member of. Every member whose path was taken and received votes
  the share of its value evidence each value received (F11).
- **The frame whose members agree most speaks,** with c = lam · e(path → frame) · conviction(members). The path's
  own record comes last, as own evidence always does.
- **It is read only from the paths that were ever followed by a value,** so a mind never received has no cost from it.

**With F33: expectations per relationship.**
- How a person received the mind's act is noted with that person, the one sensed signal that is (the world sensed
  directly stays no one's in particular).
- What a path led to is read in present company, so what pleased one person is an expectation of that relationship.
- **The world `--people`.** Two people hand it things in the rounds: ayşe receives the polite answer well and the
  banter badly; mehmet, who jokes with it, the other way round. Habits are watched equally.

**Measured**:

| | without F35 | F35 | F35 + F33 |
|---|---|---|---|
| polite to the things it was handed / to new things | 9/9 / 0/9 | 9/9 / **8/9** | |
| new things answered the way ayşe / mehmet receives well | 6/9 / 3/9 | 3/9 / 3/9 | **9/9 / 9/9** |

- **It also learns faster.** With F35 one bad reception was enough: the first round already went "ver bakayım",
  "teşekkür ederim", "teşekkür ederim", because the other things' paths share the frame.
- **One new thing in one seed was met with silence:** habit and consequence were balanced there.
- **The two relationships.**
  - Without F33 their consequences cancel (3/9 each).
  - With F33 the mind answers ayşe "teşekkür ederim" and mehmet "ver bakayım" for things neither ever handed it.
  - No role, rule or person is coded: the difference is only what was lived with each, as the owner asked.
- **Nothing else moves.** No other protocol gives value feedback, so the meaning world, T-compose (75/75), the
  teacher protocol, recall and the generated worlds read exactly as before.
- **Exactness.** With the switch off, the lesson-free gate and the 17 configurations are bit-identical.

## F34 at larger windows: SCAN's own grammar (2026-11-16)

**What fits a window.** `t_compose.py --scan F` generates SCAN's grammar, all 20,910 commands with their meanings, and
keeps the commands whose words, end and answer fit a window of F items:
- F = 11: 1573 training commands, and 844 test commands with jump;
- F = 15: 4133 training and 2246 test commands.

Training is every command without jump and "jump" alone; the test is every command with jump. The mind's window is a
knob it can turn itself (3..15). `--around --window 11` adds "X around left / right" (eight actions) to T-compose.

| | the mind (F34) | Transformer (same dialogues, exact on its training commands) |
|---|---|---|
| T-compose with "around", window 11 (3 seeds) | **81 / 81** | 7 / 81 (training 49/49) |
| SCAN's grammar within a window of 11 (seed 0) | **609 / 844 (72%)** | **0 / 844** (training 1570/1574) |

- **Most of the mind's failures are one error: it does not stop.** "walk after jump" → "i_jump i_walk i_walk i_walk
  …". The prefix is right; then a weaker reading (what followed i_walk anywhere) proposes another action, while the
  members' instances ended there. Of 304 command shapes, 199 are all right.
- **Living the 4720 training dialogues took 11 minutes** (58,803 moments). Answering the 844 tests, each on a fresh
  copy of the mind, took about 20.
- **Commands longer than the window remain out of reach:** 70% of SCAN at a window of 15. They need the question held
  as one item while the answer is said (a working memory).

## Word Forms (F36): a word never met read as the form it is of (2026-11-16)

**The gap.** A word never met had no place, no category and no frame. The mind could not relate "nasılsınız" to the
lived "nasılsın", or "merhabalar" to "merhaba", and stayed silent: the meaning world's form variant and unseen
greeting stood at 0.

**F36** (`read.form_base`, `read.form_reading`, `Config.forms`, on in the speaking configuration). The layer below
words.
- **The words the mind lived are read as letters.** The base of a word never met is the lived word sharing its
  longest stem.
- **The stem must be a boundary of the lived words themselves** (Harris's successor variety): a lived word, or a
  beginning that lived words go on from in two ways. A beginning only one lived word has, cut where the new word
  happens to differ ("mas" of "masa" for "maske"), is not a stem.
- **How far the base stands for the word.** f = the share of the longer word the stem covers. c = conviction(the lived
  words beginning with the stem).
- **The reading.** A context holding a word never met is read, when nothing would be said otherwise (no world target
  over the speaking line), as the lived context with the base in its place. Its reading is carried over with
  e ← c f (e' − ½) + ½, the analogy of NARS.

**Measured** (`meaning_switches.py --set forms=0|1`, 3 seeds):

| | without F36 | with F36 |
|---|---|---|
| "merhabalar" (never met) greeted as a greeting | 0.00 | **1.00** ("merhaba , hoş geldin") |
| "nasılsınız ?" (never met) | 0.00 | **1.00** ("iyiyim , teşekkür ederim") |
| every other probe of the meaning world | | unchanged |

- **What failed first.** The reading was first used only when no world target exceeded ½. Then "nasılsınız" was cut
  after "iyiyim ,": another reading held something just over ½ but under the speaking line. It is now used when
  nothing would be said.
- **Nothing else moves.** Teacher protocol 120 exact, listening worlds 0.783 / 0.812, T-compose 75/75, relations world
  unchanged. Recall cues are lived words, and the reading needs a word never met.
- **Limits.** Only beginnings are stems: an ending shared with lived words ("-lar") is not yet evidence of its own,
  and a stem that changes ("keyif" / "keyfin") is not found.
- **Exactness.** With the switch off, the lesson-free gate and the 17 configurations are bit-identical.

**An audit of what the mind can and cannot do** (relational and causal inference, composition, analogy, exploration,
new categories, autonomous goals, chains of inference, contradictions, sources, verification) is in
[DENETIM_2026_11_16.md](DENETIM_2026_11_16.md), in Turkish. It includes the controlled experiment on "Yağmur nasıl
yağar sence ?": none of the owner's three minds here produces a causal account of rain (cevahir5 and cevahir are
silent; cevahir3b says "bilmiyorum , öğrenmek isterim .", all five words from lived evidence), and the private
documents hold no text on how rain forms.
