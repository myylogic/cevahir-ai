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
- `undo_last` gives the moment's rows back: every stored array, not only the values, is as before.
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
single runs on a shared machine). The small run takes 1.0 s, against 0.8-0.9 s. Reading a tie now assembles its
row instead of indexing it.

**What remains, by size per tie (68.6 B):**

- The other tie columns, 44 B: t_src and t_dst are int64, as are t_last and t_birth. int32 would save 16 B, with
  a guard at 2^31.
- Python caches beside the tensors: `tie_index` (a dict of 4-tuples) and the `out`/`inc` lists. These are
  probably the next memory problem, but their size was not measured. The 4.0 GB process peak also includes
  converting the old file on load.
- A disk-backed store and consolidation in sleep, (c) and (d) above. Not started.
