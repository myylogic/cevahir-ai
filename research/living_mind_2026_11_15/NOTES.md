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

| exact answer | mean token F1 | "bilmiyorum" | silent |
|---|---|---|---|
| 17% (10/60) | 0.38 | 23% (14/60) | 0 |

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
