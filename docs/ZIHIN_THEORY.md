# The Cevahir Zihin Theory: a Living Evidence Mind

*Muhammed Yasin Yılmaz, November 2026. This text describes the theory and architecture of the `zihin/` core in this
repository. Measurements and negative results for every claim are in the notes and result files under
`research/living_mind_2026_11_14/` and `research/living_mind_2026_11_15/`. Turkish original:
[ZIHIN_KURAM.md](ZIHIN_KURAM.md).*

## In short

Cevahir Zihin is not a Transformer. It has no neural network, weight matrices, embedding vectors, backpropagation or
softmax attention. It is a **Living Evidence Mind**:

- whatever it perceives becomes a unit, and every relation between units is a tie;
- each tie keeps graded evidence for and against the relation, **separately by where the evidence came from**: seen
  with its own eyes, told, imitated, its own act, inferred or imagined.

It reads what to expect from a single rule and learns at every moment. While its ledger is kept, the evidence of any
single experience can be taken back exactly.

Training and use are not separate phases: **to learn is to live**. The model learns in every conversation.

## 1. Basic entities

**Channels** are how the mind touches the world and itself. There are eight channel kinds:

- `sense`: the turn end, the gist of an image or of a sound, the inner not-knowing;
- `act`: words, whoever says them, and the mind's own acts;
- `feel`: surprise, relief, conflict, load;
- `knob`: its own settings;
- `source`: where evidence comes from;
- `facet`: how a thing looks or sounds, or a piece of a word;
- `value`: a good or bad outcome;
- `context`: who speaks, tone, the self.

A mind has named channels, each of one kind. Words arrive on the act-kind channel `text` whoever says them; whether a
word was told by someone, imitated or said by the mind itself is kept as its provenance (below), not as a channel.

**Units** carry kind flags:

- an atom: a symbol on a channel, such as a word, an image digest or a turn end;
- a composed unit: a lived sequence, a window of up to seven items;
- a frame: a composed unit one of whose places is a category, such as "_ eats fish"; frames are the generalisations
  section 5 relies on;
- a category: a class of units lived in the same places (a category induced from a frame is marked as induced).

**Ties** relate two units:

- `EXPECT` with lag 0 (together), lag 1 (right after) or lag 2 (a consequence later in the same episode);
- `PART`: part of a sequence;
- `ABSTRACT`: member of a category;
- `RELY` (optional): how reliable a generalisation has been; made only when the reliability setting (F9) is on, which
  the chat program does not turn on.

**Provenance Evidence.** Each tie keeps evidence for (W+) and against (W−) separately for six origins: *sensed, told,
imitated, acted, inferred, imagined*. The amounts are graded, not counts: a confirmation adds an amount scaled by how
active the expecting unit was and how strong the occurrence was; a failure adds an amount weighted by the tie's own
expectation (section 4). Where knowledge came from is never lost.

## 2. One rule: Conviction-weighted Expectation

How much a tie is expected is read from its evidence, weighted by the trust in each origin:

- frequency `f = W+ / W`;
- conviction `c = W / (W + k)`, with `k = 1`;
- expectation `e = c · (f − ½) + ½`.

This is the expectation of a truth value in NARS (Pei Wang's Non-Axiomatic Reasoning System) with `k = 1`. This core
reads it from evidence kept per origin and weighted by learned trust. With no evidence the expectation is ½, which
means "I don't know". Conviction grows with evidence. Every reading rests on this rule: what to say, what to expect,
what to be surprised by.

**Trust Judgment.** Sensed evidence and the mind's own acts are direct experience; their trust is 1 by definition.
Only sensed evidence, experience of the world, judges: the echo of the mind's own act is certain but says nothing
about whether a claim about the world holds. The trust in the four indirect origins (told, imitated, inferred,
imagined) starts at ½, the rule's value with no evidence, and is learned: indirect evidence is compared with the
sensed evidence that comes later on the same tie; what holds raises the trust in that origin, and what fails lowers
it. So the mind learns by
itself how far to believe whom.

## 3. Attention: the Seven-Place Focus

In a Transformer, attention lets each token look at all earlier tokens in its context window through learned vectors
and a softmax. It is dense, and its cost grows with the length of the context. Cevahir Zihin attends by a different
mechanism.

**Seven-Place Focus, or Focus Settling.** At each moment only seven units enter the focus (k-winners-take-all,
F = 7). The candidates are:

- what is perceived now;
- what is carried over from earlier moments; carried activation is multiplied by `1 − 1/F = 6/7` each moment;
- the recognised composed context, meaning the lived sequence of the episode;
- their categories and frames, which are the generalisations;
- what lived togetherness calls to mind.

Settling takes at most six rounds: association spreads, winners are chosen again, and it stops when nothing changes.
What is present comes first, then the most specific generalisation of each place.

Consequences:

- there is no context window: a moment reads the ties of the units in focus and never looks back over the past;
- the cost of a moment therefore depends on those ties, not on any context length; but frequent words gain ties as the
  mind grows, so the time per moment grows slowly with memory (about 1.5 ms on a small mind, 2.6 ms at 1.33 million
  ties and 3.3 ms at 2.54 million ties after the speed work);
- attention is readable: you can see which seven units are in focus and why;
- there are no attention weights to train, because attention arises from evidence.

## 4. The flow of a moment

At each moment, roughly each word, the mind lives these steps:

1. **Perception.** What comes from outside and what comes from inside are perceived together. Inside means the echo
   of its own words, its feelings and the felt not-knowing. When the mind itself speaks or thinks, the **Self Unit**
   accompanies the moment.
2. **Learning what comes next** (lag 1). What was in focus a moment ago learns to expect what happens now.
   - **Present-Only Learning:** only what really was the present a moment ago teaches; units still fading in the
     focus do not.
   - **Expected-Failure Rule:** a tie receives evidence against it only when its target's channel reported something
     else in that moment, and in the chat program only if the tie was expected at least `1/(2F) = 1/14`. The amount
     is weighted by the tie's expectation, so a rare follower settles near the floor instead of being refuted at full
     weight.
3. **Learning consequences.** A good or bad outcome is credited to every context in the episode's **Consequence
   Trace**. This is how the mind learns which path leads where.
4. **Focus.** The Seven-Place Focus settles, as described above.
5. **Learning togetherness** (lag 0). Atoms in focus learn to expect each other. Facets such as looks, sounds and
   word pieces bind to the item they accompany.
6. **Expectation and decision.** What is expected for the next moment is read. If it is the mind's turn, its options
   race.
7. **Felt Not-Knowing.** If it is the mind's turn and it has nothing to say, it perceives its not knowing as an inner
   event.
8. **Feelings.** Surprise, relief, conflict and load become inner percepts of the next moment.
9. **Ledger.** Every increment of the moment is written to a ledger. This makes **Exact Retraction** possible: the
   evidence a single experience added is subtracted bit for bit, and the increments of later moments stay untouched.
   It is exact for the evidence, not a replay of a life without that experience: units and ties it created stay when
   later moments used them, and what was learned later in its presence stays. In the chat program the ledger covers
   the current session, because the mind is saved without it.

## 5. Generalisation: categories speak with their members

- **Members' Reading.** A category reads what to expect for an instance from what its members themselves lived, not
  from rules of its own, and the instance being read is left out of the vote. On instances never told the asked
  dimension, it improved with age: 93% / 81% (two domains) after 600 experiences (5 seeds), 100% / 90% after 1500
  (3 seeds).
- **Self/Next Slots.** Relations such as "the answer repeats the asked word" or "then comes that word's
  continuation" count as the same relation even when the words differ. Variables arise from words.
- **Resemblance Vote.** Two things resemble each other through shared facets, and a facet is more telling the fewer
  things share it. The members most like a new thing speak loudest, weighted as in exemplar models of categorisation
  (the generalized context model). Seeing and hearing recognise this way: new images 73% / 83% / 67%, new sounds of
  three known synthetic voices 93% / 97% / 93% (chance 33%).
- **Situation Resemblance.** An utterance never lived as a whole is answered with what followed the most similar
  lived situations. If the resemblance is weak, for example because the utterance has a word nothing lived, nothing
  is made up.

## 6. Speaking, thinking, not knowing

- **Continuation Rule.** When nothing in the present situation expects anything above ½, the mind reads the items
  after the last perceived event, and the longest tail of them that has lived continuations speaks, with reduced
  conviction. This rule never starts an utterance, and it leaves out the partner's words still inside the present
  context, so it does not repeat what the partner just said.
- **Advantage Race.** Options on one channel exclude each other. The leader is said if its advantage over the
  strongest rival reaches the threshold; if all options are equal, the mind stays silent.
- **Trying in Hesitation.** When the mind hesitates between well-supported options, it tries the one whose
  consequences it knows least instead of staying silent. It does this only at the start of its turn, or where good or
  bad was already met in this situation. An answer that was called wrong is not tried again next time; another one
  is.
- **Inner Voice.** With `!dusun`, or after ten seconds of silence, the mind thinks about the best topic of what it
  last heard (the most telling and least known word). Its thoughts are lived as inner voice with the imagined origin
  and can shape what it later decides only through what they leave in it; nothing is said for the mind. (Until
  15 November 2026 the program also called thoughts before each reply and said the first real-enough one for the mind,
  "Think, then Speak"; that injection was removed.)
- **Felt Not-Knowing.** Not knowing is a measured inner state, not an output:
  `u = 1 − 2·max(0, best expectation − ½)`. The state is perceived on its own after a silent moment. What one says
  then is learned from the people the mind lives with and from its data; the core holds no sentence for it.
  How to ask is learned in the same way (the coded curiosity reflex that asked back an unknown word with '?' was
  removed on 15 November 2026). A mind that has not lived how one says it stays silent. Hesitating between two strong answers is not not-knowing, and it does not raise the
  feeling.

## 7. Imagination, dreams and reasoning

- **Structural Imagination.** A concept is likened to another that stands in the same places in lived sequences
  ("fire warms the house" / "the sun warms the world": fire ↔ sun). The thought about one is carried over to the
  other. Each step gets a **reality score** read from all evidence except imagined evidence and the mind's own acts.
- **Shared-Stem Inference.** "A cat is an animal" and "animals breathe" are joined through the shared stem into
  "a cat breathes". This is a hypothesis, and it is lived as imagined. On real data a shared stem is sometimes a
  coincidence, so the inference itself runs only in dreams and in the inner voice (`!dusun`), never while answering.
  What it constructs is lived as imagined evidence and, weighted by the learned trust in imagination, stays in the
  evidence answers are read from until life judges it.
- **Dream Sorting.** In sleep the mind dreams about the day's topics and the words it is unsure of.
  - A dream is a chain of inferences and likenings linked by association.
  - An image seen together with a word is "seen" again in the dream.
  - Each dream is kept with its new claims; when life later touches them, the dream is sorted as **true**,
    **false** or **open**.
  - Imagined evidence is weighed against real evidence, and the **trust in its own imagination** follows.
  - Nothing imagined is deleted (only dream records older than the last 500 fade); where real evidence exists, it
    outweighs the imagined through trust.
- **Self Unit.** An innate "self" unit is perceived in every moment the mind speaks or thinks. Introspection reads
  it: how much the mind spoke, how much it thought, and what it binds itself to most.

## 8. Learning to learn, and learning from consequences

- **Voice Credibility.** Every generalisation that could speak for an instance is a "voice". A voice's kind is how
  many members speak and how much they agree. The mind learns from its own outcomes how far each kind of voice can
  be believed, and since this knowledge holds no words, it carries into a new domain. The result is mixed: early
  learning of a new domain sped up in one direction (+0.058 ± 0.025 held-out over a fresh mind; 7 lives better, 2
  equal, 1 worse) but not in the other. Off by default.
- **Learning code from its consequences** (a research test with six tiny functions, three candidate completions each,
  `research/living_mind_2026_11_15/t_code.py`):
  - The mind first watches mostly wrong code completions.
  - Then the code it writes is actually run, and the test result comes back as a good or bad signal.
  - With Trying in Hesitation it writes passing code for all six tasks by round 4.
  - Without the signal it stays at 0%, and without trying at 11%.
  - Writing new code, beyond choosing among completions it watched, is still open.

## 9. Memory and speed

- **One-Origin Row.** 99.6% of ties carry evidence of a single origin. Such a tie stores only its origin and two
  numbers (W+, W−); a full row is given only to ties with more than one origin or with judged amounts (the learned-trust
  bookkeeping). Memory per tie went from 239 to 69 bytes: the core of a mind with 3.25 million ties shrank from 777 MB
  to 223 MB, with bit-identical results.
- **Speed.** The speed work made a moment 2.0–3.8× faster with bit-identical results: 3.2–3.5 ms on a mind with 2.54
  million ties. The One-Origin Row then cost 13–32%: about 3.9–4.4 ms per moment on the mind with 3.25 million ties
  (times are indicative; the machine was shared). Both changes were checked by comparing the substance digest on the
  small public run, in 17 configurations and on the large minds.
- **Open problem.** Memory grows linearly with lived data, at about 11 ties per moment, and the process of the
  3.25-million-tie mind holds about 4 GB, most of it outside the tie evidence. The road to millions of examples, a
  disk-backed store and consolidation in sleep, is planned in the notes.

## 10. Compared with a Transformer

| | Transformer | Cevahir Zihin |
|---|---|---|
| Learning | train, freeze, use | learns at every moment; no separate training |
| Representation | embedding vectors, weight matrices | units, ties, evidence counted by origin |
| Attention | dense softmax over its context window | Seven-Place Focus: local, sparse, readable; no context window |
| Update | backpropagation, gradients | local evidence counting, one rule |
| Forgetting | exact removal needs retraining | Exact Retraction of an experience's evidence while its ledger is kept |
| Not knowing | always a probability distribution | expectation ½: a measured felt not-knowing |
| Sources | not kept per fact | origin and trust of every piece of evidence kept |
| Hardware | large models need GPUs to train; small ones train on a CPU | an ordinary CPU, no training phase; RAM grows with what it lived |

Against small Transformers on the identical stream of experiences, in the small-data, learn-while-living regime (v5;
Zihin in its "members" configuration, two small generated worlds, 5 seeds, 6–8 held-out probes per life):

- On instances never told the asked dimension, Zihin reached 90% / 97% and 75% / 95%. A Transformer trained online in
  one pass reached 67% / 57% and 60% / 50%; one trained to convergence reached 77% / 67% and 53% / 53%.
- A second domain lowered the one-pass Transformer's accuracy on the first domain from 0.85 to 0.74; in Zihin it
  stayed at 1.00. The converged Transformer was not tested for this.
- The configuration the chat program runs ("speaking") was not in that comparison. In the same worlds it does poorly
  (held-out 0.55 and 0.04 after 600 experiences); it also speaks into the stream, which these worlds were not made
  for, and this was not analysed further.

The models were small and the data scarce; with far more data the gap may close.

## 11. What we do not claim

- This is not a claim of consciousness or of general intelligence; neither can be measured for any machine today.
- We do not claim novelty for every part: the one rule is the NARS expectation, the Resemblance Vote follows
  exemplar models of categorisation, and the focus is a k-winners-take-all competition. The names below are the
  names this theory uses.
- Zihin is not a fluent language model. On real data its conversation comes largely from what it lived and from
  imitation.
- Negative results are kept, with their measurements in the notes:
  - the first attempt at learning to learn was negative;
  - a mind raised with word pieces from birth ran out of memory;
  - the one-shot evidence fix made conversation worse;
  - the chat configuration does poorly in the generated worlds where the "members" configuration does well;
  - until 15 November 2026 the chat program held scripted lessons about the mind and about saying "bilmiyorum",
    written by the developer; they were removed from the core at the owner's decision. Minds raised before lived
    them, and the measurements that used them say so.

## Names used in this theory

| Name | Turkish | Note | Related prior work |
|---|---|---|---|
| Living Evidence Mind | Yaşayan Kanıt Zihni | the architecture | |
| Provenance Evidence | Köken Kanıtı | W+/W− per six origins | source monitoring in psychology |
| Conviction-weighted Expectation | Kanaat Beklentisi | e = c(f − ½) + ½, the one rule | the NARS truth-value expectation |
| Trust Judgment | Güven Yargısı | trust in the four indirect origins learned from direct experience | |
| Seven-Place Focus / Focus Settling | Yedili Odak / Odak Yerleşmesi | the attention mechanism | k-winners-take-all |
| Exact Retraction | Kesin Unutma | an experience's evidence removed exactly through the ledger (named in v5, T9) | exact machine unlearning |
| Members' Reading | Üye Okuması | F11 | |
| Self/Next Slots | Kendi/Sonraki Yuvaları | F12 | |
| Present-Only Learning | Şimdiden Öğrenme | F13 | |
| Expected-Failure Rule | Beklenen Başarısızlık | F14 | |
| Consequence Trace | Sonuç İzi | F15, L5 | eligibility traces |
| Continuation Rule | Sürdürme Kuralı | F16 | |
| Resemblance Vote | Benzerlik Oyu | F18 | exemplar models (the generalized context model) |
| Advantage Race | Avantaj Yarışı | F19 | |
| Structural Imagination | Yapısal Hayal | F20, a demonstration | |
| Self Unit | Benlik Birimi | F21, a demonstration | |
| Dream Sorting | Rüya Ayıklama | F22 | |
| Situation Resemblance | Durum Benzeşimi | F23 | |
| Voice Credibility | Ses Güvenilirliği | F24, off by default | |
| Born Failures | Doğuş Başarısızlıkları | F25, studied, off by default | |
| Felt Not-Knowing | Bilmeme Hissi | L7 | feeling of knowing (metacognition) |
| Inner Voice | İç Ses | thinking lived as imagined (the earlier "Think, then Speak" injection was removed) | |
| Trying in Hesitation | Kararsızlıkta Deneme | exploration | |
| Shared-Stem Inference | Ortak Kök Çıkarımı | T8, second form | |
| Online Life | Çevrimiçi Yaşam | lives in its own time during a session | |
| One-Origin Row | Tek Köken Satırı | memory layout | |

## Running it on your computer

The model learns in every conversation, so please speak to it kindly: everything you say becomes part of its life.

It needs Python 3.11 or newer. Linux / macOS:

```
git clone https://github.com/myylogic/cevahir-ai
cd cevahir-ai
python3 -m venv .venv && . .venv/bin/activate
pip install torch numpy           # optional: python-docx (Word), pypdf (PDF), pillow (images)
python -m zihin
```

Windows PowerShell:

```
git clone https://github.com/myylogic/cevahir-ai
cd cevahir-ai
pip install torch numpy
.\cevahir.ps1                     # if scripts are blocked: powershell -ExecutionPolicy Bypass -File .\cevahir.ps1
```

Start it inside the `cevahir-ai` folder. The program speaks Turkish: it first asks "Seninle konuşan kim?" (who is
talking to you?; type your name). On first start the mind is empty and knows no language; it learns by living with
you. Commands are typed at the program's prompt, not in the shell; each has an English alias.

```
!oku my_documents/                # !read: .txt .md .docx .pdf documents, .json/.tsv dialogues, source code
!ogret dialogues.tsv              # !teach: each line question<TAB>answer, saved as UTF-8
!izle hello || hi                 # !watch: watch an exchange
!dogru  /  !yanlis                # !good / !bad: how you received its last answer
!uyu    !dusun <topic>   !kendin  # !sleep, !think, !introspect: dream, inner voice, introspection
!cik                              # !quit
```

While it is open it lives in its own time: it lives silent moments, thinks inwardly after ten seconds of silence and
sleeps after ten minutes (`--yasamsiz` turns this off); it says something only when its own decision produces words. The mind is saved at the end of every session to `~/.cevahir/zihin/cevahir.zihin`
(Windows: `%USERPROFILE%\.cevahir\zihin\cevahir.zihin`; `CEVAHIR_HOME` changes the folder; `--kaydetme` does not
save).

The minds raised so far lived the author's private Turkish data and are not in this repository. Your copy learns the
language you teach it with `!ogret` and `!oku`. For a language other than Turkish, start a new mind with `--dil diger`
so that capital letters are lowered without Turkish rules (a mind keeps the rule it was raised with). After a dozen
pairs it answers the questions it was shown; how much data a new language needs for questions it has not seen has
not been measured. A mind file carries what it has read, so do not share a mind raised on private documents. Full
guide (Turkish): [ZIHIN_KILAVUZ.md](ZIHIN_KILAVUZ.md).
