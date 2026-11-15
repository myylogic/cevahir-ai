# The Cevahir Zihin Theory: a Living Evidence Mind

*Muhammed Yasin Yılmaz, November 2026. This text describes the theory and architecture of the `zihin/` core in this
repository. Measurements and negative results for every claim are in the notes and result files under
`research/living_mind_2026_11_14/` and `research/living_mind_2026_11_15/`. Turkish original:
[ZIHIN_KURAM.md](ZIHIN_KURAM.md).*

## In short

Cevahir Zihin is not a Transformer. It has no neural network, weight matrices, embedding vectors, backpropagation or
softmax attention. It is a **Living Evidence Mind**:

- whatever it perceives becomes a unit, and every relation between units is a tie;
- each tie counts how often the relation was confirmed and how often it was refuted, **separately by where the
  evidence came from**: seen with its own eyes, told, imitated, its own act, inferred or imagined.

It reads what to expect from a single rule, learns at every moment, and can retract any single experience exactly.

Training and use are not separate phases: **to learn is to live**. The model learns in every conversation.

## 1. Basic entities

**Channels** are how the mind touches the world and itself:

- `sense`: heard or seen;
- `act`: said or done;
- `feel`: surprise, relief, conflict, load;
- `knob`: its own settings;
- `source`: where evidence comes from;
- `facet`: how a thing looks or sounds, or a piece of a word;
- `value`: a good or bad outcome;
- `context`: who speaks, tone, the self.

**Units** come in three kinds:

- an atom: a symbol on a channel, such as a word, an image digest or a turn end;
- a composed unit: a lived sequence, a window of up to seven items;
- a category: a class of units lived in the same places.

**Ties** relate two units:

- `EXPECT` with lag 0 (together), lag 1 (right after) or lag 2 (later in the same episode);
- `PART`: part of a sequence;
- `ABSTRACT`: member of a category;
- `RELY`: how reliable a generalisation has been.

**Provenance Evidence.** Each tie keeps evidence for (W+) and against (W−) separately for six origins: *sensed, told,
imitated, acted, inferred, imagined*. Where knowledge came from is never lost.

## 2. One rule: Conviction-weighted Expectation

How much a tie is expected is read from its evidence, weighted by the trust in each origin:

- frequency `f = W+ / W`;
- conviction `c = W / (W + k)`, with `k = 1`;
- expectation `e = c · (f − ½) + ½`.

With no evidence the expectation is ½, which means "I don't know". Conviction grows with evidence. Every reading
rests on this rule: what to say, what to expect, what to be surprised by.

**Trust Judgment.** The trust in each origin is not fixed. Sensed and acted evidence count as direct experience.
Told, imitated, inferred and imagined evidence are compared with the direct experience that comes later; what holds
raises the trust in that origin, and what fails lowers it. So the mind learns by itself how far to believe whom.

## 3. Attention: the Seven-Place Focus

In a Transformer, attention lets each token look at all earlier tokens through learned vectors and a softmax. It is
dense and grows with the length of the past. Cevahir Zihin attends by a different mechanism.

**Seven-Place Focus, or Focus Settling.** At each moment only seven units enter the focus (k-winners-take-all,
F = 7). The candidates are:

- what is perceived now;
- what is carried over from earlier moments, fading by `1 − 1/F` per moment;
- the recognised composed context, meaning the lived sequence of the episode;
- their categories and frames, which are the generalisations;
- what lived togetherness calls to mind.

Settling takes at most six rounds: association spreads, winners are chosen again, and it stops when nothing changes.
What is present comes first, then the most specific generalisation of each place.

Consequences:

- the cost of a moment depends on local ties, not on the length of the past;
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
   - **Expected-Failure Rule:** what did not happen receives evidence against it only if it was expected.
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
9. **Ledger.** Every change of the moment is written to a ledger. This makes **Exact Retraction** possible: a single
   experience is removed exactly, without disturbing what was lived after it.

## 5. Generalisation: categories speak with their members

- **Members' Reading.** A category reads what to expect for an instance from what its members themselves lived, not
  from rules of its own, and the instance being read is left out of the vote. Generalisation to never-seen instances
  improved with age: 93% / 81% after 600 experiences, 100% / 90% after 1500.
- **Self/Next Slots.** Relations such as "the answer repeats the asked word" or "then comes that word's
  continuation" count as the same relation even when the words differ. Variables arise from words.
- **Resemblance Vote.** Two things resemble each other through shared facets, and a facet is more telling the fewer
  things share it. The members most like a new thing speak loudest. Seeing and hearing recognise this way: new
  images 73% / 83% / 67%, new sounds 93% / 97% / 93% (chance 33%).
- **Situation Resemblance.** An utterance never lived as a whole is answered with what followed the most similar
  lived situations. If the resemblance is weak, for example because the utterance has a word nothing lived, nothing
  is made up.

## 6. Speaking, thinking, not knowing

- **Continuation Rule.** An utterance under way continues along its own lived sequence. This rule never starts an
  utterance and never echoes the partner.
- **Advantage Race.** Options on one channel exclude each other. The leader is said if its advantage over the
  strongest rival reaches the threshold; if all options are equal, the mind stays silent.
- **Trying in Hesitation.** When the mind hesitates between well-supported options, it tries the one whose
  consequences it knows least instead of staying silent. It does this only at the start of its turn, or where good or
  bad was already met in this situation. An answer that was called wrong is not tried again next time; another one
  is.
- **Think, then Speak.** The least certain words of what it heard call its knowledge to mind. These thoughts are
  lived as inner voice with the imagined origin. If it has no answer at once, it says the first thought that is real
  enough by lived evidence.
- **Felt Not-Knowing.** Not knowing is a measured inner state, not an output:
  `u = 1 − 2·max(0, best expectation − ½)`. The state is perceived on its own after a silent moment. What one says
  then, such as "bilmiyorum , öğrenmek isterim ." ("I don't know, I'd like to learn."), is learned by living, and
  **curiosity** then asks back the unknown word with a question tone. Hesitating between two strong answers is not
  not-knowing, and it does not raise the feeling.

## 7. Imagination, dreams and reasoning

- **Structural Imagination.** A concept is likened to another that stands in the same places in lived sequences
  ("fire warms the house" / "the sun warms the world": fire ↔ sun). The thought about one is carried over to the
  other. Each step gets a **reality score** read from all evidence except imagination.
- **Shared-Stem Inference.** "A cat is an animal" and "animals breathe" are joined through the shared stem into
  "a cat breathes". This is a hypothesis, and it is lived as imagined. On real data a shared stem is sometimes a
  coincidence, so inference is used only in dreams and inner voice, never in answers.
- **Dream Sorting.** In sleep the mind dreams about the day's topics and the words it is unsure of.
  - A dream is a chain of inferences and likenings linked by association.
  - An image seen together with a word is "seen" again in the dream.
  - Each dream is kept with its new claims; when life later touches them, the dream is sorted as **true**,
    **false** or **open**.
  - Imagined evidence is weighed against real evidence, and the **trust in its own imagination** follows.
  - No dream is deleted; real evidence outweighs it.
- **Self Unit.** An innate "self" unit is perceived in every moment the mind speaks or thinks. Introspection reads
  it: how much the mind spoke, how much it thought, and what it binds itself to most.

## 8. Learning to learn, and learning from consequences

- **Voice Credibility.** Every generalisation that could speak for an instance is a "voice". A voice's kind is how
  many members speak and how much they agree. The mind learns from its own outcomes how far each kind of voice can
  be believed, and since this knowledge holds no words, it carries into a new domain. The result is mixed: early
  learning of a new domain sped up in one direction (+0.055, better in 7 of 10 lives) but not in the other. Off by
  default.
- **Learning code from its consequences.**
  - The mind first watches mostly wrong code completions.
  - Then the code it writes is actually run, and the test result comes back as a good or bad signal.
  - With Trying in Hesitation it writes passing code for all six small tasks by round 4.
  - Without the signal it stays at 0%, and without trying at 11%.

## 9. Memory and speed

- **One-Origin Row.** 99.6% of ties carry evidence of a single origin. Such a tie stores only its origin and two
  numbers (W+, W−), and a full row is given only to ties with several origins. Memory per tie went from 239 to 69
  bytes: the core of a mind with 3.25 million ties shrank from 777 MB to 223 MB, with bit-identical results.
- **Speed.** A moment takes about 3–4 ms on a mind with 2.5 million ties. Measurement tools check on every change
  that results stay bit-identical.
- **Open problem.** Memory grows linearly with lived data, at about 11 ties per moment. The road to millions of
  examples, a disk-backed store and consolidation in sleep, is planned in the notes.

## 10. Compared with a Transformer

| | Transformer | Cevahir Zihin |
|---|---|---|
| Learning | train, freeze, use | learns at every moment; no separate training |
| Representation | embedding vectors, weight matrices | units, ties, evidence counted by origin |
| Attention | dense softmax over the whole past | Seven-Place Focus: local, sparse, readable |
| Update | backpropagation, gradients | local evidence counting, one rule |
| Forgetting | retraining | Exact Retraction: one experience removed exactly |
| Not knowing | always a probability distribution | expectation ½: a measured felt not-knowing |
| Sources | unknown | origin and trust of every piece of evidence kept |
| Hardware | GPU | an ordinary CPU |

Against small Transformers on the identical stream of experiences, in the small-data, learn-while-living regime:

- On never-seen instances Zihin reached 90% / 97% and 75% / 95%. A Transformer trained online in one pass reached
  67% / 57% and 60% / 50%; one trained to convergence reached 77% / 67% and 53% / 53%.
- A second domain lowered the Transformer's knowledge of the first from 0.85 to 0.74; in Zihin it stayed at 1.00.

The models were small and the data scarce; with far more data the gap may close.

## 11. What we do not claim

- This is not a claim of consciousness or of general intelligence; neither can be measured for any machine today.
- Zihin is not a fluent language model. On real data its conversation comes largely from what it lived and from
  imitation.
- Negative results are kept, with their measurements in the notes:
  - the first attempt at learning to learn was negative;
  - a mind raised with word pieces from birth ran out of memory;
  - the one-shot evidence fix made conversation worse.

## Named findings

| Name | Turkish | Note |
|---|---|---|
| Living Evidence Mind | Yaşayan Kanıt Zihni | the architecture |
| Provenance Evidence | Köken Kanıtı | W+/W− per six origins |
| Conviction-weighted Expectation | Kanaat Beklentisi | e = c(f − ½) + ½, the one rule |
| Trust Judgment | Güven Yargısı | trust in origins learned from direct experience |
| Seven-Place Focus / Focus Settling | Yedili Odak / Odak Yerleşmesi | the attention mechanism |
| Exact Retraction | Kesin Unutma | a single experience removed exactly through the ledger |
| Members' Reading | Üye Okuması | F11 |
| Self/Next Slots | Kendi/Sonraki Yuvaları | F12 |
| Present-Only Learning | Şimdiden Öğrenme | F13 |
| Expected-Failure Rule | Beklenen Başarısızlık | F14 |
| Consequence Trace | Sonuç İzi | F15, L5 |
| Continuation Rule | Sürdürme Kuralı | F16 |
| Resemblance Vote | Benzerlik Oyu | F18 |
| Advantage Race | Avantaj Yarışı | F19 |
| Structural Imagination | Yapısal Hayal | F20 |
| Self Unit | Benlik Birimi | F21 |
| Dream Sorting | Rüya Ayıklama | F22 |
| Situation Resemblance | Durum Benzeşimi | F23 |
| Voice Credibility | Ses Güvenilirliği | F24, off by default |
| Born Failures | Doğuş Başarısızlıkları | F25, studied, off by default |
| Felt Not-Knowing | Bilmeme Hissi | L7 |
| Think, then Speak | Önce Düşün | T8 in conversation |
| Trying in Hesitation | Kararsızlıkta Deneme | exploration |
| Shared-Stem Inference | Ortak Kök Çıkarımı | T8, second form |
| Online Life | Çevrimiçi Yaşam | lives in its own time during a session |
| One-Origin Row | Tek Köken Satırı | memory layout |

## Running it on your computer

The model learns in every conversation, so please speak to it kindly: everything you say becomes part of its life.

```
git clone https://github.com/myylogic/cevahir-ai
cd cevahir-ai
pip install torch numpy           # optional: python-docx (Word), pillow (images)
python -m zihin                   # Linux / macOS
.\cevahir.ps1                     # Windows PowerShell
```

On first start the mind is empty and learns by living with you. To teach it:

```
!oku my_documents/                # .txt .md .docx .pdf documents, .json/.tsv dialogues, source code
!ogret dialogues.tsv              # each line: question<TAB>answer
!izle hello || hi                 # watch an exchange
!dogru  /  !yanlis                # how you received its last answer (good / bad)
!uyu    !dusun <topic>   !kendin  # sleep and dream, inner voice, introspection
```

Cevahir was raised on Turkish data. To speak another language with it, let it live a small dataset in that language
(a few hundred question-answer pairs or texts) with `!ogret` or `!oku`. A mind file carries what it has read, so do
not share a mind raised on private documents. Full guide (Turkish): [ZIHIN_KILAVUZ.md](ZIHIN_KILAVUZ.md).
