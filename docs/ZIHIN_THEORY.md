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
- **Exact Relations (F28).** A relation counts for a member only where it names the very value the member lived,
  resolved as it would be for the instance being read. Of the relations that name the same value only the most
  specific counts, and the value itself always counts. Before, a member's value that was merely one of the words it
  goes on with counted as "next" and never as the value. In a world where the value follows the asked thing directly
  ("<relation> <entity> <value>"), members that had all lived the same value were read as agreeing on "next", which
  pointed at the asked thing's own habits. Held-out answers in that world went from 3% to 81%.
- **Analogy by Interchangeability (F29).** Two words are interchangeable as far as they filled the same lived places
  before the utterance being read; a place shared by few is more telling than one shared by everything. What a
  category's members lived carries over to a new instance through its interchangeability with them (the analogy rule
  of NARS); the members still decide what is claimed, the analogy only how firmly it holds. "selam , adın ne ?" is now
  answered like the lived "merhaba , adın ne ?", and "ismin ne ?" like "adın ne ?" where adın and ismin filled the same
  places in what was read; a word never met carries nothing over.
- **Corroboration (F30).** Independent lived evidence about the same claim adds up (the revision rule of NARS): the
  claims of the frame that speaks are revised with what the most similar lived situations say about the same values,
  each lived context counted once. "peki adın ne ?" is answered because the frame of questions with a word before
  "adın ne" and the lived bare question agree; where similar situations disagree, nothing is added.
- **Paradigmatic Relation (F31) and Relational Binding (F32).** A frame's value may stand where its filler stands:
  the item most interchangeable with the filler (not a rival filler of the frame's own place) is a relation of the
  value to the filler, as identity and continuation are; and a place binds the filler's stand-in as it binds its
  repeats. From four definitions ("otomobil ne demek ?" -> "otomobil , araba demektir .") and the places words filled
  in what was read, "selam ne demek ?" is answered "selam , merhaba demektir". This is the paradigmatic (vertical)
  axis of language beside the syntagmatic one (Saussure, Jakobson).
- **Relational Memory (F33).** One mind, one memory, recalled in the company of the person present. What is lived
  while someone the mind knows is present is also noted with them. Recall then depends on who is there: what was
  lived with no one in particular is recalled with anyone, what was lived with the person present is theirs and comes
  last, and what was lived with others is recalled only as far as other people corroborate it (one person's alone is
  theirs). Each person is answered from their own history, a secret lived with one person is told to no one else, and
  what everyone taught is known with a stranger; no role or relation is coded.
- **Reply Relation (F34).** A frame's value may be what its filler, said on its own, was answered with, and several
  places after a prefix may be bound to the filler (its stand-in, its reply, or an item every member has there); a
  frame's claim is corroborated by the lived suffixes of its context that keep every fixed word of the frame (a
  suffix that leaves one out is another context: "walk ⏎" of "jump left and walk ⏎"). On SCAN's "add primitive" split, made small for
  the mind's window, a primitive seen only alone ("jump" -> "i_jump") is used in every template the first time it is
  asked ("jump twice", "walk and jump", "jump left twice"): 75 of 75 commands, where a Transformer trained from
  scratch on the same data, exact on all its training commands, answers 4. On SCAN's own grammar within a window of
  11, with the bound frames of every prefix read: 789 of 844 (the Transformer 0), every command that fits the window
  whole answered exactly; with a window of 12, all 844 (with a window of 13, all 1,188 that fit 12 items).
- **Generalised Consequences (F35).** What an act would lead to (good, bad) is read from what similar paths led to,
  through the frames of the path, and the path's own record comes last. A mind that learned from how its answers were
  received in a few situations acts the same way in situations it never met (polite to things never handed to it: 0
  -> 8 of 9, against the habit it watched); with F33 what pleased one person is an expectation of that relationship
  (each of two people with opposite tastes answered their way, 9 of 9).
- **Word Forms (F36).** The layer below words: a word never met is read as the lived word it shares its longest
  stem with, the stem being a boundary of the lived words themselves (Harris's successor variety), carried over by
  how much of the word the stem covers. "merhabalar" is greeted as "merhaba", "nasılsınız ?" answered as "nasılsın ?";
  a beginning one lived word has by chance is not a stem.
- **Inheritance (F37, off by default).** Deduction with exceptions, read from the places a word filled: the class of
  a place is named by its frame's most telling fixed word, and the class spoken of on its own is a lived word sharing
  a stem with it ("kuşlar" for "kuştur"); what that word goes on with is carried over to the member by the deduction
  of NARS (f = f1 f2, c = f1 f2 c1 c2 f_n), and the member's own record, what its class-mates mostly did not go on
  with, comes last. Facts read apart ("kartal bir kuştur .", "kuşlar uçar .") give a conclusion never taught
  ("kartal uçar"), and an exception stated of itself ("penguen yüzer") wins over its class: 48 of 48 in T-deduce
  against 2, no wrong generalisation. A class word inherits in turn from the classes it is said to be of (what it goes
  on with where it is said to be of a class, "bir" of "balıklar bir hayvandır .", is the form of the saying, not a
  property), so a conclusion two steps away is reached: T-chain 27 of 27 ("levrek solur" from "levrek bir balıktır .",
  "balıklar bir hayvandır .", "hayvanlar solur ."), none without F37 or with one step; the nearer property 9 of 9, the
  exception 3 of 3. Off by default: on the largest mind an answer takes about eleven times as long.
- **Witness Reliability (F38, off by default).** A shorter lived context that leaves out a fixed word of a frame's claim is a witness
  of the claim as far as the mind has seen it agree: the lived contexts that end with the left-out word and the
  shorter one went on as the shorter one did. What the witness says is carried by the deduction of NARS (f = f1 f2,
  c = f1 f2 c1 c2), so it says nothing of a value it never went on with. "adın ne ?" witnesses "selamlar , adın ne ?"
  (what ended with ", adın ne ?" went on as "adın ne ?" did); "look thrice" does not witness "jump thrice and look
  thrice". The reliability of a source, measured on the mind's own lived contexts, for the part of a context a
  reading leaves out. Off by default: answering with it loses nothing, but a SCAN life lived with it learned
  otherwise (789 → 767).
- **Replacement Witness (F39, off by default).** Mill's method of difference, read from the mind's own life: when
  nothing would be said, a lived situation that differs from the present one by one replacement is a witness of it as
  far as the lived pairs that differ by the same replacement went on alike (the replacement made no difference where
  it was lived). With clouds the common cause of the umbrella advice and the rain, the words alone lead the mind to
  rain on a sunny day with the advice (0.556); with F39 the fine weather leads (0.577), because the two advices went
  on alike on cloudy days while the two skies with the same advice did not. It leans, it does not yet assert. Lived
  with it, SCAN within 11 keeps its 789 answers, the same ones.
- **Urge to Go On (F40, off by default).** Two expected words are either alternatives (only one can be right) or two
  things known about the same item (both are right; they compete only for the order). Their share of an item's lived
  continuations says how far: the leader then races against silence instead of against its rival. A first step to
  long answers: more facts said about things never asked, no more confabulated (T-long 20 → 22 of 54: two sentences
  that stopped half way finished; confabulated 4 and 4).
- **Unsaid Known (F41, off by default).** What is known of a thing is how the lived sequences that began with it go
  on: not what was only asked of it (a sequence that another turn answered is a question), and not what the present
  episode has already said. A frame's value may be such a thing: the members of a place went on with things known of
  themselves, and of the read filler the known thing most like what they said is claimed (two words are alike as far
  as they go on alike, as words or as the lived sequences they carry on; so a question about one kind of fact is
  answered with the thing's own fact of that kind). When such a claim is acted on, its filler is the topic of the
  answer. Where no word would be said, the lived contexts of the place's other members in which their own known
  sequence stood the same way say how to go on (the words between facts), each read as an answer about its own
  filler. The urge to go on is the unsaid share of what is known of the topic: nothing is said when it is 0, and
  while it is above 0 the members' ends of turn count only beyond their goings-on (an answer that ends for having
  nothing left has gone on at least once for every end); these links of the urge to speech are a model, not learned.
  Asked about things never asked about, the mind tells their facts read apart, whole and no more: T-long 20 → 54 of
  54 in the watched order and 0 → 54 of 54 in any order (on seeds never looked at, 108 of 108 and 107 of 108, no
  sentence unfinished or confabulated). In a world it was not developed on (T-tell: two to four facts, three kinds
  of question lived by one mind), 2 → 104 of 113 facts for things never in a dialogue, and the place alone when asked
  where (1 → 24 of 31). On the owner's documents the teacher protocol rises from 120 to 130 exact. A SCAN life lived with it answers 810 of the 844 commands within 11 (789 without): 21 answers that ran on now end, and none is lost.
  Off by default: recall with a two-word cue loses one answer (90 → 89), a question about a kind of fact that is not
  known of the thing is more often answered with another fact of it, and an answer on the largest mind takes about 2.4 times as long (453 s against 190 s for the teacher questions).
- **Episode Echo (F42).** What already followed an item in the present episode, heard or said, is an echo, not
  knowledge of what follows it now: it is not read as the lived continuation of the last word (F16 held this for the
  partner's words only), nor as how a frame's filler goes on (NEXT, START). The rule was part of F41 and gave it its
  SCAN answers; alone it gives the same (SCAN within 11: 789 → 810 of 844, answers that ran on now end) and changes
  nothing else that is measured (T-compose, meaning, relations, T-purpose, T-cause, T-deduce, T-chain, the listening
  worlds, T-tell, the teacher protocol and recall on the owner's documents; T-long 66 → 68 of 162). On in the speaking
  configuration.
- **Wholes (F43, off by default).** The mind decides from its last F items; an answer longer than that is said with
  the question out of sight. What holds more is not a longer window but larger items. A sequence that was said alone
  and answered is a whole, and every watched dialogue is lived once more as two items, the utterance and its answer,
  so that a whole's reply is one lived continuation however long. When the answer of a watched dialogue is made of the
  replies of wholes that stand in its utterance, the dialogue is lived once more in summary: each such whole as its
  form (its own summary with the wholes inside it as bare wholes; a whole with no whole inside is a bare whole), each
  reply as a pointer to the place of its whole. Of the ways to read an answer so, the fewest words left unexplained,
  then the fewest items; when two tie, nothing is concluded; and a summary must fit the window. A summary is also
  lived as its construction (its words as they are, every whole bare). An utterance never answered is heard in the
  summary that was lived and answered as it is; when none was, in the one whose construction was, every item of it
  lived before (what was never lived as an item is not known to stand as one). The forms say which reading of an
  utterance is its summary ("A after (B twice)", not "(A after B) twice"); what is answered to a construction does
  not depend on them. A part is read as an utterance is, so a part never said alone is a whole by its form. When the
  mind decides a pointer it sets its working state aside, answers that part as if it had been said alone (remembered
  whole if it was answered before; in its own summary if it is new), and takes its working state up again; the echo
  of all it said meanwhile is one item. The window bounds each level, not the answer. A summary is another level and
  is lived on a channel of its own, the words that stand in it too, so what the mind reads word by word is what it was
  (on the words' own channel the summaries entered the frames of plain words, and one world broke). SCAN's add-jump
  split with commands of every length (T-scan; 7,706 test commands, answers of up to 48 actions): after watching
  1,082 of SCAN's 13,204 training dialogues twice (six and a half minutes of life on one core) the mind answered
  **7,706 of the 7,706** exactly, with each of the three forms F43 has had (the present one included). With **92
  dialogues** (0.7% of the training set: the 81 commands without a connective, "jump", ten joined commands; ten
  seconds of life) it answers **500 of a sample of 500**. What it needs
  is each construction once and each form once as an item: with two joined commands that hold both connectives and
  both repeating words, 500 of 500; with two that both say "after", the 259 that do not say "and" (the rest silent).
  A Transformer trained from scratch on the 1,082 dialogues answers 1 of 7,706 (298 of 300 of its own training
  commands). SCAN's length split (training answers of at most 22 actions, test 24 to 48; never looked at before): as
  first found **101 of 500** (182 silent, 217 wrong: a part of 24 actions was never said alone, and only the
  utterance being answered could be read by what had not been lived as it is); with parts read as utterances are,
  **500 of 500**, after 1,094 of the 16,990 training dialogues and also after 104 of them; a Transformer on the 1,094
  dialogues 0 of 500. In a world it was not developed on (T-görev: jobs of two words made of
  two or three steps, joined and repeated by Turkish words that stand before, around and between them; the new job
  watched only alone): 492 of 492 over three seeds (0 without F43); as first found, before one line was changed, 51
  of 164. An earlier step is gone: summaries never lived used to be considered (brought to mind in imagination, what
  the frames would answer read, the moments undone); once parts were read by their construction it was reached 236
  times in the battery and decided nothing, and it had been what kept the mind silent with few dialogues (367 of 500
  at 92 dialogues with it). What F43 does not do: an answer that changes a part other than word for word (F45) has
  no summary; summaries are learned from watched dialogues only; the parts' own questions are imagined (what a long
  life of such answers leaves was not measured); a whole does not say of what kind it is; and an answer of the full
  run takes about 1.8 s on one core, nearly all of it the mind's own decisions at every moment of every part.
- **Lived Exclusion across channels (F44).** F27 counts what an instance was lived to go on with on a channel against
  what it never went on with on that channel. The end of a turn and a word are on two channels, so an instance that
  always went on with the end of the turn said nothing against a word a sibling frame proposed there, and a lived
  summary did not stop ("jump twice" said as three jumps). But an instance's record holds all that followed it: what
  is not in it followed in none of its lived moments, on any channel. The largest of the instance's positive masses on
  a world channel counts against every world target it never went on with. Alone it changes one thing in the whole
  battery: in the relations world a known person's greeting no longer runs on into the greeting of no one in
  particular (0.75 → 0.00 of the probes; her own greeting 1.00 as before); SCAN within 11, the teacher protocol and
  recall are what they were. F43 needs it where a construction has a sibling that goes on: without it T-görev falls
  from 492 to 315 of 492 ("iki kere kahve yap" said three times) and T-compose from 75 to 67 of 75. On in the speaking
  configuration.
- **Open Claims (F46, off by default).** Wanting to learn what it found it did not know. A frame that speaks claims a
  relation of its filler's own record: "what follows is how the sequence that begins with the filler went on" (what it
  was answered with, what stands where it stands, what is known of it). For every member the relation named
  something; when for the present filler it names nothing, the mind's own evidence says that something follows here
  and that of this item it has nothing: a known unknown, measured, not declared. Asked "leylek nedir ?" about a thing
  it never met, the mind says "leylek" and stops; that moment's claim (its expectation e over 1/2) is written as
  evidence on a channel of the mind's own, W+ of the tie (the item there -> holds), and W- when a claim about that item
  is later met. What the mind looks up first (`!kendini_gelistir`) are the items whose claims are left open by that
  evidence, before what it is merely unsure of (the older measure, tau x drive, named "bir" and class words: the
  uncertainty of a function word is not a gap). T-want (generated; a library stands where the web stands, the same
  interface): of the 20 topics the mind wants most after being asked about four things it never met (5 seeds),
  **20 are those things** (0 without F46); it looks them up, and asked again answers **20 of 20** (0 of 20 when nothing
  is looked up, 0 of 20 when it reads as much about other things); asked once more for real, the claims are met and
  none of the 20 is wanted any longer. On a second question form it was not developed on ("X nerede bulunur ?"), as
  found: 20 of 20 wanted, 20 of 20 answered with the fact's own words (never the whole sentence: the mind does not
  repeat "bulunur", a word of the question, there, for things it read about as well). With the switch on, T-compose,
  the meaning world, relations, T-purpose, T-deduce, T-chain, T-long, T-tell and SCAN's add-jump are what they were.
  What it is not yet: the mind does not decide by itself *when* to look something up (the owner starts it), it wants
  only what a frame of its own asked for (not what it has never been asked), and the library of the test is clean.
- **Looking Up by Itself (F47, off by default; with F46).** When to look something up is the mind's own decision.
  In a silent moment, for each item a claim was left open for, as for any act: rate = the drive + what this led to
  before = (2 e_open - 1) + (2 e_look - 1), where e_open is how firmly the claim is open (F46) and e_look what
  looking that item up has led to (its own record over the record of every look; 1/2 while nothing was ever looked
  up, so the first look is driven by the want alone). It acts when ceil(theta / rate) <= T: the line it speaks at.
  The reading is carried out by the session, only when the owner lets it (`--ac look`; `!bakma`, `!bak`), from the
  sources the owner gave (a folder of the owner's documents, `--kaynak`, before the web; `--agsiz`: never the web),
  every page printed. Having read, the mind reads the claim again where it was left open: that is the outcome of the
  act. Met: the want is over and looking gains evidence. Still open: evidence against looking that item up, and
  against looking in general, so a mind whose looks lead to nothing stops by its own evidence, not by a counter.
  T-want, left to itself (5 seeds; nobody tells it to look): before it was asked anything, 0 looks; after four
  questions it could not answer, **20 looks, all 20 of those things and nothing else**; the claim met by what it
  read 20 of 20; asked again **20 of 20**; no look in the silence that follows. A library that holds nothing: one
  look per mind, then none (the wants stay). A library that holds half: each thing once, one failed thing once
  more, then none; 10 of 20 answered. What it is not: the clock of thinking and sleeping is still the program's;
  the mind looks up single words; and it reads text only (it cannot yet learn from what it hears or sees).
- **Thinking by Itself (F48; the online life uses it, `--saatli` gives the program's clock back).** When to think
  is the mind's own decision. What it takes in leaves its surprise over those moments (0.26 for what it expected,
  1/2 exactly for words it never met, 0.60 when a thing it knows is said to be otherwise), and as for any act:
  rate = (2 surprise - 1) + (2 e_think - 1) against the line it speaks at, e_think what its thoughts have come to
  (their reality scores; a thought in which nothing was said counts against). It thinks about the item its
  expectation was of where it failed most, once. Told 24 sentences of each kind and left in silence: it thought
  about 0 of the expected ones, **24 of 24 of those that went against what it knew** (bringing back what it knew:
  "çam: yeşil bir ağaçtır"), 15 of 24 of the neutral ones once thinking had paid, where the program's clock thinks
  after every sentence. Chosen by the older topic measure the thought was about "bir", said nothing, and the mind
  learned from that one empty thought never to think again: found, and fixed by thinking about what failed. **F47,
  stricter:** a claim is met only when the mind, brought back to the situation, says what follows and stops where
  the end of a turn is expected; a first word is not knowing ("bakır ," no longer counts), the claim stays open and
  the way of reading is judged by it. Sleep is still on the program's clock.
- **Fault Tracing (F49, off by default; `zihin/hata.py`).** The core writes a failed expectation on the tie that
  expected. But most of what the mind says of a thing it never lived is a *reading* (a frame's members and the kind
  of their claim, a continuation, a form, similar situations), and a reading has no tie: when it fails, nothing is
  written, the one instance is corrected and the next is read as before. Here every way of reading has ties of its
  own on the mind's own channel and is judged as everything is, by what was then lived. The chain is kept where it
  is computed (which way made each claim, how strongly). What each way claimed where the mind was free to speak is
  kept with its context, said or not (rivals that kept each other silent are all kept). When that context is lived
  again and goes on from outside, each claim is settled: (way → holds) W+ or W−, (way → claimed) by the strength
  it was made with. So the break is found by living: the words before it are confirmed, the word at it is the
  fault, nothing told to the mind about it. Told only "good" or "bad", each link of the answer takes the blame the
  chain itself gives it, (1 − e_i) / (1 − Π e_j): the link it was least sure of most. The record is read back
  general → specific as everywhere: e ← e0; e ← c e_level + (1 − c) e, with
  e_level = 1/2 + (f_held − 1/2) / (f_claimed − 1/2) (e0 − 1/2): the claim's advantage over 1/2 rescaled by how
  far reality fell short of, or exceeded, what the level claimed. The general levels (every way; the kind of claim)
  are the vote of the ways that have a record of their own, each counted once as a member is in a frame. A way
  never tried has only that vote; where the vote is against it, the mind weighs holding the claim back against
  saying it once to find out, by its own record of whether what it held back was then answered. Measured
  (`research/living_mind_2026_11_22`, section 11; 5 seeds; the teacher's loop: asked, then shown the true dialogue
  whatever it said): in frames whose members merely share a value, wrong answers 66 → 29 and on things never asked
  about 25 and 24 of 40 → 0 and 0; eight such frames one after another cost 5, 4, 2, 2, 1, 0, 0, 0 wrong answers
  (120 → 14; with the frame's own record alone 53: every frame pays for itself), at the price of lawful frames
  withheld until shown (29 answers of 80, fewer with each lawful frame); where two readings disagree it learns
  which to believe in half the rounds (all of 40 fresh things right after 8 rounds instead of 16); told only good
  or bad, 33 wrong → 12 with the lawful frame untouched; and with F46 and F47, after looking things up by its own
  decision it says what it read (1 wrong of 80 where the mind without it still asserts 38). Where nothing misleads,
  nothing changes. Eight forms were built and taken out on the way (the notes say which and why). What it does not
  do without the switch for it (below): learn about its ways where it was not free to speak; and only the frame
  that spoke is in the chain. `!hata` prints what the mind has learned about its own ways and its last
  faults.
  **Restructuring** (the same switch; added the same day, notes section 12). A frame's category is every item that
  ever filled its place; when its own vote has been found wanting (its record), the vote is taken again among the
  members that share *another lived place* with the thing asked about (a category both were members of: F29 reads
  likeness from the same memberships), and as among frames the place whose members agree most speaks. That claim is
  a way of its own, settled and read back like any other, so the category is split where the error showed it had
  to be, by a place the mind lived, and the split is kept only as far as its claims hold. Measured
  (`t_fault_kinds.py`, 5 seeds): what a thing eats follows its kind, which was read in a sentence about each
  thing, and the watched dialogues are all of one kind. On 8 things never asked about, after 8 rounds of being
  asked and then shown: **40 of 40 right**, where the mind with the switch off is wrong on 40 of 40 to the end (it
  answers with the sentence it read about the thing) and the mind that only doubts says nothing (0 wrong, 0
  right). It does not find a likeness that no single lived place carries.
  **While it only watches** (`Config.silent_claims`, a switch of its own, off by default; notes section 13). At a
  context it has no record of its own of, the mind makes its full reading although it cannot speak, and what its
  ways claim there, with the word it would have said, is settled by what it hears next: everything new it watches
  is a test of its ways. Measured (5 seeds), a mind that is **never asked**: after watching eight dialogues about
  things of all kinds it answers 40 of 40 fresh things right (40 of 40 wrong without the step), having found from
  watching alone which reading misleads and how the category splits. It costs the full reading at every new
  context (the watching took 1.8 times as long), and before it has watched anything of another kind it is worse
  than the mind without it (it asserts what held for everything it watched: 30 wrong of 40 where the other says
  nothing).
- **Finding Out (F50, off by default; with open claims, looking and fault tracing).** Asked about a thing it has
  read nothing of, the mind used to say the word the things of that kind share: nothing marked that word as not
  being knowledge of the thing asked about, and it looked things up only later, in a silence. F49's chain says
  which ways claim the word the mind is about to say. A claim *rests on what was lived of the thing* when a
  relation that reads the thing's own record makes it, when the context has a record of its own, or when the word
  is how a sentence the mind read went on from where its own utterance began; everything else rests on what
  others share. Of a word that rests on others the mind knows only how far that way is proven by its own record:
  proven = W+ / (W+ + W- + k), u = 1 - max(0, 2 proven - 1) (what the record does not yet say counts for
  nothing: one lucky hit is not knowledge). As for any act, rate = (2u - 1) + (2 e_look - 1) against the line it
  speaks at: it withholds the word and looks the thing up first, in the turn it was asked in. That it is looking is
  an inner event; the word one says then is learned by living it, never given. In the online life what it sent for
  is fetched while it goes on living (it answers other things meanwhile), and when it has come the mind reads it,
  is brought back to what it was asked and says what it can say now. With nowhere to look, or when the look found
  nothing, it does not assert the shared word as long as holding back is answered (someone shows it; its own
  record of that), and says it where no one ever does. What was lived of the thing itself stands against what the
  others share (e_v <- (1 - c_r) e_v). Measured (5 seeds; 16 new things asked once each, a library that holds one
  sentence about each; whole answer / the value only / another value / nothing, of 80): off 9 / 0 / 44 / 27; with
  looking only in later silences 4 / 0 / 10 / 66; with F50 56 / 24 / 0 / 0; with F50 and F41 (what is known of the
  thing and not said yet) **80 / 0 / 0 / 0**. A library that holds every second thing: 40 of the 40 it holds whole
  and right; of the 40 it does not hold, 39 times the asked word and nothing more, once a wrong value (an induction
  its record supported: three new things in a row had eaten the same). Fetched in the background: the turn ended
  at once 45 of 45, it said the word it had heard three times in that state 44 of 45, answered something else
  meanwhile 45 of 45, and its late answer was whole and right 45 of 45. It needs a question the mind has lived
  being answered and somewhere to look; it does not ask its partner; not measured on a long life or on the web.
- **Asking Itself (F51, off by default; with open claims and looking).** Until here an open claim arose only when
  someone asked. An item is *new* while the mind has met it in one run of items only (the words every sentence
  holds are lived again at once and never stay; nothing lists which words are content). In a silent moment, as
  for any act, rate = (2u - 1) + (2 e_wonder - 1), u = 1: it brings to mind the question it knows best (a turn
  that was answered, generalised over one place) with the new item in its place and answers inwardly; what it
  cannot answer from what it lived of the thing is an open claim, which it looks up by itself; what it reads
  brings the next new thing. Patience: over a silence in which nothing else happens the drive goes on adding up
  (acted on when ceil(theta / rate) <= max(T, the moments it has been silent)), so an act of its own time that
  has rarely paid is taken again after a longer wait and never given up; a look that failed counts against
  looking that item up again only as far as looking was expected to pay then. Measured (5 seeds): told one
  sentence about the first of a chain of seven things, then 40 silent moments with no one saying anything: 35
  questions of its own, 45 looks, and all 30 things that eat something answered right afterwards (5 of 30 without
  it), nothing said aloud, and of the last thing, about which nothing could be found, the asked word and nothing
  more. Two chains at once: 30 of 30. A library that holds nothing: it asks and looks ever more rarely and does
  not stop; when the library is then filled it comes back by itself to everything it could not find (50 of 50).
  A youth of 20 looks that led nowhere does not keep it from the chain (30 of 30). It asks one question of
  anything new, oldest first, and does not reason over what it found.
- **Echo of the Same Words (F52, off by default).** F16 leaves out, of the lived continuation of the mind's last
  words, "what the partner just said", as written every word of the partner's utterance wherever it stood. In its
  exact form (F42's rule, for last words of any length) a continuation is an echo only where it followed these
  very words in what is present. Measured in F50's world only: 19 more whole answers of 80 without F41, nothing
  changed with it. Not run on the whole battery.
- **Frames of Frames (F53, off by default).** A frame generalises contexts that differ in one item. Frames that
  differ from each other in one fixed word and are the same otherwise are a *family*, a generalisation with two
  places: the frames' own place and the place of the word that varies. Nothing is stored for it; it is read from
  the frames that are there. A context a family fits is read by it: every frame votes once, with the conviction of
  its members, on the *kind* of what its members went on with (a relation of the value to the frame's filler; the
  frame's own varying word; the thing known of the filler whose lived sequence goes on to that word; a word all of
  them said): vote(kind) = sum c_frame f_frame(kind) / sum c_frame, e = c vote + (1 - c)/2 with
  c = lam e_new conviction(sum c_frame). A family claims only what its frames agree on, a place holds items of one
  kind, and where a frame of its own already speaks the family's claim is its prior in place of 1/2. So what was
  learned about kinds of question carries over to a new kind. Measured (5 seeds): a mind that has mastered two or
  more kinds of question answers a new kind at first sight, 40 of 40, where the same mind without it needs two
  dialogues and gets 34, whatever it had mastered before; a kind answered otherwise is not made up and is learned
  as fast; of a thing whose fact was never read only the asked word. With F49 its kinds of claim have records:
  where the family's claim is false the mind is wrong about the first new kind and holds the claim back for the
  next three (wrong first answers 20 -> 5); where lawful and false kinds alternate, one outcome cannot tell them
  apart and the right first answers are lost. Two places only.
- **Relations Lived Apart (F54, off by default).** A frame's relation "what the filler goes on with" was checked,
  for a member, against the member's own continuation. Where an answer repeats the asked word and goes on, the
  instance's continuation is the member's continuation (one event, written on both), so the relation held for every
  member by the dialogue itself. Now it counts only as far as the member lived it apart from its own instance:
  W+_apart(x -> v) = W+(x -> v) - W+(instance -> v), the strongest of what is left. Measured: with two facts read
  of every thing and one kind of question mastered a fresh thing was not answered (0 of 40) and now is (40 of 40);
  kinds of question learned one after another stayed answered (160 of 160) where three of four had gone silent.
  In the kinds world of F49 it takes away a silence that was an accident and leaves the plain induction (30 wrong
  of 40 before the first round, corrected by what is then lived).
- **The Asked Place (F55, off by default).** A question is a statement the mind knows with one item replaced, and
  the answer is that item. The run of items a thing begins in the context is read against what is known of the
  thing (the sequences that began with it, without what was only ever asked of it): a known sequence that is the
  run in every item but one names what it holds at that place. Nothing says which word is the question word: it is
  the one place where what was asked and what is known differ. It is a relation of a frame like the others: the
  members vouch for it (the value each went on with is what its own known sequence holds there) before it is read
  for a new filler, and a family can agree on it. F41 and F53 read what is known of a thing only where the answer
  names the thing first; this reads it wherever the question names it. Measured (5 seeds; answers that are the
  value alone, several facts read of every thing): a fresh thing asked a mastered kind 0 of 40 -> 40 of 40; with
  F53 a new kind at first sight 40 of 40; of a thing nothing was read of, nothing is said. Two watched things that
  happen to carry the same value mislead the mind after two dialogues (6 wrong of 40) unless what was lived of the
  thing itself stands against what the others share (F50's rule: 0 wrong). One place, and the run must begin with
  the thing.
- **The Kind of a Whole (F56, off by default; with wholes).** A whole stands in a summary as its form: which
  wholes stand in it. The form was read from the whole's own summary, and a lived command whose answer reads as the
  replies of its wholes in more than one order ("walk twice and walk": three walks) has none: two readings tie and
  nothing is concluded. Such a command stood as a bare whole and took what an action phrase takes
  ("[walk twice and walk] right"): the one source of all four answers of the published SCAN splits that were not
  exact (traced in the mind's own list of readings). A form says which wholes stand in a whole, not in what order
  their replies come: it is now given also there, and the dialogue's summary stays unconcluded. Measured: the four
  commands exact with it (each split's life lived with the switch off and on); a small world reproduces the error
  and its correction (a test).
- **Trials (F57, off by default).** Until here which ways of reading a mind uses was decided outside it: a
  mechanism was written after a failure was traced, measured by its builders and turned on by hand. Here that
  decision is the mind's, taken as its builders took it: a controlled experiment on what it lives, one thing
  changed. A context at which the mind decided, and which then goes on from outside (someone shows the answer), is
  read again before what came is taken in: once as the mind decides it (the same reading and race, run dry) and once
  for each of its ways with that one way changed. Where the two differ the way gets one experience on a tie of
  its own (better where the configuration with it said what came; saying what came is better than silence, silence
  better than another word). A sleeping way is taken into use when it did better more often than worse by more than
  one experience (W+ / (W+ + W- + k) > 1/2: F50's rule), a way in use is put away by the same rule the other way
  round, and a way in use stays on trial. One change at a time: two ways that mend the same failure are proven by
  the same cases and nothing was shown about the two together, so one is taken and the trials of the sleeping ways
  begin again against the new configuration. The configuration is the mind's own state (saved with it, undone
  with a moment), and every case and change is kept with what each configuration would have said (`!deney`).
  Measured (5 seeds, a curriculum of eight kinds of question in the teacher's loop): the mind as it was answers 2
  of 320 questions about fresh things; with trials it takes the asked place and frames of frames into use in every
  life and answers the five kinds read of the thing itself (200 of 200), a sixth at first sight (40 of 40); what it
  was taught before is answered after every change. In a world where answers name the thing it takes other ways
  (F41, F53, F52, F37) and answers 200 of 200, where the setting made by hand answers 194. A way that does harm
  where it is tried is not taken (0 of 5). No way was put away in any life measured: where a claim failed it was
  stopped where it is made, after one failure, by the frame's members, the family's agreement and F49.
- **Built Relations (F58, off by default; with trials).** Where the configuration did not say what came, the
  failure is looked into before what came is taken in: for the filler of each one-place frame of the context,
  every chain of two of the mind's relations is read, and one that names what came is a relation to try (the steps:
  how its sequences went on, what it goes on with, what it was answered with, what stands where it stands, what a
  known sequence holds at the asked place, the things known of it). A proposed chain is a way like any other
  (F57); in use it is a relation of a frame, vouched for by the frame's members like the relations the mind was
  built with. A relation it built is a step of the next chain, so a chain of three is found as a chain of two once
  the shorter one has proven itself, and so on: the relations grow out of the ones that worked. Measured in the
  same curriculum: asked what was read only of a thing's class, the mind builds "what is known of it, then what
  stands at the asked place of that" in every life (0 of 40 -> 40 of 40 for fresh things; the same structure met
  again is learned in one round instead of three); asked what was read only of the class's order it builds the
  chain of three out of that one (0 of 40 -> 40 of 40). The mind set by hand with the mechanisms of the sections
  before answers 8 of those 120 questions (and says a wrong word to 14). Nobody wrote either relation. What is still given: the primitive
  relations, that a new relation is a chain, the experiment and what counts as better.
- **Conjectures (F59, off by default).** Everything the mind said of a thing it had been told of that thing, or
  read from what it knew at the moment it was asked. Here it comes, unasked and in its own time, to statements
  nobody told it, and keeps them apart from what it knows. A thing told to go on with an item that is itself told
  about is linked to it ("x y ..."; the rest of that sentence is the link's tail). Every statement told of the
  item, with the thing in its place, is a candidate. Where the thing was told the candidate too the link carried
  it (one experience for); where the thing was told it with one item different it did not (one against); the
  records are read level by level, each the vote of the one below, every member once: this statement through
  links with this tail, links with this tail, links of any kind. A level has decided when it is for or against by
  more than k; the most specific level that has decided speaks, and one that leans against keeps the more general
  from speaking for it. A candidate nobody told, whose record has decided for, is a conjecture: the inner voice
  thinks it (imagined, as every thought; so sleep sorts it against what is lived and it weighs as the mind trusts
  its imagination), and it is kept with its source and with what becomes of it when the thing is told more (held,
  or failed). When to think a thing over is the mind's decision in a silent moment, as for any act. Nothing names a
  word, a field or a kind of link: that what is told of the item before one word is told of the thing, and before
  another word is not, is found. Measured (5 seeds; three fields with words of their own, the class's statements
  told of six things of one field only): the mind comes to all 220 statements that hold of the other things and
  were told of none (60 of 60 in that field, 80 of 80 and 80 of 80 in the two fields where nothing was told of any
  thing) and to none that does not hold; to none with no such evidence, with one experience, or where the evidence
  is against; one statement told in each of two fields, neither enough alone, is enough together (270 of 270). It
  does not believe of another word for the link what one word carried. Links in other shapes, running text, and a
  conjecture as the premise of another are not there.
- **At the same time, with nothing lost (not a mechanism: a property).** What a moment writes depends on what the
  mind expected, so a life is lived in turn: copies of the mind that live different things at once and are added
  into one ("sibling minds") were measured and eliminated (no loss where the copies share a beginning that holds the
  structure of what they read; on natural text recall fell, 148 valid of 200 to 95, and parting more often made the
  loss smaller, never nothing). What writes nothing runs at the same time exactly: many questions or thoughts, each
  answered by a process that is the mind as it stands (`zihin/paralel.py`; word for word the answers given one after
  another, the asking mind untouched; 60 answers in parts 4.8 times sooner on 4 processors). The readings inside one
  moment could run at the same time too; that needs the substance's indices in shared memory and is not built.
- **Told Apart (F45, off by default; with F43).** A word that was never said alone has no reply of its own, and stands
  in a summary as the word it is: what was lived with "left" says nothing of "right" in the same place, and the mind
  said left's answer ("around right", held out with every command it stands in: 0 of 500, 379 of them wrong). But it
  had lived "walk left" and "walk right". When two lived dialogues differ in one word of the utterance and their
  answers, of one length, differ in one pair of words wherever they differ, each of the two words answers with its
  own: Mill's method of difference, on two dialogues the mind can hold. What it gives is lived as a dialogue of its
  own, inferred ("left" answered "i_lturn"; weighed by the trust the mind has in what it infers), unless the word was
  itself said alone and answered. The residue is taken out first: the difference is assigned only when what the two
  answers share is what the shared words answer with, and a word of the answer that a shared word answers with is
  not the differing word's share ("walk twice" and "walk left" differ in "i_walk" and "i_lturn", and "i_walk" is
  walk's: without this "twice" became a word with a reply, and T-compose fell from 75 to 63 of 75); a word told
  apart in several ways, none of them most of the evidence, has no reply. The word is a whole from then on, so
  "walk around left" is lived as [whole around whole: the reply of the third, the reply of the first, four times],
  and "walk around right", never met, is the same summary. "around right": 0 → **500 of 500**; the three other
  template splits of the same paper: "jump around right" 200 of 200, "V right" 200 of 200, "V opposite right" 200 of 200 (a
  Transformer from scratch on the around-right dialogues: 3 of 500). These were measured on splits generated here
  from the grammar; compared with the published files (2026-11-22) the generated "right" and "opposite right" splits
  are easier than the published ones (they keep the "turn" commands in training). On the **published** files
  (about 1,090 of the published training commands watched twice): "right" **4,475 of 4,476**, "opposite right"
  **4,476 of 4,476**, "around right" **4,474 of 4,476**, "jump around right" **1,173 of 1,173**, and the length
  split (answers of 24 to 48 actions) **3,919 of 3,920**, "add jump" **7,706 of 7,706**: 26,223 of 26,227 over the
  six published test files, the four wrong answers one source that F56 closes (with it the whole "right" file is
  4,476 of 4,476; the other files with it are running) (`research/living_mind_2026_11_22`, notes sections 19, 20
  and 22). In a world it was not developed on (T-dans:
  Turkish moves said alone, directions never said alone and placed before the move, "iki kez" and "üç kez" between
  them; held out, one direction with both): the first run **600 of 600** over three seeds (0 of 600 without F45; a
  Transformer 108 of 200); when the new direction is met in **one dialogue only** ("geriye zıpla"), 600 of 600 of the
  other commands it stands in (1 of 600 without F45; a Transformer 7 of 200). The lives of SCAN's add-jump and of
  T-görev with the switch off are bit for bit what they were; with it on, add-jump 300 of a sample of 300, T-görev 492 of
  492 with an unchanged life (nothing there is told apart), and every world of the battery, the teacher protocol and recall are what they were (SCAN within 11: 844 of 844). With words as wholes two readings of an
  answer can be equally cheap. When every whole of the one lies inside a whole of the other ("turn left" and the
  "left" in it answer alike) they are one reading at two grains and the larger wholes are taken; when their wholes
  cross (in SCAN "walk after run left" answers the same whether "left" is run's or the conjunction's) they are two
  readings and nothing is concluded (taking the larger whole there made "(A after B) twice" a reading: six wrong of
  200 on "jump around right"; concluding nothing from either tie left every command with a "turn left" part
  unanswered). What it does not do: one word for one word only (not a word whose share of the answer is several
  words, or none); an inference can be wrong where two frames agreed on every filler they share; a whole does not
  say of what kind it is; and a told-apart word is answered as any part is, so an answer costs more.
- **Resemblance Vote.** Two things resemble each other through shared facets, and a facet is more telling the fewer
  things share it. The members most like a new thing speak loudest, weighted as in exemplar models of categorisation
  (the generalized context model). Seeing and hearing recognise this way: new images 73% / 83% / 67%, new sounds of
  three known synthetic voices 93% / 97% / 93% (chance 33%).
- **Situation Resemblance.** An utterance never lived as a whole is answered with what followed the most similar
  lived situations. If the resemblance is weak, for example because the utterance has a word nothing lived, nothing
  is made up.
- **Lived Exclusion (F27).** A channel carries one item per moment, so what a situation was itself lived to be
  followed by is evidence against what a generalisation proposes there that the situation never went on with: such a
  proposal is read with no evidence for it and the situation's own lived amount on that channel against it, nested
  last as own evidence. The specific already overrode the general for what it lived; now it also counts against what
  it did not. Read on the same lives of the generated worlds, exceptions are kept more often (nitelik 0.74 → 0.84).

## 6. Speaking, thinking, not knowing

- **Continuation Rule.** When nothing in the present situation expects anything above ½, the mind reads the items
  after the last perceived event, and the longest tail of them that has lived continuations speaks, with reduced
  conviction. This rule never starts an utterance, and it leaves out the partner's words still inside the present
  context, so it does not repeat what the partner just said.
- **Sequence Starts (F26).** What followed a word where a sequence began with it (a sentence read, an utterance after
  silence) is kept apart from what followed it anywhere. Nothing new is stored: every episode that began x, v
  recognised the two-item unit (x, v) once, and those recognitions are counted, the other beginnings counting against.
  A frame's members can agree that the answer goes on as the sequence that began with the asked word (START), an
  utterance of one word so far goes on that way, and so does a word brought to mind after silence. In a template-free
  test (a mind that read the documents once, 200 random sentences, their first two words brought to mind) it said a
  read sentence beginning with those words in 150 (145 with Exact Relations, below), and the rest of that very
  sentence in 60 of the 61 whose first two
  words begin only one sentence. The strongest small Transformer trained on the same documents (7.4 million
  parameters, 50 minutes) said a valid one in 41. With one word as the cue the Transformer said a valid sentence more
  often (51 against 44; 47 with Exact Relations), because the mind stays silent when a word begins several
  sentences.
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
- **Compact indices and a day store.** The indices beside the evidence were Python objects, 78–87% of a loaded
  mind's RAM. They are now packed keys, 32-bit id arrays, and a tie lookup kept as a long-term store with a day store
  of the ties made since, merged in sleep (exact; the same results bit for bit). A loaded mind with 3.25 million ties
  went from 1784 MB to 762 MB beyond Python and torch, about 234 bytes per tie.
- **Open problem.** Memory still grows linearly with lived data, at about 11 ties per moment. The next steps, reading
  only the ties that can still change and a memory-mapped long-term store on disk, are planned in the notes.

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
- The configuration the chat program runs ("speaking") was not in that comparison. In the same worlds it did poorly
  (held-out 0.57 and 0.04 after 600 experiences). Two causes were found. First, it spoke into the world's stream:
  these worlds let the mind act in every moment, so it lived its own words between the world's, and learned five
  times as many categories, mostly from them. Second, F12's "next" took the members' agreement on a value (Exact
  Relations). Listening while the world speaks, as in a chat, and with Sequence Starts, Lived Exclusion and Exact
  Relations, it reaches 0.78 and 0.81, with members, exceptions and the hidden classes all at 1.00. The "members"
  configuration's 0.93 in the first world is not reached.
- Recall of what was read once (F26, F27) is compared with the duel's Transformers in section 6, under Sequence
  Starts. The earlier same-data duel asked both sides the held-out subjects as a bare word, a form the teacher
  dialogues never used, so it measured both on a question neither had learned; the template-free test replaces it
  (research/living_mind_2026_11_15/NOTES.md).

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
  - the chat configuration did poorly in the generated worlds where the "members" configuration does well (mostly
    because it spoke into the world's stream and F12's "next" took the members' agreement; with both addressed it
    reaches 0.78 and 0.81 against 0.93 and 0.81);
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
| Sequence Starts | Dizi Başları | F26 | |
| Lived Exclusion | Yaşanmış Dışlama | F27 | |
| Exact Relations | Kesin İlişkiler | F28 | |
| Analogy by Interchangeability | Yerine Geçebilirlik Benzetmesi | F29 | the distributional hypothesis; NARS analogy |
| Corroboration | Bağımsız Kanıtların Birleşmesi | F30 | NARS revision |
| Paradigmatic Relation | Dizisel İlişki | F31 | the paradigmatic axis (Saussure, Jakobson) |
| Relational Binding | İlişkisel Bağlama | F32 | variable binding of a frame's place |
| Relational Memory | İlişkisel Bellek | F33 | context-dependent memory; source monitoring |
| Reply Relation | Cevap İlişkisi | F34 | compositional generalisation (SCAN); backoff to lived suffixes |
| Generalised Consequences | Genelleşmiş Sonuçlar | F35 | model-based choice: expected value of consequences |
| Word Forms | Kelime Biçimleri | F36 | morpheme boundaries by successor variety (Harris) |
| Inheritance | Kalıtım | F37, off by default | NARS deduction; default inheritance with exceptions |
| Witness Reliability | Tanığın Güvenilirliği | F38, off by default | source reliability; NARS deduction; backoff in variable-order models |
| Replacement Witness | Değişiklik Tanığı | F39, off by default | Mill's method of difference |
| Urge to Go On | Sürdürme İsteği | F40, off by default | drive and habit (Hull): whether to speak apart from what to say |
| Unsaid Known | Söylenmemiş Bilinen | F41, off by default | recall inside a generalisation (the frame is transparent to the filler's own lived sequences); distributional likeness; drive as the unsaid share |
| Episode Echo | Bölüm Yankısı | F42 | inhibition of return; repetition avoidance in sequence generation |
| Wholes | Bütünler | F43, off by default | chunking (Miller); working memory as a stack of suspended contexts; roles and fillers, pointers; analysis by synthesis |
| Lived Exclusion across channels | Kanallar Arası Dışlama | F44 | closed-world reading of an instance's own record |
| Told Apart | Fark ile Cevap | F45, off by default | Mill's method of difference; minimal pairs; contrast in word learning |
| Open Claims | Açık Kalan Beklenti | F46, off by default | feeling of knowing; information gaps (Loewenstein); known unknowns as the drive to look up |
| Looking Up by Itself | Kendi Kararıyla Bakmak | F47, off by default | information seeking as an act chosen by its expected outcome; learned helplessness when looking never helps |
| Thinking by Itself | Kendi Kararıyla Düşünmek | F48, off by default (on in the online life) | surprise as the drive to think (prediction error); reflection valued by what it comes to |
| Fault Tracing | Hata İzi | F49, off by default | error monitoring and metacognitive confidence; calibration of confidence by observed outcomes; the reject option (abstaining when unsure); credit (blame) assignment along a chain; partial pooling (each frame one vote) |
| Finding Out | Öğrenmek İçin Harekete Geçmek | F50, off by default | epistemic (information seeking) actions; knowing versus guessing by the source of a claim; the reject option with a learned policy; acting while waiting (asynchronous retrieval) |
| Asking Itself | Kendine Sorma | F51, off by default | curiosity as self-posed questions; intrinsic motivation; active learning; exploration of a frontier; persistence under sparse reward |
| Echo of the Same Words | Aynı Kelimelerin Yankısı | F52, off by default | |
| Frames of Frames | Çerçevelerin Çerçevesi | F53, off by default | learning to learn (a learning set); schema abstraction over schemas; two-place (relational) generalisation; hierarchical priors (the general level is the specific level's prior) |
| Relations Lived Apart | Ayrı Yaşanmış İlişkiler | F54, off by default | held-out validation (a rule is tested on evidence that did not make it); circular evidence |
| The Asked Place | Sorulan Yer | F55, off by default | question answering by aligning a question with a known statement; analogy as structure alignment; the wh-word as the one unaligned slot |
| The Kind of a Whole | Bütünün Türü | F56, off by default | constituency (what a phrase is made of) kept apart from its denotation; attachment ambiguity |
| Trials | Kendi Üzerinde Deney | F57, off by default | model selection by the learner itself; controlled (single-variable) experiments; counterfactual evaluation of one's own policy on logged situations; algorithm configuration; metareasoning |
| Built Relations | Kurulan İlişkiler | F58, off by default (with trials) | program / relation induction by composition; multi-hop inference found from a failure; library learning (what was built becomes a building block) |
| Conjectures | Varsayımlar | F59, off by default | hypothesis formation; property induction over a learned relation (which links license inference); abduction kept apart from knowledge by provenance; hypothesis testing against later evidence; default reasoning with exceptions |
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
