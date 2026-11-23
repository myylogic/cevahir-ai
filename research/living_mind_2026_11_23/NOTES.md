# The ground: what is perceived and what is said meet in one evidence calculus (2026-11-23)

What was asked (the owner): "a human mind does not work with the mind alone; deepen the mathematical models of what
we have built; computing over word sequences is not an example of intelligence." The model is written in
[MODEL.md](MODEL.md) (Turkish); this file holds what was built after it and what was measured, failures included.
Everything is measured on one core of a 4-core cloud machine, pure Python + numpy/torch on CPU; no neural network is
part of the mind (the Transformer below is a baseline, in `t_m1.py` only).

## 1. The world (`dunya.py`)

A small world of things, people and actions, and a teacher who speaks about what happens, in Turkish or in English
(the same situations in both):

* 10 kinds of things (horse, cat, dog, bird, ball, apple, book, house, school, water), 6 people, the teacher; every
  thing is a tracked index with 15 features on opaque dimensions (`d3:v1`): perception is complete (what a thing has
  not is seen too: no mane, no wings);
* 8 actions the teacher orders (look at, hit, run to, come to, go to, laugh at, hold, take) and one it only does
  (pointing); an event has 6 opaque action features and, for each participant, 8 participation features (who starts
  it, who moves, who is looked at or turned to, who is touched, who changes, who goes along), 1 or 0;
* the teacher's lines: naming while pointing ("bu bir at" / "this is a horse"), "bu ne ?" asked while it points,
  orders ("ali ata bak" / "ali look at the horse") followed by the doing and its description while it happens ("ali
  ata bakıyor" / "ali is looking at the horse"), "kim V-ıyor ?" / "N ne yapıyor ?" asked while it happens, facts about
  each kind told, "X nedir ?" asked of half of the kinds. A colour is said only when another thing of the same kind
  stands beside (to tell it apart).

The splits come from the seed alone, before any learner lives the world (`python dunya.py --splits`): one kind for
each verb done to three or more kinds and one verb for each person are never shown together (T1 asks the 53 to 54
orders that combine them; T1 seen asks as many shown ones); of each kind one colour is never said with it (T3, with
another thing of the kind beside), one colour is never seen at all (T4: "bu ne ?" of such a thing); five kinds are
never asked "X nedir ?" (T2); T6 asks who does what, and what a person does, of 24 held-out combinations. 626 to 660
lines a life, about 3,700 words. World digests are in every row.

## 2. The baseline (M1, `t_m1.py`, `results/m1.jsonl`): today's whole mind and a Transformer from nothing

The same life and the same questions. Today's mind hears facts and namings and watches the dialogues; with `--scene`
what is seen is shown as today's core can take it (one image-like thing with all its facets, without which thing has
which). The Transformer (3 layers, d 64, from nothing on the same lines, 4,000 steps) gets the scene's facets as tokens
before the line with `--scene`. Three seeds; right answers (word for word):

    learner                          T1 new    T1 seen   T2 X nedir  (facts)   T3 colour  T4 name   T6 situation
    the mind, Turkish, words          144/158   158/158   0/15        15/45     8/45       0/23      1/72
    the mind, Turkish, + scene        111/158   158/158   0/15        15/45     22/45      4/23      1/72
    the mind, English, words          126/158   158/158   0/15         0/45     0/45       0/23      1/72
    the mind, English, + scene        104/158   158/158   0/15         2/45     0/45       0/23      1/72
    Transformer, Turkish, words       119/158   158/158   0/15         0/45     6/45       1/23      4/72
    Transformer, Turkish, + scene      96/158   158/158   0/15         0/45     0/45       21/23     29/72
    Transformer, English, words       128/158   158/158   0/15         4/45     1/45       3/23      3/72

What it shows:

* An order and its description differ only in the verb's form: today's mind does it from the words alone, 144 of 158
  in Turkish, by keeping the order's words and changing the end of the verb. It has no event; with the scene shown as
  facets it does worse (111). The small world of the owner's example (`t_paradigm.py`: 14 order–description pairs) had
  too few examples for it (1 of 6).
* What can only be answered from what is seen (T4 naming a thing, T6 who is doing what) today's mind does not answer
  at all, with the scene or without; a Transformer given the scene as tokens learns naming (21 of 23) and part of the
  situation questions (29 of 72).
* "X nedir ?" for a kind never asked: no learner says the three facts; today's mind says the first one (15 of 45 facts).

## 3. The ground (`zihin/zemin.py`, `Config.grounded`, off by default)

Built after MODEL.md, measured in `t_m2.py`. With the switch off the four gates are bit-identical (digest, plain,
wholes, the 17 configurations). What it is, briefly (the module's docstring is the full account): perception binds
features to things, events and participations; a word refers to one of the things talked about or to the event, by an
alignment of the utterance in which what one word explains another gets little of, and names the features of what it
refers to; an utterance's form becomes a construction (places of participants, of modifiers, of the action, fixed
words) whose places hold the participation profile of what filled them (a role is a profile, never a name); what the
answerer said back is learned per turn (the very words, the construction with its action word, the construction);
answering is from the meaning: the event going on, or the one an order asks for, told with the construction that
answered such a question, its places filled from the event.

### 3.1 Found on the way (each measured, each changed)

* Words naming features directly (cross-situational alignment of features, as Fazly et al. 2010): every word of an
  order named the person's features too, since a person is in every order; two meanings that always come together
  are not parted that way. Changed: a word refers first (to a thing, the event or nothing), then names the features of
  its referent.
* Concepts induced from perceived resemblance put a dog with the horses: the features every thing shares (no wings,
  no stem) outweighed the few that differ. Kept for later; here a word's meaning is the profile of what it referred
  to.
* Likelihoods averaged over dimensions (a geometric mean) left the English alignment flat after the whole life
  (shares 0.14 to 0.24 for every word and referent): the shared features drown the few that differ, and English
  utterances have more words. Changed: the likelihood ratio over the dimensions, measured against a background word
  that is in every utterance (IBM Model 1's NULL word), so that a word that predicts what is talked about no better
  than the background ("the", "is") refers to nothing.
* Turns kept per construction alone mixed "bu ne ?" with "kim bakıyor ?" (both two places: "bu ne ?" was answered
  "bu bakıyor"), and whatever was said next counted as an answer (a description, then the next order: "ali ne
  yapıyor ?" was answered in the imperative). Changed: only what the answerer says back to another is an answer, and
  the very words of a question are looked up first (recall), then its construction with its action word, then its
  construction.
* "kim bakıyor ?" was answered "kim bakıyor": the question's own word was reused for the doer. Changed: a word that
  expects the identity of the thing (a proper name) is chosen before one that names any doer.
* A ball described as "elmaya" (an apple) where looking at a ball was never shown: a word is not chosen for a thing
  it contradicts (it expects, at least as firmly as the mind speaks, another value of one of its dimensions: "elmaya"
  expects a stem).
* A background word (IBM Model 1's NULL word: in every utterance, learning what is talked about at all, the reference
  of every word's likelihood ratio) took every referent: real words, measured against it while their meanings were
  still empty, predicted less than it did and learned nothing (every Turkish order silent, every "bu ne ?" answered
  "bu bir kuş"). Taken out: the ratio is against knowing nothing (1/2 on each dimension); a word that predicts only
  the features every thing shares wins no referent where a word predicts what tells it apart, since every word
  predicts the shared features alike.
* A kind word heard twice with one thing (an order and its description name the same thing) expects that thing's
  identity for a while (0.67, over the speaking line) and contradicted every new thing of the kind. An identity is no
  property of a kind: it is left out of the contradiction (a proper name is preferred where it names the identity).
* A modifier read on every dimension of its thing ("kahverengi", heard with brown dogs only, expects a tail)
  contradicted a brown horse. A modifier tells apart what its head leaves open: it is read on the dimensions its head
  word does not expect a value of as firmly as the mind speaks.
* The words' shares normalized once (each word's over the referents, then each referent's over the words) left the
  English orders at 0 of 10 and the situation questions at 2 of 10 on seed 0: the words heard in every utterance kept
  part of everything. Changed: a soft one-to-one assignment, Sinkhorn's balancing of the likelihood ratios (every word
  to one referent or to nothing, every referent to one word or to none, balanced in turn until nothing moves), each
  value weighted by how rare it is in what is talked about (1 - e(the unit of what is talked about at all has it): no
  wings tells nothing, a mane or a person's identity much). A referent's head is the word with the greatest share of
  it, where that is more than no word has; what is pointed at is what is talked about (joint attention); a person's
  own name is found from the person (the words that expect her identity), whatever construction is said. Seed 0
  after it: Turkish orders never shown 8 of 10, naming 7 of 8, situations 7 of 10, colour 6 of 8; English 8 of 10,
  0 of 8, 8 of 10, 0 of 8.
* English naming 0 of 8: "what is this ?" was answered while the teacher went on pointing, but the world did not
  show the pointing with the answer, so "this is a horse" was about everything in view. The world now shows it
  (`dunya.py`, the answer's scene of "bu ne ?" / "what is this ?"; the world digests changed only by that field: the
  baselines of M1 read the scene's facets, which do not hold it, so their inputs are the same).
* "bu bir bu": a construction's fixed words filled its places. They do not now, and of the words that may fill a
  place the one that tells most of the thing is said (what it fits this thing beyond what it fits things lived on
  average: "bu" fits every pointed thing). That made "kediye" a proper name (it fits one cat more than the others): a
  word names a thing's very identity only as firmly as the mind speaks.
* "bu bir elma" said of a horse: a form kept from before its words meant anything. A construction whose fixed word
  has named participants and names none of the things now talked about is not said.
* "kim alıyor ?" has no action place, so the time relation of its turns was "-" and nothing it was answered with
  matched: the word of a question that says most of an event (its action word, else the word not naming a
  participant that expects the most of the dimensions events are perceived on) is what it asks about.
* A book of another colour was not "kitap" ("kitap" expected one colour 0.62, the other 0.49): a dimension on which a
  word met another value it is not sure is absent (above 1 - e*) varies for that word and is no mark against it.
* An answer is said only from turns at least as expected as the mind speaks.
* "ali topa bak" (looking at a ball never shown) was answered "ali elmaya bakıyor": the words lived in that verb's
  place were preferred to the question's own. The question's word for a participant is kept where it was lived in
  that place of the answer (with any verb).

### 3.2 Tried after that and not kept (each measured on seed 0, the probe of the first 10 / 6 / 8 / 10 / 8 items)

The version kept answers, on the probe, Turkish 9 of 10 orders never shown (understanding), 5 of 6 (seeing), naming
8 of 8, situations 9 of 10, colour 5 of 8; English 10, 6, 7, 9, 6. One failure of it led to the four tries below: a
ball being looked at was called "elmaya" (an apple) in both languages, because a dimension is weighted by how rare
the thing's own value is, so "elmaya" expecting a stem tells almost nothing against a ball (having no stem is common):
the rarity-weighted ratio gives "topa" 2.32 and "elmaya" 2.03 for the ball.

* The pointwise mutual information in place of the rarity-weighted ratio (ln P(value | word) - ln P(value), P(value)
  how often the value is met in talk), with P(value | word) the one rule's e: a word less sure than talk of the common
  values "expects" the rare ones more than talk does, so every word fit a horse by about 10 (person names 9.8 to 10.6,
  "ata" 14.3). Turkish 4 / 5 / 1 / 6 / 0.
* The same with P(value | word) the one rule with talk in place of "I do not know" (c f + (1 - c) P(value)): the
  alignment became sharp ("ali ata bakıyor": shares 0.98, 0.99, 0.98, against 0.67, 0.70, 0.46), but in "bu bir elma"
  the words heard with every pointed thing took the thing ("bu" 0.90) and the pointing ("bir" 0.71), and "elma" was
  left with nothing (0.74 to nothing): a kind word heard with one kind is shut out by words heard with every kind.
  Turkish 9 / 5 / 4 / 8 / 3.
* That, with the exclusion value by value (each value of a referent explained by one of the words that refer to it,
  or by none; a word refers to a referent or to nothing without the referents being shared out), so that "bu" and
  "elma" could both speak of the apple, telling different things of it: Turkish 8 / 4 / 4 / 3 / 1 (the verbs went
  wrong: "ali kediye gülüyor" for "bakıyor").
* The information (the second form) only where the mind chooses what to call a thing, with the form a verb's place
  takes judged from the other places the word was lived in: as a sum over the dimensions it chose the form heard most
  ("bu bir kitaba"); as a mean (the features of one thing are one observation of it) English situations went to 10 of
  10 (the ball no longer an apple) but "ali look at the ball" was answered with the apple, and Turkish was unchanged
  (the ball "elmaya" still); judging the form by whether what the word's places took this place took too, Turkish
  naming fell to 5 of 8 ("bu bir ata": the namings were parsed into two constructions, and the place of one took none
  of the other's words). Telling which form a place takes needs the words of one meaning compared across places; that
  is the next step, not this version.

### 3.3 The measurement (`t_m2.py`, `results/m2.jsonl`): the version kept, three seeds, against the baseline of M1

The same lives and questions as M1 (`World.training`, `World.tests`), lived by today's whole mind with the ground on
(the core hears every line as before). Each question is answered by the ground alone (silent where it does not
understand), by the core alone on the same mind (its answers as in M1), and by both (the ground, else the core). For
orders never shown, `a` is the order heard with only the things in view and answered from what it asked for (the
event is never shown); `b` is the order heard, then done, then answered while it goes on. Right answers, word for
word, summed over seeds 0, 1, 2:

    learner                                  T1 new    T1 seen   T2 X nedir (facts)   T3 colour  T4 name   T6 situation
    the ground, Turkish (a and b alike)      128/158   158/158   0/15                 31/45      16/23     48/72
    the core of the same mind, Turkish       144/158   158/158   0/15  (15/45)         8/45       0/23      1/72
    both, Turkish                            151/158   158/158   0/15  (15/45)        31/45      16/23     48/72
    the ground, English (a and b alike)      152/158   158/158   0/15                 25/45      19/23     54/72
    the core of the same mind, English       126/158   158/158   0/15   (0/45)         0/45       0/23      1/72
    both, English                            152/158   158/158   0/15   (0/45)        25/45      19/23     54/72
    M1: the mind with the scene, Turkish     111/158   158/158   0/15  (15/45)        22/45       4/23      1/72
    M1: Transformer with the scene, Turkish   96/158   158/158   0/15   (0/45)         0/45      21/23     29/72
    M1: the mind with the scene, English     104/158   158/158   0/15   (2/45)         0/45       0/23      1/72
    M1: Transformer with the scene, English  145/158   158/158   0/15   (0/45)         0/45      23/23     42/72

What it shows:

* Who is doing what, said from what is seen (T6): the ground 48 and 54 of 72, a Transformer given the scene as tokens
  29 and 42, today's mind 1. A colour never said with the kind (T3): the ground 31 and 25 of 45, the Transformer 0.
* Naming a thing of a colour never seen (T4): the Transformer is better (21 and 23 of 23 against 16 and 19).
* Orders never shown (T1 new): in English the ground says the description from the meaning, the event never shown,
  152 of 158 (the core, keeping the order's words and changing the verb, 126; the Transformer with the scene 145; the
  six misses are one, seed 2's "X look at the house" said of "the school" for the six people: section 3.4). In
  Turkish the core does more (144 against 128): the ground is silent where the order uses a form of a word it never
  heard ("ali elmayı al": the two verbs that take that form, hold and take, were both kept from apples in seeds 0 and
  1, "kuşu" in seed 2), and the core copies the form. Those are 30 orders, exactly the ground's 30 misses: every
  Turkish order whose words were all heard was answered right, 128 of 128. Both together: 151 of 158. The answers
  from the meaning alone (`a`) are right as often as those given while the event is seen (`b`) in every one of the
  six lives.
* The core answers in the same mind as in M1 (Turkish orders 144 of 158 in both): the ground's ties, written in the
  same store, do not change what the core reads.
* "X nedir ?" for a kind never asked: no learner says it (the ground has nothing to say of it yet).
* Cost: living the world 90 to 134 s of one core, answering every question three ways 179 to 302 s.
* With the switch off the four gates are bit-identical (digest, plain, wholes, the 17 configurations), on this code.

### 3.4 What fails (seen in the examples of the rows and on the probes), and what comes next

* "bu bir bu" (Turkish, seeds 1 and 2) and "this is a this" (English, seed 2): where the namings were parsed with the
  deictic word as the head, the answer's place is filled with it.
* "this is a school" for a house (English, all three seeds; "X is looking at the school" for the six orders of seed 2
  that look at a house): the two words name nearly the same features.
* "ali ne alıyor": the question word "ne", lived in that place of the question, is kept as the participant's word.
* A ball being looked at called "elmaya" / "the apple" (section 3.2).
* Colour orders: mostly silent ("ali kırmızı eve gel", "can look at the white horse"), once a wrong verb and colour
  ("ayse is hitting the yellow apple" for taking the green one).
* An event never shown told with another verb ("ali is laughing at the horse" while he looks at it).
* Next, in this order: words of one meaning compared across the places they were lived in (which form a place
  takes; the deictic word and the question word told apart from names), the colour orders, telling ("X nedir ?" from
  what was seen and told), and concepts as gathering points (MODEL.md, M5).
