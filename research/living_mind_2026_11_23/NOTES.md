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
