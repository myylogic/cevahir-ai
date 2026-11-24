# The open mind (release v7)

`cevahir-open.zihin.gz` is the program's **whole mind** (every mechanism of the full configuration on, silent claims
too), raised from nothing on public material only. Minds raised on the author's private documents are not in this
repository and never will be.

The folder also holds **the open Cevahir** (`cevahir/`, below): the same lessons lived by Cevahir, the one layer of
the Living Intelligence study.

## Use it

```
python -m zihin --load minds/cevahir-open.zihin
```

On the first load the compressed file is unpacked beside it, as `cevahir-open.zihin` (about 17 MB; ignored by git).
The mind speaks Turkish, the language of its lessons. Ask a question after a moment of silence:

```
merhaba                      -> merhaba
nasılsın                     -> iyiyim teşekkürler
adın ne                      -> adım cevahir
asal sayı nedir ?            -> yalnız bire ve kendisine bölünen sayıdır
python nedir ?               -> python yorumlanan bir programlama dilidir
```

It keeps learning from everything you tell it (`!ogret`, `!oku`), like any mind of the program.

## What it lived

[`raise_open.py`](raise_open.py) is the whole recipe. The mind was created empty and studied with
`research/living_mind_2026_11_22/etut.py`. Each lesson was taught, its dialogues asked in a shuffled order, a wrong
attempt taken back exactly and the answer shown, round after round. The 36 lesson files, in order
([sources.json](sources.json)), were:

1. the dialogue and text files of the first experiments (`research/living_mind_2026_11_14/data/`, 5 files);
2. the seven school subjects of `research/living_mind_2026_11_22/t_school.py` (`subjects(0)`), written out by the
   recipe;
3. the 24 lessons of public knowledge in `research/living_mind_2026_11_22/dersler/`.

**Privacy check.** Every word the mind holds (1,282 symbols on its word channel) is a word of those files.

## How the study went

Counts only; the full record is [study.json](study.json):

- **Dialogues asked:** 440, of which 433 were right in the first round. Two lessons needed a second round: school
  biology (6, then 8 of 8) and the biology lesson (7, then 12 of 12). After the rounds, 440 of 440.
- **Examination lines never taught:** 6 of 41.
- **Asked again after all 36 lessons:** 433 of 440. Two lessons lost answers:
  - the first dialogues lost 2 of 5: "kedi ne yapar" was first taught "kedi miyavlar", and the biology lessons later
    told it what cats do as mammals;
  - school biology lost 5 of 8.
- **Size and cost:** 19,040 moments, 14,107 units, 132,229 ties; 1,006 s of one processor core.

## Files

| file | what it is |
|---|---|
| `cevahir-open.zihin.gz` | the mind, saved without its ledger, gzip-compressed (sha256 a72a5dac9ab47ebaa4c6b0bd66b70aa7616a85da869970be04b931da87c16704) |
| `cevahir-open.zihin.json` | its manifest: the sha256 of the unpacked file (c9cbafbe…), its substance digest, moments, units, ties |
| `raise_open.py` | the recipe |
| `sources.json` | the lesson files, in the order lived (`<generated>/` are the school subjects the recipe writes) |
| `study.json` | the study record: rounds, examination and review per lesson |

# The open Cevahir (24 November 2026)

`cevahir/` is Cevahir ([`living/cevahir.py`](../living/cevahir.py);
[MODEL.md, section 12](../research/living_intelligence_2026_11_24/MODEL.md)): Zihin's whole mind and the cortex of
the Living Intelligence study meeting in one harmony, with the hippocampus's episodes. It was raised from nothing on
the same public lessons as the open mind above, in the same order ([`raise_cevahir.py`](raise_cevahir.py)), and
studied as a teacher would: each lesson taught, its questions answered by Cevahir in conversation, a right answer
confirmed, a wrong one forgotten exactly and the answer shown, round after round (three at most), then every dialogue
asked again.

## Use it

```
python -m living --load minds/cevahir
```

On the first load its compressed stores are unpacked beside them (`zihin.zihin`, about 17 MB, and `core.bin`, about
106 MB; both ignored by git) and every file is checked against `manifest.json`. Loading takes about 12 s and an
answer about 0.7 s on one processor core. It speaks Turkish, the language of its lessons, keeps learning from what you
tell it, and answers `!dogru` / `!yanlis` (how its last answer was received), `!gecmis`, `!unut N`, `!uyu`,
`!kaydet FOLDER`.

## How the study went

Counts only; the full record is [study_cevahir.json](study_cevahir.json):

- **Dialogues asked:** 440, of which 435 were right in the first round. Two lessons needed a second round: school
  biology (6, then 8 of 8) and the biology lesson (9, then 12 of 12). After the rounds, 440 of 440.
- **Examination lines never taught:** 7 of 41 (the open mind above: 6 of 41).
- **Asked again after all 36 lessons:** 433 of 440. The same two lessons lost answers as for the open mind above: the
  first dialogues 2 of 5 and school biology 5 of 8.
- **Size and cost:** 16,825 moments of Zihin (14,042 units, 126,050 ties); the core holds 129,132 nodes and 157,418
  entries over 74 letters (55,971 letters lived); 959 s of one processor core.
- **Asked once more, as kept:** of 12 lesson questions drawn at random, 11 are answered with the lesson's sentence
  (without its final full stop); the twelfth, a coding order written with a leading "?", is met with silence.

**Privacy check.** Every word Cevahir holds in `cevahir.json` (1,264) is a word of the lesson files, and so is every
word symbol of its Zihin (1,302), apart from 17 names the program gives its own channels (where a thing was learned
from and what was felt: `told`, `inferred`, `surprise`, ...).

**Kept without the study's ledger.** The study kept Zihin's ledger in memory, so that a wrong answer could be
forgotten exactly. A loaded mind has none (its history begins at loading), so the mind was loaded once and kept again
without it: the same substance digest, core and `cevahir.json`, and 2.3 MB instead of 20 MB. Both keepings gave the
same answers to the 12 lesson questions and three greetings.

## Files

| file | what it is |
|---|---|
| `cevahir/zihin.zihin.gz` | Zihin, gzip-compressed (sha256 b64174dc…; unpacked a6e8c9c0…) |
| `cevahir/zihin.zihin.json` | its manifest: the sha256 of the unpacked file, its substance digest, moments, units, ties |
| `cevahir/core.bin.gz` | the cortex's core, gzip-compressed (sha256 fc341579…; unpacked cd2606c3…) |
| `cevahir/cevahir.json` | what Cevahir itself learned: the forms of words, the records of its voices, the episodes |
| `cevahir/manifest.json` | the sha256 of every file as it is used (unpacked) |
| `raise_cevahir.py` | the recipe |
| `study_cevahir.json` | the study record: rounds, examination and review per lesson |
