# The open mind (release v7)

`cevahir-open.zihin.gz` is the program's **whole mind** (every mechanism of the full configuration on, silent claims
too), raised from nothing on public material only. Minds raised on the author's private documents are not in this
repository and never will be.

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
