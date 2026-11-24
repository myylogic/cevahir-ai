# Research release 2026.11.24 (v7)

- Git tag: `research-2026.11.24-v7`, a GitHub Release from main. It is made by the author; Zenodo archives it and
  gives the version DOI, which is then recorded in [release.json](../release.json) and CITATION.cff.
- All versions: DOI [10.5281/zenodo.22864660](https://doi.org/10.5281/zenodo.22864660). The previous version is v6,
  DOI [10.5281/zenodo.24279940](https://doi.org/10.5281/zenodo.24279940); its record is in
  [research-2026.11.16-v6.json](research-2026.11.16-v6.json).

## What is new

- **The paper.** *Cevahir Zihin: a Living Evidence Mind* (CEV-2026-11,
  [PAPER.md](../../research/living_evidence_mind_2026_11_24/PAPER.md)) describes the whole work in English. It covers
  the core and its one rule, the 75 mechanisms and the ones not kept, and every evaluation against Transformers trained
  from scratch on the same data.
- **The work since v6.** These are recorded step by step in `research/living_mind_2026_11_22/` (sections 1 to 43) and
  `research/living_mind_2026_11_23/`:
  - F41 to F75, and the whole mind (everything built on by default);
  - the six published SCAN test files, 26,227 of 26,227; COGS measured at 0 with its causes;
  - recall and a race on real text (146 to 151 of 200 after one reading, against 47 for a Transformer after an hour);
  - speed with nothing changed (median answer 2.70 s → 0.64 s);
  - the study and the school examination;
  - the grounded meaning layer (who does what, 48 and 54 of 72, against 29 and 42 for a Transformer given the scene).
- **An open mind.** [minds/cevahir-open.zihin](../../../minds/README.md) is the program's whole mind raised from
  nothing on public material only. Minds raised on the author's private documents are not in the repository.
- **English editions.** The ten earlier reports have English editions: abstract, interpretation of the findings and
  the note on where the research record is. The edition template is English.

## Withdrawn

On 20 November 2026 the author withdrew the Turkish documents from the repository: the research records of
CEV-2026-01 to CEV-2026-10, the Turkish technical book, the Turkish theory and guide, and other Turkish notes.

In this release each of them is replaced by a note giving the release that holds its full text: v6, at the same
path. Two exceptions:

- The two Turkish help topics of the console were removed; the console shows the English topics in their place.
- README-TR.md is kept as its author made it.

The preservation manifest lists the 38 preserved research files that were withdrawn, with the digests of their
archived texts. The publication check requires each of them to be exactly its note.

## Checks

The following pass on the release commit:

- `python scripts/check_book.py` (the reviewed-source record renewed after the book's withdrawal);
- `python scripts/build_publications.py`;
- `python scripts/check_publications.py`;
- the test suite of `zihin/`.

The gates of the core (store digest, plain run, working memory, the 17 configurations) are bit-identical with the
grounded layer off.
