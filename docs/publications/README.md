# Cevahir — Research Reports

**Muhammed Yasin Yılmaz** · research reports and preprints · release v7, 24 November 2026

**DOI of all versions: [10.5281/zenodo.22864660](https://doi.org/10.5281/zenodo.22864660)** · the DOI of this
version is recorded in [release.json](release.json) once the archive is published.

This series collects the papers of the Cevahir research, with their negative results and reproducible
computations. Its question is still open: **how can a system change its future computation and behaviour through the
experiences it lives, persistently, under control and in a way that generalises?**

The texts are research reports that have not been externally peer reviewed. Each edition holds an abstract, an
interpretation of the findings and the research record. Source code, measurements, counter-examples and literature
links are kept. A supplied feature language, privileged feedback or a small finite world is not taken to mean that
general learning is solved.

## The series

| Identifier | Work | Main result |
|---|---|---|
| CEV-2026-01 | [Experience-conditioned computation](papers/01-experience-conditioned-computation.md) | Strategy choice from persistent external feedback; no real quality gain measured yet. |
| CEV-2026-02 | [Insufficiency witnesses in representation search](papers/02-representation-witness-negative.md) | A claimed learning advantage was eliminated: the same complete search. |
| CEV-2026-03 | [Persistent procedures, transfer and retesting](papers/03-persistence-transfer-retesting.md) | Knowledge, access cost and benefit under change were separated. |
| CEV-2026-04 | [Representation, credit and state transport](papers/04-representation-credit-state-transport.md) | Keeping today's answer may not keep future learning. |
| CEV-2026-05 | [One lifetime stream and temporal relations](papers/05-joint-lifetime-temporal-relations.md) | Combined mechanisms tested together under errors and distribution shift. |
| CEV-2026-06 | [Update behaviour acquired from experience](papers/06-learned-update-transfer.md) | Different learning curves from the same first prediction; transfer can reverse. |
| CEV-2026-07 | [An acquired recurrent state procedure](papers/07-acquired-recurrent-state.md) | A small execution graph separated from the relearning archive. |
| CEV-2026-08 | [Correctable learning state](papers/08-correctable-learning-state.md) | Even today's complete candidate set may not suffice for correction. |
| CEV-2026-09 | [Future queries and the state they require](papers/09-future-query-state-classes.md) | Different information bounds for the same history, by query and correction rights. |
| CEV-2026-10 | [A multi-directional campaign for learning while living](papers/10-living-learning-campaign-wave1.md) | 48 conditional comparisons in four engines: 30 supported, 9 contradicted, 9 inconclusive. |
| CEV-2026-11 | [Cevahir Zihin: a Living Evidence Mind](papers/11-living-evidence-mind.md) | A mind of evidence with no network: SCAN's six published test files 26,227 of 26,227; real-text recall and a grounded layer against Transformers trained on the same data. |

The research records of CEV-2026-01 to CEV-2026-10 were written in Turkish. Their full texts were withdrawn from the
repository on 20 November 2026 and remain in release v6
([DOI 10.5281/zenodo.24279940](https://doi.org/10.5281/zenodo.24279940)). Their editions here keep the English
abstract, the interpretation of the findings and the note that says where the full text is.

## Reading the evidence

- The [machine-readable catalog](catalog.json) links every edition to its research record.
- The [preservation manifest](evidence/prior_manifest.json) holds the byte digests of the research files published
  before the series. Files withdrawn in v7 are listed there with the digest of their archived text.
- The [validation record](evidence/validation.json) separates what was re-run when the first archive was prepared
  from earlier run records.
- A result file being kept does not mean that the experiment was run again on this date.

## Citation and archive

The collection's machine-usable citation is in [CITATION.cff](../../CITATION.cff). Cite a work as:

> Yılmaz, M. Y. (2026). *Title* (CEV-2026-XX). In *Cevahir Zihin and Cevahir AI: research reports*, version
> 2026.11.24. Zenodo.

A DOI identifies a specific archive version of the content. It is not a confirmation of scientific correctness or
originality, a personal identity document, or a cryptographic signature. The Zenodo record is a software and research
archive; it is not claimed that each text of the series has a DOI of its own.

Every earlier release keeps its record: [v1](releases/research-2026.09.21-v1.json),
[v2](releases/research-2026.09.21-v2.json), [v4](releases/research-2026.10.13-v4.json),
[v5](releases/research-2026.11.14-v5.json), and v6 in its [release note](releases/research-2026.11.16-v6.md). The
archive verifications and DOI resolution checks are in [evidence/](evidence/).

## Contribution statement

The project and its research direction are Muhammed Yasin Yılmaz's. AI assistance was used for code, experiments,
literature review and text preparation. No external peer review, institutional affiliation or ORCID verification is
implied. The licence is the repository's [Apache-2.0](../../LICENSE). The [publishing principles](PUBLISHING.md)
record how each new result becomes a release.
