# Experience-conditioned computation: provenance-aware feedback and execution limits

**Muhammed Yasin Yılmaz**

CEV-2026-01 · Version 2026.11.24 · Published 2026-11-24

Research report / preprint. Not peer reviewed.

## Abstract

We implement provenance-aware feedback, a shared request budget, and experience-conditioned strategy selection. Persistent experience statistics are distinct from weight training. Synthetic table replay tests mechanism behavior, not real-model quality or efficiency. Model, tokenizer, task and feedback identities constrain admissible comparisons.

## Interpretation of the findings

Experience can influence the choice of future computation through a persistent record. There is no evidence for presenting this effect as general learning or as a gain in real task performance.

The routing groups are heuristic; the outcomes of untried actions cannot be inferred from the logs of chosen actions. A fixed MoE prior is not a learned expert meaning.

## Publication and evidence

This edition presents the research record below with its method, results, negative controls and source links, or
where that record was withdrawn, the note that says where its full text is archived. The original file is kept as
[docs/research/EXPERIENCE_CONDITIONED_COMPUTE.md](../../research/EXPERIENCE_CONDITIONED_COMPUTE.md); links are rebased to this edition's location. Dates and earlier
publication statuses inside the record belong to the historical record; this edition's date is above.

This work is a separate text in the [publication series](../README.md). The DOI identifies the release archive that
holds every text and the code; it does not mean that each paper has a DOI of its own. Cite the work's identifier
together with the release DOI. The scope of the computational checks is described in the
[validation record](../evidence/validation.json) and the [publishing principles](../PUBLISHING.md).

The project and its research direction are Muhammed Yasin Yılmaz's. AI assistance was used for code, computational
experiments, literature review and text preparation. Neither this statement, the DOI record nor the automated tests
are external scientific peer review.

---

## Research record: Withdrawn

This document was withdrawn from the repository on 20 November 2026 by its author. Its full text is archived in
release v6 (tag `research-2026.11.16-v6`, the same path), DOI
[10.5281/zenodo.24279940](https://doi.org/10.5281/zenodo.24279940).

The work is described in English in [Cevahir Zihin: a Living Evidence Mind](../../research/living_evidence_mind_2026_11_24/PAPER.md).
