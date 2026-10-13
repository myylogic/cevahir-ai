# Living-learning research — round 20

Muhammed Yasin Yılmaz · 2026-09-23 · Local research only.

Actual round-19 learned hypothesis revisions are installed into a separate, never-reset live stream. Six state migration policies separate replay-state containment, hypothesis error, confident error and abstention. The reset-based calibration channel remains; this is not single-channel no-reset learning or a new fundamental principle.

- [Turkish report](../../docs/research/living_learning_live_revision_2026_09_23/REPORT_TR.md)
- [Preperformance protocol](../../docs/research/living_learning_live_revision_2026_09_23/PROTOCOL.md)
- [Mathematics and primary literature](../../docs/research/living_learning_live_revision_2026_09_23/MATHEMATICS_AND_PRIOR_ART_TR.md)
- [Whole-problem assessment](../../docs/research/living_learning_live_revision_2026_09_23/WHOLE_PROBLEM_TR.md)

72 source worlds, 2 live-stream strides, 6 methods: 864 paths and 235,008 prediction points. Source models include failures. The learned tables change only 70 times across the 72 source histories; this becomes 140 migration events per method over two strides. These are paired comparisons, not 864 independent worlds.

Main negative result: old-model relation plus an eight-bit trace gives exactly the same state sets and predictions as parity plus the eight-bit trace alone, at additional virtual work. Resetting the working hypothesis to its initial state creates 713 errors even when its program is globally correct. Five replay-sound methods always contain the program's full-history state; this does not make incorrect hypotheses correct.

Reproduce from repository root (Python standard library):

```powershell
python research/living_learning_live_revision_2026_09_23/verify_live_revision.py --rerun
python research/living_learning_live_revision_2026_09_23/analyze_results.py
python research/living_learning_live_revision_2026_09_23/diagnose_results.py
python research/living_learning_live_revision_2026_09_23/final_audit.py
```

The verifier includes full exact rerun, 18 state-renaming paths, and 36 fresh-process continuations. `live_revision.py` without flags also checks exact reproduction. `--write` intentionally creates/replaces this round's main result and is not needed for verification. The unchanged round-19 engine and raw result are dependencies; their hashes are recorded.

- [Raw experiment](results/live_revision.json)
- [Summary](results/summary.json)
- [Verification](results/verification.json)
- [Revision/error diagnostics](results/diagnostics.json)
- [1,039-file preservation manifest](results/preservation_baseline.json)
- [Final audit](results/final_audit.json)

Costs include common source calibration separately from added migration/live work. Counts are declared virtual operations, not complete CPU/RAM accounting. No global resource cutoff was imposed. Full archive retains 528 or 4,112 main-stream bits; suffix methods retain eight; all methods still require their programs, belief sets, parity and common calibration state. No main-app/book edits, commit, push or publication.
