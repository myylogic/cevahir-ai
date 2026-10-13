# Living-learning research — round 18

Author: Muhammed Yasin Yılmaz. Date: 2026-09-22. Local research only.

The experiment reuses the rule acquired in round 17 and compares measured single/pair edit effects, persistent versus reset model-order choice, and exact subset search. It is a controlled F17 program experiment, not a new general learning principle.

- [Turkish report](../../docs/research/living_learning_interactions_2026_09_22/REPORT_TR.md)
- [Protocol fixed before the performance run](../../docs/research/living_learning_interactions_2026_09_22/PROTOCOL.md)
- [Mathematics and primary-source boundaries](../../docs/research/living_learning_interactions_2026_09_22/MATHEMATICS_AND_PRIOR_ART_TR.md)
- [Whole problem and next discriminating tests](../../docs/research/living_learning_interactions_2026_09_22/WHOLE_PROBLEM_AND_NEXT_TEST_TR.md)

Run from the repository root with Python and NumPy:

```powershell
python research/living_learning_interactions_2026_09_22/interaction_learning.py
python research/living_learning_interactions_2026_09_22/verify_interactions.py --rerun
python research/living_learning_interactions_2026_09_22/analyze_results.py
python research/living_learning_interactions_2026_09_22/diagnose_coverage.py
python research/living_learning_interactions_2026_09_22/final_audit.py
```

The first command without `--write` checks exact reproduction against the existing record. `--write` is for explicit result creation and would replace this round's result; it is unnecessary for verification. The verifier includes the full rerun, so those two commands need not both be run for routine checking.

Dependencies: unchanged round-17 `repair_learning.py` and its `results/repair_learning.json`. Their hashes and the primary source/protocol hashes are recorded in the main result.

Artifacts:

- [Raw experiment](results/interaction_learning.json): 24 paired lives, 168 paths, 2,688 target attempts.
- [Summary](results/summary.json): phase counts, virtual work and paired-life differences.
- [Verification](results/verification.json): scalar probe checks, complete finite-domain acceptance checks, fresh-process persistence and exact rerun.
- [Post-run coverage diagnosis](results/coverage_diagnostics.json): explains required interaction orders without changing the solver or giving it target masks.
- [Preservation manifest](results/preservation_baseline.json): 1,007 prior file hashes and unchanged HEAD.
- [Final audit](results/final_audit.json): claim checks and artifact hashes.

Primary outcome: first-order 210/384; second-order and both adaptive policies 372/384; exact subset 384/384. Persistent choice is more costly than resetting in all 24 whole lives, with identical accepted mask sequences. No false acceptance or budget exhaustion was observed. The stored program library is not reused for subsequent search in this round; the persisted model preference is one bit. No commit, push or publication.
