# Living-learning research — round 19

Muhammed Yasin Yılmaz · 2026-09-23 · Local research only.

Unknown binary finite-state worlds are learned using classical observation-table refinement. The state count and distinguishing suffixes are not supplied; the alphabet, reset/query interface, finite-state hypothesis and refinement algorithm are supplied. This is a controlled extension of the research, not a novel general learning principle.

- [Turkish report](../../docs/research/living_learning_distinctions_2026_09_23/REPORT_TR.md)
- [Protocol saved before performance runs](../../docs/research/living_learning_distinctions_2026_09_23/PROTOCOL.md)
- [Mathematics and checked primary literature](../../docs/research/living_learning_distinctions_2026_09_23/MATHEMATICS_AND_PRIOR_ART_TR.md)
- [Whole-problem assessment](../../docs/research/living_learning_distinctions_2026_09_23/WHOLE_PROBLEM_TR.md)

72 worlds (24 paired seeds × 3 families), 4 methods, 288 paths, 4,608 block snapshots and 18,432 passive predictions. Exact equivalence is checked on the reachable product of target and learned machines, without feeding its counterexamples to the learner.

Results: adaptive and same-evidence rebuild each identify 57/72 worlds exactly; exhaustive continuation coverage identifies 58/72; fixed-test ablation identifies 0/72. Adaptive/rebuild states, query sequences and evidence agree in all 1,152 paired blocks, with less recomputation for the persistent method. Four wrong adaptive models pass every sampled long test. All successful final models are minimal under an independent partition check.

Reproduction from repository root (Python standard library only):

```powershell
python research/living_learning_distinctions_2026_09_23/verify_distinctions.py --rerun
python research/living_learning_distinctions_2026_09_23/analyze_results.py
python research/living_learning_distinctions_2026_09_23/diagnose_results.py
python research/living_learning_distinctions_2026_09_23/final_audit.py
```

The verifier includes the full exact rerun. Running `distinction_learning.py` without flags also checks exact reproduction. `--write` is only for intentional creation/replacement of this round's main result and is unnecessary for verification.

- [Raw experiment](results/distinction_learning.json)
- [Summary](results/summary.json)
- [Independent verification and 24 fresh-process restarts](results/verification.json)
- [Post-run diagnosis](results/diagnostics.json)
- [1,023-file preservation manifest](results/preservation_baseline.json)
- [Final claim/hash audit](results/final_audit.json)

No path reached either resource limit. The report explicitly documents that the implemented environment limit rejects new uncached queries, while cache-only work may continue; the original protocol's automatic-freeze wording was broader. This boundary difference did not affect the recorded paths, and primary source/protocol/results were preserved. Internal work is a declared operation model, not CPU/RAM. The full query history remains stored; memory-constrained or archive-free learning was not tested. No commit, push, publication or main-app/book change.
