# Living-learning round 22 — corrected experiment and rejected claims

Muhammed Yasin Yılmaz · 2026-09-23 · Local only

The initial experiment was invalid as a whole-life comparison. Its engine/raw output remain for audit. The corrected experiment keeps exactly the same 24 worlds but repairs accounting, source parsing, comparison methods and validation. It is a **post-result correction**, not preregistration or independent replication.

- [Corrected Turkish report](../../docs/research/living_learning_crossmap_2026_09_23/REPORT_TR.md)
- [Superseding amendment](../../docs/research/living_learning_crossmap_2026_09_23/AMENDMENT.md)
- [Mathematics, primary literature and new hypotheses](../../docs/research/living_learning_crossmap_2026_09_23/MATHEMATICS_AND_NEXT_TR.md)
- [Original protocol, superseded where conflicting](../../docs/research/living_learning_crossmap_2026_09_23/PROTOCOL.md)

Use `corrected.py`, not the initial `crossmap.py`. `analyze_results.py` and `results/summary.json` are also initial invalid artifacts. Archives `results/invalid_initial_run/` and `results/correction_audit_v1/` preserve the actual earlier code/results, including errors.

Valid files:

| File | Purpose |
|---|---|
| [corrected.py](corrected.py) | Seven learners, raw observation boundary, meter, evaluator, process continuation |
| [crossmap_corrected.json](results/crossmap_corrected.json) | 24 paired worlds, 168 paths, 4,032 query records |
| [summary_corrected.json](results/summary_corrected.json) | Full scalar totals including all setup; probability scores and byte sensitivity |
| [verify_corrected.py](verify_corrected.py) | Independent backtracking maps, score checks, renaming, corruption, continuations |
| [verification_corrected.json](results/verification_corrected.json) | Final verification evidence |
| [tree_audit.json](results/tree_audit.json) | All 945 trees: at most two single-node classes, all four as unions, minimum six references |
| [model_scope_audit.json](results/model_scope_audit.json) | Extra generator-law information and dependent-query counterexample; post-result diagnostic |
| [preservation_baseline.json](results/preservation_baseline.json) | 1,075 earlier files and HEAD |
| [final_audit.json](results/final_audit.json) | Final artifact hashes and consistency checks |

Reproduce from repository root with Python standard library:

```powershell
python research/living_learning_crossmap_2026_09_23/verify_corrected.py --rerun
python research/living_learning_crossmap_2026_09_23/analyze_corrected.py
python research/living_learning_crossmap_2026_09_23/final_audit.py
```

`corrected.py` without flags independently reruns and compares canonical JSON. `--write` overwrites only its corrected raw result; ordinary verification needs no raw rewrite. Verifier and analysis write their own derived evidence. The verifier includes 84 fresh-process loads, 63 with nonempty continuation, 72 renamed paths, exact rational probabilities and prior-file hashes. `tree_audit.py` and `model_scope_audit.py` rebuild their separate diagnostics; final audit recomputes them directly.

Main finding: persistent enumeration, full reconstruction and compact completion return identical probabilities. Mean whole-life scalar work: 6,612 / 130,170 / 694. This is not an AGI principle, structural-alignment discovery, a new inference rule or wall-time claim. Direct identity bridges and the inference language are supplied. No book/main-app changes, commit, push or publication.
