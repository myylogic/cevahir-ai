# Living learning: separately acquired relations and new composition

Local research round 12 · 2026-09-22 · Muhammed Yasin Yılmaz

[Türkçe ana rapor](../../docs/research/living_learning_relations_2026_09_22/REPORT_TR.md) · [Protokol](../../docs/research/living_learning_relations_2026_09_22/PROTOCOL.md) · [Matematik denetimi](../../docs/research/living_learning_relations_2026_09_22/MATHEMATICAL_AUDIT_TR.md) · [Literatür](../../docs/research/living_learning_relations_2026_09_22/LITERATURE_TR.md) · [Yeni kullanıcı gözlemi ve açık hipotezler](../../docs/research/living_learning_relations_2026_09_22/NEXT_HYPOTHESES_TR.md)

This finite experiment studies transfer of observation-derived relation classes to independently renamed target actions, held-out compositions, and failures when the relation family changes or acquires a local exception. It does not establish a general living-learning principle, AGI, consciousness, or a biological mechanism. Object identities, the hypothesis language, and the composition rule are supplied.

## Local reproduction

From the repository root, use Python with NumPy. Primary run and full replay were executed with Python 3.14.3 and NumPy 2.4.2 on this machine. No trained model, API, network service, or GPU is used for the experiment. The optional plot needs Matplotlib.

```powershell
python research/living_learning_relations_2026_09_22/verify_relations.py --structural-only
python research/living_learning_relations_2026_09_22/relation_learning.py
python research/living_learning_relations_2026_09_22/analyze_results.py
python research/living_learning_relations_2026_09_22/structural_witness.py
python research/living_learning_relations_2026_09_22/verify_relations.py --rerun
python research/living_learning_relations_2026_09_22/final_audit.py
```

`relation_learning.py`, `analyze_results.py`, and `structural_witness.py` check existing records by default. Their explicit `--write` option initially generated the respective result. Verification writes its own audit file; final audit writes only its own audit record. Rebuilding a stored result deliberately changes its provenance and may require a fresh verification. The preservation check is relative to this workspace and the recorded HEAD; it is not intended to claim that future intentional repository edits are invalid.

```powershell
python research/living_learning_relations_2026_09_22/plot_results.py
```

The default and bundled Python runtimes lacked Matplotlib. Plot dependencies are isolated in a temporary directory; the project and system Python package installations are not changed. Figure rendering does not affect numerical learning results. Metadata in regenerated graphics can differ.

## Files

- [relation_learning.py](relation_learning.py): source observations, candidate construction, eight learners, 72 target worlds, 576 trajectories; hidden truth is confined to the evaluator.
- [verify_relations.py](verify_relations.py): independent class checks, renaming invariance, composition witnesses, scalar labels, uninterrupted versus fresh-process continuation, preservation and exact full replay.
- [analyze_results.py](analyze_results.py): paired 24-life descriptive comparisons. First exact composition checkpoint is an additional descriptive diagnostic, not a fitted success threshold.
- [structural_witness.py](structural_witness.py): post-result rational-arithmetic witness; same current marginal predictions need not imply the same response to new evidence.
- [plot_results.py](plot_results.py): standalone PNG/SVG plot from recorded results.
- [results/relation_learning.json](results/relation_learning.json): all worlds, all timepoint metrics, prequential probabilities and state hashes.
- [results/paired_summary.json](results/paired_summary.json): averages, paired better/tied/worse counts, complete-composition diagnostic.
- [results/verification.json](results/verification.json): full rerun and correctness checks, including 24 separate process restarts.
- [results/structural_witness.json](results/structural_witness.json): exact `0` versus `1` learning response after identical evidence.
- [results/composition_curves.png](results/composition_curves.png), [SVG](results/composition_curves.svg): descriptive mean learning curves.
- [results/preservation_baseline.json](results/preservation_baseline.json): SHA256 snapshot of 904 pre-existing files and unchanged Git HEAD.
- [results/final_audit.json](results/final_audit.json): final source/result/document integrity audit.

## Main limitations

The actions have independent learner posteriors, although compatible world generation ensures one action of each source type; that privileged dependency is deliberately withheld. Mean transition-matrix multiplication is exact for two **different** actions under this learner model, not for reusing the same unknown action. Source acquisition and target acquisition have different costs. Candidate-filter counters measure logical active-candidate checks, not actual NumPy comparisons or complete runtime. The frozen control intentionally rejects contradictory evidence while preserving a factual audit table; predictions do not silently use that table to repair contradictions.

Previous research and the book remain untouched. This round is local only: no commit, push, release, or DOI publication.
