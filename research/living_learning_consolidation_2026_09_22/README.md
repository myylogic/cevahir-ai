# Living learning: acquisition, consolidation, timing and information limits

Local round 13 · 2026-09-22 · Muhammed Yasin Yılmaz

[Türkçe rapor](../../docs/research/living_learning_consolidation_2026_09_22/REPORT_TR.md) · [Sonuç öncesi protokol ve kullanıcı açıklaması](../../docs/research/living_learning_consolidation_2026_09_22/PROTOCOL.md) · [H18–H41](../../docs/research/living_learning_consolidation_2026_09_22/HYPOTHESES_TR.md) · [Koşullu matematiksel kanıtlar](../../docs/research/living_learning_consolidation_2026_09_22/MATHEMATICAL_LIMITS_TR.md) · [Biyolojik kaynaklar](../../docs/research/living_learning_consolidation_2026_09_22/BIOLOGY_AND_LITERATURE_TR.md)

This is a finite learning experiment, not a neural sleep simulation or an AGI claim. A supplied parity language supports temporary episode caching, delayed and online error-count updates, context-visible/hidden tasks, corrupted observations and replay controls. It does not learn its own representation language or context boundaries. The primary positive result depends on a hypothesis language containing the target and a visible context routing signal. Old replay is unnecessary in that separated-context setting; this negative result is retained.

## Reproduction

Python 3.14.3 and NumPy 2.4.2 were used. From repository root:

```powershell
python research/living_learning_consolidation_2026_09_22/verify_consolidation.py --structural-only
python research/living_learning_consolidation_2026_09_22/consolidation.py
python research/living_learning_consolidation_2026_09_22/analyze_results.py
python research/living_learning_consolidation_2026_09_22/exact_witnesses.py
python research/living_learning_consolidation_2026_09_22/verify_consolidation.py --rerun
python research/living_learning_consolidation_2026_09_22/final_audit.py
```

The main experiment, summary, and exact witness check stored results by default; explicit `--write` generated their original records. Verification and final audit write their own audit records only. No external service or GPU is used. Preservation checks intentionally refer to the recorded local workspace and HEAD, not an assertion that future intended edits must never occur.

## Artifacts

- [consolidation.py](consolidation.py): supplied 14/84-candidate languages, source acquisition, eight learning methods, 72 worlds and 576 trajectories.
- [verify_consolidation.py](verify_consolidation.py): scalar semantic checks, equal-schedule counts, contradictory-context lower bound, isolated snapshots, episode purge and 24 process continuations.
- [analyze_results.py](analyze_results.py): descriptive life-level results and paired comparisons.
- [exact_witnesses.py](exact_witnesses.py): post-result exact rational ridge check and a two-world ambiguity in the recorded noisy run. The selected failure is diagnostic, not a new benchmark.
- [results/consolidation.json](results/consolidation.json): complete worlds, trajectories, prequential probabilities, metrics and final learner states.
- [results/paired_summary.json](results/paired_summary.json): stage averages, paired counts and resources.
- [results/verification.json](results/verification.json): correctness and exact full replay record.
- [results/exact_witnesses.json](results/exact_witnesses.json): independently checkable arithmetic and ambiguity witness.
- [results/preservation_baseline.json](results/preservation_baseline.json): 924 previous file digests and unchanged HEAD.
- [results/final_audit.json](results/final_audit.json): final source/result/doc integrity record.

## Interpretation limits

`seen` and `unseen` are fixed eventual 24 training / 40 held-out input sets. At intermediate checkpoints they are not the prefix-specific seen/unseen partition. Prequential inputs are distinct, so an exact cache never hits before its label arrives. Final state stores learned candidate identities and sufficient error counts, not raw episodes. The fixed output bank is shared supplied code data and is not included in serialized learner-state byte counts. Memory is not a minimal representation: controls may allocate unused counters; no total-RAM efficiency claim follows.

Online and delayed rich learning both perform 4,032 candidate–example error checks beyond common source acquisition. Online selects the current best candidates more frequently, so complete CPU equality is not claimed. Update continuity after restart is tested with an explicitly shared diagnostic continuation operator; it is not an additional benchmark of the frozen control's second-day policy.

All earlier research and book files remain unchanged. Local work only; no commit, push or publication.
