# Living learning: inferred context, confounded cues, self-selected experience

Round 14 · 2026-09-22 · Muhammed Yasin Yılmaz · Local research

[Türkçe rapor](../../docs/research/living_learning_context_2026_09_22/REPORT_TR.md) · [Ana protokol](../../docs/research/living_learning_context_2026_09_22/PROTOCOL.md) · [Aktif deneyim protokolü](../../docs/research/living_learning_context_2026_09_22/ACTIVE_EXPERIENCE_PROTOCOL.md) · [Matematik denetimi](../../docs/research/living_learning_context_2026_09_22/MATHEMATICAL_AUDIT_TR.md) · [Sürekli sistem hedefi](../../docs/research/living_learning_context_2026_09_22/CONTINUOUS_SYSTEM_TARGET_TR.md) · [Yeni kuramın yenilik sınırı](../../docs/research/living_learning_context_2026_09_22/THEORY_NOVELTY_BOUNDARY_TR.md)

This experiment does not discover its representation language or claim a new learning principle. Supplied four-bit history summaries and affine-parity expert/gate languages make context attribution tractable. Perfectly confounded acquisition data cannot distinguish history and surface explanations. Subsequent separating evidence changes persistent beliefs and held-out predictions. An independently planned extension lets the learner select new queries through a supplied noiseless labeling interface.

## Reproduction

Python 3.14.3, NumPy 2.4.2. No API, GPU, external model or main-system integration. Original runs used one OpenBLAS thread for reproducibility and to avoid oversubscription:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python research/living_learning_context_2026_09_22/verify_context.py --structural-only
python research/living_learning_context_2026_09_22/context_learning.py
python research/living_learning_context_2026_09_22/active_experience.py
python research/living_learning_context_2026_09_22/analyze_results.py
python research/living_learning_context_2026_09_22/verify_context.py --rerun
python research/living_learning_context_2026_09_22/verify_active.py --rerun
python research/living_learning_context_2026_09_22/final_audit.py
```

Experiment and summary commands compare stored results by default; explicit `--write` generated initial records. Verification commands write their audit records. Exact JSON replay assumes the same numerical runtime; other BLAS versions or architectures may require a documented tolerance comparison. Preservation checks refer to the recorded local workspace and HEAD; they are not a rule against later intentional edits.

## Files

- [context_learning.py](context_learning.py): supplied function bank, ordinary and privileged-oracle learners, 72 worlds, 504 primary trajectories.
- [active_experience.py](active_experience.py): 96 extra query-selection trajectories; labels revealed only after selection.
- [verify_context.py](verify_context.py): all 8,388,608 candidate-output values checked independently, prior/odds/world invariants, snapshots, 21 fresh-process continuations and full replay.
- [verify_active.py](verify_active.py): query selection and scalar outcomes, independent greedy-score calculation, four continuations including RNG/seen state, and full active replay.
- [analyze_results.py](analyze_results.py): paired descriptive summaries; 24 lives are the comparison unit.
- [results/context_learning.json](results/context_learning.json): worlds, snapshots, prequential probabilities, final-state hashes.
- [results/active_experience.json](results/active_experience.json): chosen experience traces, costs and snapshots.
- [results/paired_summary.json](results/paired_summary.json): main and active results kept distinct.
- [results/verification.json](results/verification.json), [active verification](results/active_verification.json): replay and correctness evidence.
- [results/preservation_baseline.json](results/preservation_baseline.json): 941 previous file digests and HEAD.
- [results/final_audit.json](results/final_audit.json): source/result/document integrity record.

## Limits needed to interpret the results

The history vector is provided and is not extracted from a running temporal stream. The true context is withheld from ordinary learners but supplied to the oracle control. History-only models receive privileged exclusion of surface gates. Priors are over 16,384 parameter representations, not 15,904 distinct functions. Positive `.05` likelihood noise is a fixed learner assumption despite clean environment labels; the evaluated prediction is the clean latent function mean.

The entire 512-input evaluation includes seen inputs; unseen subsets are separately reported. Full-table counts retain experience information even though raw rows are absent. The shared supplied float64 bank occupies 67,108,864 bytes and is excluded from learner JSON size; table methods do not intrinsically require it. Query selection can inspect all 512 candidate inputs and request a correct label for any of them; this interface is not discovered or necessarily available in realistic environments.

After 16 active queries, uncertainty-based selection scores 134,217,728 candidate-input cells versus 262,144 selected-input prediction cells in the uniform control. Better external-sample efficiency is not equal total computation or an energy advantage. Every learner's 64-label candidate-update cost is also reported. The new formal notations are scope definitions, not proofs that the general living-learning problem is solved.

No previous result, book, system file, commit, release or publication was changed.
