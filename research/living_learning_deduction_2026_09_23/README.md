# Living-learning research — round 21

Muhammed Yasin Yılmaz · 2026-09-23 · Local research only

This round studies one component of the open problem: whether a running system can turn internally derived, independently checked consequences into later computational capacity without treating them as new world observations. It uses small guarded CNF worlds, changing premises, a proof checker independent of the solver, and a whole-life resource account.

- [Turkish protocol](../../docs/research/living_learning_deduction_2026_09_23/PROTOCOL.md)
- [Turkish report](../../docs/research/living_learning_deduction_2026_09_23/REPORT_TR.md)
- [Hypotheses](../../docs/research/living_learning_deduction_2026_09_23/HYPOTHESES_TR.md)
- [Mathematics and literature](../../docs/research/living_learning_deduction_2026_09_23/MATHEMATICS_AND_LITERATURE_TR.md)
- [Raw results](results/deduction.json)
- [Summary](results/summary.json)
- [Verification](results/verification.json)
- [Initial pre-audit run](results/initial_run/)

The final amended run has 24 paired seeds, 2,304 queries per method and seven methods. `scoped` retained only lemmas whose proof dependencies remained in the current base. `blind` deliberately retained them after premise removal and was unsound. `full_table_cached` is an audit-added stronger compilation control; the original six-method run remains under `results/initial_run/`.

Reproduce:

```powershell
python research/living_learning_deduction_2026_09_23/deduction.py
python research/living_learning_deduction_2026_09_23/analyze_results.py
python research/living_learning_deduction_2026_09_23/verify_deduction.py --rerun
```

The normal engine command does not write results; `--write` does. The verifier ran 1,000 randomized solver cases, 16,128 recorded answer checks, 815 certificate checks, 33,940 active-lemma truth checks and 2,304 exact replay pairs. Fresh-process continuation passed for three seeds, three learned methods and two cut points. No book or main application integration, commit, push or publication occurred.
