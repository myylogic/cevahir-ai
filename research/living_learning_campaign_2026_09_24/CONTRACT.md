# Campaign module contract (round 23)

Local-only, Python standard library (NumPy permitted but unnecessary), deterministic.
Modules live beside this file. Each module owns its code and a catalog JSON file.

Required exports:

- `hypotheses()` returns list of dicts: id, family, title_tr, prediction_tr, intervention_tr, falsifier_tr, status (`wave1` or `queued`), prior_art (known-family description; no novelty claim). Each wave1 item additionally has metric, direction=`lower`, contrast={candidate, control}, regime, practical_margin (>=0, in metric units). A claim is conditional on the stated regime; count families separately from conditional claims.
- `run(seed: int)` returns `{hypotheses: {id: {candidate: float, control: float, costs: {candidate: number, control: number}, diagnostics: dict}}, checks: {str: bool}, meta: dict}` for all wave1 items. Lower metric is always preferable. All metrics bounded or range stated in catalog. Costs virtual and explicit, not wall time. Diagnostics include memory and errors where relevant. No NaN/Inf. Reuse same generated stream across candidate/control; avoid hidden target access.
- `selfcheck()` returns `{checks: int, passed: bool, details: ...}`. Use at least an independent mathematical/scalar reference and a mechanism ablation. Include deterministic serialized-state continuation if stateful model allows it; no fake assertions.

Catalog: 20 meaningful research propositions per assigned family. At least 12 wave1 executable propositions for primary implemented family. Additional families may remain queued and explicitly non-executable. Write catalog JSON before performance runs. Do not count seeds or parameter points as hypotheses. Mechanism/regime conditional contrasts can be claims, but registry must retain their grouping and not sell them as independent principles.

Engine training generates outcomes; evaluation hidden truth must not feed learners. Candidate/control compute includes their actual selected operations; do not grant free offline oracle selection. Any oracle is labeled diagnostic, never primary comparator. A lower error alone does not demonstrate resource superiority. Store all outcomes including negative and ties.

Orchestrator runs discovery seeds 0..23 and untouched confirmation seeds 1000..1047. Code and catalog hashes freeze before either wave; no hyperparameter adaptation between phases. Paired sign tests concern synthetic seed sampling only; familywise Holm adjustment across all executed hypotheses, practical margin alongside. Whole-life metric means prequential/held-out metric declared per family, acquisition cost separate. Hashes, rerun, preservation and catalog coverage checked by root.
