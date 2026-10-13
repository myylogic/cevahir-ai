# Round 24 / campaign wave 2 — module contract

Keep every previous file immutable. Work only in the new wave2 directory.
Local-only; no commit, push, publish or app/book integration. Python stdlib or NumPy.

Three engines: context, credit, capacity. Eight executable conditional comparisons
per engine, 24 total; these are not 24 independent discoveries. Do not inflate seeds,
metrics or regime variants into mechanisms. Link each comparison to `source_ids`
from the 200-entry previous registry and state `coverage_note`: partial operational
test, not blanket completion of that original proposition. Old registry stays frozen.

Required exports:
- hypotheses(): list of 8 dicts with id (CTX2-01 / CRD2-01 / CAP2-01 etc), family,
  title_tr, prediction_tr, intervention_tr, falsifier_tr, prior_art, status='wave2',
  metric, metric_range=[lo,hi] or descriptive string, direction='lower',
  contrast={candidate,control}, regime, practical_margin>=0, mechanism_group,
  source_ids:list, coverage_note:str.
- run(seed): {hypotheses:{id:{candidate:number,control:number,
  costs:{candidate:number,control:number},diagnostics:dict}},checks:{str:bool},meta:dict}.
  All scores lower-better, finite; report acquisition, lifetime, memory and phase
  effects where material. Equal observations across methods unless explicitly a
  tested information intervention. Hidden truth/eval probes never learner-visible.
- selfcheck(): {checks:int,passed:bool,details:...}. Use independent scalar/exact
  reference, actual mechanism ablation, serialized continuation and information
  collision counterexample when appropriate. Checks are not performance trials.

Write catalog_<engine>.json and <ENGINE>_PROTOCOL.md BEFORE performance runs.
No performance runs until root freezes all sources. Selfchecks okay with dedicated
fixed mechanics fixtures, not the main seeds. Root runs discovery 2000..2031 (32)
and confirmation 3000..3063 (64). No outcome-dependent tuning or stopping.
Whole wave: 288 engine-lifetimes and 2304 paired outcomes. 24*2=48 one-sided paired
sign tests corrected together by Holm; margin on mean is descriptive extra filter.
Support/reverse requires adjusted p<=.05, mean delta >= margin in direction and
at least 2/3 of all confirmation seeds in that direction; otherwise inconclusive.
Wave2 is a new exploratory campaign after wave1; no global all-time FWER claim.

Prefer strong same-information controls. Teacher-provided context IDs, mappings,
operator banks, clocks and revisions must be explicit, not called discovered.
Charge learner acquisition/rebuild/selection/storage work with declared virtual
primitives. Costs aren't CPU time and not comparable between engines. No universal
or biological principle claims from a finite supplied language. Keep runtime small
(ideally under 1 sec per seed) without weakening substantive controls.
