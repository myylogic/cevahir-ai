# g5 — Prior-work digest: consolidation and timing (Tur 13), memory from a raw symbol stream (Tur 15), installing a learned rule in a live stream (Tur 16), acquired repair procedures (Tur 17)

Analyst digest for the new-core initiative (leave Transformer / embedding table / softmax-argmax; one process for learning and running; weights as graded beliefs; concepts formed when needed; imagination measured for reality; a way of constructing that improves and transfers).
Repository: `/workspace/cevahir-ai` (read-only; nothing was written there, no repo script was executed). All numbers below were read from the studies' own reports and re-read from their result JSON files; where I recomputed or inferred something it is marked **[analyst]**. Nothing heavy was run (JSON reads and a few exact-fraction checks only, §6).

Conventions (identical to the sibling file `g3-correction-query.md` in this folder; `g1` uses bracketed tags with the same meaning): **PROVED** = algebraic proof or exhaustive exact enumeration inside the stated finite scope. **SUPPORTED** = descriptive result over the study's 24 paired synthetic lives (no population claim, no confidence intervals in any of the four studies). **REFUTED / NOT SUPPORTED** = an exact counterexample or a measured contrary result. **SUGGESTED** = interpretation the study itself does not claim to have shown. **OPEN** = named open by the study.

The five design pillars are abbreviated P1 (learning = running), P2 (weights = graded beliefs), P3 (concepts formed when needed), P4 (imagine and measure reality), P5 (the way of constructing improves and transfers).

---

## 0. Source map, existence check, integrity

All four named folders exist under **both** roots. Chronological order (used below) is not the order of the task list: consolidation is round 13, temporal 15, migration 16, repair 17 (round 14, `living_learning_context_2026_09_22`, lies between and is not in this group).

| Folder | Round | `docs/research/<folder>/` | `research/<folder>/` |
|---|---|---|---|
| `living_learning_consolidation_2026_09_22` | Tur 13 | `REPORT_TR.md`, `PROTOCOL.md`, `HYPOTHESES_TR.md`, `MATHEMATICAL_LIMITS_TR.md`, `BIOLOGY_AND_LITERATURE_TR.md` | `consolidation.py`, `analyze_results.py`, `exact_witnesses.py`, `verify_consolidation.py`, `final_audit.py`, `README.md`, `results/{consolidation (5.7 MB), paired_summary, exact_witnesses, verification, final_audit, preservation_baseline}.json` |
| `living_learning_temporal_2026_09_22` | Tur 15 | `REPORT_TR.md`, `PROTOCOL.md`, `MATHEMATICAL_AND_LITERATURE_TR.md`, `LEARNING_STATE_TR.md` | `temporal_learning.py`, `analyze_results.py`, `verify_temporal.py`, `verify_learning_state.py`, `final_audit.py`, `README.md`, `results/{temporal_learning (10.2 MB), summary, verification, learning_state_verification, final_audit, preservation_baseline}.json` |
| `living_learning_migration_2026_09_22` | Tur 16 | `REPORT_TR.md`, `PROTOCOL.md`, `MATHEMATICAL_AND_LITERATURE_TR.md`, `CORRECTIONS.md`, `WHOLE_PROBLEM_AND_NEXT_TEST_TR.md` | `migration_learning.py`, `analyze_results.py`, `verify_migration.py`, `final_audit.py`, `README.md`, `results/{migration_learning (7.6 MB), migration_learning_initial_cost_omission, summary, verification, final_audit, preservation_baseline}.json`, `results/initial_source_cost_omission.txt` |
| `living_learning_repair_2026_09_22` | Tur 17 | `REPORT_TR.md`, `PROTOCOL.md`, `MATHEMATICS_AND_LITERATURE_TR.md` | `repair_learning.py`, `analyze_results.py`, `verify_repair.py`, `final_audit.py`, `README.md`, `results/{repair_learning (7.8 MB), summary, verification, final_audit, preservation_baseline}.json` |

Publication status: none of the four is in `docs/publications/` (no match for the folder names or round numbers there); each report calls itself "Yerel araştırma kaydı" (local research record) and states that no commit, push or publication was made (e.g. Tur 16: "commit, push veya yayın yapılmadı").

**What kind of systems these are.** None of the four contains a neural network, a gradient, an embedding or a softmax. All learners are exact, discrete and inspectable: integer counters, enumerated hypothesis tables, sets of possible states, expression trees with rewrite proposals. They already sit outside the Transformer paradigm in mechanism — but at toy scale (64 inputs; ≤ 8-bit contexts; 64 two-state automata; a 289-point finite field) and always inside a hypothesis language supplied by the researcher. Every report says so in its first lines (Tur 13: "Bu sonuç, beynin çalışma formülünün veya genel yapay zekânın bulunduğu anlamına gelmez"; Tur 15: "Ana yaşayan öğrenme problemi hâlâ çözülmedi"; Tur 16: "Ana yaşayan öğrenme problemi çözülmüş değildir"; Tur 17: "Genel yaşayan öğrenme problemi hâlâ açık").

**Shared experimental template (all four).** 24 "lives" = seeds 0–23; every method and regime runs on the same life (paired design); the independent unit is the life, and each report warns that the hundreds of trajectories are *not* independent samples. Statistics are descriptive only: means over 24 lives and counts of lives better/tied/worse. Protocol written before the performance results and hashed into the result file (`protocol_sha256`); the protocols are local, and Tur 13 says explicitly "dış bir ön kayıt değildir" (not an external pre-registration). Prediction is always produced before the label. Each study has (i) an exact full re-run, (ii) independent scalar re-implementations of the semantics, (iii) *fresh-process resume*: serialise the learner state, restore it in a new OS process, feed the same future and require bit-identical predictions and final state (24 / 21 / 24 / 15 such checks), (iv) a SHA-256 preservation baseline of all earlier files (924 / 961 / 977 / 994 files) and of HEAD `f4eb1dc550b3efc8b2a62fc1c5db262c804b219c`. Environment recorded in the audits: Python 3.14.3 (NumPy 2.4.2 where NumPy is used; Tur 16 uses the standard library only).

**[analyst] Number check.** I re-derived every table quoted below from `results/paired_summary.json` (Tur 13) and `results/summary.json` (Tur 15, 16, 17). All report figures match the JSON to the printed precision; I found no discrepancy between reports and result files.

**How the four chain.** Tur 13: what becomes durable, and is a separate offline phase needed? → Tur 15: drop the ready-made history summary; which *computation* over the past must be kept, and is the state that answers today enough to learn tomorrow? → Tur 16: drop the per-episode reset; a correctly learned rule must be *installed* into a running history. → Tur 17: can experience change not only the stored solution but the *way the next solution is constructed*? Each later report cites the earlier one as its premise.

---

## 1. `living_learning_consolidation_2026_09_22` — Tur 13: "Deneyimin kalıcı kurala dönüşmesi: konsolidasyon, zamanlama ve bilgi sınırları"

Files: `docs/research/living_learning_consolidation_2026_09_22/{REPORT_TR,PROTOCOL,HYPOTHESES_TR,MATHEMATICAL_LIMITS_TR,BIOLOGY_AND_LITERATURE_TR}.md`; `research/living_learning_consolidation_2026_09_22/consolidation.py` (250 lines), `exact_witnesses.py`, `verify_consolidation.py`, `results/*.json`.

### (1) Question
When events are recorded fast and integrated later, what exactly becomes durable — and is a separate offline ("sleep") phase, a replay of old episodes, a context signal, or a perfect supervisor *necessary* for that? (The round takes the owner's verbatim day/sleep/"üst görevciler … kusursuz" description, `PROTOCOL.md:9`, splits it into 24 hypotheses H18–H41 and tests six.)

### (2) Mechanism / state representation — symbolic hypothesis bank with integer error counts (no neural part)
- **World.** `x ∈ {0,1}^6`, 64 inputs (`consolidation.py:22-23`).
- **Supplied language.** All parities of ≤ 3 features with a bias bit: `PROGRAMS` (84 functions) with truth table `OUTPUTS[84×64]` (`:24-27`). "Narrow" language = the first 14 (2 constants + 12 single-feature rules), `:28`. The learner does not invent this language (`PROTOCOL.md:26`).
- **Durable state** (`Learner.__init__`, `:53-77`): `errors[2][|bank|]` (int64) = N(c,h) for the two observable contexts (`:67`); `core_ids[c]` = current minimal-error set (`:65`); `evidence_counts[c]`; cost counters (`learning_checks`, `selection_calls`, `synthetic_labels`). 168 integer cells = 1,344 bytes; serialised state ≈ 990 bytes after purge.
- **Episodic state:** `old_rows` (24 source triples) and `day_rows` (appended by `observe`, `:105`); both deleted by `purge_episodes` (`:128-130`).
- **Update rule** (`_add`, `:79-85`): `self.errors[context] += OUTPUTS[self.bank, x] != y` (`:83`), i.e. (`REPORT_TR.md:33-35`)
  `N_{t+1}(c,h) = N_t(c,h) + 1[c=c_t]·1[h(x_t) ≠ y_t]`.
- **Selection** (`_select`, `:87-92`): `core_ids[c] = {h : N(c,h) = min_h N(c,h)}` (`:91`).
- **Readout** (`predict`, `:94-100`): `p(1|x,c) = mean_{h ∈ core_ids[c]} h(x)` (`:96`) — all minimal-error candidates weighted equally, no complexity penalty; if the exact key `(c,x)` is in the day cache the cached label replaces the rule output (`:97-99`, a hard override).
- **Old rule is learned, not given:** the unique zero-error candidate among the 14 on the 24 source rows (`:60-64`); both contexts start from it (`:65`).
- **The eight methods differ only in *when* and *with which rows* `_add` is called** (`consolidate`, `:110-126`): `online_rich` adds every event as it arrives and re-selects (`:106-108`); the `sleep_*` variants add once at the boundary; `sleep_self_replay` replaces the day labels by the old rule's own outputs (`:117-119`, counted in `synthetic_labels`); `sleep_mispaired_replay` rotates the labels by one position (`:120-122`); `sleep_rich_recent` skips the old rows (`:123-124`).
- **Continuation operator** `continue_observed` (`:132-139`) adds future evidence to the retained counts with no raw episodes present.

### (3) Experimental design
- 24 lives × 3 regimes × 8 methods = **576 acquisition paths**; 7 snapshots each (day 0/6/12/18/24, consolidated-with-episodes, consolidated-no-episodes) = 4,032 state evaluations; 13,824 pre-label predictions (`REPORT_TR.md:23`).
- Per life (`make_world`, `:164-187`): old rule = one feature or its negation; new rule = parity of 2 or 3 features or its negation — old and new disagree on exactly 32 of 64 inputs (`verify_consolidation.py:62`); 24 source inputs (always including 0 and the six unit vectors, which make the old rule unique in the 14-language); 24 distinct "day" inputs from the new rule; **40 inputs never used in training**; 4 pre-drawn corrupt positions; 6 future rows.
- Regimes: `visible_context` (old = context 0, new = context 1, clean labels); `hidden_context` (both observed as 0, `:177` — same inputs and rules, only the distinction removed); `noisy_day` (context visible, 4 of the 24 new labels flipped; clean labels exist only in the evaluator).
- Methods (`PROTOCOL.md:40-51`): `frozen_core`, `cache_only`, `sleep_linear_replay` (14-language), `sleep_rich_replay` (84), `online_rich` (84), `sleep_rich_recent`, `sleep_self_replay`, `sleep_mispaired_replay`.
- Metrics: scalar MSE of p against the clean label on old-64 / new-64 / held-out-40 / seen-24, each for the durable core alone and for core+cache; accuracy with ties = ½; balanced (old+new)/2 MSE in the hidden regime; costs = candidate–example checks, selection calls, raw rows held, statistic cells, JSON bytes (explicitly not CPU/RAM).
- Statistics: descriptive (`analyze_results.py:13-16`: mean difference; better / tied / worse lives).
- Audits (`results/verification.json`, `results/final_audit.json`): exact full re-run; 5,376 candidate outputs and 9,216 world labels re-derived with an independent scalar parity; 73,728 final-query checks; 13,824 predictions verified to precede their label; order-reversal invariance of the counts (`verify_consolidation.py:88-93`); 24 fresh-process continuations after purge; state export/restore isolation — an aliasing risk in export/restore was found by independent code review and fixed, only in this round's files (`REPORT_TR.md:134`).

### (4) Results (after consolidation **and** removal of all raw episodes; durable core only; means over 24 lives; `results/paired_summary.json`, `REPORT_TR.md:54-63`)

Visible, clean context:

| Method | old-64 MSE | new-64 MSE | held-out-40 MSE | lives with the new rule exact | learning checks | selection calls |
|---|---:|---:|---:|---:|---:|---:|
| frozen_core | 0 | 0.500000 | 0.509375 | 0/24 | 0 | 0 |
| cache_only | 0 | 0.500000 | 0.509375 | 0/24 | 0 | 0 |
| sleep_linear_replay (14) | 0 | 0.455729 | 0.540611 | 0/24 | 672 | 2 |
| **sleep_rich_replay (84)** | 0 | **0** | **0** | **24/24** | 4,032 | 2 |
| **online_rich (84)** | 0 | **0** | **0** | **24/24** | 4,032 | 49 |
| sleep_rich_recent | 0 | 0 | 0 | 24/24 | 2,016 | 1 |
| sleep_self_replay | 0 | 0.500000 | 0.509375 | 0/24 | 4,032 (24 synthetic labels) | 2 |
| sleep_mispaired_replay | 0 | 0.439062 | 0.427955 | 0/24 | 4,032 | 2 |

Further figures:
- **Cache ≠ rule.** `cache_only` with its cache at day end: seen-24 accuracy 100 %, all-64 accuracy 68.16 %, held-out-40 accuracy 49.06 %; after the cache is removed all-64 falls to 50.00 % (`REPORT_TR.md:50`).
- **Service during the day** (pre-label true MSE): every deferred method 0.484375; `online_rich` 0.080072 (visible; better in 24/24 lives), 0.189135 (noisy), 0.456163 (hidden; better in 7 lives, tied in 17) (`REPORT_TR.md:73`). [analyst, from `consolidation.json`] in the clean visible regime the online core is exact on all 64 inputs in 6/24 lives after 6 day events and in 24/24 lives after 12 of the 24 events; under noise the count of exact lives is 2 / 8 / 17 / 23 after 6 / 12 / 18 / 24 events.
- **Online = deferred at the end.** In all 72 life × regime pairs the final error vectors, minimal-error sets and core predictions of `online_rich` and `sleep_rich_replay` are identical; both make 4,032 candidate–example checks beyond the common 336 (`REPORT_TR.md:71`; `verify_consolidation.py:206-212`).
- **Hidden context** (`REPORT_TR.md:81-86`): frozen old rule: old 0 / new 0.500000 / balanced 0.250000; rich replay = online: old 0.177083 / new 0.260417 / balanced **0.218750**; recent-only: old 0.500000 / new 0 / balanced 0.250000. Information floor 1/8 = 0.125 (below). [analyst] the minimal-error set is {old rule} in 11 lives, {new rule} in 7, a tie of both in 6; the minimal error count is 8–14 of 48 observations, never 0.
- **Noise** (4 of 24 labels flipped; `REPORT_TR.md:100-104`): rich replay = online = recent-only: new-64 MSE 0.005208, held-out 0.006250, exact in **23/24** lives. In life 22 two candidates tie: `x₂⊕x₄⊕1` (id 35) and `x₀⊕x₂⊕x₃` (id 52, the true one); each fits the 24 observations with exactly four "errors", at positions [0,6,7,23] and [11,13,18,20]; they differ on 32 of 64 inputs and on 24 of the 40 held-out ones (`results/exact_witnesses.json`; added *after* the results and labelled so).
- **Noisy cache overrides a correct rule.** [analyst, from `consolidation.json`] in `noisy_day`, at the "consolidated with episodes" snapshot, the core's seen-24 MSE is 0.003472 but core+cache is 0.166667 (= 4/24: the cache returns the four flipped labels with certainty); new-64: 0.005208 vs 0.066406. The report states the mechanism (`REPORT_TR.md:106`) and that no reliability gate is learned.

Decision labels (the study's own table, `REPORT_TR.md:116-128`):

| Hypothesis | Study's label | Evidence |
|---|---|---|
| H18 episodic recall ≠ durable generalisation | **Supported** ("Desteklendi") | cache 68.16 % / 49.06 % → 50 % after purge; rich core 24/24 exact on never-seen inputs |
| H19 adequacy of the supplied representation language | Supported, with open confound | 14-language 0/24 vs 84-language 24/24 — but 6× more candidate checks and the language is given |
| H20 same summary ⇒ timing equivalence | 72/72 identical + additivity proof; **general necessity of a separate phase not supported** | above |
| H21 limit of the model's own labels | New rule not acquired (0/24); general limit by the two-world proof | `sleep_self_replay` |
| H22 missing context and contradiction | Supported: exact lower bound + visible/hidden contrast | 0.218750 ≥ 0.125; visible 0 |
| H23 limit of a perfect supervisor | One failing noisy life + two-compatible-worlds witness; **no learned supervisor was tested** | life 22 |
| H31 effect without the raw record | Supported in the narrow sense: rule and update statistic persist in a fixed language | 24 fresh-process continuations |
| H33 importance of the input–label binding | Mispairing ⇒ exact new rule 0/24; not generalised to all tasks | 0.439062 |
| H24–H30, H32, H34–H41 | Recorded; **not tested** (beyond the partial measures named in the protocol) | — |

Negative results kept by the study: (a) replaying the *old* episodes was unnecessary here — `sleep_rich_recent` reaches the same result with half the checks (2,016), because old and new skills live in separate context compartments (`REPORT_TR.md:67`); (b) the mispaired control has lower mean error than the frozen model on some measures (0.439 < 0.5) without acquiring the rule (`REPORT_TR.md:65`); (c) hidden context: joint learning (0.21875) does not reach the 0.125 floor — the method is not shown optimal (`REPORT_TR.md:94`).

### (5) Laws / invariants (`MATHEMATICAL_LIMITS_TR.md`, written before the results; all conditional, none claimed as new)
1. **Timing equivalence (Önerme 1, `:32-56`). PROVED.** If the summary is additive over disjoint parts with an associative, commutative `⊕` (`T(D₁⊎D₂)=T(D₁)⊕T(D₂)`), the fixed objective depends on data only through `T(D)`, the final step returns the exact solution `F(T(D))` with a timing-independent tie-break, all schedules start from the same prior, process the same `D` and lose no summary, then one-by-one, block-wise and end-of-period integration give the **same final solution** (intermediate answers need not agree). Stated failure modes: bounded-step updates (`θ←(θ+y)/2`, `θ₀=0`: order (0,1) → 1/2, order (1,0) → 1/4); time-varying objective, weighting, forgetting, non-linear local optimisation; closed-loop data (behaviour changes what is observed, so `D` is no longer the same); lossy compression in one schedule; and reaching the right answer late does not cancel the cost of the wrong answers used on the way (end point and life-long performance are to be measured separately). Keeping only `θ` and deleting `T(D)` preserves the last answer but changes later learning. Empirical counterpart: 72/72.
2. **Internal processing adds no information about the world (Önerme 2, `:60-81`). PROVED.** Two worlds that produced the same accessible state `z` and require different answers cannot both be answered correctly by any consolidation `C` that receives no new external observation (deterministic or randomised with world-independent randomness); with equal weights the mean squared loss is ≥ 1/4. Complement stated in the same section: `C` *can* make consequences that are already determined by `z` available within a short answer budget (find a proof, compile a program part, build an index) — "İlki bilgi sınırıdır, ikincisi hesaplama kapasitesidir"; the cost of `C` must be booked.
3. **Contradiction without context (Önerme 3, `:85-99`). PROVED.** One answer `a` for two valid evaluators demanding 0 and 1: `min_a max{a²,(a−1)²} = 1/4`. In this world half the inputs conflict, so the balanced MSE is ≥ 1/8, and the bound is tight (`verify_consolidation.py:117-120`). With an observable context `a(x,c)=c` both are met with zero loss. "“Üst görevci” adı verilen bir birim bu bilgiyi yoktan üretemez; ek duyum veya geçerli önbilgi edinmesi gerekir."
4. **A new experience does not need a new dimension (Önerme 4, `:103-148`). PROVED.** For fixed features `v=φ(x)∈ℝ^d` and ridge objective, `G=Σvvᵀ`, `b=Σvy` are sufficient; block and recursive solutions coincide (`k=Pv/(1+vᵀPv)`, `θ⁺=θ+k(y−vᵀθ)`, `P⁺=P−Pvvᵀ P/(1+vᵀPv)`); storage `O(d²)` reals independent of the number of experiences. Exact-rational check: 4 steps, λ=2, final θ = (10/11, 3/11) (`exact_witnesses.py:23-42`). Caveats in the text: adding stored `A_i=λI+G_i` double-counts the prior; a fixed number of reals is not a fixed number of bits; `θ` alone is not enough for the next exact update (`P` or `(G,b)` is); nothing here learns `φ`.
5. **A re-organisation must commute with the update (Önerme 5, `:152-169`). PROVED.** If `f′(Tz,x)=f(z,x)` and `T(U_e(z))=U′_e(Tz)` for all reachable `z` and admissible `e`, every later answer after the transformation `T` equals the old system's. Preserving only the first equation is insufficient: `z=(a,b)`, `f=a`, `U_e(a,b)=(a+b,b)`, `T(a,b)=a` keeps today's answer and loses `b`, which the next update needs.
6. **Approximate commutation drifts (`:171-190`). PROVED (under the stated Lipschitz assumptions).** If `d(U′_e(Tz),T(U_e z)) ≤ ε` and `U′_e` is `L`-Lipschitz: `Δ_{t+1} ≤ LΔ_t+ε`; `L<1` ⇒ bounded by `ε/(1−L)`; `L=1` ⇒ grows as `tε`; `L>1` ⇒ geometric growth.
7. **Stability ≠ correctness ≠ adaptability (§7, `:192-206`).** A contraction `q<1` gives a unique fixed point; a two-way (top-down/bottom-up) loop or a supervisor does not supply the contraction by itself: `F(h)=−h` oscillates, `F(h)=2h` diverges, `F(h)=1` is perfectly stable and wrong if the target is 0.

Also in this document: four distinct things that can change after an experience (`:21-28`) — (1) the answer to the same query, (2) the cost of producing the same answer, (3) a correct relation on an unseen query, (4) what the *next* experience can change — and a ten-row audit checklist (`:208-223`).

### (6) Limits
Admitted by the study: the candidate language and the context bit are supplied; no common neural representation forms (`REPORT_TR.md:9`); old-skill retention is a structural consequence of separate context compartments — "Paylaşılan sinir ağı parametrelerinde girişim giderildiği … gösterilmez" (`:27`); the new rule need not share an abstraction with the old one, so no acceleration by old knowledge is tested (`:27`); the rich language makes six times as many comparisons, so the language effect is not isolated under equal compute (`:65`); online is not better under equal CPU (49 vs 2 selection calls) and no response-time or energy constraint was imposed, so the value of a separate processing period under such constraints stays open (`:75`); day inputs never repeat, so the exact-key cache can never hit before the label (`:75`); the selection rule is not a Bayes posterior for the noise model (`PROTOCOL.md:51`); the continuation check is a shared diagnostic operator, not evidence that the frozen control learns (`REPORT_TR.md:132`); local audit, no external review (`REPORT_TR.md:136`).

**[analyst]** (i) H20 is true here by construction: with counts and exact arg-min the equality is an identity, so the experiment functions as a null model, not as a discovery. (ii) The learner has no way to tell *why* no hypothesis reaches zero error: hidden context leaves a minimal error of 8–14/48, label noise leaves 4/24 — the same symptom with opposite remedies. Worse, in life 22 the two tied rules err on disjoint observation sets, so a two-rule cover with a free per-item selector would explain the *noisy* data perfectly; a selector that is not priced makes noise look like a hidden context. (iii) The cache/rule arbitration is a hard override (`consolidation.py:97-99`); graded arbitration was not built. (iv) Scale: 64 inputs, 84 hypotheses, 24 observations.

### (7) Implications for a core in which learning and running are one process
- **Requirement (P1).** A separate train/consolidate phase is not logically required: with additive, commutative evidence statistics, event-by-event updating ends in exactly the same durable state as deferred integration (law 1; 72/72) and serves better meanwhile (0.080 vs 0.484). The price is paid in selection work at every event (49 vs 2 calls). So the core's durable trace should be an *additive sufficient statistic* wherever possible; any update that is not of this kind (bounded steps, forgetting, non-linear local moves) makes the result depend on the order of life and must be documented as such.
- **Requirement (P2).** The durable trace is `N_t` (evidence per hypothesis and context), not the selected rule: "kalıcı deneyim izi, yalnız bugünkü cevap değildir" (`REPORT_TR.md:44`). A weight that is a graded belief must carry what the next update needs (law 5; `θ` vs `(G,b)`).
- **Requirement (test).** Any memory component must pass the *purge test*: remove raw episodes, then query inputs never seen (H18). Seen-item accuracy is not learning.
- **Warning (P4).** Replaying self-generated labels cannot move the system toward the world (0/24; law 2) — it re-confirms the old model exactly (`verify_consolidation.py:95-99`). Imagination is legitimate as *computation* (making consequences available within budget), never as *evidence*. Evidence units need a provenance flag; the study already counts `synthetic_labels` separately (`consolidation.py:72,119`).
- **Warning (P3).** Contradictory demands on the same observable input have a hard floor (1/4 per conflicting input). A new distinction (context) is the only remedy, and it needs *information* — a supervisor cannot manufacture it. Context discovery itself was not attempted (H24 open).
- **Warning.** A raw episodic store that overrides the consolidated belief on exact hits re-injects noise (0.1667 vs 0.0035). Arbitration between a fast record and a consolidated belief must be graded by reliability — open (`REPORT_TR.md:106`).
- **Mechanism to reuse.** Per-context integer error-count bank + minimal-error-set averaging (`consolidation.py:79-100`): additive, order-free, mergeable, resumable from the serialised state without raw data; ambiguity is visible as the size of the minimal-error set (2 in life 22).
- **Mechanism to reuse (controls).** Equal-evidence arms that separate information from computation: self-replay, mispaired replay, recent-only, narrow language (`consolidation.py:110-126`).
- **Design rule from law 6 [analyst].** Any compaction/re-clustering of the belief store that only approximately commutes with the update is safe when the update forgets (`L<1`: error bounded by `ε/(1−L)`) and can accumulate linearly when the update is a pure counter (`L=1`: bound `tε`). Exact counters therefore need exact re-organisation; decaying beliefs tolerate approximate consolidation. (This reads the study's bound as a design rule; the study states only the bound.)
- **Open questions handed on:** H24 (context from experience), H25 (value of a separate phase under latency/energy limits), H29 (a supervisor that itself learns), H30 (value vs truth), H32 (provenance traces for selective correction), H34 (a summary sufficient for one language is not for another), H35 (selective replay under budget), H38 (reversibility after consolidation), H39 (service quality *during* a transformation).

### 1.A `HYPOTHESES_TR.md` — H18–H41 (no results in this file by design, `:9`)
Four axes (`:65-70`): (1) information source and durability — H18, H21, H23, H31–H35, H38; (2) context and representation formation — H19, H22, H24, H26, H30, H36, H37; (3) timing, resources, supervision — H20, H25, H27–H29, H39; (4) scope — H40, H41. Statements worth carrying into the new design because they define controls:
- H26 two-way correction: remove the bottom-up and the top-down path in turn, with correct *and* misleading priors; "Bağlantı çizmek veya iki yönde mesaj göndermek tek başına başarı sayılmaz".
- H30 value ≠ truth: a high learned importance may protect a rare fact or reinforce a striking error.
- H34 validity range of a summary: two histories that collapse to the same summary while a *new* query or a *changed* language needs them apart (Tur 13 uses its additive summary only for a fixed language and loss; the language-change experiment is marked open here, complementing the earlier exact information-loss counterexamples digested in `g3`).
- H36 formation of the representation itself: the produced feature must be tested on held-out future tasks against a wide fixed language *with equal total search/memory budget*.
- H37 open-endedness under finite resources: lengthen streams that contain both reusable structure and endless independent distinctions; measure memory, work, old error and new acquisition as functions of life length.
- H39: a good end state does not certify the intermediate states of the transformation.
- H40 (logical counterexample, also `BIOLOGY_AND_LITERATURE_TR.md:85-91`): a frequency estimator `p̂_t = p̂_{t−1} + (x_t − p̂_{t−1})/t` learns while it lives (expected squared error `p(1−p)/t`) and has no general competence — "yaşarken öğrenme" and general intelligence are not the same concept.
- H41: a claimed "consciousness" advantage must be written as a separately measurable mechanism or prediction.

### 1.B `BIOLOGY_AND_LITERATURE_TR.md` — what the biology does and does not license
Five primary sources with the reading depth stated per source (de Vivo 2017 read at abstract level only): McClelland–McNaughton–O'Reilly 1995 (fast episodic vs slow integrative learning as a *computational* rationale); de Vivo 2017 and Diering 2017 (sleep-related synaptic down-scaling of *existing* synapses, selective); Girardeau 2009 (suppressing ripples during post-learning rest impairs a spatial memory task); Foster & Wilson 2006 (reverse replay in the *awake* state). Conclusions of the note: a new representation is not a new neuron; replay is not exclusive to sleep; none of the sources shows a perfect supervising unit or perfect transfer; "Bir evreye “uyku” adı verilmesi açıklama değildir". One currency flag: Jadhav 2012 was screened but not used because an erratum dated 9 July 2026 could not be read (`:67`). Design relevance: the fast-record / slow-integration split is a hypothesis about resources and timing (H25), not a law; Tur 13 shows it is unnecessary for the *end state* when the statistic is additive.

---

## 2. `living_learning_temporal_2026_09_22` — Tur 15: "Deneyimden bellek kullanımı ve gelecekte öğrenmeye yeterlilik"

Files: `docs/research/living_learning_temporal_2026_09_22/{REPORT_TR,PROTOCOL,MATHEMATICAL_AND_LITERATURE_TR,LEARNING_STATE_TR}.md`; `research/living_learning_temporal_2026_09_22/temporal_learning.py` (259 lines), `verify_temporal.py`, `verify_learning_state.py`, `results/*.json`.

### (1) Question
If the learner receives raw bits one by one and only a single terminal label per episode, can it acquire *which distinctions of the past to use* — and is the state that suffices for today's answer also sufficient for tomorrow's learning? (Protocol hypotheses H46–H49, `PROTOCOL.md:11-17`: variable-depth memory use from terminal outcomes; horizon vs number of states; what survives beyond the training lengths; how far the supplied memory form decides success.)

### (2) Mechanism / state representation — counting statistics and enumerated automata; persistent learned state vs per-episode working state
- **Working state** (reset by `begin()`, `temporal_learning.py:62-65`): `buffer` = last ≤ depth bits (`:75-76`); `running_states[64]` = current state of each candidate automaton (`:72`); `current_length`. Advanced on every bit by `feed` (`:67-76`).
- **Persistent learned state** — changed *only* in `observe_label` (`:133-151`), i.e. only when the terminal label arrives (`PROTOCOL.md:5`: "Öğrenilen parametreleri her bit gelişinde gizli bir bellek olarak değiştiren yöntem bu deneyde yoktur"):
  - *Suffix tables* (`kt_constant`, `suffix_1/5/8`): `counts[2^d, 2]` (`:45`); update `counts[key,label] += 1` (`:139`); readout = KT estimate `p = (n₁+½)/(n₀+n₁+1)` (`:116-117`).
  - *Context tree, depth 8* (`map_tree_8`, `mixture_tree_8`): 511 nodes with `counts`, `log_kt`, `log_mix`, `map_score`, `split` (`:45-52`). On a label, for each of the 9 nodes on the suffix path (`path()`, `:78-84`): `log_kt[node] += log((counts[label]+.5)/(counts.sum()+1)); counts[node,label] += 1` (`:143-146`); then `_refresh` bottom-up (`:148`).
  - `_refresh` (`:96-108`): `stop = log½ + log_kt[node]`; `log_mix[node] = logaddexp(stop, log½ + log_mix[L] + log_mix[R])`, i.e. the context-tree-weighting recursion `P_w = ½·P_KT + ½·P_w(L)·P_w(R)` (leaf: `P_w = P_KT`); `map_score[node] = max(stop, log½ + map_score[L] + map_score[R])`; `split[node] = (branch > stop)` — strict, ties stop.
  - MAP readout: KT estimate at the first non-split node on the path (`:119-122`). Mixture readout: from the deepest node upward `p ← β·p_KT(node) + (1−β)·p` with `β = exp(log½ + log_kt[node] − log_mix[node])` (`:123-131`) — the exact Bayes-mixture predictive over all prunings of the depth-8 tree.
  - *Two-state bank* (`two_state`): 16 transition tables × 4 readout tables = 64 candidate automata, start state 0 (`:21-23`); persistent `errors[64]`; on a label `errors += (READOUT[j, state_j] != label)` (`:136`); weights `w_j ∝ (0.05/0.95)^{errors_j}` (`:91-94`); readout `p = Σ_j w_j·READOUT[j, state_j]` (`:112-114`). The 5 % "noise" is the learner's assumption; the world is noise-free (`PROTOCOL.md:39`).
- No raw completed episode is kept by the learner (`raw_completed_episodes=0`, `:189`). Sizes: tree 20,951 bytes of numeric arrays; two-state bank 1,040 bytes; `suffix_8` 4,096 bytes.

### (3) Experimental design
- 24 lives × 3 tasks × 7 methods = **504 learning paths** ("504 bağımsız veri kümesi değildir"); each 1,024 distinct training episodes of 8–24 bits → 516,096 pre-label predictions; exactly one terminal label per episode, no intermediate labels (`run_one`, `:226-239`).
- Tasks (`truth`, `:31-35`): `last_bit`; `delay_5` (fifth bit from the end); `parity` of the whole episode. The task name is not given to the learner; the maximal tree depth, the two-state bound, the update rule and the candidate families are.
- Held-out sets per life: 128 unseen short words (8–24 bits) and 128 long words (32–64 bits); all 1,280 words of a life are distinct (`make_life`, `:198-208`); evaluated on frozen copies at 0/32/128/512/1,024 labels (`evaluate`, `:217-223`). No label from these sets is learned, no hyper-parameter is chosen on them.
- Metrics: scalar MSE (always answering 0.5 scores exactly 0.25) and accuracy (ties ½) on short/long tests; prequential MSE for the first 128, last 128 and all 1,024 episodes; selected leaves and depth; root stop mass; effective number of candidates `1/Σw²` (`:180`); work counters and bytes.
- Statistics: descriptive; per-life paired differences with better/tied/worse counts (`analyze_results.py:43-55`).
- Audits (`results/verification.json`, `results/final_audit.json`): exact re-run of 504 paths / 516,096 predictions; 2,520 evaluation snapshots; 21 fresh-process resumes; 168 checks of MAP and mixture against explicit enumeration of *all* trees of depth 1–3 (`verify_temporal.py:33-63`); 32,640 scalar-automaton outputs; 80 exact-fraction KT checks; evaluation does not mutate the learner; 961 earlier files + HEAD unchanged.

### (4) Results (means over 24 lives; `results/summary.json`; `REPORT_TR.md:33-41`)

Final **long-test** (32–64-bit words, never trained on) scalar MSE after 1,024 labels; accuracy in parentheses:

| Method | last_bit | delay_5 | parity |
|---|---:|---:|---:|
| kt_constant | 0.250189 (50.4 %) | 0.250134 (50.3 %) | 0.250231 (50.4 %) |
| suffix_1 | 0.000000952 (100 %) | 0.250008 (50.9 %) | 0.250606 (49.0 %) |
| suffix_5 | 0.000249592 (100 %) | 0.000249592 (100 %) | 0.257862 (49.8 %) |
| suffix_8 | 0.019317581 (99.2 %) | 0.019317581 (99.2 %) | 0.289163 (50.6 %) |
| map_tree_8 | 0.000000952 (100 %) | 0.000249592 (100 %) | 0.250231 (50.4 %) |
| mixture_tree_8 | 0.000000978 (100 %) | 0.000301700 (100 %) | 0.250234 (50.4 %) |
| two_state | numerical zero (100 %) | **0.470584 (51.95 %)** | numerical zero (100 %) |

- **Structure acquired.** The MAP tree selected depth 1 / 2 leaves for `last_bit`, depth 5 / 32 leaves for `delay_5` and the single root leaf for `parity`, each in 24/24 lives (`REPORT_TR.md:45-47`); it equals the `suffix_5` table (which was *given* the right length) on `delay_5`, and beats the fixed 8-bit table in 24/24 lives on both suffix tasks. [analyst, from `summary.json`] leaf count on `delay_5` over time: 1 leaf in 24/24 lives at 32 labels, 22–32 leaves at 128 labels, 32 leaves in 24/24 at 512; on `parity` a spurious 2-leaf split is selected in 3 lives at 32 labels, in 1 life at 128, in none at 512 — splits are retracted. Mixture root stop mass at the end: 1.7e−306 (`last_bit`), 1.5e−260 (`delay_5`), 0.9767 (`parity`).
- **Length generalisation.** The two-state bank answers `parity` on 32–64-bit words with 100 % accuracy in all 24 lives already after 32 labelled 8–24-bit episodes (long MSE 4.1284e−17); effective candidates 1.0 (`parity`) and 2.0 (`last_bit`, two tables compute the same function).
- **Cross failure.** The suffix family cannot represent `parity` (MAP stays at the root; `suffix_8` is *worse* than ignorance, 0.289163); the two-state family cannot represent `delay_5` and becomes confidently wrong: long MSE rises from 0.25 to 0.470584, effective candidates fall to 1.8867 (`REPORT_TR.md:49`).
- **Cost of learning over time** (`delay_5`, long-test MSE; `REPORT_TR.md:55-61`):

| labels | MAP tree | tree mixture | two-state bank |
|---:|---:|---:|---:|
| 0 | 0.250000 | 0.250000 | 0.250000 |
| 32 | 0.256645 | 0.247308 | 0.422312 |
| 128 | 0.046826 | 0.053830 | 0.476094 |
| 512 | 0.001017 | 0.001546 | 0.458616 |
| 1,024 | 0.000250 | 0.000302 | 0.470584 |

  Pre-label MSE of the first 128 training episodes: MAP 0.232339, mixture 0.177188; over all 1,024: MAP 0.031628, mixture 0.026671 [analyst, `summary.json`]; final long test: MAP lower in 24/24 lives. "Erken belirsizlik ve son uzmanlaşma farklı karşılaştırmalardır."
- **Resources** (`REPORT_TR.md:67-69`): tree methods 9,216 counter updates + 9,216 node refreshes per path and statistics for all 511 nodes throughout ("Seçilen iki yaprak, bütün öğrenicinin iki durum veya iki sayaç kadar yer kapladığı anlamına gelmez"); two-state bank 1,051,210.67 candidate-state transitions and 65,536 candidate–label comparisons.

Decision labels. The report gives no per-hypothesis verdict table; the mapping to H46–H49 is **[analyst]**:

| Hypothesis | Verdict | Basis |
|---|---|---|
| H46 variable-depth memory use can be acquired from terminal outcomes alone | SUPPORTED inside the supplied depth-8 suffix family | 2 leaves / 32 leaves in 24/24 lives; long test 100 % |
| H47 horizon and number of states are different things | PROVED (Önerme 3, 4) and SUPPORTED by the cross failure | below |
| H48 some acquisitions survive beyond the training lengths | SUPPORTED when the target is in the family (two-state parity; tree on suffix tasks); nothing transfers when it is not | 100 % vs ≈ 50 % |
| H49 the supplied memory form decides success | SUPPORTED: each family fails on the other's task | 0.4706; 0.2502 |

The report's own headline: success "önceden verilmiş bellek ailelerinin içinde gerçekleşti"; "Yeni bir temel algoritma keşfedildiği de ileri sürülmüyor" (`REPORT_TR.md:7-11`).

### (5) Laws / invariants
From `MATHEMATICAL_AND_LITERATURE_TR.md` (written before the results; mapped by the study to Nerode, Shalizi–Crutchfield causal states, Littman–Sutton–Singh predictive state representations, Willems–Shtarkov–Tjalkens CTW — "yeni kuram diye sunulmaz"):
1. **Continuation classes are the state (Önerme 1–2, `:31-43`). PROVED.** For a fixed known `F: Σ*→{0,1}` let `h ∼_F h′ ⟺ ∀z∈Σ*: F(hz)=F(h′z)`. Then `δ([h],b)=[hb]`, `o([h])=F(h)` are well defined, and every everywhere-correct deterministic predictor has at least as many reachable states as there are classes; the class machine attains the bound. Scope: one fixed, known, deterministic target; says nothing about how to *find* the classes from finite experience.
2. **Unbounded horizon ≠ unbounded state (§3, `:45-77`). PROVED.** Parity has exactly two classes (`q_{t+1}=q_t⊕b_{t+1}`), necessary and sufficient. **No fixed finite suffix suffices for parity at all lengths (Önerme 3):** `0^{D+1}` and `1·0^D` share length and last `D` bits and have different parity; any predictor depending only on suffix and length gives both the same `p`, so the mean scalar squared loss on the pair is `(p−½)²+¼ ≥ ¼` (vector Brier ≥ ½ — the note warns not to mix the two scales). Scope: the witness pair or a balanced distribution; not a claim about every test distribution, and not against a window as long as the whole word.
3. **Short horizon can need many states (Önerme 4, `:79-87`). PROVED.** Delay-k needs ≥ 2^k deterministic states (all 2^k words of length k are pairwise separated by padding with ≤ k−1 zeros); 32 for k=5. "“Bellek daha uzağa uzanıyor”, “daha çok durumu ayırt ediyor” ve “daha fazla öğrenilmiş bilgi taşıyor” ayrı kapasite özellikleridir."
4. **A mixture of inadequate candidates is inadequate (§5, `:95-103`). PROVED (conditional).** A convex mixture with weights fixed during the episode that outputs exact 0/1 on every word forces every positively weighted candidate to equal `F` everywhere; hence no fixed mixture of two-state candidates realises delay-5 exactly. Not extended to other combiners or to weights that change inside the episode.
5. **Finite experience is not a certificate for the unbounded future (Önerme 5, `:105-117`). PROVED.** `G(h)=P(h)⊕1[h=w*]` agrees with parity on every training word, differs on one longer word `w*`, and is finite-state (≤ 2(|w*|+2) states).
6. **Information access is part of the experiment (§7, `:119-127`).** The suffix counterexample is valid only if (1) no intermediate target labels are given, (2) both counterexample words start from the same trained state, (3) parameters are not updated inside the episode — "Model etiketsiz her bitten sonra parametrelerini değiştiriyorsa ağırlıkları bir parity biti saklayabilir. Bu durumda nominal sonek penceresi toplam belleğin sınırı değildir", (4) no unbounded "counter" smuggles content, (5) all methods get the same input/label history.
7. **Changing the memory rule creates a state-transport problem (§8, `:129-143`). PROVED.** A deterministic `T` with `ψ=T∘φ` exists iff `φ(h)=φ(h′) ⇒ ψ(h)=ψ(h′)`. Witness: parity kept, last bit wanted — `00` and `11`.

From `LEARNING_STATE_TR.md` (theoretical appendix; exact-arithmetic witnesses in `results/learning_state_verification.json`):
8. **Answer-equivalence vs learning-equivalence (§1, `:9-31`). PROVED.** For a given deterministic learning procedure `A`: `h ∼_cevap h′ ⟺ ∀q A(h,q)=A(h′,q)`; `h ∼_öğrenme h′ ⟺ ∀z,q A(hz,q)=A(h′z,q)`. The second implies the first, not conversely. Any implementation that compresses history to `S(h)`, updates only from `S`, and reproduces all future answers of `A` must satisfy `S(h)=S(h′) ⟹ h ∼_öğrenme h′`; conversely the classes of `∼_öğrenme` are a valid state (`U([h],e)=[he]`). "Bu bir varlık/koruma koşuludur, uygulanabilir keşif algoritması değildir" — the classes may be infinite, undecidable or unaffordable.
9. **Exact witness (§2, `:33-51`). PROVED (exact fractions; 100 states checked).** KT counters `(0,0)` and `(99,99)` both answer 1/2 today; after one new `1` they answer 3/4 and 201/400. All `(k,k)` answer 1/2 today and `½ + 1/(4k+4)` after one `1`: infinitely many classes, so *this exact counting procedure* cannot be continued forever with finitely many states. Explicitly not claimed: that approximate, bounded-precision or forgetting learners fail.
10. **Bits needed to move to a new representation (§3, `:53-68`). PROVED.** With old summary `φ` and wanted summary `ψ`, a lossless fixed-length auxiliary code needs `b ≥ ⌈log₂ M⌉` bits, `M = max_s |{ψ(h): φ(h)=s}|`, and that many suffice (given the tables). Witness: `φ` = parity, `ψ` = (parity, last five bits): 32 suffixes under each parity value ⇒ 5 extra bits must have been kept *beforehand*; later inputs rebuild them over time but never retroactively. Serving parity and delay-5 together along all continuations needs 64 working states (2,016 pairs of the 64 six-bit histories separated by at most four zeros).
11. **Three preservation questions and one impossibility (§4, `:70-82`).** (i) working memory for the current task; (ii) evidence and uncertainty needed to keep learning; (iii) distinctions not used now but needed for a competence not yet built. Compressing (i) well preserves neither (ii) nor (iii). And if future queries may ask for any single old bit, `n` bits must be kept: "Bu şart altında sabit boyutlu evrensel kayıpsız bir özet yoktur."

### (6) Limits
Admitted by the study: memory families, depth bound, state bound and update rule are supplied (`REPORT_TR.md:27`); "Bu deney, ailenin yanlışlığını fark edip genişletme mekanizması içermiyor" (`:49`); the parity success "doğru iki-durumlu algoritmanın küçük ve hazır aday ailesinde bulunmasına bağlıdır" (`:65`); the tree keeps statistics for all 511 nodes, so no minimal-memory claim (`:67`, `PROTOCOL.md:41`); 64 tables are not 64 functions, so the uniform prior is not uniform over functions (`MATHEMATICAL_AND_LITERATURE_TR.md:91`); resources are not an equal-compute comparison and exclude code and shared tables (`REPORT_TR.md:69`); episodes are reset, so a change of representation in the middle of an uninterrupted episode is untested (`MATHEMATICAL_AND_LITERATURE_TR.md:143`); accepted words are not exactly i.i.d. (duplicates rejected); the 24-life sample "bütün olası çevreler için bir sertifika değildir"; finite audits do not replace the proofs.

**[analyst]** (i) The two out-of-family failures differ in kind, and the difference is whether the family contains a *null member*. The tree family contains the context-free predictor (the root): on `parity` it ends at MSE 0.2502 with root stop mass 0.977 — an honest "nothing here". The automaton bank has only deterministic 0/1 experts scored with a sharp likelihood: on `delay_5` the posterior collapses on the luckiest expert and the loss is 0.4706, almost twice the loss of ignorance. An expert answering ½ would have likelihood `0.5^n` against roughly `(0.95·0.05)^{n/2} ≈ 0.218^n` for a chance-level deterministic expert and would take the posterior. (ii) My enumeration: the 64 tables realise only 22 distinct functions on non-empty words up to length 8; parity is table 26 alone, last-bit tables 21 and 42; no table agrees with delay-5 on more than 134 of the 256 eight-bit words (52.3 %), which matches the measured 51.95 %. (iii) No learner here changes durable parameters on unlabeled symbols — the regime a one-process core lives in by default. (iv) Deterministic noise-free targets, uniform random bits, no drift. (v) No single supplied family covers both tasks; the 64-state joint machine is proved necessary and never learned.

### (7) Implications for a core in which learning and running are one process
- **Requirement (P2).** A belief must store its evidence mass, not only its current estimate: `(n₀,n₁)`, not `p` (law 9). Because the exact procedure has infinitely many learning classes, the core must *choose and declare* an approximate learning-equivalence (bounded precision, decay) — the study leaves that choice open.
- **Requirement (P1).** Budget three stores separately — working distinctions, learning evidence, reserve for representations not yet built (law 11) — and say for each memory cell which of the three it serves.
- **Warning (P1).** Law 6(3) removes a comfort the one-process core cannot have: when parameters move on every event, fast weight change *is* working memory, and neither "window length" nor "number of state units" bounds what the system remembers. Capacity must be accounted over the whole mutable store, against the continuation classes of the operations promised (laws 1, 8).
- **Warning.** Horizon, number of distinguishable states and amount of learned information are three capacities. Lengthening a suffix/context window buys horizon by storage; it does not give the two-state computation that parity needs (law 2), and a 5-bit horizon can still need 32 states (law 3).
- **Warning.** Confidence inside a hypothesis family is not evidence that the family is adequate (0.4706, effective candidates 1.89; `REPORT_TR.md:100`). **[analyst] requirement:** every hypothesis bank carries a null member and is monitored against the context-free predictor's prequential loss (here 0.4795 vs 0.2515 over the life; already 0.4223 vs 0.25 at 32 labels); mass on the null member is the trigger for forming a new kind of concept.
- **Warning.** More addressable memory is not better learning at equal data: `suffix_8` 0.0193 vs MAP tree 0.00025 on `delay_5`, and 0.289 > 0.25 on `parity`.
- **Mechanism to reuse (P3).** The context tree: every candidate distinction keeps its own evidence; a split is *taken* when the evidence of the two children beats the stop option (`temporal_learning.py:103-108`) and is *retracted* when it no longer does. Distinctions are formed when warranted and are reversible because the statistics of unselected alternatives are never discarded. Cost: all 511 nodes are maintained whether selected or not.
- **Mechanism to reuse.** A bank of small recurrent candidates advanced in parallel on the raw stream and re-weighted by terminal outcomes (`temporal_learning.py:67-73, 91-94, 133-137`): exact generalisation from 8–24-bit episodes to 32–64-bit words (up to 2.7 times the longest episode ever seen), after 32 labels, when the target is in the bank.
- **Mechanism to reuse.** Hedge early, commit late: the mixture has the lower cumulative loss, the MAP structure the lower final loss.
- **Open questions handed on** (`REPORT_TR.md:104`, `MATHEMATICAL_AND_LITERATURE_TR.md:159`): the origin of the memory language itself; choosing between languages with a resource account; changing representation inside a running episode; environments where the task change is not announced; several competences kept together; "Sistem, gelecekteki öğrenmesini destekleyecek ayrımları hangi deneyimlere dayanarak, hangi bütçeyle ve hangi kayıp güvencesiyle koruyabilir?" (`LEARNING_STATE_TR.md:82`).

---

## 3. `living_learning_migration_2026_09_22` — Tur 16: "Doğru öğrenilmiş kuralı yaşayan akışta çalıştırmak"

Files: `docs/research/living_learning_migration_2026_09_22/{REPORT_TR,PROTOCOL,MATHEMATICAL_AND_LITERATURE_TR,CORRECTIONS,WHOLE_PROBLEM_AND_NEXT_TEST_TR}.md`; `research/living_learning_migration_2026_09_22/migration_learning.py` (180 lines), `verify_migration.py`, `results/*`.

### (1) Question
Does having learned a new rule *correctly* guarantee that it can be started in the right internal state at the present moment of an uninterrupted stream — and what must have been kept so that the old competence survives the installation? (`PROTOCOL.md:5`.)

### (2) Mechanism / state representation — enumerated programs, version-space acquisition, set-of-possible-states execution
- **Program families** (`migration_learning.py:18-21`, `:26-39`): new = 8 delay rules (k = 1..8; state = last k bits, `step: (2s+b) mod 2^k`, output = oldest bit `s >> (k−1)`) + 6 modulo-3 counters (state `q∈{0,1,2}`, `step: (q+b) mod 3`, readout = one of the six permutations of {0,1,2}) = 14 candidates; old = 64 two-state automata (as in Tur 15).
- **Acquisition = version-space elimination** (`calibrate`, `:55-65`): for each of 128 labelled words from a *separate, reset* calibration channel, predict with the uniform vote of the surviving candidates, then keep `candidates = [idx … if values[idx]==label]` (`:62`); compile the lowest-indexed survivor (`:64`); stop with an error if none survives (the hidden teacher is never used).
- **Running controller** (`Controller`, `:68-120`): `old_state` (one automaton state); `new_states` = sorted list = **the set of states the new program may currently be in**; reserves kept *before* installation only if the policy says so — `recent` (last 8 raw bits), `mod6` (number of ones mod 6), `archive` (all bits); counters `new_transition_checks`, `install_replay_bits`, `repairs`, `labels`.
- `feed(bit)` (`:75-84`): always advance `old_state` (`:77`); before installation update the policy's reserve (`:79-81`); after it advance every possible state and merge duplicates: `new_states = sorted({step(new_program, s, bit) for s in new_states})` (`:84`).
- `install(program)` (`:86-101`): initial set by policy — `[0]` (`guess_zero`, `reset_all`); all states (`possible_states`, and a reserve that does not match the task); `[run(program, recent)]` (`reserve_8` with a delay rule); `[mod6 % 3]` (`reserve_mod6` with a mod-3 rule); `[run(program, archive)]` (`archive`); `reset_all` additionally zeroes `old_state` (`:99`); reserves are then dropped (`:101`).
- `predict()` (`:103-107`): uniform probability over the **distinct outputs of the possible states** (set-valued belief, not a posterior). `observe_new_label(label)` (`:109-115`): keep the states whose output equals the label; if none remains, count a `repair` and expand to all states with that output.

### (3) Experimental design
- 24 lives × 2 prefix lengths (256, 4,096 bits before installation) × 2 new tasks (`delay_5`, `mod3`) × 6 policies = **576 paths**; 65 evaluation instants each (t = 0..64 new bits) = 37,440 instants and 74,880 competence outputs (old + new). The main stream is never reset.
- Calibration: 3 per life (old parity among 64; delay-5 and a mod-3 readout among 14) = 72 calibrations, 9,216 pre-label predictions, on reset words of 8–24 bits that are *not* part of the main stream — an explicit assumption: "bütün veri erişimi sıfırlamasız tek bir dünyadan gelmiş değildir" (`REPORT_TR.md:19`).
- Feedback on the main stream: none, except **one** label for the new skill at t = 16, delivered after that instant's prediction (`run_one`, `:147`); never a label for the old skill.
- Policies (`PROTOCOL.md:17-22`): `guess_zero`, `possible_states`, `reserve_8`, `reserve_mod6`, `archive`, `reset_all` (negative control; the real environment is not reset).
- Metrics: three-class Brier sum for the new skill (0 = certain and right, 2 = certain and wrong, 0.5 = even split over two outputs, 2/3 over three); binary squared error for old parity (a different scale — not comparable with Tur 15's scalar MSE); state-set size; `certain_wrong`; `true_state_in_set` (`record`, `:123-135`); stored raw bits, auxiliary counter bits, replayed bits, candidate transitions.
- Audits: exact re-run of 576 paths; 9,216 calibration and 37,440 stream records recomputed with independent scalar semantics; 24 fresh-process resumes (after t = 3 and after t = 16); exhaustive scan of all 8,192 thirteen-bit histories for the transport fibres (`verify_migration.py:40-47`); 977 earlier files + HEAD unchanged.

### (4) Results (means over 24 lives, all instants of the interval included; `results/summary.json`; `REPORT_TR.md:46-53`)

New-skill three-class Brier, prefix 256 / 4,096:

| Policy | delay-5, t = 0–4 | delay-5, t = 5–15 | mod-3, t = 0–15 | new skill, t = 17–64 | old parity error, whole continuation |
|---|---:|---:|---:|---:|---:|
| guess_zero | 1.016667 / 0.833333 | 0 / 0 | 1.000000 / 0.916667 | 0 / 0 | 0 / 0 |
| possible_states | 0.5 / 0.5 | 0 / 0 | 0.666667 / 0.666667 | 0 / 0 | 0 / 0 |
| reserve_8 | 0 / 0 | 0 / 0 | 0.666667 / 0.666667 | 0 / 0 | 0 / 0 |
| reserve_mod6 | 0.5 / 0.5 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| archive | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| reset_all | 1.016667 / 0.833333 | 0 / 0 | 1.000000 / 0.916667 | 0 / 0 | **0.458333 / 0.375** |

- **Rule acquisition succeeded everywhere; installation still failed.** In all 72 calibrations exactly one consistent candidate remained and it was the correct one (24 × parity, 24 × delay-5, 24 × the right mod-3 readout); the task identity was never given to the learner (`REPORT_TR.md:21`). Yet starting the correct rule from the zero state gave history-dependent wrong answers.
- **Inputs synchronise the delay rule, not the counter.** Possible-state set of the delay rule: 32 → 16 → 8 → 4 → 2 → 1 in every life; the influence of the five old bits ends after five new bits, and the zero-guess is right from t = 5 on ("erken yanlış cevaplar geçmiş kayıtlarından silinmedi"). For mod-3 the set stays at three states from t = 0 until the label; the zero-guess starts wrong in 12/24 (256) and 11/24 (4,096) lives and stays wrong by the same offset until the external label at t = 16 repairs it (`REPORT_TR.md:55-57`). After t = 17 all policies are exact on the new skill.
- **Honest uncertainty vs a guess** [analyst, `summary.json`]: `possible_states` is never certain-and-wrong and always contains the true state; `guess_zero` on the delay rule is certain-and-wrong in 37.5–58.3 % (256) and 29.2–54.2 % (4,096) of lives at t = 0..4, and its singleton contains the true state in only 1/24 (256) and 2/24 (4,096) lives at t = 0; on mod-3 it is certain-and-wrong in 50 % / 45.8 % of lives at every instant until the label.
- **Old competence.** All five policies that keep the old coordinate answer old parity correctly at every instant; `reset_all` flips it in 11/24 (256) and 9/24 (4,096) lives for the whole continuation, and the label for the *new* skill cannot repair it (`REPORT_TR.md:59`).
- **Each reserve serves only the task it anticipated.** `reserve_8`: delay 0, mod-3 0.6667; `reserve_mod6`: mod-3 0, delay 0.5; only `archive` serves both.
- **Costs** (`REPORT_TR.md:65-67`): archive 256 / 4,096 raw bits kept and replayed at installation; `reserve_8` 8 raw bits kept, 8 replayed; `reserve_mod6` a six-valued counter (3 bits in a plain fixed code); carrying 32 delay states through 64 new bits costs 121 candidate transitions instead of 64, three mod-3 states until the label 96 instead of 64. Only the new coordinate is counted.

Decision labels. The protocol's section "Önceden beklenen ayırıcı sonuç" (`PROTOCOL.md:38-40`) made four predictions; all four are **SUPPORTED/PROVED**: delay uncertainty ends within five new bits; the mod-3 remainder is never recovered from inputs; one external label fixes it (injective readout); a wrong reset does not bring old parity back. The naive expectation "a correctly learned rule can simply be switched on" is **REFUTED**. Headline of the report (`REPORT_TR.md:11`): "Böylece kural edinimi, mevcut duruma yerleştirme ve eski beceriyi sürdürme ayrı gereklilikler olarak sınandı."

### (5) Laws / invariants (`MATHEMATICAL_AND_LITERATURE_TR.md`; mapped by the study to Rabin–Scott subset construction, Eppstein reset sequences, Travers–Crutchfield exact synchronisation — "yeni ad vererek temel buluş iddia edilmedi")
1. **Knowing the rule is not knowing its state (§1, `:5-26`). PROVED.** `∃T: ψ=T∘φ ⟺ ∀h,h′: φ(h)=φ(h′) ⇒ ψ(h)=ψ(h′)`. Otherwise carry the set compatible with all kept information `I`: `C₀(I)={ψ(h): h compatible with I}`, `C_t=δ*(C₀,w_{1:t})`. Under a fixed deterministic transition `|C_{t+1}| ≤ |C_t|`, and the true state stays in the set if it started there. The set is not a probability distribution; if the *program* is uncertain, states must stay attached to their program.
2. **Future inputs can make the past irrelevant without recovering it (§2, `:28-46`). PROVED.** The current state is exactly known after continuation `w` iff `|δ*(C₀(I),w)|=1`. Delay-k: `|C_t| = 2^{max(k−t,0)}`. "Bu işlemde eski beş bit öğrenilmiş olmaz; onların yerini artık gözlenmiş yeni bitler alır."
3. **Some initial ignorance is never removed by inputs (§3, `:48-66`). PROVED.** For mod-3 every input acts as a permutation, `δ*(q,w)=q+Σw_i (mod 3)`, so distinct start states stay distinct for every continuation. Mixing is not knowing: averaged over random inputs the chain converges to uniform, but conditioned on the *observed* inputs the belief is only rotated; if inputs are independent of the hidden start state, `H(Q_t | I, W=w) = H(Q_0 | I)`.
4. **Uniform over the possible outputs is minimax, not posterior (§4, `:68-89`). PROVED.** With `r` distinct possible outputs and the truth among them, `min_{p∈Δ₃} max_{y∈O} Σ_j (p_j − 1[j=y])² = 1 − 1/r` (0, 1/2, 2/3; a wrong certain answer costs 2). The true posterior for `N` fair bits and known parity, `Pr(Q=r|p,N) = Σ_{j≡r (3), j≡p (2)} C(N,j) / 2^{N−1}`, is not uniform for finite `N`; the experiment does not use it.
5. **An external label is separately counted information (§5, `:91-101`).** `C⁺={q∈C: o(q)=y}`; with an injective readout one label determines the state. Re-expansion after an emptied set is an explicit repair (the set may grow at that step — no contradiction with law 1, which concerns input transitions only).
6. **When is the old competence preserved? (§6, `:103-116`). PROVED.** With a projection `π: S′→S` from the extended state to the old one, `π(δ′(s′,b)) = δ(π(s′),b)` and `π(s′₀)=s₀` imply by induction that the old state — hence every old answer read from it — is preserved along every continuation. `reset_all` violates the initial condition; because parity transitions are permutations the error persists without an old-skill label or a reconstruction of the past. "Bu negatif kontrol sinir ağlarına özgü bir unutma sonucu değildir; doğru devam durumunun silinmesinin sonucudur."

General constraint the report carries to the main problem (`REPORT_TR.md:83`): "**Yeni hesaplama kapasitesi edinildiğinde, o kapasitenin mevcut yaşam geçmişiyle tutarlı çalışmasını sağlayacak bilgi de gereklidir.**"

### (6) Limits
Admitted by the study: candidate programs, reserve forms and the calibration channel are supplied (`REPORT_TR.md:13`); the calibration data come from reset examples outside the stream (`:19`); after acquisition the reference examples and the eliminated candidates are not kept in the compiled controller, so the evidence needed to *re-learn* the rule later is not preserved — "tur 15'teki öğrenmeye devam etme ayrımı açık kalır" (`:21`); the reserves are comparisons chosen in advance, "Bunların ileride faydalı olacağını sistemin öğrendiği söylenmiyor" (`:38`); starting from "all states" is justified for these families and prefix lengths only, not a general transport algorithm (`PROTOCOL.md:24`); one label suffices only because the readout is one-to-one (`REPORT_TR.md:57`); old-skill preservation is "ayrı korunan koordinatın yapısal sonucu", with no claim about shared representations or forgetting in neural networks (`:61`); costs cover only the new coordinate (`:67`); "Her gelecekteki kavram dönüşümünün böyle olacağı söylenmiyor" (`:85`).

**[analyst]** (i) Apart from the base rates of a wrong guess, the outcomes are deterministic consequences of the dynamics (identical set sizes in every life); the 24 lives mainly confirm the implementation of the proofs. (ii) No noise in inputs or label; one installation event (repeated revisions are a later round, `living_learning_live_revision_2026_09_23`). (iii) The study measures the price of honest uncertainty only in transitions; with larger state spaces the possible-state set can be exponentially large (32 states for a 5-bit delay), so set transport needs a representation, not an enumeration.

### (7) Implications for a core in which learning and running are one process
- **Requirement (P3).** Forming a concept in the middle of a life is two acts: acquiring the rule and *placing it in the present*. The second needs information the rule does not contain. Installation protocol (all parts tested here): (a) derive the set of states compatible with what was kept; (b) keep running with the set and answer with the output-uniform hedge (minimax `1−1/r`); (c) let inputs synchronise what they can; (d) for what inputs cannot synchronise, use a reserve kept beforehand, a replay of an archive, or an external observation; (e) never start from a point guess; (f) count repairs.
- **Requirement (P1).** A new competence is added as an *extension* of the running state with a projection onto the old state that commutes with the dynamics (law 6). In a continuing life "reset" is an action with consequences, not a neutral default.
- **Requirement [analyst].** For every state variable of a formed concept the core should know whether its dynamics are *forgetting* (many-to-one: ignorance about the start decays as `2^{max(k−t,0)}`) or *conserving* (permutation: ignorance never decays). Only forgetting variables can be created late for free; conserving ones must be tracked from before or anchored by an observation.
- **Requirement [analyst].** Declare a *reserve contract*: which families of future concepts can be installed exactly and at what storage cost. Tur 16 shows each fixed reserve serving one family; Tur 15 law 11 shows no fixed-size reserve serves all. A bounded raw recent window covers every forgetting concept whose horizon fits in it; cheap conserved counters cover the anticipated group-like concepts; everything else waits, hedged, for an external observation.
- **Mechanism to reuse (P2, P4).** Set-valued belief over states with label filtering and repair-expansion (`migration_learning.py:75-115`), and the two coverage measures `true_state_in_set` and `certain_wrong` (`:134-135`) as direct measures of whether a belief contains reality.
- **Mechanism to reuse.** Version-space acquisition with a pre-label vote (`:55-65`).
- **Warning.** Honest uncertainty is cheap here (121 vs 64, 96 vs 64 transitions) but is not free and not small in general.
- **Open questions handed on** (`MATHEMATICAL_AND_LITERATURE_TR.md:130`; `REPORT_TR.md:87`): acquisition of the representation language; choosing from experience which auxiliary distinctions will be needed; safe expansion when the *program* is uncertain; many competences under limited resources; the system finding useful feedback by its own actions.

### 3.A `CORRECTIONS.md` — one audit correction, kept visible
After the first main run an independent code review found that `reserve_8` replays eight buffered bits when it installs a delay rule but did not add them to `install_replay_bits` (0 instead of 8). No prediction, state set, label order or acquisition changed. The first result and the source that produced it are preserved (`results/migration_learning_initial_cost_omission.json`, `results/initial_source_cost_omission.txt`); a re-run that was in progress while the source changed was *not* counted as a verification; the final audit checks that all calibration records and all 37,440 stream records of the two runs are identical and that only the counter and the hashes derived from it differ (`final_audit.json`: `all_predictions_unchanged_by_correction: true`). Reusable practice: corrections supersede, they do not overwrite.

### 3.B `WHOLE_PROBLEM_AND_NEXT_TEST_TR.md` — the owner's own status map after Tur 16 and the design of the next test
Not an experiment. It restates the main question — "Bir yapay sistem, çalıştığı süre boyunca yaşadığı deneyimler nedeniyle gelecekteki hesaplama ve davranış kapasitesini kalıcı, kontrollü ve genellenebilir biçimde nasıl değiştirebilir?" — and guards against sliding into a sub-problem ("genel yaşayan öğrenmenin mutlaka otomata öğrenimi veya senkronizasyon demek olduğu sonucunu vermiyor"). Its table (`:13-20`) is the owner's summary of what each region left and what it never showed:

| Region | Contribution that remains | Still not shown |
|---|---|---|
| acquisition of program pieces | earlier solutions can change later search; negative transfer is possible | acquisition of the piece-extraction and valuation procedure itself |
| cross-linking of relations | transfer to new compositions inside a given relation language | formation of the object/relation language and the kind of link |
| integration and durability | raw event storage, durable rule and replay are different things | a general memory budget and a long multi-skill life |
| context and access to experience | separating knowledge that explains the same outcomes needs different experience | acquisition of task, goal and feedback interface |
| memory use from a stream | time horizon, state capacity and learning statistics are distinct | construction of the usable memory language |
| adding a rule to a running state | correctness of the rule, correct start and preservation of old skill are distinct | representation transformation and resource sharing chosen by the system |

Proposed next discriminating question (`:26-38`): can experience build not only *which solution is stored* but *how the computation is changed when new evidence arrives* — a learned repair procedure `r: (p, E, τ) ↦ p′` (current candidate, accessible input–output evidence, own execution trace). It notes that `r` is itself an ordinary program runnable by a fixed interpreter, so "iddia, hesaplanabilir fonksiyon sınıfını genişletmek olamaz"; the testable claim is that lived experience changes what is *reachable within a resource budget*. The decisive causal comparison (`:38`): two copies with the same old solutions and the same initial answers; remove only the acquired repair procedures from one. Required strong baseline (`:42`): a synthesiser that receives the same old solutions and traces and learns a solution library and task-dependent edit priorities; plus a fixed interpreter running the learned procedures in the same order as an equivalence check. Pre-declared counterexamples (`:46`): programs agreeing on short evidence and diverging on long inputs; an old repair misleading in a new family; a right result through a wrong internal change; the same meaning in different syntax; a life too short to amortise the production cost. Novelty audit (`:52-58`): AutoML-Zero, Synthesize-Execute-Debug and SOAR already cover "learning the learning algorithm" and "repairing programs from one's own failures" — "geniş yenilik cümlelerini geçersiz kılar". Tur 17 implements this design (with arithmetic trees instead of lists).

---

## 4. `living_learning_repair_2026_09_22` — Tur 17: "Edinilmiş düzeltme yordamı yararlı; temel yenilik iddiası desteklenmedi"

Files: `docs/research/living_learning_repair_2026_09_22/{REPORT_TR,PROTOCOL,MATHEMATICS_AND_LITERATURE_TR}.md`; `research/living_learning_repair_2026_09_22/repair_learning.py` (330 lines), `verify_repair.py`, `results/*.json`.

### (1) Question
Can experience change not only which solution is stored but *how the next solution is constructed* — do variable-bound repair schemas, extracted from repairs the system found by its own search, make later, deeper computations acquirable under the same evidence and budget — and is that effect more than what experience-ranked generic bulk edits already give? (`PROTOCOL.md:9`: "Hipotez" and "Güçlü karşı hipotez"; "Her iki sonuç geçerli araştırma sonucudur".)

### (2) Mechanism / state representation — expression trees, a charged search, and three carriers of experience
- **Language (given).** Trees over `x`, `y`, constants, `add/sub/mul` in the field F17 (`repair_learning.py:15, 43-51`); meaning is checkable on all 289 inputs (`GRID`, `:20`).
- **Evidence and acceptance.** A task = a faulty candidate tree + 16 input–output examples (`examples`, `:236-238`). Evidence loss = Σ (numeric residual)² over the 16 examples (`evidence_loss`, `:54-57`); the first candidate with loss 0 is accepted (`solve`, `:221-224`). The teacher's tree, the regime and the full table are never given to the learner.
- **Generic local edit language (given).** At any node open a two-layer frontier of three sub-expressions (`boundaries`, `:66-71`) and rebuild it with one of 108 templates = 2 shapes × 3 outer ops × 3 inner ops × 6 permutations (`TEMPLATES`, `:18-19`; `rebuild`, `:74-76`).
- **Search (given).** `solve` (`:201-233`): evaluate the candidate; try the library of earlier accepted programs (`:208-212`); then at most eight greedy rounds — evaluate all proposals, accept on zero loss, otherwise move to the strictly best-improving candidate (`:213-230`); a budget exception ends the task (`:231`).
- **What experience writes** (`acquire`, `:274-292`), after solving four source tasks with the plain generic search (`:277`):
  1. `library` — accepted programs;
  2. `priorities` — the successful (context signature, template) pairs with counts (`:281-286`); a signature is (shape, outer op, inner op);
  3. `rules` — repair schemas from `derive_rules` (`:128-137`): for every pair of source edits, anti-unify the two *before* trees and the two *after* trees with **one shared memo** (`generalize`, `:109-120`; shared at `:131`), so the same sub-expression pair becomes the same hole on both sides; reject a result whose left side is a bare hole, whose two sides are equal, or whose right side has an unbound hole (`:133`); keep the provenance pairs and sort by support (`:137`);
  4. `source_edits`, `source_examples`.
- **How the carriers are used** (`proposals`, `:185-198`): `learned_rules` — for each schema one bulk bottom-up single pass over the whole tree (`bulk_rule`, `:163-171`; nodes created in the pass are not revisited, so it terminates) and every single-site application (`:189-195`), then the ranked local edits; `bulk_ranked` — the learned priority applied to every matching node in one pass (`bulk_template`, `:174-182`), then ranked local edits; `local_ranked` — learned priorities first, then all templates (`local_proposals`, `:85-106`); `local_plain` — fixed order; `rules_disabled_paid` — the same durable state and the same booked acquisition cost as `learned_rules`, schema execution switched off (`:186, :297-298`).
- **Cost meter** (`Work`, `:36-40`; `run_targets`, `:295-310`): 1,000,000 abstract units per life, at most +120,000 per target (`:301`); every evaluation (`nodes × examples`), traversal, virtual construction, match, equality check and generalisation step is charged; an operation that does not fit is not started.

### (3) Experimental design
- 24 lives × 3 regimes × 5 methods = **360 paths** × 8 targets = 2,880 target attempts; independent unit = 24 lives.
- Sources (4 per life; `make_life`, `:254-257`): the faulty `(u+c)*w` must become `u*w+c` (`u,w ∈ {x,y}` alternating, `c` a different non-zero constant each time). The repair is *found by the learner's own generic search*, not handed over.
- Targets (`:259-270`): 1, 2 or 4 planted fault sites with operand sub-expressions of depth 1, 2 or 3, in the fixed order (1 site, depth 1), (2, 2), (4, 3), (1, 1), (2, 2), (4, 3), (2, 2), (4, 3), joined by `add`; operands are random sub-expressions that do not occur in the sources.
- Regimes sharing the same faulty candidates (`:263-266`): **compatible** (every site needs the source repair), **changed** (sites need `(u+v)+w` instead), **partial** (the last planted site is already right and must be left alone; single-site partial targets are therefore correct from the start).
- Methods: `local_plain`, `local_ranked`, `bulk_ranked` (the strong control), `learned_rules`, `rules_disabled_paid` (the causal ablation).
- Measure: a target counts as correct only if the accepted program equals the hidden target on all 289 field inputs (`:303-307`); this check is not charged and never returned to the learner. Also: accepts, false accepts, work, candidates evaluated, schema proposals, rejected schema candidates, persistent JSON size.
- Statistics: descriptive counts and per-life paired differences (`analyze_results.py:33-44`).
- Audits (`results/verification.json`, `results/final_audit.json`): exact re-run of 360 paths; 13,872 scalar field outputs; each of the 1,095 accepted target programs re-checked on 289 inputs with an independent scalar evaluator; shared-hole, repeated-hole and free-right-variable rules tested; one-pass termination; budget accounting; 15 fresh-process resumes after the fourth target; 994 earlier files + HEAD unchanged. Two accounting defects (tree construction before the budget check; uncharged schema-validation scans) were found by independent review **before** the performance run and fixed; after the results nothing was changed (`REPORT_TR.md:108`).

### (4) Results (`results/summary.json`; `REPORT_TR.md:59-65`)

Correct acquisitions out of 192 (= 24 lives × 8 targets) per cell:

| Method | compatible | changed | partial |
|---|---:|---:|---:|
| local_plain | 48 | 48 | 51 |
| local_ranked | 54 | 48 | 120 |
| **bulk_ranked** (strong control) | **84** | 48 | 120 |
| **learned_rules** | **84** | 48 | 120 |
| rules_disabled_paid (ablation) | 54 | 48 | 120 |

- **Source phase.** 96/96 source repairs found by the generic search and exact on all 289 inputs. Every life yields exactly three schemas — two x/y-specific with one hole and the general `mul(add(A,B),C) ↦ add(mul(A,C),B)` — and exactly one priority entry. Source search and priority set-up cost 5,690 units, schema extraction a further 468 (6,158 booked to both schema arms).
- **Causal ablation (compatible).** Executing the schemas: 84 vs 54 with identical memory and booked cost; per-life difference positive in 19 lives, zero in 5, never negative (at most +3 per life).
- **Strong control.** `learned_rules` and `bulk_ranked` produce the *same accepted program sequences* in all 72 life × regime pairings (`final_audit.json`: `identical_learned_and_bulk_program_sequences: 72`), not merely the same counts; the schema arm costs on average 195.79 units more in the compatible regime (per life between −3,436 and +4,955).
- **Where the gain is** [analyst, per-target counts in `summary.json`]: single-site targets are solved by every method (24/24 each, 48 in total); two-site/depth-2 targets: schemas = bulk 10 + 12 + 12 = 34 of 72, ranked-local 1 + 1 + 4 = 6, plain 0; four-site/depth-3 targets: schemas = bulk **2 of 72**, all others 0 (`REPORT_TR.md:71`: "derin bileşim sorunu çözülmüş değil").
- **Changed family.** Every method stays at 48 (the single-site targets). The schema arm is more expensive than its paid-disabled twin in all 24 lives: +3,591.71 units on average (min +1,997, max +5,959), no extra correct target.
- **Partial family.** 120 for every method that uses experience — of these 48 are the already-correct single-site controls and 72 the two-site targets needing exactly one repair; plain search reaches 51; four-site partial targets fail in every method. The schema arm costs +2,638.13 units over the paid-disabled twin; the plain ranked-local method reaches the same 120 at the lowest mean cost.
- **False accepts.** 0 of 1,095 accepted target programs (and 0 of 96 sources).
- **Rejected schema proposals** (evaluated and failing the evidence): 776 / 934 / 684 in compatible / changed / partial (of 1,070 / 1,363 / 897 schema proposals issued).
- **Durable state size** (serialised JSON, compatible regime, mean): bulk control 2,078 bytes, schema arm 2,447 bytes (`REPORT_TR.md:94`) — the whole acquired "way of constructing" is a few kilobytes of inspectable structure.
- **Whole-life work** (mean abstract units; `REPORT_TR.md:79-84`):

| Method | compatible | changed | partial |
|---|---:|---:|---:|
| local_plain | 724,627.62 | 720,700.54 | 704,798.42 |
| local_ranked | 691,659.75 | 721,066.75 | 370,224.92 |
| bulk_ranked | 548,624.79 | 721,758.00 | 373,897.83 |
| learned_rules | 548,820.58 | 725,126.46 | 373,331.04 |
| rules_disabled_paid | 692,127.75 | 721,534.75 | 370,692.92 |

  No path reached the life cap (maximum 726,737). **[analyst]** In all 15 cells the number of attempts stopped by the budget equals 192 minus the number accepted (144 / 138 / 108 / 108 / 138; 144 ×5; 141 / 72 ×4): *every* failure is an exhausted per-target cap. The work table is therefore essentially "failures × 120,000 + small terms" (plain search: 6 failures per life × 120,000 = 720,000) and is not evidence independent of the success counts. The report says "ağırlıkla" (mostly) and states that the result holds for this budget split only (`REPORT_TR.md:92`).

Decision labels (the study's own):

| Claim | Study's verdict | Evidence |
|---|---|---|
| Schemas derived from one's own repairs change later acquisition under equal memory, evidence, budget and booked cost | **Supported** (narrow) | 84 vs 54; +19 / =5 / −0 lives |
| The effect is fully explained by experience-ranked generic bulk edits (strong counter-hypothesis) | **Supported** | identical accepted programs in 72/72 |
| Learned schema extraction is a necessary new fundamental principle / beats the strong known method | **Not supported** | same; +195.79 units |
| Schemas transfer to a changed task family | **Not supported**; negative transfer as pure cost | +0 correct; +3,591.71 units in 24/24 lives |
| Transfer to deeper compositions | Weak | 2 of 72 |
| Open-ended self-improvement | Not shown ("Genel yaşayan öğrenme problemi hâlâ açık") | — |

### (5) Laws / invariants (`MATHEMATICS_AND_LITERATURE_TR.md`, written before the results; `REPORT_TR.md:96-102`)
1. **Fitting the evidence is not being right (§2, `:17-46`). PROVED.** With 16 observed pairs the set `S_x` of observed first coordinates is a proper subset of F17, and `g(x,y) = f(x,y) + Π_{a∈S_x}(x−a)` reproduces all 16 examples while differing from `f` wherever `x ∉ S_x`; `g` is expressible in the given language. Conversely even full-field equality is neither program identity nor integer semantics (`x^17 = x` on F17). Consequence drawn by the study: the false-accept rate is a required measure; acceptance counts alone are not progress.
2. **Shared-variable generalisation is syntactic (§3, `:48-64`).** The same sub-expression pair must use the same pattern variable on the before and the after side; right-side variables ⊆ left-side variables; a repeated variable demands identical sub-trees. The resulting schema is **not** an algebraic identity (`(u+v)w ≠ uw+v` in general) — "bazı görevlerde istenen davranış değişikliğine adaydır"; if the true target is `(A+B)*C` the schema breaks a correct program. Termination of bulk application is a property of the supplied one-pass interpreter, not something learned.
3. **Conditions for a causal reading (§4, `:66-80`).** `Δ(L,B) = Y(s,on,L,B) − Y(s,off,L,B)` is the effect of executing schemas only if both arms start from the same state `s`, the same task evidence and the same budget meter. Over eight successive targets it is a **total** effect that includes the divergence of the two libraries after the first differing result; a direct per-target effect would need branching from a common state before every target (not done).
4. **Whole-life cost (§5, `:82-97`).** `C_life = C_source + C_setup + C_schema + Σ_t C_target,t`; schema production and failed matches are charged; an equal number of candidates is not an equal cost.
5. **The learned general schema equals a generic bulk edit (`REPORT_TR.md:96-102`). PROVED for this schema.** Its three left variables are independent, so matching is equivalent to "root `mul`, left child `add`"; the right side rebuilds `A*C+B`, which the generic record `L/add/mul/(0,2,1)` already expresses; equality on whole trees follows by structural induction under the same bottom-up single pass (40 independent tree checks). Explicitly not a theorem that every learned schema reduces to the generic edit language.
6. **What can grow (§1, `:15`).** There are finitely many (`17^289`) functions F17² → F17; "olası kapasite kazanımı, verilen bütçe içinde doğru fonksiyona ulaşma yeteneğidir. Hesaplanabilir fonksiyon sınıfının genişletilmesi … sonucu çıkarılamaz."

### (6) Limits
Admitted by the study: the tree language, the 108 local rebuilds, the generalisation algorithm, variable binding and the bulk interpreter are supplied — "Oluşan desen ağaçları deneyimden geldi; fakat bunları çıkaran öğrenme algoritması deneyimden icat edilmedi" (`REPORT_TR.md:37`); schema extraction is frozen after the sources and targets create no new schema (`:53`); the ablation measures a total effect (`:53`); the evidence loss is the ordinary square of residues 0…16, an arbitrary search geometry shared by all methods (`MATHEMATICS_AND_LITERATURE_TR.md:46`); fixed budget split, other budget managers untested (`REPORT_TR.md:92`); abstract work units, not CPU or RAM (`:94`); the 120 partial successes are not "120 new skills" (`:73`); F17 only; direct methodological predecessors Getafix and REFAZER, and SOAR for the idea of improving later repair from one's own search attempts (`:106`).

**[analyst]** (i) One source fault family produces exactly one general schema, and that schema lies inside the generic two-layer template language; the design could not have separated "learned schema" from "ranked bulk template". A discriminating version needs sources whose common generalisation is *not* one two-layer template (repeated variables, deeper context, side conditions). (ii) All failures are per-target cap stops, so the study measures reachability under one budget split. (iii) The constructor is improved once, from four sources, and then frozen: nothing during the eight targets lowers the priority of a schema that keeps being rejected (776–934 rejected evaluations are paid without consequence). (iv) In the partial regime a mere re-ordering of a fixed proposal set (`local_ranked`, 120 vs 51) carries the whole benefit at the lowest cost — the simplest "graded preference" already does the work there. (v) Greedy descent on squared residues of a finite field is a weak guide; part of the depth failure is a search failure.

### (7) Implications for a core in which learning and running are one process
- **Requirement (P5).** A claim that "the way of constructing improves" must pass two tests: (a) the paid, same-memory ablation (`rules_disabled_paid`) and (b) the strongest simple baseline that uses the same experience (`bulk_ranked`). Tur 17 passes (a) — the effect is real, +30 of 192 — and ties (b). The new core should therefore begin with the simplest carrier of constructive experience (context signature → edit, with a count, applied at all matching sites; `repair_learning.py:281-286, 174-182`) and demand that any richer mechanism beat it on a task family where the two can actually differ.
- **Requirement (P4).** Keep "zero loss on my evidence" and "exact on the domain" as two different quantities, report false accepts, and never let the hidden check flow back into construction (`:303-307`).
- **Requirement.** Whole-life accounting: the production of a construction aid and its failed proposals are charged; a life can be too short to repay them.
- **Warning (P5).** Transfer is signed by the regime: outside the source regime the acquired proposer only costs (+3,592 units per life, 24/24). An applicability estimate that outcomes can revise was not built.
- **Warning.** One schema applied in bulk does not give compositional depth (2/72).
- **Mechanism to reuse.** Shared-memo anti-unification of one's own before/after pairs with a validity filter and provenance (`:109-137`); one-pass terminating bulk application (`:163-182`); experience-ranked proposal order (`:85-99`); the charged `Work` meter with a per-task cap and atomic refusal (`:36-40, :301`); the paid-disabled arm (`:297-298`).
- **Open questions handed on** (`REPORT_TR.md:120`): the system must (1) separate by experience *which internal computation step* caused a failure, (2) construct the new *form* of change it needs, (3) test it on other tasks against total cost and preservation of old skills; plus (`MATHEMATICS_AND_LITERATURE_TR.md:113`) acquisition of the verification criterion, choice of goals and experiences, and change of the method that produces repair methods.

---

## 5. Cross-study synthesis for the new core

### 5.1 Laws of this group (numbered for citation)

| # | Law (precise form) | Scope / what it rests on | Status in the studies |
|---|---|---|---|
| G5-L1 | **Continuation classes are the state.** `h∼h′ ⟺ ∀z F(hz)=F(h′z)`; the classes carry a well-defined transition and output; an everywhere-correct deterministic predictor needs at least that many reachable states. | one fixed, known, deterministic target | PROVED (Tur 15, Önerme 1–2) |
| G5-L2 | **Three capacities.** Time horizon, number of distinguishable states and amount of learned information are independent: parity = unbounded horizon / 2 states; delay-k = horizon k / ≥ 2^k states; no fixed suffix serves parity at all lengths (pair loss ≥ 1/4). | witnesses; balanced pair | PROVED; cross failure in 24/24 lives |
| G5-L3 | **Learning-state law.** Answer-equivalence (`∀q A(h,q)=A(h′,q)`) is strictly coarser than learning-equivalence (`∀z,q A(hz,q)=A(h′z,q)`); a compressed state that continues `A` exactly must refine the latter. Exact KT counting has infinitely many learning classes. | a given deterministic learning procedure | PROVED; (0,0) vs (99,99): 3/4 vs 201/400 |
| G5-L4 | **Transport law.** `ψ=T∘φ` exists iff `φ(h)=φ(h′) ⇒ ψ(h)=ψ(h′)`; otherwise `⌈log₂ M⌉` bits, `M = max_s card{ψ(h): φ(h)=s}`, must have been kept *before* the change; if any old bit may be asked, `n` bits — no fixed-size universal lossless summary. | lossless, finite histories; tables assumed known | PROVED; 5 bits; 64 states; 2,016 pairs |
| G5-L5 | **Timing equivalence.** With an additive commutative summary, a fixed objective that depends on data only through it, and an exact final solve, every schedule (per event, per block, at the end) reaches the same end state — not the same life-long service; bounded steps, closed-loop data, lossy compression or a changing objective break it. | Tur 13, Önerme 1 | PROVED; 72/72; `θ←(θ+y)/2`: 1/2 vs 1/4 |
| G5-L6 | **Internal processing is computation, not evidence.** Worlds that left the same accessible state cannot be told apart by any processing without a new external observation (pair loss ≥ 1/4); processing can still make consequences already determined by the state available within the answer budget. | Tur 13, Önerme 2 | PROVED; self-replay 0/24; life-22 witness |
| G5-L7 | **Contradiction floor.** One answer for two valid opposite demands costs `min_a max{a²,(a−1)²}=1/4` per conflicting input; only a distinguishing observable removes it. | Tur 13, Önerme 3 | PROVED; 0.21875 ≥ 0.125; visible context 0 |
| G5-L8 | **Re-organisation must commute with updating.** `f′(Tz,x)=f(z,x)` and `T∘U_e=U′_e∘T` ⇒ identical future; preserving answers alone is not enough; with mismatch ε and Lipschitz constant L: `Δ_{t+1} ≤ LΔ_t+ε`. | Tur 13, Önerme 5 | PROVED; counterexample `(a,b) ↦ a` |
| G5-L9 | **Installation law.** A correctly learned rule's present state is known only as the set `C₀(I)` compatible with what was kept; it evolves as `C_t=δ*(C₀,w)`, never grows under input transitions, and is exact iff `card δ*(C₀,w) = 1`. | Tur 16, §1–2 | PROVED; 576 paths |
| G5-L10 | **Forgetting vs conserving dynamics.** Many-to-one transitions erase ignorance about the start (`2^{max(k−t,0)}`); permutation transitions conserve it for ever (`H(Q_t ∣ I,w) = H(Q_0 ∣ I)` for inputs independent of the state); then only an output observation helps. | Tur 16, §2–3, §5 | PROVED; 32→1 in five bits; 3 states until the label |
| G5-L11 | **Hedge law.** Uniform probability over the `r` possible outputs is minimax for the Brier sum (`1−1/r`); it is not the posterior. | truth inside the set | PROVED; 0.5 and 0.6667 measured |
| G5-L12 | **Preservation law.** An old competence survives an extension of the state iff the projection onto the old state commutes with the dynamics and starts right. | Tur 16, §6 | PROVED; `reset_all` wrong in 11/24 and 9/24 lives |
| G5-L13 | **Learning without growth.** With fixed features the exact recursive update equals the batch solution with `O(d²)` statistics; `θ` alone does not continue the learning. | Tur 13, Önerme 4 | PROVED; θ = (10/11, 3/11) |
| G5-L14 | **Budgeted reachability.** What experience adds to construction is reachability within a budget, not computable functions; the effect is real and signed by the regime. | Tur 17 | SUPPORTED: +30/192 (19/24 lives); +0 with +3,592 units in a changed family |
| G5-L15 | **Fit is not truth.** Finite evidence certifies neither full-domain correctness (`f+Π(x−a)`) nor the unbounded future (`P⊕1[h=w*]`); certainty inside a family does not certify the family. | Tur 17 §2; Tur 15 Önerme 5; Tur 15 data | PROVED / SUPPORTED (0.4706 > 0.25) |
| G5-L16 | **Stability ≠ correctness ≠ adaptability.** A contraction converges to one fixed point; a loop or a supervisor supplies neither the contraction nor the truth. | Tur 13, §7 | PROVED by counterexamples |

None of these is claimed by the owner as a new law; each report maps its statements to existing literature (Nerode/causal states/PSR/CTW; subset construction/synchronising words; complementary learning systems; anti-unification repair). Their value for the new core is as constraints to satisfy and tests to pass.

**[analyst] Common form.** L1, L3, L4, L9, L12 and L14 are instances of one statement: *fix the family of future operations the system promises to support — continuations and queries (L1), further learning (L3), a change of representation (L4), the installation of a later rule (L9), the continued use of an older competence (L12), constructions within a budget (L14). Two histories may share a state only if no operation of that family separates them; the number of separable classes is the state that must be held; whatever lies outside the family has to be bought later with new external information.* The unified wording is mine (it coincides with G3-L1 "operation-relative sufficiency" of the sibling digest); each instance is proved separately in the cited document.

### 5.2 One table: which future operation needs which state

| Future operation the core promises | State shown sufficient | Shown insufficient | Source |
|---|---|---|---|
| answer a terminal query on a stream for a known rule | the rule's continuation class (2 states for parity, 32 for delay-5) | any fixed suffix for parity; 2 states for delay-5 | Tur 15 |
| answer parity *and* delay-5 along all continuations | (parity, last five bits): 64 states | either alone | Tur 15 `LEARNING_STATE_TR.md` §3 |
| keep learning exactly with a KT counter | `(n₀,n₁)` (unbounded) | the current estimate `p` | Tur 15 `LEARNING_STATE_TR.md` §2 |
| keep selecting among a fixed hypothesis family after raw episodes are purged | per-context error counts `N(c,h)` | the selected rule alone; an exact-key cache | Tur 13 |
| keep an exact ridge solution | `(G,b)` or `(θ,P)` | `θ` | Tur 13, Önerme 4 |
| be indifferent to the schedule of learning | additive commutative statistic + exact solve | bounded-step iterates | Tur 13, Önerme 1 |
| serve two rules that conflict on the same input | both rules + an observable that tells them apart | any single-answer state without it | Tur 13, Önerme 3 |
| generalise a rule to unseen inputs after the memory is purged | selected member of a language that contains the rule | cache; a language without the rule (0/24) | Tur 13 |
| start a newly learned delay-k rule in a running stream | last k raw bits kept beforehand — or k further inputs with a possible-state set | a zero-state guess (wrong until k inputs have arrived) | Tur 16 |
| start a newly learned mod-3 counter in a running stream | a mod-6 (or mod-3) counter kept beforehand, or archive replay, or one output observation | a zero guess; the last-8-bit reserve; any amount of further input | Tur 16 |
| keep the old competence while adding a new one | old coordinate behind a commuting projection | resetting the running state | Tur 16 |
| change representation `φ → ψ` without raw history | `φ` whose fibres are constant on `ψ`, else `⌈log₂M⌉` auxiliary bits | parity alone for (parity, last five bits) | Tur 15 |
| answer any later question about any old bit | `n` bits | every fixed-size summary | Tur 15 `LEARNING_STATE_TR.md` §4 |
| reach related constructions later within the budget | library + experience-ranked generic edits applied in bulk (≡ the learned schema here) | library + ranked single-site edits (54 vs 84) | Tur 17 |

### 5.3 What the group says about each pillar

- **P1 learning = running.** Supported in the strongest available sense for additive evidence: the per-event learner ends in the identical durable state as the deferred one (72/72) and serves better on the way (0.080 vs 0.484), at the price of selection work at every event (L5). Three warnings: results become order-dependent as soon as updates are bounded steps or the loop is closed (L5); when everything may change on every event, the whole mutable store is "state" and must be accounted against the promised operations (Tur 15 §7); a rule acquired during life has to be *installed* into the running history (L9–L12) and "reset" is not available.
- **P2 weights = graded beliefs.** Four concrete belief forms appear: a count pair with its evidence mass (KT), an error-count vector per context with the minimal-error set as readout, a likelihood-weighted bank of candidate machines, and a non-probabilistic *set* of possible states with a minimax hedge. Conditions: a belief must contain what the next update needs (L3, L8, L13); a bank needs a null member [analyst, from Tur 15 data]; a set-valued belief is the honest form when no distribution is justified (L11).
- **P3 concepts formed when needed.** Tested pieces: priced, reversible splits in a context tree (1 → 22–32 → 32 leaves; spurious splits withdrawn); installation of a later rule with honest initial uncertainty. Tested obstacles: a later concept needs distinctions kept before it existed (L4); conserving variables cannot be created late (L10); a hidden context and label noise produce the same symptom [analyst]. Not tested: discovery of the context or of the hypothesis language.
- **P4 imagine and measure reality.** The only "imagination" in the group is replay of self-generated labels, and it adds nothing (0/24; L6). Processing without new evidence is legitimate as pre-computation. Reality measures that were actually implemented are listed in 5.6. The limit of any internal judge is the two-world witness (life 22).
- **P5 the way of constructing improves and transfers.** One positive causal result (84 vs 54) and its deflation by a strong baseline (identical programs 72/72); transfer limited to the source regime, costly outside it, weak in depth; the constructor was improved once and then frozen.

### 5.4 Requirements extracted (each traceable)

1. **Declare, per memory component, the future operations it must support, and derive the state from that** (L1, L3, L4; 5.2). Test: serialise, restore in a fresh process, feed the same future, demand identical predictions and updates (done 84 times across the four studies).
2. **The durable trace is the evidence, not the answer**: counts, error vectors, `(G,b)` — never only the selected rule or the current estimate (L3, L13; `REPORT_TR.md:44` of Tur 13).
3. **Prefer additive, commutative evidence statistics**; then learning at every event equals any deferred schedule and the state is mergeable and resumable (L5). Where an update is not of this kind, record that the outcome depends on the order of life.
4. **Purge test.** Every memory mechanism is evaluated after raw episodes are removed and on inputs never seen (Tur 13 H18).
5. **Provenance of evidence.** Self-generated labels never count as evidence (L6); keep them countable and separate (`synthetic_labels`); an external label that repairs an emptied belief is booked as a repair (`repairs`).
6. **Every hypothesis bank has a null member and is monitored against the context-free predictor** [analyst strengthening of Tur 15 `REPORT_TR.md:100`]; mass on the null member, not confidence inside the bank, is the signal that a new kind of concept is needed.
7. **Distinctions are priced and reversible**: a split is taken only when its evidence beats the stop option, and the evidence for unselected alternatives is kept (Tur 15 `_refresh`). A selector that is not priced makes noise indistinguishable from a hidden context [analyst, life 22 of Tur 13].
8. **Conflicting demands need an observable that separates them**; without it the system reports the floor instead of pretending (L7).
9. **Any re-organisation of beliefs (compaction, re-clustering, "sleep") must commute with updating, or carry an explicit error budget** (L8). [analyst] For exact counters (L = 1) the bound on accumulated re-organisation error grows linearly (`tε`); for decaying beliefs (L < 1) it stays below `ε/(1−L)`.
10. **Installation protocol for a concept formed during life** (L9–L11): set of compatible states → hedge over possible outputs → synchronise by inputs where the dynamics forget → reserve, archive replay or an external observation where they conserve → never a point guess.
11. **Extension, not reset** (L12): a new competence is added behind a projection that leaves the old state's dynamics intact.
12. **Reserve contract** [analyst, from L4 and Tur 16]: state which future concept families can be installed exactly and at what storage; no fixed-size reserve serves all.
13. **Construction-improvement claims pass the paid same-memory ablation and the strongest simple baseline, in a related, a changed and a partial regime, with whole-life cost and false accepts** (Tur 17).
14. **Fit and truth are separate quantities** (L15): keep ambiguity (size of the minimal-error set, effective number of candidates) and coverage (`true_state_in_set`, `certain_wrong`) next to accuracy.
15. **Evidence discipline of the programme, worth keeping**: lives are the unit, not trajectories; protocol hashed before results; post-result witnesses labelled as such; corrections supersede without overwriting; equal data is not equal compute.

### 5.5 Reusable mechanisms (with locations)

| Mechanism | Where | Reuse in the new core |
|---|---|---|
| Per-context integer error-count bank, minimal-error-set readout; additive, order-free, resumable | `research/living_learning_consolidation_2026_09_22/consolidation.py:79-100` | graded belief over an enumerated hypothesis set; ambiguity = size of the set |
| Core vs core+cache readouts, purge step, snapshots before/after | `consolidation.py:94-100, 128-130, 190-225` | purge test |
| Equal-evidence control arms (self-replay, mispaired, recent-only, narrow language) | `consolidation.py:110-126` | separate information from computation in any consolidation or imagination step |
| Diagnostic continuation operator; fresh-process resume harness | `consolidation.py:132-139`; `verify_consolidation.py:127-163` | persistence test |
| Exact rational recursive least squares | `exact_witnesses.py:23-42` | exactness check for additive statistics |
| KT counter (estimate + evidence mass) | `research/living_learning_temporal_2026_09_22/temporal_learning.py:116-117, 139` | weight = graded belief |
| Context tree with mixture and MAP structure selection | `temporal_learning.py:78-84, 96-131, 141-149` | priced, reversible formation of distinctions |
| Parallel bank of recurrent candidates, error-weighted; effective-candidate count | `temporal_learning.py:21-23, 67-73, 91-94, 112-114, 133-137, 180` | hypothesis bank over state-update rules; ambiguity measure |
| Frozen-copy evaluation; prequential scoring | `temporal_learning.py:217-239` | honest online measurement |
| Explicit enumeration of all small trees as an oracle; scalar automaton re-implementation | `verify_temporal.py:26-63` | independent verification of a structured learner |
| Exact-fraction "same answer, different learner" witnesses | `verify_learning_state.py:8-28` | regression test for requirement 2 |
| Version-space acquisition with pre-label vote | `research/living_learning_migration_2026_09_22/migration_learning.py:55-65` | acquire a rule from labelled resets |
| Possible-state set transport, output-uniform hedge, label filter, repair-expansion | `migration_learning.py:75-115` | install a state variable under honest uncertainty |
| Reserves (last-8 window, mod-6 counter) and archive replay with cost counters | `migration_learning.py:79-81, 88-98` | explicit reserve contract |
| Coverage measures `true_state_in_set`, `certain_wrong` | `migration_learning.py:123-135` | does the belief contain reality? |
| Exhaustive fibre scan over all 13-bit histories | `verify_migration.py:40-47` | check which states are compatible with what was kept |
| Shared-memo anti-unification of own before/after edits, validity filter, provenance | `research/living_learning_repair_2026_09_22/repair_learning.py:109-137` | acquired change schema |
| Context-signature → template priorities with counts; ranked proposal order | `repair_learning.py:85-99, 281-286` | cheapest carrier of constructive experience |
| One-pass bottom-up bulk application (terminating) | `repair_learning.py:163-182` | apply an acquired change at all matching sites |
| Charged work meter, per-task cap, atomic refusal | `repair_learning.py:36-40, 301` | whole-life cost accounting |
| Paid-disabled ablation arm | `repair_learning.py:186, 297-298` | causal test with equal memory and cost |
| Hidden full-domain check that never flows back | `repair_learning.py:303-307` | reality measure for constructions |
| Correction record that preserves the superseded run | `docs/research/living_learning_migration_2026_09_22/CORRECTIONS.md` | auditability |

### 5.6 Reality measures already implemented in this group (P4) **[analyst compilation]**

| Measure | What it tells | Where |
|---|---|---|
| held-out exactness after purge | the construction lives in the durable state, not in a record | Tur 13 snapshots `consolidated_no_episodes` × `unseen` |
| size of the minimal-error set / effective number of candidates `1/Σw²` | how many constructions the evidence cannot separate | Tur 13 (2 in life 22); Tur 15 (1.0, 2.0, 1.89) |
| root stop mass of the mixture | "no structure of this kind is supported" | Tur 15 (0.977 on parity) |
| `true_state_in_set`, `certain_wrong` | whether the belief contains the truth; confident error | Tur 16 |
| minimax hedge value `1−1/r` | the honest price of what is not known | Tur 16 |
| evidence loss vs full-domain errors; false accepts | fit vs truth | Tur 17 (0 of 1,095) |
| `synthetic_labels`, `repairs`, schema `provenance` | where a belief came from | Tur 13, Tur 16, Tur 17 |
| loss relative to the context-free predictor | adequacy of the whole family (not implemented as a monitor; visible in the data) | Tur 15 (0.4795 vs 0.2515) |

### 5.7 What this group never built

- The hypothesis language is supplied in all four studies (84 parities; depth-8 suffix tree and 64 two-state tables; 14 programs; F17 trees with 108 rebuilds). No study forms it.
- No discovery of context or task boundaries: the context bit, the reset calibration channel and the delimited targets are given.
- No learner whose durable parameters move on unlabeled events; Tur 15 excludes it by design, Tur 13's online learner moves only on labelled events.
- No detection that a family is inadequate, and no extension of a family.
- No learned policy for what to reserve; no learned gate between a fast record and a consolidated belief.
- No approximate, bounded-precision or forgetting version of the exact laws was run (only the drift bound L8 is stated).
- No combination: counting beliefs, priced distinctions, state installation and acquired proposers were each studied alone.
- No continuing improvement of the constructor (frozen after four sources); no schema revised by its own outcomes.
- No neural or continuous substrate, no noise beyond four flipped labels, no scale beyond 64 inputs / 8-bit contexts / a 289-point field.

---

## 6. Analyst tiny checks (not part of the owner's studies)

All checks read result JSON files or use exact fractions; none wrote into the repository; none imported repository modules.

| # | Check | Result |
|---|---|---|
| A | Recomputed the report tables from `paired_summary.json` / `summary.json` of the four studies | all figures match the reports |
| B | Tur 13 hidden context, `consolidation.json`: minimal-error sets of the rich learner | {old}: 11 lives, {new}: 7, {old,new}: 6; minimal error 8–14 of 48 |
| C | Tur 13 noisy day, with episodes present: core vs core+cache on the 24 seen inputs | 0.003472 vs 0.166667 (= 4/24) |
| D | Tur 13 online learner, lives exact on all 64 inputs after 6 / 12 / 18 / 24 events | clean: 6 / 24 / 24 / 24; noisy: 2 / 8 / 17 / 23 |
| E | Tur 15: the 64 two-state tables enumerated on all non-empty words up to length 8 | 22 distinct functions; parity = table 26; last-bit = tables 21, 42; delay-5: none, best agreement 134/256 |
| F | Tur 15 structure trajectory, `summary.json` | delay-5 leaves: 1 (24/24) at 32 labels, 22–32 at 128, 32 (24/24) at 512; parity: spurious split in 3 / 1 / 0 lives at 32 / 128 / 512 |
| G | Exact fractions: KT `(k,k)` after one `1` = `½+1/(4k+4)`; 64 six-bit histories pairwise separated by ≤ 4 zeros; ridge θ; `θ←(θ+y)/2` | 3/4 and 201/400; 2,016/2,016; (10/11, 3/11); 1/2 vs 1/4 |
| H | Tur 16 set sizes re-derived | delay-5: 16, 8, 4, 2, 1; mod-3: 3 throughout |
| I | Tur 17: budget stops vs failures in all 15 method × regime cells | budget_stopped = 192 − accepted in every cell |
| J | Tur 17 per-target-type counts | two-site: 34 / 6 / 0 of 72 (schemas=bulk / ranked-local / plain); four-site: 2 / 0 / 0 of 72 |
| K | Every Turkish quotation in this file searched verbatim in the four studies' documents (`g5_quote_check.py`) | 58 of 58 found |
| L | `file:line` citations of this file checked against the cited lines (`g5_lineref_check.py`) | 209 of 209 contain the cited text |
| M | Numbers of this file re-derived from the result JSON files (`g5_number_check.py`) | 51 of 51 check groups pass |

The three check scripts and their outputs are in `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/prior/_checks_g5/`.

---

## 7. Established / Refuted / Open — across the group

| Established (with the study's evidence) | Refuted or not supported (with the study's evidence) | Open (named by the studies unless marked) |
|---|---|---|
| An episodic record is not a durable rule: cache 68.16 % on all inputs / 49.06 % held-out, 50 % once purged; the consolidated 84-language core is exact on 40 never-seen inputs in 24/24 lives (Tur 13) | "A separate offline ('sleep') phase is necessary": online and deferred end states identical in 72/72, online better in service (0.080 vs 0.484). Scope: additive statistics; no statement about biological sleep (Tur 13) | Value of a separate processing period under latency or energy limits (H25); service quality *during* a transformation (H39) |
| With an additive commutative statistic and an exact final solve, the schedule of learning does not change the end state (proof; 72/72; order reversal) (Tur 13) | "Old episodes must be replayed to keep the old skill": recent-only gives the same result with 2,016 instead of 4,032 checks. Scope: separate context compartments only (Tur 13) | Retention and interference in a *shared* learned representation (none of the four addresses it) |
| Processing without new external evidence cannot identify the world (proof; life-22 witness: two rules, each consistent with exactly four flipped labels, differing on 32 inputs) (Tur 13) | "Replaying one's own outputs teaches the new rule" (0/24); "a perfect supervisor removes every error" (life 22) (Tur 13) | A supervisor that itself learns (H29); a reliability gate between fast record and consolidated belief (cache override 0.1667 vs 0.0035) |
| Opposite demands on the same observable input cost ≥ 1/4 per conflicting input (≥ 1/8 here); with the context observable both rules are exact (Tur 13) | "More consolidation resolves a hidden-context contradiction": joint learning 0.21875, recent-only destroys the old rule (0.5) (Tur 13) | Discovering context from experience (H24); telling label noise from a hidden context [analyst: same symptom, minimal error 4/24 vs 8–14/48] |
| The pairing of input and label carries the information (mispaired replay 0/24); a language without the target cannot acquire it (14-language 0/24) (Tur 13) | "Every new experience needs a new unit or dimension": exact recursive ridge with fixed `O(d²)` statistics, θ = (10/11, 3/11) (Tur 13) | Formation of the representation language itself (H36): supplied in all four studies; isolated effect of language vs compute (H19 confound: 6× the checks) |
| The state needed to answer is the set of continuation classes; horizon ≠ number of states (parity 2 states / unbounded horizon; delay-5 32 states / 5-bit horizon) (Tur 15) | "A longer suffix or a bigger table learns better": the fixed 8-bit table loses to the learned tree in 24/24 lives and is worse than ignorance on parity (0.289 > 0.25); no fixed suffix serves parity (pair loss ≥ 1/4) (Tur 15) | Acquisition of the memory language; a learner covering both kinds of task (the 64-state joint machine is proved necessary, never learned) |
| The depth of useful past distinctions can be acquired from terminal labels alone inside a bounded family: 2 leaves / 32 leaves in 24/24 lives, 100 % on 32–64-bit words (Tur 15) | "A mixture always beats one selected structure": final loss MAP 0.000250 vs mixture 0.000302 (24/24); early loss mixture 0.177 vs MAP 0.232 (Tur 15) | Structure selection without maintaining statistics for every unselected alternative (511 nodes always kept) |
| A recurrent hypothesis inside the family generalises exactly beyond the training lengths: parity 100 % on 32–64-bit words after 32 labelled 8–24-bit episodes, 24/24 lives (Tur 15) | "Confidence inside a candidate family shows that the family is adequate": two-state bank on delay-5, effective candidates 1.89, MSE 0.4706 > 0.25 (Tur 15) | Noticing that the family is wrong and extending it ("Bu deney … genişletme mekanizması içermiyor") |
| Today's answer does not determine tomorrow's learning: KT (0,0) and (99,99) both answer 1/2, then 3/4 and 201/400; infinitely many learning classes (Tur 15) | "Finite consistent evidence certifies the unbounded future" (`P⊕1[h=w*]`); "a fixed-size universal lossless summary exists" (n-bit bound) (Tur 15) | Approximate, forgetting or bounded-precision learning-equivalence with guarantees; which distinctions to keep for future learning, on what evidence and budget |
| Changing representation costs `⌈log₂M⌉` bits kept in advance: 5 bits for parity → (parity, last five bits); 64 joint states (Tur 15) | "A representation built later recovers what was not kept" (Tur 15) | A change of representation chosen by the system inside an uninterrupted episode |
| A correct rule is not a correct present state: 72/72 rules acquired, yet zero-start Brier 1.0167 / 0.8333 (delay, t 0–4) and 1.0 / 0.9167 (mod-3, t 0–15) (Tur 16) | "A correctly learned rule can be switched on from a default state"; "resetting everything is harmless": old parity wrong in 11/24 and 9/24 lives for the whole continuation (Tur 16) | Learning the rule inside the unreset stream (the calibration channel is supplied); joint uncertainty over program and state |
| Forgetting dynamics are synchronised by inputs (32→16→8→4→2→1); conserving dynamics never are (3 states until one external label); a set-valued belief is never certain-and-wrong and always contains the truth (Tur 16) | "Enough further input always removes ignorance of the start state" (mod-3: `H(Q_t ∣ I,w) = H(Q_0 ∣ I)`); "mixing of the average chain = knowing the state" (Tur 16) | Choosing reserves from experience under a budget (each fixed reserve served only the task it anticipated: 0 vs 0.6667 / 0.5) |
| An old competence survives an extension whose projection commutes with the dynamics: five policies, every old answer right (Tur 16) | "A stable top-down/bottom-up loop or a supervisor guarantees correctness" (`F(h)=1`: stable and wrong) (Tur 13 §7) | Many competences under shared, limited resources |
| An experience-derived change proposer alters what is acquired later under equal memory, evidence, budget and booked cost: 84 vs 54 of 192; +19 / =5 / −0 lives (Tur 17) | "Learned schema extraction is a necessary new principle or beats the strong known method": ranked bulk edits give identical accepted programs in 72/72; schemas cost +195.79 units (Tur 17) | A task family in which a learned schema and a ranked generic edit can differ [analyst]; acquisition of the extraction procedure itself |
| Acceptance on finite evidence is not correctness (`f+Π(x−a)` fits all 16 examples); in this run 0 false accepts of 1,095 (Tur 17) | "Acquired schemas help when the family changes" (+0 correct, +3,591.71 units in 24/24 lives; partial +0, +2,638.13); "schema transfer solves deep composition" (2 of 72) (Tur 17) | An applicability estimate revised by outcomes; diagnosis of the internal step that failed; allocation of budget across tasks (every failure is a per-target cap stop [analyst]) |
| Persistence is testable and was tested: a serialised state restored in a fresh process reproduces all later predictions and updates exactly (24 + 21 + 24 + 15 checks) (all four) | "Learning while living is the same thing as general intelligence" (frequency-counter counterexample, H40) (Tur 13) | A learner whose durable parameters move on every unlabeled event (excluded by design in Tur 15); any neural or continuous substrate; scale beyond 64 inputs / 8-bit contexts / F17 |
