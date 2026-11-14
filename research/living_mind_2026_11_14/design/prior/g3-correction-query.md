# g3 — Prior-work digest: correctable state, future-operation state classes, representation geometry, acquired program library (+ papers 07, 08, 09, working note)

Analyst digest for the new-core initiative (leave Transformer / embedding table / softmax-argmax; one process for learning and running; weights as graded beliefs; concepts formed when needed; imagination measured for reality; a way of constructing that improves and transfers).
Repository: `/workspace/cevahir-ai` (read-only). All numbers below were read from the study's own reports and re-read from its result JSON files; where I recomputed something it is marked **[analyst]**. Nothing heavy was run.

Conventions: **PROVED** = algebraic proof or exhaustive exact enumeration inside the stated finite scope. **SUPPORTED** = descriptive result over the study's 12 paired synthetic lives (no population claim). **REFUTED** = an exact counterexample or a measured contrary result. **SUGGESTED** = interpretation the study itself does not claim to have shown. **OPEN** = named open by the study.

The five design pillars are abbreviated P1 (learning = running), P2 (weights = graded beliefs), P3 (concepts formed when needed), P4 (imagine and measure reality), P5 (the way of constructing improves and transfers).

---

## 0. Source map, existence check, integrity

| Item asked | `docs/research/…` | `research/…` | Publication status |
|---|---|---|---|
| `living_learning_correction_2026_09_20` | exists: `REPORT_TR.md` only | exists: `correction_state.py`, `verify_correction.py`, `results/{correction_state,verification,prior_manifest}.json` | **CEV-2026-08** = paper 08 |
| `living_learning_query_state_2026_09_21` | exists: `REPORT_TR.md` only | exists: `query_state.py`, `results/query_state.json` | **CEV-2026-09** = paper 09 |
| `living_learning_geometry_2026_09_21` | exists: `REPORT_TR.md`, `PROTOCOL.md`, `MATHEMATICS_AND_LITERATURE.md`, `POSTHOC_CONTROLS.md`, `REVIEW.md` | exists: `geometry.py`, `posthoc_controls.py`, `endpoint_diagnostic.py`, `verify_geometry.py`, `final_audit.py`, `README.md`, `results/{geometry,posthoc_controls,endpoint_diagnostic,verification,final_audit,preservation_baseline}.json` | "local research round 10"; **not** in `docs/publications` (no match for the folder name there) |
| `living_learning_library_2026_09_21` | exists: `REPORT_TR.md`, `PROTOCOL.md`, `MATHEMATICS_AND_LITERATURE.md`, `POSTHOC_PORTFOLIO.md` | exists: `library_learning.py`, `portfolio_control.py`, `analyze_results.py`, `verify_library.py`, `final_audit.py`, `README.md`, `results/{library_learning,portfolio_control,paired_summary,verification,final_audit,preservation_baseline}.json` | "local research round 11"; **not** in `docs/publications` |
| paper 07 | `docs/publications/papers/07-acquired-recurrent-state.md` | — | CEV-2026-07, edition of `docs/research/living_learning_state_2026_09_20/REPORT_TR.md` |
| paper 08 | `docs/publications/papers/08-correctable-learning-state.md` | — | CEV-2026-08, edition of the correction report |
| paper 09 | `docs/publications/papers/09-future-query-state-classes.md` | — | CEV-2026-09, edition of the query-state report |
| working note | `docs/publications/working-notes/future-state-review.md` | — | internal pre-publication mathematical note (dated 20 Sept, addendum 21 Sept) |

No named folder is missing under either root.

**[analyst] Edition check.** For papers 07, 08, 09 I compared the body after the `---` separator with the source `REPORT_TR.md` after normalising link targets: 45/45, 76/76 and 44/44 non-empty lines identical except the title line (`# …` → `## Araştırma kaydı: …`). The papers add only a front matter (abstract TR/EN, "interpretation of findings", publication notice: "not externally peer reviewed", AI assistance disclosed). So paper 08 ≡ study 1, paper 09 ≡ study 2; they are digested together with their studies. Paper 07's underlying study (`living_learning_state_2026_09_20`) is digested in depth in the sibling file `g2-state-update.md` §1; §6 below is self-contained but compact.

**How the four studies chain.** paper 07 (execution state ≠ relearning archive) → study 1/CEV-08 (even the full current version space is not enough to correct the past) → working note + study 2/CEV-09 (count exactly how much state each future-operation contract needs) → geometry round 10 (the same question in a *learned shared representation*: protection geometry decides future acquisition) → library round 11 (the same question for *acquired composites*: proposal priority decides future acquisition). Each report states "the general living-learning problem is still open" in its first lines.

---

## 1. Study `living_learning_correction_2026_09_20` = CEV-08 = paper 08 — "Correctable learning state: what does today's answer not preserve?"

Files: `docs/research/living_learning_correction_2026_09_20/REPORT_TR.md`; `research/living_learning_correction_2026_09_20/correction_state.py`, `verify_correction.py`, `results/correction_state.json`, `results/verification.json`.

### (1) Question
Does a state that preserves all of today's predictions **and the complete set of currently consistent candidates** suffice to produce the correct learning state when one past experience is later corrected? (`REPORT_TR.md:7`)

### (2) Mechanism / state representation
Purely symbolic, exact integers; no network, no probabilities, no learning-rate. "Experience" = an identified `(x, y)` record with `x∈{0,1,2}`, `y∈{0,1}`.

- Hypotheses: `h∈{0..7}` is the bit code of a truth table, `prediction(h,x) = (h >> x) & 1` (`correction_state.py:25-26`).
- **State A — `CorrectionSummary`** (`correction_state.py:38-69`). Stored: the fixed hypothesis tuple, `counts[i]` = number of violations of *every* hypothesis in `H`, **including eliminated ones**, and `events`. The docstring says "Event contents/IDs are NOT stored here" (line 39).
  - append: `counts[i] += prediction(h, x) != y` (lines 46-49).
  - correction: `replace_label(old_x, old_y, new_y)`: `count − [h(old_x)≠old_y] + [h(old_x)≠new_y]` (lines 51-58); guard `0 ≤ count ≤ events` else `ValueError` (56-57). "The sender must supply the actual old event, not just its ID" (line 52).
  - readout: version space `V = {h : c_h = 0}` (`survivors_mask`, 34-35).
  - Report equations (`REPORT_TR.md:41-50`): `c_h(D)=Σ_i 1[h(x_i)≠y_i]`, `V(D)={h: c_h(D)=0}`, `c'_h = c_h − 1[h(x_j)≠y_j] + 1[h(x_j)≠y'_j]`.
- **State B — `ObservationHistogram`** (lines 72-89), the "simpler strong rival": six bins `bins[x][y]`; append `bins[x][y] += 1`; correction `bins[old_x][old_y] −= 1; bins[old_x][new_y] += 1` with guard "old bin ≥ 1"; readout `c_h = Σ_x bins[x][1−h(x)]` (87-89). Six counters < eight violation counters, and it also supports hypotheses added later.
- Reference: `replay_counts` = full rescan of the corrected history, not sharing the incremental code (29-31).
- Ablation 1 "survivors only": `kept = before_mask & {h : h(x)=new_y}` (174-176). Ablation 2 "violation bit": `int(c>0) − [h(x)≠old_y] + [h(x)≠new_y]` (179-181).
- Persistence: `snapshot()/restore()` = `{hypotheses, counts, events}` (60-69); the continuation packet supplies `[old_x, old_y, new_y]` (245-262).

### (3) Experimental design
- Universe: 3 inputs → 8 Boolean functions → **255** non-empty hypothesis subsets; all histories of length 1–4 (6+36+216+1,296 = **1,554**); every event's label flipped once → **5,910** corrections in the full universe → ×255 = **1,507,050** class–correction cases (`correction_state.py:136-207`). Sub-class results are *projections* of the full-universe counts; the report insists "there are not 1.5 million independent worlds or statistical samples" (`REPORT_TR.md:75`).
- Methods compared against full replay: all-candidate violation counters; six-bin histogram; survivors-only; violation-bit subtract/add.
- Repeated edits: every length-3 sequence of correction indices on every history of length 1–3: **6,126** sequences, **18,378** intermediate states (210-229).
- Persistence: 80 random events saved, 50 corrections applied in a separate OS process (`--probe`), compared with replay (232-262).
- Verification (`verify_correction.py`): exact rerun equality; empty-history edit rejected; no-op edit unchanged; a deliberately wrong provenance packet passes the numeric guard; 88 prior files hash-identical.
- Statistics: none needed — deterministic exhaustive enumeration with assertions.

### (4) Results and decision labels (`results/correction_state.json`, `REPORT_TR.md:77-88`)

| Method | Candidate set different from full replay | by history length 1/2/3/4 |
|---|---:|---|
| All-candidate violation counters | **0 / 1,507,050** | 0/0/0/0 |
| Six-bin observation histogram | **0 / 1,507,050** | 0/0/0/0 |
| Only previously surviving candidates | **360,000 / 1,507,050** | 1,440 / 12,096 / 64,224 / 282,240 |
| Violation yes/no bit with subtract/add | **1,338,336 / 1,507,050** | 0 / 12,096 / 138,528 / 1,187,712 |

Other totals: `nonempty_before` 360,000; `nonempty_after` 360,000; `nonempty_both` 78,534; `empty_before_nonempty_after` 281,466; `binary_false_admission_class_cases` 1,338,336. Repeated edits: 0 mismatches in 18,378 states. Fresh process: exact (final counts `[33,37,36,40,40,44,43,47]`). `verification.json`: `exact_rerun_equal: true`, `wrong_provenance_can_pass_count_guard: true`.

Exact witnesses (`correction_state.py:97-133`, re-derived independently **[analyst]** in `_checks_g3/g3_tiny_checks.py`):
1. `H={h0=[0,0,0], h3=[1,1,0], h5=[1,0,1]}`, `D_A=[(0,0),(1,0)]`, `D_B=[(0,0),(2,0)]`. Both give version space `{h0}`; the same correction "event 0: (0,0)→(0,1)" gives `{h5}` and `{h3}`.
2. `[(0,0),(1,0)]` vs the reversed order: identical full count vector `[0,1,1,2,0,1,1,2]` and identical histogram; "set label of event 0 to 1" gives `{1,5}` vs `{2,6}`.
3. `H_old={const0,const1}`, histories `[(0,0)]×3` and `[(0,0),(1,0),(1,0)]`: same counts `[0,3]`; a hypothesis added later, `[0,1,0]`, has 0 vs 2 violations.

| Claim | Label | Evidence |
|---|---|---|
| "The complete current version space is sufficient for correcting the past" | **REFUTED** (exact) | witness 1; survivors-only fails in **all** 360,000 cases whose correct new set is non-empty (after a genuine flip every old survivor predicted the old label, so the kept set is always empty — a structural, expected failure, `REPORT_TR.md:84`) |
| "A contradicted/not-contradicted bit per candidate is enough" | **REFUTED for the stated ablation** | 1,338,336 false admissions; the report says this is not an impossibility theorem for all bit representations (`REPORT_TR.md:86`) |
| "Violation multiplicities + truthful old content reproduce full replay" | **PROVED** (algebra + induction for all finite histories, fixed finite `H`); verified 0/1,507,050, 0/18,378, fresh process | `REPORT_TR.md:52` |
| "The six-bin histogram does the same with less" | **PROVED** in this closed domain | 0/1,507,050 |
| "An event ID alone identifies what to correct" | **REFUTED** | witness 2 |
| "Counters over the current hypotheses can score a hypothesis added later" | **REFUTED** | witness 3 |
| "The numeric guard is a provenance check" | **REFUTED** | `verify_correction.py:42-49` |
| "The counter method is superior" | **not claimed** — the histogram rival is at least as good | `REPORT_TR.md:54` |

### (5) Law / invariant claimed
**Operation-relative sufficiency** (`REPORT_TR.md:29-35`). Let `c(D)` be the summary, `F(D)` the desired result, `u` a permitted operation. If the answer after the operation is to be computed as `F(uD)=G(c(D),u)`, then for every operation valid on both histories
`c(D)=c(D') ⟹ F(uD)=F(uD')`; and if the summary itself is to be reused, the stronger **closure** condition `c(D)=c(D') ⟹ c(uD)=c(uD')`.
Slogan (`REPORT_TR.md:112`): *"Yeterli durum, gelecekte izin verilen işlemlere göre yeterlidir"* — a state is sufficient only relative to the operations permitted in the future.
Scope: deterministic, exact answers. Rests on witness 1 (necessity side) and on the additive identity (constructive side, fixed finite `H`, truthful content-carrying corrections). The study explicitly says this is the previously known sufficiency-under-append condition applied to correction — "not a new learning algorithm or original basic law" (`REPORT_TR.md:35`); literature: Mitchell's version spaces, Doyle's TMS, Gupta–Mumick–Subrahmanian incremental view maintenance.

### (6) Limits
Admitted: fixed known finite `H`; trusted old contents supplied from outside (that external memory is "not removed from cost or scope"); projected subsets are not independent trials; correctness of the correction is an external assumption (a wrong new label is replayed faithfully); cost is `O(|H| log(n+1))` bits and `O(|H|)` evaluations per update — "a fixed number of counters is not a fixed number of bits"; no byte/time benchmark; no deletion/unlearning guarantee; no automatic error diagnosis.
Seen by me: (a) only binary label replacement was enumerated — no input correction, no deletion, no insertion (deletion is the same subtraction but untested); (b) the readout is exact consistency (`c_h=0`); with noise the version space is empty and the study says nothing about graded choice; (c) both exact states need an *enumerated finite alphabet* — `|H|` counters or `2|X|` bins — which is exactly what a core that forms concepts on demand does not have; (d) no approximate or bounded-memory summary was tested; (e) universe is 3 inputs, histories ≤4 (the algebra is general, the enumeration is not).

### (7) Implications for a core where learning and running are one process
- **Requirement (P2).** A belief must be stored as an *additive, invertible evidence quantity* (a count that can be decremented), not as the current winner, the surviving set, or a saturated bit. Any lossy non-linearity applied to the count before storage (threshold, survivor filter) makes retraction impossible: 360,000 and 1,338,336 failures.
- **Requirement (P2/P3).** Evidence against *currently rejected* alternatives must be kept; pruning a candidate is an irreversible loss with respect to later correction ("the lost information is which other evidence contradicted the eliminated candidates", `REPORT_TR.md:27`).
- **Warning (P3).** A summary indexed by the *current concepts/hypotheses* cannot evaluate a concept formed later (witness 3). On-demand concept formation needs evidence stored at the *observation* level (or an explicit re-acquisition path), not at the hypothesis level.
- **Warning (P1).** A numeric sanity guard is not provenance; "which event" needs its own record or the correction must carry the old content.
- **Mechanism to reuse.** `CorrectionSummary` / `ObservationHistogram` update rules; the *differential replay harness* — every incremental update asserted equal to an independent full rescan over an exhaustive small universe (`exhaustive_single_corrections`), plus the separate-process continuation probe. This is a ready acceptance test for any new belief-update rule.
- **Open question.** Which information a *bounded* memory should keep across unknown future representation expansions and corrections (`REPORT_TR.md:114`); who decides that a correction is trustworthy.

---

## 2. Study `living_learning_query_state_2026_09_21` = CEV-09 = paper 09 — "State required by future learning operations: an exact finite audit"

Files: `docs/research/living_learning_query_state_2026_09_21/REPORT_TR.md`; `research/living_learning_query_state_2026_09_21/query_state.py`, `results/query_state.json`.

### (1) Question
Of the two constructive rivals of study 1 (all-hypothesis counters, full histogram), which one carries the information that is *actually necessary* — i.e. how many distinguishable states does each future-operation contract require on the same histories? (`REPORT_TR.md:11`)

### (2) Mechanism / state representation
Symbolic, exact integers. `X={0..m−1}`, `Y={0,1}`, history of `N` events.

- **Signed summary** `(N, d)` with `d_x = n[x,0] − n[x,1]`: `signed()` does `d[x] += 1 − 2*y` (`query_state.py:22-26`).
- **Readout for every Boolean hypothesis** `h:X→{0,1}` (`query()`, lines 29-32):
  `c_h = (N − Σ_x d_x)/2 + Σ_x h(x)·d_x` (`REPORT_TR.md:19`). So the loss of any hypothesis — including one invented later — is an affine readout of `m` signed integers.
- Updates (`REPORT_TR.md:23`): append `N←N+1, d_x←d_x+1−2y` (code 81-83); content-carrying label change `d_x←d_x+2(y−y')` (code 73-75).
- Histogram: `histogram()` (35-36). ID contract signature: current risk vector followed by the risk vector after *every* `(j,b)` "set label of event j to b" command, stored as exact integer bytes, not a hash (68-79).
- Class-count formulas (`risk_classes()`, 39-45; `REPORT_TR.md:25-31`):
  `A_m(0)=1, A_m(s)=Σ_{k=1}^{min(m,s)} C(m,k)·2^k·C(s−1,k−1)`;
  `K_risk(N,m)=Σ_{j=0}^{⌊N/2⌋} A_m(N−2j)`; `K_hist(N,m)=C(N+2m−1, 2m−1)`; `K_ID=(2m)^N`.
  Ideal fixed-length code: `bits(k)=(k−1).bit_length()` = `⌈log₂K⌉` (48-49).

### (3) Experimental design
21 cells: `m∈{1,2,3}`, `N∈{0..6}`; all `(2m)^N` histories per cell (`audit_cell`, 52-95). Per history: replay (direct scan) vs signed formula for every hypothesis; bijection between `(N,d)` and the full risk vector asserted (64-65); every content edit (both labels, no-ops included) and every single append compared with replay; three kinds of signatures counted exactly and compared with the formulas (87-89). Negative controls in `negative_controls()` (98-115). No statistics; exhaustive. Naive JSON payload sizes are recorded as a secondary, explicitly non-ideal measure.

### (4) Results and decision labels (`results/query_state.json`, `REPORT_TR.md:43-53`)

| Measure | Result |
|---|---:|
| Histories enumerated exactly | 61,575 (127 + 5,461 + 55,987) |
| Content label writes checked (no-ops included) | 712,632 |
| Single-record appends checked | 358,020 |
| Summary-vs-replay query mismatches | 0 |
| Cells where a class-count formula disagrees | 0 / 21 |

Selected cells: `m=3,N=4`: 85 risk classes / 126 histogram classes / 1,296 ID classes → ideal bits **7 / 7 / 11**. `m=3,N=6`: 231 / 462 / 46,656 → 8 / 9 / 16 bits. `m=2,N=6`: 49 / 84 / 4,096 → 6 / 7 / 12. `m=1`: risk = histogram = `N+1` (no gap). Asymptotics for fixed `m`: `Θ(N^m)` risk classes, `Θ(N^{2m−1})` histograms, and a *linear* `N·log₂(2m)` bit lower bound for the ID contract. Mean naive JSON bytes at `m=3,N=6`: all risks 17.0, signed 12.07, histogram 13.0, ID records 37.0.

Negative controls: `[(0,0),(0,1)]` and `[(1,0),(1,1)]` share `N=2, d=(0,0)` and all risk answers, but the count of observation `(0,0)` is 1 vs 0 — and a claimed correction of a non-existent `(0,0)` record in the second history still passes the numeric bounds. `[(0,0),(1,0)]` and its reverse share the histogram; "set event 0 to label 1" gives different risks.

| Claim | Label | Evidence |
|---|---|---|
| "Future hypotheses are unknown, therefore the full histogram (or everything) must be kept" | **REFUTED** | `(N,d)` suffices for all `2^m` Boolean hypotheses, including ones added later (algebra; 0 mismatches) |
| "`(N,d)` is exactly the equivalence class of all-risk answers" | **PROVED** both ways | `d_x = c_{e_x} − c_0`, `N = c_0 + c_1`; bijection asserted in code |
| "The histogram is the minimal state for every contract" | **REFUTED** | 85 < 126 classes etc.; it is sufficient, not minimal, for risk queries |
| "The histogram is necessary once observation-indicator queries are allowed" | **PROVED** (the answers are the bins) | working note §2 |
| "ID-only label edits need all `(2m)^N` histories separated" | **PROVED** + enumerated in 21 cells | separation argument `REPORT_TR.md:31` |
| "Fewer classes means fewer bits" | **caution** — 85 vs 126 are both 7 bits | `REPORT_TR.md:51` |
| "`(N,d)` can validate that a claimed old record exists" | **REFUTED** | negative control 1 |
| "The histogram validates event identity" | **REFUTED** | negative control 2 |
| Minimal state for weaker queries (e.g. only "which candidate is best") | **OPEN** | `REPORT_TR.md:21` |
| Update/query runtime or primitive-operation cost | **not measured** | `REPORT_TR.md:39` |

### (5) Law / invariant claimed
**"The family of future operations determines the state that must be kept."** Precisely (working note §1, `future-state-review.md:13-22`): for a family `Q` of future queries, `D ≡_Q D' ⟺ q(D)=q(D') ∀q∈Q`; with permitted operation sequences `u`, `q(uD)=q(uD')` is required as well. An exact deterministic summary that merges two classes answers some separating query wrongly; hence with `K` classes the worst-case fixed-length state needs `≥ ⌈log₂K⌉` bits. Three instances on the same histories: `K_risk`, `K_hist`, `K_ID` above.
Paper 09's own reading (`09-future-query-state-classes.md:19`): "The required information depends first on which queries and operations will be supported, before whether the hypotheses are known."
Scope: fixed finite `X`, binary labels, exact answers, the family of *all* Boolean risks, the right to edit *every* ID. Rests on the algebraic expansion, the two-case separation argument for IDs, and the exhaustive 21-cell enumeration. The report says the result is not a general algorithm, that the specific three-contract presentation is "not verified to be first in the literature", and maps it to Mitchell (1977) and Cao & Yang (2015).

### (6) Limits
Admitted: exactness, fixed domain, all-Boolean risks and full ID-edit right are "strong assumptions"; approximate learning, bounded correction horizon, re-acquirable records, open input language or distributional performance "give different bounds"; class counts do not supply an efficient encoder; content edits assume truthful old contents; "no minimal neural-network memory can be derived from any of these results" (`REPORT_TR.md:61`). The choice of which future operations to protect was made by the researcher (`REPORT_TR.md:71`: "Deney bu seçimi araştırmacıdan almadı" — I read this as "the experiment did not take this choice out of the researcher's hands"; the wording is ambiguous but the three contracts are in fact hard-coded in `query_state.py:122-124`).
Seen by me: (a) the hypothesis family "all Boolean functions on X" has no inductive structure — evidence at `x` says nothing about `x'` — so this is a *memory* law, not a *generalisation* law; (b) only `m≤3, N≤6` enumerated; (c) the ID separation argument uses no-op writes as legal commands (stated in `limits`), a slightly artificial contract; (d) `(N,d)` keeps one global `N`, so per-cell evidence volume (confidence) is not recoverable from it — see (7).

### (7) Implications for the core
- **Requirement (P1).** The core must carry an explicit, typed **future-operation contract** — which operations may ever be applied to a piece of learned state: append evidence, retract with content, retract by identity, add a concept, ask about an observation feature, re-represent. The state layout must be *derived* from that contract and tested against it; "what it answers today" is not a specification.
- **Mechanism to reuse (P2) — the most direct fit with "weights are graded beliefs" in the whole group.** A weight as a *signed evidence balance* `d_x` (plus a total `N`) is closed under continued learning and content-carrying correction, order-independent, and gives every Boolean hypothesis' loss by an affine readout. Update: `d_x += 1−2y`; correction: `d_x += 2(y−y')`.
- **Warning (P2).** The single signed number loses *how much* evidence stands behind a cell and *where* contradictory pairs sit: `(N,d)` cannot answer observation-indicator queries, cannot give per-cell confidence, cannot check that a retracted record ever existed. If the core wants confidence or validity checking, a belief must be the **pair** (support, counter-support) = the histogram, not their difference.
- **Warning (P1).** Identity-addressed correction ("the thing I told you at event j was wrong") costs memory linear in life length; there is no summary that avoids it under exactness. Either keep an episodic identity→content ledger, bound the correction horizon, or require corrections to carry content — and count the external memory honestly.
- **[analyst] small extension, not a result of the study.** For the weaker query "which Boolean hypotheses are currently best?" under continued appends, the distinguishable classes equal the distinct `d` vectors (brute force `m=2`, `N≤4`, all append sequences up to length 4: 1/4/9/16/25 classes = number of distinct `d`; sign-only gives 1/4/9/8/9). So even the weakest sensible readout needs the full integer balance; only the global `N` can be dropped. A graded belief cannot be reduced to its sign or to a bounded level without losing future learning.
- **Open question.** "An open-world learner must also learn which future operations are worth protecting"; and, under bounded memory and compute, "which irreversible losses are accepted against a later-changing query/representation language and which re-acquisition paths are kept" (`REPORT_TR.md:71`).

---

## 3. Working note `docs/publications/working-notes/future-state-review.md` — "Sufficient state for future learning operations: query classes and identity memory"

### (1) Question
Before running anything: for which outputs *and which operations* is a state "sufficient", and can the minimal state be counted exactly for different operation contracts?

### (2) Mechanism
Pure derivation (status line: "direct derivations and an experiment proposal not yet run when written", `:5`). Objects: finite observation universe `Z`, history `D=(z_1..z_N)`, query family `Q`, equivalence `≡_Q` (`:13-22`). Additive queries `q_g(D)=Σ_i g(z_i) = Σ_z g(z)·n_z`; with the query rows stacked in a matrix `A`, two histograms are equivalent iff `An=An'` (`:26`). The note then derives the signed summary and identity `c_h = (N−Σd)/2 + Σ h(x)d_x` (`:40-58`), its closure under appends and content-carrying edits (`:66-72`), the reachable set `D_{N,m}={d∈Z^m: ‖d‖₁≤N, N−‖d‖₁ even}` (`:78-85`), the counts `A_m(s)`, `K_risk`, `K_hist` (`:87-97`), and the identity bound: for a fixed histogram `K_{ID|n}=N!/Π_z n_z!`, overall `(2m)^N`, i.e. `⌈N log₂(2m)⌉` bits (`:99-124`).

### (3) Design (proposed, §6)
`m∈{1,2,3}`, `N=0..6`; three independently produced signatures; replay must not share the signed-count function; method comparison "all-hypothesis counters; `(N,d)`; histogram; histogram+ID mapping; raw ID records"; measures "query mismatch; class count; ideal code bound; real serialised bytes; primitive operations per update and query"; three negative controls.

### (4) Results / status
The 21 Sept addendum (`:7`) states the class counting, append/correction audit and negative controls were later run (study 2) and that "this does not mean all proposed performance measures were done". **[analyst]** comparing the proposal with `query_state.py`: executed — three signatures, formula counts, replay separation, three negative controls, naive JSON bytes; **not executed** — primitive operations per update/query, a separate "histogram+ID mapping" method, a paid external old-content service.
Worked numbers given as *formula values, not measurements*: `m=3,N=4`: `K_risk=66+18+1=85`, `K_hist=C(9,5)=126`, both 7 bits; 1,296 ID assignments → 11 bits (later confirmed by enumeration).

### (5) Laws stated (all derivations, **PROVED** within their assumptions)
- Class-count lower bound `⌈log₂K⌉` bits for an exact deterministic summary (`:20`).
- If operations are *partial* and the system must also *reject* invalid ones, equality of "which operations are valid" is part of the equivalence (`:22`) — this is why `(N,d)` is fine only under the weaker contract "commonly valid operations".
- The exact minimal bit count is **not** the rank of `A`: it is the number of distinct values `An` takes over reachable integer histograms; counters, their digits and hypothesis definitions are separate costs; "encoding the whole history in one infinite-precision real number does not count as a fixed-memory solution" (`:28`).
- Two-part main finding (`:9`): unknown future hypotheses on a known finite domain do not force the full histogram; ID-only arbitrary edits may force the event mapping. "These belong to different operation contracts; they cannot be merged into a single memory necessity."
- §5 (`:126-132`): "'future hypotheses are unknown, so everything must be stored' is too strong"; but a later query about a discarded observation feature or about record order breaks the fixed-`X` assumption.

### (6) Limits
Admitted: internal AI-assisted review, not peer review; originality scan incomplete — "a publication claim of 'new general lower bound' or 'basic principle of living learning' is not appropriate" (`:156`); the heaviest contract is "artificially" exact and all-ID.
Seen by me: the matrix formulation `An=An'` is the general tool, but only one instance (`A` = Boolean risk rows) was ever counted; other query families (linear, threshold, structured) remain uncounted.

### (7) Implications for the core
- **Mechanism.** The matrix view gives a design calculus: write the future queries of a memory cell as rows `g`; the state the cell must keep is the image `An` of the evidence histogram — nothing more, nothing less. A new query family means a new row set, hence possibly a larger state.
- **Requirement.** Decide for each operation whether the core must *reject invalid uses* (then validity information must be stored) or may assume validity.
- **Warning.** Do not hide state in precision (one real number encoding a history) — count distinguishable states.

---

## 4. Study `living_learning_geometry_2026_09_21` (local round 10, unpublished) — "Keeping old skill and acquiring new capacity in a learnable representation"

Files: `docs/research/living_learning_geometry_2026_09_21/{REPORT_TR,PROTOCOL,MATHEMATICS_AND_LITERATURE,POSTHOC_CONTROLS,REVIEW}.md`; `research/living_learning_geometry_2026_09_21/{geometry,posthoc_controls,endpoint_diagnostic,verify_geometry,final_audit}.py`, `results/*.json`.

### (1) Question
When the representation itself is learned (shared hidden weights), how does the *way the past is protected* change which new skills can still be acquired? (`REPORT_TR.md:5`)

### (2) Mechanism / state representation
Neural, tiny: `2 → 8 tanh → 1`, **33** parameters, NumPy float64, hand-derived Jacobian.
- Parameter layout `unpack` (`geometry.py:38-39`): 8 output weights `a`, 8×2 input weights `w`, 8 hidden biases `b`, 1 output bias `c`. `forward = tanh(x·wᵀ+b)·a + c` (42-44). Analytic Jacobian (47-53). Loss gradient `Jᵀ(pred−y)/n` (56-58).
- Protection basis: SVD row basis with cut-off `rcond·σ_max` (65-68); projection `g − Bᵀ(B g)` (71-72), i.e. `P = I − A⁺A` on the retained anchor-output Jacobian `A`.
- **Learner state** (`Learner`, 107-203): weights `θ`; `mode`, `rate`, `rcond`; stored old labelled examples ("anchors") `x, y`; `reference` = the *initial* outputs on the anchors (112); a fixed `basis` for `tangent_fixed`; cost counters. `snapshot()/restore()` serialise all of it (189-203).
- Update rules in `step` (124-187): `frozen`; `head_only` (hidden gradient zeroed, 136-137); `online` `δ=−η g_new`; `replay` `δ=−η (g_new+g_old)/2` (147); `loss_halfspace` (A-GEM-like: if `g_new·g_old<0` subtract the component along `g_old`, 141-145); `tangent_fixed` (project on the null space of the *initial* anchor Jacobian, 119-122); `tangent_current` (recompute Jacobian + SVD every step, 152-156); `tangent_guard`, `replay_guard` = propose, then try up to 13 candidates `θ+δ·2^{−attempt}` and accept the first whose mean squared anchor-output drift from the fixed reference is `≤1e−4`, else leave the state unchanged (164-178).
- Post-hoc learners (`posthoc_controls.py:14-50`): `online_loss_guard`, `replay_loss_guard` accept if anchor MSE against the **true labels** `≤ initial + 1e−4` (36-48); `replay_two_steps` does two replay updates per minibatch (24, 31-33).
- World (`make_world`, 75-104): teacher in the same class with 3 active tanh units (`a=[1.2,−.7,.6]`, rotated `w`, `b=[0,.2,−.3]`); random rotation per seed; old region `u∈[−1,−.2]`, new `u∈[.2,1]`, bridge `u∈[−.2,.2]`, `v∈[−1,1]`; warm-up 600 full-batch steps at rate .05 on 128 old examples; new stream 128 minibatches × 8; 32 continuation minibatches; 256 held-out points per region. Target stationary and noiseless.

### (3) Experimental design
- Local protocol written **before** results (`PROTOCOL.md`; not an external registry). Four falsifiable claims (`PROTOCOL.md:13-20`), listed in (4).
- 12 seeds × 3 anchor sizes (4/16/64, nested) × 2 rates (.03/.12) × (9 methods at `rcond=1e−8` + 3 tangent methods at `1e−6`) = **864** conditions, 72 summary groups; **36** continuation pairs; post-hoc **216** conditions + 12 offline joint references + 12 endpoint diagnostics. The report: "these are not 1,080 independent worlds" — the same 12 lives reused (`REPORT_TR.md:7`).
- Measures: prequential MSE (prediction before the label), held-out old/new/bridge MSE, anchor drift, projection rank, projected/raw gradient norm ratio, accept/reject counts, cost counters (gradient rows, Jacobian rows, SVD calls, candidate forward rows).
- Statistics: per-group mean, min, max, and `mean ± 1.96·SEM` over 12 seeds (`geometry.py:251-256`); paired per-seed differences; no multiple-comparison correction (`REVIEW.md` point 7). Equal samples, **not** equal compute.
- Continuation intervention (`continuation_probe`, 235-248): clone the final state twice; one copy keeps the protection policy, the other only has `mode='online'` set; same next 32 minibatches.
- Verification (`results/verification.json`): Jacobian vs central finite difference max error 8.23e−11, vs PyTorch autograd 2.22e−16; projector idempotence/orthogonality ~1e−15; fresh-process continuation of all 9 main methods exact; 866 pre-existing files byte-identical; label-leak test in `REVIEW.md` (hold-out labels replaced by −999: identical learner results).

### (4) Results and decision labels
Main table — 64 anchors, rate .12, `rcond=1e−8`, 12-life means (`REPORT_TR.md:41-53`; re-read from `results/geometry.json` **[analyst]**). Start: old MSE .00806143, new .19626949, bridge .03491092.

| Method | Old MSE | New MSE | Prequential MSE | Anchor output drift |
|---|---:|---:|---:|---:|
| frozen | .00806143 | .19626949 | .20458583 | 0 |
| head_only | .05148927 | .00653977 | .01221860 | .04315893 |
| online SGD | .04103444 | .00474944 | .00979798 | .03537611 |
| replay | .01028929 | .00641810 | .01647353 | .00587973 |
| loss half-space | .01045777 | .00399829 | .01109306 | .00698476 |
| tangent_fixed | .00805737 | .19580429 | .20455255 | .00000114 |
| tangent_current | .00806143 | .19626921 | .20458569 | ≈0 |
| tangent_current + output guard | .00806143 | .19626921 | .20458569 | ≈0 |
| replay + output guard | .00802845 | .17455036 | .18265219 | .00009999 |

- `tangent_current`: mean projection rank **31.5 / 33** (not full rank); mean raw new-gradient norm .70445; projected/raw norm ratio **.00010842** (≈0.0108 %). With `rcond=1e−6`: rank ≈25.49, ratio .003694, new MSE .19604527 — still almost no acquisition.
- Anchor count (`tangent_current`, rate .12): 4 anchors → old .01192201 / new .01014484 / rank 4 / ratio .4237; 16 → .00805568 / .17963712 / 16 / .0377; 64 → .00806143 / .19626921 / 31.5 / .000108. With 4 anchors the anchor drift is only ~1.09e−6 while held-out old MSE rises .0081→.0119: protecting stored examples is not protecting the old region.
- `tangent_fixed`, 16 anchors, rate .12: new .02512812 but old .04163159 and anchor drift .03886594 — orthogonal to the *initial* derivatives is not orthogonal to the derivatives after the representation moved.
- Paired: `tangent_current` new MSE > replay new MSE in **12/12** lives at 16 and 64 anchors and both rates; at 64/.12 mean difference .18985111, range [.04519730, .35635837]; at 4 anchors mixed (8/12 at .03, 7/12 at .12 **[analyst]**).
- Output guard on replay rejected a mean **121.58 of 128** steps despite 104,389 candidate forward rows; the guard on `tangent_current` never halved (the proposals were already negligible) — "the guard accepting everything is not learning success".
- Continuation pairs (36, first-prediction difference exactly 0 in all): from `tangent_current`, keeping protection → new .19626914; releasing it → new **.00700841** but old .00806143 → **.04683179**. From replay+guard: .17455036 vs .00698898, old → .04681292. Releasing is "not a free repair".
- Post-hoc (after seeing results, same 12 lives; 64 anchors, .12): online + true-label guard old .00830013 / new .15360665 / 120.17 full rejections; replay + true-label guard .00834211 / .14737338 / 117.75; `replay_two_steps` old **.00720101** / new **.00414022** / prequential .01150301 — both errors below the start *on average*, in 7/12 lives; in the other 5 old MSE rose; maximal transient anchor-MSE increase .00942514 ≫ 1e−4.
- Offline joint reference (extra data and compute, 1,000 joint steps): old .00507431, new .00289452, bridge .00516680. Endpoint diagnostic: **10/12** endpoints satisfy the true-label budget on the 64 anchors, yet **12/12** of the paths taken violate it at some step; the zero-error teacher violates the *output*-preservation budget in **12/12** lives.
- Costs per life at 64 anchors: online 1,024 gradient rows; replay 9,216; two-step replay 18,432; `tangent_current` +8,192 Jacobian rows and 128 SVDs; `tangent_fixed` +64 rows and 1 SVD.

| Protocol claim (to be falsified) | Label | Evidence |
|---|---|---|
| 1. "A step that does not conflict with the old *loss* gradient suffices to keep the old skill" | **REFUTED** (exact) | at a perfectly fitted old example the gradient is 0; witness `f_o=ab+c`: two separately safe acquisitions, combined old loss `h⁴/2 = 1/32` at `h=1/2` (`MATHEMATICS…:34-49`) |
| 2. "A step orthogonal to the old *output* Jacobian preserves old outputs exactly for a finite step" | **REFUTED** (exact + measured) | `f=ab`, `(1,1)`, `δ=(h,−h)`: `Jδ=0` but output `1−h²`; `tangent_fixed` drift .0389 |
| 3. "Protecting more old outputs automatically improves new acquisition" | **REFUTED** in this family | new MSE .0101 → .1796 → .1963 for 4 → 16 → 64 anchors |
| 4. "A finite-step accept test that really tries the candidate turns good retention into good acquisition" | **REFUTED** in this family | 117–122 of 128 steps rejected; new MSE stays .147–.175 |
| "Preserving the initial outputs = preserving the correct old skill" | **REFUTED** | bound in (5); teacher outside the output budget 12/12 |
| "Identical prediction weights ⇒ identical future acquisition" | **REFUTED** (intervention) | 36/36 pairs |
| "A feasible endpoint has a feasible path under the tested proposal rule" | **not for the 12 sample paths**; general statement **INCONCLUSIVE** ("12/12 sample-path violations are not an impossibility for all paths") | `POSTHOC_CONTROLS.md` |
| "More replay solves it" | **PARTIAL / post-hoc** | means improve; 7/12 lives; no uninterrupted protection; not equal compute |
| "A general plasticity-loss mechanism was found" / "loss-constrained optimisation is impossible" | **not claimed** | `REPORT_TR.md:55, 93` |

### (5) Laws / invariants (all in `MATHEMATICS_AND_LITERATURE.md`; the study calls them known linear algebra / known linearisation limits, mapped to GEM, A-GEM, OGD, NTK-overlap, TRPO)
- **First-order reachability under protection** (§3, PROVED): with `A=J_old(θ)`, `B=J_new(θ)`, `P=I−A⁺A`: `Aδ=0 ∧ Bδ=r ⟺ r∈range(BP)`. A strictly descending direction for the new loss inside the protected tangent space exists **iff** `Pg≠0`. Scope: one point, Euclidean metric on the given parametrisation; exact for finite steps only in affine output models. Counter-scope witness: old output `a`, new output `b²`, start `(0,0)`: `BP=0` yet `(0,h)` keeps the old exactly and lowers the new loss — "no first-order direction is not global unlearnability".
- **Finite-step sufficient bounds** (§4): `L_o(θ+δ)−L_o(θ) ≤ ∇L_oᵀδ + (β/2)‖δ‖²`; `‖F_o(θ+δ)−F_o(θ)−Aδ‖₂ ≤ (K/2)‖δ‖²`, so with `Aδ=0` the mean squared drift is `≤ K²‖δ‖⁴/(4m)`.
- **Output drift vs true risk** (§5): `R−R₀ = (2/m)(F₀−y)ᵀd + D ≤ 2√(R₀D) + D`; `D=0` also freezes the initial errors. Mean risk is not each skill's risk: `L_A=(θ−1)²/2, L_B=(θ+1)²/2`, `θ: 1/2→1/4` lowers the mean 5/8→17/32 and raises `L_A` 1/8→9/32.
- **Three separations** (`REPORT_TR.md:27-37, 97-103`): local direction ≠ finite step ≠ reachable output; feasible endpoint ≠ feasible path; *capacity* ≠ *capacity permitted by the constraint* ≠ *capacity reachable with the available computation*.
- **Learning state includes evidence and policy** (`REPORT_TR.md:81`): "the stored evidence and the protection/update policy applied to it are part of the learning state".
All exact witnesses re-derived with fractions **[analyst]**; the risk bound spot-checked on 2,000 random cases.

### (6) Limits
Admitted (`geometry.json:limits`, `REVIEW.md`): the old-data cut and the first training phase are supplied; fixed finite network, no growth; stationary noiseless target, no actions; equal samples not equal compute, counters are not FLOPs; 12 lives reused; post-hoc controls chosen after seeing results; `1.96·SEM` normal intervals without multiplicity correction; the "0" in rank/linearised-change fields of non-tangent methods means *not measured*; no new principle or algorithm.
Seen by me: (a) at 64 anchors there are more constraints than parameters (33) — near-saturation is built in; the more informative case is 16 anchors, where 17 directions remain free and the projected gradient is still only 3.8 % of the raw one, i.e. old and new skill are genuinely entangled in the shared units; (b) `head_only` already reaches new MSE .0065 while wrecking old (.0515): the conflict sits in the shared *readout* as much as in the features; (c) one architecture, one teacher family, one metric (Euclidean projection — no function-space metric tried); (d) the guard can only shrink a given direction (acknowledged), so "accept tests fail" is a statement about that proposal generator; (e) the alternative most relevant to P3 — *adding* units/capacity for the new region, or a local/sparse representation where `A` and `B` have nearly disjoint support — was not tested; (f) the budget 1e−4 is ~1.2 % of the initial old MSE, a very tight contract.

### (7) Implications for the core
- **Warning (P1, the central one).** If running and learning share one dense parameter vector and the past is protected by constraining updates of that vector, the system either freezes (projected gradient 0.01 % of raw) or forgets (online: old MSE ×5). In this experiment only *keeping evidence and refitting jointly* (replay) moved both errors down, and even that with transient violations. Retention by constraint on shared weights is not a foundation for learning-while-running.
- **Requirement (P3).** New skill needs somewhere to go that is not the old skill's parameters: on-demand capacity (a new concept/unit) or structural locality, so that `range(BP)` is not empty by construction. Compare study 5, where old programs are stored as independent immutable expansions and retention is "structurally easy".
- **Requirement (P2/P4).** The retention contract must be stated against *correct targets with a budget, per skill* — not against frozen outputs (which also freeze errors) and not against a mean (which hides a sacrificed skill).
- **Requirement (P1).** Learning state = (function, retained evidence, update/protection policy). All three must be persisted and all three must be named when two states are called "the same"; the continuation-pair intervention is the test.
- **Mechanism to reuse (P4).** "Propose → actually evaluate the candidate on retained evidence → commit or leave unchanged", with a *fixed* reference (not a tolerance that accumulates step by step). Use it as a safety layer only: the study shows it does not create good candidates, and that the number of accepted updates is not acquisition.
- **Mechanism to reuse.** The reachability test `r∈range(BP)` / `Pg≠0` as a cheap *diagnostic* that tells the core "no first-order room here → allocate new structure or compensate jointly" rather than as an update rule.
- **Open question.** "When a feasible end state exists, how can a good update path be found from data with bounded computation?" — direction change, joint compensation (`(h,h,−h²)`), curved paths (`(1+h, 1/(1+h))`) or representation change; "which is necessary under which assumptions has not been shown" (`REPORT_TR.md:140`). Also the three next lines the report names (`:138`): acquiring a genuinely new internal feature/operation and using it in *another* problem; assigning a delayed outcome to the internal change that caused it; separating "fits this family better" from "acquires better in a new family" by intervention.

---

## 5. Study `living_learning_library_2026_09_21` (local round 11, unpublished) — "How does experience change future computation? Acquired program pieces"

Files: `docs/research/living_learning_library_2026_09_21/{REPORT_TR,PROTOCOL,MATHEMATICS_AND_LITERATURE,POSTHOC_PORTFOLIO}.md`; `research/living_learning_library_2026_09_21/{library_learning,portfolio_control,analyze_results,verify_library,final_audit}.py`, `results/*.json`.

### (1) Question
Given the same base language and the same experiences, can pieces of previously acquired programs change — for better and for worse — which *later* programs are found within a fixed computation budget? (`REPORT_TR.md:5`, `MATHEMATICS…:11`)

### (2) Mechanism / state representation
Symbolic program synthesis; no network.
- Language: values in `F_101`; four primitives `op0: x+1`, `op1: 2x`, `op2: x²`, `op3: −x` (`library_learning.py:29-34`); a program is a finite sequence (`execute`, 37-41).
- **Search** `synthesize` (44-98): breadth-first over *token* sequences; each queue entry holds the output vector on the 8 search inputs; candidates whose 8-output signature was already seen are pruned (91-96); **every token is charged its full primitive length** (77-80) — a macro is not free; the first candidate matching the 8 search outputs is tested **once** on 8 *separate acceptance* examples and is either `accepted` or `acceptance_rejected` (the learner then abstains; acceptance labels are never reused to pick another candidate) (65-70, 89-90). Other end states: `search_budget_exhausted`, `budget_before_acceptance`, `observed_signature_space_exhausted`.
- **Learner state** `build_state` (122-124): `programs` (the accepted programs, fully expanded to primitives), `library` (≤4 pieces), `mining` diagnostics. Raw input–output rows are *not* kept. ~441 bytes of JSON after 12 tasks.
- **Update rule = library mining** `mine_library` (101-119): count contiguous sub-sequences of length 2 or 3, at most once per program; eligible if support ≥2 and `score = support·(len−1) − len > 0`; sort by (−score, −len, lexicographic); keep the top 4. Example recorded for life 0: programs' pieces `[0,1,0]` support 7 score 11; `[1,0]` 10/8; `[0,1]` 8/6; `[0,0,1]` 2/1.
- **How state enters future computation** `token_policy` (127-149): `learned_library` and `expanded_proposals` both use `tuple(library) + PRIMITIVES` — library pieces are proposed *first*; `frequency_prior` reorders primitives by past frequency; `whole_program_cache` proposes up to 4 whole past programs first; `static_all_chunks` = all 80 length-2/3 pieces + primitives; `random_library` = same number/lengths of random pieces; `oracle_motifs` = the two generating motifs (privileged).
- History acquisition `acquire_history` (237-245): the 12 warm tasks are all solved by primitive-only search (budget 32,768); the learner's *own* accepted programs go to the bank, never the teacher's.
- World `make_world` (194-220): two "home" 3-operation motifs (chosen among triples not equal on the whole domain to any program of length ≤2); 12 warm tasks = a motif alone or with one primitive prefixed/suffixed (the first two are the bare motifs — "a favourable curriculum"); probes: 8 **related** (new compositions of the two home motifs, half with an extra primitive) and 8 **shifted** (same construction from two *other* motifs of the same language). All 28 targets per life differ as whole-domain functions.

### (3) Experimental design
- Local protocol before results (`PROTOCOL.md`), four claims (below). 12 lives × 3 history sizes (0/6/12) × 2 budgets (512 / 2,048 primitive-vector steps; one step = one primitive applied to the 8 examples) × 16 probes × 8 methods = **9,216** conditions; post-hoc portfolio **384**. "These are not 9,600 independent worlds."
- Probes are run on *copies* of the state; probe results are not written back — the study measures acquisition *after* a history, not a continuing life.
- Success = the accepted program is correct on all 101 inputs (evaluator only).
- Statistics: counts out of 96 (12 lives × 8 tasks), paired gained/lost task counts, per-life differences, cost on jointly solved tasks; explicitly descriptive, "no multiple-comparison or population-generalization claim" (`paired_summary.json:scope`).
- Interventions/verification (`results/verification.json`): library erase with program bank intact (192 future searches); teacher/task-name/family metadata changed (48 cases, learner result unchanged); instrumented primitive counting (48); 16 fresh-process resumes with exact JSON equality; independent scalar interpreter on 85 programs and 40 macro compositions; full reruns; 886 prior files byte-identical.

### (4) Results and decision labels
After the 12-task history (`REPORT_TR.md:47-56`; re-read from `results/library_learning.json` **[analyst]**), successes out of 96:

| Method | Related / 512 | Shifted / 512 | Related / 2,048 | Shifted / 2,048 |
|---|---:|---:|---:|---:|
| primitive-only (no history) | 25 | 21 | 73 | 69 |
| primitive frequency prior | 34 | 20 | 73 | 68 |
| **learned pieces** | **86** | 10 | **94** | **14** |
| same priorities as primitive expansions | 86 | 10 | 94 | 14 |
| whole-program memory | 80 | 7 | 95 | 15 |
| all length-2/3 pieces (static) | 6 | 6 | 23 | 19 |
| random pieces | 14 | 15 | 28 | 37 |
| oracle home motifs | 96 | 10 | 96 | 26 |

- Learned library by history size (related/512, shifted/512, related/2,048, shifted/2,048): 0 → 25/21/73/69; 6 → 84/9/90/24; 12 → 86/10/94/14.
- Paired learned vs primitive-only at history 12 (`paired_summary.json`): related/512 **+62 / −1** (every one of 12 lives better); related/2,048 +23 / −2; shifted/512 +4 / −15; shifted/2,048 **+0 / −55** (every one of 12 lives worse).
- History 6 → 12 for the learned library: related/512 +8 / −6; related/2,048 +6 / −2; shifted/512 +4 / −3; shifted/2,048 +1 / −11. Witness: life 7, `related4`, budget 2,048 — after 6 tasks the target is found in **478** steps (`[2,2,0,3,0,0,2]`, correct on the whole domain); after 12 tasks the re-selected library exhausts **2,048** steps with no answer. No wrong evidence, no old answer changed (libraries **[analyst]**: `[[0,0,2],[2,0,3],[2,0]]` → `[[0,0,2],[0,2],[0,0],[2,0]]`).
- Equality control: `learned_library` ≡ `expanded_proposals` in **1,152 / 1,152** paired conditions (program, status, every counter).
- Library-erase intervention: 144 old programs' whole-domain answers unchanged; 192 future searches equal the primitive-only results and counters.
- Post-hoc portfolio (half budget prior-free + half budget learned, primitive branch has priority): shifted/2,048 14 → **40** (base at full budget 69); related/2,048 94 → **96**; related/512 86 → 71; shifted/512 10 → 12 (base 21).
- Cost: warm-up 1,374.08 steps per life [1,018; 2,154]; mining scans 49.83 windows; state 441.25 bytes [431; 450]; largest queue 969; related/2,048 mean charged steps 231.78 (learned) vs 1,152.20 (base); on the 71 jointly solved tasks 156.56 vs 876.28; on the 14 jointly solved shifted tasks 489.21 vs 441.93. False accepts: 0 in all main and post-hoc conditions. Warm-up: 144/144 acquired programs correct on the whole domain.

| Claim | Label | Evidence |
|---|---|---|
| P1 "Pieces mined from experience can ease acquisition on related later tasks (not guaranteed everywhere)" | **SUPPORTED** | 86 vs 25, 94 vs 73 |
| P2 "A library can raise cost/failure on unrelated tasks" | **SUPPORTED** (negative transfer measured) | 14 vs 69; +0/−55 |
| P3 "A fixed interpreter proposing the same pieces as primitive expansions behaves identically" | **SUPPORTED** (1,152/1,152) — see limit (i) | `library_learning.py:288` |
| P4 "Simpler rivals must be compared" | done: whole-program memory ≥ learned pieces at 2,048 (95 vs 94) | "it cannot be argued that extracting shared sub-pieces is necessary in this curriculum" (`REPORT_TR.md:62`) |
| "More correct experience ⇒ more future acquisition in every direction" | **REFUTED** | 6→12 losses; life-7 witness |
| "A larger fixed library is better" | **REFUTED** here | 6/96, 23/96 |
| "A prior learned from correct experience is at least as good as a random one" | **REFUTED** on shifted tasks | random 37 vs learned 14 (2,048); 15 vs 10 (512) — "does not show randomness is a general solution" |
| "An acquired library adds expressive power / a new computability class" | **REFUTED** (proof + equality control) | (5a) |
| "Reserving prior-free budget removes the negative transfer" | **PARTIAL, post-hoc** | 40 < 69; costs 15 related successes at 512 |
| "A new primitive or language was invented" | **not claimed** | `REPORT_TR.md:27` |

### (5) Laws / invariants (`MATHEMATICS_AND_LITERATURE.md`; literature: DreamCoder, OOPS, Stitch — "not an implementation of these")
- **(a) Expansion equivalence** (§2, PROVED): for a finite, acyclic macro library `L` over base language `P₀`, `⟦p⟧_L = ⟦expand_L(p)⟧_0` for all `p∈P_L`; hence the two languages compute the same function set (assuming no program-length bound, no side effects/shared state/randomness, no external oracle, no new recursion).
- **(b) Budgeted reachability is not invariant** (§3): the tried sequence `S(L,π,E,B)` depends on library, policy, examples and budget. Witness: over `{a,b}` in length-then-lexicographic order `x↦xaaa` is the 8th candidate; with macro `M=aaa` proposed first it is the 2nd. And (§4) `x↦xb` is 3rd in the base order but 4th with the macro: a budget of 3 candidates now fails. Counting: `Σ_{j≤d} b^j` vs `Σ_{j≤d}(b+k)^j` candidates.
- **(c) Costs of abstraction** (§4): description gain of a macro used `k` times `=(k−1)ℓ − h − kc`; lifetime gain only if `C_build + Σ_i C_{L,i} < Σ_i C_{0,i}`, i.e. `n > C_build/s`, and `s` can become negative when the future distribution changes.
- **(d) Evaluation principle** (`REPORT_TR.md:135`, stated as "an evaluation requirement, not a sufficient algorithm"): *a learner state must be judged not only by the answers it gives today but by which changes it can acquire tomorrow under the same new evidence and the same resources.*
- **(e) Portfolio guarantee** (`POSTHOC_PORTFOLIO.md`): every task the prior-free branch accepts correctly at *half* budget is preserved; nothing is guaranteed relative to the full-budget baseline; a false accept in the primitive branch could displace a correct learned program.
- **(f) Definition vs claim** (§6): a program's *definition* (immutable composition) and the *claim that it solves a task* (supported by finite evidence) are different objects; correction should re-verify dependent claims, not silently rewrite shared definitions — "immutable versioned definitions + evidence/claim dependencies".
- The report's own placement (`:121`): "the role played by permitted-update geometry in the previous round is played here by candidate-generation priority. The two mechanisms are different; they are not reduced to one universal law."

### (6) Limits
Admitted: language, interpreter, search procedure, task boundaries and label meaning are given; no new primitive; favourable curriculum; noiseless finite field; the "shift" is a motif-family change inside the same language; signature pruning uses only 8 observed inputs and may prune a correct program; only the first fitting candidate is ever tested; the program bank grows with the number of tasks (no bounded-memory solution); mining, sorting and object costs are outside the step budget; no correction of wrong old evidence; no delayed outcomes; the selector itself is not changed by experience.
Seen by me: (i) the `learned ≡ expanded` equality holds **by construction** — both methods receive the same token tuple and the same BFS; they differ only in lines 82-87 (apply the token as one call or primitive by primitive). It is a faithful demonstration that the knowledge lives in the proposal order, not an independent empirical confirmation. (ii) The miner does not recover the generating concepts (life 0: home motifs `[0,1,0]`, `[1,0,3]`; mined `[0,1,0]`, `[1,0]`, `[0,1]`, `[0,0,1]`) and still wins — success is about useful fragments, not about identifying "the" concept. (iii) There is no feedback from probe outcomes to the library (no utility estimate, no demotion): P5 was tested for one step (history → library), never recursively. (iv) With tokens tried first in a depth-ordered search, negative transfer is nearly mechanical: branching 4 → 4+k per level, each piece costing 2–3 steps. (v) Because the world is noiseless the acceptance test was never stressed (0 false accepts), so P4 is untested here.

### (7) Implications for the core
- **Mechanism to reuse (P3).** A concept formed from use: a composite that recurs in ≥2 *own, accepted* constructions with positive compression score; stored as an immutable expansion; old constructions stay expanded so re-ranking the library cannot change their meaning. Retention is then structural, not a constraint on updates (contrast study 4).
- **Requirement (P5).** A concept is a *prior over the order of construction*, and it must pay: charge every use by its full expansion; cap or price the library (static-all 6/96; life-7 regression from a 4th piece); keep a **reserved prior-free construction budget** so that learned priorities can never fully starve fresh search (portfolio, OOPS-style).
- **Requirement (P5/P4).** Evaluate every change to the construction method by *acquirability*: clone the state, give related **and** family-shifted probes under a fixed budget, report paired gained/lost counts against the prior-free control and against the simplest reuse control (whole-construction memory), and against a same-priority fixed-interpreter control.
- **Warning (P5).** "More correct experience" is not monotone, and a correct prior can be worse than a random one off-family. A core that improves its way of constructing must be able to *demote* a construction habit, and must expect transfer to be signed.
- **Warning.** Do not claim new expressive power for acquired composites; claim changed reachability under a stated budget (law a/b).
- **Mechanism (P4).** Separate acceptance evidence, tested once, with abstention on failure; and the definition/claim split of (5f) for later correction.
- **Open question** (`REPORT_TR.md:137`): "with what evidence does experience earn the right to change the candidate generator or the update rule, and does that change support acquisition in a new family?" and the acquisition of observational distinctions absent from the initial language — "finding a program earlier must not be confused with a system obtaining information it could not represent at all".

---

## 6. Paper 07 (CEV-2026-07) `docs/publications/papers/07-acquired-recurrent-state.md` — "A recurrent state procedure acquired from experience, and relearning memory"

Edition of `docs/research/living_learning_state_2026_09_20/REPORT_TR.md`; details in `EXPERIMENT.md`, `FAILURE_IDENTIFIABILITY.md`; code `research/living_learning_state_2026_09_20/{state_merging,recurrent_state,evidence_audit}.py`. (In-depth digest: `g2-state-update.md` §1.)

### (1) Question
Can the terminal outcomes of short episodes yield a persistent transition procedure that processes much longer unseen sequences without reading the raw past — and what must still be kept in order to go on learning? (`07-…:53-56`)

### (2) Mechanism / state representation
Symbolic; a deterministic finite transition graph over a binary alphabet.
- `state_merging.fit` (`state_merging.py:122-200`): build the prefix tree of all labelled words; unlabelled prefixes are `UNKNOWN`; RPNI-style red/blue merging in short-lex order; `_try_merge` (62-112) merges two states plus the closure needed for determinism and **rejects the merge if two observed, different terminal labels would collide** (89-92); if a blue state merges with no red state it is **promoted to a new state** (178-180); afterwards unknown outputs → 0, missing transitions → a rejecting sink. No bound on the number of states is given to the learner.
- Running state: the graph + the current state in the episode (one transition per symbol; raw past not read). Learning state: additionally the **archive of all labelled words** (256 words, mean 2,094 symbols) from which the graph is re-inferred at 16/32/64/128/256 labels, only at episode boundaries.
- Baselines: exact elimination over all DFAs with ≤3 states (5,898 descriptions → 1,054 distinct languages: 2/24/1,028), read as first survivor or as vote; last-8-symbols counts; whole-word table.

### (3) Design
Four world families: 16 distinct three-state targets; 4 parity; 4 "ones mod 4 = 0" (needs 4 states); 8 corrupted (32 of 256 labels flipped, each on a *different* word). 256 distinct words of length 1–12 per life; the learner predicts at the end of a word, then receives one terminal label. Evaluation: 128 new short + 64 new long words (lengths 32–256); Brier `(p−y)²`; exact equivalence to the hidden target by a product-automaton check never shown to the learner.

### (4) Results and labels (`EXPERIMENT.md:50-78`; `results/recurrent_state.json`)

| Final Brier on long words | 3-state family | parity | 4-state rule | corrupted labels |
|---|---:|---:|---:|---:|
| learned merged graph | 0 | 0 | 0 | .287109 |
| elimination, first survivor | 0 | 0 | .25* | .25* |
| elimination, vote | 0 | 0 | .25* | .25* |
| last eight symbols | .206508 | .264526 | .240724 | .227182 |
| whole-word table | .25 | .25 | .25 | .25 |

`*` family emptied → abstention scored as .5. Equivalence on all non-empty words: 16/16, 4/4, 4/4 (14/16 including the never-observed empty word; kept as recorded and explained, not silently fixed). Online service Brier in the clean family: merged .050781, first survivor .016113, vote .009146 — the bounded-enumeration rival learns with fewer online errors. Corrupted: **36–42 states** (mean 39.375) for 3-state truths, 0/8 equivalent, range .109375–.609375; with trusted external relabelling the stored words repair 8/8. Persistence: graph + current state resumed in a new process, remaining 385 symbols identical with 127 prefix symbols not stored; full learning resumed at episode 37 for 53 episodes incl. a re-inference. Cost: copied states in merge attempts 8,530.6 (clean) vs 440,572.8 (corrupted).

| Claim | Label |
|---|---|
| "Experience can become an executable transition procedure that generalises to much longer sequences" | **SUPPORTED** in the clean families (exact equivalence on the used domain) |
| "State merging is better than the bounded exact rival" | **not supported** — same end result, rival better online (with a privileged 3-state bound) |
| "Training consistency + structure growth ⇒ a valid new structure" | **REFUTED** (36–42 states, 0/8) |
| "An emptied candidate family tells what went wrong" | **REFUTED** — wrong labels and a true 4-state rule give the same symptom |
| "First identify the physical type of the error, then learn" as a necessary general principle | **not supported** (`07-…:58`) |
| "Small execution state ⇒ small learning memory" | **REFUTED** — ~3 states to run vs 256 words to relearn |
| "A fixed window of the last k symbols can carry a long dependency" | **REFUTED** (exact: `0^{k+1}` vs `1·0^k` for parity) |
| "The system discovered which labels were wrong" | **not shown** — the repair used external truth |

### (5) Law
`h ∼ h' ⟺ for every permitted continuation u: f(hu)=f(h'u)` (`07-…:92-97`). If the relation is preserved when a symbol is appended, the classes form a transition procedure `δ([h],a)=[ha]`: the system carries the *distinction needed for future use*, not each past event. Scope: fixed deterministic target; known automata / predictive-state theory (Oncina & García 1992; Shalizi & Shalizi 2004) — "no claim of a new discovery". Companion negative (`EXPERIMENT.md:96`): for every finite training set there is a different finite-state target agreeing on it (parity vs a machine differing only on `0^13`), so no unconditional certainty without class/evidence assumptions. Also recorded (`recurrent_state.json:mathematical_checks`): a numeric state identity carried from the old graph into a re-inferred graph can be wrong (old id 1 ↔ correct new id 2).

### (6) Limits
Admitted: binary alphabet, episode boundaries, correct terminal attribution and the finite-state language are given; the full archive is kept (no bounded lifelong memory); re-inference only at boundaries (state transport mid-episode not solved); corrections used external truth; counters are not total cost.
Seen by me: the merge criterion is hard consistency — one label per word, no evidence multiplicity — which is exactly why noise is "explained" by growth; the study has no graded evidence at all.

### (7) Implications for the core
- **Mechanism (P3).** "Promote to a new state only when no merge with an existing state is consistent" is a concrete, tested rule for *forming a concept when needed*; states = equivalence classes of histories under future continuations.
- **Warning (P4).** The same rule manufactures concepts out of noise (39 states for a 3-state truth). On-demand formation must be paired with a *reality measure* that is not training consistency: graded evidence multiplicity (study 1/2), held-out acceptance (study 5), a price on structure (study 5c).
- **Warning (P1).** Running memory, relearning memory, correction memory and diagnostic information are four different things with four different sizes; a design that persists only the running state cannot keep learning.
- **Open question.** Which retained experience supports *both* new-structure acquisition and correction of wrong structure; how to tell noise from a richer truth with accessible evidence; how to carry the in-episode state across a change of the graph.

---

## 7. Cross-study synthesis for the new core

### 7.1 Laws of this group (numbered for citation)

| # | Law (precise form) | Scope / what it rests on | Status in the studies |
|---|---|---|---|
| G3-L1 | **Operation-relative sufficiency.** A retained state must separate two histories whenever some permitted future operation sequence followed by a permitted query separates them; for reuse it must be closed: `c(D)=c(D') ⟹ c(uD)=c(uD')`. With `K` classes, exact deterministic state needs `≥⌈log₂K⌉` bits. | deterministic, exact; witness 1 of CEV-08; counts of CEV-09; Nerode relation of paper 07 | PROVED in finite scope; declared "known, applied" |
| G3-L2 | **Present answers under-determine future learning.** Equal version space (CEV-08), equal risk vector (CEV-09 under ID edits), equal weights and predictions (geometry, 36 pairs), equal old answers (library, life 7) are each compatible with different futures. | four independent constructions | REFUTES "same function ⇒ same learner" |
| G3-L3 | **Additive invertible evidence.** A summary that is a sum over events is closed under append and under content-carrying retraction: `c' = c − φ(old) + φ(new)`. Thresholding the sum breaks this. | fixed finite alphabet; trusted content | PROVED; 0/1,507,050; 0/712,632 |
| G3-L4 | **Hypothesis-indexed summaries do not survive language growth; observation-indexed ones do while the observation alphabet is fixed.** `(N,d)` suffices for every later Boolean hypothesis; counters over old `H` do not. | CEV-08 witness 3; CEV-09 identity | PROVED |
| G3-L5 | **Identity needs its own memory.** ID-addressed edits require the ID→content map: `(2m)^N` classes, `⌈N log₂(2m)⌉` bits; numeric guards are not provenance. | exact, all IDs editable | PROVED |
| G3-L6 | **First-order reachability under protection.** `Aδ=0, Bδ=r` solvable iff `r∈range(BP)`, `P=I−A⁺A`; descent exists iff `Pg≠0`. | one point, Euclidean metric; exact only for affine outputs | PROVED; measured ratio 1.08e−4 |
| G3-L7 | **Endpoint ≠ path; capacity ≠ permitted capacity ≠ reachable capacity.** | 10/12 feasible endpoints, 12/12 sample paths violating; teacher outside output budget 12/12 | SUPPORTED (sample paths; not an impossibility) |
| G3-L8 | **Protect correct targets, per skill, with a budget.** `R−R₀ ≤ 2√(R₀D)+D`; a mean can fall while a skill is lost. | algebra | PROVED |
| G3-L9 | **Acquired composites change reachability, not expressibility.** `⟦p⟧_L=⟦expand_L(p)⟧_0`; the same knowledge can be carried as a library or as a proposal order. | finite acyclic macros | PROVED; 1,152/1,152 (by construction) |
| G3-L10 | **Transfer is signed and non-monotone in experience.** | 12 lives, one language | SUPPORTED: +62/−1 related, +0/−55 shifted |
| G3-L11 | **Consistency plus growth is not validity.** | 8 corrupted lives | SUPPORTED: 36–42 states, 0/8 |
| G3-L12 | **Four memories.** Execution state, relearning evidence, correction provenance and diagnostic information are distinct and differently sized. | paper 07, CEV-08, CEV-09 | SUPPORTED/PROVED per part |

None of these is claimed by the owner as a new law; every report maps them to existing literature. Their value for the new core is as *constraints the design must satisfy and tests it must pass*.

### 7.2 One table: which future operation needs which state

| Future operation the core promises | Minimal state shown sufficient | Shown insufficient | Source |
|---|---|---|---|
| answer today's queries | current function | — | — |
| process future symbols of an episode | Nerode class (graph + current node) | fixed window of last k symbols | paper 07 |
| re-infer structure from later labels | archive of labelled words | the graph alone | paper 07 |
| append evidence, read any Boolean hypothesis' loss (even one added later) | `(N, d)` | counters over the current hypotheses | CEV-09, CEV-08 |
| pick the best hypothesis after further appends | `d` (integer) **[analyst]** | `sign(d)` | §8 check C |
| answer "how often was this exact observation seen" / per-cell confidence / reject a bogus retraction | histogram `n[x,y]` | `(N, d)` | CEV-09 control 1 |
| retract with the old content supplied | additive counts (either form) | version space; violation bit | CEV-08 |
| retract by event identity only | ID→content map (linear) | histogram | CEV-08 witness 2, CEV-09 |
| score a concept formed later | observation-level counts (fixed alphabet) | hypothesis-level counts | CEV-08 witness 3 |
| use a feature of past observations that was never encoded | raw records or re-acquisition | every summary | working note §5 |
| acquire a new skill without losing an old one in shared weights | retained examples + joint refitting (replay), with transient violations | gradient/Jacobian projection, accept-only guards | geometry |
| find later constructions sooner | bank of own accepted constructions + mined pieces (+ reserved prior-free budget) | pieces alone on a shifted family | library |

### 7.3 What the group says about each pillar

- **P1 learning = running.** Every step that both runs and learns must leave a state closed under the declared future operations (L1). The policy that decides how evidence may move the state is itself state (geometry continuation pairs). A library used at run time and a proposal order are the same object (L9) — so "running" already *is* the place where experience acts.
- **P2 weights = graded beliefs.** The only construction in this group where a "weight" is literally a belief is `(N,d)`: an integer evidence balance, additive, invertible, order-free, with affine readouts for all hypotheses. The studies add three sharp conditions: keep it as a count (L3), keep both sides if confidence/validity matter (CEV-09 control 1), keep the losers' evidence (CEV-08).
- **P3 concepts formed when needed.** Two tested formation rules exist: promote-on-conflict (paper 07) and mine-recurring-composites (library). Two tested obstacles: later-formed concepts need observation-level evidence (L4); a fixed shared substrate has no room (L6).
- **P4 imagine and measure reality.** Tested pieces of a reality measure: separate acceptance evidence with abstention (library; 0 false accepts but noiseless), try-then-commit against retained evidence (geometry; safe but sterile), exact replay as oracle (CEV-08/09). Tested failure: consistency and growth as a reality proxy (L11). Not tested anywhere in this group: a *graded* reality score under noise, or imagination (constructing unobserved cases) at all.
- **P5 the way of constructing improves and transfers.** Shown once, one step deep: history → library → different search order; strong positive transfer in-family, strong negative transfer off-family (L10); partial remedy by reserved budget. Not shown: a construction method that is revised by its own outcomes, or that transfers across families.

### 7.4 Requirements extracted (each traceable)

1. **Declare the future-operation contract per memory cell and derive the state from it** (L1; CEV-09 §7). Test: count the distinguishable classes under the contract on an exhaustive small universe and compare with the implementation's reachable states.
2. **Beliefs are additive evidence counts, never thresholded or winner-only** (L3; 360,000 and 1,338,336 failures).
3. **Two-sided beliefs (for, against) wherever confidence, indicator queries or validity checks are promised** (CEV-09 negative control 1).
4. **Evidence is indexed by observations, not by the current concept set**, or the loss is declared and a re-acquisition path exists (L4).
5. **Corrections either carry the old content or are backed by an identity ledger whose linear cost is budgeted**; a separate provenance check exists (L5).
6. **Old skill is retained structurally (immutable, versioned composites; separate capacity), not by projecting updates of a shared dense vector** (L6, L9; geometry vs library).
7. **Retention contracts are stated on correct targets, per skill, with an explicit budget and a fixed reference** (L8).
8. **Learning state = function + retained evidence + update/protection/proposal policy; all three persist** (L2, L12). Test: continuation pairs and fresh-process resume.
9. **Every acquired composite pays for its expansion, the library is priced, and a prior-free construction budget is reserved** (L9, L10; portfolio).
10. **Any change to the construction method is evaluated by acquirability on related *and* shifted probes with paired gained/lost counts and three controls: prior-free, whole-construction memory, same-priority fixed interpreter** (library design).
11. **Structure formation needs a reality measure other than training consistency** (L11).
12. **Report honestly: reused lives are not independent worlds; post-hoc controls are labelled; equal samples are not equal compute** (the practice of all four studies; worth keeping as the project's evidence discipline).

### 7.5 Reusable mechanisms (with locations)

| Mechanism | Where | Reuse in the new core |
|---|---|---|
| Violation-count ledger with subtract/add | `correction_state.py:38-69` | invertible belief update |
| Observation histogram, readout `c_h=Σ_x n[x,1−h(x)]` | `correction_state.py:72-89` | two-sided belief store |
| Signed balance `(N,d)`, `c_h=(N−Σd)/2+Σh(x)d_x`, updates `±` | `query_state.py:22-32, 73-83` | weight = graded belief; affine readout for hypotheses formed later |
| Exhaustive differential test against an independent replay; exact class counting by signatures | `correction_state.py:136-207`; `query_state.py:52-95` | acceptance test for any state/update design |
| Separate-process continuation probe | `correction_state.py:232-262`; `geometry.py` `--resume-input`; `verify_library.py` | persistence test |
| RPNI-style merge with promote-on-conflict | `research/living_learning_state_2026_09_20/state_merging.py:62-200` | concept formation when needed |
| Reachability diagnostic `range(BP)`, `Pg≠0`, SVD row basis | `geometry.py:65-72`; `MATHEMATICS_AND_LITERATURE.md` §3 | "no room here → allocate structure" signal |
| Try-then-commit guard with fixed reference and halving | `geometry.py:164-178`; `posthoc_controls.py:36-48` | safety layer for imagined updates |
| Continuation-pair intervention (clone, change only the policy) | `geometry.py:235-248` | test that policy is part of state |
| Budget-charged BFS with signature pruning and one-shot separate acceptance | `library_learning.py:44-98` | construction with abstention |
| Library mining `support·(len−1)−len`, top-4 | `library_learning.py:101-119` | concept = recurring own composite |
| Library-erase intervention, expanded-proposal equality control | `verify_library.py:113-137`; `library_learning.py:288` | causal role test; "no new expressibility" control |
| Half-budget portfolio with prior-free priority | `portfolio_control.py:13-32` | bounded negative transfer |
| Definition/claim split with versioned definitions | library `MATHEMATICS_AND_LITERATURE.md` §6 | correcting concepts without breaking dependants |
| Local protocol before results; post-hoc notes kept separate; SHA-256 preservation manifests | all four studies | evidence discipline |

### 7.6 What this group never built (gaps the new core would be first to fill inside this programme)

- All exact states rely on a **given finite alphabet** (`X`, `H`, binary symbols, four primitives). No study forms the *observation distinctions* themselves.
- Evidence ledgers (studies 1–2), on-demand structure (paper 07), protection geometry (study 4) and construction priors (study 5) were each studied **alone**; no system combines an additive two-sided belief store with on-demand concepts and a priced, self-revising construction order.
- No study has **graded evidence under noise** deciding whether a new structure is real; no study has imagination (evaluating constructed, unobserved cases).
- No study lets the **construction method be changed by its own outcomes**, and none shows transfer across families.
- No study bounds **memory over an unbounded life** (archives, banks and ID ledgers all grow).

---

## 8. Analyst tiny checks (not part of the owner's studies)

Script and output: `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/prior/_checks_g3/g3_tiny_checks.py`, `…/_checks_g3/g3_tiny_checks.json` (standard library, 0.2 s; imports none of the research code).

| Check | Result |
|---|---|
| A. CEV-08 three witnesses re-implemented | version spaces `{0},{0}` → `{5},{3}`; reversed-order histories same counts, after-edit `{1,5}` vs `{2,6}`; old counts `[0,3],[0,3]`, new hypothesis 0 vs 2 — all as recorded |
| B. CEV-09 classes for `m=3,N=4` by brute force with independent code | 85 / 126 / 1,296 = formulas; signed identity holds on all 1,296 histories |
| C. **Extension**: classes needed for the "which hypotheses are best" query under all append sequences up to length 4, `m=2`, `N=0..4` | 1/4/9/16/25 = number of distinct `d`; sign-only would give 1/4/9/8/9 → the integer balance is necessary, its sign is not enough |
| D. Geometry witnesses with exact fractions | `Jδ=0`, output 3/4, loss 1/32, curved path 1; `ab+c`: 0/0/1/32/0; `b²` example loss 1/2 → 9/32 with zero first derivative; mean 5/8 → 17/32 while `L_A` 1/8 → 9/32; bound `R−R₀ ≤ 2√(R₀D)+D` holds on 2,000 random cases |
| E. Library reachability witness | `aaa` is 8th in base order (13 primitive applications if each candidate is run from scratch) and 2nd with the macro (3); `b` is 3rd vs 4th; mining scores 11/8/6/1 reproduce the recorded life-0 selection |
| F. Result files re-read | every number quoted in §1–§6 was re-read from the JSON files; additional values computed from them: 4-anchor paired counts 8/12 and 7/12; life-7 libraries; edition identity of papers 07–09 |

---

## 9. Established / Refuted / Open — across the group

| Established (with the evidence the study itself gives) | Refuted (counterexample or contrary measurement) | Open (named by the studies unless marked) |
|---|---|---|
| A state is sufficient only relative to the operations permitted later; closure condition `c(D)=c(D') ⟹ c(uD)=c(uD')` (CEV-08, exact witness + algebra) | Keeping only the current winner or the complete current version space is enough to correct the past (360,000/360,000 non-empty cases fail) | Which future operations an open-world learner should protect — the learner must learn the contract (CEV-09) |
| Additive violation counts with truthful old content equal full replay: 0/1,507,050; 0/18,378; fresh process exact | A contradicted/not bit per candidate with subtract/add (1,338,336 false admissions; ablation, not a theorem) | What a bounded memory should keep across unknown representation expansions and corrections (CEV-08) |
| Six-bin histogram is equally exact and supports hypotheses added later (closed domain) | An event ID alone identifies what to correct | Minimal state for weaker queries such as "best candidate" (CEV-09); [analyst] `d` is needed under appends for the all-Boolean class |
| `(N,d)` is exactly the class of all Boolean-risk answers; closed under append and content-carrying edit; 0 mismatches over 61,575 histories, 712,632 edits, 358,020 appends | Counters over the current hypotheses can score a hypothesis added later | Bounds under approximate answers, bounded correction horizon, re-acquirable records, open input language (CEV-09) |
| Class counts `K_risk=Θ(N^m)`, `K_hist=Θ(N^{2m−1})`, `K_ID=(2m)^N`; 21/21 cells match | "Unknown future hypotheses ⇒ store the full histogram / everything" | Who certifies that a correction is right; automatic discovery of wrong experience (CEV-08, paper 07) |
| ID-only correction needs `⌈N log₂(2m)⌉` bits (exact contract) | The histogram is minimal for every contract; the histogram validates identity; `(N,d)` validates that an old record existed | An efficient encoder achieving the class-count bounds; update/query operation costs (not measured) |
| First-order reachability: `r∈range(BP)`; descent iff `Pg≠0` (linear algebra at a point) | A numeric range guard is a provenance check | Finding a feasible update *path* with bounded compute when a feasible endpoint exists (geometry) |
| In the 2→8→1 family, current-tangent protection leaves 0.0108 % of the new gradient (rank 31.5/33); new MSE .19627 → .19627; worse than replay in 12/12 lives at 16 and 64 anchors | Non-conflict with the old loss gradient preserves old skill; Jacobian-orthogonal finite steps preserve old outputs exactly | Whether new capacity / local structure removes the freeze-or-forget dilemma ([analyst]: untested in this group) |
| Fixed tangent ≠ current tangent once the representation moves (drift .0389 at 16 anchors) | More protected anchors ⇒ better acquisition | Acquiring a genuinely new internal feature/operation and using it in another problem (geometry §9) |
| Preserving initial outputs ≠ preserving the correct skill: `R−R₀ ≤ 2√(R₀D)+D`; teacher outside the output budget 12/12 | A try-and-accept guard turns retention into acquisition (117–122 of 128 steps rejected) | Assigning a delayed outcome to the internal change that caused it (geometry §9) |
| Same weights, different protection policy ⇒ different future acquisition (36/36 pairs; new MSE .196 vs .007, old .008 → .047) | Identical predictions ⇒ identical future learning | Separating "fits this family" from "acquires better in a new family" (geometry §9) |
| Feasible endpoint without the tested path being feasible (10/12 vs 12/12) — sample paths only | Preserving outputs = preserving correctness; a mean budget protects every skill | With what evidence experience may change the candidate generator / update rule, and whether that helps in a new family (library) |
| Acyclic macro libraries leave the computable function set unchanged; they change budgeted reachability | An acquired library adds expressive power | Acquisition of observational distinctions absent from the initial language (library) |
| Mined pieces help on related tasks: 86 vs 25 (512), 94 vs 73 (2,048); +62/−1 | More correct experience ⇒ more future acquisition (life 7: 478 steps → failure at 2,048; shifted 6→12: +1/−11) | A selector/library that is itself revised by outcomes; bounded bank over an unbounded life (library) |
| The same pieces hurt on a shifted motif family: 14 vs 69 (2,048); +0/−55 in 12/12 lives | A learned prior is never worse than a random prior (37 vs 14); a bigger fixed library is better (6/96, 23/96) | A graded reality measure for formed structure under noise ([analyst]; noise case only shown to fail in paper 07) |
| Library knowledge can be carried in proposal order: 1,152/1,152 identical (by construction) | Extracting shared sub-pieces is necessary beyond whole-program reuse (95 vs 94 — not supported) | Telling wrong labels from a richer truth when the candidate family empties (paper 07) |
| Erasing the library changes 192 future searches and no old answer (144 programs): the pieces are causal for future search only | Reserved prior-free budget fully removes negative transfer (40 < 69; post-hoc) | Carrying the in-episode state across a change of the learned graph (paper 07) |
| Terminal labels of short words yield transition graphs exact on all non-empty words: 16/16, 4/4, 4/4; a 4-state rule is acquired without a state bound | Training consistency + structure growth ⇒ validity (36–42 states, 0/8, Brier .287109) | A general computational principle of living learning — every report: "the main problem is still open" |
| Execution state (≈3 states) ≠ relearning memory (256 words, 2,094 symbols) | Small running state ⇒ small learning memory; a fixed last-k window carries long dependencies | |
