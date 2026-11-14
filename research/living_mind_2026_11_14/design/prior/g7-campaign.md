# G7 — The living-learning campaign of 2026-09-24 (wave 1, wave 2), paper 10, and continuity notes on CEV-11 and KA

Reader: group G7. Date of reading: 2026-11-14. Repository read-only; nothing in it was changed.

## 0. What exists, what does not, and how to read this file

| Named source | Exists? | What it holds |
|---|---|---|
| `docs/research/living_learning_campaign_2026_09_24/` | yes | `REPORT_TR.md` (87 lines), `PROTOCOL_TR.md`, `RESULTS_TR.md` (48 rows), `CATALOG_TR.md` (200 entries, 2,520 lines), `SOURCES_TR.md` |
| `research/living_learning_campaign_2026_09_24/` | yes | four engines `memory.py` `representation.py` `update.py` `environment.py`, orchestrator `campaign.py`, four catalog JSONs, `registry.json`, four engine protocols, `CONTRACT.md`, `results/{freeze,preflight,summary,verification}.json`, `results/{discovery,confirmation}/*.jsonl` |
| `research/living_learning_campaign_wave2_2026_09_24/` | yes, but **contract only** | `CONTRACT.md` (42 lines) and `results/preservation_baseline.json`. No engine, no catalog, no protocol, no outcome |
| `docs/research/living_learning_campaign_wave2_2026_09_24/` | **does not exist** | wave 2 has no report |
| `docs/publications/papers/10-living-learning-campaign-wave1.md` | yes | front matter + the wave-1 report verbatim (see §8) |
| `docs/research/living_learning_paradigm_2026_11_13/REPORT_TR.md` | yes | CEV-11 "Paradigma" report (1,397 lines) |
| `research/concept_address_2026_11_13/results/confirm/tables.md` | yes | KA confirmatory decisions. There is **no** `docs/research` folder and no report for KA; its `README.md` is stale (§10) |

Three kinds of statement are kept apart below:

- **STUDY** — backed by the study's own files (path and line given).
- **MY CHECK** — a tiny recomputation I made with the study's frozen code on its own confirmation seeds. It is not a
  result of the study. Script: `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/prior/_g7work/g7_checks.py`
  (7 s on one core, writes nothing; the folder holds byte copies of the four engines).
- **MY READING** — an inference; it was not tested by anyone.

Replay note (MY CHECK 0): seed 1000 of all four engines reproduces the stored confirmation record exactly on Linux /
Python 3.13.18 / NumPy 2.5.4; the originals were made on Windows / Python 3.14.3 / NumPy 2.4.2
(`research/living_learning_campaign_2026_09_24/results/freeze.json`).

---

## 1. Campaign wave 1 as a whole ("Tur 23")

**(1) Question.** In which conditions does each of four known mechanism families (keeping/forgetting/retracting
evidence, acquiring features, adapting the update coefficient, choosing actions that produce the data) change a running
system's later computation for the better, and where does it fail — measured over the whole life against a strong rival?

**(2) Mechanism / state.** Four *separate* engines, no integration (`PROTOCOL_TR.md:9`). None is neural. Every learner is
a handful of sufficient statistics updated in closed form once per observation; there are no epochs and no
backpropagation. Each has exact snapshot/restore.

| Engine | State that is stored | Update rule | Where |
|---|---|---|---|
| MEM | per key (16 keys) two counts; optionally raw records, per-key windows, or a table event-id → (key, label) | add / decay / evict; revision = subtract old, add new | `memory.py:29-143` |
| REP | active feature set S, RLS weights w and matrix P, residual–feature trace, archive of all (x, y) | recursive least squares; error-directed feature installation with replay | `representation.py:26-128` |
| UPD | one estimate m, six log-weights over a supplied gain family, six one-step proposals | exponentially weighted choice of the gain from one-step prediction error | `update.py:44-115` |
| ENV | per action count and reward sum; optionally the last 64 (t, action, reward) | sample mean, ε-greedy, UCB bonus, windowed UCB | `environment.py:17-70` |

**(3) Design.** 200 catalogued propositions in ten families of 20 (MEM, CON, RES, REP, XFR, LAN, UPD, CRD, CTX, ENV);
48 executable conditional comparisons (12 per engine), 152 `queued` (`campaign.py:63-65`, `PROTOCOL_TR.md:9`).
Discovery seeds 0–23, confirmation seeds 1000–1047 (`campaign.py:25`); 4 engines × 72 seeds = 288 engine-lifetimes,
48 × 72 = 3,456 paired outcomes. 17 source/catalog/protocol files hashed before any run (`results/freeze.json`).
Statistics (`campaign.py:230-267`): d = control − candidate per seed (lower metric is better); exact one-sided paired
sign test in both directions; Holm over all 96 tests; **supported** iff Holm p ≤ 0.05 **and** mean d ≥ the
pre-declared practical margin **and** ≥ ⌈2·48/3⌉ = 32 seeds positive; **contradicted** symmetrically; otherwise
**inconclusive**. Error, virtual operations, memory cells and acquisition are separate ledgers, never merged
(`PROTOCOL_TR.md:21`). Hidden truth is read only by the scorer; any oracle is diagnostic only (`CONTRACT.md:14`).

**(4) Results** (`REPORT_TR.md:11-17`, `RESULTS_TR.md:10-57`, `results/summary.json`):

| Engine | run | supported | contradicted | inconclusive |
|---|---:|---:|---:|---:|
| MEM | 12 | 12 | 0 | 0 |
| REP | 12 | 8 | 4 | 0 |
| UPD | 12 | 5 | 3 | 4 |
| ENV | 12 | 5 | 2 | 5 |
| total | 48 | 30 | 9 | 9 |

Audit: 28,811 preflight software checks (56 statistics + 26,520 MEM + 266 REP + 83 UPD + 1,886 ENV,
`results/preflight.json`); all 288 lifetimes recomputed and identical (`results/verification.json`); again on
2026-10-13 (`docs/publications/evidence/research_validation_2026_10_13.json`, `exact_deterministic_reruns: 288`).
Discovery and confirmation agree in sign on 47 of 48 claims; the exception is ENV-11 (discovery mean −0.0015, 8 wins /
15 losses; confirmation +0.0226, "supported").

**(5) Law / invariant claimed.** None of general scope. The report says so in bold: "Genel living-learning ilkesi
bulunmuş veya kanıtlanmış değildir" (`REPORT_TR.md:7`). What it does state are conditional regularities, each tied to a
named generator and rival (§2–§5), plus one methodological principle: *a learned part that beats a simple rival has not
thereby beaten the strong ready-made procedure* (`REPORT_TR.md:65`). The only things **proved** (algebraically, by
independent reference in the selfchecks) are four identities:
1. identified-event retraction on additive counts ≡ full recount (`memory.py:319-344`, and `answer_sha256` equal in all 72 MEM lifetimes, `memory.py:263-266`);
2. installing a feature by replaying history ≡ the ridge solution that had the feature from the start (`representation.py:244-252`);
3. the gain learner with meta-rate 0 ≡ a fixed gain at the family mean (`update.py:322-328`), and its trace depends on observations only, never on hidden truth (`update.py:350-356`);
4. the sign-test tail and Holm adjustment are exact (exhaustive for n ≤ 8, `campaign.py:84-110`).

Proved in scope, statistically: each of the 30 + 9 decided comparisons, read literally — "method A has a lower whole-life
metric than method B on generator G" — and nothing wider. Only suggested by the report's wording, and not carried by
its own data: that the gain learner's wins come from its learning (§4 (6)); that an acquired coefficient "can be
transferred" (title of report section 3; UPD11 is inconclusive); that selectivity explains the two wins over the full
representation (§3 (6)); that incremental revision is faster in real time (the counter is virtual, `REPORT_TR.md:35`).

**(6) Limits.** Admitted: four unconnected engines; representation language, labels, actions, candidate update family
and metrics are all supplied by the researcher; no language, perception, long life, open-ended task; local freeze is not
a public preregistration; the sign test tests direction frequency, not the size of the gain; catalog entries overlap
and are not 200 mechanisms (`REPORT_TR.md:9,23,27,71`, `SOURCES_TR.md:17-19`).
Limits I see: (a) the p-values measure how deterministic a synthetic generator is — MEM11's difference is the same
number, 604.25, in all 72 seeds, so its "48/0/0, p = 3.41e-13" carries no statistical information; (b) two engines
lack the *mechanism-off* comparator as a registered claim, and in both the headline attribution does not survive it
(§3 (6), §4 (6)); (c) memory budgets are not matched between rivals (34 vs 146 vs 853 cells in MEM); (d) every
learner holds at most about 1,000 numbers; (e) 152 of 200 propositions — including every one about consolidation,
resources, transfer, procedure language, credit and context — were never run.

**(7) For a core where learning and running are one process.** *Mechanism to reuse:* all four engines already are
"one process" learners — predict, observe, closed-form belief update, exactly resumable — so the campaign is a toy-scale
existence proof of the loop itself. *Requirement:* score every claim the way this campaign does (predict before
observe, whole life, acquisition included, cost ledgers separate, frozen code, untouched confirmation seeds, three-part
rule) **plus** a registered mechanism-off control. *Warning:* nothing here shows a *general* advantage of learning
while running: the complete ready-made representation won wherever a product mattered (§3), and the adaptive update
lost in the one regime where the preset constant was the right one (§4).

---

## 2. Engine MEM — keeping, forgetting, retracting (`memory.py`, `MEMORY_PROTOCOL.md`)

**(1) Question.** Given 16 supplied keys with noisy binary labels, which way of keeping evidence gives the lowest
whole-life error when the world is stationary, changes abruptly or key by key, arrives unevenly, or is corrected later
by identified revisions?

**(2) Mechanism.** Symbolic table of statistics. State per key k: counts c_k = (c_k0, c_k1) (`memory.py:34`).
Readout: majority vote, 0.5 on a tie (`memory.py:21-22, 43-55`) — a hard decision, never a probability.
Updates on an observation (k, y):
- `count`: c_ky += 1 (`:124`). `last`: keep only the latest label (`:87-90`).
- `decay`: every count × 0.99, then c_ky += 1 (`:91-95`).
- `window`: last 6 labels per key; the evicted label is subtracted (`:116-123`).
- `fifo` / `reservoir`: 96 raw records in total; FIFO evicts the oldest, reservoir replaces a uniformly drawn slot (`:96-115`).
- revision (id, k, y′), `revision_delta`: (k, y_old) ← E[id]; E[id] ← (k, y′); c_k,y_old −= 1; c_k,y′ += 1 (`:62-72`).
  `revision_append`: c_k,y′ += 1 without removing the old contribution (`:73-75`). `revision_ignore`: nothing.
  `revision_rebuild`: stores only E and recounts every event at every query (`:47-54`).

**(3) Design.** Streams of 1,024 steps (512 for revisions), each step = answer a uniformly drawn key, then receive one
event (`:162-207`). Label noise 0.25. Regimes: `stable`; `clustered` (56-step blocks per key); `rare` (96% of events on
4 keys); `abrupt` (8 keys flip at t = 512); `staggered` (key k flips at 256 + 32k); `revisions` (first 256 observations
wrong with probability 0.85 on keys 0–5 and 0.10 elsewhere, then 256 identified, correct revisions in random order,
`:199`). Metric: whole-life squared loss against the noiseless label, in [0, 1]; MEM11: virtual operations per event.
Margin 0.005 (5.0 for MEM11).

**(4) Results** (confirmation, n = 48; `RESULTS_TR.md:22-33`). All twelve **supported**.

| Id | Regime | Candidate / control | Loss cand. | Loss control | + / = / − |
|---|---|---|---:|---:|---|
| MEM02 | stable | count / reservoir | 0.02299 | 0.09239 | 48/0/0 |
| MEM03 | stable | count / last | 0.02299 | 0.24871 | 48/0/0 |
| MEM07 | stable | count / decay | 0.02299 | 0.05120 | 47/0/1 |
| MEM12 | stable | count / window | 0.02299 | 0.07859 | 48/0/0 |
| MEM04 | abrupt | window / count | 0.10151 | 0.23188 | 48/0/0 |
| MEM06 | abrupt | window / reservoir | 0.10151 | 0.22886 | 48/0/0 |
| MEM05 | staggered | decay / count | 0.13372 | 0.38730 | 48/0/0 |
| MEM01 | clustered | reservoir / fifo | 0.12314 | 0.19676 | 45/0/3 |
| MEM08 | rare | window / fifo | 0.08373 | 0.17454 | 48/0/0 |
| MEM09 | revisions | delta / ignore | 0.25800 | 0.36852 | 48/0/0 |
| MEM10 | revisions | delta / append | 0.25800 | 0.30690 | 48/0/0 |
| MEM11 | revisions | delta / rebuild (ops per event) | 15 | 619.25 | 48/0/0 (constant) |

Read from the confirmation records (`results/confirmation/memory.jsonl`, diagnostics): peak memory in numeric slots —
count 34, last 18, decay 34, window 146, fifo 226, reservoir 853 (627 of them RNG state), `revision_delta` **802**,
`revision_rebuild` 770, append/ignore 34. Total virtual operations per life — delta 7,680 vs rebuild 317,056 (41.3×,
`REPORT_TR.md:35`). After the revisions (second half of the life) the loss is 0.1548 (delta = rebuild), 0.2526
(append), 0.3759 (ignore); the first half is identical for all four (0.3611).

**(5) Invariants.**
- *Exact (proved):* with additive counts and an identified, stored original event, incremental retraction equals a
  full recount — same answer sequence in every lifetime, at 15 instead of 619.25 virtual operations per event. Scope:
  the revision may change a label, never its key (`memory.py:64-65`); majority readout; the cost ratio is for this
  workload and this declared counter, not processor time.
- *Conditional (measured):* "ne sınırsız birikim ne de sürekli unutma bütün koşulların kazananıdır"
  (`REPORT_TR.md:33`): cumulative counts are best when the world is stationary (0.0230 against 0.0512–0.2487 for every
  forgetting rule), and the worst rule measured when it changes (0.2319 against 0.1015 for the window; 0.3873 against
  0.1337 for decay). The counterexample that forbids a single rule is this pair of rank reversals on the same code.
- *Conditional:* a correction treated as fresh independent evidence is worse than retract-and-add (0.3069 vs 0.2580).

**(6) Limits.** Admitted: keys, labels and event identities are supplied; revisions are externally given and correct —
"sistem doğruluğu kendisi keşfetmedi" (`REPORT_TR.md:35`); memory counts fields, not bytes; the growing bit length of
counters is not charged; all of it is known statistics (`MEMORY_PROTOCOL.md:54-73`).
Limits I see: the numbers are corollaries of binomial majority arithmetic (the `last` rule scores 0.2487 ≈ the noise
rate 0.25; a six-label window in a stationary world must lose 0.0706 by hand calculation and scores 0.071 in the second
half), so the experiment confirms arithmetic rather than discovering anything; rivals are not memory-matched; the
readout is a hard vote, so nothing is learned about graded confidence or calibration; whether a compressed side record
could support retraction (MEM15) was not run — exact correctability here costs 3 cells per correctable event (802 vs
34 slots).

**(7) For the new core.** *Mechanism to reuse:* beliefs as additive sufficient statistics with an identity-indexed
ledger, so that "correct" = subtract the old contribution and add the new (`memory.py:62-72`). *Requirement:* if
revision is among the operations the core must support, the readout state alone is not enough — two histories with the
same counts need different counts after the same revision (the falsifier written into MEM15, untested but evident).
*Requirement:* the time-scale of a belief has to be a quantity the system infers; no constant is right (MEM13,
"learned change detector", is queued). *Warning:* stating the price — auditable revision grows with the number of
events that may still be revised.

---

## 3. Engine REP — acquiring features from the error trace (`representation.py`, `REPRESENTATION_PROTOCOL.md`)

**(1) Question.** If experience changes not only the weights but which features are used for the next prediction, when
does that help against (a) never adding features, (b) adding random ones, (c) having all of them from the start?

**(2) Mechanism.** Statistics, closed form. Feature language supplied: φ(x) = [1, x₁..x₄, six pairwise products]
(`:22-23`). Recursive least squares on the active set S (`:53-63`):

    v = φ_S(x);  u = P v;  d = λ + vᵀu;  w ← w + u (y − vᵀw)/d;  P ← (P − u uᵀ/d)/λ      P₀ = I, w₀ = 0, λ ∈ {1, 0.97}

w is the mean and P, up to the noise variance, the covariance of the ridge-regression posterior (unit prior; with
λ < 1 an exponentially forgetting one): literally weights as a graded belief, updated in one step per observation.
Need signal (`:72-88`): s ← 0.98 s + r·φ_products(x), m ← 0.98 m + 1 with r the prediction residual; every
24 labels, if |S| < 8, the inactive product with the largest |s_j|/m is installed when that exceeds the fixed threshold
0.08. Installation with replay (`:91-102`): P ← I, w ← 0 and the whole stored history is run again through RLS;
ablation `zero_history` (`:103-108`): old block kept, new coordinate gets an independent unit prior. Random control:
same trigger, index from seed and time (`:85-87`). Every adaptive learner archives all (x, y) (`:72`).

**(3) Design.** 192 samples per stream, x ∈ {±1}⁴, predict then label (`:131-158`). Regimes: `interaction` (one hidden
product), `additive` (none), `noisy` (none, noise sd 1.2), `rare` (two coordinates +1 with probability 0.08),
`covariate` (x₂ = x₁ with probability 0.95 in the first half), `onset` (product appears at t = 96), `switch` (product
changes identity and sign at t = 96). Rivals: `linear` (5 features), `full` (all 11 from the start), `random`.
Metric: whole-life MSE in [0, 144], margin 0.01; REP11: virtual operations, margin 1,000.

**(4) Results** (confirmation, n = 48; `RESULTS_TR.md:34-45`).

| Id | Regime | Candidate / control | cand. | control | + / = / − | Decision |
|---|---|---|---:|---:|---|---|
| REP01 | interaction | selective / linear | 0.22858 | 1.16095 | 48/0/0 | supported |
| REP04 | interaction | selective / random | 0.22858 | 0.76208 | 44/4/0 | supported |
| REP08 | rare | selective / linear | 0.13799 | 0.49712 | 47/0/1 | supported |
| REP12 | interaction | replay / zero-history | 0.22858 | 0.26545 | 48/0/0 | supported |
| REP05 | switch | λ = 0.97 / λ = 1 | 0.46871 | 0.75144 | 48/0/0 | supported |
| REP03 | additive | selective / full | 0.03471 | 0.04817 | 48/0/0 | supported |
| REP07 | noisy | selective / full | 1.66571 | 1.78165 | 47/0/1 | supported |
| REP11 | interaction | selective / full, **operations** | 72,271 | 160,662 | 48/0/0 | supported |
| REP02 | interaction | selective / full | 0.22858 | 0.09332 | 0/0/48 | **contradicted** |
| REP06 | switch | selective / full (λ = 0.97) | 0.46871 | 0.30091 | 0/0/48 | **contradicted** |
| REP09 | covariate | selective / full | 0.25003 | 0.09055 | 2/0/46 | **contradicted** |
| REP10 | onset | selective / full (λ = 0.97) | 0.17651 | 0.15591 | 11/0/37 | **contradicted** |

Memory: selective 1,009 numeric cells on average (archive included), full 132 (`REPORT_TR.md:43`, formula
`representation.py:169`).

Read from the confirmation records (`results/confirmation/representation.jsonl`): in `interaction` the second-half MSE
is 0.02398 (selective) against 0.02496 (full); the whole-life deficit of REP02 is therefore paid entirely before and
around the installation (first half 0.433 against 0.162).

MY CHECK 1 (which product was installed; truth recomputed from the generator):

| Regime, method | lives | true product installed | true product installed *first* | spurious installs |
|---|---:|---:|---:|---:|
| interaction, selective | 48 | 48 | 48 (at t = 24, the first decision) | 1 |
| interaction, random control | 48 | 26 | 4 | 96 |
| interaction, zero-history | 48 | 48 | 48 | 58 |
| rare, selective | 48 | 47 | 40 | 22 |
| covariate, selective | 48 | 48 | 37 | 20 |
| onset, selective (λ = 0.97) | 48 | 48 | 24 | 79 |
| switch, selective (both λ) | 48 | 48 (both products) | 48 | 47–48 |
| additive, selective | 48 | — (nothing to find) | — | 39, in 33 lives |
| noisy, selective | 48 | — (nothing to find) | — | **144 = all three free slots in every life** |

MY CHECK 2 (a learner that never installs anything, same seeds):

| Regime | linear | selective | full |
|---|---:|---:|---:|
| additive | **0.03431** | 0.03471 | 0.04817 |
| noisy | **1.62138** | 1.66571 | 1.78165 |
| interaction | 1.16095 | 0.22858 | **0.09332** |
| covariate | 1.15090 | 0.25003 | **0.09055** |
| onset | 0.56582 | 0.17651 | **0.15591** |
| switch | 1.18830 | 0.46871 | **0.30091** |

**(5) Invariants.**
- *Exact (proved):* replay installation reproduces the posterior of a learner that had the feature all along
  (`representation.py:244-252`).
- *Measured:* a distinction selected from the error trace changes later predictions (REP01, REP04); carrying the
  observed past into the new feature beats starting it empty (REP12).
- *Measured, negative:* in this closed 11-feature language the complete representation beats selective growth wherever
  a product matters (REP02, 06, 09, 10); fewer active features meant less arithmetic (72,271 vs 160,662) but **more**
  memory (1,009 vs 132) — "az etkin özellik, az toplam bellek anlamına gelmedi" (`REPORT_TR.md:43`).

**(6) Limits.** Admitted: "Hazır özellik bankası dışındaki bir işlem keşfedilmedi" (`REPORT_TR.md:45`); thresholds
fixed in advance; full ridge is a strong same-information control, not the best possible; REP02 and REP11 are two views
of one run (`REPRESENTATION_PROTOCOL.md:8, 98-99`).
Limits I see:
1. REP03 and REP07 are "supported", but the never-installing learner does better there (MY CHECK 2; `additive`:
   better in 31 lives, identical in the 15 lives where nothing was installed, worse in 2; `noisy`: better in 44, worse
   in 4). The selective learner wins over `full` only because its cap is 8 features instead of 11; every installation
   it made in those worlds was a false positive. On all six regimes the selective learner lies **between** the two
   fixed choices and never beats both. (MY READING: its real merit is as a hedge — it avoids the 1.16 of `linear` and
   most of the excess of `full` — which the study neither claims nor tests.)
2. The need threshold (0.08, fixed) is not calibrated against chance: with nothing to find it fills every free slot in
   48 of 48 noisy lives, and half of the `onset` lives install a wrong product before the real one exists.
3. There is no removal of a wrong feature (REP18 queued), so a false positive is permanent and uses the budget.
4. The decision cadence (every 24 labels) is an unexamined constant that sets the whole-life penalty of REP02.
5. The posterior matrix P is computed but never used as uncertainty (not for the need signal, not for anything).

**(7) For the new core.** *Mechanisms to reuse:* RLS as the one-step belief update with optional forgetting; the
residual × candidate-feature trace as "a concept is needed here" (right product first in 48/48 clean lives); replay as
the way to give a new concept its past. *Requirements:* (i) the need signal must be a test against the null — e.g.
residual standardised by the predictive variance that P already provides — and (ii) concepts must be retractable.
*Requirement on state:* a concept formed later can only be evidenced from what was kept **before** it existed. In a
closed language the cheap alternative to the archive is the cross-moment table of all candidates, which is the full
learner's information (REP14, queued); in an open language, where candidates cannot be listed, only stored experience
can do it (MY READING — REP19 "same answer, different update" is the queued test). *Warning:* on-demand concept
formation was **not** shown to beat a complete ready-made representation; its case has to be made where the candidate
space is too large to instantiate (REP16, LAN03 — queued).

---

## 4. Engine UPD — adapting the update coefficient, and carrying it elsewhere (`update.py`, `UPDATE_PROTOCOL.md`)

**(1) Question.** Can experience change not just the estimate but *how much the next observation changes the
estimate*, and does that acquired coefficient help on a new task when the estimate itself is reset?

**(2) Mechanism.** Statistics; a belief over a supplied family of update rules. Gains G = {0.01, 0.03, 0.1, 0.3, 0.7, 1}
(`:15`). Fixed rule: m ← m + a (y − m) (`:56`). Candidate `gain_meta` (`:80-88`), on each new y:

    ℓ_a ← 0.97·ℓ_a − 4·(z_a − y)²/16          (z_a: the proposal made for gain a one step earlier)        (:32)
    w = softmax(ℓ);   a_t = Σ_a w_a·a                                                                    (:82)
    z_a ← m + a·(y − m)   for every a in G     (six counterfactual next states from the shared state)   (:85)
    m ← m + a_t·(y − m)                                                                                  (:86)

Strong control `expert_bank` (`:94-115`): six independent states, one per gain, the same weight law on their own
errors, prediction Σ w_a m_a. Transfer (`:205-232`): run `gain_meta` on 256 drift observations, then reset m to 0 and
run a *fixed* gain equal to the acquired one on 256 target observations.

**(3) Design.** 384-step streams (`:136-173`); hidden mean μ in [−0.9, 0.9], observation = μ + uniform noise.
Regimes: `stationary_low` (noise ±0.08), `stationary_noisy` (±0.8), `drift`, `jumps` (every 32 steps), `reversals`
(every 24), `mixed` (48-step phases). Fixed controls: a = 0.05 in stationary regimes, 0.2 elsewhere (`:240`), set in
advance. Metric: mean ((prediction − μ)/3)², in [0, 1], margin 0.0005 (`:187-188`). A retrospective best-fixed-gain
oracle is recorded as diagnostic only (`:245-249`).

**(4) Results** (confirmation, n = 48; `RESULTS_TR.md:46-57`).

| Id | Regime | Candidate / control | cand. | control | d (margin 0.0005) | + / = / − | Decision |
|---|---|---|---:|---:|---:|---|---|
| UPD01 | stationary_low | meta / fixed 0.05 | 0.000178 | 0.000789 | +0.000610 | 41/0/7 | supported |
| UPD04 | jumps | meta / fixed 0.2 | 0.003043 | 0.004875 | +0.001833 | 48/0/0 | supported |
| UPD05 | reversals | meta / fixed 0.2 | 0.001136 | 0.002098 | +0.000962 | 48/0/0 | supported |
| UPD06 | mixed | meta / fixed 0.2 | 0.004465 | 0.005893 | +0.001429 | 48/0/0 | supported |
| UPD08 | drift | meta / bank | 0.000526 | 0.001280 | +0.000754 | 48/0/0 | supported |
| UPD02 | stationary_noisy | meta / fixed 0.05 | 0.003244 | 0.001214 | −0.002031 | 0/0/48 | **contradicted** |
| UPD07 | stationary_noisy | meta / bank | 0.003244 | 0.002058 | −0.001186 | 1/0/47 | **contradicted** |
| UPD12 | transfer, mismatched target | carried gain / fixed 0.2 | 0.002890 | 0.001847 | −0.001044 | 0/0/48 | **contradicted** |
| UPD03 | drift | meta / fixed 0.2 | 0.000526 | 0.000742 | +0.000216 | 48/0/0 | inconclusive (below margin) |
| UPD09 | jumps | meta / bank | 0.003043 | 0.003372 | +0.000329 | 46/0/2 | inconclusive (below margin) |
| UPD10 | mixed | meta / bank | 0.004465 | 0.004536 | +0.000071 | 30/0/18 | inconclusive |
| UPD11 | transfer, matched target | carried gain / fixed 0.2 | 0.000541 | 0.000763 | +0.000222 | 48/0/0 | inconclusive (below margin) |

Transfer detail (`REPORT_TR.md:53-55`, records): target phase only, mismatched: 0.005236 against 0.002921; acquisition
cost 28,444 against 1,282 virtual operations; the carried gain is 0.350 on average (0.341–0.359 over the 48 lives).
Diagnostic oracle in the records (best single fixed gain in hindsight, never shown to a learner): 0.3 in all 48 drift
lives; 0.7 in 47 of 48 jump lives and all 48 reversal lives; 0.03, 0.1 or 0.01 in 22, 17 and 9 of the noisy stationary
lives. The adaptive learner's final gain was 0.349, 0.367, 0.362 and 0.211 in those regimes.

MY CHECK 3 — the same learner with its learning switched off (meta-rate 0, which the study proves equal to a fixed
gain of mean(G) = 0.3567 but never scores):

| Regime | meta | learning off | fixed control | bank | what learning adds | final weights over G |
|---|---:|---:|---:|---:|---:|---|
| stationary_low | 0.000178 | 0.000180 | 0.000789 | 0.000300 | +0.000002 | 0.167 0.167 0.167 0.167 0.166 0.165 |
| drift | 0.000526 | 0.000529 | 0.000742 | 0.001280 | +0.000003 | 0.168 0.169 0.169 0.170 0.166 0.159 |
| reversals | 0.001136 | 0.001150 | 0.002098 | 0.001618 | +0.000014 | 0.162 0.163 0.165 0.170 0.172 0.168 |
| jumps | 0.003043 | 0.003173 | 0.004875 | 0.003372 | +0.000131 | 0.160 0.161 0.164 0.171 0.176 0.170 |
| mixed | 0.004465 | 0.004862 | 0.005893 | 0.004536 | +0.000397 | 0.130 0.133 0.144 0.174 0.213 0.208 |
| stationary_noisy | 0.003244 | 0.005258 | 0.001214 | 0.002058 | +0.002014 | 0.215 0.216 0.217 0.199 0.108 0.046 |

Transfer, target phase: carried "learned" gain 0.000538 — unlearned prior mean 0.000537 — fixed 0.2: 0.000753
(matched); 0.005236 — 0.005350 — 0.002921 (mismatched).

**(5) Invariants.**
- *Exact (proved):* meta-rate 0 ⇒ fixed mean gain; the learner's trace is a function of the observations alone.
- *Measured, negative:* an acquired update coefficient that persists is not thereby useful in a new environment —
  "Öğrenilmiş bir güncelleme katsayısının kalıcı olması, her yeni ortamda yararlı olacağını garanti etmiyor"
  (`REPORT_TR.md:55`); counterexample UPD12. And "'Kendi güncellemesini değiştirmek her zaman daha iyidir' sonucuna
  ulaşılamaz" (`REPORT_TR.md:51`); counterexamples UPD02, UPD07.
- *Not established:* positive transfer (UPD11 is below the margin).

**(6) Limits.** Admitted: six supplied gains, no new operator language or optimizer; regimes are not models of
anything real (`UPDATE_PROTOCOL.md:7, 15`).
Limits I see: the section title "öğrenilen katsayı aktarılabiliyor" (`REPORT_TR.md:47`) is not carried by the data.
(i) In the four regimes where `gain_meta` beat the fixed control, switching the learning off loses +0.000002 to
+0.000397 — below the study's own margin every time; the win is the family's prior mean (0.357) being a better constant
than the control's (0.05 or 0.2). (ii) The belief weights end practically uniform in the low-noise regimes; the
adaptation is inert there because the evidence term (z − y)²/4 is in raw units and is tiny when the noise is small —
a scale defect, not a property of meta-learning. It always moves in the right direction (44–48 of 48 lives better than
"off") and moves materially only where errors are large. (iii) What UPD11/UPD12 "transfer" is in effect the prior
mean: 0.350 carried against 0.357 unlearned, with indistinguishable target error. (iv) The "strong" bank is itself
weaker than the plain fixed gain under drift (0.001280 against 0.000742).

**(7) For the new core.** *Mechanism to reuse:* the loop itself — from the present state generate alternative next
states, let the next real observation score them, and move the belief over *how to update* (`update.py:80-88`); it is
the smallest working instance of "imagine alternatives and measure them against what actually happens".
*Requirements:* the evidence must be scale-free (standardise the error by an estimated noise scale) or the meta-level
is asleep exactly when the world is clean; every claim that "the way of learning improved" needs the learning-off
control as a registered rival. *Warning for transfer:* the only transfer evidence in my group is one null (UPD11) and
one harm (UPD12); a carried way-of-updating must be probed on arrival, tagged with the conditions it was acquired in,
and be forgettable (XFR07, UPD14 — queued). *Open question:* whether an acquired update rule transfers at all (UPD13,
UPD15, LAN05).

---

## 5. Engine ENV — actions that produce the data (`environment.py`, `ENV_PROTOCOL.md`)

**(1) Question.** When the learner's own choices decide what it observes next, which selection rule keeps whole-life
regret low in stationary and reversing two-action worlds?

**(2) Mechanism.** Statistics: per action a count n_a and a reward sum s_a (`:20`). `greedy`: pull each once, then
argmax s_a/n_a (`:35-38, 48`). `epsilon`: deviate with probability 0.10 (`:40-42`). `ucb`: s_a/n_a + 0.5·√(ln(max(2,
n₀+n₁))/n_a) (`:43-46`). `window_ucb`: the same on counts rebuilt at every step from the last 64 (t, action, reward)
(`:28-34, 54-60`).

**(3) Design.** 512 steps, two Bernoulli arms (0.8/0.2 or 0.55/0.45, common jitter ±0.02); best arm fixed, flipped once
at 256, or every 128 (`:73-87`). All rules face the same table of potential outcomes but see only their own choices —
explicitly *not* a same-data comparison (`ENV_PROTOCOL.md:9-12`). Metric: mean pseudo-regret, margin 0.01.

**(4) Results** (confirmation, n = 48; `RESULTS_TR.md:10-21`).

| Id | Regime | Candidate / control | cand. | control | + / = / − | Decision |
|---|---|---|---:|---:|---|---|
| ENV-07 | single reversal | ucb / greedy | 0.07275 | 0.30127 | 48/0/0 | supported |
| ENV-10 | recurring reversal | ucb / greedy | 0.11353 | 0.30112 | 48/0/0 | supported |
| ENV-12 | recurring reversal | window_ucb / ucb | 0.06638 | 0.11353 | 39/1/8 | supported |
| ENV-08 | single reversal | epsilon / greedy | 0.28313 | 0.30127 | 34/3/11 | supported (Holm p 0.0218) |
| ENV-11 | recurring reversal | epsilon / greedy | 0.27856 | 0.30112 | 33/5/10 | supported (Holm p 0.0164) |
| ENV-03 | stationary, large gap | window_ucb / ucb | 0.02710 | 0.00562 | 0/0/48 | **contradicted** |
| ENV-06 | stationary, small gap | window_ucb / ucb | 0.03108 | 0.01842 | 9/0/39 | **contradicted** |
| ENV-01 | stationary, large gap | ucb / greedy | 0.00562 | 0.05127 | **4/0/44** | inconclusive |
| ENV-02 | stationary, large gap | epsilon / greedy | 0.03335 | 0.05127 | 4/0/44 | inconclusive |
| ENV-04 | stationary, small gap | ucb / greedy | 0.01842 | 0.04190 | 20/0/28 | inconclusive |
| ENV-05 | stationary, small gap | epsilon / greedy | 0.02051 | 0.04190 | 20/0/28 | inconclusive |
| ENV-09 | single reversal | window_ucb / ucb | 0.03655 | 0.07275 | 31/2/15 | inconclusive |

Read from the records (`results/confirmation/environment.jsonl`): in the stationary large-gap world greedy has median
regret 0.00117 and maximum 0.59883, with 4 of 48 lives above 0.1; UCB's maximum is 0.01055. Greedy's less-used arm is
pulled 1.2 times per life on average (once, and then never again). Virtual operations: greedy 7,157, UCB 12,257,
windowed UCB about 167,264.

**(5) Invariant.** Measured: *mean gain and gain in most lives are different properties* — UCB is nine times better
than greedy on the mean and worse in 44 of 48 lives, because a few greedy lives lock onto the wrong action for good
(`REPORT_TR.md:61`). The pre-registered rule called this inconclusive and the criterion was not changed afterwards.
Conditional: optimism beats exploitation under reversal; recent-only statistics help under repeated reversal and hurt
when nothing changes.

**(6) Limits.** Admitted: two ready actions, reward meaning, selector and window given; no internal state, new goal or
new action language is discovered; classical UCB guarantees are not claimed for c = 0.5 or the 64-step window
(`ENV_PROTOCOL.md:14-18, 32-33`). Limits I see: ENV-08 and ENV-11 are small effects whose discovery run was null or
negative (12/12 at +0.0070; 8 wins / 15 losses at −0.0015) — fragile; the largest-gap finding rests on four lives.

**(7) For the new core.** *Requirement:* a system that acts on its beliefs must carry their uncertainty into the
choice, because acting on the current best belief removes the evidence that could correct it (one pull of the other
arm, then never again). *Requirement on evaluation:* declare in advance which feature of the distribution over lives
counts — mean, share of lives, or the worst lives; they disagreed here. *Warning:* the same time-scale dilemma as MEM
and UPD appears a third time (window helps under change, hurts under stationarity).

---

## 6. The 200-entry catalog — what was written down but not run (`CATALOG_TR.md`, `registry.json`)

**(1) Question.** Which propositions about learning while living are worth testing, each with a prediction, an
intervention and a falsifier, across ten families?

**(2) Mechanism.** None: 152 entries are text only (`status: queued`); only the 48 above have code. Entries carry
`prediction_tr`, `intervention_tr`, `falsifier_tr`, `prior_art` (`campaign.py:52-56`).

**(3) Design.** 20 per family. Run: MEM01–12, REP01–12, UPD01–12, ENV-01–12. Queued: MEM13–20, REP13–20, UPD13–20,
ENV-13–20 and **all** of CON (consolidation), RES (resources), XFR (transfer), LAN (procedure language), CRD (credit),
CTX (context).

**(4) Results.** None for the 152. "152 önerme henüz sınanmadı" (`REPORT_TR.md:9`).

**(5) Laws.** None claimed. Several queued entries are *statements of the earlier law* "the family of future operations
determines the state that must be kept" in new settings, each with a collision-style falsifier, and all untested here:
REP19 (two representations equal today, the one that kept cross-moments learns a new feature better), CTX20 and UPD18
(same present answers, different later learning), CON14 (a summary that preserves old answers can destroy
updatability), CRD10 (a credit summary sufficient for old outcome types fails for a new type), MEM15, CTX07.

**(6) Limits.** Admitted: entries overlap and are neither 200 mechanisms nor 200 findings (`SOURCES_TR.md:19`). Limit I
see: the six families never run are exactly the ones the new initiative is about.

**(7) For the new core — the queued propositions that bear on each design intention** (all OPEN):

| Design intention | Catalog entries (id — content) |
|---|---|
| Concepts formed when needed | REP16 — a learner confined to pairwise products has a permanent error floor on a pure third-order target; LAN03 — extend the operation language on a counterexample; LAN04 — expand a newly acquired operation back into the old language to tell a mere abbreviation from a real gain in what can be represented; MEM20 — a state language that grows when new label values appear; REP18 — replace a wrong feature, REP20 — fast provisional store and slow permanent installation |
| Contexts (the owner's "bağlam kümeleri") | CTX01 — same input, different right behaviour needs history; CTX02 — more clusters is not learning; CTX03/CTX04 — merge wrongly split, split wrongly merged contexts; CTX05/XFR09 — the same event belongs to different partitions for different tasks; CTX08 — committing early makes repair expensive; CTX12 — context discovery is not name matching; CTX14 — a local exception must not rewrite the whole category; MEM14, MEM19 |
| Cross links and transfer (the owner's "çapraz bağlar") | XFR01 — mapping without a ready identity bridge; XFR06 — keep confidence in the bridge apart from confidence in the source knowledge; XFR07 — probe compatibility with a few target samples before transferring; XFR12 — round-trip consistency; XFR13 — break-even of building a bridge; XFR18 — separate a mapping error from a knowledge error; XFR20 — composed bridges accumulate error; CTX17, CTX18 |
| The way of constructing improves and transfers | LAN05 — acquire the update procedure from a task sequence; LAN20 — bound an acquired updater by counterexamples on old knowledge; UPD13–15, UPD20; LAN06 — local patch versus resynthesis; LAN10 — which state to keep when the procedure changes; LAN11 — keep the grounds of an acquired procedure; LAN17 — a search prior learned in one family can hurt in another |
| Imagining, and measuring how real a construction is | CON13 — self-generated replay loses the sparse regions of the real evidence; CON06 — replaying a wrong record strengthens the wrong belief when no independent correction arrives; MEM16 — correlated repeats must not count as independent evidence; CRD12 — trust set by agreement with the current model shuts out correct dissent; ENV-18 — interventions aimed at model disagreement; ENV-19 — rewarding prediction error wastes effort on irreducible noise; LAN14 — proved versus merely observed equivalence; LAN15 — a fixed verifier cannot vouch for a language that outgrew it; CON17 — a small protected witness set that catches a bad consolidation |
| Delayed credit | CRD02 — totals cannot identify individual causes (an identifiability counterexample); CRD05 — right label bound to the wrong event; CRD06, CRD16 — keep a set of candidate attributions; CRD20 — accurate credit is not the same as useful action |
| Cost over a life | RES01 — break-even of an acquired accelerator; RES06 — charge the measuring; RES11 — a fixed number of counters is not fixed memory; RES17 — relearn versus keep; CTX19, XFR13, ENV-20 |

The report's own priority list for the next wave: REP16/LAN03, REP19/CTX20, CTX03/CTX04, CRD05/CRD10, CON02/CON10,
XFR01 (`REPORT_TR.md:73`).

---

## 7. Campaign wave 2 ("Round 24") — a contract that was never executed

**(1) Question.** Would have been: partial operational tests of queued propositions with three new engines — *context*,
*credit*, *capacity* (`research/living_learning_campaign_wave2_2026_09_24/CONTRACT.md:6`).

**(2) Mechanism.** None exists. The contract fixes only interfaces: `hypotheses()` with ids CTX2-01 / CRD2-01 / CAP2-01,
`source_ids` pointing into the frozen 200-entry registry and a `coverage_note` ("partial operational test, not blanket
completion"), `run(seed)`, `selfcheck()` (`CONTRACT.md:12-25`). "Capacity" is not defined in the contract; by the
wave-1 priority list it would be REP19/CTX20, same answers today and different learning later (MY READING).

**(3) Design (planned).** 8 comparisons per engine, 24 in all; discovery seeds 2000–2031, confirmation 3000–3063;
288 engine-lifetimes, 2,304 paired outcomes; 48 one-sided sign tests under Holm; same three-part rule; "no global
all-time FWER claim" (`CONTRACT.md:29-35`). New relative to wave 1: the selfcheck must include an **information
collision counterexample** where appropriate (`:24-25`); equal observations across methods unless information is the
tested intervention (`:21-22`); "Teacher-provided context IDs, mappings, operator banks, clocks and revisions must be
explicit, not called discovered" (`:37-38`).

**(4) Results.** None. The folder holds `CONTRACT.md` and a preservation baseline of 1,149 file hashes. Confirmed by
`docs/publications/evidence/research_validation_2026_10_13.json` (`"performance_results_present": false`) and
`docs/publications/releases/research-2026.10.13-v4.md:9` ("Wave 2 is a protocol only").

**(5) Law.** None. **(6) Limits.** It is an intention, not a study; the programme moved on to the 2026-11-13 studies
without it.

**(7) For the new core.** *Open question, inherited whole:* context, credit and capacity were chosen as the next three
engines and none was built. *Mechanism to reuse:* the contract's rule that anything the teacher supplies (context ids,
mappings, operator banks, clocks, revisions) is named as supplied — a necessary honesty rule for a core that claims to
form its own concepts.

---

## 8. Paper 10 — `docs/publications/papers/10-living-learning-campaign-wave1.md` (CEV-2026-10, 2026-10-13)

**(1) Question.** The wave-1 question, published as a preprint. **(2)–(4)** Identical to §1–§5: after a front matter
(Turkish and English abstract, an interpretation paragraph, publication and AI-assistance statements) the body is the
research report verbatim — I compared them line by line with links normalised: 50 of 50 non-empty lines equal except
the title line. Numbers: 200 entries, 48 comparisons, 3,456 paired outcomes, 30 / 9 / 9.

**(5) Law.** Its interpretation paragraph (`:19`): "The campaign maps conditional mechanisms and negative controls
rather than discovering a general computational principle. Experience can change later computation in these supplied
languages, but a stronger pre-supplied representation or a shifted environment can remove the apparent advantage."

**(6) Limits.** Not peer reviewed (`:7`); "local synthetic evidence, not external peer review, biological validation,
AGI evidence or proof of novelty" (`:21`). Limit I see: the abstract's sentence that strong controls show learned
representation *and update behaviour* to be "conditional and harmful under shift" is accurate, but the body keeps the
section title about a transferable learned coefficient that §4 (6) shows to be unsupported.

**(7) For the new core.** Nothing beyond §1–§5; it fixes the public wording: no general principle was found.

---

## 9. Continuity (brief): CEV-11 "Paradigma" — `docs/research/living_learning_paradigm_2026_11_13/REPORT_TR.md`

**(1) Question.** When a fact is "lived" in one inflected sentence and written into the network, how far does the write
reach across the Turkish case forms of the entity, their apostrophe-less spellings, a left context, and other entities
that share the name's last root token (`:112-115`)?

**(2) Mechanism.** Neural, the project's own decoder (3,445,824 parameters, D = 192, 6 layers, `:232-234`). The write
is a rank-one term added by a hook to one block's FFN output projection: M = (v* − W k̄)(C0⁻¹k̄)ᵀ / (k̄ᵀC0⁻¹k̄), v* =
W k̄ + δ, δ found by an external optimiser (NLL of the new object + KL on another relation), norm-capped at 8× the
median output norm in the selected configuration (`:279-300, 502`). C0 = uncentred second moment of the FFN inputs over ≈ 20,000 positions. A float64
ledger W = W0 + B(λC0 + A)⁻¹ gives exact insert / correct / retract (`:142, 413`). Similarity of a query key to the
written key: τ(k′, k̄) = k′ᵀC0⁻¹k̄ / k̄ᵀC0⁻¹k̄ (`:381-382`). The report's own scope line: "This is lifelong knowledge
editing inside the network, not learning from raw experience" (`:80-81`).

**(3) Design.** Unit = one independently pretrained base; 2 discovery worlds, 8 confirmatory (1000–1007); 70 edits per
world; arms STD, ROOT, LAST, PARA-ORTH, FT-L, FT-ALL, NORM, BASE; exact sign-flip test on margin-shifted effects, Holm
(`:386-437`).

**(4) Results** (`:519-523, 642-645`).

| Hyp. | Effect | Margin | Decision |
|---|---|---:|---|
| H1 formal − apostrophe-less success, same cases | +24.3 pts [21.0, 27.5] | 15 | supported (p 1/256, Holm 0.0117) |
| H2 that gap, fused-token forms − identical-key forms | +69.7 pts [62.5, 76.9] | 30 | supported |
| H3 leak to last-root-token sharers − random entities | +33.4 pts [29.1, 37.7] | 5 | supported |
| S1 form-averaged keys | +3.1 pts [1.5, 4.7] | 10 | **refuted** |
| S2 prefix averaging and left context | +12.3 pts [6.4, 18.2], p 0.18 | 10 | inconclusive |
| S3 entity-joint ledger over a 100-operation life | — | 10 | **not run** |
| S4 key similarity to neighbours falls with depth | −0.086 [−0.134, −0.039] (it rose) | 0.15 | **refuted** |

Pooled detail: identical-key forms 84.6% formal / 84.1% informal; fused forms 83.4% / **14.9%** (`:574-575`); leak to
sharers 34.0% against 1.0% to random entities (`:828`). Discovery lives (not confirmatory): the joint ledger collapsed
at every λ — end-of-life accuracy 2.5–5.7% where writing nothing would give 13.3%, answers changed on 63–70 of 70
random probes (`:715-745`); diagnosis: about 52 capped writes drive ΔW·h to ≈ 2× the layer's natural output (`:751-759`).
Fine-tuning arms generalised better to fused forms (55.3% / 71.7%) with leaks ≤ 0.4%, but were weaker after a left
context (54.5% / 76.1% against 83.8%) — descriptive only (`:884-890`).

**(5) Invariant.** A reading, not a proof: *the write fires where the query's key at the written site is close to the
written key*; reach equals key identity (gap −2.3…+2.0 pts when the key is bit-identical, 58–81 pts when the tokenizer
fuses root and suffix) and leakage equals key sharing. Holds only within a layer; no intervention on the fused token
was made (`:966-981`). By construction and unit-tested: exact retraction, order independence (V4), and the earlier
programme's counterexample that a state holding only W cannot tell two histories apart while fact records can (V5,
`:1004-1005`).

**(6) Limits.** Admitted (`:1152-1198`): the STD configuration failed both pre-registered leak constraints in discovery
("not installable under constraints"), δ sat at the cap in 92% of edits and leaked 26.2% into another relation of the
same entity; tiny model, synthetic templated world, one tokenizer, edits applied one at a time; S3 left unrun while its
probable negative outcome was known; one S4 value seen early.

**(7) For the new core.** *Warning:* a surface-token position is not a concept address — it misses the same entity
when the surface segmentation changes and hits other entities that share a token. *Warning:* exact bookkeeping makes a
write reversible, not harmless; capacity of the written site is a separate constraint. *Mechanism to reuse:* τ as a
measurement made **before** writing of where a write will act (AUC 0.86 for success on the informal form, 0.69 within
fused forms, `:970-972`).

---

## 10. Continuity (brief): KA "Kavram Adresi" — `research/concept_address_2026_11_13/results/confirm/tables.md`

Context from `research/concept_address_2026_11_13/PROTOCOL.md` and `results/confirm/decisions.json`. State of the
record: `PROTOCOL.md:3` says "Status: FROZEN 2026-11-14" and §12 records the freeze; confirmatory results were
committed (`f2a208f`); but `README.md:6-8` still says "NOT FROZEN … there is no result", there is no `AMENDMENTS.md`
and no written report. Treat as fresh and unreviewed.

**(1) Question.** Which code inside the frozen decoder is a real concept address for writing a fact — reaching all case
forms and fused spellings without touching neighbours — and does a calibrated, optimiser-free memory built on it beat
sequential fine-tuning (`PROTOCOL.md:39-44`)?

**(2) Mechanism.** Neural base, statistical memory outside the weights. Address: a read-only code at the prediction
position, whitened, scored against stored keys by a two-covariance PLDA log-likelihood ratio and calibrated by logistic
regression on 60 never-written calibration entities (`PROTOCOL.md:152-166`). Read with an explicit "no stored episode"
slot (`:168-175`):

    ρ_j(q) = e^{s(q,k_j)} / ( e^{θ_N} + Σ_l e^{s(q,k_l)} ),   θ_N = log N − logit P_t,   P_t = 0.5,   ρ_0 = 1 − Σ_j ρ_j

Values are analytic and relative to the base model: the smallest step along the margin gradient that makes the new
object win by m* = 6 (`:179-181`), injected before the output norm in proportion to ρ_j. Episodes live in a ledger
outside the module tree; insert / correct / retract are exact and the base weights never change (`:190-196`).

**(3) Design.** 8 confirmatory worlds (CEV-11's bases); stream of 400 inserted facts, checkpoints 25–400; sequential
fine-tuning streams of the first 100 facts at CEV-11's configurations; unit = world; sign-flip, Holm; primary family
{KA-H2, KA-H3}, secondary {S2} after the discovery screens (`PROTOCOL.md:505-520`).

**(4) Results** (`tables.md`, `decisions.json`).

| Hyp. | Statistic | Decision |
|---|---|---|
| KA-H1 the attention-written stream is the better *write* address | on both discovery worlds every gated mid-stack address and the standard editor were infeasible under the leak constraints (F* = 0); fine-tuning references reached 0.39–0.64 | **discovery-stage negative**, not in the confirmatory family |
| KA-H2 memory vs best sequential fine-tuning, success on fused forms at N = 100 | d = 0.5823 [0.5212, 0.6434], p 0.0078 (Holm 0.0156); memory 0.834 vs best FT 0.252 | **"memory better"** |
| KA-H3 tuning-free calibrated scope | ECE 0.0148 (≤ 0.05); growth of leak to never-written entities from N = 25 to 400: 0.59 pt (≤ 1 pt); with the threshold frozen at θ₂₅ it grows 3.7 pts | **supported** (p 0.0193) |
| S2 prediction before writing (part i only) | item-level AUC 0.967 [0.940, 0.994] (> 0.85) | **supported** |
| S1a, S1b | — | not tested (the attention code failed its screen) |
| Gaussian false-firing law | observed ≤ law + 1 pt at every checkpoint on 8 of 8 worlds | holds (count rule, outside Holm) |

Guards and descriptives (`decisions.json`): leak to never-written entities — memory 0.35%, sequential fine-tuning
60.5%; lived form still produced at N = 100 — memory 100%, FT-ALL 45.2%, FT-L 32.5%; change in bits per byte — memory
+0.0001, FT-ALL +0.0075, FT-L +0.0597; scorer consistency 0 disagreements in 438.

**(5) Law.** The scope rule: with a calibrated log-likelihood ratio and the Bayes prior P_t/N per stored episode, the
threshold θ_N = log N − logit P_t keeps false firing roughly independent of the number of stored episodes; predicted
count N·Φ̄((θ_N + m)/√(2m)) under a Gaussian null (`PROTOCOL.md:202-212`). The Markov form is N-invariant but admitted
to be "vacuous at the Leak_U ≤ 1% scale" (`:204-206`). Tested on 8 worlds at N ≤ 400 only.

**(6) Limits.** Admitted: 3.4M-parameter model, synthetic templated world, one tokenizer, N = 100 (`:492-503`). Limits
I see: the fine-tuning rivals use configurations chosen for single edits and no replay or regularisation, so "memory
better" is against sequential fine-tuning *as configured*; the memory is an external episodic store added at the
output, not a change of the network's own weights; the mid-stack address question (KA-H1) ended negative; stale README
and no report.

**(7) For the new core.** *Mechanisms to reuse:* (i) separate **address** ("is this about a concept I hold?", a
calibrated likelihood ratio) from **value** ("what do I add?"); (ii) the **null slot** whose threshold grows with log N
— the same reading that says "none of my concepts applies" is the natural trigger for forming a new one; (iii)
calibration fitted on material that is never written, then checked on what is written. *Warning:* updating shared
dense weights fact after fact destroyed 60% of unrelated answers within 100 facts at this scale.

---

## 11. What my group implies for the new core

**The five design intentions against what these studies actually hold**

| Intention of the new core | What my group established | What it did not |
|---|---|---|
| Learning and running are one process | All four engines are exactly that at toy scale: predict, observe, one closed-form update, resumable to the bit (§1) | Any such loop over a representation the system built itself; anything beyond ~1,000 numbers of state |
| Weights are graded beliefs | RLS weights with their covariance, counts, and log-weights over update rules are beliefs in the literal sense (§3, §2, §4) | The belief's uncertainty was never *used*: hard-vote readout in MEM, P unused in REP; calibration measured only in KA (ECE 0.0148, §10) |
| Concepts are formed when needed | An error-trace need signal finds the right product first in 48/48 clean lives; a new feature can be given its past exactly (§3) | A need signal that stays silent when nothing is there (48/48 noisy lives filled with spurious products); removal; any win over the complete ready-made representation; any concept outside a supplied bank |
| The system imagines and measures how real its constructions are | Two small instances only: one-step counterfactual proposals scored by the next real observation (§4); predicting before a write where it will act (τ, ρ: §9, §10) | Everything about self-generated experience — no executed test; the risks are catalog entries (§6) |
| The way of constructing improves and transfers | Nothing positive. One below-margin result and one demonstrated harm (UPD11, UPD12), and the "learned" coefficient equals its prior in effect (§4) | Whether an acquired way of updating can transfer at all |

**Requirements**

1. **Whole-life, predict-before-observe scoring with separate ledgers** (error, operations, memory, acquisition). A
   learner that is equal after acquiring (0.0240 vs 0.0250) still lost the life (0.2286 vs 0.0933) — §3.
2. **An identity-indexed evidence ledger wherever correction or retraction is among the future operations**; the
   readout state alone cannot do it (0.3069 vs 0.2580), and the price is explicit (802 vs 34 slots) — §2. KA and CEV-11
   use the same construction outside and inside a network.
3. **The time-scale of a belief is itself a belief.** Three engines independently show a rank reversal between
   stationary and changing worlds (count vs window/decay; UCB vs windowed UCB; best fixed gain 0.03 vs 0.7) — §2, §4, §5.
4. **A concept-need signal calibrated against chance, and retractable concepts.** The fixed threshold picked the right
   product first in 48/48 clean lives and filled every free slot with noise in 48/48 noisy lives — §3.
5. **Experience kept in a form richer than the current concept set**, so that a concept formed later can be given its
   past (replay beat zero-history, 0.2286 vs 0.2654) — with its memory cost counted (1,009 vs 132 cells) — §3.
6. **Scale-free evidence at the meta level and a registered learning-off control.** Where the gain learner "won", its
   learning contributed +0.000002 to +0.000397 and its belief stayed uniform — §4.
7. **Transfer on probation.** Carried ways of updating are probed on arrival, tagged with their origin, and
   forgettable; the only evidence is one null and one harm — §4.
8. **Uncertainty drives action, and the success criterion over lives is declared beforehand** (mean, majority, or
   tail) — §5.
9. **Rivals at both ends:** the strongest ready-made same-information procedure *and* the version with the new
   mechanism switched off. The selective learner sat between `linear` and `full` on every regime — §3.
10. **An address that is a concept, not a surface token, with an explicit "none of these" outcome** whose threshold
    scales with the number of stored concepts — §9, §10.

**Warnings**

- Beating a weak rival is not beating the ready-made procedure (REP02, 06, 09, 10).
- Fewer active parts did not mean less memory (REP11 against the memory ledger).
- An acquired update coefficient harms in the wrong regime (UPD02, 07, 12).
- Acting on the current best belief produces rare, permanent lock-in (4 of 48 lives, regret up to 0.599).
- Tiny p-values on synthetic seeds measure determinism; one of the 48 is a constant (MEM11).
- Token-anchored writes miss fused forms (14.9% vs 83.4%) and leak to token sharers (34.0% vs 1.0%); about 52 capped
  writes into one layer collapsed it; sequential fine-tuning of shared weights leaked 60.5% within 100 facts.
- Exact bookkeeping guarantees reversibility, not usefulness (the collapsed ledger had retraction residual 0.0).
- Nothing self-generated was ever used as experience in these studies; every risk of imagination (CON06, CON13,
  CRD12, ENV-19, MEM16) is still only a catalog entry.

**Mechanisms to reuse**

| Mechanism | Where | Evidence |
|---|---|---|
| One-step closed-form belief update as the unit of "learning = running", exactly resumable | counts `memory.py:124`; RLS `representation.py:53-63`; belief over update rules `update.py:29-41, 80-88`; counts + optimism `environment.py:43-46` | 28,811 preflight checks incl. serialized continuation |
| Identity-indexed delta revision | `memory.py:62-72` | ≡ rebuild in 72/72 lives, 15 vs 619.25 operations per event |
| Residual × candidate trace as need signal | `representation.py:72-88` | true product first in 48/48 (MY CHECK 1); needs a null calibration |
| Replay installation of a new feature | `representation.py:91-102` | ≡ fixed-basis solution (`:244-252`); REP12 |
| Counterfactual one-step proposals scored by the next real observation | `update.py:80-88` | works where errors are large; needs normalisation |
| Three-part decision rule, frozen hashes, untouched confirmation seeds, selfcheck contract (independent reference, mechanism ablation, continuation; wave 2 adds a collision counterexample) | `campaign.py:230-267`, both `CONTRACT.md` | separated 30 / 9 / 9, including 48/48-but-below-margin (UPD03, UPD11) and mean-against-majority (ENV-01) |
| Calibrated likelihood-ratio address with a null slot, θ_N = log N − logit P_t | KA `PROTOCOL.md:152-175` | ECE 0.0148; leak growth 0.59 pt from N = 25 to 400 |
| Predict, before the value is written, where a write will act (key similarity τ in CEV-11; read weight ρ in KA) | CEV-11 `REPORT_TR.md:381-382`; KA `PROTOCOL.md:274-282` | τ: AUC 0.86 (exploratory); ρ: AUC 0.967 (S2, supported) |

**Open questions** (none answered in my group)

1. Does on-demand concept formation win when the candidate language is open and cannot be pre-instantiated (REP16, LAN03, LAN04)?
2. Which extra state does which future operation require — same answers today, different learning tomorrow (REP19, CTX20, UPD18, CON14, CRD10; wave 2's "capacity")?
3. Context discovery, split and merge without names (CTX03, CTX04, CTX12, MEM14, MEM19).
4. Delayed credit bound to the right past, and its identifiability limits (CRD02, CRD05, CRD10).
5. Consolidation under interruption and the risks of replay (CON02, CON06, CON10, CON13, CON14, CON17).
6. Transfer without an identity bridge (XFR01, XFR06, XFR07, XFR12, XFR20).
7. Can an acquired update rule transfer at all (UPD13–15, LAN05, LAN20)?
8. Imagination and its reality measure: no executed test exists.
9. Does any of this survive leaving closed-form statistics of at most ~1,000 numbers?

---

## 12. Established / Refuted / Open across the group

"Established" and "Refuted" mean: decided by the study's own pre-registered rule on its confirmation units, inside the
stated synthetic scope. Rows marked † rest on MY CHECK, not on the study.

| # | Statement | Status | Evidence |
|---|---|---|---|
| 1 | Retracting an identified observation on additive counts equals a full recount, at 15 vs 619.25 virtual operations per event | **Established** (exact) | MEM11; `memory.py:263-266, 319-344` |
| 2 | A correction added as new evidence is worse than retract-and-add (0.3069 vs 0.2580); ignoring it is worse still (0.3685) | **Established** | MEM09, MEM10 |
| 3 | Cumulative counts are the best rule in a stationary noisy world (0.0230 vs 0.0512–0.2487) | **Established** | MEM02, 03, 07, 12 |
| 4 | Recent-only or decayed evidence is better when keys change (0.1015 vs 0.2319; 0.1337 vs 0.3873) — so no fixed retention rule wins everywhere | **Established** | MEM04, 05, 06 |
| 5 | A per-key quota or uniform sample covers uneven arrival better than global FIFO | **Established** | MEM01, MEM08 |
| 6 | A feature chosen from the error trace changes later predictions (0.2286 vs 1.1610 linear, vs 0.7621 random choice) | **Established** | REP01, 04, 08 |
| 7 | Giving a newly added feature its observed past beats starting it empty (0.2286 vs 0.2654) | **Established** | REP12 |
| 8 | Forgetting helps when the relation switches (0.4687 vs 0.7514) | **Established** | REP05 |
| 9 | Selective growth uses fewer operations than the full representation (72,271 vs 160,662) but more memory (1,009 vs 132 cells) | **Established** | REP11, `REPORT_TR.md:43` |
| 10 | Under reward reversal an uncertainty bonus beats exploitation (0.0728 vs 0.3013; 0.1135 vs 0.3011); a recent window helps under repeated reversal (0.0664 vs 0.1135) | **Established** | ENV-07, 10, 12 |
| 10b | A fixed 10% random deviation beats exploitation under reversal, by little (0.2831 vs 0.3013; 0.2786 vs 0.3011) | **Established** by the rule, fragile (discovery runs null and negative) | ENV-08, ENV-11 |
| 11 | The gain learner beats the pre-set fixed gain in low-noise stationary, jump, reversal and mixed regimes, and the filter bank under drift | **Established as stated**; attribution to learning **not** established † | UPD01, 04, 05, 06, 08; MY CHECK 3 |
| 12 | (CEV-11) A one-sentence rank-one write loses 24.3 pts without the apostrophe, the loss sits in fused-token forms (+69.7 pts), and it leaks to last-root-token sharers (+33.4 pts) | **Established** (8 worlds) | H1–H3 |
| 13 | (KA) A calibrated null-slot episodic memory beats sequential fine-tuning as configured at N = 100 (d = 0.58); its scope stays calibrated (ECE 0.0148) and its false firing does not grow with N (0.59 pt) | **Established** (8 worlds; no report yet) | KA-H2, KA-H3, S2 |
| 14 | Selective feature growth beats the complete ready-made representation when an interaction matters (also after a switch, under input shift, with late onset) | **Refuted** | REP02, 06, 09, 10 |
| 15 | Adapting the gain beats a fixed gain, or a bank of filters, in a stationary high-noise world | **Refuted** | UPD02, UPD07 |
| 16 | A gain acquired under drift helps on a mismatched target (target phase 0.005236 vs 0.002921) | **Refuted** | UPD12 |
| 17 | Recent-window statistics help a stationary bandit | **Refuted** | ENV-03, ENV-06 |
| 18 | Selective acquisition is the reason for the wins over `full` in additive and noisy worlds | **Refuted** † (never-installing learner is better: 0.0343 vs 0.0347; 1.6214 vs 1.6657) | MY CHECK 2 |
| 19 | The transferred "learned" gain carries something learned | **Refuted** † (target error 0.000538 vs 0.000537 with the unlearned prior mean) | MY CHECK 3 |
| 20 | (CEV-11) Form-averaged keys close the informal gap (+3.1 pts, margin 10); key similarity to neighbours falls with depth (it rose) | **Refuted** | S1, S4 |
| 21 | (KA) A gated mid-stack address, including the attention-written stream, is a usable write address under the leak constraints | **Refuted at discovery** (not confirmatory) | KA-H1 screen, `PROTOCOL.md:513-519` |
| 22 | (CEV-11) An entity-joint ledger sustains a 100-operation life at the pre-registered settings | **Refuted at discovery**; confirmatory test not run | `REPORT_TR.md:706-765` |
| 23 | Gain adaptation beats a fixed gain under plain drift; beats the bank under jumps or mixed regimes | **Open** (inconclusive: below margin or mixed signs) | UPD03, 09, 10 |
| 24 | An acquired gain helps on a matched new task | **Open** (+0.000222, margin 0.0005) | UPD11 |
| 25 | Uncertainty bonus or random deviation beats exploitation in a stationary bandit | **Open** (mean better, most lives worse: 4/48, 20/48) | ENV-01, 02, 04, 05 |
| 26 | A recent window helps after a single reversal | **Open** | ENV-09 |
| 27 | (CEV-11) Prefix averaging buys robustness to left context | **Open** | S2 |
| 28 | Everything about consolidation, resources, transfer, procedure language, credit and context (CON, RES, XFR, LAN, CRD, CTX: 120 entries) and MEM13–20, REP13–20, UPD13–20, ENV-13–20 | **Open** (never run) | `registry.json`, `REPORT_TR.md:9` |
| 29 | Context, credit and capacity engines of wave 2 | **Open** (contract only) | wave-2 `CONTRACT.md` |
| 30 | A general principle of learning while living | **Open** — "bulunmuş veya kanıtlanmış değildir" | `REPORT_TR.md:7` |
