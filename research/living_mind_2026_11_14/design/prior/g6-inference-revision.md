# g6 — Prior-work digest: inference and revision — distinctions (round 19), live revision (round 20), deduction (round 21), cross-map (round 22)

Repository root: `/workspace/cevahir-ai` (read-only). All paths below are relative to that root unless absolute.
Digest written 2026-11-14 for the "new core" design (learning and running as one process; weights as graded
beliefs; concepts formed when needed; the system imagines and measures how real its constructions are; the way
of constructing improves and transfers).

---

## 0. What was read, how the studies connect, integrity

**Named folders.** All four named studies exist under **both** roots; nothing is missing.

| Study (round) | Docs (`docs/research/<study>/`) | Code + results (`research/<study>/`) |
|---|---|---|
| `living_learning_distinctions_2026_09_23` (round 19) | `REPORT_TR.md`, `PROTOCOL.md`, `MATHEMATICS_AND_PRIOR_ART_TR.md`, `WHOLE_PROBLEM_TR.md` | `distinction_learning.py`, `verify_distinctions.py`, `analyze_results.py`, `diagnose_results.py`, `final_audit.py`, `README.md`; `results/{distinction_learning.json (65.9 MB), summary.json, verification.json, diagnostics.json, final_audit.json, preservation_baseline.json}` |
| `living_learning_live_revision_2026_09_23` (round 20) | `REPORT_TR.md`, `PROTOCOL.md`, `MATHEMATICS_AND_PRIOR_ART_TR.md`, `WHOLE_PROBLEM_TR.md` | `live_revision.py`, `verify_live_revision.py`, `analyze_results.py`, `diagnose_results.py`, `final_audit.py`, `README.md`; `results/{live_revision.json (46.0 MB), summary.json, verification.json, diagnostics.json, final_audit.json, preservation_baseline.json}`. Loads round 19's engine and raw result by path (`live_revision.py:13-17`). |
| `living_learning_deduction_2026_09_23` (round 21) | `REPORT_TR.md`, `PROTOCOL.md`, `HYPOTHESES_TR.md`, `MATHEMATICS_AND_LITERATURE_TR.md` | `deduction.py`, `check_proof.py`, `verify_deduction.py`, `analyze_results.py`, `final_audit.py`, `README.md`; `results/{deduction.json (9.5 MB), summary.json (= analysis_stdout.txt), verification.json (= verification_stdout.txt), final_audit.json, preservation_baseline.json}`; `results/initial_run/{PROTOCOL.md, deduction.py, deduction.json}` = the six-method run before the audit amendment |
| `living_learning_crossmap_2026_09_23` (round 22) | `REPORT_TR.md`, `PROTOCOL.md` (superseded where it conflicts), `AMENDMENT.md`, `MATHEMATICS_AND_NEXT_TR.md` | **valid:** `corrected.py`, `analyze_corrected.py`, `verify_corrected.py`, `tree_audit.py`, `model_scope_audit.py`, `final_audit.py`, `README.md`; `results/{crossmap_corrected.json, summary_corrected.json, verification_corrected.json, tree_audit.json, model_scope_audit.json, final_audit.json, preservation_baseline.json}`. **invalid, kept for audit:** `crossmap.py`, `analyze_results.py`, `results/{crossmap.json, summary.json, analysis_stdout.txt}`, `results/invalid_initial_run/`, `results/correction_audit_v1/` |

**Order and dependencies.** The sections below follow the order in which the rounds were done (19, 20, 21, 22),
not the order of the task list, because each round uses the one before it:

- Round 20 takes round 19's *actually learned* machines — including the 15 wrong ones — as the programs it has to
  install into a running stream (`live_revision.py:105-115`).
- Round 21 leaves finite-state machines and asks whether computation *without new outside evidence* can change
  later capacity; its hypothesis list (`HYPOTHESES_TR.md`) names H21.10 + H21.11 as the next test.
- Round 22 is that next test, and its report also corrects a wording error in round 21's table
  (`living_learning_crossmap_2026_09_23/REPORT_TR.md:73`: the "185 lemma/yaşam ortalaması" of round 21 is the
  total over 24 lives; the mean is 7.7083).
- Round 23 (the campaign) links back to rounds 22 and 21 in one paragraph each
  (`docs/research/living_learning_campaign_2026_09_24/REPORT_TR.md:65,67`, repeated in
  `docs/publications/papers/10-living-learning-campaign-wave1.md:107,109`). That is the only place in
  `docs/publications/` where any of these four folders is cited. **There is no paper edition of rounds 19–22.**

The preservation manifests show the same order: 1,023 earlier files hashed by round 19, 1,039 by round 20, 1,055
by round 21, 1,075 by round 22, all against base HEAD `f4eb1dc550b3efc8b2a62fc1c5db262c804b219c`.

**Integrity checks I ran (read-only, seconds; scripts and outputs in
`/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/prior/_checks_g6/`).**

- `g6_integrity.py → g6_integrity.json`: all artifact hashes stored in the three `results/final_audit.json`
  files that carry them (15 distinctions + 15 live_revision + 34 crossmap = 64 files: every `.py`, `.md` and result
  file, including the archived invalid crossmap runs) equal the sha256 of today's files (`mismatching: []` three
  times). Round 21's `final_audit.json` stores **no** artifact hashes; for it I checked the three hashes stored
  in the raw result (`source_sha256`, `checker_sha256`, `protocol_sha256`) against today's `deduction.py`,
  `check_proof.py` and `PROTOCOL.md` — all equal. Round 20's stored hashes of round 19's engine and raw result
  also match.
- `git status --short` was empty before and after; no `__pycache__` was created (I never imported a repository
  module; every check reads stored JSON with `python3 -I`). HEAD today is `e436b04…`; the files of these
  studies are byte-identical to what their audits hashed at base `f4eb1dc…`.
- Original environment: Python 3.14.3, standard library only (`final_audit.json › python`; READMEs). This
  machine: Python 3.13.18. **I did not rerun any experiment.** Every number below is read from the stored
  result files; where I re-aggregated raw records the script is named and the item is marked **[ANALYST]**.

**Substrate fact.** Pure Python standard library — not even NumPy. No neural network, no gradient, no embedding,
no softmax anywhere in this group; the only commit-to-one readouts are the deliberately weak controls
(`reset_start` in round 20, `point` in round 22). What is "learned" is one of:

| Round | Persistent state that experience changes | Form of belief |
|---|---|---|
| 19 | two sets of binary strings `S` (access histories) and `E` (tests), a dictionary word → label, and the machine compiled from them | crisp: two histories are in the same state or not |
| 20 | the installed program, a **set** of states the program may be in, one parity bit, and 0 / 8 / all bits of the stream | crisp set: a state is possible or not; answer only if all possible states agree |
| 21 | a list of clauses, each with its proof, the clauses the proof used and the identities of the base clauses it rests on | crisp: a lemma is active (all supports present) or dormant |
| 22 | the list of observed identity pairs (`known`, `inverse`), or the list of surviving permutations | counts over surviving hypotheses = exact rational probabilities under a stipulated uniform model |

So every law below is a statement about **state and its future use**, not about a substrate. None of it is
evidence for or against any particular neural or non-neural core; it constrains what *any* core must keep.

**Common experimental template (stated once; deviations are noted per study).**
(a) a protocol written before the performance run (round 22: superseded by a post-result amendment);
(b) 24 deterministic seeds as the paired independent unit ("lives"); every method sees the same world and the
same inputs; method × seed products are explicitly *not* counted as independent;
(c) a declared **virtual work meter** (counted table reads, comparisons, transitions, written scalars/bytes) —
never CPU time or RAM;
(d) a **same-evidence replay control** that rebuilds the learner from the full history and must reach the
identical state (rounds 19, 21, 22; in round 20 the corresponding reference is the full-archive replay of the
installed program);
(e) a deliberately weak or unsound control and at least one strong control;
(f) an evaluator that knows the hidden world and never feeds anything back to the learner;
(g) an independent verifier (different algorithm), exact rerun, fresh-process continuation from serialized state,
and a hash manifest of all earlier files;
(h) statistics: **descriptive paired counts and means only** — no confidence intervals, no significance tests
("No significance test across 24 constructed seeds is claimed", crossmap `PROTOCOL.md:43`; round 21's analysis
script is headed "no independent-world inflation or significance claim", `analyze_results.py:1`).

**Evidence grades used in this digest.**
- **[EXACT]** — a proof, a finite algebraic witness, or an exhaustive finite check in the study's own documents
  or code.
- **[EMP n=24]** — measured over 24 paired lives (in rounds 19–20: 24 seeds × 3 world families = 72 worlds,
  seeds shared across families), descriptive only.
- **[DESIGN-CONSTANT]** — a number that is identical in every life by construction, so "24 of 24" is one fact,
  not 24 (round 22's cost figures; see §4 (6)).
- **[ANALYST]** — my own re-aggregation of the stored raw records or my own reading; not a claim of the study.
- Decision labels are the studies' own: **supported / partially supported / rejected-in-scope / not supported /
  not met**. Where a study gives no label, I say so.
- "Given" lists what the researcher supplied to the learner; it matters as much as the result, and every one of
  these four reports states that the *procedure* that builds the new structure was supplied, not acquired.

---

## 1. Study `living_learning_distinctions_2026_09_23` (round 19) — building new distinctions from experience, and staying stuck with a missing one

Report title: "Deneyimle yeni ayrımlar kurmak ve eksik ayrımda kalmak" (`REPORT_TR.md:1`).

### (1) Question

Without being told how many states the world has or which continuations tell them apart, can the outcomes of
histories that looked the same enlarge the distinctions the learner uses and the executable state graph it
runs — and what computation does the retained structure save compared with rebuilding it from the same evidence?
(`PROTOCOL.md:7`)

### (2) Mechanism / state representation actually used

A symbolic **observation table** (Angluin's L\* without the equivalence teacher), compiled to a finite-state
machine. No statistics, no weights. All code references are to
`research/living_learning_distinctions_2026_09_23/distinction_learning.py`.

**What is stored** (`Learner.export`, `:72-74`): `S` (set of access histories, binary strings), `E` (set of
distinguishing continuations = "tests"), `known` (dictionary word → 0/1 label), `splits` (one record per
distinction ever created), `counterexamples`, `model` (start state, transition table, output table), `status`.
Around the learner, per life (`:213-215`): the world's answer cache and the list of the learner's own queries,
a work meter by category, the passive batches so far, a `frozen` flag.

**Start** (`:65-67`): `S = {ε}`, `E = {ε}`, no labels; machine = one state, output 0.

**The update rule — one stabilisation** (`stabilize`, `:83-129`) repeats the following until nothing fires:

1. *Rows.* For every `s ∈ S ∪ S·{0,1}` compute `row(s) = (y(s·e))_{e∈E}` (`:86-87`). A label not yet known is
   asked from the world (`value`, `:75-78`): the world is reset and the word is run, at `len(word)+1`
   environment units for a new word (`Environment.query`, `:49-55`).
2. *Closure* (`:92-97`). If the row of some one-step extension is not the row of any member of `S`, add that
   history to `S` — a new state appears.
3. *Consistency → a new distinction* (`:98-116`). For `s ∈ S` with representative `r` (first member with the
   same row): if for some symbol `a` the rows of `s·a` and `r·a` differ, take the first test `e` on which they
   differ and add the new test **`a·e`** to `E` (`new = bit + suffix`, `:109`). A split record is written
   (`:113-115`): `{left, right, action, suffix, new_test, previous_tests, old_row, outcomes, added}` — the two
   histories that used to be together, why they were together, and the outcome pair that separates them.
4. *Compile* (`:117-120`). States = the distinct rows of `S`; start = `row(ε)`; `δ(row(s), a) = row(s·a)`;
   output = first column (= `y(s)`, because ε sorts first in `E`).
5. *Own-evidence check* (`:121-128`). Run the compiled machine on **every** known word, shortest first. At the
   first word whose stored label it contradicts, add all prefixes of that word to `S`, log the word in
   `counterexamples`, and go back to 1. (The "counterexamples" are the learner's own stored labels, not a
   teacher's.)
6. Otherwise accept: `model ← compiled`, `status ← 'consistent'` (`:129`).

In the study's notation (`MATHEMATICS_AND_PRIOR_ART_TR.md:9-41`):

```
h ~_E h'   ⟺   ∀ e ∈ E :  y(he) = y(h'e)          provisional sameness under the tests held now
h ~_* h'   ⟺   ∀ e ∈ Σ* : y(he) = y(h'e)          true sameness
ρ_E(h) = ρ_E(h')  and  y(hae) ≠ y(h'ae)  for some e ∈ E   ⟹   the new test  ae  separates h and h'
closed:     ∀ s∈S, a∈Σ :  ρ_E(sa) ∈ ρ_E(S)
consistent: ρ_E(s) = ρ_E(t)  ⟹  ∀a : ρ_E(sa) = ρ_E(ta)
then        δ(ρ_E(s), a) = ρ_E(sa),   g(ρ_E(s)) = y(s)       is a well-defined machine
```

**The life** (`run_life`, `:177-221`). One initial stabilisation on no passive evidence (it asks at least `ε`,
`0`, `1`). Then 16 blocks of four passive words. Each passive word is predicted with the current machine **before** its label
is opened (`:192-196`); the machine does **not** change inside a block; at the block end the four labelled
words are recorded (`observe`, `:79-82`) and `stabilize()` runs (`:209-210`).

**The four methods** (`:14`):
- `adaptive` — the rule above, with `S`, `E`, `known` carried through the life.
- `rebuild` — at every block end a fresh learner replays all blocks so far, stabilising after each
  (`:201-207`). The world's answer cache is kept, so no word is asked twice. Same-evidence control.
- `exhaustive_tests` — when a new test `a·e` is needed, add *every* binary word up to that length to `E`
  (`:111`). Broader coverage; may ask more of the world.
- `fixed_tests` — `E` stays `{ε}`; at the first inconsistency it returns and keeps its last consistent machine
  (`:108`). Capacity ablation, not a competitor.

**Given vs acquired** (`REPORT_TR.md:11`): alphabet, binary output, determinism, finiteness, reset before every
word, the right to ask about any word, truthful labels and the table procedure are given. "ayırıcı **içerik**
edinildi; onu oluşturma yordamı verildi" — the *content* of the distinctions was acquired; the procedure that
makes them was supplied.

### (3) Experimental design

- **Worlds** (`make_world`, `:139-157`; `PROTOCOL.md:13-17`): 24 seeds × 3 families = 72 hidden binary-input
  machines whose binary output is read after the last symbol.
  - `cycle_reset`: 5–8 states (`5 + seed%4`); one symbol steps round a cycle, the other returns to the start;
    exactly one state outputs 1.
  - `random_cycle`: 6/10/14 states; one symbol steps round a cycle, the other jumps to a random state; random
    outputs (not all equal).
  - `delay`: a 3/4/5-bit shift register (8/16/32 states); the output is the bit *d* steps back.
  Symbols and state names are shuffled per seed (`shuffled`, `:131-137`).
- **Experience**: 64 passive words per life, length 0–24 uniform, fair bits, the *same* stream for the three
  families of a seed (`:151-152`), plus whatever the learner chooses to ask.
- **Evaluation**, at the start and after each block (`evaluate`, `:169-175`): (a) errors on 128 unseen words of
  length 32–64; (b) **exact equivalence on all finite words**, by breadth-first search over the reachable
  product of learned and hidden machine (`equivalent`, `:159-167`); (c) contradictions with known labels;
  (d) split witnesses. Nothing returns to the learner; the evaluator's counterexample is never used for learning
  (`PROTOCOL.md:44`).
- **Units**: 72 worlds, 4 methods, 288 paths, 4,608 block results, 18,432 label-blind predictions. "Bunlar 288
  bağımsız dünya tekrarı değildir" (`REPORT_TR.md:13`).
- **Caps**: 100,000 environment units and 20,000,000 internal work units per life (`:16`) — never reached.
- **Statistics**: counts of exactly learned worlds per family; paired per-seed differences
  (`analyze_results.py:38-52`). No tests, no intervals.

### (4) Results (the study's numbers) and decision labels

**Exact models after the life** (`REPORT_TR.md:27-32`; `results/summary.json › groups[*].final_exact`):

| Method | cycle_reset /24 | random_cycle /24 | delay /24 | Total /72 |
|---|---:|---:|---:|---:|
| `adaptive` (persistent table) | 10 | 23 | 24 | **57** |
| `rebuild` (same evidence, from scratch) | 10 | 23 | 24 | **57** |
| `exhaustive_tests` | 10 | 24 | 24 | **58** |
| `fixed_tests` (no new distinctions) | 0 | 0 | 0 | **0** |

- All 172 exact final models (57 + 57 + 58) have exactly the independently computed minimal number of states
  (`diagnostics.json › independently_minimal_exact_final_models: 172`); the 8/16/32-state delay worlds were
  found without the count being given (`REPORT_TR.md:36`).
- New tests were built in 58 of 72 `adaptive` lives; 57 of those ended exact (`REPORT_TR.md:19`). 592 split
  witnesses were verified against the hidden machine (old tests agree, the new test differs)
  (`verification.json › recorded.split_witnesses`); **[ANALYST]** = 212 `adaptive` + 212 `rebuild` + 168
  `exhaustive_tests` (`g6_distinctions_checks.json`).
- **High sample accuracy without the right structure.** On the 9,216 unseen long words (72 × 128) the
  persistent method makes only **26** errors, yet **15 of 72** final models are not exactly right; **4** of the
  wrong models make **zero** errors on their 128 long tests (`REPORT_TR.md:38,44`;
  `diagnostics.json › wrong_adaptive_models_with_zero_long_test_errors: 4`). All 15 wrong models agree with
  every label the learner holds (`cache_errors_by_method.adaptive: 0`).
- **Where it fails** (`REPORT_TR.md:42-46`; `diagnostics.json › adaptive_failures`): 14 `cycle_reset` worlds in
  which *every* label ever received was 0 — the learner stays with the one-state constant-zero machine, which
  explains all its evidence; the shortest separating word (e.g. `111111` for seed 2) was never tried. One
  `random_cycle` world (seed 19): 9 learned states against 10 minimal, no contradiction with any record; the
  unseen word `000000011` separates them; `exhaustive_tests` learned this world.
- **Persistence = saved recomputation, nothing else.** `adaptive` and `rebuild` have the identical learner
  state, identical query order and identical evidence in all 72 worlds and all 1,152 paired blocks
  (`REPORT_TR.md:52`; `final_audit.json › equal_adaptive_rebuild_blocks: 1152`). Internal learning work:

  | Family | `adaptive` | `rebuild` | ratio of means | [ANALYST] per-life ratio, min–max |
  |---|---:|---:|---:|---:|
  | cycle_reset | 15,417.17 | 104,718.67 | 6.79 | 4.93–8.31 |
  | random_cycle | 57,136.00 | 470,542.17 | 8.24 | 7.68–8.69 |
  | delay | 47,385.58 | 419,849.96 | 8.86 | 7.63–9.50 |

  `adaptive` is cheaper in all 72 lives (`REPORT_TR.md:54-60`).
- **Selective vs broad refinement.** `exhaustive_tests` mean internal work 58,090.96 / 82,335.67 / 200,964.54
  (3.77× / 1.44× / 4.24× `adaptive`) and mean environment units 4,002.33 / 4,001.29 / 7,421.79 against
  1,226.83 / 2,828.08 / 2,186.17 (`REPORT_TR.md:62-68`). But in 6 of the 24 `random_cycle` worlds the
  *selective* method spent more environment units than the broad one: "Daha az yeni test eklemek tek başına daha
  az toplam sorgu garantilemez" (`REPORT_TR.md:70`). The broad method also solved one more world. The pair is
  not a same-evidence comparison (it asks different questions).
- **Ablation**: `fixed_tests` 0/72, 1,677 contradictions with its own labels at the end (25 + 794 + 858).
- **Resources**: maximum internal work 1,057,573 of 20,000,000; maximum environment units 33,256 of 100,000 —
  "bu tur eşit sınırlı kaynak nedeniyle gerçekleşmiş başarı farkı göstermiyor" (`REPORT_TR.md:76`).
- **Verification** (`REPORT_TR.md:84-90`; `final_audit.json`): all 288 paths rerun exactly; 4,608 exactness
  verdicts confirmed by a different algorithm (partition refinement of the joined machines,
  `verify_distinctions.py:16-29`); 589,824 long-test outputs recomputed; 57,799 stored labels checked; 24
  fresh-process continuations exact; 4,096 pairs of two-state machines cross-checked.

**[ANALYST] re-aggregations of the stored raw records** (`g6_distinctions_checks.json`,
`g6_distinctions_initial.json`):

- In `cycle_reset`, exact learning coincides **perfectly** with whether the 64 passive words happened to
  contain a positive label: 10 worlds with ≥ 1 passive positive → 10 exact; 14 worlds with none → 14 wrong.
  Failures by number of target states: 1/6 (5 states), 3/6 (6), 5/6 (7), 5/6 (8). The learner's own questions
  never found a positive that the stream had not already shown.
- 67.3 % of all labels the persistent learner holds at the end were **asked by itself** from the resettable
  world (mean own queries per life 35.9 / 168.7 / 126.0, against at most 64 passive words, 59.75 distinct).
- The initial hypothesis is the one-state machine in 57 worlds and a two-state machine in 15 `random_cycle`
  worlds; it is exact in none. First exact block: `delay` 23 worlds after block 1 and 1 after block 2;
  `random_cycle` 12 after block 1, all 23 by block 6; `cycle_reset` spread out to block 16. An exact model was
  never lost again (0 lives).
- 64.3 % of the persistent learner's internal work is step 5 — re-running the hypothesis on the whole label
  archive (`model_execution`); row lookups 9.6 %, closure 8.6 %, consistency 7.1 %.
- Label-blind passive predictions wrong: `adaptive` 111 of 4,608 (2.4 %), `exhaustive_tests` 103, `fixed_tests`
  1,247.
- The learner's own stored labels forced 81 re-openings of the table (`counterexamples`) over the 72 lives.
- Longest test ever built: 6 symbols; longest access history: 24; shortest separating word of the 15 wrong
  models: 4–9 symbols.

**Decision labels.** The protocol states no numbered hypotheses. The report's own claims and their status:

| Claim | Status | Evidence |
|---|---|---|
| Distinctions and an executable state graph can be built from outcomes without the state count or the list of distinguishers | **shown, within a supplied finite-state language** | 57/72 exact on all finite words |
| The retained table gives no other information than rebuilding from the same evidence; it saves the recomputation | **shown** | identical states in 1,152 blocks; 6.79–8.86× |
| The same mechanism stays undeveloped when the separating experience never occurs | **shown** | 14 all-zero worlds |
| More selective creation of distinctions is unconditionally better | **refuted** ("koşulsuz üstünlük yoktur", `REPORT_TR.md:46`) | seed 19; 6/24 worlds with higher environment cost |
| A new principle was found | **explicitly not claimed** — "klasik gözlem tablosu öğrenimidir" (`REPORT_TR.md:5`) | Angluin 1987; Rivest & Schapire 1989 read as prior art (`MATHEMATICS…:90-92`) |

### (5) Laws / invariants claimed (all labelled as known automata theory, `MATHEMATICS_AND_PRIOR_ART_TR.md:5`)

- **D1 — tests only refine [EXACT].** `E ⊆ F ⟹ (h ~_F h' ⟹ h ~_E h')`. A new test can separate histories that
  were held together; it can never join histories that an old test separated. In a stationary world, histories
  that are truly equivalent (`~_*`) are never separated later; what changes is the learner's partition under
  finite evidence, not the world's meaning (`:23-25`). *Scope*: fixed deterministic `y`, truthful labels.
  **[ANALYST]** This is the automaton form of the programme's statement that the family of future operations
  determines the state: the state is the class of histories under the tests currently held, and each newly
  relevant continuation refines it.
- **D2 — a new distinction is constructed, not looked up [EXACT].** If two histories have equal rows but after
  the same action their rows differ on a held test `e`, then `a·e` separates them (`:29-41`). The separator is
  built by prefixing one action to an already-held separator. *Witness*: the three-state machine at `:43-51`
  (ε and `0` both output 0; `01 → 1`, `1 → 0`), checked in code (`verify_distinctions.py:31-38`).
- **D3 — closed + consistent ⟹ executable [EXACT].** `δ(ρ(s),a) = ρ(sa)`, `g(ρ(s)) = y(s)` is well defined
  (`:55-68`).
- **D4 — fitting all evidence certifies nothing beyond it [EXACT + EMP].** A closed, consistent table that
  explains every held label is "dünyayla bütün dizilerde eşdeğerlik sertifikası değildir" (`:70`).
  *Constructive counterexample* (`:72`): an all-zero record is consistent with the constant-zero machine and
  with a machine that accepts exactly one untried word. *Measured*: 15 wrong models with 0 contradictions; 4 of
  them also pass 128 unseen long tests.
- **D5 — the learner cannot certify itself; the evaluator can [EXACT].** Equivalence on all finite words is
  decidable by the reachable product, but only for someone who holds the hidden machine; "rapordaki tam
  eşdeğerlik, öğrenicinin kendi başına elde ettiği bir sertifika değildir" (`:78`).
- **D6 — an exact model makes the raw archive unnecessary for *prediction in the same stationary world*, and
  only for that [EXACT, not exercised].** (`:80`.) The implementation never deleted the archive; nothing
  follows for a new world, a new kind of output, or a later change of representation.
- **D7 — same evidence + same deterministic procedure ⟹ same state (induction) [EXACT + EMP].** Therefore the
  difference between persistent and rebuilt is *only* recomputation (`:84-86`).

### (6) Limits

**Admitted by the study.**
- Classical observation-table learning; no new principle (`REPORT_TR.md:5,98`).
- Stationary, noise-free, deterministic worlds with reset and free truthful queries; the mechanism only *adds*
  distinctions — "Birleştirme, değişen dünya, duygusal değerlendirme veya geçmiş olayın yeni bağlamda farklı
  anlam kazanması bu mekanizmayla bütünüyle açıklanmaz" (`MATHEMATICS…:25`).
- Learning happens in resettable episodes with block-wise updates: "kesintisiz beden/akış güncellemesi
  gösterilmez" (`PROTOCOL.md:19`).
- `adaptive` vs `exhaustive_tests` is not an equal-evidence comparison (`REPORT_TR.md:70`).
- No path met a resource cap, so nothing is shown about learning under a binding budget; the cap wording in the
  protocol ("freeze at budget") and the code (only *new uncached* queries are refused) differ, documented and
  not retro-fitted (`REPORT_TR.md:78`).
- Virtual work is not CPU or memory; the full query archive is kept; bounded-memory learning untested
  (`REPORT_TR.md:72`).
- Local reproducible record; no external preregistration or review (`REPORT_TR.md:92`).

**Limits I see.**
- **Not one process.** The machine is frozen while it runs a block and rebuilt between blocks; running and
  learning alternate.
- **Not archive-free, and the dominant cost is re-validation.** 64 % of the "persistent" learner's work is
  re-executing the hypothesis on every stored label at every stabilisation; that cost grows with the archive.
  Persistence removed the *rebuilding*, not the *re-checking*.
- **Most of the experience is experiment on a resettable world** (67 % self-asked labels). The result says
  little about a stream that cannot be rewound.
- **Questions are asked only when the table needs them.** There is no step that tries something the current
  structure does not call for. In `cycle_reset` the outcome was decided entirely by the passive stream
  (10/10 vs 0/14 above). The default hypothesis "nothing ever happens" is never challenged from inside.
- **Failure is silent.** In all 15 wrong worlds the learner's own status is `consistent`. It has no quantity
  that says "this structure has never been tested beyond what I happened to see".
- **Crisp table, no graded belief.** One wrong label would create a permanent spurious state; two conflicting
  labels for one word stop the program (`assert`, `:81`). Splits are never undone.
- **One configuration, descriptive counts.** Family size is tied to `seed % 3` or `seed % 4`, so state count and
  seed are not varied independently.

### (7) Implications for a core in which learning and running are one process

- **Mechanism to reuse — "split on a witnessed inconsistency".** Keep two histories in one state while every
  held test agrees; when the same action leads them to different outcomes, create the new test `a·e`, and store
  the witness (`{left, right, action, suffix, outcomes}`). This is a concrete, auditable rule for *forming a
  concept when it is needed*: need = a contradiction inside the present grouping; the concept comes with the
  record of why it exists; 592 such records were verified. It also gives the minimal structure (172/172
  minimal machines) — no distinction is created without a witnessed need.
- **Mechanism to reuse — compile on closure.** A distinction becomes part of what the system *does* (states and
  transitions), not a note beside it: rows → machine (`:117-120`).
- **Requirement — incremental re-validation.** In a one-process core the check "does the new structure still
  explain what I hold?" cannot be a re-run over the whole archive at every change (64 % of the work here). It
  has to be local to what changed.
- **Warning — "explains all my evidence" is not "is real".** D4: 15/72 wrong with zero contradictions; 4 pass
  128 unseen long tests; 26 errors in 9,216 samples hide 15 wrong structures. A measure of how real a
  construction is must include *what has never been tried*, not only agreement with what has.
- **Warning — need-driven formation alone is blind where nothing contradicts.** 14 worlds stayed at one state
  because no evidence disagreed. If concepts form only on contradiction, a system whose experience never
  contradicts its default never grows. The broader control helped only marginally (58 vs 57) and at 1.4–4.2×
  the work — breadth of imagination is not free and was not sufficient either.
- **Warning — a retained structure's value is computational.** D7: identical state from replay. Do not credit
  persistence with knowledge; credit it with the 6.8–8.9× it saves, and count that against what it costs to
  keep.
- **Open question — forming the forming procedure.** "Deneyim geldiğinde yeni ayrımın nasıl kurulacağı bu turda
  hazırdı" (`WHOLE_PROBLEM_TR.md:27`). Nothing here improves or transfers the way of constructing.
- **Open question — no reset.** Rivest & Schapire's homing sequences are named as prior art, so removing reset
  "başlı başına keşif sayılmaz" (`WHOLE_PROBLEM_TR.md:47`); the open part is doing it while the built structure
  is already in use (taken up, in part, by round 20).

### 1B. `WHOLE_PROBLEM_TR.md` — what distinction-forming explains and what it does not

A direction record, not an experiment. Its content:

- **Status table** (`:13-21`): for each research need, what round 19 did and what stays open — e.g. *new
  computational distinction*: built from observed inconsistency, state count not given / open: acquiring a new
  language of operations, observations and evaluation; *continuous operation*: learning in successive resettable
  episodes / open: building a new representation "dünyayı sıfırlamadan ve mevcut çalışan durumu kaybetmeden";
  *reorganisation of concepts*: provisionally merged histories were split / open: merging, cross-task meaning,
  change of value, a concept shrinking, different old concepts joining.
- **Three gaps that do not reduce to each other** (`:25-31`): (1) the *content* that makes a distinction
  possible (the separating experience may never have occurred — and "yalnız daha çok sorgu seçmeye odaklanmak
  ana problemi tekrar keşif problemine daraltır"); (2) the *form of computation* that builds the distinction
  (here supplied); (3) *attaching* the new structure to the life already running ("'Yeni modeli öğrendim'
  mevcut konumunun doğru taşındığı anlamına gelmez"). They are "zorunlu ayrı yazılım modülleri önerisi
  değildir" — one structure may serve more than one.
- **Two untested hypotheses** (`:37-41`): **H19-A** — the use of a new distinction may lie not in solving the
  present target better but in the cost of acquiring or carrying the *next* task; a growing state count alone
  must not count as success. **H19-B** — a complete old model and a state sufficient for *future learning* can
  differ; when new observable outcomes or operations arrive, either the separating information is still
  accessible in old experience, or it was never observed or was lost, and then "temsil büyütmek bilgiyi yoktan
  çıkaramaz". No protocol was written; no result is claimed.
- **On novelty** (`:45`): the contribution is narrow and checkable — the ready-made distinctions of earlier
  rounds were removed; the conditions under which acquired distinctions are and are not sufficient on all
  finite words were measured; the same-evidence control separated new information from repeated computation.

---

## 2. Study `living_learning_live_revision_2026_09_23` (round 20) — attaching a newly learned structure to a history that is already running

Report title: "Öğrenilen yeni yapının çalışan geçmişe bağlanması" (`REPORT_TR.md:1`). First sentence of the
report: "**Yeni modeli öğrenmek, onun mevcut geçmişteki doğru çalışma durumunu kendiliğinden vermez.**"

### (1) Question

In a stream that is never reset, is knowing the new program the same as knowing which state that program should
be in *now* — and what can the old learned state tell about the new state without the raw history?
(`PROTOCOL.md:7`)

### (2) Mechanism / state representation actually used

Symbolic; sets of automaton states. All code references are to
`research/living_learning_live_revision_2026_09_23/live_revision.py`.

**What a tracker stores** (`Tracker.export`, `:64-67`): `model` (the installed program: start, transitions,
outputs), `possible` (the **set** of states the program may be in), `parity` (one bit: exact parity of all
stream bits — a skill given at the start and never modified), `position`, `suffix` (the last ≤ 8 bits; two
policies only), `history` (all bits; `full_archive` only), work categories, counters.

**Running** (`advance`, `:68-73`), for each real bit `a`:
`parity ← parity ⊕ a`; `possible ← { δ(q, a) : q ∈ possible }`; append the bit to the trace where one is kept.

**Answering** (`predict`, `:100-103`): if all possible states have the same output, give it; otherwise return
`None` — *undetermined* ("kararsız"). No probability, no random guess (`PROTOCOL.md:32`).

**Installing a revised program** (`install`, `:74-99`) — six policies; `A` = old program, `B` = new program,
`U` = old possible set, `p` = current parity, `s` = kept suffix, `h` = whole history:

| Policy | New set of possible states | What it uses |
|---|---|---|
| `reset_start` | `{ start_B }` | nothing (weak negative control) |
| `all_states` | every state of `B` reachable from its start | the new program |
| `parity_suffix8` | `{ b : (p, b) ∈ R_B^s }` | new program, parity, last ≤ 8 bits — **not** the old learned state |
| `model_relation` | `{ b : ∃ a ∈ U, (p, a, b) ∈ R_{A,B} }` | old and new program, old possible set, parity |
| `model_relation_suffix8` | `{ b : ∃ a ∈ U, (p, a, b) ∈ R_{A,B}^s }` | all of the above + last ≤ 8 bits |
| `full_archive` | `{ δ_B(start_B, h) }` | the whole bit history |

The relation (`relation`, `:34-52`; `MATHEMATICS_AND_PRIOR_ART_TR.md:17-43`):

```
R_{A,B}   = { (P(w), q_A(w), q_B(w)) : w ∈ {0,1}* }            all triples co-reachable from (0, a0, b0); ≤ 2|A||B|
R_{A,B}^s = { (p ⊕ P(s), δ_A(a,s), δ_B(b,s)) : (p,a,b) ∈ R }   pushed through the kept suffix on all three parts together
T(p, U)   = { b : ∃ a ∈ U, (p, a, b) ∈ R^(s) }                 then filtered by the current parity and the old set (:95)
```

Without the old program the same is done with pairs `(P(w), q_B(w))`. No label from the live stream is ever
used to narrow the set, to change the program or to choose the policy (`PROTOCOL.md:36`). The relation is
recomputed at every revision and is not cached (`REPORT_TR.md:92`).

**Where the programs come from** (`sources`, `:105-115`): round 19's `adaptive` path in each of its 72 worlds —
its initial machine and its machine after each of the 16 learning blocks. A revision is installed at stream
position `stride × (block + 1)`, and only if the machine actually changed (`:76`).

**Given vs acquired**: the parity skill, the trace length 8, the six policies and the relation construction are
given; the programs were acquired (in round 19, in a separate resettable channel).

### (3) Experimental design

- 24 seeds × 3 families — the 72 worlds of round 19, wrong final models included ("gizli doğruluğa göre başarılı
  dünya seçilmez", `PROTOCOL.md:9`) — × 2 strides (32 or 256 real bits between revisions) × 6 policies =
  **864 paths**.
- Stream: `16 × stride + 16` fair bits per (seed, stride), the same for all families and policies
  (`stream_for`, `:117-119`). The world is never reset.
- At each of the 16 boundaries, predictions at t = 0…16 bits after the boundary are recorded: 272 per path,
  **235,008** in all, 39,168 per policy.
- The evaluator (`run_life`, `:126-137`) computes, independently of the tracker, the state the *installed
  program* would be in after reading the whole real history (`replay_state`) and the true world output. This
  gives three separate verdicts per point: **migration** — is the replay state inside the tracker's set;
  **model** — does the program's replay output equal the world's output; **answer** — right, wrong or withheld.
  State numbers of target and learned machine are never compared (`REPORT_TR.md:27`).
- Revision events (`REPORT_TR.md:17`; `diagnostics.json`): 14 of the 72 source worlds never change their table;
  48 change once, 8 twice, 2 three times → 70 source revisions → **140 migration events per policy** over the
  two strides. In 27 source revisions the old machine had more than one state. **[ANALYST]** 47 of the 70
  happen at the first block; all 27 multi-state cases are `random_cycle`; 57 of 70 install an exact machine.
- Costs: added live and migration work per path, on top of the common calibration cost carried over from round
  19 (mean 39,979.58 internal units per life, `REPORT_TR.md:90`).
- Statistics: counts; paired per-world differences (`analyze_results.py:49-64`); no tests.

### (4) Results (the study's numbers) and decision labels

**Points at which the installed program is exactly right on all finite words** — 28,458 per policy
(`REPORT_TR.md:33-40`; `summary.json › methods[*].exact_model_points`):

| Policy | right, certain | **wrong, certain** | undetermined |
|---|---:|---:|---:|
| `reset_start` | 27,745 | **713** | 0 |
| `all_states` | 26,877 | 0 | 1,581 |
| `parity_suffix8` | 27,270 | 0 | 1,188 |
| `model_relation` | 26,886 | 0 | 1,572 |
| `model_relation_suffix8` | 27,270 | 0 | 1,188 |
| `full_archive` | **28,458** | **0** | **0** |

**All points, right and wrong programs together** — 39,168 per policy (`REPORT_TR.md:60-67`):

| Policy | right | wrong | undetermined | replay state outside the set |
|---|---:|---:|---:|---:|
| `reset_start` | 38,010 | 1,158 | 0 | 1,471 |
| `all_states` | 36,945 | 305 | 1,918 | 0 |
| `parity_suffix8` | 37,403 | 324 | 1,441 | 0 |
| `model_relation` | 36,956 | 305 | 1,907 | 0 |
| `model_relation_suffix8` | 37,403 | 324 | 1,441 | 0 |
| `full_archive` | 38,750 | **418** | 0 | 0 |

- **At the 140 migration instants (t = 0)** (`REPORT_TR.md:44`): `reset_start` 79 right, **61 wrong** (its
  state differs from the replay state in 115); `full_archive` 133 right, 7 wrong — all 7 are errors of the
  program itself; the two trace policies 81 right, 3 wrong, 56 undetermined; `all_states` and `model_relation`
  undetermined in all 140.
- **The old learned state added nothing to a short trace.** `model_relation_suffix8` and `parity_suffix8`
  produced "**144 eşleştirilmiş yaşamın bütün 39.168 noktalarında aynı olası durum kümelerini ve aynı
  cevapları**" (`REPORT_TR.md:50`), while the relation policy cost on average 721.58 (stride 32) and 653.19
  (stride 256) more work per life, was cheaper in no life and equal in the lives without a revision.
- Without a trace, `model_relation` gave only 11 more certain answers than `all_states`, in two lives at
  stride 32 — and it also conditions on parity, which `all_states` does not, so even these 11 cannot be credited
  to the old *learned* state (`REPORT_TR.md:52`).
- **Eight bits are not a general fix**: in `cycle_reset` and `delay` the short trace fixed the state at every
  change; in `random_cycle` 28 of the 36 changes per stride stayed undetermined (`REPORT_TR.md:54`).
- **Right migration does not repair a wrong program**: for the five covering policies a certain answer never
  contradicted the same program's full-history replay, yet 305–418 answers were wrong because the program was
  (`REPORT_TR.md:69`).
- **Error accounting** for `reset_start` (`REPORT_TR.md:71-73`; `diagnostics.json`): the archive's own program
  errors 418; answers spoiled although the replay was right 782; answers accidentally right although the replay
  was wrong 42: **1,158 = 418 + 782 − 42**.
- **Parity** was right in all 235,008 measurements — "ayrı ve değiştirilmeyen bir sayaçtır" (`REPORT_TR.md:46`).
- **Cost** (`REPORT_TR.md:81-88`), added work per life:

  | Policy | stride 32 | stride 256 | live trace kept at the end |
  |---|---:|---:|---|
  | `reset_start` | 1,328.97 | 8,496.97 | 0 bits |
  | `all_states` | 1,512.10 | 8,911.35 | 0 bits |
  | `parity_suffix8` | 2,227.86 | 13,205.11 | 8 bits |
  | `model_relation` | 1,871.24 | 9,270.93 | 0 bits |
  | `model_relation_suffix8` | 2,949.44 | 13,858.31 | 8 bits |
  | `full_archive` | 1,922.75 | 13,135.19 | 528 / 4,112 bits |

  Peak temporary relation: 356 triples; the trace-only control at most 64 pairs. Full archive needed *less* work
  on average than relation + trace in these lives and answered more, at the price of more stored raw history:
  "arşivsiz yöntemin genel hesap üstünlüğü gösterilmedi" (`REPORT_TR.md:92`).
- **Verification** (`REPORT_TR.md:102-110`; `final_audit.json`): 864 paths rerun exactly; 235,008 points checked
  against an independent replay; 195,840 containment checks for the five covering policies; 258,048 + 258,048
  relation witnesses on all pairs of the 64 two-state machines; 39,168 subset checks (all equal); 18 paths with
  renamed states identical; 36 fresh-process continuations exact.

**[ANALYST] look at the stored raw records** (`g6_live_revision_checks.json`; the bit stream was regenerated from
the three-line rule at `live_revision.py:117-119` and validated against all 235,008 recorded parities, and my
re-simulated sets equal the stored sets at the end of every block):

- Of the 140 migrations, the old program had a **single state in 86** — there the "old learned state" is empty
  of information by construction. It had several states in 54.
- In **49 of those 54** the old possible set was a singleton: the old program's state was known *exactly*. In
  **43 of these 49** the new set after migration still had more than one member. In **0 of the 54** was the new
  set smaller than the trace-only control's.
- Without a trace: 48 of 54 old sets were singletons; all 48 new sets had more than one member; the old state
  made the new set smaller than `all_states` in 5 of 54 events (mean set size over all 140 migrations 11.98
  against 12.07).
- **How long a wrong start lasts** (`reset_start`, 140 migrations, no feedback): state different from the
  replay state at t = 0 / 4 / 8 / 16 bits: 115 / 51 / 35 / **26**; answer different from the replay answer:
  60 at t = 0, 9 at t = 16.
- **How uncertainty decays without feedback**: `parity_suffix8` undetermined 56 → 27 and ambiguous states
  64 → 48 from t = 0 to t = 16; `all_states` undetermined 140 → 36, ambiguous 140 → 54. All remaining ambiguity
  is in `random_cycle` (64 of its 72 migrations start ambiguous; 0 of 20 in `cycle_reset`, 0 of 48 in `delay`);
  two `random_cycle` lives per set policy end the whole life with an ambiguous state.

**Decision labels** — the study's own table (`WHOLE_PROBLEM_TR.md:15-22`):

| Claim | Status after round 20 |
|---|---|
| A more correct new program interprets the past correctly by itself | **False**: 713 wrong answers in phases with an exact program |
| No correct transfer is possible without storing the raw history | **False as a generalisation**: when the present summary or the recent trace suffices, the right state or output is found |
| If the new state cannot be known exactly, a confident wrong answer is unavoidable | **False**: a covering set and withholding avoid some; coverage is the price |
| The old learned structure adds benefit to every new transfer | **Not supported**: identical to the strong eight-bit/parity control on every set |
| A set policy with fewer wrong answers has learned a better model | **False here**: all policies received the same programs; part of the difference is withholding |
| Replaying the archive is in general the most expensive | **Not shown**: cheaper on average than relation + trace in these short lives, but stores more raw history |

### (5) Laws / invariants claimed (labelled as known: program updating, state observers, automata — `MATHEMATICS_AND_PRIOR_ART_TR.md:5`)

- **V1 — coverage (replay-soundness) [EXACT + EMP].** If `q_A(h) ∈ U` and `p = P(h)` then
  `q_B(h) ∈ T_R(p, U)`. *Proof*: `h` itself is the witness triple. The statement does not depend on whether `A`
  or `B` is right about the world; applying the next real bit to every candidate preserves it; so with a covering
  start and this transfer at every change, the program's history-replay state stays in the set through any
  sequence of right or wrong hypotheses (`:17-31`). Not claimed: that the set is the tightest possible (`:33`).
- **V2 — a trace narrows only if every component is pushed through it [EXACT].** `R^s` as above;
  `T^s_{A,B}(p, U) ⊆ T^s_B(p)` — with the same trace, the old learned state can only add a filter (`:37-51`).
  Filtering by the present parity without first pushing parity through the suffix would be wrong (`:43`).
  *Measured*: equality in all 39,168 points.
- **V3 — when one new state can be read off the old summary [EXACT].** A function `f` with
  `q_B(h) = f(P(h), q_A(h))` exists **iff** `(P(h), q_A(h)) = (P(h'), q_A(h')) ⟹ q_B(h) = q_B(h')`
  (`:57-63`). When the condition fails, the options are new information, more trace, or carrying several
  states; "'Daha güçlü model öğrendik' cümlesi eski özetin kaybettiği ayrımı kendiliğinden geri getirmez"
  (`:65`). Stated as the earlier representation-transport condition applied to finite machines, not as a new
  theorem.
- **V4 — passing through a coarser representation can lose what a later one needs [EXACT witness].**
  `A` = number of ones mod 3; `B` = one-state constant; `C` = `A` again. `A → C` directly: the mod-3 state
  suffices. `A → B → C` with the mod-3 state dropped: under parity alone all three residues are possible, and an
  equal eight-bit suffix does not recover it (prefix `ε` against `11`: same parity, different residue). Keeping
  the old program's *code* does not help if its *current state* was not kept: "Program belleği ile çalışma
  durumu belleği ayrılır" (`:69-73`). Checked in `verify_live_revision.py:25-38`. Not an event of the main run.
- **V5 — a certain answer is not a right answer, and totals hide cancellations [EXACT identity + EMP].** If the
  set covers the replay state and all members agree, the answer equals the program's own full-replay answer —
  never more (`:77-79`). For binary outputs,
  `wrong_total = program errors under replay + spoiled although replay was right − accidentally right although
  replay was wrong`, measured 1,158 = 418 + 782 − 42 (`:81-89`). "Sırf doğru cevap yüzdesiyle taşımanın
  doğruluğunu değerlendirmek, 42 tesadüfi düzeltmeyi yanlış biçimde yarar sayabilir."

### (6) Limits

**Admitted by the study.**
- The channel that learns the programs still resets, and the live stream supplies no labels: "sıfırlamasız
  tek-kanallı öğrenme başarısı da iddia edilmiyor" (`REPORT_TR.md:100`).
- Parity is a separate, unmodified counter; its perfect retention says nothing about skills kept in a shared
  representation that changes (`REPORT_TR.md:46`).
- The history of model change is short (70 source revisions): "uzun ve açık uçlu bir yaşam başarısı sayılmaz"
  (`REPORT_TR.md:17`).
- Updates are applied at logical times; no deadline, no background computation (`REPORT_TR.md:19`).
- Virtual costs; no budget cut; no asymptotic conclusion for long lives (`REPORT_TR.md:94`).
- The sets are sound, not tightest: the experimenter knows the time index, the policies do not use it
  (`MATHEMATICS…:33`).
- Prior art: dynamic software updating (Hicks & Nettles 2005), detectability of discrete-event systems (Shu &
  Lin 2011), homing sequences (Rivest & Schapire 1989); "yeni bir genel fikir sayılmıyor" (`REPORT_TR.md:100`).

**Limits I see.**
- **Two channels, not one process.** Programs arrive from a separate learner; the running stream neither teaches
  nor is taught. What is tested is the *seam* between a learner and a runner, not their unity.
- **No feedback on the live stream**, so a wrong working state can only heal through the program's own dynamics,
  never by noticing an error: 26 of 140 wrong starts were still wrong 16 inputs later.
- **Belief is a crisp set and the readout is all-or-nothing.** Whether two or thirty-two states are possible
  (the largest sets: 14 with a trace, 32 without), a disagreement on the output means the answer is simply
  withheld; there is nothing in between.
- **The negative result about the old learned state rests on 54 events in one world family**; in the other 86
  the old program had one state.
- **The trace length, the side invariant and the policy are chosen by hand.**
- The relation is rebuilt from scratch at every change; a cached or incrementally maintained relation was not
  tried, so the cost comparison with the archive is specific to this implementation.

### (7) Implications for a core in which learning and running are one process

- **Requirement — every structural change needs a migration of the working state.** Starting the new structure
  at its initial state while the world keeps running is wrong even when the new structure is perfectly right:
  713 confident errors in 28,458 points; at the instant of change 115 of 140 wrong states and 61 wrong answers.
  In a core where structure changes *while* it runs, this is the normal case, not an edge case.
- **Requirement — three separate measurements, never one through another.** (i) Is the program adequate to the
  world? (ii) Is the working state covered, relative to that program? (iii) Is the answer right, wrong or
  withheld? V5 shows that a total accuracy can contain 782 damages and 42 accidental repairs.
- **Mechanism to reuse — set-valued working state with agree-or-withhold readout**, a sound transfer through
  the co-reachability relation, and narrowing by a short raw trace (`:34-52`, `:74-103`). The invariant to test
  at every step is V1: the program's replay state is in the support. For graded beliefs the same invariant
  applies to the support of the weights.
- **Mechanism to reuse — a small invariant that survives every revision** (here parity). Carried unchanged
  across revisions, it narrows every migration without risk; in the trace-only control it is, with the eight
  bits, the only thing carried across the change — and adding the old learned state to that control changed
  nothing.
- **Warning — the old abstraction did not help to seat the new one; a few bits of raw recent experience did.**
  Identical sets in 39,168 points, 653–722 extra work per life; **[ANALYST]** an exactly known old state left
  the new state ambiguous in 43 of 49 cases. A reading consistent with V3: a revision exists *because* the new
  program separates histories the old one held together, so the old state is silent exactly where the new one
  needs information; raw trace is not. Hence: keep a short raw trace next to every abstraction that may be
  revised — an abstraction cannot be refined out of itself.
- **Warning — V4.** Reorganising through a coarser form destroys working state that a later, finer form needs;
  keeping the code of the old concept is not keeping where it was.
- **Warning — withholding has a price.** 1,188–1,581 of 28,458 answers not given; fewer wrong answers is not a
  better model.
- **Warning — the archive won here.** Replay was the only policy that was always right when the program was
  right, and it cost less than relation + trace. "Archive-free" was not shown to be better; the choice of what
  raw trace to keep is a real design decision and nothing in this round makes it.
- **Open question — single-channel learning.** Forming the revision from the live stream itself and migrating
  in the same step. Not done anywhere in the programme up to this round.
- **Open question — does old learned state ever narrow a migration beyond trace?** V2 allows it; it was not
  observed.

### 2B. `WHOLE_PROBLEM_TR.md` — parts that now work together, and the explanation that is still missing

A direction record. Beyond the claims table quoted in (4):

- "Giderek daha çok yapı eklemek otomatik olarak daha iyi öğrenme sağlamıyor" (`:24`) — said with reference to
  round 17 (the strong control matched learned repair), round 18 (unconditional persistence cost more) and
  round 19 (perfect sample accuracy hid a wrong model).
- **The risk of the method** (`:28`): deterministic finite machines give exact verification and explicit
  counterexamples; the risk is to see living learning more and more as machine identification plus state
  transport. Most of the owner's proposal — concepts branching, merging, cross-linking, old events acquiring a
  new reading — "bu dünyalarda bulunmuyor".
- **Three areas kept apart** (`:32-36`): **A** — the *way of learning* developing with experience, not only its
  content; the test should show which computation can no longer be built, or becomes more expensive, from the
  same new experience once an acquired procedure is removed. **B** — shared meaning reorganised across several
  tasks; so far the target function was fixed and a new need never changed which old distinctions suffice.
  **C** — the life's common resource decision: eight bits, the whole archive or the old working state was a
  ready-made choice; the open part is deciding by experience which trace to keep before the next task is known —
  there is no free complete summary for all futures, "fakat bu, gerçekçi dağılımlarda işe yarayan kontrollü
  seçimlerin bulunamayacağı anlamına gelmez".
- **Criteria for the next discriminating study** (`:42-44`): structure acquired from *different* sets of
  experience must change the form of computation of a new task; which mapping or reusable operation to choose
  must be acquired without the correct identity being given; a wrong transfer must be correctable by the system;
  controls: strong one-shot inference, recomputation from the same history, the learned part switched off with
  its evidence kept, whole-life resources; how much of the common interface is supplied must be written down.
  (Round 22 tried this and, by its own account, did not meet the "no identity given" condition.)

---

## 3. Study `living_learning_deduction_2026_09_23` (round 21) — the system's own deductions turned into capacity, and kept valid when their premises change

Report title: "içsel çıkarımın yaşayan hesap kapasitesine dönüşmesi" (`REPORT_TR.md:1`).

### (1) Question

With no new label from outside, can a system keep intermediate results it has proved by its own computation
and then answer later, never-seen questions equally correctly at lower service cost — and can those results be
taken safely out of use when what they rest on changes? (`REPORT_TR.md:7`)

The report presents this as the logical counterpart of the owner's suggestion that events in the mind,
cross-linked, make new ideas easier — and adds at once: "bu küçük sistemde kavram, duygu, bilinç, beyin veya
genel yapay zekâ yoktur. Sonuç, önerinin yalnızca bir hesap biçimindeki sınamasıdır" (`REPORT_TR.md:9`).

### (2) Mechanism / state representation actually used

Symbolic propositional logic: clauses, a DPLL solver, a separate proof checker. No statistics, no weights.
Code references are to `research/living_learning_deduction_2026_09_23/deduction.py` unless `check_proof.py` is
named.

**The world as the learner sees it.** A *base* — a dictionary clause-id → clause, the id being a hash of the
clause's content (`cid`, `:31-32`; "identity is content", `:133`) — and a *query*: three assumption literals
over six public variables. The task: is base ∧ assumptions satisfiable?

**What is stored** (`Learner`, `:171-177`): a list `lemmas` of records (`:235-238`)

```
{ id, clause, dependencies, assumptions, proof, used_clauses, parent_dependencies }
```

`clause` is the negation of a minimal unsatisfiable set of assumptions; `dependencies` are the sorted ids of the
**base** clauses the proof rests on, with any lemma used in the proof expanded recursively to its own base
clauses (`:226-232`); `proof` is the search tree; `used_clauses` the clauses it cites. Plus the signature of the
last base and five cost counters.

**The update rule — one query** (`Learner.query`, `:188-247`):

1. *Which results may be used now* (`:196-200`). A lemma is **active** iff every id in its `dependencies` is in
   the current base. Inactive lemmas stay stored (**dormant**). (`blind` uses all of them; `erase` deletes all
   of them when the base signature changes, `:192-194`.)
2. *Answer first* (`:201-204`). Run DPLL (`solve`, `:35-103`: unit propagation; branch on the first variable of
   the shortest open clause, negative branch first) on active lemma clauses + base under the assumptions. The
   verdict is the service; its counted operations are the service work.
3. *Then learn from the same computation* — only if the verdict was UNSAT and no active lemma already covers
   it (`:206-242`):
   a. minimise the assumptions by dropping one at a time and re-solving (`:215-223`) — acquisition work;
   b. check the UNSAT tree of the minimal core with the **separate** checker `check_unsat`
      (`check_proof.py:33-149`), which validates every unit reason, every conflict leaf and both children of
      every branch, and returns exactly the clauses used (`:224-225`);
   c. flatten the used clauses to base-clause ids (`:226-232`);
   d. **discharge**: store the clause `¬a₁ ∨ … ∨ ¬a_k` (`:233`) with its record, and charge the serialized bytes
      of the whole record as write work (`:240`).

"Learners answer first, then may construct a lemma from the same query. No oracle feedback trains them"
(`PROTOCOL.md:32`).

The two logical steps used (`MATHEMATICS_AND_LITERATURE_TR.md:21-33`):

```
F ∧ a₁ ∧ … ∧ a_k  unsatisfiable    ⟹    F ⊨ (¬a₁ ∨ … ∨ ¬a_k)        discharge of local assumptions
D_l ⊆ F,  D_l ⊨ l,  D_l ⊆ F'        ⟹    F' ⊨ l                       validity carried by the supports
```

**Seven methods** (`:19`; `run_world`, `:250-327`; `PROTOCOL.md:25-30` and the amendment at `:5`):

1. `fresh` — solve every query from scratch on the current base.
2. `scoped` — the rule above.
3. `blind` — same acquisition, but every stored lemma is used whatever its supports. Deliberately unsound; "not
   a deployable candidate".
4. `erase` — as `scoped`, but all lemmas are deleted whenever the base changes.
5. `replay` — before every answer, rebuild a `scoped` learner from the complete chronological history
   (`:269-281`). Same-evidence control.
6. `full_table` — whenever the base changes, solve all 64 complete public assignments, then answer by scanning
   the satisfiable rows (`:283-318`).
7. `full_table_cached` — keep each table under the exact set of clause ids and reuse it on return; pay a version
   check per query. Added by the audit amendment as a stronger control.

**Given** (`PROTOCOL.md:44`): "SAT syntax, solver, checker, implication-based lemma language, query interface,
minimizer, retention policy and resource meter are supplied by the researcher. The system does not invent
these."

### (3) Experimental design

- **Worlds** (`world`, `:106-142`; `PROTOCOL.md:19-21`): 24 seeds. Six public variables and four *gadgets*.
  Each gadget is the unsatisfiable three-pigeons/two-holes clause set (9 clauses on six private variables) with
  a two-literal **guard** over public variables appended to every clause: 36 four-literal clauses, 30 variables.
  A gadget can be satisfied iff its guard is true, so the public meaning of a base is four two-literal clauses —
  which the learner is never told: "The learner receives only flat clauses with immutable content identities,
  not guards, pigeonholes or target answers."
- **Life**: four phases of 24 queries — base; same base; **changed base** (one gadget's guard with both signs
  reversed); **return** to the first base. A query is three distinct public variables with random signs.
  96 queries per life, 2,304 per method; in the first, second and fourth phase nothing new arrives from outside.
- **Truth reference**: independent enumeration of each gadget's private assignments for every public assignment
  (`projected_truth`, `:145-164`; a second, differently factored one in `verify_deduction.py:26-44`).
- **Work meter** (`PROTOCOL.md:36`): inspected literals and clause headers, assignments, branch and proof-node
  operations, checked dependencies, and written certificate/history **bytes** (table construction: seven scalar
  writes per row) — "heterogeneous counts form an explicit virtual proxy". Two totals: service only, and whole
  life including acquisition, checking, dependency scans and writes.
- **Amendment** (`PROTOCOL.md:5`; `REPORT_TR.md:27`): after the six-method run an audit found that `replay`'s
  memory diagnostic counted a base not yet seen (no behaviour changed); it was corrected and
  `full_table_cached` was added. The reported numbers are the rerun after the amendment — "amendment sonrası
  yeniden çalıştırmadır, önceden kaydedilmiş bağımsız bir tekrar değildir". The earlier run is preserved in
  `results/initial_run/`.
- **Statistics**: per-method counts and means; paired per-life differences against `scoped`
  (`analyze_results.py:30-47`); no tests.

### (4) Results (the study's numbers) and decision labels

(`REPORT_TR.md:31-39`; `results/summary.json`)

| Method | correct / 2,304 | mean service work per query | mean whole-life work | what it shows |
|---|---:|---:|---:|---|
| `fresh` | 2,304 | 1,166.3 | 111,967.5 | always right; every query solved again |
| `scoped` | 2,304 | 440.3 | 111,812.3 | right and checked |
| `blind` | **2,218** | 378.1 | 97,554.1 | looks cheaper; **86 wrong answers** |
| `erase` | 2,304 | 612.4 | 156,070.3 | safe; deleting and re-acquiring is expensive |
| `replay` | 2,304 | 440.3 | 6,560,128.5 | identical to `scoped` in answers and lemma state; rebuilding is expensive |
| `full_table` | 2,304 | 28.2 | 128,739.6 | very fast service; recompiles at every change |
| `full_table_cached` | 2,304 | 28.2 | **92,136.3** | cheapest strong control |

- **Service.** 1,755 of the 2,304 queries are first occurrences in their life. `scoped` served more cheaply than
  `fresh` on 1,702 points (1,261 of them first occurrences) and more expensively on 474
  (`summary.json › paired[0]`). Answered correctly within 500 operations: `fresh` 669, `scoped` 1,788; within
  2,000: 1,973 against 2,247. Phase means of `scoped` service: 718.2 / 303.3 / 532.8 (changed base) / 207.0
  (return); of `erase`: 718.2 / 303.3 / 714.1 / 714.0.
- **Whole life**, mean paired difference `scoped` − other (`summary.json › paired`): − 155.2 against `fresh`
  (`scoped` cheaper in 15 of 24 lives); − 44,258.0 against `erase` (24/24); − 16,927.3 against `full_table`
  (18/24); **+ 19,676** against `full_table_cached` (5/24); `replay` costs 58.7 × `scoped`.
- **Composition of `scoped`'s whole-life work** (mean per life): service 42,273.4; acquisition 30,446.7;
  certificate writes 21,780.7; dependency scans 14,752.9; proof checking 2,558.7.
- **Unsoundness measured, not asserted** (`verification.json › recorded`): `blind` used an invalid active lemma
  1,705 times and created 5 invalid new lemmas; every accepted `scoped`/`erase` lemma and every active `scoped`
  lemma at every use was true in the independent reference (33,940 checks including controls).
- **`scoped` ≡ `replay`** in answer, service work, active lemma set, new lemma and stored count on all 2,304
  pairs: "kalıcılığın aynı tarihsellikte yeniden çıkarımdan farklı hesap maliyeti yaratabildiğini gösterir;
  bilgi üstünlüğü göstermez" (`REPORT_TR.md:43`).
- **Stored state** (mean of the per-life maximum, serialized bytes): `scoped` 22,973; `full_table_cached`
  2,797; `full_table` 387; `replay` 29,420.
- **Verification** (`REPORT_TR.md:63`): 1,000 random formulas against enumeration (198 satisfying witnesses, 802
  refutation proofs); 16,128 recorded answers; 815 certificates re-checked; 2,304 replay pairs; 18 fresh-process
  continuations; checker self-test: 5 valid proofs accepted, 14 corrupted ones rejected.

**[ANALYST] re-aggregations of the stored raw records** (`g6_deduction_checks.json`, `g6_small_numbers.json`):

- **185 lemmas in 24 lives**: 5–12 per life, mean 7.71 — confirming the correction in round 22's report.
- **What the lemmas are.** 183 of 185 are two-literal clauses over public variables and 2 are unit clauses;
  **128 of 185 are exactly a planted guard** of the base in force. The system's own proofs recover, when a query
  runs into a gadget, the compact public meaning of that 9-clause private structure; the other 57 are further
  consequences (combinations of guards). A lemma rests on 24.3 base clauses on average (9–36).
- **When they are formed**: 80 in the first phase, 37 in the second, 64 in the changed base, 4 on return.
- **Dormancy works both ways.** At the change of base 81 lemmas (3.4 per life; in 22 of 24 lives) became
  dormant; on return exactly these 81 were active again. Active / stored before a query: 3.0 / 6.4 in the
  changed phase, 6.1 / 7.6 on return.
- **`blind`'s 86 errors are of one kind** — "unsatisfiable" said of a satisfiable query — 50 in the changed
  base and **36 after the return** to the original base (results proved during the changed phase, or derived
  from stale ones, do not hold in the original world). 22 of 24 lives have at least one.
- **Where the service saving comes from.** Mean service `fresh` → `scoped`: on satisfiable queries 1,051.8 →
  455.0 (n = 1,717), on unsatisfiable ones 1,501.2 → 397.6 (n = 587). Results that are formed only from
  failures also shorten the search on queries that succeed. `scoped` was more expensive than `fresh` on 418
  satisfiable and 56 unsatisfiable points.
- **The whole-life verdicts depend on the price of a written byte.** `scoped` − `fresh` per life: mean − 155.2,
  median − 9,684.5, standard deviation 26,094, range − 54,142 … + 47,644; a sign test on 15 of 24 gives a
  two-sided p ≈ 0.31. Re-pricing the certificate bytes by a factor *w*: *w* = 0 → − 21,936 (cheaper in 19/24);
  *w* = 0.5 → − 11,046 (17/24); *w* = 1, the study's unit → − 155 (15/24); *w* = 2 → + 21,626 (7/24). Against
  `full_table_cached`: *w* = 1 → + 19,676 (5/24); *w* = 0 → − 2,105 (13/24).

**Decision labels** (`REPORT_TR.md:49-57`):

| Hypothesis | Study's verdict | Evidence |
|---|---|---|
| **H21.1** internally verified intermediate results reduce the service computation of never-seen queries | **supported** ("bu yapay mantık ailesinde") | 440.3 vs 1,166.3; 1,261 novel points cheaper |
| **H21.2** the whole life including acquisition and checking is cheaper than solving afresh | **partially supported** — "ortalamada azıcık daha ucuzdu … bir dağılım ve ölçüm birimine özgüdür" | − 155.2; 15/24 |
| **H21.3** tracking supports prevents wrong reuse after premises are removed | **supported** — "genel yeterli politika teoremi değildir" | `scoped` 0 wrong, `blind` 86 |
| **H21.4** scoped persistence reproduces recomputation from the same history at a different cost | **supported** | 2,304 identical pairs; 58.7 × |
| **H21.5** experience-dependent acquisition beats exact compilation of the whole query interface | **rejected in scope** — "Bu, genel derlemelerin her zaman üstün olduğu anlamına gelmez" | 111,812 vs 92,136 |

"Böylece 'kalıcı lemma, her güçlü derlemeyi aşar' iddiası reddedildi" (`REPORT_TR.md:41`).

### (5) Laws / invariants claimed (none presented as new: "yeni keşfedilmiş genel öğrenme kuramı diye sunulmuyor", `MATHEMATICS_AND_LITERATURE_TR.md:5`)

- **I1 — internal computation adds no information about the world [EXACT].** World `W`, outside evidence so far
  `H`, internal randomness `R` independent of `W` given `H`; any internal result or retained state
  `Z = f(H, R)` gives the chain `W → H → Z`, so `I(W;Z) ≤ I(W;H)` and `I(W;Z | H) = 0` (`:9-13`). *Scope*: no
  side channel, no interaction with the environment, no randomness correlated with `W`. It does **not** say that
  internal computation is not learning (`:13`).
- **I2 — under a finite budget, entailed results change what can be reached [EXACT definition + EMP].**
  `K_B(F, L)` = the queries answered *correctly* within `B` work from base `F` with retained results `L`. If
  `F ⊨ L` then `Mod(F ∧ L) = Mod(F)` — the same possible worlds — and still `K_B(F, L) ≠ K_B(F, ∅)` is possible
  (`:15`). *Measured*: 669 → 1,788 of 2,304 within 500 operations. Not monotone: more true results need not
  enlarge `K_B`, because the same algorithm also scans the useless ones (`:17`) — consistent with the 474 dearer
  points, not tested as such.
- **I3 — the correct way to turn a local result into a global one [EXACT].** Discharge, as above (`:21-25`).
  *Counterexample to the naive version* (`:27`): `F = (a ∨ b)`, assumption `¬a`; keeping the locally derived `b`
  as a fact makes the later query `¬b` look unsatisfiable although `a = 1, b = 0` satisfies it; the safe result
  is `a ∨ b`. Second example: `F = (a ∨ b) ∧ (¬b ∨ c)`; under `¬a`, `c` follows; the global result is `a ∨ c`.
  The checker's self-test contains it (`check_proof.py:212-217`).
- **I4 — validity travels with the supports [EXACT + EMP].** Monotonicity, as above (`:31`). The converse
  fails — another proof may keep the result true after a support is gone — so the test is "**yeterli, gerekli
  olmayan** tutucu bir ölçüttür" and not shown to be the cheapest revalidation (`:33`). Also: "Temel maddeler
  dünyayı yanlış anlatıyorsa, kusursuz mantık denetimi dünya doğruluğu sağlamaz." *Necessity shown* by `blind`.
- **I5 — not every proof check gives the contract a kept result needs [EXACT example].** Steps that preserve
  only satisfiability (RAT) are harmless for one verdict and harmful later: add `a` to an empty `F`, then ask
  `¬a` (`:37`, after Wetzler, Heule & Hunt). A kept result must preserve the *set of models*.
- **I6 — symmetry bound for learning a mapping [EXACT; stated for the next round].** If two candidate mappings
  give every accessible source history the same likelihood, source observations cannot separate them; a
  discriminating experience, an extra structural assumption or target feedback is needed. Full Bayes with the
  right model minimises expected proper loss; a learned transformation can only approach that under bounded
  computation (`:41-43`).
- **I7 — crossing classes cannot all be single subtrees of one tree [EXACT].** `{1,2}/{3,4}` and
  `{1,3}/{2,4}`: the leaf sets under two nodes of a rooted tree are nested or disjoint; `{1,2}` and `{1,3}`
  cross. "Çapraz bağ, çoklu temsil veya kopyalama gerekir; bunlardan hangisinin ekonomik olduğu ayrı sorudur"
  (`HYPOTHESES_TR.md:35-37`). Round 22 sharpens it (§4, T2).

### (6) Limits

**Admitted by the study.**
- Everything that builds, checks and retains a lemma is supplied (`PROTOCOL.md:44`).
- In the stationary phases no outside label arrives; the changed base *is* outside information
  (`REPORT_TR.md:43`).
- Virtual work with mixed units; the service thresholds are retrospective counts, "**not** a trained system
  actually learning under interrupted budgets" (`PROTOCOL.md:36`).
- The final numbers are a post-amendment rerun (`REPORT_TR.md:27`).
- The support test is sufficient, not necessary; the checker is application-level, "Coq/Lean gibi bir çekirdekte
  makineyle doğrulanmış metateori değildir" (`MATHEMATICS…:35`).
- `full_table_cached` exists because the public interface has six variables: "genel problemlerde tam tabloyu
  kurmak mümkün veya ekonomik olmayabilir" (`REPORT_TR.md:41`).
- Prior art: explanation-based learning and its utility problem (Minton 1988), CDCL, proof checking (DRAT-trim),
  knowledge compilation (Darwiche & Marquis): "sonuç yeni temel ilke sahiplenmesi değildir"
  (`REPORT_TR.md:59`).
- The wording "185 lemma/yaşam ortalaması" in the report's table (`REPORT_TR.md:34`) is wrong; corrected in
  round 22's report.

**Limits I see.**
- **[ANALYST] Neither "consolidation pays" nor "it does not" is established.** Both whole-life verdicts (H21.2,
  H21.5) turn on one exchange rate — a written byte equals one operation — and change sign at *w* = 0 or
  *w* = 2; and the mean difference to `fresh` is 0.14 % of the total with a standard deviation 168 times its
  size. What *is* established is the service effect (H21.1) and the validity effect (H21.3).
- **Results are formed only from failures** (unsatisfiable queries, 25.5 % of all); nothing is kept from a
  success.
- **The store only grows.** There is no rule for which true result is worth keeping (H21.13, H21.14 untested).
  The record keeps the whole proof and all cited clauses: the lemma store is eight times the size of two cached
  truth tables.
- **Validity is re-established from scratch at every query** — all supports of all stored lemmas are scanned
  (13 % of the life's work); nothing is invalidated incrementally.
- **Noticing the change is free.** The learner is handed the current base with every query; detecting that the
  world changed is not part of the test.
- **One kind of change** (a guard reversed, then restored), whole clauses with content identities, no noisy or
  graded premise.
- **No new symbol.** The lemma language states new relations among given variables; it cannot name a new one.
- Only 160 distinct queries exist (`C(6,3) · 2³`); "novel" means first in that life.

### (7) Implications for a core in which learning and running are one process

- **Mechanism to reuse — "answer first, then consolidate from the same computation"** (`:201-242`). This is the
  one place in this group where learning happens *inside* running: the search that produced the answer is
  minimised, checked and stored, and the very next query runs on the changed structure. In a symbolic setting,
  with batch size one, it is the closest prior instance of the target.
- **Mechanism to reuse — a construction that carries its certificate and its scope**:
  `{clause, proof, used clauses, supports}`, active iff the supports are present, dormant otherwise. For
  *derived* content this is a working answer to "measure how real a construction is":
  real = has an independently checked derivation ∧ what it rests on still holds.
- **Mechanism to reuse — a checker independent of the generator**, with its own work meter and tamper tests.
- **Mechanism to reuse — content identities for primitives.** Because "identity is content", the return of an
  old world reactivates everything that depended on it without any search: 81 of 81.
- **[ANALYST] Observation — concepts formed on demand.** 128 of 185 lemmas are the hidden guard: a compact
  public relation standing for a private nine-clause structure, formed only when a query ran into it. The rule
  "a failure, minimised and discharged, becomes a reusable relation" is a concept-formation rule with a proof
  attached.
- **Requirement — a retained result must carry (a) its scope, with the local assumptions discharged into its
  statement (I3), and (b) its supports (I4).** Without (b): 86 wrong answers, 36 of them after the world had
  already gone back.
- **Requirement — dormancy, not deletion.** `erase`: + 44,258 work per life, dearer in 24 of 24 lives.
- **Requirement — derived content is not evidence** (I1). What imagination or deduction adds is access under a
  budget (I2); it must be accounted net of acquisition, checking, storage and lookup over the whole life.
- **Warning — where the question interface is small and declared, compiling it is at least as good.** Under the
  study's unit the exact table with a version key beat the lemma store by 19,676 per life, and it serves a
  query with 16 times less work (28.2 against 440.3). **[ANALYST]** The whole-life gap closes if written bytes
  are free; the service gap does not. Lemmas are the option when the interface cannot be enumerated — and that
  case was not tested.
- **Warning — more true results is not more capacity** (474 dearer points; no selection rule).
- **Open questions**: which true results to keep and which to let go (H21.13/14); graded or partial support; a
  cheaper test that is necessary as well as sufficient; noticing that a premise changed; naming new symbols.

### 3B. `HYPOTHESES_TR.md` — the hypothesis map that re-opens the research field

Sixteen hypotheses in four groups; "Bir hipotezi yazmış olmak, onu doğrulamak değildir" (`:5`). Status as of the
end of this group:

| # | Hypothesis (short) | Discriminating test named by the note | Status |
|---|---|---|---|
| H21.1 | deduction becomes a lasting computational advantage | compare with a solver reprocessing the same evidence; if the gain is only on repeats the claim narrows | **tested, supported** |
| H21.2 | acquisition pays for itself over the life | count learning, checking, storing and use | **tested, partially supported** |
| H21.3 | validity is carried with content | support-checked vs unconditional vs erase-all | **tested, supported** |
| H21.4 | persistence may be avoided recomputation, not more information | exact equality with chronological replay | **tested, supported** |
| H21.5 | experience-dependent acquisition is cheaper than complete compilation up front | full-interface table, with and without version cache | **tested, rejected in scope** |
| H21.6 | shared coordinates make the next update cheaper (an acquired transformation) | wrong-transformation transfer; symmetry control; full hierarchical inference from the same source history | not run |
| H21.7 | a learned update operation is more than a new initial value | keep content and representation, reset only the update operation; if only a step size or prior was learned, say so | not run |
| H21.8 | the learner's own choices can distort its error information | same data tape for all rules vs self-chosen data; report cost of selection and the unvisited regions | not run |
| H21.9 | the need for a distinction can come from later learning, not from the present answer | evaluate output-equivalence and update-equivalence separately | not run here (mathematical basis in earlier sufficiency results) |
| H21.10 | acquiring a mapping between two independently learned domains can build a new task, without a ready common interface | strong one-shot synthesis on the joint history; "Doğru eşleme açıkça verilirse hipotez sınanmış sayılmaz" | attempted in round 22 — **not met** (bridge meaning supplied) |
| H21.11 | joining need not require one universal tree of meaning | crossing task partitions: one tree, several views, direct query; count copies | examined in round 22 — single-node version **true**, "one tree must lose a task" **false** |
| H21.12 | re-reading the past need not erase knowledge | same event record, changed use mapping; ask old and new queries alternately | not run |
| H21.13 | usable capacity does not grow monotonically with the number of true results | count the audit and search cost of new true results | not run |
| H21.14 | forgetting affects correctness and accessibility separately | at the same byte limit, delete a reproducible result vs delete source evidence; no hidden copy elsewhere | not run (only the narrow `erase` result) |
| H21.15 | self-progress is not unlimited production of knowledge | I1; "ampirik AGI karşıtı sonuç değil" | mathematical |
| H21.16 | the general principle may not be a single update equation | show with concrete pairs why achieving one obligation does not give another | organising hypothesis: "kanıtlanmış zorunlu dört modüllü mimari değildir" |

H21.16 names four obligations — acquiring the right content, deriving new computation from the evidence in
hand, keeping validity, choosing resources — and leaves open that one mechanism may meet them all (`:49`).
The selection rule for what comes next (`:53`): no automatic continuation into another SAT variant; first
H21.10 and H21.11 together, with the strong joint-history synthesis, the wrong-mapping control, the
mapping-switched-off control, the whole-life budget and old-skill queries written down before the results.

### 3C. `MATHEMATICS_AND_LITERATURE_TR.md`

Its propositions are I1–I6 above. Its literature section (`:45-53`) states what was actually read: Minton 1988
(full text; the utility of learned search knowledge must exceed the cost of applying and matching it — "Bu
turun ara sonuçların toplam maliyeti sorusu doğrudan önceki çalışmayla eşleşir"), Darwiche & Marquis (abstract
page only), Wetzler–Heule–Hunt 2014 (section 2), Baxter 2000 (opened; recorded for H21.6–7, not used as support),
GRASP (access failed; not used). "Bu tarama kapsamlı yenilik taraması değildir." The round's stated value:
separating the acquisition of information from access to computation, and correctness from lasting validity,
"aynı denetlenebilir kayıtta".

---

## 4. Study `living_learning_crossmap_2026_09_23` (round 22) — joining two bodies of knowledge: a success, an unnecessary mechanism, and the limit of a false claim

Report title: "iki bilgi alanının bağlanması: başarı, gereksiz mekanizma ve yanlış iddia sınırı"
(`REPORT_TR.md:1`). The report's own summary of what the round is worth (`:5`): three distinctions were
confirmed — "deneyimle verilen eşleşmeyi kullanmak ortak anlamı keşfetmek değildir; bütün adayları kalıcı
saklamak aynı bilgiyi daha küçük durumda tutmaktan üstün olmayabilir; kesişen kavramlar tek bir ağacın her türlü
kullanımını imkânsız kılmaz."

### (1) Question

Can two relational domains that were learned separately be joined by experience so that a later task that was
never taught directly becomes computable — and does the same stored event need several crossing organisations
instead of one nested tree? (`PROTOCOL.md:9`)

### (2) Mechanism / state representation actually used

Symbolic: tables and permutations. The valid engine is `research/living_learning_crossmap_2026_09_23/corrected.py`
(all references below unless another file is named). It reuses, unchanged, only the world generator of the
archived first engine (`:27-32`, loading `results/invalid_initial_run/crossmap.py:26-58`).

**What the learner receives** (`observation`, `:35-40`; `packets`, `:251-258`): 24 source records
`[domain, 'next' | 'group', symbol, value]` — per domain six successor facts and six category labels — and,
before each query phase, new **bridge** pairs `[x, y]`: "symbol `x` of domain X and symbol `y` of domain Y name
the same object". Targets and latent identities stay in the evaluator.

**What is stored** (`Learner.__init__`, `:61-86`): `source` (the 24 records), `tables` (parsed successor and
category tables), `bridges`, `known[x] = y` and `inverse[y] = x` (− 1 when unknown), `maps` (a list of candidate
bijections, for the methods that keep one), an `enabled` flag, nine cost counters, byte counters.

**Update on a bridge** (`ingest`, `:123-138`): write the pair into `known`, `inverse`, `bridges`; a method that
keeps a candidate list filters it by the new pair (`:131-136`), or the first time enumerates all 720
permutations against the bridges seen (`enumerate`, `:108-121`).

**Answer** (`query`, `:179-211`): build a histogram of answers over the hypotheses in use; **if exactly one
answer has a non-zero count, give it; otherwise abstain** (`:205-207`). Three kinds of question (`outcome`,
`:43-58`): `xy` — the Y-name of the X-successor of `x`, `π(next_X(x))`; `yx` — the X-name of the Y-successor of
`y`, `π⁻¹(next_Y(y))`; `cat` — the pair (X-category of `x`, Y-category of `π(x)`), four classes.

**Seven methods** (`:17`; `AMENDMENT.md:15`):

| Method | Keeps | Answers by |
|---|---|---|
| `separate` | all records | never composes across domains → always abstains (deliberate non-use control) |
| `persistent` | the list of **all** bijections consistent with the bridges; filtered at each new bridge | counting over the list |
| `recompute` | the raw records only | re-parsing the source and re-enumerating all 720 maps before **every** query, no cache |
| `completion` | only the observed pairs | the closed form (`analytic_histogram`, `:148-177`): if the needed pair is known, its image; otherwise each still-unmatched symbol of the other side once. No map is ever listed |
| `point` | the list | the *first* surviving map, treated as certain (overconfidence control) |
| `disabled` | the list, as `persistent` | nothing in phases 0–2; use switched on in phase 3 (`:182-185`) |
| `oracle` | the true map, given (six entries, counted as a gift) | the true map (ceiling) |

The model behind `persistent`, `recompute` and `completion`: every bijection consistent with the bridges,
equally weighted — "o varsayım sınıfında tam hesap; dünyanın üretim yasasını kullanan Bayes hesabı olduğu iddia
edilmiyor" (`REPORT_TR.md:45`). Its closed form (`MATHEMATICS_AND_NEXT_TR.md:7-9`):

```
|Ω(M)| = (n − k)!          P( π(x) = y | M ) = 1 / (n − k)     for x, y not yet matched     (n objects, k pairs seen)
```

**Scores** (`scores`, `:230-237`), exact fractions: Brier `Σ_i (n_i/N − 1[i = answer])²`; expected accuracy of
a forced choice with fair tie-breaking `1[answer among the top] / #top`.

**Given vs acquired** (`REPORT_TR.md:11`): the symbols, the finite relations, the *meaning of a bridge*
(identity), the bijection assumption, the composition operation, the candidate language and the task format are
given. The learner does not know the whole mapping at the start; that is all it acquires. The stated aim — to
remove the ready-made common interface of round 12 — "**tam karşılanmadı**".

### (3) Experimental design

- **Worlds** (`results/invalid_initial_run/crossmap.py:26-58`): 24 seeds, six latent objects. X names are a
  random permutation and the X relation a six-cycle; Y names an independent permutation and the Y relation three
  two-cycles. Categories in latent ids: X `{0,1,2}/{3,4,5}`, Y `{0,2,4}/{1,3,5}` — they cross.
- **Life**: 24 source records; then four phases of six questions. Phase 0 (two bridges known): `xy` for every
  `x`; phase 1 (three): `yx` for every `y`; phase 2 (four): `cat` for every `x`; phase 3 (five): `xy` again.
  Surviving candidates 24 / 6 / 2 / 1. 576 questions per method — 432 new, 144 repeats; 240 need a pair that was
  never shown (`REPORT_TR.md:13-15,37`).
- 7 methods × 24 lives = 168 paths, 4,032 answers: "168 yol ve 4.032 cevap noktası, 168 bağımsız dünya değildir".
- **Cost** (`AMENDMENT.md:27,39`): nine scalar counters — input reads, source parsing, candidate slots, pair
  tests, query reads, answer counting, state writes, transient writes, mode changes — summed to whole-life
  work, including setup and ingestion; setup + acquisition + service must telescope to the final totals in every
  path. Written bytes are reported separately, with sensitivity weights 0 / 1 / 4.
- **Structural audit, separate from the learners** (`tree_audit.py`): all 945 rooted binary trees on six
  labelled leaves against the four category sets.
- **Scope audit, after the results** (`model_scope_audit.py`): what a learner that knew the generator's law
  could infer.
- **History of the experiment** (`AMENDMENT.md`): the first engine was invalid; the analysis of the first
  correction was invalid again; both are archived. "This amendment is **post-result repair**, not
  preregistration" (`:5`).
- **Statistics**: counts, exact rational scores, paired differences; no tests.

### (4) Results (the study's numbers) and decision labels

(`REPORT_TR.md:19-27`; `results/summary_corrected.json`)

| Method | certain and right | abstained | certain and wrong | mean whole-life work |
|---|---:|---:|---:|---:|
| `separate` | 0 | 576 | 0 | 394 |
| `persistent` | 374 | 202 | 0 | 6,612 |
| `recompute` | 374 | 202 | 0 | 130,170 |
| `completion` | 374 | 202 | 0 | **694** |
| `point` | 450 | 0 | **126** | 6,009 |
| `disabled` | 144 | 432 | 0 | 5,840 |
| `oracle` | 576 | 0 | 0 | 625 |

- **Same beliefs, three costs.** `persistent`, `recompute` and `completion` return the identical probability
  distribution, as exact fractions, on all 576 questions (1,728 pairwise equalities,
  `verification_corrected.json`). `persistent` is cheaper than `recompute` in 24 of 24 lives (19.7 ×).
  `completion` is cheaper than `persistent` in 24 of 24 lives (**9.53 ×**) and does no more work in *any* of
  the nine counter categories (`final_audit.json › completion_scalar_category_dominance_lives: 24`); with byte
  weights 0 / 1 / 4: 694 / 1,255 / 2,938 against 6,612 / 7,639 / 10,720. "Bu negatif sonuç, bu kategorilere
  farklı negatif olmayan ağırlıklar vermekle tersine dönmez" (`REPORT_TR.md:31`).
- The conclusion drawn is narrow: not "learning is unnecessary" — `completion` also changes its persistent state
  with experience (it keeps the pairs) — but "**bu sorgu ailesinde öğrenilmiş bütün olasılık listesini saklamanın
  gerekli olmadığıdır**"; and its inference rule was supplied, not invented (`REPORT_TR.md:33`).
- **What generalised and what was only reused** (`REPORT_TR.md:37`): on the 432 new questions the three
  uncertainty-carrying methods were certain and right on 230 and abstained on 202. Of the 240 questions whose
  needed pair was never shown, only **38** could be answered with certainty; the other 202 stayed undetermined.
- **Committing vs keeping uncertainty** (`REPORT_TR.md:39`): Brier 0.237847 for the uncertainty-carrying
  methods, 0.4375 for `point`. `point` has more certain-and-right answers (450 against 374) at the price of 126
  certain-and-wrong ones; expected forced-choice accuracy 76.215 % against 78.125 %. "Kalibrasyon/kesin yanlış
  riskinin korunması, en yüksek zorunlu seçim doğruluğuyla aynı sonuç değildir."
- **The use switch** (`REPORT_TR.md:41`): `disabled` keeps evidence and candidates; when use is switched on for
  the last six questions, 144 of 144 are right. "Kullanım yolunun nedensel etkisini gösterir; iç yapının
  yanlışlığını kendiliğinden düzeltme mekanizması değildir."
- **The learner's uncertainty is not the world's** (`REPORT_TR.md:45-47`; `model_scope_audit.json`): a learner
  that knew the generator's category/cycle law would have 6 candidates before any bridge, 2 after one bridge in
  all 24 lives, 1 after two bridges in 20 lives (2 in four), 1 after three in all — against 24 / 6 / 2 / 1 after
  two / three / four / five bridges for the main learner. This uses information the methods did not have; it is
  there to stop the candidate counts being read as "no system could learn from less".
- **Crossing categories and one tree** (`REPORT_TR.md:53-61`; `tree_audit.json`): no tree of the 945 shows all
  four classes as single nodes — at most two. If the union of nodes may be read, every class is shown without
  copying an object; the minimum is six node references, reached by 2 trees (reference histogram 6:2, 7:16,
  8:57, 9:112, 10:224, 11:336, 12:198). The stronger claim "one tree must lose one of the tasks" is **false**.
- **Verification** (`REPORT_TR.md:65-67`; `final_audit.json`): 4,032 answers by an independent
  free-coordinate count; 3,024 probability scores; 672 state snapshots; 4,032 retained local answers; cost
  telescoping in 168 paths; 12 renamings × 6 methods = 72 paths with unchanged probabilities; 84 fresh-process
  loads, 63 of them with a non-empty continuation; a corrupted candidate and a corrupted bridge rejected;
  3,780 tree covers recomputed.

**[ANALYST] re-aggregations of the stored raw records** (`g6_crossmap_checks.json`):

- **[DESIGN-CONSTANT]** The cost vector of every method is identical in all 24 lives (394 / 6,612 / 130,170 /
  694 / 6,009 / 5,840 / 625; `life_work_min = life_work_max`). The cost comparison is one deterministic fact of
  the design, repeated under 24 renamings — not 24 observations. The only quantity that varies between lives is
  the number of category coincidences in phase 2: 17 lives have 15 certain answers, 7 lives have 17.
- The 38 certain answers on never-shown pairs are 24 in phase 3 (the sixth pair follows from five by the
  bijection assumption) and 14 category questions in phase 2 in which both remaining candidates fall in the same
  category.
- `point`: 450 right in fact; **439.0 expected under the learner's own model — exactly the expected
  forced-choice score of the uncertainty-carrying methods (439)**. The eleven extra right answers are a
  fluctuation. In expectation, committing buys nothing in forced-choice accuracy and costs 126 confident errors
  (at least one in every one of the 24 lives) and a Brier score of 0.4375 against 0.2378.

**Decision labels** — the hypotheses as *re-stated after the defects and results were seen* (`AMENDMENT.md:35`):

| Hypothesis (corrected form) | Verdict | Evidence |
|---|---|---|
| **H22.1** bridge experience enables composition, against deliberate non-use | holds in that weak form; **not** discovery of a common meaning | 374 against 0; bridge meaning supplied |
| **H22.2** a cached candidate set is cheaper than explicit rebuilding, with identical predictions | **supported** | 6,612 against 130,170; 576/576 identical |
| **H22.3** committing under ambiguity gives confident systematic errors; switching use off removes them | **supported**, with the caveat "disabling all computation prevents both wrong and useful answers, so it alone is not a solution" | 126 confident errors; `disabled` 0 right while off |
| **H22.4** one class = one tree node is impossible for crossing partitions | **true in that restricted form**; "one fixed hierarchy loses at least one task" **refuted** | 945 trees; union readout, six references |
| **H22.5** the persistent candidate mechanism beats a same-information direct procedure | **rejected** | `completion`: identical distributions at 694 against 6,612 |
| original aim: no common interface supplied | **not met** | `REPORT_TR.md:11` |

### (5) Laws / invariants claimed

- **T1 — closed form of the candidate list [EXACT].** With `n` objects, `k` observed pairs and no other
  constraint, `|Ω(M)| = (n − k)!`; an observed `x` has a certain image; an unobserved `x` goes to each unused
  `y` in `(n − k − 1)!` of the candidates, so `P(π(x) = y | M) = 1/(n − k)`. Every question that depends on
  **one** entry of `π` or `π⁻¹` can therefore be answered without listing bijections; a category question sums
  the category counts of the remaining `y`'s (`MATHEMATICS_AND_NEXT_TR.md:7-11`). "Bu sonuç bir yeni öğrenme
  ilkesi değil, seçili aday sınıfının simetrisi ve sorgunun biçimidir" (`:13`). *Consequence for state*: the
  record of pairs together with the stipulated model regenerates every joint constraint; the list is a cache.
- **T1′ — marginals alone do not suffice for coupled questions [EXACT witness].** Two unmatched `x` and two
  unmatched `y`: two legal maps; each pairing has marginal 1/2, yet both `x` going to the same `y` is
  impossible, while multiplying the marginals gives that collision probability 1/2 (`:13`;
  `model_scope_audit.json › joint_query_counterexample`). The compact pair record can recompute a coupled
  question; the single-reference routine does not do so by itself.
- **T2 — the right and the wrong tree theorem [EXACT, exhaustive over 945 trees].** In a rooted tree the leaf
  sets under two nodes are disjoint or nested, so two partially overlapping categories cannot both be single
  nodes. Here every X class partially overlaps every Y class: at most two of the four classes are single nodes,
  the other two need at least two nodes each — lower bound `1 + 1 + 2 + 2 = 6`, attained (`:23`). **But** if
  even unions of leaves may be read, every finite subset is representable; "'kesişen kavramlar ağacı imkânsız
  kılar' savı fazla güçlüdür. Temsilin düğümleri, izinli okuma işlemi ve maliyet birlikte değerlendirilmelidir"
  (`:25`). "Bir ağacı, onu nasıl okuyabildiğinden bağımsız değerlendiremeyiz" (`REPORT_TR.md:61`).
- **T3 — absence of information is provable only relative to an explicit family [EXACT + audit].** In the
  abstract family — local tables fixed, cross-domain bijection completely free — every `π` produces the same
  local history when there is no bridge, so no method can guarantee the cross answer from that history; under
  equal weights one cross output is uniform, 1/6. The planted generator is a *different* family: its rule leaves
  6 candidates with no bridge (`:17-19`). Three things must be kept apart: the evidence that is actually
  accessible; the assumptions the learner uses; the hidden law the evaluator knows. "Bunların karıştırılması
  hem sahte öğrenme başarısı hem sahte imkânsızlık sonucu üretebilir."
- **T4 — answer-if-unanimous against commit-to-one [EMP n=24 + ANALYST].** On the same evidence: 0 confident
  errors and Brier 0.2378, against 126 confident errors and Brier 0.4375, with equal expected forced-choice
  accuracy (439 = 439 of 576).

### (6) Limits

**Admitted by the study.**
- The bridge states identity outright; the bijection assumption and the composition are supplied; the aim was
  not met (`REPORT_TR.md:11`).
- Post-result correction; two invalid stages archived; "dış ön kayıt, dış hakem değerlendirmesi veya bağımsız
  bilimsel tekrar değildir" (`REPORT_TR.md:7`).
- Source tables complete and noise-free; nothing about getting objects or concepts out of perception
  (`REPORT_TR.md:15`).
- The equal-weight posterior is exact within the stipulated model, not the generator's posterior
  (`REPORT_TR.md:45-47`).
- The kind of question changes with the phase, so the fall of error across phases cannot be credited to the
  number of bridges alone; only same-phase comparisons between methods are valid (`REPORT_TR.md:49`).
- `oracle` has extra information; `separate` is a deliberate non-use control (`REPORT_TR.md:41`).
- Work = declared scalar counters; Python object handling, some initialisation, sorting and diagnostic JSON are
  not all counted; "bütün gerçek donanımlar için hız teoremi değildir" (`REPORT_TR.md:31`).
- The restore path is not a security boundary against hostile state (`verify_corrected.py:83`).
- The tree result is a representation/readout fact, "kavramların deneyimden oluştuğu bir öğrenme deneyi
  değildir" (`REPORT_TR.md:61`).

**Limits I see.**
- **The cost comparison has n = 1** (design constant, above).
- **Six objects, one hypothesis family.** The gap between listing candidates and the closed form grows
  factorially with `n`, so the negative result becomes *stronger* with size inside this family; for a family
  with no closed form a list or a sampler may be needed. The result is about this model and this family of
  questions — which is also what the report says.
- **Bridges are never wrong.** A contradicting pair stops the program (`assert`, `corrected.py:127`). Retracting
  a wrong correspondence — the "correct its own effect after a wrong transfer" asked for after round 20 — is not
  tested.
- **Belief is uniform over survivors**: no graded prior, no noisy evidence, no partial weight.
- **Both domains are complete before the first bridge.** Nothing is learned across domains except the pairing;
  the two relations (a six-cycle and three two-cycles) are deliberately unrelated under the learner's model, so
  there is no structure to align.
- The valid engine imports its world generator from a file under `results/invalid_initial_run/` — deliberate
  (identical worlds), but it means the "invalid" folder is a live dependency.

### (7) Implications for a core in which learning and running are one process

- **Requirement — keep the evidence and the assumption; derive the belief when asked.** A stored list of
  surviving hypotheses was dominated in every cost category by "observed pairs + closed form", with identical
  distributions. A persistent belief structure is justified only for a family of questions that needs it and
  cannot regenerate it cheaply.
- **Requirement — graded weights are marginals unless stated otherwise.** T1′ exhibits a perfectly reasonable
  set of per-item weights (1/2 each) that gives probability 1/2 to an impossible joint event. If weights are to
  be graded beliefs, the core must either hold the evidence from which the joint can be recomputed, or represent
  the constraint that couples them.
- **Requirement — uncertainty about the language of hypotheses is separate from uncertainty inside it** (T3;
  H22.9). 24 candidates in the learner's language, 6 under the generator's law. Premature certainty in a wrong
  language is the danger on the other side.
- **Mechanism to reuse — counts over surviving hypotheses with answer-if-unanimous-else-withhold**, scored with
  exact proper scores: zero confident errors (`:179-211`, `:230-237`).
- **Mechanism to reuse — the use switch.** Keep evidence and state, turn the *use* of the learned structure off
  and on (`disabled`, `:182-185`): the causal test that the learned structure, and nothing else, produces the
  answers (0 right while off, 144/144 when on).
- **Mechanism to reuse — closed-form completion from a sufficient record** (`analytic_histogram`, `:148-177`).
- **Mechanism to reuse (process)** — a cost meter that must telescope (setup + acquisition + service = final)
  in every path; targets that never enter the learner; invariance under renaming; rejection of corrupted state;
  invalid runs archived rather than deleted.
- **Warning — a forced single choice.** 126 confident errors for no expected gain in forced-choice accuracy
  (439 = 439). In this six-object world the argument for leaving "pick the top one" as the default readout is as
  clean as it can be.
- **Warning — using a correspondence that experience hands over is not discovering a common meaning.** Nothing
  was aligned; identity was told (`REPORT_TR.md:5`).
- **Warning — a concept is not the stored set alone** (T2). One node per concept fails for crossing categories;
  a readout that may take unions holds all four at six references with no copies. Node set, permitted readout
  and cost have to be fixed together — and the readout was supplied here, not acquired (H22.7).
- **Warning (process) — the first engine of this round was wrong in twelve listed ways and the first repair
  mis-accounted costs again** (4B). A learning comparison is fragile exactly where it matters: the cost account,
  the information boundary, the strength of the control.
- **Open questions**: finding the correspondence without identity bridges (H21.10; listed as still open in the
  campaign report, `living_learning_campaign_2026_09_24/REPORT_TR.md:73`, "hazır kimlik köprüsü olmadan aktarım
  (XFR01)"); acquiring the readout (H22.7); a new *kind* of relation against a parameter in a given language
  (H22.8); language uncertainty (H22.9); an acquired update operation separated from content (H22.6); retracting
  a wrong bridge.

### 4B. `AMENDMENT.md` — what was wrong with the first engine, and with the first repair

The record of defects is itself a result worth carrying over. **Found in the first engine** (`:9`): (1) the
summary added per-query deltas but left out source and bridge acquisition; (2) filtering counted survivors
instead of candidates tested; (3) `recompute` cached across queries instead of rebuilding; (4) the `tree`
method never built a tree — its abstention was imposed; (5) the disabled use never resumed; (6) restoring a
snapshot turned integer dictionary keys into strings, and the resume path omitted bridges that arrive later;
(7) the learner's constructor copied the generator's tables instead of parsing raw records; (8) targets were
passed into `query()` for scoring — not used for inference, but an unnecessary access; (9) proper scores and the
promised verification were missing; (10) "no common interface" was too strong a description; (11) phase 3
repeats phase 0, so not all queries are new; (12) an oracle with extra information is not an information-matched
strong control.

**Found in the first repair** (`:39-41`): the corrected analysis again summed only the setup *writes* and left
out the setup input and parsing counters (six methods under-counted by 216 and `recompute` by 96: 6,396 instead
of 6,612; 478 instead of 694; 130,074 instead of 130,170 —
`results/correction_audit_v1/summary_corrected.json`); the oracle's largest-candidate counter
was 0 instead of 1; the verifier compared raw histogram scales instead of normalised fractions; a complete rerun
then exposed integer keys in generated worlds against string keys after JSON loading.

What did **not** change across the three stages: the answers (374 / 202 / 0 and 450 / 0 / 126 are already in
the invalid first summary). What changed: every cost figure (first summary: `persistent` 265.4, `recompute`
10,345.4, `oracle` 72), the controls (`completion` added; `tree` removed as a learner and replaced by the
exhaustive audit; `disabled` actually re-enabled: 0 → 144), and the whole verification layer. The same-information
strong control — the one that overturned the headline — did not exist in the first design.

### 4C. `MATHEMATICS_AND_NEXT_TR.md` — literature relation and four new hypotheses

- **Relation to literature** (`:29-33`): Gentner's structure mapping (1983) carries the relational pattern
  between domains; this experiment, which is handed identity pairs, "bu kuramın uygulaması veya yeni bir analoji
  algoritması değildir". Whittington et al.'s TEM is named as a nearby line (structure separated from sensory
  content); not reproduced. Round 12 carried source relations to a new target with the common interface given;
  round 22 "bu açığı kapatamadı". Round 11's learned program pieces changed search priority and a strong
  control produced the same computation; "Burada da daha fazla öğrenilmiş mekanizma eklemek, iyi bir aynı-bilgi
  kontrolünün ötesinde kapasite getirmedi."
- **H22.6 — an acquired update operation can be separated from content.** Two states that give the same old
  answers and hold the same event records may differ in future capacity after the same new experience. Open
  burden: from which experience the changed update operation was acquired, and whether it reduces to a
  ready-made catalogue prior. Discriminating counterfactual: keep content, source evidence and representation
  fixed, change only the acquired update operation, and compare with a strong method that reprocesses the same
  history. Not run.
- **H22.7 — a concept is fixed not by the stored set alone but by the operations available on it in the task.**
  The same tree with different readouts gives different task capacity (round 22 is the finite witness). Open:
  acquiring the readout from different task experiences; an intervention must separate *switching on* a
  ready-made union operation from *acquiring* it. If the task hands over the right category by its label, no
  discovery of a concept can be claimed. Not run.
- **H22.8 — producing a new *kind* of relation as an explanation must be kept apart from finding a parameter in
  an existing language.** For unbounded computation every program the interpreter can write is there from the
  start; that does not mean no new procedure can be acquired operationally. The claim has to be tested by what
  the new procedure makes reachable under the same computation and memory, and at what acquisition cost. More
  fixed primitives, or the right program leaking from the generator, do not count. Not run.
- **H22.9 — uncertainty about the right hypothesis language is separate from uncertainty inside the language.**
  The 24 remaining candidates were not forced by the real generator. Letting source experience change the
  candidate language could give fast learning; committing early in a wrong language could hurt. The test needs
  a wrong-structural-assumption control and a correct-wide-model control together. Not run.
- Closing position (`:45`): these four are not parts of one required architecture; before the next round,
  separating the query, the strong control and the acquired-versus-written boundary "küçük bir modeli
  büyütmekten daha belirleyici olacaktır".

---

## 5. Cross-study synthesis for the new core

**The group in one paragraph.** Four rounds dated the same day (23 September 2026), each removing one
convenience of an earlier round: the ready-made distinctions (round 19), the reset of the running stream (round
20), new outside information (round 21), the ready-made common interface (round 22 — by its own account not
achieved). In each, the learned or persistent structure did what it was built for: 57 of 72 machines exact on
all finite words; zero confident errors across 140 program changes; every answer right at 2.65 times less
service work; 374 composed answers where non-use gives none. And in each, the strongest control on the same
information matched or beat it: rebuilding reached the identical state and a broader refinement solved one more
world; a few raw bits without the old learned state gave the identical sets for less work; a cached compilation
of the interface was cheaper over the life (92,136 against 111,812); a closed-form completion gave the identical
distributions at 1/9.5 of the work. In each, what persistence added was computation avoided, never information.
In none was the *way of building* the structure acquired; each report says so in its own words.

### 5.1 The laws of this group in one list

Study-level laws (stated with scope and evidence in §1–4): **D1–D7** (round 19: tests only refine; a new
distinction is constructed by prefixing an action; closed + consistent ⟹ executable; fitting certifies nothing
beyond the evidence; only the evaluator can certify; an exact model replaces the archive for prediction in the
same world; same evidence ⟹ same state). **V1–V5** (round 20: coverage under co-reachability; trace narrows only
if every component is pushed; single-valued migration iff equal summaries ⟹ equal new states; loss through a
coarser intermediate; certain ≠ right, with the error identity). **I1–I7** (round 21: no new world information
from internal computation; access changes under a budget; discharge; validity by supports; model-preserving
checks only; symmetry bound for mappings; crossing classes and one tree). **T1–T4** (round 22: closed form of
the candidate list; marginals insufficient for coupled questions; the right and the wrong tree theorem; absence
of information only relative to a family; unanimous against committed).

Read across the four rounds — **[ANALYST]**, regularities of this group rather than theorems:

- **G1 — Persistence is computation saved, not information gained.** Three independent same-evidence replays
  reach the identical state at 6.8–8.9 × (round 19), 58.7 × (round 21) and 19.7 × (round 22) the work. I1 and
  D7 are the reason. A retained structure is to be valued by what it saves, against what it costs to keep.
- **G2 — The strongest control on the same information matched or beat the learned structure in every round.**
  `rebuild` ≡ `adaptive` and `exhaustive_tests` 58 against 57; `parity_suffix8` ≡ `model_relation_suffix8`;
  `full_table_cached` < `scoped`; `completion` ≡ `persistent` at 1/9.53 of the work. The studies draw the same
  conclusion: "Giderek daha çok yapı eklemek otomatik olarak daha iyi öğrenme sağlamıyor" (round 20,
  `WHOLE_PROBLEM_TR.md:24`).
- **G3 — Content needs its conditions of validity, and "fits what I have seen" is not one of them.** Scope and
  supports (I3, I4); coverage relative to a program (V1); agreement of the surviving hypotheses (T4); against
  D4.
- **G4 — Three ways of being right are independent**: the structure is adequate to the world; the working state
  is right relative to the structure; the answer is right. Totals mix them: 1,158 = 418 + 782 − 42 (V5);
  26 errors in 9,216 samples with 15 wrong structures (D4).
- **G5 — What must be kept is fixed by what will be asked of it later.** The tests held fix the states (D1);
  the new program fixes whether an old summary can be migrated (V3); the family of questions fixes pairs
  against marginals against the list (T1, T1′); the declared interface fixes table against lemmas (H21.5).
  These are four more instances of the programme's central statement that the family of future operations
  determines the state that must be kept (g1 digest: L13).

### 5.2 One table: which future operation needs which state

| Future operation the system must still be able to perform | State that suffices | State that is insufficient, with the witness | Source |
|---|---|---|---|
| predict every later output of the same stationary world | an exact minimal machine; the raw archive is not needed | a machine that only fits the evidence: 15 of 72 wrong, 4 of them with zero errors on 128 unseen long words | D4, D6 |
| keep refining the machine when a contradiction arrives | `S`, `E` and the labels — or a resettable world to ask again | the compiled machine alone (it holds no tests and no evidence; with reset the loss is re-acquisition cost, distinctions `PROTOCOL.md:50`) | §1, [ANALYST] reading |
| continue running a *revised* program on a stream that was not reset | the new program's replay state: from the whole history; or a covering set from (old set, parity) through co-reachability; or from parity + recent trace | the new program's start state: 713 confident errors; the old learned state without a trace: all 140 migrations undetermined | V1, §2 |
| read one new state off the old summary | `(P(h), q_A(h))`, iff equal summaries ⟹ equal new states | any summary that merges two histories the new program separates | V3 |
| return to a finer structure after a detour through a coarser one | the finer structure's working state kept through the detour, or raw trace | the finer structure's code + parity + 8 recent bits: the residue mod 3 is gone | V4 |
| give an answer now without a confident error | a covering set whose members agree on the output; hypotheses that agree | one committed state or hypothesis: 713; 126 | V5, T4 |
| reuse a derived result after its premises changed, and after they changed back | the result with its scope discharged + the identities of the base facts it rests on | the result alone: 86 wrong, 36 of them after the return | I3, I4 |
| answer any question of a small declared interface cheaply | a compiled table under a version key of the premises | the lemma store is sufficient but dearer here: + 19,676 per life | H21.5 |
| answer one-reference questions about a partly known correspondence | the observed pairs + the stated model | — (the candidate list adds nothing: identical distributions at 9.53 × the work) | T1 |
| answer coupled questions about it | the pairs + the model (joint recomputed), or the candidate list | per-item marginals: probability 1/2 for an impossible collision | T1′ |
| hold categories that cross | any tree **with** a union readout (six references, no copies), or several views | one node per category: at most 2 of 4 | T2 |
| resume after a restart with the same future | the serialized state including meters, traces and caches | — | 24 + 36 + 18 + 84 fresh-process checks, all exact |

### 5.3 What the group says about each design pillar

**(P1) Learning and running are one process.**
*Reached only in round 21*, in symbolic form and with batch size one: the search that answers the query is
minimised, checked and stored, and the next query runs on the changed structure. Round 19 alternates (machine
frozen during a block, rebuilt between blocks); round 20 keeps learner and runner in two channels and gives the
runner no feedback; round 22 ingests between phases.
*Established for P1:* (a) a change of structure during running requires a migration of working state (round
20); (b) what is formed inside running must carry scope and support (round 21); (c) acquisition belongs in the
same account as the service it speeds up (rounds 21, 22); (d) a retained structure is to be judged as saved
computation (G1).
*Missing:* forming distinctions from the live, un-resettable stream **and** migrating in the same step — gap 3
of round 19, still open after round 20.

**(P2) Weights are graded beliefs.**
*No graded weight exists in this group.* Beliefs are crisp (same state or not; possible or not; active or
dormant) or counts over survivors under a stipulated equal-weight model (round 22).
*Constraints the group puts on any graded belief:* its support must cover (V1); weights are marginals unless a
joint constraint is represented or the evidence is kept (T1′); a belief needs its scope and its supports (I3,
I4); being certain and being right are different quantities (V5, T4); belief inside a language says nothing
about the language (T3); agreement with evidence is not certification (D4).
*Evidence on the readout:* answer-if-unanimous-else-withhold gave zero confident errors in round 20 (five
policies) and round 22 (three methods); commit-to-one gave 713 and 126, with no expected gain in forced-choice
accuracy in round 22 (439 = 439). The price of withholding was measured too: 1,188–1,581 of 28,458; 202 of 576.

**(P3) Concepts are formed when needed.**
*Two working formation rules, both triggered by a failure of the present structure.* Round 19: a witnessed
inconsistency between histories held together → the new test `a·e` and a new state (592 verified witnesses;
172 of 172 exact machines minimal). Round 21: a failed query → a minimised, checked, discharged clause
(**[ANALYST]** 128 of 185 are the hidden guard of a private nine-clause structure).
*Their common weakness is the same property:* no failure, no concept. 14 worlds with no contradicting evidence
stayed at one state (**[ANALYST]** exact learning 10/10 with a positive in the passive stream, 0/14 without);
lemmas come only from the 25.5 % of queries that fail.
*Neither rule merges, re-scopes or retires:* round 19 only splits; round 21 suspends and resumes but never
revises.
*Round 22 adds:* a concept is a stored structure **with** its readout; crossing concepts need a union readout or
several views (T2).

**(P4) The system imagines and measures how real its constructions are.**
*The strongest measure in the group is for derived content:* an independently checked derivation whose supports
still hold (round 21). I1 bounds what imagination can do — it adds no information about the world; I2 says what
it does do — it changes what is reachable under a budget.
*For induced content there is no internal measure here.* Round 19's learner reports `consistent` in all 15
wrong worlds; exactness was visible only to an evaluator holding the hidden machine (D5).
*Round 20 gives the measure for working state* (replay containment) and shows that it is distinct from the
adequacy of the program. *Round 22 gives* exact proper scoring of beliefs and separates the learner's
uncertainty from the world's (T3).

**(P5) The way of constructing improves and transfers.**
*Not reached in any of the four*, and each says the procedure was supplied: round 19 `REPORT_TR.md:11`; round
20 `WHOLE_PROBLEM_TR.md:32` ("Bu turdaki taşıma ilişkisi ve tur 19'daki ayrım kurma yordamı araştırmacı
tarafından yazıldı"); round 21 `PROTOCOL.md:44`; round 22 `REPORT_TR.md:33`.
*The hypotheses that would test it are written and unrun:* H21.6, H21.7, H22.6, H22.8, area A of round 20.
*The group's negative regularity bears on it:* a learned extra structure beat a strong written procedure
nowhere (G2). Any claim under P5 has to pass that control, and has to show — in round 20's formulation — which
computation can no longer be built from the same new experience once the acquired procedure is taken away.

### 5.4 Requirements, warnings, reusable mechanisms, open questions

**Requirements** (each traceable to a result above).

- **R1.** Every retained construction states its scope and its supports; it is used only while its supports
  hold; otherwise it is kept dormant, not deleted. (I3, I4; 86 wrong, 36 after the return; `erase` + 44,258 in
  24/24 lives; 81 of 81 reactivated.)
- **R2.** Every change of structure while running comes with a migration of the working state; a new structure
  never starts at its initial state by default. (713 confident errors with exact programs; 115 of 140 wrong
  states at the instant of change.)
- **R3.** Working state may be a set or graded, and it must *cover*: the structure's replay state is in the
  support at every step. An answer is given only when the support agrees; otherwise it is withheld and counted.
  (V1: 195,840 checks; T4.)
- **R4.** Three verdicts are kept apart in every test: adequacy of the structure; coverage of the working
  state; right / wrong / withheld of the answer. (V5; D4.)
- **R5.** A short raw trace is kept beside every abstraction that may be revised; an abstraction is never the
  only record of what it abstracts. (Identical sets in 39,168 points; V3, V4; **[ANALYST]** 43 of 49.)
- **R6.** The stored thing is the evidence and the stated assumption; beliefs are derived when asked, unless a
  declared family of future questions needs them stored and cannot regenerate them cheaply. (T1; 9.53 ×;
  H21.5.)
- **R7.** Graded weights declare the joint constraints they obey, or the evidence is kept from which the joint
  can be recomputed. (T1′.)
- **R8.** Uncertainty about the language of hypotheses is represented apart from uncertainty inside it. (T3;
  24 against 6.)
- **R9.** Derived content — imagined, deduced, composed — is marked as derived: it counts as access, not as
  evidence, and is accounted net of acquisition, checking, storage and lookup over the whole life. (I1, I2.)
- **R10.** A concept is specified together with its readout; crossing concepts are allowed (unions or
  compositions of nodes, or several views) and the cost of the readout is counted. (T2.)
- **R11.** The check that a changed structure still explains what is held is local to what changed — not a
  re-run over the archive. (Round 19: 64 % of the persistent learner's work; round 21: 13 % spent re-scanning
  supports at every query.)
- **R12.** Every claimed benefit passes four controls: same-evidence replay; the strongest direct procedure on
  the same information; use switched off with the evidence kept; an unsound variant that shows the safeguard is
  needed. (All four rounds; the headline of round 22 was overturned by the second of these, which its first
  design lacked.)

**Warnings.**

- **W1.** More learned structure was not more capacity in any of the four rounds (G2).
- **W2.** That consolidation pays for itself over a life is not established: − 155 on average with a standard
  deviation of 26,094, and the sign depends on the price given to a written byte.
- **W3.** Growth only on contradiction leaves the system at its default wherever nothing contradicts it.
- **W4.** Agreement with all held evidence, and with unseen samples, does not certify a structure (15; 4).
- **W5.** Errors of state do not wash out by themselves: 26 of 140 wrong starts were still wrong 16 inputs
  later; stale derived results were wrong after the world changed (50) **and** after it changed back (36).
- **W6.** Committing to one state or one hypothesis produced 713 and 126 confident errors, with no expected
  gain in forced-choice accuracy where that could be computed.
- **W7.** All costs in this group are declared virtual counters; in round 22 they are a design constant, in
  round 21 they turn on one exchange rate. No wall-clock or memory conclusion can be carried over.
- **W8.** The first build of a learning comparison is likely to be wrong exactly where it matters: round 22
  (twelve defects, then a second accounting error), round 21 (a diagnostic that looked at a base not yet seen),
  round 19 (the semantics of the resource cap).

**Mechanisms that can be reused** (file:line as in §1–4).

- **M1.** Split on a witnessed inconsistency, with the witness stored — `distinction_learning.py:98-116`.
- **M2.** Compile on closure: rows → executable machine — `distinction_learning.py:117-120`.
- **M3.** Own-evidence check that re-opens the structure at the first contradicted record —
  `distinction_learning.py:121-128` (to be made incremental, R11).
- **M4.** Set-valued working state; advance every candidate; answer if unanimous —
  `live_revision.py:68-73,100-103`.
- **M5.** Sound migration through co-reachability, a trace pushed through every component, and a small
  invariant that survives revisions — `live_revision.py:34-52,74-99`.
- **M6.** Answer first, then consolidate from the same computation: minimise → check → discharge → store —
  `deduction.py:188-247`.
- **M7.** A certificate record with support identities; active or dormant by support; identity = content —
  `deduction.py:31-32,196-200,235-238`.
- **M8.** A checker independent of the generator, with its own work meter and a tamper self-test —
  `check_proof.py:33-222`.
- **M9.** A sufficient record with closed-form belief; counts with unanimous readout; exact proper scores —
  `corrected.py:148-177,205-207,230-237`.
- **M10.** A use switch with the evidence kept — `corrected.py:182-185`.
- **M11.** The harness: same-evidence replay equality; an exact evaluator the learner cannot see; three-verdict
  reporting; a telescoping cost meter with bytes apart and a sensitivity to their weight; fresh-process
  continuation; invariance under renaming; rejection of corrupted state; invalid runs preserved; a
  change-and-return schedule (A → B → A) to expose stale knowledge in both directions.

**Open questions** (named by the studies unless marked).

- **Q1.** One channel: forming distinctions from the live, un-resettable stream and migrating the working state
  in the same step.
- **Q2. [ANALYST]** An internal measure of how real an *induced* construction is — one that reports what has
  never been tried, not only what has been matched.
- **Q3. [ANALYST]** Growth without contradiction: what makes the system test its default where nothing
  disagrees, and at what cost (the broader control cost 1.4–4.2 × and gained one world).
- **Q4.** Acquiring the procedure that builds structure (H21.6, H21.7, H22.6, H22.8; area A), shown against the
  strong written control.
- **Q5.** Which trace to keep before the next task is known (area C); which true results to keep (H21.13,
  H21.14).
- **Q6.** Finding a correspondence between separately learned domains without being told identities (H21.10;
  XFR01), and retracting a wrong one.
- **Q7.** Merging, re-scoping and retiring concepts under change or noise.
- **Q8.** Validity that is graded or partial; a revalidation test that is necessary as well as sufficient;
  noticing that a premise changed.
- **Q9.** Whether old learned state can ever narrow a migration beyond raw trace (V2 allows it; not observed).
- **Q10.** Acquiring the readout (H22.7) and the hypothesis language (H22.9); naming new symbols.
- **Q11.** Whether a distinction pays off in the *next* task rather than the present one (H19-A), and whether a
  complete old model is a sufficient state for future learning (H19-B, H21.9).

### 5.5 [ANALYST] The step these four rounds jointly imply — and what none of them ran together

No round ran the following together; rounds 19 and 20 were joined only across two separate channels. Put end to
end, the mechanisms describe one step of a core that learns while it runs:

1. **Run.** Push the input through every state the present structure may be in (M4); keep a short raw trace and
   an invariant that survives revisions (M5).
2. **Answer.** Give the output if the support agrees; otherwise withhold, and count the withholding (M4, M9).
3. **Notice a failure of the present structure.** Two histories held together disagree after the same action
   (M1); or the question cannot be satisfied (M6); or a held record contradicts the structure (M3).
4. **Form.** Build the new distinction or relation out of the failure itself — prefix the action to a held test
   (D2), or discharge the minimal set of assumptions (I3) — and store it with its witness or checked
   certificate, its scope and its supports (M7, M8).
5. **Re-seat.** Migrate the working state into the changed structure through co-reachability and the trace,
   never from the initial state (V1, R2); structures whose supports are gone become dormant (R1).
6. **Account.** Charge forming, checking, storing and looking up to the same life as the service they speed up
   (R9); keep the evidence and the assumption, derive beliefs when asked (R6).

What the four rounds did **not** test, and so what this composition does not establish: steps 3–5 on one
un-reset stream (O1); a step 3 that fires without a failure (O3); any measure in step 4 for constructions that
are induced rather than derived (O2); graded weights anywhere (P2); and whether the rules of the loop themselves
can change with experience (O4).

---

## 6. Analyst tiny checks (not part of the owner's studies)

All read-only; all on stored result files; no engine of the repository was imported or rerun. Scripts and their
JSON outputs are in
`/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/prior/_checks_g6/`.

| Check | Script → output | What it found |
|---|---|---|
| A. Integrity | `g6_integrity.py` → `g6_integrity.json` | 64 stored artifact hashes of rounds 19, 20, 22 equal today's files; round 21's three stored hashes equal; round 20's hashes of round 19's engine and result equal; `git status` empty |
| B. Round 21, what was learned and what it cost | `g6_deduction_checks.py` → `g6_deduction_checks.json` | 185 lemmas = 7.71 per life; 183 two-literal, 128 exactly a planted guard; 81 dormant at the change, 81 reactivated on return; `blind`'s 86 errors all "unsatisfiable said of satisfiable", 50 + 36; service saving on satisfiable (1,051.8 → 455.0) and unsatisfiable (1,501.2 → 397.6) queries; whole-life difference to `fresh` mean − 155, median − 9,684.5, s.d. 26,094; sign of the mean flips with the weight of written bytes |
| C. Round 19, where the evidence came from | `g6_distinctions_checks.py`, `g6_distinctions_initial.py` → `.json` | 592 splits = 212 + 212 + 168; 67.3 % of labels self-asked; 64.3 % of work is re-running the hypothesis on the archive; in `cycle_reset` exact learning = "a positive occurred in the passive stream" (10/10 against 0/14); failures by state count 1/6, 3/6, 5/6, 5/6; exact models never lost; per-life rebuild/persistent ratios 4.93–9.50 |
| D. Round 20, what the old state knew | `g6_live_revision_checks.py` → `.json` | stream regenerated and validated on 235,008 recorded parities; 86 of 140 migrations from a one-state old program; of the other 54, old state known exactly in 49, new state still ambiguous in 43 of those, never narrower than the trace-only control; wrong starts persisting 115 → 26 over 16 inputs; ambiguity decaying 64 → 48 (trace) and 140 → 54 (no trace) without feedback |
| E. Round 22, what varies | `g6_crossmap_checks.py` → `.json` | every method's cost vector identical in all 24 lives; only the phase-2 category coincidences vary (17 lives with 15 certain answers, 7 with 17); the 38 = 24 + 14; `point` 450 right in fact, 439.0 expected under the learner's own model = the mixture's forced-choice expectation |
| F. Small derived numbers | `g6_small_numbers.py` → `.json` | sign test 15/24 two-sided p = 0.3075; ratios 58.67, 19.69, 9.53; per-stride cost table of round 20 reproduced from `summary.json` |

Caveats on my own checks: B's "planted guard" is recovered as the public literals common to all clauses of a
gadget in the stored world; D re-simulates the tracker with the published rule and asserts equality with the
stored sets at the end of every block; none of these is a rerun of an experiment, and none changes a number the
studies report.

---

## 7. Established / Refuted / Open — across the group

Grades as in §0. "Established" means: shown by this group's own evidence **within the stated scope** — in every
case a supplied finite symbolic language, 24 paired lives where empirical, descriptive statistics, declared
virtual costs, no external replication. Nothing below is claimed by the studies as a new principle; all four
reports say so.

### Established

| # | Statement | Grade | Scope / what is given | Where |
|---|---|---|---|---|
| E1 | Distinctions and an executable state graph can be built from outcomes without the number of states or the list of distinguishers: 57 of 72 worlds exact on all finite words; all 172 exact final machines minimal. | [EMP n=24 × 3 families] | stationary deterministic finite-state worlds; reset; free truthful queries; the table procedure given | §1 (4) |
| E2 | A new distinction is constructed by prefixing one action to a held test when two histories held together diverge after that action; closed + consistent tables compile to a machine. | [EXACT] + 592 verified witnesses | Angluin's observation table | D2, D3 |
| E3 | Tests only refine: provisional sameness can be split by new evidence, never joined; in a stationary world truly equivalent histories are never separated. | [EXACT] | fixed world, truthful labels | D1 |
| E4 | Fitting all held evidence certifies nothing beyond it: 15 of 72 final machines wrong with zero contradictions; 4 of them with zero errors on 128 unseen long words; 26 errors in 9,216 samples. | [EXACT] counterexample + [EMP] | — | D4 |
| E5 | Where the separating experience never occurs, the structure does not grow: 14 worlds stay at one state. In that family, exact learning coincides with a positive label occurring in the passive stream (10/10 against 0/14). | [EMP]; the coincidence is [ANALYST] | need-driven queries only | §1 (4) |
| E6 | Persistent state ≡ state rebuilt from the same evidence; the difference is recomputation only: 6.79–8.86 × (round 19), 58.7 × (round 21), 19.7 × (round 22). | [EXACT] induction + [EMP]; costs virtual, round 22 [DESIGN-CONSTANT] | deterministic procedures | D7, H21.4, H22.2 |
| E7 | A correct new program started at its initial state on an un-reset stream gives confident wrong answers: 713 of 28,458; wrong state in 115 of 140 changes; still wrong in 26 of 140 after 16 inputs. | [EMP]; persistence [ANALYST] | no feedback on the live stream | §2 (4) |
| E8 | Coverage: migrating a set through the co-reachability relation keeps the program's replay state in the set, whatever the adequacy of old and new program. | [EXACT] + 195,840 checks | finite deterministic programs | V1 |
| E9 | One new state can be read off an old summary iff equal summaries imply equal new states. | [EXACT] | — | V3 |
| E10 | Passing through a coarser representation can lose a working state that a later finer one needs; keeping the old program's code, parity and 8 recent bits does not restore it. | [EXACT] witness (mod 3) | not an event of the main run | V4 |
| E11 | Certain ≠ right; totals hide cancellations: 1,158 = 418 + 782 − 42. A covering set's unanimous answer equals the program's full-replay answer and nothing more. | [EXACT] identity + [EMP] | binary outputs | V5 |
| E12 | Set-valued state with answer-if-unanimous gives zero confident migration errors at the price of withholding 1,188–1,581 of 28,458 answers. | [EMP] | exact-program phases | §2 (4) |
| E13 | Internal computation adds no information about the world (`I(W;Z│H) = 0`); under a finite budget it changes what is reachable: 669 → 1,788 of 2,304 queries answered within 500 operations; mean service 1,166.3 → 440.3 with every answer right (H21.1). | [EXACT] + [EMP n=24] | no side channel; guarded formulas, supplied solver | I1, I2, §3 (4) |
| E14 | A local result becomes a global one only by discharging its assumptions into its statement (`F∧a… unsat ⟹ F ⊨ ¬a∨…`); keeping the bare local conclusion is unsound. | [EXACT] + counterexample | — | I3 |
| E15 | Validity travels with the supports (sufficient, not necessary). Without it: 86 wrong answers, 1,705 invalid uses, 5 invalid new results; all errors "unsatisfiable said of satisfiable", 36 of them after the world had returned. | [EXACT] + [EMP]; error kind and timing [ANALYST] | whole-clause premises with content identities | I4, H21.3 |
| E16 | Dormancy beats deletion: deleting on change costs + 44,258 per life (24/24 lives); 81 results suspended at the change, the same 81 active again on return. | [EMP]; 81/81 [ANALYST] | change-and-return schedule | §3 (4) |
| E17 | Results formed from the system's own failed queries are compact public relations: 183 of 185 two-literal; 128 exactly the hidden guard of a private nine-clause structure. | [ANALYST] from recorded data | — | §3 (4) |
| E18 | Under equally weighted bijections with `k` of `n` pairs seen, `P(π(x)=y) = 1/(n−k)`; one-reference questions need no list. Stored list, full re-enumeration and closed form give identical distributions on 576/576 questions at 6,612 / 130,170 / 694 work. | [EXACT] + [DESIGN-CONSTANT] | stipulated model; identity bridges | T1, H22.2, H22.5 |
| E19 | Per-item marginals do not determine coupled questions (probability 1/2 for an impossible collision). | [EXACT] witness | — | T1′ |
| E20 | No rooted tree holds two crossing two-class partitions as single nodes (at most 2 of 4 classes); with a union readout all four are held at 6 node references without copies. | [EXACT], 945 trees | six leaves | T2 |
| E21 | Absence of information is relative to an explicit family: 24 / 6 / 2 / 1 candidates for the learner against 6 / 2 / 1 under the generator's law. | [EXACT] + post-result audit | — | T3 |
| E22 | Answer-if-unanimous against commit-to-one on the same evidence: 0 against 126 confident errors; Brier 0.2378 against 0.4375; expected forced-choice score equal (439 = 439). | [EMP n=24]; the equality [ANALYST] | six objects | T4 |
| E23 | Switching the *use* of a learned structure off and on, with its evidence kept, isolates its causal effect (0 right while off; 144/144 when on). | [EMP] | — | §4 (4) |
| E24 | Learner state survives a process restart with the same future (24 + 36 + 18 + 84 continuations exact); behaviour is invariant under renaming of states (18 paths) and tokens (72 paths). | [EXACT] checks | the four engines | verify scripts |

### Refuted or not supported

| # | Idea | Verdict and evidence | Where |
|---|---|---|---|
| X1 | A more correct new program interprets the running past correctly by itself. | **Refuted**: 713 confident errors in phases with an exact program. | live_revision `WHOLE_PROBLEM_TR.md:17` |
| X2 | No correct transfer is possible without storing the raw history. | **Refuted as a generalisation**: a present summary or a recent trace sometimes suffices. | `:18` |
| X3 | If the new state cannot be known exactly, a confident wrong answer is unavoidable. | **Refuted**: covering set + withholding. | `:19` |
| X4 | The old learned structure adds benefit to every transfer. | **Not supported**: identical sets and answers in all 39,168 points; + 653–722 work per life; [ANALYST] exactly known old state left the new one ambiguous in 43 of 49 cases. | `:20`; §2 (4) |
| X5 | A method with fewer wrong answers has learned a better model. | **Refuted here**: same programs; part of the difference is withholding. | `:21` |
| X6 | Replaying the archive is in general the most expensive option. | **Not shown**: cheaper on average than relation + trace in these short lives. | `:22` |
| X7 | More selective creation of distinctions is unconditionally better. | **Refuted**: the broader control learned one more world; in 6 of 24 worlds the selective method asked more of the environment. | distinctions `REPORT_TR.md:46,70` |
| X8 | Passing unseen tests shows that the right structure was acquired. | **Refuted**: 4 wrong machines with 0 errors on 128 long words. | distinctions `REPORT_TR.md:44` |
| X9 | Keeping every derived result regardless of what it rests on is safe. | **Refuted**: 86 wrong answers. | deduction `REPORT_TR.md:45` |
| X10 | Persistent lemmas beat every strong compilation (H21.5). | **Rejected in scope**: 111,812 against 92,136. | deduction `REPORT_TR.md:41,57` |
| X11 | Consolidation pays for itself over the life (H21.2). | **Only partially supported** by the study (− 155.2; 15/24); [ANALYST] not distinguishable from zero (s.d. 26,094; p ≈ 0.31) and dependent on the price of a written byte. | deduction `REPORT_TR.md:51`; §3 (4) |
| X12 | The number of true results held and usable capacity grow together. | **Not supported**: 474 points served more dearly with lemmas than without; no selection rule (H21.13 untested). | `MATHEMATICS_AND_LITERATURE_TR.md:17` |
| X13 | A stored list of all surviving hypotheses is needed, or beats direct inference on the same information (H22.5). | **Rejected**: identical distributions at 1/9.53 of the work, dominated in every cost category. | crossmap `REPORT_TR.md:31-33` |
| X14 | Committing to one surviving hypothesis improves forced-choice accuracy. | **Not supported in expectation** (439 = 439); 450 against 439 in fact is a fluctuation; 126 confident errors. | §4 (4) [ANALYST] |
| X15 | One tree must lose one of two crossing tasks (strong H22.4). | **Refuted**: union readout, six references, no copies. | crossmap `REPORT_TR.md:55-61` |
| X16 | Round 22 removed the ready-made common interface / discovered a shared meaning. | **Not met**; withdrawn by the report: identity bridges, bijection and composition supplied. | crossmap `REPORT_TR.md:5,11` |
| X17 | The learner's candidate count is a bound for any learner. | **Refuted** by the scope audit (6 / 2 / 1). | crossmap `REPORT_TR.md:47` |
| X18 | The figures of round 22's first engine (costs; the `tree` method; `disabled`) and of its first repair. | **Invalid, withdrawn**, archived. | crossmap `AMENDMENT.md:9,39` |
| X19 | "185 lemmas per life" (round 21's table). | **Erratum**: 185 is the total over 24 lives; mean 7.7083. | crossmap `REPORT_TR.md:73` |
| X20 | Any of the four rounds found a new principle or solved living learning. | **Explicitly denied** in each report. | distinctions `:5`; live_revision `:7`; deduction `:59`; crossmap `:5` |

### Open

| # | Question | Named by | Why it matters for the new core |
|---|---|---|---|
| O1 | Forming distinctions from the live, un-resettable stream and migrating the working state in the same step. | round 19 gap 3; round 20 `REPORT_TR.md:100` | this *is* "learning and running are one process" |
| O2 | An internal measure of how real an induced construction is — one that knows what was never tried. | [ANALYST] from D4, D5 | the pillar "measures how real its constructions are" has a working instance only for derived content |
| O3 | Growth where nothing contradicts the default; self-chosen experience beyond what the present structure demands. | round 19 gap 1; H21.8 | need-driven formation alone stalled in 14 worlds |
| O4 | Acquiring the constructing procedure itself, and showing it against a strong written control. | H21.6, H21.7, H22.6, H22.8; round 20 area A | the pillar "the way of constructing improves and transfers" is untouched |
| O5 | Which trace and which true results to keep before the next task is known. | round 20 area C; H21.13, H21.14 | every mechanism here was given its trace length and kept everything |
| O6 | A correspondence between separately learned domains without identities being told; retraction of a wrong one. | H21.10; campaign XFR01; [ANALYST] retraction | round 22 did not meet it |
| O7 | Merging, re-scoping and retiring concepts under change and noise. | round 19 `MATHEMATICS…:25`; round 20 area B; H21.12 | both formation rules only add |
| O8 | Graded or partial validity; necessary-and-sufficient revalidation; noticing that a premise changed. | I4 (`sufficient, not necessary`); [ANALYST] | support checks here are binary and the change is announced |
| O9 | Whether old learned state can ever narrow a migration beyond raw trace. | V2; round 20 `REPORT_TR.md:50` | decides whether abstractions or traces carry continuity |
| O10 | Acquiring the readout of a concept and the language of hypotheses; naming new symbols. | H22.7, H22.9; [ANALYST] symbols | T2 and T3 show both matter; both were supplied |
| O11 | Whether a distinction pays off in the next task rather than the present one; whether a complete old model suffices for future learning. | H19-A, H19-B, H21.9 | the value of a concept formed "when needed" may lie later |
| O12 | Whether consolidation pays over a life under a physical cost measure. | H21.2; [ANALYST] | the only measured answer is unit-dependent |
| O13 | Whether the obligations — acquiring content, deriving computation, keeping validity, choosing resources — need one mechanism or several. | H21.16 | organising hypothesis, deliberately left undecided |
