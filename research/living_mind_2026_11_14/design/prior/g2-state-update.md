# g2 — Prior-work digest: acquired state, learned update, joint life (+ papers 04, 05, 06)

Repository root: `/workspace/cevahir-ai` (read-only). All paths below are relative to that root unless absolute.
Digest written 2026-11-14 for the "new core" design (learning and running as one process; weights as graded
beliefs; concepts formed when needed; the system imagines and measures how real its constructions are; the way
of constructing improves and transfers).

---

## 0. What was read, how the documents map, integrity

**Named folders — all three exist under both roots.**

| Study | Docs (`docs/research/<study>/`) | Code + results (`research/<study>/`) |
|---|---|---|
| `living_learning_state_2026_09_20` | `REPORT_TR.md`, `EXPERIMENT.md`, `FAILURE_IDENTIFIABILITY.md`, `STATE_DISCOVERY_LITERATURE.md`, `REVIEW.md` | `state_merging.py`, `recurrent_state.py`, `evidence_audit.py`, `verify_state.py`, `preserve_prior.py`; `results/{recurrent_state,evidence_audit,verification,prior_manifest}.json` |
| `living_learning_update_2026_09_20` | `REPORT_TR.md`, `EXPERIMENT.md`, `UPDATE_THEORY.md`, `MEMORY_AND_SCOPE_AUDIT.md`, `REVIEW.md` | `learned_update.py`, `analytic_risk.py`, `verify_update.py`, `preserve_prior.py`; `results/{learned_update,analytic_risk,verification,prior_manifest}.json` |
| `living_learning_joint_2026_09_20` | `REPORT_TR.md`, `JOINT_STREAM.md`, `TEMPORAL_RELATION.md`, `PRINCIPLE_AUDIT.md`, `ADVERSARIAL_REVIEW.md` | `joint_stream.py`, `temporal_relation.py`, `joint_ablation.py`, `verify_joint.py`, `preserve_prior.py`; `results/{joint_stream,joint_no_burst,temporal_relation,verification,prior_manifest}.json` |

**Papers.**

| Paper | Is the edition of | Note |
|---|---|---|
| `docs/publications/papers/04-representation-credit-state-transport.md` (CEV-2026-04) | `docs/research/living_learning_reassessment_2026_09_20/REPORT_TR.md` | NOT one of the three named folders. Its study is `docs/research/living_learning_reassessment_2026_09_20/` (5 files) + `research/living_learning_reassessment/` (code folder has **no date suffix**; `research/living_learning_reassessment_2026_09_20` does not exist). I read that study too so that paper 04 can be digested with mechanisms. |
| `docs/publications/papers/05-joint-lifetime-temporal-relations.md` (CEV-2026-05) | `docs/research/living_learning_joint_2026_09_20/REPORT_TR.md` | Body identical to the report. |
| `docs/publications/papers/06-learned-update-transfer.md` (CEV-2026-06) | `docs/research/living_learning_update_2026_09_20/REPORT_TR.md` | Body identical to the report. |
| (`07-acquired-recurrent-state.md`, not in my list) | `docs/research/living_learning_state_2026_09_20/REPORT_TR.md` | The state study's edition is paper 07, not 04–06. |

Checked by diff (link targets normalised): for 04/05/06 the text after the `---` rule equals the respective
`REPORT_TR.md` except for the title line. **The only new text in the papers is the header** (Özet, Abstract,
"Bulguların yorumu"). One header is factually wrong — see §5.

**Integrity checks I ran (read-only, seconds).** Every `source_sha256` stored in a result JSON equals the sha256
of the current source file (state 2/2, update 2/2, joint 3/3, reassessment 3/3); every result hash stored in the
three `verification.json` files equals the current result file; `git status` stayed clean. So the numbers
quoted below are the numbers the committed code produced.

**Substrate fact.** None of the four studies contains a neural network or imports numpy/torch — standard-library
Python only. Everything "learned" is one of: a DFA transition table; a set of surviving candidate DFAs; count
tables; a linear coefficient vector; Gram / inverse-Gram statistics; a 2×2 symmetric gain matrix; finite-field
linear heads over product features. The laws below are therefore functional statements about *state*, not about
any substrate. Nothing here is evidence for or against a neural core.

**Evidence grades used in this digest.**
- **[EXACT]** finite algebraic/arithmetic witness or exhaustive finite check executed in the study's own code.
- **[EMP n=k]** measured over k independent lives/sources, with the study's own interval.
- **[SUGGESTED]** stated or interpreted by the study without a dedicated test.
- **[ANALYST]** my inference or my tiny check — *not* a result of the owner's studies (collected in §8).

Every study in this group states, repeatedly, that the main problem is open and that no new general principle is
claimed ("Ana problem hâlâ açık"). The studies' own decision words are kept in Turkish where they are labels.

---

## 1. Study `living_learning_state_2026_09_20` — an executable state procedure acquired from experience

### (1) Question
Can terminal labels of short episodes alone (no states, no transitions, no state count) make a learner acquire a
persistent transition procedure that afterwards processes much longer unseen sequences one symbol at a time
without reading stored raw sequences — and must a system first diagnose the physical cause of an error before it
can learn from it? (`EXPERIMENT.md:9`, `REPORT_TR.md:11-16`)

### (2) Mechanism / state representation actually used
Symbolic, deterministic, ungraded.

- **Executed state.** A DFA table `{'transitions': [[n0,n1],…], 'outputs': [0/1,…]}` plus one integer (current
  state). Prediction = one table lookup per symbol: `recurrent_state.py:33-37` (`answer`). The stored training
  words are not read on the prediction path.
- **Learning state.** The full archive of labelled words `self.samples` (`recurrent_state.py:92,114`).
- **Learner under test, `merged`** — RPNI-style red/blue state merging, `state_merging.py:122-220`:
  prefix tree of all labelled words in shortlex order, unlabeled prefixes = UNKNOWN (`:131-140`); repeat: blue =
  children of red states (`:157-162`); take the shortlex-first blue (`:165`); try to merge it into each red in
  shortlex order (`:166-177`); `_try_merge` (`:62-112`) does the union + determinism closure and **rejects only
  if two *observed, different* terminal labels meet** (`:89-92`); if no red accepts, promote blue to red
  (`:178-180`). Completion: unknown outputs → 0 (`:186`), missing transitions → rejecting sink (`:190-206`).
  A final assertion checks consistency with every sample (`:216-219`). Two different labels for the same word
  raise an error (`:41-42`). No state-count bound is given.
  → The merge criterion is "no contradiction seen so far": absence of evidence licenses a merge, one label
  vetoes it, and defaults (0 / sink) are emitted as ordinary answers.
- **Update rule in the life.** Predict first (`recurrent_state.py:206-209`), then `observe` (`:110-129`): append
  the sample; **refit from scratch on the whole archive** when the number of labels is in {16, 32, 64, 128, 256}
  (`:122-128`); between refits the last graph serves. Refits only at observed episode boundaries.
- **Comparators.** `enumerated_compiled` / `enumerated_vote`: version space over all 1,054 distinct languages of
  minimal reachable binary DFAs with ≤3 states (`candidate_library`, `recurrent_state.py:68-85`; 5,898 labelled
  descriptions → 1,054 languages = 2 + 24 + 1,028; re-derived independently by table-filling minimisation,
  `evidence_audit.py:34-77`). In `recurrent_state.py`: every label deletes inconsistent candidates (`:113`);
  "compiled" answers with the first survivor, "vote" with the survivors' mean answer (`:102-106`); **abstains
  when the set is empty** (scored as p = .5, `:166`). `suffix8`: Laplace counts keyed by the last 8 symbols
  (`:107,117-120`). `word_table`: exact-word lookup (`:108`).
- **The relation the study is about** (`EXPERIMENT.md:88-94`, `REPORT_TR.md:50-57`):
  `h ~ h'  ⟺  ∀ permitted continuation u: f(hu) = f(h'u)`; it is preserved on the right
  (`h~h' ⇒ ha~h'a`), so `δ([h],a) = [ha]` is well defined; finitely many classes ⇒ a finite recurrent
  computation suffices.

### (3) Experimental design
- **Unit:** a life = 256 distinct binary words (length uniform in 1–12, exact repeats rejected), terminal label
  only, episode start/end observable, every episode starts from the same start state (`EXPERIMENT.md:13-17`).
- **Worlds** (`EXPERIMENT.md:21-26`; `recurrent_state.py:182-196,350-353`):

  | Regime | Lives | Hidden target |
  |---|---:|---|
  | `random_three_state` | 16 | 16 distinct minimal 3-state DFAs |
  | `parity` | 4 | odd number of ones (2 states) |
  | `outside_three_state_bound` | 4 | (number of ones) mod 4 == 0 (needs 4 states) |
  | `corrupted_unique_labels` | 8 | first 8 targets of the main family; 32 of 256 labels flipped (12.5 %), all words distinct so no contradiction exists |

  The same seed uses the same training words in every regime (`words(81000+seed)`), i.e. a paired design.
- **Evaluation:** after each refit, 128 unseen short words + 64 long words (16 each of length 32, 64, 128, 256),
  never shown to the learner; Brier loss `(p−y)²`, abstention scored as .5 with coverage reported separately
  (`recurrent_state.py:164-168`); and an **exact language-equivalence check against the hidden truth** by product
  graph (`witness`, `recurrent_state.py:40-53`; non-empty-domain variant `evidence_audit.py:19-31`) — evaluator
  only, never returned to the learner.
- **Statistics:** descriptive normal intervals over lives within a regime; single configuration; no tuning on
  held-out results; no simultaneous significance claim (`recurrent_state.py:369-370`).
- **Separate post-result audit** (`evidence_audit.py`): domain correction for the empty word, an explicitly
  privileged repair with one extra label, erasure checks, an unseen-alternative witness.

### (4) Results (the study's numbers) and decision labels

Final long-sequence Brier loss (`EXPERIMENT.md:50-58`; `results/recurrent_state.json › summary`):

| Method | 3-state family | Parity | 4-state rule | Wrong labels |
|---|---:|---:|---:|---:|
| merged (learned graph) | 0 | 0 | 0 | .287109 (CI [.164, .410], range .109375–.609375) |
| enumeration, compiled | 0 | 0 | .25 (abstains) | .25 (abstains) |
| enumeration, vote | 0 | 0 | .25 (abstains) | .25 (abstains) |
| suffix8 | .206508 | .264526 | .240724 | .227182 |
| word table | .25 | .25 | .25 | .25 |

- **Exactness.** Full language 14/16; on the non-empty domain actually used **16/16** — the two mismatches are the
  empty word only, never part of train/service/test (`EXPERIMENT.md:66-70`; `results/evidence_audit.json › lives`).
  Parity 4/4. Four-state control 4/4 with exactly 4 learned states, while the ≤3-state version space emptied
  after 8, 8, 11, 10 labels. Corrupted: **0/8**, 36–42 states (mean 39.375) for 3-state truths
  (`EXPERIMENT.md:74`); shortest counterexamples have length 1–4 (`evidence_audit.json`).
- **Learning curve** (from the JSON checkpoints): main family, long Brier .2637 / .0576 / 0 / 0 / 0 at
  16 / 32 / 64 / 128 / 256 labels; mean learned states 5.0 / 3.69 / 2.94 / 2.94 / 2.94.
- **Whole-life service loss** (clean Brier, main family): merged **.050781**, compiled **.016113**, vote
  **.009146** (`EXPERIMENT.md:62`). The learner that was given the correct ≤3 bound and updates on every label
  made fewer errors while living; equal final result.
- **Trusted correction.** When the evaluator replaces the 32 flipped labels, the graph rebuilt from the kept
  archive is exact on the non-empty domain in 8/8 lives (6/8 including ε) (`EXPERIMENT.md:78`;
  `recurrent_state.py:219-224`).
- **Persistence.** (a) Full learning state loaded in a fresh OS process at episode 37: 53 further episodes,
  including the refit at 64, identical. (b) Graph + current state only, the 127-symbol prefix not serialised:
  the remaining 385 symbols identical (`recurrent_state.py:321-338`; `EXPERIMENT.md:104-109`).
- **Erasure** (`evidence_audit.py:99-115`): on `0^129` vs `1 0^128` the learned parity graph answers [0,1];
  resetting the running state after the first symbol gives [0,0] (really executed); "graph erased" [0,0] is an
  analytic substitution of the initial constant-0 procedure, not a run with a missing graph (`REVIEW.md:39`).
- **Memory / cost.** Final graph ≈ 2.94 states, 5.875 transition entries; archive 256 words, 2,094 symbols
  (`EXPERIMENT.md:109`). Merge-closure pair visits 1,546.3 (clean) vs 7,460.1 (corrupted); states copied in merge
  attempts **8,530.6 vs 440,572.8**; enumeration filter transitions 18,551.6 vs 14,572.0 (`EXPERIMENT.md:111-117`).
  Counters are not a common unit; no FLOP/time claim.

| Claim | Label | Evidence | Reference |
|---|---|---|---|
| Terminal labels of short words can yield a persistent transition procedure that generalises to much longer words without reading raw history | **supported, in scope** | [EXACT per target] product-graph equivalence on all non-empty words: 16/16, 4/4, 4/4 | `EXPERIMENT.md:60-64`, `REVIEW.md:45` |
| The learner can acquire a state count the fixed family excludes | **supported** | [EMP n=4] 4 states learned; ≤3 family empty | `EXPERIMENT.md:64` |
| State merging is better than bounded candidate elimination | **not supported** | [EMP n=16] service .050781 vs .016113 / .009146; equal final | `EXPERIMENT.md:62`, `REVIEW.md:23` |
| Training consistency + structure growth ⇒ a valid new distinction was acquired | **refuted** | [EMP n=8] 0/8 exact, 36–42 states, Brier .287109 | `REPORT_TR.md:32-38`, `EXPERIMENT.md:72-76` |
| An emptied bounded family diagnoses "labels are wrong" | **refuted** | [EXACT] same symptom in the clean 4-state world (empty at 8–11) and the corrupted world (empty at 8–13) | `REVIEW.md:31`, `EXPERIMENT.md:76` |
| "First diagnose the physical type of the error, then learn" is a mandatory principle | **not supported** ("desteklenmedi") | [EXACT] aliasing witnesses, §(5) S4 | `REPORT_TR.md:16`, `FAILURE_IDENTIFIABILITY.md:116` |
| The kept archive suffices for repair once a *trusted* correction is given | **supported** | [EMP n=8] 8/8 | `EXPERIMENT.md:78` |
| The system discovers which labels are wrong | **not shown** | correction locations come from the evaluator | `REVIEW.md:33` |
| merged vs enumeration under corruption (which is worse?) | **inconclusive** | paired diff +.0371, CI [−.0857, +.1599] | `recurrent_state.json › summary`, `REVIEW.md:25` |

### (5) Laws / invariants claimed (all explicitly "known, not new")

- **S1 — Future-congruence law.** For a deterministic, stationary terminal target `f`: histories may share a
  state iff every permitted continuation gives the same answer, `h~h' ⟺ ∀u f(hu)=f(h'u)`; the relation is a
  right congruence, hence `δ([h],a)=[ha]`. **Equality of the present labels is not sufficient for merging**
  ("İki geçmişin şu an aynı cevabı vermesi, onları aynı iç duruma koymaya yetmez", `REPORT_TR.md:50`). Necessary
  condition for any state code `q` with a fixed transition: `q(h)=q(h') ⇒ q(ha)=q(h'a)`
  (`STATE_DISCOVERY_LITERATURE.md:49-53`). Scope: deterministic, fixed `f`, given alphabet and episode
  boundaries; stochastic version needs equality of conditional future distributions
  (`FAILURE_IDENTIFIABILITY.md:94`). Rests on: the right-congruence argument in `EXPERIMENT.md:94` (the classical
  Myhill–Nerode construction; the study does not name it, it says "bilinen düzenli dil ve öngörü durumu fikri",
  `:98`) and the constructive runs. Mapped by the study to RPNI, CSSR, causal states, PSRs.
- **S2 — No fixed window replaces a recurrent state.** [EXACT] For parity and every `k`, `0^(k+1)` and `1 0^k`
  have equal length and equal last `k` bits but opposite labels ⇒ any predictor reading only that window has
  error ≥ 1/2 (Brier ≥ .25) on the balanced pair (`EXPERIMENT.md:84`; checked for k = 1, 8, 32, 128 in
  `recurrent_state.py:264-273`). The fairer witness with full input/output pairs in the window:
  SET0/SET1/WAIT/QUERY (`FAILURE_IDENTIFIABILITY.md:70-86`). "Bu yalnız temsil yeterliliği ispatıdır" — it proves
  representational adequacy, not that the procedure was acquired.
- **S3 — Finite evidence never certifies the congruence.** [EXACT] For any finite sample `D` and regular `L`,
  `L' = L △ {w*}` (`w*` unseen, longer) is regular and agrees with `L` on `D`
  (`STATE_DISCOVERY_LITERATURE.md:73-77`). Executed: a 16-state machine equal to parity on all 256 training
  words and different only on `0^13` (`evidence_audit.py:80-96,145-149`). "Not seeing a separating continuation
  does not prove equivalence" — generalisation needs a class/evidence assumption.
- **S4 — A prediction error does not say what must change.** [EXACT] Same observation stream, different useful
  correction: target changed (`q=1−x, y=q`) vs feedback channel inverted (`q=x, y=1−q`)
  (`FAILURE_IDENTIFIABILITY.md:19-32`; `verify_state.py:52-55`); new instantaneous target vs one-step label delay
  on `x=(0,1,0)`, separated only by a naturally occurring `x[4]=0` (`:34-45`); missing recurrent XOR state vs an
  externally scheduled coefficient on a 6-step record (`:47-51`). **Critical qualifier:** if the goal is to
  predict the observed `y` itself, the two worlds need the same new rule and no diagnosis is required; the
  ambiguity matters only when success is defined on the hidden `q` (`:30`). Three questions are separated:
  prediction sufficiency / correction sufficiency / cause identification (`:11-15`).
- **S5 — A summary can erase the relation a later correction needs.** [EXACT] Inputs `0,0,1,1` vs `0,1,0,1`
  with feedback `0,0,0,1` have identical unordered `(x,y)` pairs and identical normal-equation summaries, but
  the lag statistic `C1 = Σ x[t−1]y[t]` is 1 vs 0 (`FAILURE_IDENTIFIABILITY.md:53-68`; `verify_state.py:56-60`).
  Stated law: **"Ne saklanması gerektiği, daha sonra desteklenmek istenen düzeltme ailesine bağlıdır"** — what
  must be kept depends on the family of corrections one wants to support later (`:68`). Not "keep everything":
  the ordered raw record suffices, and so does the pre-computed `C1` for this one candidate.
- **S6 — More states cannot repair an observation conflict.** [EXACT] The same word with both labels cannot be
  satisfied by any deterministic DFA; at least `min(n0,n1)` training errors
  (`STATE_DISCOVERY_LITERATURE.md:79-83`; `state_merging.py:41-42`; `recurrent_state.py:274-281`).
- **S7 — State identifiers are not portable across procedure versions.** [EXACT] Two graphs with the same
  language but different numbering: carrying state id 1 gives output 1, the correct mapped id 2 gives 0
  (`recurrent_state.py:282-291`). Mid-episode migration needs prefix replay or a verified state map; the study
  avoids it by refitting only at episode ends (`STATE_DISCOVERY_LITERATURE.md:92`).
- **S8 — Execution memory ≠ learning memory ≠ diagnosis information.** [EXACT/EMP] 385 symbols continue from
  graph + one integer, but continuing to *learn* needed the whole archive
  ("Küçük çalışma durumu ile küçük öğrenme belleği aynı başarı değil", `REPORT_TR.md:65`).
- **S9 — "Parameter change" vs "state change" are not natural kinds.** [SUGGESTED] `s[t+1]=F(s[t],x[t])` can be
  read either way; comparisons must be defined by what is kept, what the update may read, the operation family
  and the cost (`FAILURE_IDENTIFIABILITY.md:17`).

### (6) Limits
**Admitted by the study.** Given: binary alphabet, episode boundaries, correct terminal attribution, the
deterministic finite-state language, the merge primitive and its order preference, storage of all examples
(`EXPERIMENT.md:34,123`). Not solved: cause diagnosis, acceptance of structure under noise, mid-episode
representation migration, lifetime memory bounds, perception of symbols, goals, action-generated experience
(`REPORT_TR.md:71-75`). Exactness is the *evaluator's* certificate, not something the learner holds
(`REVIEW.md:19`). The comparison with enumeration is not compute-matched (`EXPERIMENT.md:62`). The 14/16 → 16/16
reading is a post-result domain correction, kept transparent; the ε repair is privileged (`EXPERIMENT.md:70`).

**Limits I see [ANALYST].**
1. "Learning while running" holds only weakly: five batch re-inferences from the full archive, nothing
   incremental; under corruption the re-inference work explodes (copied states ×52).
2. The learner has **no graded evidence and no notion of its own ignorance**: it never abstains, and its two
   "ε failures" come purely from the silent default `outputs = max(label, 0)` (`state_merging.py:186`) — an
   unobserved quantity emitted as an answer.
3. Two learner-accessible signals are present in the study's own files but not exploited:
   (a) **hypothesis size vs evidence**: clean worlds saturate (mean states 2.94 / 2 / 4, constant from 64 labels
   on), the corrupted world grows without saturating — 6.1 → 10.1 → 15.3 → 25.0 → 39.4 states at
   16/32/64/128/256 labels, i.e. 33–39 extra states for 32 flipped labels (≈ one invented state per false
   label). From 64 labels on the regimes are separated (clean ≤ 4 states, corrupted ≥ 12). Caveat: a genuinely
   large clean target also grows until it saturates, so this is a rate signal, not a verdict.
   (b) **unanimity of the surviving alternatives**: in all 20 clean lives inside the family, every surviving
   bounded candidate agreed on the whole non-empty language
   (`evidence_audit.json › all_remaining_bounded_candidates_agree_on_nonempty_language = true`, version-space
   size 1 or 2; computed at `evidence_audit.py:127-129`). That is a certificate the learner could compute itself
   — but only relative to the family; outside the family it is empty, not "false".
4. Small n for the special regimes (4 lives each, one fixed target each); corruption is uniform random flips
   only — no structured or adversarial corruption, and by design no duplicate words.

### (7) Implications for a core where learning and running are one process
- **REQ — merge by futures, not by present agreement (S1).** Any concept/state the core forms by identifying
  two experiences must be justified against a declared set of continuations, and must be splittable when a
  separating continuation appears. Present-output equality is the wrong merge criterion.
- **REQ — keep three state tiers apart and test each by trajectory equality (S8):** what is needed to *run*
  (procedure + current state), to *keep learning* (here: the archive), and to *diagnose/repair* (here: nothing —
  external). Persist and audit each; the study's fresh-process continuation test is the template.
- **WARN — ungraded structure learning turns noise into structure.** Consistency with experience plus growth
  is not evidence of validity; a deterministic "explain everything" learner builds ≈ one new state per false
  label. A core whose weights are graded beliefs needs *graded structural* beliefs too (support per
  transition/merge), otherwise the grading stops exactly where concepts are formed.
- **WARN — a failure signal is not a diagnosis.** "My family emptied / my residual is systematic" means "at
  least one of {model bound, stationarity, measurement, attribution} is violated", nothing more (S4).
- **MECH to reuse.** (i) `state_merging.fit` as the simplest "concept formation = quotient of experience by
  future behaviour" primitive; (ii) version space with abstention and unanimity voting
  (`recurrent_state.py:101-116`) as an *honest* answerer; (iii) exact equivalence by product graph and the
  "toggle one unseen word" alternative builder (`evidence_audit.py:80-96`) — i.e. *imagining the nearest
  alternative construction that fits all evidence and measuring where it differs* — today evaluator-side, a
  natural internal "how real is my construction" probe; (iv) the erase/transport/fresh-process protocol.
- **OPEN.** Incremental (per-event) revision of the quotient with bounded learning memory; migrating a running
  state across a revised procedure (S7); deciding which archive items may be dropped without losing future
  distinctions (S5); self-detection of false evidence.

---

## 2. Study `living_learning_update_2026_09_20` — experience changes how later experience is learned

### (1) Question
Can what a system lived through leave a state that changes how the *same new evidence* will change it —
independently of any carried answer — and on which commonality between past and future does the benefit depend?
(`REPORT_TR.md:13`, `EXPERIMENT.md:7`)

### (2) Mechanism / state representation actually used
Numeric statistics; no symbolic structure, no neural net.

- **Functional split** `s = (w, m)`: `w` gives the current answer, `m` decides what a new label does to `w`
  (`REPORT_TR.md:42-49`, `UPDATE_THEORY.md:9-18`):
  `ŷ_t = x_tᵀ w_t`, `w_{t+1} = w_t + P_m x_t (y_t − x_tᵀ w_t)`, `∂w_{t+1}/∂y_t = P_m x_t`.
- **Stored:** `w ∈ R²`; `P` symmetric 2×2 as three numbers `[a, d, b]`; three forward-sensitivity vectors; an
  accumulated meta-gradient; a block counter (`learned_update.py:63-73`).
- **Base update** every sample: `w ← w + P x e` (`learned_update.py:78-79,88`).
- **Meta update** (the "learning of the update"), exact *within a block of 64 samples with P held fixed*:
  sensitivities `s^j = ∂w/∂m_j`, `s^j ← (I − P x xᵀ) s^j + E_j x e`, `g_j −= e xᵀ s^j`
  (`learned_update.py:80-87`; `EXPERIMENT.md:39-46`; boxed recursion `UPDATE_THEORY.md:60-63`); at the block end
  `P ← Π( P − (0.1/64)·[g_a, g_d, g_b/2] )` (`:91-96`; the ½ on the off-diagonal is required because one
  parameter represents two matrix entries), with `Π` = clipping of both eigenvalues to [.002, .6] (`:51-60`);
  sensitivities and gradient reset, `w` not reset (`:99-101`). Start `P = .08·I`.
  Scope of the derivative: block-local, not through all past changing gains (`EXPERIMENT.md:48`;
  `learned_update.py:282`).
- **What is transferred to a new life:** only the three numbers of `P`; `w = 0`, sensitivities, gradient and
  counter are fresh; no raw example, no old coefficient (`learned_update.py:171,174-187`); the code asserts that
  the first prediction of all nine methods is identical (`:197`).
- **World:** hidden 2-D target does a random walk, fast along a hidden direction (sd .04/step) and slow along
  the orthogonal one (sd .001); inputs uniform on the unit circle; label noise sd .15; order: target moves →
  input → prediction → noisy label (`learned_update.py:113-123`).
- **Independent model of the learner** (`analytic_risk.py:39-45,80-96`): with `C_t = E[d dᵀ]`, `d = θ − w`,
  `E[xxᵀ] = I/2`, `E[xxᵀCxxᵀ] = (tr(C) I + 2C)/8`:
  `M_P(C) = C − ½(PC + CP) + ⅛ P (tr(C) I + 2C) P + (r/2) P²`, `C_{t+1} = M_P(C_t) + Q`, expected clean
  error `tr(C_t)/2` (`EXPERIMENT.md:100-120`).

### (3) Experimental design
- **Units:** 24 independent source lives × 16,384 samples (hidden direction differs per source). After each
  source: 4 regimes × 4 new lives × 1,536 samples = 384 new lives, 589,824 samples. All nine methods see the
  same new samples and labels (`EXPERIMENT.md:13`). The independent unit for intervals is the source (24); the
  four new lives are averaged within a source first (`learned_update.py:231-235`).
- **Regimes** (`EXPERIMENT.md:19-24`): `same_geometry`; `reversed_geometry` (fast and slow directions swapped);
  `isotropic_change` (both fast); `cold_restart` (same geometry but unknown initial target, components
  N(0, .75²)). All fixed before the first full run.
- **Methods** (`EXPERIMENT.md:54-64`; `learned_update.py:174-187`): `initial_fixed` (.08·I); `learned_frozen`;
  `learned_continue` (meta-learning continues); `fresh_meta`; ablations of the transferred matrix —
  `diagonal_only`, `isotropic_same_trace`, `rotated_90`; two baselines that *also learn from the source* —
  `train_selected_scalar` (best of 8 scalars by observed source error) and `train_selected_matrix` (best of 32
  fixed candidates).
- **Metric:** mean squared error of the pre-label prediction against the clean target over all 1,536 steps;
  also first 64 / 65–512 / 513–end and the final query (`learned_update.py:199-203`).
- **Statistics:** approximate normal 95 % intervals over 24 sources; no multiplicity control; no pre-registered
  acceptance test (`EXPERIMENT.md:98`).
- **Checks:** meta-gradient vs central finite differences (max 4.94×10⁻¹²); whole-problem orthogonal rotation
  equivariance (1.94×10⁻¹⁶ over 512 steps; *not* invariance under general invertible recoding); fresh-process
  resume mid-block, 149 steps identical — which needs `P`, `w`, sensitivities, gradient and block position, not
  `P` alone (`REPORT_TR.md:84-86`); post-result independent second-moment computation.

### (4) Results and decision labels

Service MSE over the whole new life (`EXPERIMENT.md:74-86`; `results/learned_update.json › summary`):

| Method | Same structure | Reversed | Fast everywhere | Unknown start |
|---|---:|---:|---:|---:|
| initial_fixed | .011426 | .011251 | .021592 | .016174 |
| **learned_frozen** | **.005199** | **.034495** | .036181 | .013223 |
| learned_continue | .005179 | .011996 | .015551 | .012988 |
| fresh_meta | .008774 | .008623 | .016619 | .013219 |
| diagonal_only | .007207 | .014041 | .019104 | .011162 |
| isotropic_same_trace | .007762 | .007699 | .013704 | .010516 |
| rotated_90 | .034933 | .005220 | .038364 | .045334 |
| train_selected_scalar | .006868 | .006868 | .010456 | .008538 |
| train_selected_matrix | .005333 | .067916 | .063193 | .023619 |

- Learned gain along the true fast direction **.261987**, along the slow one **.025604** (both started at .08;
  the direction was never given) (`EXPERIMENT.md:88`). Source cost: 16,384 base updates + 49,152 sensitivity
  updates + 256 meta-blocks; the grid baseline does 524,288 base updates (`EXPERIMENT.md:70`).
- Paired differences, `learned_frozen − fresh_meta`: same **−.003575** [−.003693, −.003456]; reversed
  **+.025872** [.023307, .028438]; fast-everywhere **+.019562** [.017142, .021982]; unknown start
  **+.00000377** [−.001554, +.001561].
- `learned_frozen − train_selected_scalar`: same −.001670 [−.001707, −.001632]; unknown start **+.004685**
  [.002856, .006514] (scalar better).
- `learned_continue − fresh_meta`: reversed **+.003373** [.003277, .003469] (harm reduced from .034495 to
  .011996 but not removed); fast-everywhere **−.001068** [−.001252, −.000883] (continuing from the transferred
  matrix is slightly *better* than starting fresh — in the JSON, not discussed in the prose).
- Unknown start, by time (JSON): first-64 MSE learned_frozen **.164766**, fresh_meta .129302, scalar
  **.047019**; steps 513–end learned_frozen **.005193** (best), scalar .006883. The learned rule is a good
  tracker and a slow acquirer.
- Analytic vs observed (`EXPERIMENT.md:122-128`): learned/same .00519828 vs .00519868; learned/reversed
  .03453095 vs .03449503; learned/fast-everywhere .03792460 vs .03618068; learned/unknown-start .01367785 vs
  .01322295; scalar/unknown-start .00851145 vs .00853796. For all 28 method × regime pairs with a frozen matrix
  the interval of the observed-minus-expected gap contains 0 (`results/analytic_risk.json`). Formula vs 16-point
  circle quadrature: 5.55×10⁻¹⁷.

| Claim | Label | Evidence | Reference |
|---|---|---|---|
| A state acquired from experience can change the effect of new evidence without carrying any answer | **supported** | [EMP n=24] −.003575 vs fresh meta with identical first predictions; removing/rotating the direction information removes or inverts the gain (+.002009 diagonal, +.002563 isotropic, +.029734 rotated) | `EXPERIMENT.md:90-92` |
| The acquired update helps new problems in general | **refuted** | [EMP n=24] +.025872 reversed, +.019562 fast-everywhere | `EXPERIMENT.md:94` |
| Continuing to adapt removes the harm of a wrong habit | **refuted as a full fix** | [EMP n=24] still +.003373 vs fresh over the life | `EXPERIMENT.md:94` |
| The acquired update gives better acquisition of an unknown initial relation | **not shown** ("üstünlük gösterilmedi") | [EMP n=24] +.00000377 [−.001554, +.001561]; scalar better by .004685 | `EXPERIMENT.md:96` |
| Online meta-learning is uniquely better than picking a fixed matrix from source data | **not established** | −.000134 [−.000229, −.000039] vs a coarse 32-candidate grid | `EXPERIMENT.md:90` |
| Total-life efficiency | **not claimed** | acquisition cost reported, not amortised | `REPORT_TR.md:38` |

### (5) Laws / invariants claimed
- **U1 — Same answer today ≠ same learning tomorrow.** [EXACT + EMP] Two systems with identical predictions can
  differ in `∂w_{t+1}/∂y_t = P_m x_t`. "Aynı bugünkü tahmin, aynı yarınki öğrenme kapasitesi demek değildir"
  (`REPORT_TR.md:49`). The split `(w, m)` is functional, not architectural: both may live in one big state
  updated by one fixed program; the distinction is tested by which future computations an erase/transport
  intervention changes (`REPORT_TR.md:51`). No infinite tower of meta-learners is required.
- **U2 — A transferred way of learning is a bet on the future's change geometry.** [EMP n=24] The same three
  numbers help (−.003575), hurt (+.025872) or do nothing (+.00000377) depending on whether the new life shares
  the source's change directions and start condition; rotating the matrix by 90° swaps the outcomes
  (.034933 ↔ .005220). "Öğrenilmiş bir öğrenme alışkanlığı da düzeltme maliyeti yaratabiliyor"
  (`REPORT_TR.md:34`).
- **U3 — Responsiveness has an exact price list.** [EXACT, scalar tracking model] `θ_{t+1}=θ_t+ν`, `y=θ+ε`,
  `w←w+α(y−w)`, variances `q`, `r`: `S_∞(α) = (q + α² r)/(2α − α²)`, optimum `α* = 2q/(q + √(q²+4qr))`;
  finite life: `S_t = aᵗ S_0 + S_∞(1−aᵗ)`, `a=(1−α)²`, so the best finite-horizon rate also depends on how
  good the starting knowledge is; the `q→0` limit says "α→0", yet α = 0 never learns — the limits do not
  commute (`UPDATE_THEORY.md:77-129`). Neither "always keep" nor "always move fast" is a rule.
- **U4 — Local correction is not validity.** [EXACT] After an update the same sample's residual is
  `e⁺ = (1 − xᵀPx) e`; `0 ≤ xᵀPx ≤ 2` guarantees the *sample's* error does not grow. Counterexample: true
  target 0, `w=0`, `x=1`, one wrong label `y=1`, `P=1` ⇒ sample error 1→0, true error 0→1
  (`UPDATE_THEORY.md:133-143`).
- **U5 — Expressive limits of cheap update families.** [EXACT] Scalar gain: `‖η(u+v) − v‖² ≥ ½` for all `η`
  while `K = vvᵀ` is exact; diagonal gain: `‖Dg − vvᵀg‖² ≥ ¼` for `g=(1,0)`, `v=(1,−1)/√2`; diagonal matrices
  are not preserved by rotation `K' = RKRᵀ` (`MEMORY_AND_SCOPE_AUDIT.md:13-39`).
- **U6 — What to keep depends on what will be asked.** [EXACT] Two fair bits, room to keep one, query asks bit 1
  with probability `p`: errors `(1−p)/2` vs `p/2`; at `p=.9`, .05 vs .45, and the ranking flips with the query
  distribution (`MEMORY_AND_SCOPE_AUDIT.md:41-49`). "Korunması gereken bilgi, ileride yapılacak işe bağlıdır."
- **U7 — Time scale of the past.** [EXACT, restricted] with fixed `P`, iid unit-circle inputs and a stationary
  target, the old start error decays by `(1−λ/2)ᵗ` along an eigen-direction of gain `λ`
  (`MEMORY_AND_SCOPE_AUDIT.md:97`): a small gain is exactly "long retention of the old state".
- **U8 — The learner's risk can be computed without living it.** [EXACT model + EMP agreement] the
  second-moment recursion predicts the finite-time risk of any frozen `P` in any of the four worlds and matches
  the runs. It does not cover the two adaptive methods.

### (6) Limits
**Admitted.** Two dimensions, linear predictor whose function class never grows; given observation vector,
loss, matrix family, meta-procedure, 64-step horizon and sensitivity reset; labels are correctly aligned
targets; no raw-memory choice, no representation discovery, no action; matches IDBD / meta-gradient literature
— no new principle (`EXPERIMENT.md:134-146`, `REVIEW.md:68-75`). "Güncelleme hızını daha fazla iyileştirmeyi
araştırmanın merkezi yapmamak gerekir" (`REPORT_TR.md:80`).

**Limits I see [ANALYST].**
1. The transferred object is a **steady-state gain**, i.e. it encodes "how fast the world moves" but not "how
   much I currently do not know". That is exactly why it loses at the unknown start (first-64 MSE .164766 vs
   .047019) while being the best tracker afterwards (.005193). A belief state (mean + covariance) with the same
   transferred drift model removes that negative on the study's own streams (§8, check B: .005596 vs .013223,
   best study method .008538).
2. The same check shows the brittleness is worse, not better, for a *sharper* transferred belief: an exact
   source drift model applied to the reversed world gives .0867 (study's learned matrix .034495, no transfer
   .008623). The study's learned matrix was simply less committed than the exact filter: judging by those two
   risks it responds about 2.5× more along the old slow direction (learned gain .0256; eigenvalues floored at
   .002; credit truncated at 64 steps). Its smaller loss under reversal is a by-product of that, not a design.
3. Source life is 10.7× longer than a new life; four hand-picked regimes; all noise Gaussian; the mismatch is
   never detected by the learner — `learned_continue` cannot react before its first block ends (first-64 MSE is
   identical to the frozen rule by construction).

### (7) Implications for the core
- **REQ — measure learning, not answers (U1).** Any claim that the core "learns better because of its past"
  must use this protocol: identical initial answers, identical new evidence, only the carried state differs,
  and resetting that state must remove the effect.
- **REQ — the "way of constructing" that transfers must itself be a graded, revisable belief with a declared
  scope (U2).** A transferred habit without a confidence and without a hedged fallback is a liability: of the
  seven source-informed methods, only the two *direction-free* ones were never worse than fresh learning in all
  four worlds — one scalar picked on the source (.006868 / .006868 / .010456 / .008538) and the learned matrix
  reduced to its trace (.007762 / .007699 / .013704 / .010516), against fresh .008774 / .008623 / .016619 /
  .013219. Every method that carried a direction lost in at least one world.
- **REQ — a graded belief needs two different quantities: current uncertainty and expected change (U3, U7).**
  One number (a rate) cannot serve both acquiring an unknown and tracking a drifting known.
- **WARN — bounding the update protects the sample, not the truth (U4).** Graded weights do not by themselves
  handle false evidence.
- **MECH to reuse.** (i) forward-sensitivity meta-update with spectral projection (`learned_update.py:77-101`)
  as a minimal, exactly checkable "learning-rule state"; (ii) the ablation set for a transferred state
  (delete off-diagonal / equalise / rotate) as a causal test of *what* was transferred; (iii) the second-moment
  recursion (`analytic_risk.py:39-45,80-96`) as a prototype of *imagination with a measurable error*: the
  consequence of adopting an update rule in an assumed world is computed, then compared with the lived outcome;
  (iv) full-state resume test showing which fields are needed to continue.
- **OPEN.** Detecting from the stream that a transferred habit no longer fits; learning the change model at
  acceptable cost; whether any of this survives a growing function class.

---

## 3. Study `living_learning_joint_2026_09_20` — one life, shared state, no ready identifiers

### (1) Question
Can acquisition, retention and correction run together on one shared changing state without task names or
change flags, and can the relation between an outcome and several past inputs be learned without a ready
per-outcome experience identifier? (`REPORT_TR.md:5-16`)

### (2) Mechanism / state representation actually used
Numeric statistics over a given finite feature language; no neural net.

**Experiment A — joint stream (`joint_stream.py`).**
- Predictor: one shared weight vector over the active subset of the supplied language
  `[1, x, z, x·z, max(x,0)·z]` (`:18,37-39,111-112,127-130`).
- Memory variants: (a) *regional*: a supplied 4×4 grid over `(x,z)`, each cell keeps its 8 newest rows
  `[x, z, y, serial]`, ≤128 raw records (`:49-52,113,176-179`); (b) *rolling*: the 128 newest rows
  (`:171-174`); (c) *all-history*: Gram 5×5 `G=Σφφᵀ` and cross `b=Σφy`, no raw rows (`:115-116,169-170`).
- Update: every 16 observations, ridge solve (λ = .001) on the current memory (`:181-189`, solver `:55-77`).
- "Concept when needed" (`growth_coverage`, `consider_growth`, `:132-162`): starts with `[1,x,z]`; with ≥48 rows
  and ≥12 per fold, split memory by parity of the internal sample counter; for each inactive candidate, two-fold
  cross-fit; activate the best candidate if the gain is ≥ .001 in *both* folds and the mean gain ≥ 10 % of the
  base loss (`:156`). Activated terms are never removed (`growth_structural_deletion: False`, `:381`).
- World: `f0 = .5 + .7x − .4z + .8xz`; from sample 1025 on, `f1 = f0 − 1.6·max(x,0)·z` (only the right side
  changes) (`:42-46,225`). Seven phases of 256 samples: left acquisition, right acquisition, left recurrence,
  right label burst + recovery, right real change, left retention, right changed recurrence (`:34`);
  `x ∈ ±[.2, 1]`; label noise sd .1; **samples 769–800 carry the labels of the future changed target while the
  world has not changed** — a structured false burst (`:229-230`).

**Experiment B — temporal relation (`temporal_relation.py`).**
- State: `w` (8 numbers), `P` (8×8 = 64 numbers) (`:32-45`), plus the last 7 inputs, kept by the driver and
  saved with the model (`:142`).
- Rule: recursive least squares with forgetting `ρ` (`:47-58`; `TEMPORAL_RELATION.md:17-21`):
  `k = Pφ/(ρ + φᵀPφ)`, `w ← w + k (y − wᵀφ)`, `P ← (P − k φᵀP)/ρ`, `w0 = 0`, `P0 = I`, `ρ ∈ {1, .995}`,
  `φ_t = (1, u_t, …, u_{t−6})`.
- World: `y_t = .2 + 1.1 u_{t−2} − .7 u_{t−5} + ξ`, `ξ ~ N(0, .05²)` (`:73-78`).

**Theory (`PRINCIPLE_AUDIT.md`)** — three propositions and two counterexamples, see (5).

### (3) Experimental design
- **A:** 24 paired lives × 1,792 observations; predict → outcome → update; five methods share the stream
  (`JOINT_STREAM.md:37-45`): growing + regional; expanded-from-start + regional; expanded + rolling; expanded +
  all-history statistics; fixed base + regional. Per life 400 held-out inputs per side whose labels never enter
  learning. Metrics: whole-life service MSE against the clean target, terminal left/right MSE, phase and
  checkpoint MSEs. Extra: (i) a counterfactual "hidden global change" evaluation — the worlds "right changed"
  and "both sides changed" give identical observations during 1025–1280 (`joint_stream.py:280-281,310-316`);
  (ii) a **post-result** ablation removing only the 32 false labels (`joint_ablation.py:25-31`), declared as a
  mechanism check, not confirmation; (iii) JSON round trip at 896 then 896 further updates identical; fresh
  OS-process continuation of 64 updates for each method (`verify_joint.py:67-105`).
- **B:** four regimes × 24 paired lives × 1,800 observations (`TEMPORAL_RELATION.md:29-40`): iid stationary; iid
  with a coefficient change at t = 900 (1.1, −.7 → −.4, 1.2); strictly alternating inputs ±1; second effect at
  lag 10 (outside the 0–6 vocabulary). Methods: no learning; current input only; all lags ρ = 1; all lags
  ρ = .995; a privileged reference that knows the two true lags. 512 fresh independent histories per life for
  evaluation. Choices fixed before results; inputs are not chosen by the learner.
- **Statistics:** approximate normal 95 % intervals over 24 lives; single configuration.

### (4) Results and decision labels

**A — joint stream** (`JOINT_STREAM.md:51-59`; `results/joint_stream.json`):

| Method | Whole-life service | Final left (old rule still valid) | Final right (rule changed) |
|---|---:|---:|---:|
| growing + regional | .012425 | .000478 | .000417 |
| expanded + regional | **.012216** | .000478 | .000417 |
| expanded + rolling window | .029270 | **.204387** | .000293 |
| expanded + all-history statistics | .048933 | .000052 | **.078358** |
| fixed base + regional | .056672 | .012557 | .012463 |

- Growing minus expanded, service: **+.000209** [+.000134, +.000283] (in favour of the fixed expanded model);
  linear solves **320.17 vs 112**; moment multiply-adds 613,810 vs 392,106 (`JOINT_STREAM.md:61-63`).
- Every life activated both `x·z` and the hinge. Hinge activation steps (JSON): 16 of 24 inside the false-burst
  window (784–816), 7 after the real change (1040–1136), 1 spurious at 272.
- False burst: right-side MSE **.000464 → .132789 → .000673** at samples 768 / 800 / 896; the all-history
  learner is more robust to the burst (.005082 at 800) but slow on the real change (`JOINT_STREAM.md:69`).
- Post-result ablation without the burst (`results/joint_no_burst.json`): hinge active before the real change in
  **1/24** lives (was 17/24 counting the spurious one); same final right MSE **.000417**; service in the
  real-change phase .042377 → .044044; growing now needs 364.83 solves and is worse than expanded by +.000613
  [+.000511, +.000716].
- Hidden global change: any predictor that is the same in the two indistinguishable worlds has average left MSE
  ≥ gap/4 = **.088415**; the regional learner, which keeps the old left rule, would have **.355531** at sample
  1280 if the left had changed too (`JOINT_STREAM.md:75`).
- Reset state: left .304707, right 1.152122 — everything measured comes from the lived state.

**B — temporal relation** (MSE on fresh independent histories; `TEMPORAL_RELATION.md:44-52`):

| Method | iid, stationary | coefficient change | after alternating input | effect outside the lag vocabulary |
|---|---:|---:|---:|---:|
| no learning | .609923 | .570297 | .609923 | .613588 |
| current input only | .569447 | .531311 | 1.651309 | .572413 |
| all lags, all history | **.00001344** | .488246 | **.415140** | .165005 |
| all lags, forgetting .995 | .00005410 | **.00029226** | .415222 | .167184 |
| knows the true lags (privileged) | .00000565 | .487417 | .026810 | .00000587 |

- Alternating inputs: last-200-step service MSE **.00000318** but **.415140** on new independent histories
  (`TEMPORAL_RELATION.md:60`). The learned lag weights are ±.2568 on all seven lags (JSON): the minimum-norm
  spread of the total effect 1.8 over 7 lags.
- Forgetting: under change, whole-life service .492554 → .114709, paired −.377846 [−.383780, −.371911];
  in the stationary world final error is higher by +.00004066 [+.00003160, +.00004973]
  (`TEMPORAL_RELATION.md:56`).
- Outside vocabulary: floor `.7²·Var(u) = .49/3 = .163333`; measured .165005 (`TEMPORAL_RELATION.md:64-70`).
- Keeping `w` but resetting `P`: identical answers to all probes at the reset, then different learning — mean
  maximal future prediction difference **.067295** (stationary) and **2.742197** (changing)
  (`TEMPORAL_RELATION.md:76`; `temporal_relation.py:144-146`).
- RLS vs an independent direct ridge solve: max difference 3.33×10⁻¹⁶ (`verify_joint.py:47-66`).

| Claim | Label | Evidence | Reference |
|---|---|---|---|
| Acquisition, retention and correction can run on one shared state with no task id / change flag | **supported as an existence witness** | [EMP n=24] final .000478 / .000417 | `JOINT_STREAM.md:5-7` |
| Forming the features when needed is better than having them from the start | **not supported — reversed** | [EMP n=24] +.000209 service, 320 vs 112 solves | `JOINT_STREAM.md:61-63`, `ADVERSARIAL_REVIEW.md:9` |
| Regional retention solves "which past is still valid" | **refuted** | [EXACT] two worlds identical on the observed prefix; bound .088415; learner .355531 | `JOINT_STREAM.md:75-77` |
| Working under noise = recognising false information | **refuted** | [EMP n=24] .000464 → .132789 after 32 structured false labels | `JOINT_STREAM.md:69` |
| The false burst was what made the later adaptation succeed | **refuted as necessary** (post-result check) | same final .000417 without it | `JOINT_STREAM.md:71-73` |
| Every outcome needs a ready experience identifier | **refuted for the linear lag family** | [EMP n=24] .00001344 vs .569447 | `TEMPORAL_RELATION.md:54` |
| Success on the lived stream ⇒ the relation is good for unseen continuations | **refuted** | [EMP n=24] .00000318 vs .415140; [EXACT] alias `y=u_t` vs `y=−u_{t−1}` | `TEMPORAL_RELATION.md:60-62` |
| Forgetting is always an improvement | **refuted** | [EMP n=24] helps under change, hurts when stationary | `TEMPORAL_RELATION.md:56` |
| Good temporal prediction ⇒ causal discovery | **refuted** | [EXACT] structural alias: probability of Y=1 under do(A=1) is .5 in one world and 1 in the other, observations identical | `PRINCIPLE_AUDIT.md:52-59` |
| Randomness is needed to identify the relation | **refuted** | [EXACT] deterministic pulse design has rank 8 | `TEMPORAL_RELATION.md:82`, `verify_joint.py:106-110` |
| Updates that are each good are good together | **refuted** | [EXACT] `R(a,b)=(a+b−1)²` | `PRINCIPLE_AUDIT.md:73-81` |
| Explicit scalar optimisation is logically necessary for learning | **refuted** | [EXACT] version-space construction | `PRINCIPLE_AUDIT.md:101-117` |

### (5) Laws / invariants claimed (all mapped by the study to known literature)
- **J1 — Identifier-free acquisition, conditional (Proposition 1).** Keep `G_n = Σ z zᵀ`, `b_n = Σ z y`;
  `θ̂_n = (λI + G_n)⁻¹ b_n`. Exact identity `θ̂_n − θ* = (λI+G_n)⁻¹(Σ z ξ − λθ*)`, hence
  `‖θ̂_n − θ*‖ ≤ (‖Σ z ξ‖ + λ‖θ*‖)/(λ + λ_min(G_n))`; converges if `λ_min(G_n) ≥ κ n` and
  `‖Σ z ξ‖ = o(n)` (`PRINCIPLE_AUDIT.md:17-38`). Storage `O(L²)`, no raw history. Scope: linear relation, given
  lag bound, order and alignment; the noise condition is an assumption.
- **J2 — What experience never excited stays undetermined (Counterexample 1).** If every lived `z_t` satisfies
  `z_tᵀ v = 0` for some `v ≠ 0`, then `θ` and `θ + v` are observationally identical however long the stream;
  they separate only on a query with `zᵀv ≠ 0` (`PRINCIPLE_AUDIT.md:46`). Measured instance: the
  alternating-input regime.
- **J3 — Separately good updates can be jointly bad (Counterexample 2).** `R(a,b) = (a+b−1)²` from `(0,0)`:
  `a←1.5` alone 1 → .25, `b←1.5` alone 1 → .25, both → 4. Writing to separate memory does not help: they meet at
  the shared output (`PRINCIPLE_AUDIT.md:73-81`). The programme's own one-line form: "Parça başarıları birleşik
  sistem başarısı olarak toplanamaz" (update `REPORT_TR.md:65`).
- **J4 — Conditional composition budget (Proposition 2).** If each applied step raises each fixed risk `R_j` by
  at most `ε_{t,j}` on an event of conditional probability ≥ `1−δ_t`, then with probability ≥ `1−Σδ_t`:
  `R_j(s_T) ≤ R_j(s_0) + Σ_t ε_{t,j}`. No independence needed. Not a learning algorithm (never updating
  satisfies it); adaptively reused validation data break the precondition; a changing target makes the fixed
  `R_j` the wrong thing to protect (`PRINCIPLE_AUDIT.md:83-97`).
- **J5 — Consistent monotone expansion without an optimiser (Proposition 3).** Finite executable candidate set
  `H ∋ h*`, error-free observations, `V_t = {h ∈ V_{t−1}: h(x_t) = y_t}`, answer only when all of `V_t` agree:
  `h*` is never removed, every answer given is correct, an answerable query never becomes unanswerable or
  changes its answer. Breaks under one false observation (truth deleted irreversibly), world change, or a truth
  outside `H` (`PRINCIPLE_AUDIT.md:105-117`).
- **J6 — Three kinds of attribution.** Predictive / interventional / single-event counterfactual; the first does
  not give the second without structural assumptions (D1: `Y=U`, D2: `Y=A`, same observations;
  `PRINCIPLE_AUDIT.md:52-67`). An outcome produced by several past inputs has no single "responsible"
  experience — "bind the feedback to the one right event" can be the wrong problem statement
  (`TEMPORAL_RELATION.md:5`).
- **J7 — A retention policy is a hypothesis about how the world changes.** [EMP + EXACT] rolling window loses
  the still-valid left rule (.204387); all-history keeps the invalid right rule (.078358); region-wise refresh
  gets both (.000478 / .000417) *because* the supplied grid boundary coincides with the world's change boundary
  at `x = 0` (`JOINT_STREAM.md:27,77`); under the hidden global change the same policy keeps a wrong rule.
- **J8 — Indistinguishable-worlds bound.** [EXACT] For two worlds that agree on everything observed and differ
  by `Δ` on a query, any single prediction has average squared error ≥ `Δ²/4` over the two; .088415 here
  (`JOINT_STREAM.md:75`; same bound for two contradictory targets, `CREDIT_ASSIGNMENT.md:90`).
- **J9 — Savunulabilir koşullu edinim ilkesi** (the "defensible conditional acquisition principle",
  `PRINCIPLE_AUDIT.md:146`): in the estimation and candidate-elimination families, experience creates
  generalising capacity exactly when it actually discriminates between candidates and the persistent update
  uses that discrimination. Explicitly *not* the necessary form of all capacity change (compilation without new
  evidence is a different process, `:148`).
- **H4 (method condition, `PRINCIPLE_AUDIT.md:142`).** Evidence for a capacity increase must be independent of
  the mechanism's own acceptance criterion.

### (6) Limits
**Admitted.** Supplied: numeric observations, the feature / lag language, time order, ridge and thresholds, the
4×4 grid aligned with the change boundary, the forgetting factor. The growth thresholds are heuristics with
reused folds, no confidence guarantee (`JOINT_STREAM.md:33`). Counters are not CPU cost; the feature routine
computes all five values anyway (`JOINT_STREAM.md:65`). The burst ablation is post-result. Not tested: learning
the grid, other curricula, action-generated data, unknown time scales, anonymous labels, false-label vs
real-change discrimination (`ADVERSARIAL_REVIEW.md:20`). The first separate-process check failed because the
*checker* shared a state copy; fixed in the checker, outputs untouched (`ADVERSARIAL_REVIEW.md:18`).

**Limits I see [ANALYST].**
1. "Concept formed when needed" here means choosing 1 of 2 supplied candidates. The trigger fired 16–48 samples
   after the *false* burst began (16 lives) and 16–112 samples after the *real* change (7 lives; in the
   burst-free control 16–48 samples in 22 of 24 lives, 112 in one, one spurious activation at sample 272) — it is
   as quick for false evidence as for true, because both validation folds are cut from the same memory and share
   the false regularity.
2. Concepts are never retired.
3. The evidence age is stored in every row (the serial) but used only to split folds
   (`joint_stream.py:137,168`); nothing in the learner represents "my left-side evidence is 500 steps old".
4. The RLS state already contains the answer to "is this construction backed by experience for *this* query":
   on the study's own alternating-input lives the remaining-uncertainty form `φᵀPφ/φᵀφ` is .00014 on the lived
   stream and .585 on new independent histories, while in the iid lives both are .0013 (§8, check A). The
   learner never reads it out.
5. One fixed curriculum; the 24 lives differ only in noise and input draws.
6. The committed `results/verification.json` was assembled from two partially failed verifier invocations (its
   field names differ from what `verify_joint.py:30-41` writes; stated in the file's `reproduction_evidence`
   and in `REPORT_TR.md:87`). Hashes are consistent; it is not a single clean end-to-end verifier run.

### (7) Implications for the core
- **REQ — every construction needs an "excited-by-experience" measure separate from its fit (J2).** Fit on the
  lived stream (.00000318) said nothing about the unlived directions (.415140). "How real is this" must include
  how much of the *query* is covered by evidence.
- **REQ — every kept belief needs a validity component separate from its support (J7, J8).** Without an
  explicit, graded assumption about how the unvisited world drifts, retention is an unexamined bet; with one,
  the bet is at least visible and its worst case (`Δ²/4`) computable.
- **REQ — accept changes against the current joint state with a running budget (J3, J4).** Per-mechanism
  acceptance tests on a stale state do not compose.
- **WARN — on-demand growth is not an advantage in itself.** With the candidate language given, growing cost
  2.9× the solves for the same final skill and slightly worse service. Its case has to be made where the
  candidate space cannot be pre-enumerated — not tested anywhere in this group.
- **WARN — self-generated validation does not see structured falsehood**; and forgetting, retention and
  growth each help in one world and hurt in another.
- **MECH to reuse.** (i) `(w, P)` recursive statistics as an already-graded belief: constant memory, exact
  equivalence to batch ridge, and `P` doubles as an identifiability gauge; (ii) region-wise newest-k memory as
  a bounded store that protects unvisited regions; (iii) two-fold activation rule as a cheap "do I need a new
  concept" trigger (needs an independent validity guard); (iv) version-space answering with abstention (J5) as
  the discrete counterpart of (i); (v) the indistinguishable-world construction as an evaluation tool;
  (vi) `w`-only vs `(w,P)` restore as the test that a persistence scheme preserves learning, not only answers.
- **OPEN.** Learning the retention geometry, time weights and history vocabulary from experience (named by the
  study as its most prominent gap, `REPORT_TR.md:81`); separating false evidence from real change;
  interventional attribution; composition at longer horizons under resource pressure.

---

## 4. Paper 04 (CEV-2026-04) — representation growth, credit assignment, transport of learning state

Edition of `docs/research/living_learning_reassessment_2026_09_20/REPORT_TR.md`; details in
`REPRESENTATION_GROWTH.md`, `CREDIT_ASSIGNMENT.md`, `STATE_TRANSPORT.md`, `SCOPE_AND_FOUNDATIONS.md`; code in
`research/living_learning_reassessment/{representation_growth,credit_assignment,state_transport}.py`.
(The same underlying study is digested from the study side in the sibling file
`/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/prior/g1-foundations.md`, §6;
the shared numbers agree.)

### (1) Question
Beyond re-testing: can experience create new executed features and compositions, can delayed feedback be written
to the right place while valid old skill in shared parameters survives, and does a state that preserves today's
answers also preserve tomorrow's learning — including across a change of representation? (paper 04 `:47-49`)

### (2) Mechanisms
- **(a) Constructive features** (`representation_growth.py`). Exact arithmetic in F₁₀₁, `x ∈ F₁₀₁⁵`; base
  features `(1, x0..x4)`; a learned feature is a node storing **two parents and executed by multiplying them**
  (`:121-149,176-181`), e.g. `φ6 = x3·x4`, then `φ7 = φ6·x0`; one linear head per (given) task id (`:85`).
  `observe` (`:154-188`): predict; if a head exists and the prediction ≠ label → **delete the head at once**
  (`:157-160`); otherwise buffer rows; at 16 rows `discover` (`:110-119`): exact fit by Gaussian elimination
  mod 101 over active features (`fit`, `:47-77`), else try each candidate of the generator
  `G(Φ) = {φ·x_i : φ ∈ Φ, i ∉ vars(φ), deg φ < 3}` in degree/coordinate order (`:91-108`) and take the first
  consistent one; validate on the next 4 rows (`:169-171`); at 20 rows promote feature + head or record
  failure; the raw buffer is deleted (`:172-187`). Max 20 raw rows.
- **(b) Delayed feedback, shared parameters** (`credit_assignment.py`). One `w ∈ R³` for `x = (1,u,v)`;
  normalised LMS `w ← w − .12 (wᵀx − y) x/‖x‖²` (`:63-67`); online ridge by Sherman–Morrison holding `G⁻¹`
  (`:69-80`); feedback arrives as *(episode id, noisy label)* with delay 1..31 (`:150-151`); pending table
  id → x, capacity 64 (`:152-155`); reservoir replay of 32 `(x, y)` (`:166-175`); methods: write to the present
  input (`current_two`), to the right past input once/twice (`anchored_one/two`), right input + one replay
  (`anchored_replay`), RLS (`:82-97`).
- **(c) State transport** (`state_transport.py`). Old coordinates `x`, new `z = Ax` with a supplied invertible
  `A` (`:35-37`); update `w ← w − η·M·(wᵀx − y)x` (`:40-43`); weights-only transport `v = A⁻ᵀw` (`:69`) with
  `M = I`; full transport adds `M_z = A⁻ᵀ P A⁻¹` (`:37,86`); lossy-migration witness computed from exact
  integer summaries (`:107-137`).
- **Foundations** (`SCOPE_AND_FOUNDATIONS.md`): Propositions 1–2 and the relation `≡_{Z,Q}`, see (5).

### (3) Design
- (a) 20 seeds; three tasks per life (a pair product, a triple that uses that pair, an independent pair), 20
  labelled observations per task, 100 disjoint held-out inputs per task; four methods: fixed linear; growing;
  fixed expanded (26 monomials) with sparse search; fixed expanded with the *same activation policy* as growth.
  Interventions: reset, node deletion, random labels, one wrong label, rule shift under the same task id,
  triple task without its prerequisite pair (`REPRESENTATION_GROWTH.md:43-54`).
- (b) 32 paired seeds; 1,200 requests, two input distributions (`v = 0`, then `v = u`), same true rule
  `.3 + 1.2u − 1.7v`; label noise sd .1; five regimes: compatible delayed / immediate, insufficient pending
  capacity (delay ≤127, capacity 4), incompatible drift (slope +1.2 → −1.2), misspecified (`+.8u²`); 1,024
  held-out old, new and "composition" (independent `u,v`) inputs; Student-t 95 % intervals, 31 df; equal
  gradient budget asserted (`CREDIT_ASSIGNMENT.md:18-54`; `credit_assignment.py:196-198`).
- (c) Two regimes × 32 paired starts; 96 experiences before migration, 192 after; η = .04, noise .02; 256
  held-out queries (`STATE_TRANSPORT.md:34-44`).

### (4) Results and decision labels
- **(a)** Held-out correct: fixed linear 0/6,000 (all abstentions); growing **6,000/6,000** with 9 stored
  feature definitions; both fixed-expanded variants 6,000/6,000. Growing and "fixed expanded, same activation"
  were **identical in every seed** in predictions, acquisition events, retention, feature order and all work
  counters (489 candidate fits each; the sparse variant 565) (`REPRESENTATION_GROWTH.md:58-69`). Node deletion:
  the pair task and the dependent triple task both go to 100/100 abstention, the independent task stays 100/100.
  Random labels never promote a model. **One wrong label** removes the correct head in every seed (0 correct,
  2,000 abstentions on that task; others intact). Real rule change: 1 wrong answer + 19 abstentions, then
  100/100 on the new rule. **Curriculum counterexample:** triple without its pair — growing **0/2,000**, fixed
  expanded sparse **2,000/2,000** (`:77-83`).
- **(b)** Final held-out MSE, delayed main world (`CREDIT_ASSIGNMENT.md:58-68`):

  | Method | old inputs (rule still valid) | new inputs | unseen combinations |
  |---|---:|---:|---:|
  | writes to the present input | .495545 | .101597 | 1.494124 |
  | right input, two steps | **.216505** | .001233 | .434709 |
  | right input + replay (32) | .000957 | .000767 | .001725 |
  | online RLS (no raw examples) | **.000070** | .000047 | .000241 |

  Old-skill MSE with correct attribution alone: **.000905 → .216505** although one weight vector fits both
  phases. Wrong minus right attribution on new inputs: .100364 [.086869, .113858]. Right-two minus replay on
  old inputs: .215548 [.208600, .222497]. Price: phase-2 service .051818 (replay) vs .034694. Insufficient
  pending capacity: only 30.031 labels attributable, 1,107.219 discarded; gains vanish. Incompatible drift,
  new-input MSE: right-two **.001106**, replay **.181190**, RLS **.509150**. Misspecified: floor
  `.8²·Var(u²) = .056889`, RLS .057802 (`:86-98`).
- **(c)** At transport the predictions agree to 4.44×10⁻¹⁶. Weights-only afterwards: mean maximal prediction
  difference **.440868** (stationary) / **2.568257** (changed), final MSE **.027625 vs .000351** and
  **1.110122 vs .010339**. Weights + metric: whole trajectories equal to **2.22×10⁻¹⁵**
  (`STATE_TRANSPORT.md:46-54`). Lossy migration: histories `y = x1x2` and `y = −x1x2` on `{−1,+1}²` have the
  same complete old summary (`G = 4I₃`, `b = 0`, `Σy² = 4`, `n = 4`) but need `b' = (0,0,0,±4)` after adding
  `x1x2`; without extra information the worst-world MSE is ≥ 1 (`:60-85`).

The paper's own status table (paper 04 `:108-116`):

| Claim | Label in the paper |
|---|---|
| If current predictions are preserved, later learning is preserved | **yanlışlandı** (refuted, by invertible recoding) |
| A summary that is fully sufficient for a fixed model transports exactly to every new representation | **yanlışlandı** |
| Binding experience to the right place suffices to keep valid old skill | **yanlışlandı** |
| Replaying raw examples is the necessary retention mechanism | **yanlışlandı** (online sufficient statistics do it) |
| Growing a representation is inherently better than an equally expressive fixed one | **desteklenmedi; genel biçimi yanlış** |
| A feature that is useless today has no value for any future skill | shown dangerous by the curriculum counterexample; efficient general solution open |
| A state that respects future acquisition and correction serves wider continuation families than today's answer | **dar karşı örnekler ve yapıcı durumlarla destekli araştırma hipotezi** |

Status line of the growth experiment (`REPRESENTATION_GROWTH.md:109`): "MEKANİZMA BİLİNİYOR / SINIRLI TEMSİL
BÜYÜMESİ DENEYLE DESTEKLİ / GÜÇLÜ SABİT KARŞILAŞTIRMAYLA EŞDEĞERLİK GÖZLENDİ / MÜFREDAT VE GÜRÜLTÜ
BAŞARISIZLIKLARI KORUNDU / GENEL YAŞAM BOYU ÖĞRENME ÇÖZÜLMEDİ."

### (5) Laws / invariants claimed
- **R1 — Exact online updatability (Proposition 1).** A summary `c` of the history admits an update
  `c(hz) = u(c(h), z)` **iff** `c(h) = c(h') ⇒ c(hz) = c(h'z)` for every permitted new experience `z`
  (`SCOPE_AND_FOUNDATIONS.md:29-43`; paper 04 `:86-92`). Proof: representative-independence. It is an existence
  statement ("işlevsel varlık önermesi") — it does not make `u` computable, learnable or cheap.
- **R2 — The family of future operations determines the state that must be kept.**
  `h ≡_{Z,Q} h' ⟺ F(hz, q) = F(h'z, q) ∀ z ∈ Z, q ∈ Q` (`SCOPE_AND_FOUNDATIONS.md:47-55`). With `Z = {ε}`
  only the present behaviour is tested; with all permitted continuations the relation says which past
  distinctions must survive. Not an algorithm (it would need the future as an oracle): it states *which
  counterexample falsifies a proposed compression*. Witnesses [EXACT]: histories `(0,2)` and `(1)` have the same
  mean, new value 4 gives 2 vs 2.5 ⇒ the mean alone cannot be continued, `(sum, count)` can; but
  `(A=0,B=2)` vs `(A=1,B=1)` share `(2,2)` and "remove record A" gives 2 vs 1 ⇒ appending and deleting need
  different information (`:45,65-72`).
- **R3 — Exact representation transport (Proposition 2).** `φ1 = T∘φ0` exists **iff**
  `φ0(x) = φ0(x') ⇒ φ1(x) = φ1(x')` (`SCOPE_AND_FOUNDATIONS.md:80-90`). Three different events follow: the
  needed information was erased; it exists but the reader is incompatible; it is reachable but not within the
  compute budget.
- **R4 — Learning state transforms covariantly.** Under `z = Ax`: `v = A⁻ᵀ w` keeps the function,
  `P_z = A⁻ᵀ P A⁻¹` keeps the *future trajectory* (`v⁺ = A⁻ᵀ w⁺` by induction); weights-only transport changes
  the effective update to `w⁺ = w − η AᵀA (wᵀx − y)x` (`STATE_TRANSPORT.md:13-30`).
- **R5 — A complete summary for one model can be empty for the next feature.** The `±x1x2` witness above;
  the raw four records or the single extra moment `Σ x1x2·y` repair it; "keep all raw data forever" does not
  follow (`STATE_TRANSPORT.md:79-85`).
- **R6 — Attribution, accumulation and validity are three separate conditions** ("ilişkilendirme, bilgi
  biriktirme ve geçerlilik", `CREDIT_ASSIGNMENT.md:5`). Mechanism of the forgetting [EXACT in the noiseless
  limit]: with `v = u` every gradient changes `w1` and `w2` equally, so `w1 − w2 = 1.2` is conserved while the
  new data fix only `w1 + w2 = −.5` ⇒ `(.35, −.85)`; old-population error `.85²/3 ≈ .240833`, composition
  `≈ .481667` (measured .241632, .482351) (`:86`). And the two-target bound: one prediction for two targets
  `Δ` apart has mean squared error ≥ `Δ²/4` (.48 here) — keeping an invalid past has a floor (`:90`).
- **R7 — Measuring future learnability.** `A_{P,k,B,C}(s) = E_{f,E_k}[ R_{Q_f,B}( U^k_C(s, E_k); f ) ]`: risk
  after `k` matched new experiences under answer budget `B` and update budget `C`; initial risks can be equal
  while `A` differs (paper 04 `:98-104`). Proposed as an experimental contract, not an objective.
- **R8 — Three reporting scales.** Immediate ability / continuation ability / change in learning
  (`SCOPE_AND_FOUNDATIONS.md:131-137`).
- **R9 — The update procedure can be part of the state.** `s = (θ, η)`, `θ' = U_η(θ, z)` executed by a fixed
  interpreter; the open question is what evidence justifies changing `η` (`SCOPE_AND_FOUNDATIONS.md:115`).

### (6) Limits
**Admitted.** (a) task ids, the product grammar, exact numeric labels and the validation rule are supplied;
heads are separate so retention is easy; the number of heads grows with tasks; no noise robustness. (b) exact
episode ids are supplied, the representation and rule are known. (c) the transformation `A` is supplied; the
learner found nothing. The successes of separate rigs must not be added into one success rate (paper 04
`:72,80`).

**Limits I see [ANALYST].** All three rigs are deterministic or single-configuration; (a) uses a
single-contradiction retraction — an ungraded belief that one false label destroys; the "equal in every seed"
result means growth was, operationally, lazy allocation over a pre-determined candidate order; (c) proves a
transport formula for one linear rule and one supplied `A`.

### (7) Implications for the core
- **REQ — declare `(Z, Q)` for every stored summary (R2).** For each kind of state the core keeps, name the
  future operations it must support (answer, append, delete, re-represent, continue learning, transfer) and
  test the summary against counterexample pairs that it merges but some operation separates. This is the
  programme's most general and best-supported law.
- **REQ — when a representation is revised, transform the learning state with it, not just the readout (R4),
  and know which of the three failure events you are in (R3).**
- **REQ — attribution, accumulation and validity each need their own mechanism (R6).** Solving one does not
  give the others; replay is not the only accumulation mechanism.
- **WARN — "form the concept when needed" by extending only already-accepted concepts cannot reach a concept
  whose prerequisite is not individually useful (0/2,000).** The construction procedure needs a way to hold
  unjustified intermediate candidates, and that has a search cost nobody has paid yet in this programme.
- **WARN — growth ≡ lazy allocation when the grammar is fixed** (identical in 20/20 seeds).
- **MECH to reuse.** (i) parent-linked executed feature nodes with causal deletion tests
  (`representation_growth.py:121-149,260-266`); (ii) discover-then-validate-on-fresh-evidence before promotion
  (16 + 4); (iii) pending-trace table for delayed feedback with explicit capacity accounting;
  (iv) `(w, G⁻¹)` as retention without raw replay; (v) the transport formulas; (vi) `A_{P,k,B,C}`.
- **OPEN.** Which distinctions to keep when the future operation family is unknown and memory is finite — the
  study names this as the open problem behind every positive result (`STATE_TRANSPORT.md:85`).

---

## 5. Paper 05 (CEV-2026-05) — edition of the joint study

- **(1)–(7):** identical to §3; the body is the joint `REPORT_TR.md` verbatim.
- **Header additions.** Özet (`05-…:11`), Abstract (`:15`), "Bulguların yorumu" (`:19-21`): components that work
  separately do not guarantee reliable learning in all conditions of one life — the discriminating power of the
  data stream decides; good prediction with supplied lags is not general discovery of temporal or causal
  structure; the burst-free condition is kept separate.
- **Error in the header [ANALYST, verified against the data].** The Özet says the studied (growing) method
  matches the accuracy of the fixed expanded representation "daha az yeniden çözüm yapar" — *with fewer
  re-solves* (`:11`); the Abstract says "fewer solves do not establish total efficiency" (`:15`). The paper's
  own body (`:66`), `JOINT_STREAM.md:63` and `results/joint_stream.json › mean_costs.linear_solves` say the
  opposite: growing **320.17**, fixed expanded **112**. The body and data are right; the abstract reverses the
  direction. Any design note that cites the abstract ("growth needs fewer solves") would import a false claim.
- **Label for the paper as a whole:** supported existence witnesses plus retained negatives (§3(4)); "Genel
  yaşarken öğrenme ilkesi bulunduğu ilan edilmiyor" (`:131`).

## 6. Paper 06 (CEV-2026-06) — edition of the update study

- **(1)–(7):** identical to §2; the body is the update `REPORT_TR.md` verbatim.
- **Header additions.** Özet (`06-…:11`), Abstract (`:15`), "Bulguların yorumu" (`:19-21`): experience can change
  how learning will happen; that capacity cannot be read from the fixed initial answer and carries its own harm
  under distribution change; small linear geometry and an externally supplied meta-update family; no new
  principle because of the IDBD match. I found the header consistent with the body and the JSON.
- **Label:** one supported positive (in-family), one strong negative (out-of-family), one scope-narrowing null
  (unknown start).

---

## 7. Cross-study synthesis for the new core

### 7.1 One table: which future operation needs which state
The programme's unifying law (R2 / S5 / U6) made concrete. Every row is a witness executed in this group.

| Future operation | Shown **insufficient** | Shown **sufficient** | Source |
|---|---|---|---|
| Answer now (running mean) | — | the mean | `SCOPE_AND_FOUNDATIONS.md:45` |
| Append an observation | the mean | (sum, count) | same |
| Remove an identified record | (sum, count) | id → value record | `SCOPE_AND_FOUNDATIONS.md:65-72` |
| Answer with a linear predictor | — | `w` | `verify_joint.py:98-103` |
| Keep learning identically | `w` (mean maximal future prediction difference .067 stationary / 2.74 changing) | `(w, P)` + last inputs | `TEMPORAL_RELATION.md:76` |
| Keep learning after recoding `z = Ax` | `v = A⁻ᵀw` (final MSE .027625 vs .000351) | `(v, A⁻ᵀPA⁻¹)` (2.22×10⁻¹⁵) | `STATE_TRANSPORT.md:46-52` |
| Add a product feature later | the complete old least-squares summary | raw records or the one missing cross moment | `STATE_TRANSPORT.md:60-85` |
| Add a lag feature later | unordered `(x,y)` pairs | ordered record or the lag statistic `C1` | `FAILURE_IDENTIFIABILITY.md:53-68` |
| Continue an episode on a learned procedure | procedure without the running state | procedure + one integer (385 symbols) | state `EXPERIMENT.md:104-109` |
| Start new episodes | — | procedure alone | `evidence_audit.py:99-115` |
| Revise the procedure when new labels arrive | procedure alone | procedure + archive (256 words, 2,094 symbols) | state `EXPERIMENT.md:109` |
| Revise the procedure mid-episode | numeric state id of the old procedure | prefix replay or a verified state map (not built) | `recurrent_state.py:282-291` |
| Transfer "how to learn" to a new life | — | `P` (3 numbers), valid in-family only | `learned_update.py:171` |
| Resume meta-learning mid-block | `P` | `P`, `w`, sensitivities, gradient, block position | update `REPORT_TR.md:86` |
| Use delayed feedback | the present input (.101597 vs .001233) | id → input trace, capacity ≥ the delay spread | `CREDIT_ASSIGNMENT.md:58-78` |
| Keep valid old skill in shared weights | correct attribution alone (.000905 → .216505) | 32-record replay (.000957) or `(w, G⁻¹)` (.000070) | `CREDIT_ASSIGNMENT.md:58-70` |
| Follow a changed target | all-history statistics (.509150; .078358) | fast replacement / region-wise refresh | `CREDIT_ASSIGNMENT.md:90`, `JOINT_STREAM.md:51-59` |
| Know whether an unvisited region is still valid | **any** state built from the observed prefix | — (new evidence or an explicit assumption) | `JOINT_STREAM.md:75` |
| Decide "false label vs changed world vs model too small" | residuals, consistency, emptied family | — (trusted reference or explicit assumptions) | `FAILURE_IDENTIFIABILITY.md:19-51` |

Reading for the design: there is no single "the state". There are at least four tiers — **run**,
**keep learning**, **repair / re-represent**, **transfer how to learn** — each demanding more than "answer" and
each shown to be separable by an intervention; in these witnesses the first three are nested (what repairs also
continues learning, what continues learning also runs), the fourth is a different axis.

### 7.2 What the group says about each design pillar
- **Learning and running as one process.** Existence is established in every study (predict → observe →
  update, no training phase). What the studies add: (a) score the whole life, not the end — methods with the
  same end state lived differently (.050781 vs .009146 with equal exact end graphs; .012425 vs .012216 with
  equal end risks); (b) "one process" was in fact periodic batch re-inference in two of the four rigs (every 16
  samples; at 5 checkpoints).
- **Weights as graded beliefs.** `(w, P)` already is one, and three facts about it are established: `P` is part
  of the learning state (the `w`-only restore in §3, the weights-only transport in §4), it must be transformed
  with the representation (R4), and it measures which query directions experience has covered (J2; the readout
  itself is my check, §8-A). What is *not* graded anywhere in this group — and
  fails for that reason — is structure: DFA merges, feature activation and head retraction are all-or-nothing
  (36–42 states from noise; a head destroyed by one label; a burst accepted as change).
- **Concepts formed when needed.** Works as a mechanism in three rigs; was never better than having the
  candidates from the start (identical in 20/20 seeds; +.000209 worse and 2.9× the solves); has a proven
  curriculum hole (0/2,000); and its trigger cannot tell real need from false evidence. The one place where
  growth was *necessary* is where the fixed family was too small (4-state rule) — so the value of on-demand
  formation is tied to open-ended families, which this group never tested beyond one extra state.
- **Imagining and measuring how real a construction is.** The studies contain the ingredients only on the
  evaluator's side: exact equivalence against the truth, alternative worlds that fit all evidence
  (`L △ {w*}`, hidden global change, `y=u_t` vs `y=−u_{t−1}`, `±x1x2`), and a model that predicts the learner's
  own risk (second-moment recursion). **[ANALYST synthesis, suggested not proved]:** all four negatives have the
  same form — *a construction is only as real as the alternatives consistent with the same evidence agree on
  the queries of interest* — and the two honest internal estimators of that agreement already exist in the
  code: unanimity of version-space survivors (discrete) and `φᵀPφ` (linear). Their proven limits: they are
  relative to the candidate family (S3, 4-state control), they are corrupted by false evidence (U4, J5), and
  they say nothing about staleness unless drifted worlds are among the alternatives (J7).
- **The way of constructing improves and transfers.** Established in-family and by a causal test; established
  to invert out-of-family; continuing to adapt does not fully repair; the simplest transferred choice was the
  most robust. No study in this group lets the *construction procedure for concepts* (the generator, the merge
  order, the grid) change with experience — only a 3-number update geometry.

### 7.3 Requirements extracted (each traceable to a result above)
1. Declare, per stored summary, the future operation family it must support; test it with merged-but-separable
   counterexample pairs (R1, R2).
2. Persist and audit four tiers of state by *trajectory* equality in a fresh process, not by answer equality
   (S8, U1, R4; the w-only tests).
3. Grade structure, not only weights: support per merge / feature / head; no all-or-nothing acceptance or
   retraction (state study corruption, growth single-label failure, joint burst).
4. Report three independent components for "how real": fit on lived evidence; coverage of the query by evidence
   (identifiability); validity (age and assumed drift). The studies show each can be perfect while another is
   empty (.00000318 / .415140; .000478 / .355531).
5. Make abstention a first-class answer (version-space unanimity; scored with coverage).
6. Evaluate "better learning" only with matched new evidence, identical initial answers, and a reset control
   (U1, R7).
7. Treat every transferred habit and every retention policy as a bet with a scope; keep a hedged fallback
   (U2, J7).
8. Accept changes against the current joint state with a cumulative budget (J3, J4); never sum module-wise
   successes.
9. Count acquisition and search costs (16,384-sample source life; 320 vs 112 solves; ×52 copying under noise).
10. Keep negatives and post-result diagnostics labelled as such (the programme's append-only discipline with
    sha256 manifests is itself a reusable mechanism for "auditable").

### 7.4 Open questions the group leaves (named by the studies)
What to keep when the future operation family is unknown and memory is bounded; learning the retention
geometry, time weights and history vocabulary; telling false evidence from real change from a too-small model;
migrating a running state across a revised procedure; acquiring the symbols and the meaning of feedback;
attribution without identifiers or alignment; interventional knowledge and action-generated experience (every
stream in this group is passive); a transferable way of learning that knows when it does not apply; all of it
together in one long life under resource pressure.

---

## 8. Analyst tiny checks (not part of the owner's studies)

Script: `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/prior/_checks_g2/g2_tiny_checks.py`
Output: `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/prior/_checks_g2/g2_tiny_checks.json`
Standard library, ~10 s, repo modules loaded by path with `python3 -B` (nothing written into the repository).
Check A first reproduces the study's own numbers (probe MSE .0000134378 and .4151399, stream MSE .0000133722 and
.0000031831 — equal to the committed results), so the re-implemented loop is faithful. Check B reuses the
study's stream generator and seeds and reads the study's reference numbers from its JSON; the belief-filter
rows are new numbers of mine.

**Check A — does the RLS learner's kept state know that its construction is not backed for new histories?**
`temporal_relation.RLS`, all 24 seeds, quantity `φᵀPφ / φᵀφ` (with `P0 = I` this is the fraction of prior
uncertainty left along the query; 1 = nothing learned, 0 = fully backed).

| Regime | on the lived stream (last 200) | on 512 new independent histories | study's MSE stream / new |
|---|---:|---:|---:|
| iid, stationary | .00132 | .00132 | .0000134 / .0000134 |
| alternating input | .00014 | **.585** (min over seeds .574) | .0000032 / **.415140** |

So the failure the study reports as its "strongest negative" was visible in a state the learner already kept:
a 4,200-fold jump in remaining uncertainty for the new query population, against no jump in the regime that
generalised. Limits of this readout: within one query population it ranks individual queries only weakly
(Spearman .24 and .16 between the quantity and the squared error); the absolute scale needs a prior scale
(using the true mean square of the lag weights, .243, the predicted new-history MSE is .486 against the
measured .415 — right order, but that scale is privileged); and it presupposes that the truth lies in the
linear family (it is silent in the outside-vocabulary regime).

**Check B — the update study's streams with a belief state instead of a fixed gain.** Same 24 sources, same 384
new lives, same seeds. Learner = mean + 2×2 covariance, Kalman recursion (covariance grows by `Q` before every
observation, shrinks with every label). *Privileged reference:* the transferred object is the exact drift model
of the source world (`Q`, `r`), nothing is estimated from experience. Service MSE, mean over 24 sources:

| Variant | Same structure | Reversed | Fast everywhere | Unknown start |
|---|---:|---:|---:|---:|
| belief, source `Q`, generic wide prior (`C0 = I`) | **.005041** | **.086707** | **.077172** | **.005596** |
| belief, direction-free `Q` (only total drift), wide prior | .006806 | .006800 | .010559 | .007357 |
| belief, oracle `Q` of the new life | .004904 | .004940 | .009998 | .005598 |
| study: learned_frozen | .005199 | .034495 | .036181 | .013223 |
| study: fresh_meta | .008774 | .008623 | .016619 | .013219 |
| study: train_selected_scalar | .006868 | .006868 | .010456 | .008538 |

Intervals for the first row: same [.004952, .005130]; reversed [.075830, .097584]; unknown start
[.005457, .005735]. First-64-step MSE at the unknown start: belief .020794, learned_frozen .164766, scalar
.047019.

Reading: (i) the study's "unknown start" null is a property of transferring a *rate*; with current uncertainty
represented, the same transferred knowledge wins that regime (.005596 vs .008538 for the best study method).
(ii) The study's main negative is *amplified* by a sharper belief: an exactly optimal filter for the old world
is 2.5× worse than the study's learned matrix in the reversed world and 10× worse than no transfer. (iii) A
belief that commits to the amount of change but not to its direction is as robust as the study's most robust
method in the three tracking regimes and better at the unknown start. Hence: "weights as graded beliefs" pays
only if the belief about *how the world changes* is itself graded and revisable — which neither the study nor
this check implements. Estimating `Q`, `r` from experience at acceptable cost is untested.

**Observation C — from the state study's own JSON (no new computation).** Learned state count vs labels seen:

| Regime | 16 | 32 | 64 | 128 | 256 labels |
|---|---:|---:|---:|---:|---:|
| 3-state family (16 lives), mean | 5.00 | 3.69 | 2.94 | 2.94 | 2.94 |
| 4-state rule (4 lives), mean | 4.25 | 4.00 | 4.00 | 4.00 | 4.00 |
| corrupted (8 lives), mean [min–max] | 6.13 | 10.13 | 15.25 [12–19] | 25.0 [23–27] | 39.4 [36–42] |

The bounded family empties at the same time in the clean 4-state world (labels 8–11) and the corrupted world
(8–13) — the study's point that this signal does not diagnose. The *growth curve* does separate them from 64
labels on. It is a rate signal with the caveat given in §1(6).

---

## 9. Established / Refuted / Open — across the group

| # | Status | Statement | Evidence | Source |
|---|---|---|---|---|
| E1 | Established | A state may identify two histories only if every permitted continuation treats them alike; present agreement is not enough; then a transition on classes is well defined | [EXACT] congruence argument; runs 16/16, 4/4, 4/4 | state `EXPERIMENT.md:88-94` |
| E2 | Established | A summary can be updated online iff it is a congruence for the permitted new experiences; a representation can be derived from another iff it merges no pair the other separates | [EXACT] Propositions 1–2 | `SCOPE_AND_FOUNDATIONS.md:29-43,80-90` |
| E3 | Established | The family of future operations determines the state that must be kept: answering, appending, deleting and re-representing each need more than the previous one in the witnesses; running < keeping on learning < repairing | [EXACT] mean/sum-count/delete; `±x1x2`; lag statistic; two-bit memory; 385-symbol resume vs archive | §7.1 |
| E4 | Established | Equal answers today do not imply equal learning tomorrow; the learning state is a separate, transferable, erasable part of the state | [EXACT] `∂w⁺/∂y = P x`; [EMP n=24] −.003575; [EMP] w-only restore .067 / 2.74; transport .027625 vs .000351 | update `REPORT_TR.md:42-49`; `TEMPORAL_RELATION.md:76`; `STATE_TRANSPORT.md:46-52` |
| E5 | Established | Learning state transforms covariantly with the representation: `v = A⁻ᵀw`, `P_z = A⁻ᵀPA⁻¹` | [EXACT] 2.22×10⁻¹⁵ over all trajectories | `STATE_TRANSPORT.md:13-30` |
| E6 | Established | A procedure acquired from terminal labels of short episodes can run on far longer inputs from a compact state, with no state count given, and can exceed a fixed family's bound | [EXACT per target] 16/16 (non-empty domain), 4/4, 4/4 | state `EXPERIMENT.md:60-64` |
| E7 | Established | No fixed window replaces a recurrent state | [EXACT] parity alias for every k | state `EXPERIMENT.md:84` |
| E8 | Established | A relation between an outcome and several past inputs is learnable without per-outcome identifiers, from `O(L²)` statistics, when the stream excites the candidates | [EXACT] identity + bound; [EMP n=24] .00001344 | `PRINCIPLE_AUDIT.md:17-38` |
| E9 | Established | Acquisition, retention and correction can share one state without task ids — in a world whose change geometry matches the supplied memory geometry | [EMP n=24] .000478 / .000417 | `JOINT_STREAM.md:51-59` |
| E10 | Established | A way of learning acquired in one life changes learning in another with identical initial answers — when the change geometry is shared | [EMP n=24] −.003575 [−.003693, −.003456]; ablations | update `EXPERIMENT.md:90-92` |
| E11 | Established | Retaining valid old skill needs accumulated information, not necessarily raw replay | [EMP n=32] RLS .000070 vs replay .000957 vs none .216505 | `CREDIT_ASSIGNMENT.md:58-70` |
| E12 | Established | Learning without an explicit scalar optimiser is possible and monotone under its assumptions (candidate elimination with unanimous answering) | [EXACT] Proposition 3 | `PRINCIPLE_AUDIT.md:105-117` |
| E13 | Established | Risk increases compose additively under a per-step tolerance and failure probability | [EXACT] Proposition 2 | `PRINCIPLE_AUDIT.md:83-97` |
| E14 | Established | The finite-time risk of a fixed update rule in an assumed drifting world is computable in closed recursion and matches the lived risk | [EXACT model], agreement .00519828 vs .00519868 etc. | update `EXPERIMENT.md:100-130` |
| X1 | Refuted | Consistency with experience plus structure growth shows a valid new distinction | [EMP n=8] 0/8, 36–42 states, .287109 | state `REPORT_TR.md:32-38` |
| X2 | Refuted | An emptied candidate family (or a systematic residual) diagnoses the cause | [EXACT] same symptom, different worlds; aliasing witnesses | state `REVIEW.md:31`; `FAILURE_IDENTIFIABILITY.md:19-51` |
| X3 | Refuted (as a mandatory principle) | The cause of an error must be identified before learning | [EXACT] not needed when the goal is the observed outcome; impossible in the aliased cases | `FAILURE_IDENTIFIABILITY.md:30,116` |
| X4 | Refuted | Success on the lived stream certifies the learned relation for unlived continuations | [EMP n=24] .00000318 vs .415140; [EXACT] `L △ {w*}`, null-space alias | `TEMPORAL_RELATION.md:60`; `STATE_DISCOVERY_LITERATURE.md:73-77` |
| X5 | Refuted | Preserving current predictions preserves later learning | [EXACT + EMP n=32] | paper 04 `:110` |
| X6 | Refuted | A complete summary for the current model suffices for any later representation | [EXACT] `±x1x2`; lag statistic | paper 04 `:111`; `FAILURE_IDENTIFIABILITY.md:53-68` |
| X7 | Refuted | Correct attribution of feedback suffices to keep valid old skill | [EMP n=32] .000905 → .216505 | paper 04 `:112` |
| X8 | Refuted | Raw replay is the necessary retention mechanism | [EMP n=32] | paper 04 `:113` |
| X9 | Refuted (general form) | Growing structure on demand is inherently better than an equally expressive fixed structure | identical in 20/20 seeds; +.000209 worse and 320 vs 112 solves [EMP n=24]; curriculum 0/2,000 | paper 04 `:114`; `JOINT_STREAM.md:61-63` |
| X10 | Refuted | A learned way of learning helps in general; continued adaptation removes its harm | [EMP n=24] +.025872, +.019562; still +.003373 after continued adaptation | update `EXPERIMENT.md:94` |
| X11 | Refuted | Retention (or forgetting) is good in itself | [EMP] rolling .204387, all-history .078358; replay .181190 and RLS .509150 on a changed target; forgetting +.00004066 when stationary | `JOINT_STREAM.md:51-59`; `CREDIT_ASSIGNMENT.md:90`; `TEMPORAL_RELATION.md:56` |
| X12 | Refuted | Region-wise retention settles which past is still valid | [EXACT] indistinguishable worlds, bound .088415, learner .355531 | `JOINT_STREAM.md:75` |
| X13 | Refuted | Updates that are each good are good together | [EXACT] 1 → .25, 1 → .25, both → 4 | `PRINCIPLE_AUDIT.md:73-81` |
| X14 | Refuted | Operating under noise means recognising false information; a bounded update is a safe update | [EMP n=24] .000464 → .132789; [EXACT] sample error 1→0, true error 0→1; one label destroys a head | `JOINT_STREAM.md:69`; `UPDATE_THEORY.md:143`; `REPRESENTATION_GROWTH.md:79` |
| X15 | Refuted | Good temporal prediction is causal knowledge; randomness is required for identification; a small running state means a small learning memory; a state id survives a change of procedure | [EXACT] each | `PRINCIPLE_AUDIT.md:52-59`; `TEMPORAL_RELATION.md:82`; state `REPORT_TR.md:65`; `recurrent_state.py:282-291` |
| N1 | Not shown / inconclusive | A learned update improves acquisition from an unknown start | [EMP n=24] +.00000377 [−.001554, +.001561] | update `EXPERIMENT.md:96` |
| N2 | Not shown | The system finds which evidence is false; online meta-learning beats selecting a fixed rule from the same past; any total-life efficiency gain | corrections supplied by the evaluator; −.000134 vs a coarse grid; costs reported, not amortised | state `REVIEW.md:33`; update `EXPERIMENT.md:90`, `REPORT_TR.md:38` |
| O1 | Open | Which distinctions to keep when the future operation family is unknown and memory is bounded | named by every study | `STATE_TRANSPORT.md:85`; state `REPORT_TR.md:75` |
| O2 | Open | Learning the retention geometry, time weights, forgetting and history vocabulary from experience | named as the most prominent gap | joint `REPORT_TR.md:81` |
| O3 | Open | Separating false evidence, real change and a too-small model with accessible evidence | named open; the three produce the same symptoms in this group's rigs | state `REPORT_TR.md:38`; update `REPORT_TR.md:75,80` |
| O4 | Open | Revising a procedure or representation while it is running (state migration); incremental rather than batch re-inference | named open; only the counterexample exists | state `EXPERIMENT.md:123` |
| O5 | Open | A transferable way of learning/constructing that represents its own scope and detects mismatch | named open; "güncelleme hızını … merkezi yapmamak gerekir" | update `REPORT_TR.md:70-80` |
| O6 | Open | Forming concepts whose prerequisites are not individually useful, at a paid search cost; retiring concepts | named open; curriculum counterexample 0/2,000; no rig removes a concept | `REPRESENTATION_GROWTH.md:83,101` |
| O7 | Open | Acquiring the symbols, the meaning and reliability of feedback, attribution without identifiers, interventional knowledge, action-generated experience | named open; every stream in this group is passive and pre-symbolised | state `REPORT_TR.md:73`; `PRINCIPLE_AUDIT.md:61-67` |
| O8 | Open | All functions in one long life under shared, bounded resources | named open; separate rigs must not be summed | paper 04 `:129`; joint `REPORT_TR.md:70-81` |
| A1 | Analyst (not the studies') | The kept RLS state exposes the unbacked query population (.00014 vs .585); a belief state removes the unknown-start null (.005596) and amplifies the wrong-transfer harm (.0867); learned structure size separates noise-driven growth from saturation | §8 | `_checks_g2/g2_tiny_checks.json` |
