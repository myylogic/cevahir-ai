# AWA-3 "the order law in attention" — PROTOCOL (DRAFT, NOT FROZEN)

Status: DRAFT, NOT FROZEN. Written on 2026-11-13 (UTC) from the research program's flagship section
(scratch `PROGRAM.md` §3, including its "Critic notes" and "Response to critic") and the prototype
`attention-math/` (proposal_v2.md, awa_v2.py, test_awa_v2.py, probe_v2.py, order_law_sim.py). No Phase-1 run
exists. A first set of Phase-0 pilots (2026-11-13) was run with the original single-query training signal and no
arm left the chance floor. The diagnosis and the change of the training signal that followed are recorded in §15
("Pre-freeze design changes", 2026-11-13/14). Those pilots are void.

State of learnability on 2026-11-14, from the diagnostic runs of §15 (one seed each, not experiments of this
protocol): with the signal of §5.1 a single lookup (k = 1) is learned by the arms that were tried (STD-2L-pm,
AWA-2L); NO arm has learned a second hop (k = 2, 3) in any run, including AWA-2L after 3,000 steps. The study
cannot be frozen in this state: the multi-hop learnability problem of §15.4 has to be solved, or H1 has to be
reported as failed for a stated reason, before Phase 0 is run again. Apart from the diagnostic runs of §15, only
a pipeline smoke with tiny step counts has been run (§11.4); it tests nothing.

Freeze procedure (§14): after Phase 0 ends, close the open decisions of §12, replace this status paragraph,
remove the words "NOT FROZEN" from this file, and run `run_verify.py --freeze`. Until then
`run_verify.py --freeze` refuses to write a freeze manifest, and the Phase-1 job list cannot run: its first job
(`p1_freeze_check`) fails, every other job depends on it, and every training run (`run_train.py --phase 1`)
checks the freeze again by itself. The other Phase-1 drivers (`run_extend.py`, `run_eval.py`, `run_analyze.py`,
`run_verify.py --manifest / --rerun`) do not check the freeze themselves; started by hand, outside the job
list, they are not blocked. After the freeze, deviations go only into AMENDMENTS.md.

Naming. The program text calls the gate G0, the two primary tests P-M and P-T, and the secondary tests H3b, H4,
H5, H6. This protocol numbers them H1–H6 and keeps the program's labels in brackets: H1 [G0], H2 [P-M],
H3a [P-T], H3b, H4, H5, H6.

---

## 1. Question, claims and honest novelty

Question. Can one causal attention layer follow a reference chain in any order, and why can the published
single-layer multi-hop operators not?

Known, not claimed:
- single-layer causal multi-hop attention through a resolvent of the attention matrix: ChaCAL (Fagnou et al.,
  EMNLP 2024); Block-ChaCAL (2026);
- power-series and absorbing-chain readings of attention: Diffuser, MAGNA, InfSA, TokenRank;
- node-dependent absorption (PARW, 2012); null or sink options;
- weight-tied hops with a test-time budget: MemN2N, Universal and looped Transformers;
- function-preserving insertion at a zero gate; depth lower bounds (Sanford et al. 2024); the premise-order
  effect in large language models (Chen et al. 2024).

Claimed (the program's novelty check found no match in 34 searches; no full text was read):
- (a) The order law. After one binding layer and under assumption B (§4), a single causal multi-hop layer whose
  hops are normalised over each position's own window resolves a chain only if its links are dependency-sorted.
  Under uniform shuffling of m freely ordered links the chain is followable with probability exactly 1/m!;
  accuracy follows a mixture law that includes the guessing floor; one adjacent swap breaks the chain. All of it
  is tested on trained models. The monotone-path lemma behind it is elementary and is not claimed.
- (b) Prefix-closed (query-window) transition normalisation, which lets one causal layer walk in any order. Its
  nearest relative is ENTP's per-prefix recomputation.
- (c) Minor: content-gated absorption with a null option as the head's readout.

Not claimed:
- that the law explains premise-order effects in large models (deep stacks escape it, §4);
- that AWA beats recurrence (H6 predicts no direction).

The comparator is labelled "ChaCAL-style" until its paper has been read: whether the published operator learns
gamma, and per head, is unverified.

## 2. Mechanism

One head of one layer; `x` is the RMSNorm'd layer input. The operator is implemented in the engine
(`src/neural_network_module/ortak_katman_module/attention_manager_module/absorbing_walk.py`) as an optional
attention operator of `MultiHeadAttention`, off by default, selected through the model configuration
(`attention_operator`, `attention_operator_options`).

```
q_i = RoPE_i(W_Q x_i),  k_j = RoPE_j(W_K x_j),  v_j = W_V x_j
a_ij = softmax_{j in W_i}(q_i . k_j / sqrt(d))                         hop 1 = the engine's own attention row
kbar_j = W_K x_j  (pre-RoPE, after QK-norm when it is on),  t_j = W_T x_j     address and pointer
E_jl = exp(kappa tanh(t_j . kbar_l / (kappa sqrt(d)))) [l != j],  kappa = 20
c_j  = lambda_h sigmoid(u_h . x_j + b_h),  lambda_h = lambda_max sigmoid(rho_h),  lambda_max = 0.99
P(i)_jl = E_jl / (e^tau + sum_{l' in W_i} E_jl'),  P(i)_j,null = e^tau / (same)    normalised over the QUERY's window
pi(0) = a_i;  pi(m+1)_l = sum_{j in W_i} pi(m)_j c_j P(i)_jl;  sigma(m)_j = pi(m)_j (1 - c_j + c_j P(i)_j,null)
A~(K)_i = sum_{m<K} sigma(m) + pi(K)                                   rows stay causal probability vectors
A~(inf)_i = a_i (I - C P(i))^{-1} S(i),  C = diag(c),  S(i) = diag(1 - c + c p(i)_null)
o_i = sum_j A~_ij v_j  ->  concat heads  ->  out_proj                   (unchanged)
```

- W_i = {j <= i} in this experiment. In general W_i is the set of keys the attention mask allows for query i
  (causal mask, sliding window, padding).
- K walk steps are weight-tied; K can be changed at inference (`set_attention_hops`, or `num_hops` of
  `forward_rows`). `set_attention_hops` stores K in the model's configuration, so a saved model is restored with
  it; the checkpoint identity check ignores K.
- The operators compute in float32 (or float64) and switch autocast off for their own arithmetic: E reaches
  e^20 = 4.9e8, above the fp16 range. The study itself runs in fp32 on CPU.
- Parameters added per AWA layer: D² + H(D + 1) + 2H = 16,908 at D = 128, H = 4.
- From scratch lambda = lambda_max·sigmoid(rho) is initialised at 0.5 and is never exactly 0.

Comparators (same file):
- AWA-own (single-ingredient ablation): P_jl = E_jl / (e^tau + sum_{l' in W_j} E_jl'), i.e. every hop is
  normalised over the hopping position's own window. Only the normalisation window differs; every factor is
  lower-triangular.
- ChaCAL-style: A~ = (1 − gamma_h) A (I − gamma_h A)^{-1}, gamma_h = 0.99·sigmoid(rho_h), initialised at 0.5,
  exact by one triangular solve; H parameters.
- LOOPED-2L: a binding layer, then a weight-tied block applied T times to all positions.
- ARLOOP-2L: the final layer re-applied T times at the answer row only; no new parameters.
- STD-2L-pm: parameter-matched standard model (ffn_dim 534).

Exact reductions (each is a test in `tests/evolution/test_awa_attention.py`):

| Setting | Result | Test tolerance |
|---|---|---|
| operator off (default) | the engine before AWA, bit for bit | state_dict, logits, KV-cache decode, gradients and the random-stream position: against the engine exported from git at 0f9be09 in the same environment, and against digests recorded there |
| lambda = 0 (rho = −inf) | A~ = A | bit-exact against the explicit-softmax path; ≤ 1e-6 against the fused path |
| K = 0; tau → +inf | A~ = A | bit-exact; ≤ 1e-6 |
| K → inf | a_i (I − C P(i))^{-1} S(i) | 1e-12 (float64) |
| own window, self-hops, pointer logits = attention logits, no cap, no null, c ≡ gamma, K → inf | ChaCAL-style resolvent | 1e-10 (float64) |
| ChaCAL-style, gamma = 0 | A | 1e-15 (float64); ≤ 1e-6 through the model |
| rows-only final layer | full forward at those rows | 2e-6 (fp32), every operator, GQA / QK-norm / sliding window / padding |
| KV-cache decode (AWA) | full forward | 1e-5 over 32 decode steps, with and without a sliding window |
| LOOPED, ARLOOP at T = 1 | STD | 1e-6 (`tests/evolution/test_awa_study.py`) |

AWA-own and the ChaCAL-style operator have no KV-cache path in the engine (they raise a clear error); the study
never decodes with a cache.

## 3. Hypotheses, SESOIs and decision rules

Notation. The unit is one training run (seed). h_k is the uniform-over-literals floor, identical in all four
strata (§5). Y (primary endpoint per run) = the minimum over the four strata of the mean accuracy at k ∈ {6, 8},
at the preregistered budget: K_test = 12 for walks, T_test = 12 for loops, ChaCAL-style exact. Chance-corrected
values (acc − h_k)/(1 − h_k) are always reported next to it.

### 3.1 H1 [G0] — learnability gate (a precondition, not a contrast)

- Requirement: at least 4 of 5 AWA seeds are ID-competent: mean shuffled accuracy over k ∈ {1, 2, 3} ≥ 0.90 at
  K = 5 by step 3,000 (4,500 after the one funded extension).
- Negative control for assumption B: STD-2L-pm at k ∈ {2, 3} stays ≤ h_k + 0.10 (seed mean, shuffled stratum).
  B is declared false if k = 2 exceeds h_k + 0.15.
- The control counts only among STD-2L-pm seeds that learned what B allows: k = 1 accuracy ≥ 0.90 at step 3,000.
  With fewer than 2 competent STD seeds (1 if only one STD seed was run), the control is "uninformative" and B is
  reported as untested.
- If H1 fails: H2 is "NOT TESTED" and reported as a learnability result; H3a is still tested.

### 3.2 H2 [P-M] — primary, mechanism: query-window normalisation buys order-free depth generalisation

Intersection–union of two contrasts, each at one-sided alpha = 0.05 (no multiplicity adjustment for an IUT):
(a) AWA vs ChaCAL-style (5 vs 5); (b) AWA vs AWA-own (5 vs 5; identical except the normalisation window).

For each contrast, all three must hold:
1. at least 4 of 5 AWA seeds have Y ≥ 0.80;
2. exact permutation statistic T4 = #{AWA seeds with Y − 0.15 > max comparator Y} ≥ 4 (exact p = 6/252 = 0.024
   under the location-shift null; T4 = 5: p = 1/252); T4 tolerates one failed AWA seed;
3. the Hodges–Lehmann estimate of Y_AWA − Y_comp is ≥ 0.30 (SESOI).

Reported but not decisive: the exact Mann–Whitney test with the 0.15 shift and the exact 95.2% one-sided HL
bound (the 5th smallest of the 25 pairwise differences).

- CONFIRMED if both contrasts pass all three criteria.
- REFUTED if at most 1 of 5 AWA seeds reaches Y ≥ 0.80 and the AWA median Y is < 0.50, or if HL < 0.15 in
  either contrast.
- INCONCLUSIVE otherwise. The run is then labelled by the per-hop classifier (§3.3); no claim is made, no seeds
  are added, and no follow-up phase starts under this protocol.

### 3.3 Per-hop classifier (labels the INCONCLUSIVE zone; reported for every arm)

Per-hop error law: acc(k) ≈ h_k + (1 − h_k)(1 − eps)^{k−1}. Per seed, eps is fitted by least squares on the
in-distribution depths k ∈ {2, 3} (shuffled, plateau budget: K = 24 for walks, best T for loops), then k ∈ {6, 8}
is predicted.

- precise: Y ≥ 0.80;
- compounding (algorithm learned, imprecise): Y < 0.80 and the observed worst-order plateau accuracy at
  k ∈ {6, 8} is within ±0.07 of the prediction;
- depth cliff (no depth generalisation): observed < prediction − 0.10;
- unclassified: otherwise.

The study label is the majority over AWA seeds. eps fitted on all depths 2–12 is a continuous secondary outcome.

### 3.4 H3a [P-T] — primary, theory: the order law in trained lower-triangular models

Decoupled from H1; ChaCAL-2L × 5 and AWA-own-2L × 5.

- Informative depths: k ∈ {3, 4, 6, 8} with seed-mean sorted accuracy ≥ h_k + 0.30. k = 3 must qualify,
  otherwise the arm is "not testable".
- Predictions (per seed, pooled over the informative depths): one-swap and anti accuracy ≤ h_k + 0.10; shuffled
  accuracy within ±0.05 of the mixture prediction P_f·acc_sorted + (1 − P_f)·acc_anti at k ∈ {3, 4}.
- Supported for an arm if the upper one-sided 95% t-bound over seeds of mean(acc − h_k) is ≤ 0.10 on one-swap
  and on anti, and the upper bound of the mixture error is ≤ 0.05. The law is SUPPORTED if both arms are.
- REFUTED if, for either arm, mean(acc − h_k) on one-swap or anti exceeds 0.15 significantly (one-sided t-test,
  Holm over the two arms). Refutation means B fails in trained models or the implementation is not
  lower-triangular; the hand-set check (§4) separates the two.
- INCONCLUSIVE otherwise.
- Exploratory: accuracy against the number of out-of-order adjacent aliases, for every arm.

### 3.5 H3b — AWA order invariance (secondary)

Per seed, G = the mean over the informative depths k ∈ {4, 6, 8} (AWA's best stratum ≥ h_k + 0.30; k = 4 must
qualify, else "not testable") of (max − min accuracy across the four strata). SUPPORTED if mean G ≤ 0.05 and the
one-sided 95% t upper bound is ≤ 0.08. REFUTED if mean G ≥ 0.15. Item sampling alone gives G ≈ 0.01–0.02.

### 3.6 H4 — mechanism (AWA)

- (a) Dose-response in K. K*(k) is the smallest K with acc(k) ≥ plateau(k) − 0.05, plateau = accuracy at
  K = 24. Prediction: K*(k) − k is the same integer for every k ∈ {2..8} with plateau ≥ 0.5, and
  acc(K* − 1) ≤ acc(K*) − 0.30. Refuted if it varies by more than 1 across depths in at least 2 seeds, or if any
  depth plateaus with K < k − 1 (a shortcut that contradicts B).
- (b) Gate ablation. Setting lambda := 0 at test time brings k ≥ 2 accuracy to ≤ STD-2L-pm + 0.05.
- (c) Hop trace. The median normalised hop mass on chain element m − delta is ≥ 0.8 for every m ≤ k − 1 at
  k ∈ {3, 4, 6}.

### 3.7 H5 — no overthinking (AWA)

For K from K*(k) to min(3k, 24), accuracy stays within 0.02 of its maximum. Refuted by a drop ≥ 0.05.

### 3.8 H6 — is recurrence enough? (key secondary; no direction predicted)

LOOPED × 4 and ARLOOP × 4, each at its oracle T chosen on validation, and at T = 12.

- (i) Capability: two-sided exact Mann–Whitney on Y (5 vs 4, minimum p = 2/126 = 0.016), Holm over the two
  baselines; SESOI 0.15 on the Hodges–Lehmann estimate.
- (ii) Cost at matched accuracy. Y† = 0.9 × min(median-over-seeds plateau Y of AWA, of the baseline B). Per
  seed, b† is the smallest budget whose worst-order accuracy is ≥ Y†; C = analytic MACs per answer at b† ÷
  accuracy at b† (§9). Decision on the HL estimate of log(C_B/C_AWA) with its exact two-sided interval: AWA
  cheaper if ≥ log 2 and the interval excludes 0; B cheaper if ≤ −log 2 and the interval excludes 0; comparable
  otherwise. An arm that never reaches Y† in at least 3 of its seeds "does not reach matched accuracy".
- Reading (per baseline; `analysis.h6_reading`, shown in the summary table): (i) significant either way decides
  the capability claim; otherwise (ii) decides between "AWA matches B at ≥ 2× lower cost", "B matches AWA at
  ≥ 2× lower cost" and "equivalent".

## 4. Theory

- Monotone-path lemma (elementary; used, not claimed). For lower-triangular P(1)…P(m), causal A and diagonal
  D(s), (A D(1)P(1) ⋯ D(m)P(m))_il sums only over position sequences i ≥ p_0 ≥ p_1 ≥ … ≥ p_m = l. Hence
  Σ_m w_m A^{m+1}, (1 − gamma)A(I − gamma A)^{-1} and AWA-own receive contributions only from non-increasing
  paths.
- Binding assumption B. After one binding layer, the state at an assignment encodes its own (LHS, RHS, type) and
  no other assignment's RHS. Layer-1 queries and keys depend on one token plus relative position, so composing
  two assignments needs two content-dependent lookups. B is tested by STD-2L-pm (H1 control).
- Order law. Under B, a 2-layer model whose second layer has the lemma's form can emit the queried literal only
  through a path query → alias_0 → … → literal fact with decreasing positions. With the literal block first
  (this task):
  1. success requires the k − 1 aliases to be dependency-sorted, which has probability exactly 1/(k − 1)! under
     uniform shuffling (1/m! for m freely ordered links);
  2. one adjacent transposition makes the chain unfollowable;
  3. mixture law: acc_shuffled = P_f·acc_sorted + (1 − P_f)·acc_unfollowable, with the followable fraction P_f
     measured on the frozen items (§5).
- Hand-set check (`awa3/handset.py`, tests `test_order_law_*` in `tests/evolution/test_awa_study.py`), with
  the engine operators and idealised binding (one position per fact), K = 8:
  - exact: for every program tested at k = 3, 4, 5, exactly one of the (k − 1)! alias orders is resolved by
    AWA-own and by the ChaCAL-style resolvent, and all of them by AWA;
  - 40 programs per cell, k = 1..8: ChaCAL-style = AWA-own score 1.00 sorted and 0.00 one-swap and anti at
    k ≥ 3, and exactly the dependency-sorted share on shuffled; AWA scores 1.00 in every stratum;
  - outside the test suite, 2,000 programs per depth of the stream `unit-orderlaw`: shuffled accuracy of the
    ChaCAL-style resolvent and of AWA-own 0.4980 / 0.1765 / 0.0460 at k = 3 / 4 / 5 (equal to the
    dependency-sorted share of those programs) against 1/(k − 1)! = 0.5000 / 0.1667 / 0.0417, each within
    1.2 binomial standard errors; AWA 1.0000. About 17 CPU-seconds, from the repository root:
    `python -c "import sys; sys.path[:0] = ['research/attention_order_law_2026_11_13', '.']; from awa3 import handset; t = handset.order_law_table(depths=(3, 4, 5), trials=2000); print([t[(k, 'shuffled')] for k in (3, 4, 5)])"`
- Proposition 1 (AWA). There are weights such that one AWA layer resolves every depth ≤ K + 1 in every order,
  with mass on the correct literal ≥ (1 − n e^{−beta})·[c_min(1 − (n + e^tau)e^{−beta'})]^{k−1}·(1 − c_lit),
  beta' = kappa tanh(beta/kappa), c_min = lambda·sigmoid(beta_g) the smallest gate value at an alias,
  c_lit = lambda·sigmoid(−beta_g) the largest at a literal, beta_g the gate margin. (The earlier form used
  lambda in place of c_min and had no absorption factor; it overstated the bound.) Order invariance of a
  *trained* AWA is a prediction (H3b), not a theorem: kbar and t are pre-RoPE, but hop 1 uses RoPE scores.
- Not covered. Deep standard stacks can combine forward pointer-jumping with backward origin propagation, so
  there is no clean law for them. The contraction bound 2·lambda^K is vacuous at lambda ≈ 0.99 (1.77 at K = 12);
  depth generalisation relies on learned absorption. Fixed-K AWA is in TC⁰ like any constant-depth Transformer.

## 5. Task and data (synthetic; nothing is downloaded)

VarDeref v2 (`awa3/task.py`).
- 72-token vocabulary: `v0..v47`, `n0..n15`, `= ; ? <bos> <eos> <pad>`.
- Program: `<bos>` [literal block] [alias block] `? q`. M = 16 facts, so n = 67 tokens at every depth: length is
  not confounded with depth. The queried chain has depth k (k − 1 aliases and one literal fact); distractor
  chains have depth U{1, 2, 3}; every chain ends in a distinct literal.
- Target: the literal q resolves to, read at the last position. The readout is over the answer alphabet: a
  prediction is the arg-max over the 16 literal classes and the loss is the cross-entropy over those 16 classes,
  in training, validation and evaluation alike (`task.answer_scores`; §15 gives the reason). The floors h_k are
  floors of exactly this 16-way decision.
- Test and validation items are single-query programs in this format. Training programs carry more supervised
  queries (§5.1).

Paired strata. Each frozen base program has four variants that re-order only the queried chain's aliases among
the program positions they already occupy: sorted (definitions first), one-swap (sorted with one uniformly drawn
adjacent transposition, k ≥ 3), anti (reading order), shuffled (the training distribution). At k ≤ 2 the four
coincide.

Floors on the frozen test items, measured by this implementation (identical in all four strata, checked by a
test; h_6 and h_8 equal the program's values):

| k | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| h_k (uniform over the literals present) | 0.115 | 0.122 | 0.130 | 0.137 | 0.149 | 0.161 | 0.193 | 0.239 | 0.303 |
| last literal fact | 0.114 | 0.128 | 0.150 | 0.148 | 0.174 | 0.170 | 0.198 | 0.276 | 0.284 |
| first literal fact | 0.116 | 0.092 | 0.130 | 0.116 | 0.112 | 0.157 | 0.204 | 0.220 | 0.344 |
| P_f (dependency-sorted share, shuffled) | 1 | 1 | 0.546 | 0.166 | 0.038 | 0.009 | 0.000 | 0.000 | 0.000 |

The program text quotes P_f = 0.529 / 0.183 at k = 3 / 4 from a 1,000-item probe; the frozen sets have 500
items at those depths and give 0.546 / 0.166. The analysis uses the frozen-set values.

- The literal block removes the first design's confound (a "first literal" guess scored 0.675 on sorted at
  k = 8 when literal facts were interleaved).
- Test sets, frozen per k ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12}: 1,000 base programs at k ∈ {6, 8}, 500 elsewhere,
  each in four strata. Streams are string-keyed (`test|k=…`, `val|k=…`, `train|<seed>`, `budget|<seed>`).
- Validation sets: 600 items per depth at k ∈ {1, 2, 3} (in-distribution), 300 per depth at k ∈ {4, 5, 6, 8, 10,
  12} (oracle budget).
- Overlap of the first 40,000 training programs of `train|0` (the program with its own query, i.e. the first 67
  tokens of a training sequence) with the k ≤ 3 test items: 0 (measured again with the stream of §5.1).
- The digests of every test and validation set are written into `frozen_sets.json` at the freeze.

### 5.1 Training signal (`task.make_batch`; introduced before the freeze, §15)

A training sequence is a test-format program of the training distribution (shuffled stratum, k ~ U{1, 2, 3})
followed by more query pairs: `<bos>` [16 facts] `? q` `? x_1` … `? x_19`, 105 tokens. Every query token is a
supervised row. `task.TRAIN_SIGNAL = (8, 4, 4, 4, 0)`, in the order of the table:

| Group | Rows | Query `? x` | Target | Facts the row can see |
|---|---|---|---|---|
| forward | 8 | row 0: the program's own query q (exactly the test format); then 7 other variables of the program, without replacement | the literal x resolves to (depth 1–3) | all 16 |
| inverse | 4 | a literal of the program, without replacement | the variable that is assigned x | all 16 |
| key-value, forward (`prefix_forward`) | 4 | the variable of a fact drawn uniformly among the first m facts | its literal (depth 1) | the first m facts, m ~ U{2..number of literal facts} |
| key-value, inverse (`prefix_inverse`) | 4 | the literal of a fact drawn uniformly among the first m facts | its variable | the first m facts, m ~ U{2..number of literal facts} |
| chain (`chain`) | 0 | the variable of a uniformly drawn alias fact | its literal (depth 2–3) | the facts on its path plus U{1..4} other literal facts |

- Independent rows. An attention mask (`task.query_mask`), applied in every layer of every arm
  (`arms.row_logits`), lets a query pair see `<bos>`, the facts of its row and itself, and nothing else; the
  fact positions are not restricted. No pair sees another pair and no answer is ever fed back, so a row cannot
  use another row: no elimination among answers, no memoisation, no re-queried state of another row in the
  loop arms. Tested for every arm: row 0 equals the single-query forward of the program (2e-6); no row changes
  when the other pairs are replaced; a full or key-value row does not change when the facts it cannot see are
  replaced.
- Key-value rows. The literal block comes first, so the first m facts are m key–value pairs at the start of
  the sequence. Such a row is a single-query program with 2 to (number of literal facts) facts. These rows are
  where the lookup is discovered (§15.2); the full rows carry it to 16 facts.
- Chain rows are implemented and tested but switched off (0 rows): four of them per program did not make any arm
  learn a second hop (§15.4). A chain row is not a program of its own in the strict sense: the facts it sees were
  encoded by the binding layer in the presence of the earlier facts it does not see.
- Loss: the mean over the 20 rows of the cross-entropy over the row's answer alphabet (16 literals for forward
  rows, 48 variables for inverse rows). Validation and evaluation use forward queries only, single-query
  programs, the 16-literal readout.
- What does not change: every training depth is ≤ 3, in every row; the program distribution and the shuffled
  stratum; the data and budget streams and common random numbers; the test and validation sets (same digests),
  the four paired strata and their equal floors; all arms receive the same batches, rows and mask.
- Length. Every training sequence has 105 tokens; the length is a function of the signal, not of a depth.
- Hand-set check of the batch path (`handset.check_training_batch`, tested): the facts of every row are read
  back from the token tensor and resolved with the hand-set operators of §4. AWA reproduces the label of every
  forward row, the lower-triangular operators exactly those of the dependency-sorted rows, and the inverse rows
  are reproduced by a direct lookup (64 programs of `train|0`: 768 of 768 forward rows, 711 of them
  dependency-sorted, 512 of 512 inverse rows). With the labels shifted by one row the agreement falls to 69 of
  768 and 34 of 512.
- The training log (`curve.jsonl`) carries, next to the loss over all rows, the loss and accuracy of row 0 (the
  test-format query) and the accuracy of every group.

## 6. Arms and controls

All arms are tiny Cevahir models built with `tiny_model_config` + `ModelManager.initialize`: D = 128, 4 heads,
4 KV heads, 2 layers, SwiGLU, RMSNorm pre-norm, RoPE, tied embeddings, dropout 0, fp32. All use the same exact
rows-only final layer. Parameter counts are asserted by `awa3/arms.py` at every build.

| Arm | Final-layer computation | Params | Seeds | Role |
|---|---|---|---|---|
| AWA-2L | prefix-closed walk, K_train ~ U{3, 4, 5} | 551,308 | 5 | treatment |
| ChaCAL-2L (ffn_dim 534) | causal resolvent, exact | 551,300 | 5 | closest published operator; H2(a), H3a |
| AWA-own-2L | own-window walk, K_train ~ U{3, 4, 5} | 551,308 | 5 | single-ingredient ablation; H2(b), H3a |
| LOOPED-2L (ffn_dim 534) | binding layer + shared block × T, T_train ~ U{3, 4, 5} | 551,296 | 4 | recognised test-time depth; H6 |
| ARLOOP-2L (ffn_dim 534) | final layer × T at the answer row | 551,296 | 4 | vector re-query; H6 |
| STD-2L-pm (ffn_dim 534) | standard | 551,296 | 3 | negative control for assumption B |

Initial weights for a given seed (tested). The engine draws operator parameters after every other weight, so
arms of equal shape start from identical common weights: ChaCAL-2L, LOOPED-2L, ARLOOP-2L and STD-2L-pm share
every common weight, and AWA-2L and AWA-own-2L share every weight. Between the two groups, which differ in
ffn_dim (512 against 534), only the weights drawn before the first FFN coincide (the embedding and the first
layer's query, key and value projections). The data and budget streams are common to all arms.

## 7. Training, seeds, units

Recipe (`awa3/train.py`).
- Batch 64; AdamW betas (0.9, 0.98), eps 1e-8, weight decay 0.01 on matrices only (none on norms, biases, rho,
  tau, gate biases). Optimiser groups are built from the live model, after the operator parameters exist.
- 100 warm-up steps, cosine decay to 0.1× at step S, constant 0.1× afterwards; gradient clipping at 1.0.
- S = 3,000 steps, fixed in advance. Training depths k ∈ {1, 2, 3}, shuffled stratum.
- Training signal: the 20 independent query rows of §5.1 per program, loss over the answer alphabet. Every arm
  computes all rows of a batch in one forward (`arms.row_logits`), with a rows-only final layer.

Phase 0, learning-rate pilot. (The rules below were written for the original signal and have not been run with
the signal of §5.1; §12 item 13 lists what has to be re-decided. In particular the GO rule on k = 2 cannot pass
while no arm learns a second hop, §15.4.)
- lr ∈ {5e-4, 1e-3, 2e-3} × {AWA, AWA-own, ChaCAL, LOOPED, ARLOOP}, seed 900, 800 steps each (15 runs). A pilot
  is the first 800 steps of the main schedule (same warm-up and cosine horizon S = 3,000).
- Selection: the lowest in-distribution validation loss on the arm's competence cells at step 800; losses
  within 2% of the best go to the smaller lr. AWA-own has its own lr. STD-2L-pm uses 1e-3.
- GO for AWA if, at some lr, k = 2 accuracy is ≥ 0.30 at step 800, or the k = 2 validation loss at step 800 is
  ≤ 0.8× its step-200 value. Otherwise the best-lr AWA run is extended once by 800 steps and the same rule is
  applied at step 1,600.
- NO-GO: H2 is not started (a curriculum variant needs a new protocol). H3a still runs: Phase 1 is then
  ChaCAL × 5, AWA-own × 5 and STD-pm × 3 (`jobs_phase1_nogo.json`).

Competence check at step 3,000 (validation, budget 5).
- Walks and loops: mean shuffled accuracy over k ≤ 3 ≥ 0.90.
- ChaCAL-style and AWA-own: mean *sorted* accuracy over k ≤ 3 ≥ 0.90 (the cells they can represent).
- STD-2L-pm: k = 1 accuracy ≥ 0.90 (binding competence, H1); no extension.
- A failing run gets one +1,500-step extension from a fixed reserve of 1.0 core-hour, in the priority order
  AWA > ChaCAL > AWA-own > LOOPED > ARLOOP, then by seed (`run_extend.py`). The extension depends only on the
  in-distribution check; it runs before any evaluation job. A run that still fails, or is not funded, is "not
  ID-competent". The ledger (`extensions.json`) books the measured CPU seconds of every extension; after a kill,
  a restart first books extensions that finished without a ledger entry, from the run's own records.

RNG.
- Data come from `random.Random(f"train|{seed}")`, shared by all arms with the same seed (common random
  numbers). Budgets come from `random.Random(f"budget|{seed}")`, drawn once per step in every arm.
- `torch.manual_seed(seed)` is called before the build. Pilot seed 900 is disjoint from confirmatory seeds 0–4.
- Runs are single-threaded and bitwise reproducible; a killed run resumes from its checkpoint and continues
  bitwise identically (tested).

Launch order. Seeds are launched round-robin across arms (seed 0 of every arm, then seed 1, …), so that partial
completion stays balanced.

## 8. Evaluation (`awa3/evaluate.py`)

- One pass per (depth, stratum) gives the whole budget axis: K ∈ 0..24 for walks, T ∈ 1..16 for LOOPED,
  T ∈ 1..24 for ARLOOP. Exact counts of correct answers are stored.
- Items are the frozen single-query programs (67 tokens, no mask). A prediction is the arg-max over the 16
  literal classes (`task.predict`), the readout the models are trained with (§5); the floors h_k refer to it.
- Also recorded: the operator-off ablation (walks: K = 0, i.e. lambda := 0; ChaCAL-style: gamma := 0); hop
  traces on 200 items per (k, stratum), k ∈ {3, 4, 6}, on validation and on test items; oracle-budget validation
  sets (300 items per depth and stratum, k ≥ 4); accuracy by the number of out-of-order adjacent aliases;
  lambda and the mean c by position type (variable RHS, literal RHS, `;`, other); gamma; analytic MACs.

## 9. Cost model (`awa3/arms.py: macs_per_answer`)

Analytic multiply-accumulates for one answer at n = 67. It counts the arithmetic an exact single-answer
evaluation needs (linear maps, attention scores and value mixing, the SwiGLU FFN, and for the walk the pointer
and gate projections, E, the window normaliser and K steps of a 1×n by n×n product per head), not the batched
driver: ARLOOP is counted as re-projecting only the changed answer row. Norms, RoPE, softmax and additions are
not counted.

| Arm | Budget 1 | Budget 5 | Budget 12 | Budget 24 |
|---|---|---|---|---|
| STD-2L-pm | 21.74 M | | | |
| ChaCAL-2L (exact) | 23.40 M | | | |
| AWA-2L / AWA-own-2L | 22.91 M | 22.98 M | 23.10 M | 23.32 M |
| ARLOOP-2L | 21.74 M | 22.89 M | 24.90 M | 28.36 M |
| LOOPED-2L | 21.74 M | 98.85 M | 233.81 M | 465.15 M |

So at budget 12 a walk costs about a tenth of LOOPED and about as much as ARLOOP. ARLOOP scales as O(T n D) per
answer and AWA as O(H n² d + K H n²), so ARLOOP wins at long n unless top-r transitions are used (not
implemented).

## 10. Statistics (`awa3/stats.py`, `awa3/analysis.py`)

- The unit is the seed; there is no bootstrap.
- Per arm and cell: per-seed values, medians, Hodges–Lehmann estimates with exact rank-based intervals, and
  chance-corrected accuracy.
- Tests: exact T4 (H2); shifted exact Mann–Whitney (reported); exact two-sided Mann–Whitney with Holm (H6(i));
  HL on log costs with exact intervals (H6(ii)); one-sided t-bounds over seeds (H3a, H3b) with Holm over the two
  H3a arms.
- `analysis.compute_decisions` is a pure function of the per-run records; `run_verify.py --rerun` recomputes it.

## 11. Compute plan, gates and jobs

### 11.1 Plan (core-hours; the program's slow-probe budget, to be re-anchored on Phase-0 step times)

| Item | Detail | Budget |
|---|---|---|
| Phase 0 | 15 runs × 800 steps | 1.39 |
| AWA-2L | 5 × 3,000 | 1.29 |
| ChaCAL-2L | 5 × 3,000 | 1.17 |
| AWA-own-2L | 5 × 3,000 | 1.29 |
| LOOPED-2L | 4 × 3,000 | 2.73 |
| ARLOOP-2L | 4 × 3,000 | 1.20 |
| STD-2L-pm | 3 × 3,000 | 0.63 |
| Extension reserve | priority order (§7) | 1.00 |
| Evaluation | ≈ 19k items × 26 runs, budget axes, traces | 0.90 |
| Total | | 11.60 |
| Hard cap | | 12.0 |

This table is no longer valid. It was computed from single-query step times (0.15–0.48 CPU-s). With the signal
of §5.1 (105 tokens, 20 rows) the diagnostic runs of §15 measured, on a machine shared with other jobs, 0.27
CPU-s per step for STD-2L-pm and 0.38 for AWA-2L and ChaCAL-2L; with four more rows (113 tokens) 0.37, 0.44–0.50
and 0.41–0.50. LOOPED-2L, ARLOOP-2L and AWA-own-2L have not been timed with the new signal. At 0.38 CPU-s a run
of 3,000 steps costs 0.32 core-hours instead of the 0.26 planned for AWA-2L, and the plan has to be re-anchored
(fewer rows, a shorter S, or a higher cap) before the freeze (§12 item 14).

### 11.2 Gates

1. Code gate: the tests of the engine operator and of this pipeline pass.
2. Phase-0 GO / NO-GO for AWA (§7).
3. Freeze: the manifest is hashed before the first Phase-1 launch (`p1_freeze_check` fails otherwise).
4. Competence gate at step 3,000 with one funded extension.
5. Hard cap. The external scheduler owns CPU accounting. Cap order if the cap would be crossed: drop STD-2L-pm
   seeds 2 and 1 (seed 0 is the only control for B), then ARLOOP seed 3, then LOOPED seed 3, then the extension
   requests of LOOPED and ARLOOP. Evaluation and the H2 / H3a arms (AWA, ChaCAL, AWA-own) are never dropped.
   Dropped runs are reported as "not run (cap)"; with 3 seeds, H6(i) against that baseline is descriptive.

### 11.3 Jobs (`jobs_phase0.json`, `jobs_phase1.json`, `jobs_phase1_nogo.json`)

Lists of `{id, cmd, deps, log}`; every `cmd` runs from the repository root and every driver is single-threaded
and resumable. Phase 0: 15 pilots, then `p0_select`. Phase 1: `p1_freeze_check`, 26 training runs, `p1_extend`,
26 evaluations, `p1_analyze`, `p1_manifest`, `p1_verify`. The files are generated by `make_jobs.py` and a test
keeps them equal to the generator.

### 11.4 Pipeline smoke (not an experiment)

Recorded in README.md ("Smoke run"): six arms, 40 steps each, truncated evaluation, analysis, manifest and
recomputation, with the measured CPU seconds per step per arm. It exercises the code path only; its accuracies
are at the chance floor and mean nothing. It was run with the original training signal and has not been
repeated with the signal of §5.1 (the step times and losses in that table are those of the original signal).

## 12. Open decisions to close before the freeze

The program text leaves the following details open. The code implements the choice stated here; each must be
confirmed or changed (with the tests) before the freeze.

1. Pilot schedule. A pilot runs the first 800 steps of the S = 3,000 schedule (not a cosine compressed into 800
   steps). The AWA extension continues the same schedule to step 1,600 and the GO rule is re-applied there.
2. Extension schedule. Extension steps 3,001–4,500 run at the floor learning rate (0.1×).
3. Non-competent runs in H2. H1 governs AWA: a non-competent AWA seed stays in the analysis with its measured Y
   (it counts as a failure in the seed count and in T4). A comparator run that is not ID-competent makes its
   contrast INCONCLUSIVE: the contrast's numbers are reported, but it can neither pass (H2 cannot be CONFIRMED)
   nor supply the "HL < 0.15" refutation. H2 is then REFUTED only by the AWA-only clause of §3.2 (at most 1 of 5
   AWA seeds with Y ≥ 0.80 and median Y < 0.50; no comparator enters it) or by HL < 0.15 in the other contrast,
   if all of that comparator's runs are ID-competent; otherwise H2 is INCONCLUSIVE. `decisions.json` lists the
   inconclusive contrasts and the clauses that refuted.
4. H3a refutation test. Per arm, the p-value is twice the smaller of the two one-sided t-test p-values (one-swap,
   anti) of H0: mean(acc − h_k) ≤ 0.15, capped at 1; Holm runs over the two arms. The mixture error is the mean
   absolute error over k ∈ {3, 4} per seed, and "unfollowable accuracy" is the arm's anti accuracy. Like H2, H3a
   is INCOMPLETE (no decision, also no REFUTED) until all five seeds of both arms are in.
5. Budgets of the walk arms in H3a and H3b: K = 12. STD-2L-pm cells in the H1 control: the shuffled stratum.
6. Oracle T (loops). Per depth, the T that maximises the worst-stratum validation accuracy (ties: the smallest
   T). At the in-distribution depths, which have no oracle validation set, "best T" is taken on the shuffled
   test cell (used only for the per-hop fit).
7. H4. acc(k) is the mean over the four strata. (a) is SUPPORTED if every seed has at least two qualifying
   depths with equal offsets and a sharp step; (c) chooses, per seed, the head and delta ∈ {−1, 0, 1} with the
   largest mean validation mass, and requires the pooled test median ≥ 0.8 for every hop; (c) is reported as
   SUPPORTED / NOT SUPPORTED (no refutation rule was specified).
8. H5 is tested only at depths whose seed-mean plateau accuracy is ≥ 0.5 (a flat curve at the floor would pass
   vacuously); the curve is the seed mean of the mean over strata.
9. H6. AWA's Y at K = 12 is compared with the baseline's Y at its oracle T; a baseline with fewer than its
   planned seeds is descriptive only. In (ii), seeds that do not reach Y† are left out of the HL estimate when
   fewer than 3 seeds of an arm fail, and (ii) is "not testable" when Y† is not above the chance floor (the mean
   of h_6 and h_8). Reading per baseline: (i) decides only when it is significant after Holm and |HL| ≥ 0.15
   ("AWA beats B" / "B beats AWA"); otherwise (ii) gives one of its three readings; when (ii) has no cost
   estimate (not testable, or an arm does not reach matched accuracy) the reading is "not decided" with that
   reason; a baseline with fewer than its planned seeds reads "descriptive only". H6 as a whole is INCOMPLETE
   until all five AWA seeds and at least one run of each baseline are in.
10. MAC counts are those of §9 (single-answer arithmetic, ARLOOP re-projecting one row).
11. CPU cap and drop order are enforced by the external scheduler, not by this code; only the extension reserve
    is enforced here.
12. The ChaCAL paper has to be read (arXiv 2410.05565) before any write-up; the arm keeps the label
    "ChaCAL-style" otherwise.
13. Learnability (§15). Open, and blocking the freeze: no arm has learned a second hop. Until that is solved
    the Phase-0 rules of §7 cannot be applied as written: the GO rule reads k = 2, and the learning-rate
    selection reads the competence cells k ≤ 3, of which only k = 1 moves. To decide: (a) the remedy for the
    second hop (§15.5), or the decision to report H1 as failed for the reason of §15.4; (b) the Phase-0 rules
    for the new signal (it has only been run at lr 1e-3 and one seed, and the step at which the lookup forms
    varied between 300 and 1,500 across arms and runs, so 800-step pilots may end before it); (c) whether the
    lookup of the ChaCAL-style arm, of AWA-own and of the loop arms forms within S at all (ChaCAL-2L had not
    formed it by step 1,000–1,200 in three runs; the other three arms were not run).
14. Training signal (§5.1). The row counts (8, 4, 4, 4, 0) are those of the diagnostic runs, not the result of
    a search, and each ingredient is supported by single runs (§15.2). To confirm or change: the counts, the
    chain rows (off), and the compute plan of §11.1 that follows from them.

Not implemented from the program's list (stated so that nobody assumes them): top-r sparse transitions (E6), an
E cache for decoding (the engine recomputes E from cached pointers and addresses each step), the λ = 0 insertion
test on a CEV-11 base (E2; the reduction itself is tested on tiny models), the Turkish "aynı" phase.

## 13. Risks, and what a negative result teaches

| Outcome | Reading | Next |
|---|---|---|
| NO-GO at Phase 0 | credit assignment through products of transitions is hard to discover from answer-only loss | curriculum or hop-wise auxiliary targets (new protocol); H3a still answers the theory. Observed before the freeze in a stronger form (§15.4): no arm learns a second hop, the binding layer binds literal facts only, and a first-hop auxiliary target and short programs did not change that |
| H3a refuted | B fails in trained models (layer 1 partly composes) or the implementation is not lower-triangular; the hand-set check separates them | the law holds for idealised binding only; test a 3-layer binding stack |
| H2 INCONCLUSIVE, label "compounding" | the algorithm is right but imprecise; eps quantifies it | precision training (longer S, depth curriculum) |
| H2 INCONCLUSIVE, label "depth cliff" | the walk learned depth-specific shortcuts | H4(a) locates the shortcut |
| LOOPED or ARLOOP match AWA (H6) | recurrence suffices; AWA's value reduces to order invariance in one layer, cost against LOOPED, and the theory | an ambiguous-pointer task, where position-space walks and vector re-query differ |
| H3b refuted | order leaks through hop-1 RoPE scores or positional content in x | NoPE hop 1 inside AWA heads |
| H2 CONFIRMED | a causal head with order-free multi-hop resolution, plus a law that explains why its published relatives fail | a natural-language phase under a new protocol |

Out of scope: language-model perplexity, long contexts, full-sequence training cost (all-row AWA is O(K n³) per
head; this experiment uses the rows-only final layer).

## 14. Integrity and reproduction

- Nothing confirmatory is inspected before all confirmatory runs finish; evaluation jobs depend on the extension
  job, and the analysis job on every evaluation.
- Freeze: `python research/attention_order_law_2026_11_13/run_verify.py --freeze` writes `frozen_sets.json` and
  `freeze_manifest.sha256` (study code, PROTOCOL.md, job files, the two test files, the engine files the arms
  are built from, and the Phase-0 selection). AMENDMENTS.md is deliberately outside the manifest.
- Every Phase-1 training run re-checks the freeze before it starts.
- Results: `run_verify.py --manifest` hashes code and results; `run_verify.py --rerun` recomputes the decisions and
  compares them with the stored file and the analysis code with the frozen hashes.
- Commands are listed in README.md.

## 15. Pre-freeze design changes (2026-11-13 and 2026-11-14)

Everything in this section was done before the freeze and is part of the protocol's history, not an amendment.
The runs quoted here are diagnostic: one seed each (900 or 901), lr 1e-3, one thread, written to a scratch
directory, not part of the repository and not experiments of this protocol. "val" is the frozen in-distribution
validation set (first 200 items per depth), read with the 16-literal arg-max; the floors of §5 are h_1 = 0.115,
h_2 = 0.122, h_3 = 0.130 (measured on the test items; 200 validation items scatter by about ±0.05 around them).

### 15.1 What failed

- Phase-0 pilots, original signal (one supervised token per program, cross-entropy over all 72 classes): 5 arms
  × lr {5e-4, 1e-3, 2e-3}, 800 steps. Every run ended at loss ≈ 2.7 ≈ ln 16 with accuracy 0.07–0.11.
- 5,000 steps (320,000 programs), lr 1e-3: STD-2L-pm k = 1 accuracy 0.105, loss 2.70; AWA-2L mean accuracy over
  k ≤ 3 0.133, loss 2.58. A standard 2-layer model did not learn a single lookup.

### 15.2 Diagnosis

The pipeline is sound. The training signal was not learnable, for three separate reasons.

Pipeline checks.
- Memorisation: STD-2L-pm on 64 fixed programs reaches loss 0.0033 at step 250 (training accuracy 1.00,
  validation at chance). Optimiser, forward and labels are consistent.
- Labels and positions: the hand-set operators reproduce every label of a training batch from its tokens (§5.1).
- Forward: the training forward at row 0 equals the single-query forward used in evaluation (tested, 2e-6).

Cause 1: the 72-way loss collapses the token embeddings.
- The embedding matrix is tied to the output layer and is followed by a LayerNorm, and only literals are ever
  targets. The first thing the loss rewards is "a literal, not a variable", and the cheapest way to say it is to
  move all variable embeddings into one direction and all literal embeddings into the opposite one.
- Measured (STD-2L-pm, 6 supervised depth-1 queries per program with the answers fed back, 72-way loss): mean
  cosine between the 48 variable embeddings after the embedding LayerNorm 0.003 at step 0, 0.65 at step 100,
  0.84 at step 400; between literals 0.004 → 0.80 → 0.83; between a variable and a literal −0.008 → −0.71 →
  −0.83. Gradient norm 0.52–0.68 on the embedding and at most 0.03 on any other tensor. Answer-row attention
  entropy 0.1–1.8 nats (uniform: 4.2). The loss stays at ln 16; val k = 1 0.095 at step 1,500.
- With the loss over the 16 literal classes, same run otherwise: literal–literal cosine −0.05 at step 400, and
  the loss reaches the level of "some literal that occurs in the program" (2.22 at step 550, 2.07 at step
  1,150). The lookup is still not learned (cause 3).
- Restricting the readout further, to the literals that occur in the program, is worse: the gradient norm
  falls to 0.02–0.05 by step 500 and the loss stays at its start value through step 1,100.

Cause 2: answers fed back into the sequence are used for elimination.
- With several queries written as `? q a ? q a …` (4 key–value facts, 4 queries), the model reaches accuracy
  0.52–0.54 = (1/4 + 1/3 + 1/2 + 1)/4 with the first query at 0.28 (chance 0.25) at step 1,725: it excludes the
  literals already given and does not look anything up.

Cause 3: the lookup forms after a plateau that grows with the context.
- A lookup needs the binding layer to copy the left-hand side to the right-hand-side token and the final layer
  to match the query against that copy. Neither pays without the other.
- 4 key–value facts, 4 forward queries, no answers fed back: plateau at ln 4 from step 250 to step 1,000,
  accuracy 0.94 at step 1,250 and 0.99 at step 1,300. With 2 forward and 2 inverse queries the forward queries
  are solved at step 250.
- 16 facts: 12 forward queries per program, chance at step 1,650. 6 forward + 6 inverse queries: inverse 0.87,
  forward 0.11 at step 2,250. Programs of 3–16 facts mixed per batch: inverse 0.94–1.00, forward at chance, val
  k = 1 0.125 at step 1,700.
- Key-value rows (a query pair that sees only the first m facts), signal (8, 4, 4, 4, 0):

  | Arm | val k = 1 | val k = 2 / k = 3 | Steps run |
  |---|---|---|---|
  | STD-2L-pm | 0.605 at step 200, 1.000 at step 300 and at every validation up to step 3,000 | 0.105–0.180 / 0.085–0.165 throughout; 0.125 / 0.085 at step 3,000 | 3,000 |
  | AWA-2L | 0.280 at step 500, 1.000 at step 600 and up to step 1,400 | 0.080–0.185 / 0.085–0.165 | 1,480 (stopped) |
  | AWA-2L, plus 4 chain rows | 0.505 at step 600, 0.885 at 1,000, 0.925 at 1,500, 1.000 at 2,000, 0.995 at 3,000 | 0.120 / 0.115 at step 3,000 | 3,000 |

  The first two runs used an earlier implementation of the same signal, whose stream differs from the committed
  one only in which literals the inverse rows query. The STD-2L-pm row is also the negative control of H1 for
  this signal: after 3,000 steps a standard model answers k = 1 and stays at the floor at k = 2 and 3.
- Rows that see a scattered subset of the facts instead of a prefix did not work: with them AWA-2L had not
  formed the lookup by step 900 (val k = 1 0.080), and with literal facts only as the other facts in sight
  neither AWA-2L (0.130) nor ChaCAL-2L (0.100) had by step 1,000. A likely reason, not tested: the first facts
  of a sequence are the ones whose binding-layer attention is least diluted at initialisation.

### 15.3 The change

1. Readout over the answer alphabet (§5): loss and arg-max over the 16 literal classes, in training, validation
   and evaluation.
2. Several supervised queries per training program, as independent rows (§5.1): no answer is fed back, and an
   attention mask keeps every query pair from every other pair, in every layer of every arm.
3. Inverse rows (`? n` → the variable that is assigned n), read over the 48 variable classes.
4. Key-value rows that see only the first m facts.
5. Not changed: the task, the test and validation sets (same digests), the four paired strata and the floors,
   training depths k ≤ 3, the arms, the optimiser recipe, the streams, determinism and resumption (a run killed
   by a container restart continued with the same losses).

### 15.4 What is not solved: the second hop

No arm has answered k = 2 or k = 3 above the floor in any run.

- AWA-2L, signal (8, 4, 4, 4, 0): floors up to step 1,400. With 4 chain rows (small multi-hop programs, §5.1),
  3,000 steps: k = 2 0.120, k = 3 0.115; the chain rows themselves stay at 0.24–0.43 (mean 0.32, chance among
  the literals in sight); lambda ends at 0.49–0.61 from 0.50.
- Where it stops, read from the checkpoints (200 validation programs per depth):
  - Binding. Attention of the binding layer from a right-hand-side token to its left-hand side: 0.74–0.99 in
    one head at literal facts, 0.00 in every head at alias facts (STD-2L-pm at step 3,000; AWA-2L at step 1,000
    in both runs above and at step 3,000 in the last run below). The model binds where one lookup pays and
    nowhere else.
  - Hop 1. From the answer row, 0.91 of one head's attention is on the right-hand-side token of the queried
    literal fact at k = 1; at k = 2 and 3 it puts 0.00 on that token of the queried alias fact (at most 0.10 on
    the whole fact).
  - Walk (AWA-2L, step 1,000). Raw pointer–key scores have standard deviation 8 and reach 48 (cap 20); the
    transition row out of an alias fact has entropy 2.8–3.0 nats with 0.006–0.035 on the next fact; the continue
    probability is 0.47–0.54 at literal facts in three heads, so half of the attention mass that reached the
    right fact is moved on to an effectively random position.
- Reading. Binding an alias fact pays only if something then hops from it. In AWA that needs three things at
  once (binding at alias facts, a pointer map W_T that starts random, a gate that separates alias from literal
  facts), and none of them has a gradient alone. This is the Phase-0 NO-GO of §13 in a sharper form. It is a
  statement about this signal and this initialisation; whether the operator can learn the walk is not decided.
- ChaCAL-2L: the lookup itself had not formed by step 1,000 with (8, 4, 4, 4, 4) (val k = 1 0.080, inverse rows
  0.18), nor by step 1,200 with the last variant below (0.120; inverse rows 1.00, forward rows at chance, gamma
  0.47–0.50). For the closest comparator even k = 1 is open. AWA-own-2L, LOOPED-2L and ARLOOP-2L have not been
  trained with the new signal.
- Tried on top of the signal, without success (experimental code, not part of the committed signal):
  - inverse rows over all facts, alias facts included: AWA-2L had not formed the lookup by step 1,200 (0.070);
  - first-hop rows (`? v` → the variable v is assigned, read over the variable classes, 4 per program):
    STD-2L-pm reaches k = 1 0.970 at step 500 and 1.000 at step 600 (later than without them), and the
    first-hop rows stay at 0.00–0.14 up to step 850;
  - first-hop rows plus short programs (30% of the batches are programs of 3–6 facts in which every variable is
    queried), AWA-2L, 3,000 steps: k = 1 1.000 from step 900; k = 2 0.125 and k = 3 0.130 at step 3,000
    (0.095–0.180 and 0.110–0.175 throughout); first-hop rows on 16 facts 0.14; binding at alias facts still
    0.00. On the short programs the accuracy at depth 2 and 3 looks high (0.42–0.98) but is a shortcut: with 4
    facts and depth 2 it is 0.941 when only the queried chain has an alias fact and 0.489 (floor 0.500) when
    another chain has one too. The model returns the literal that some alias fact leads to and does not use
    the query.

### 15.5 Not tried

In the order in which they look cheapest and most decisive:

1. Tie AWA's pointer to its query map: W_T = W_Q + Delta with Delta = 0 at initialisation, and a gate bias that
   starts the walk near zero. Hop 2 then starts as a re-use of the learned query–key match, as in the
   ChaCAL-style resolvent, and only alias binding remains to be learned. The lambda = 0 equivalence is
   untouched. It is a change of the engine operator's parametrisation.
2. Start the operators nearly off (`lambda_init`, `gamma_init` are operator options; no engine change), so
   that the lookup forms as fast as in the standard model (step 300 instead of 600–1,500, or never for the
   ChaCAL-style arm so far). Note that rho moves by at most the learning rate per step under AdamW, so lambda
   cannot travel from 0.05 to 0.5 within 3,000 steps at lr 1e-3: the scalar needs its own learning rate or a
   moderate start.
3. Supervise alias binding where it lives: a target at the fact itself (at the right-hand-side token, "which
   variable is assigned here"), or a linear probe on the binding layer. This makes the first half of assumption
   B true by training, which changes what a refutation of H3a would mean; it is a decision for the protocol.
4. Short programs in which the shortcut of §15.4 cannot work (every program with at least two alias chains),
   as a scheduled curriculum; accuracy on them logged by depth.
5. A learning-rate sweep and more than one seed (the step at which the lookup forms varied from 300 to 1,500
   between arms and runs), and the three arms that were never run.
6. A second binding layer (3-layer arms).
