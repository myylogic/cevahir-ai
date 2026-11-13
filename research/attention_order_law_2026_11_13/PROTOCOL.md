# AWA-3 "the order law in attention" — PROTOCOL (DRAFT, NOT FROZEN)

Status: DRAFT, NOT FROZEN. Written on 2026-11-13 (UTC) from the research program's flagship section
(scratch `PROGRAM.md` §3, including its "Critic notes" and "Response to critic") and the prototype
`attention-math/` (proposal_v2.md, awa_v2.py, test_awa_v2.py, probe_v2.py, order_law_sim.py). No Phase-0 and no
Phase-1 run exists. Only a pipeline smoke with tiny step counts has been run (§11.4); it tests nothing.

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
- Target: the literal, at the last position; single-token cross-entropy.

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
- Overlap of the first 40,000 training programs of `train|0` with the k ≤ 3 test items: 0.
- The digests of every test and validation set are written into `frozen_sets.json` at the freeze.

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

Phase 0, learning-rate pilot.
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
are at the chance floor and mean nothing.

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

Not implemented from the program's list (stated so that nobody assumes them): top-r sparse transitions (E6), an
E cache for decoding (the engine recomputes E from cached pointers and addresses each step), the λ = 0 insertion
test on a CEV-11 base (E2; the reduction itself is tested on tiny models), the Turkish "aynı" phase.

## 13. Risks, and what a negative result teaches

| Outcome | Reading | Next |
|---|---|---|
| NO-GO at Phase 0 | credit assignment through products of transitions is hard to discover from answer-only loss | curriculum or hop-wise auxiliary targets (new protocol); H3a still answers the theory |
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
