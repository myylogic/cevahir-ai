# FWL — learning in the forward pass: protocol

Status: NOT FROZEN (draft of 2026-11-13, revised 2026-11-14 after an independent review; see §4.1)

This protocol merges two reviewed proposals of the 2026-11-13 research programme: FEC ("Forward Error Credit",
lens *beyond-backprop-forward*) and DEFTER (lens *no-freeze-learning*). Both were judged **incremental** by their
novelty reviews and **fixable** by their methods reviews. The mechanism is implemented in the engine
(`src/neural_network_module/ortak_katman_module/forward_learning.py`, optional, off by default); this folder
holds the study that measures it. **Nothing has been measured under this protocol.** The discovery jobs have not
been run, no operating point is selected, and no forward-learning arm has touched a confirmatory base. A reduced
smoke run on discovery world 0 proved the pipeline and is described in §9; its numbers are not protocol data.
An independent review (2026-11-14) changed three primary tests back to what the methods review of FEC had
required, replaced the step-cap rule of the stream and tightened the integrity checks; §4.1 lists every
deviation from the two source proposals that remains.

To freeze: settle §11, run the discovery jobs, run `run_select.py`, replace the status line by
`Status: FROZEN <YYYY-MM-DD>` and run `verify.py --freeze`. Confirmatory worlds are refused before that.

## Türkçe özet

1. **Ne ölçülüyor.** Cevahir'in çekirdeğine eklenen, varsayılan olarak kapalı bir "ileri geçişte öğrenme"
   katmanı: model bir cümleyi bir kez okur, çıkış hatasını kapalı biçimde artık akıma indirir, `ffn.fc2`
   matrislerine rank-bir hızlı ağırlıklar yazar; geri yayılım da optimizer da yoktur.
2. **İddia dar.** İki önerinin yenilik eleştirileri "artımlı" kararı verdi: kapalı biçimli hata FPO'da,
   hedef gömmesiyle yazma Backward Lens'te, yol mertebesi ailesi Highway-BP'de, RLS izdüşümü RLSEdit / AlphaEdit /
   OWM'de, MLP izdüşümünde hızlı ağırlık In-Place TTT ve Fast Weight Layers'ta, katman içi en küçük kareler
   MesaNet'te, dikkat skorlarında dur-gradyan TinT'te var. Burada yalnız eleştirilerin ayakta bıraktığı dar
   maddeler sınanıyor (§1).
3. **Hipotezler.** H1: tek okumalı FEC-A, FT-ALL'dan (güçlü karşılaştırıcı) en çok 0,04 geride (sızıntı ve bpb
   korumalarıyla). H2a: dikkatle yönlendirme (FEC-A − FEC-TOP) ≥ 0,05; H2b: FEC-A, tek adımlı tam gradyandan en çok
   0,05 geride; rastgele geri besleme denetimi ≤ %10 kurulum. H3: 100 olguluk akışta FEC'in yazım başına net olgu
   sayısı, ardışık karşılaştırıcıların **her birinden** (ardışık FT-L, izdüşümü eşlenmiş tam gradyan,
   RLSEdit-benzeri, HEBB) en çok 0,05 geride. H4a: zor olumsuzlarda sızıntı
   ≤ %2 çalışma noktasında kapılı okuma, kapısız okumayı ≥ 10 puan geçer; H4b: kapasite dizi öngörüsü ±%25
   içinde. H5: defterden geri alma ile "olgu hiç görülmedi" yeniden oynatması arasındaki fark ≤ 0,05 nat.
4. **Birim.** Dünya (bağımsız ön-eğitimli CEV-11 tabanı), n = 8 doğrulayıcı taban; yeni ön-eğitim yok.
5. **Dürüst öngörü.** H3'ün ve H5'in desteklenmemesi bekleniyor (tek keşif tabanındaki yoklamada adım tavanı
   verilen tam gradyan kolu iki FEC kolunu da geçti, §9); H2a belirsiz (FEC-TOP pilotların bir kısmında FEC-A ile
   aynı düzeyde); H4b için DEFTER v2'nin yazıldığı biçim duman koşusunda yaklaşık 2 kat şaştı (§9).
6. **Kapsam dışı.** Meta-eğitimli DEFTER kolu (yeni ön-eğitim ister), yol-sınıfı çekirdeği (PCK) ve kesik krediyle
   ön-eğitim; hepsi ileriki iş (§10).

---

## 1. Origin, prior work and what is claimed

### 1.1 Prior work (credited; this study claims none of it)

Evidence is abstract-level: `arxiv.org` was not reachable when the two novelty reviews were written, so every
method detail of another paper below is unverified against its full text, and identifiers are as the reviews
gave them. "Not found" means not found in those searches (41 and about 45 queries).

| Element here | Published as | Work |
|---|---|---|
| Closed-form output error applied to MLP weights of several layers, no cross-layer signal (our FEC-0, incl. the per-layer cosine diagnostic) | FPO | Patil et al., arXiv 2608.14563 |
| One forward pass, write FF2 from the last-position MLP input and the target embedding (our HEBB-0 / HEBB-TOP) | "forward pass shifting" | Backward Lens, Katz et al., EMNLP 2024, arXiv 2402.12865 |
| Gradient as a sum over residual paths with a truncation order whose limit is backprop (our TOP / 0 / 1 / A family) | Highway-BP | Fagnou et al., ICLR 2025, arXiv 2501.17086 |
| Stop-gradient through attention scores, value-only updates inside a forward pass | TinT | Panigrahi et al., arXiv 2307.01189 |
| Analytic top-down transport through value and gate pathways on SwiGLU transformers | Dual Path Attribution | arXiv 2603.19742 |
| Sequential edits as online regularised least squares with a Woodbury recursion (our projector and write form) | RLSEdit; AlphaEdit; OWM | arXiv 2601.15686; 2410.02355; 1810.01256 |
| The FFN's final projection as an in-place fast weight written chunk by chunk | In-Place TTT | Feng et al., ICLR 2026, arXiv 2604.06169 |
| The output-loss gradient as the value of a rank-one fast-weight write; dynamic evaluation | Fast Weight Layers (FWL); dynamic evaluation | Clark et al., EMNLP 2022, arXiv 2212.02475; Krause et al. 2018 |
| Ridge regression solved inside the layer; Kalman / RLS layers with explicit precision | MesaNet; Gated KalmaNet; Kalman Linear Attention | arXiv 2506.05233; 2511.21016; 2602.10743 |
| Key-covariance preconditioning beats the Hebbian write | test-time regression; Preconditioned DeltaNet | arXiv 2501.12352; 2604.21100 |
| The statistic nu (variance reduction at a key) | approximate linear dependence in kernel RLS, sparse online Gaussian processes, linear-bandit bonus | textbook (cited from memory by the reviews) |
| Rank-one gradient structure, learned gradient transforms for edits; reversal of rank-one edits | MEND; Youssef et al. | arXiv 2110.11309; 2505.20819 |
| Next-token-loss updates of the last blocks' MLPs with an inner backward pass | TTT-E2E | arXiv 2512.23675 |
| Forward-only, closed-form fast weights by ridge regression on key-value pairs at test time | FAAST | Bao et al., arXiv 2605.04651 |
| Backprop-free editing: forward-only (zeroth-order) gradient estimates with ROME's closed-form write | MobiEdit | arXiv 2506.13772 |
| A learned low-rank map from the final prediction error to a local correction per block, then closed-form solves (a learned counterpart of the routing here) | LoCA | Wang et al., arXiv 2608.03020 |
| Surprise-driven test-time writes with gates and forgetting | Titans | Behrouz et al., arXiv 2501.00663 |
| Editing the MLP at the last relation token (prediction site) and its over-generalisation to other subjects (our same-old-answer damage) | RETS | Liu et al., AAAI 2025, arXiv 2408.15091 |
| Interference of sequential closed-form edits with non-orthogonal keys, growing with the number of edits | Knowledge in Superposition; PRUNE | arXiv 2408.07413; 2405.16821 |
| The score identity E_{y~p}[delta] = 0; linear associative-memory algebra (kernel law); deterministic equivalents | textbook | — |

### 1.2 What the reviews left standing (the only candidates for a contribution)

From the FEC novelty review:
1. FEC-A as a type-resolved truncation (all value-path orders, no query/key and no FFN order) used as an
   autograd-free **write rule**, with the statement that it is the exact gradient of a stop-gradient graph
   (a small step beyond Highway-BP, TinT and Dual Path Attribution). The statement is a unit test here.
2. The **measured comparison of credit rules** for one-read generalisation: truncated credit against the exact
   one-step gradient (H2).
3. Edit transfer across the Turkish case paradigm as the measuring instrument (a morphology-level datum, not the
   claim).

From the DEFTER novelty review:
4. The variance-reduction statistic used as the **read licence** of fast weights, with exact reduction to the
   frozen model where the licence is closed, together with a registered, parameter-free **capacity law** for its
   false reads (H4).
5. A per-write ledger with a **measured gap** between ledger-level retraction and a counterfactual replay (H5).
   The review stated this for a meta-trained in-stream learner; here it is measured on frozen pre-trained bases,
   which is a narrower setting.
6. The test bed: recall at a fixed hard-negative leak budget (H4a).

**Not claimed:** a new learning principle, a forward-only rule, the path-order family, the closed-form error, the
RLS write, the kernel law, the familiarity statistic, the mathematics of the capacity law, or a lock against
self-learning. Before any statement leaves this repository the searches must be repeated with full texts.

## 2. Mechanism under test (engine module)

All names are from `forward_learning.py`; every numbered statement is a test in
`tests/evolution/test_forward_learning_engine.py` or `test_forward_learning_memory.py`.

- **Model.** `CevahirNeuralNetwork` in eval mode, pre-norm, sequential residual, RMSNorm, no MoE, standard
  softmax attention, plain floating-point `ffn.fc2` layers on the CPU (anything else is refused): x_0 = E[tok], u_k = x_{k-1} + attn_k(n1_k(x_{k-1})), x_k = u_k + W_2^k h_k with h_k the fc2 input,
  logits c·tanh(z/c), z = U n_o(x_L).
- **Output error** (`output_error`): delta_t = J_{n_o}(x_L)ᵀ Uᵀ[s ⊙ (p_t − e_y)], s = 1 − tanh²(z/c). Equals
  ∂(−log p_t[y])/∂x_L (≤ 1e-5 in float32, ≤ 1e-10 in float64; with and without the cap; tied and untied output:
  tying is not needed for the formula).
- **Fields** (`error_fields`), with A_k the transpose of block k's value path under the frozen attention
  pattern (`value_path_vjp`; grouped-query attention handled):
  FEC-TOP = delta at the last block (the exact gradient of the last fc2); FEC-0 = delta at every block;
  FEC-1 = delta + Σ_{k>l} A_k delta; FEC-A = Π_{k>l}(I + A_k) delta. FEC-A and FEC-0 equal the exact fc2
  gradients of the graphs built by `stop_gradient_graph` (queries, keys and FFN inputs detached; for FEC-0 also
  values). Controls: HEBB-0 / HEBB-TOP (target embedding only), RAND-0 (random feedback, seeded per edit).
- **Write** (`ForwardLearner.write`): ΔF_l = −μ Σ_s m_l[s](P_l h_l[s])ᵀ / ‖·‖_F over the written blocks;
  P_l = C_l⁻¹ with C_l ← C_l + ξ KᵀK (Woodbury), C0_l = n0·(E[h hᵀ] + ridge) from CEV-11's C0 pool, n0 = 50,
  ξ = 1. μ is the smallest step on the grid {0} ∪ {0.125·2^{i/4}} that meets the install criterion on the read
  sentence, found with forward passes only: `nll` = mean target NLL < 0.05 (the stop rule of CEV-11's fine-tuning
  arms; if not reached, the step of minimum NLL is used when it reproduces the sentence, flagged), `greedy` =
  the argmax reproduces the sentence. μ = 0 when the criterion already holds. A write that cannot be installed
  leaves the state unchanged.
- **Read gate**: nu(h) = 1 − hᵀC⁻¹h / hᵀC0⁻¹h; the fast output is added only where nu ≥ θ. Where no gate opens,
  the logits are bit-identical to the frozen model (also for the positions before the first open gate).
- **Ledger**: float64 terms (r, z, k) per write and block; F = Σ r zᵀ, C = C0 + ξ Σ k kᵀ. `retract` subtracts a
  write's terms and downdates C and P. Exact at the ledger level; **not** a replay (later gain vectors were
  computed with the retracted keys present). Guards: generic RMS ratio sqrt(tr(F E[hhᵀ] Fᵀ)/D) / RMS(native FFN
  output) ≤ 0.25 and a ledger budget, with policies evict / refuse / off.
- **Capacity law** (`predicted_familiarity`, `capacity_knee`): with whitened second moments S_w (stored-key
  population) and S_q (queries), delta = tr[S_w(I/ξ + N S_w/(1+delta))⁻¹], mean nu = 1 − tr[S_q(I + ξN
  S_w/(1+delta))⁻¹]/tr S_q; mixture form with one self-energy per stored-key class (§5.4).

Corrections to the *mechanism* carried over from the methods reviews, reflected in the code and tests (the
required changes to the *experiment* are tracked one by one in §4.1): the score identity is not
a lock (the write is normalised and step-searched; the protection is that only `read` + `write` on tokens
supplied by the caller ever write); the kernel law and the RLS guarantee are exact only for top-block writes; a
write below the top block invalidates the KV cache (cleared by the learner); every arm, BASE included, is
evaluated on the standard forward path (the tape path that returns attention weights differs by up to 1e-5 in
the logits and is used only to compute writes).

**Not implemented from DEFTER** (stated so that nothing is read into the name): the learned key map, the learned
feedback correction, the learned write gate, the uncertainty-scaled regulariser λ_t = λ_w(1 − nu), chains, the
window-wise joint write and the meta-training objective. Keys here are the raw fc2 inputs; the write is the FEC
step with the RLS projector.

## 3. Units, data, blinding

- **Unit:** the world, i.e. an independently pre-trained CEV-11 base. Discovery: worlds 0 and 1
  (`discovery_w{0,1}_K20`). Confirmatory: worlds 1000–1007 (`confirm_w1000..1007_K20`), n = 8. No new
  pre-training. The frozen CEV-11 package is imported and never edited; base weights are read from `$CEV_BASES`.
- **Scope of inference:** new synthetic worlds and initialisations with this tokenizer and prose corpus, a fixed
  relation phrase directly before the object and a closed set of 48 answers. The relation phrase makes
  prediction-site keys both transferable and answer-keyed by construction; a positive result must be stated with
  this limit.
- **Disclosure.** CEV-11's confirmatory records are on disk and its report is public: the world-level means of
  the comparators (FT-L 0.847, FT-ALL 0.931, G = mean(PE6, IE5)) were known when the margins below were chosen.
  For that reason H1 is tested against the stronger of the two (FT-ALL), as the methods review required.
  The comparators are run fresh by this study with CEV-11's frozen selection (`FT-L` layer 0, eps 0.02, lr 1e-3,
  60 steps; `FT-ALL` lr 1e-4, 40 steps) and cross-checked against CEV-11's records of the same edits
  (`verify.py --crosscheck`).
- **Blinding rule.** Everything that can be tuned is fixed on worlds 0 and 1 by `run_select.py` (§6) and enters
  the freeze manifest with the protocol, the study code and the engine module. `run_world.py` refuses
  confirmatory worlds unless the manifest matches.

## 4. Hypotheses

Decision rule for every test (unit = world, n = 8): **supported** if the Holm-adjusted exact sign-flip p ≤ 0.05
against the stated null (and, for superiority tests, the mean reaches the smallest effect of interest);
**refuted** if the two-sided 95% t-interval lies entirely on the wrong side of the margin or SESOI;
**inconclusive** otherwise. A guard fails when the bound of its 95% interval crosses its limit, not on the point
estimate; a supported test with a failed guard is labelled "guard failed". Fewer than six worlds: "not testable".
G = mean(PE6, IE5) per world over **all** edits, a failed install counting 0 in both.

**Primary family (Holm over five tests).**

| | Statistic per world | Test | Guards / controls | Honest prediction |
|---|---|---|---|---|
| **H1** single-read edit | G(FEC-A, `nll` install) − G(FT-ALL) | non-inferiority, margin 0.04 (null: ≤ −0.04; half of the resolved FT-ALL − FT-L gap) | Leak_RD, Leak_C (= RC), Leak_RN of FEC-A not above FT-ALL by more than 0.02; Δbpb ≤ 0.01 (upper 95% bounds) | supported, with low confidence (v2 pilot: 0.985 on 12 edits of one discovery world under the `nll` rule; FT-ALL 0.931 on CEV-11's confirmatory worlds). With world SDs of 0.03–0.04 this test can fail |
| **H2a** routing | G(FEC-A) − G(FEC-TOP), `greedy` install | superiority, SESOI 0.05 | RAND-0 installs ≤ 10% of edits, else "control failed" | **uncertain**: FEC-TOP has no routing and was level with FEC-A in the v1 pilot (0.91 / 0.905) and on discovery world 1 (0.89 / 0.89), but behind it on discovery world 0 (0.66 / 0.92; the smoke's six edits 0.833 / 0.950); FEC v2 expects a difference within ±0.05 |
| **H2b** truncated credit | G(FEC-A) − G(BP★), `greedy` install; BP★ = the best of {BP-1, BP-1b, BP-pred} on discovery | non-inferiority, margin 0.05 | as H2a | supported (pilot: FEC-A level with BP-pred) |
| **H3** stream | Net/write(FEC★) − Net/write(c) at N = 100 for **every** sequential comparator c, mean over the world's streams; Net/write = (retained facts − Dmg_A × never-edited relation-A facts the base knows) / N | intersection–union non-inferiority, margin 0.05 net facts per write against each comparator: p = the largest of the five sign-flip p-values; refuted when the 95% upper bound against any one comparator is below −0.05 | — | **not supported**: the tape-free rules put more of the write on entities sharing the old answer; on discovery world 0 the capped exact-gradient step was ahead of both FEC arms at N = 30 (§9) |
| **H4a** gate | recall_other(gated at θ★) − recall_other(ungated at γ★) on the gate arm, N = 100 | superiority, SESOI 0.10 | hard-negative leak of the gated read: upper 95% bound ≤ 0.05 (point limit 0.02 reported) | uncertain |

FEC★ ∈ {FEC-TOP+RLS, FEC-A+RLS} is the discovery maximum of net facts per write and is fixed in the selection
file. The comparators are {FT-L-seq, BP-1+RLS, RLSEdit, HEBB-TOP+RLS, HEBB-0+RLS}: an RLSEdit-style sequential
rank-one editor, a projector-matched exact-gradient step and the Backward-Lens-style target-embedding rule, not
only naive fine-tuning (the v1 stream claim compared a projector arm with fine-tuning that had none, and the
methods review showed that most of that gap was the projector). **No single "best comparator" is picked on
discovery.** The discovery streams run without step caps (the caps are their product), and without a cap the
exact-gradient arm collapses, so a discovery ranking of the comparators would exclude it by construction. H3
therefore asks for non-inferiority to every comparator; CMP★ (the discovery maximum of net facts per write among
the comparators) is kept in the selection file for the secondary tests only. Sequential FT-L has no projector and
is the naive reference of the set.

**Threshold hypotheses (not in the Holm family; reported as estimates against their limits).**

- **H4b — capacity knee.** Per world, the measured knee N_meas (number of stored top-block keys at which the
  median familiarity of unrelated fact probes crosses θ; θ = θ★ if it is one of 0.3, 0.5, 0.7, else 0.5) against
  the knee predicted before any write (§5.4). If the median never crosses θ within the stream, the ratio of the
  measured to the predicted mean familiarity at the largest load is used. **Supported** if at least 7 of 8 ratios
  lie within ±25% and none outside ±50%; **refuted** if at least 5 of 8 lie outside ±25%; else inconclusive.
  Honest prediction: see §9 — the form of the law as written in the DEFTER proposal is expected to fail; the
  mixture form registered here was within about 10% in level on one discovery base.
- **H5 — retraction against "never seen".** For the selected facts of stream 0 that were recalled before
  retraction (P(first object token) ≥ 0.5): KL(replay ‖ retract) of the next-token distributions at the fact's
  lived frame and one other frame, where *replay* is the same stream without that fact; and the difference in
  retention of the other facts. **Supported** ("replay-equivalent at 0.05 nat") if the upper 95% bound of the
  world-mean KL is ≤ 0.05 nat and the retention difference lies within ±0.02; **refuted** if the lower bound of
  the KL is above 0.05 nat. The gap is reported as a measurement whatever the label. Honest prediction: refuted.

**Secondary, registered, unadjusted** (same three-outcome rule): G(FEC-A) − G(FEC-0) under both install rules
(the v1 form of H2a; the methods review: FEC-0 is low because it writes a misaligned field into the lower blocks,
not because routing is absent, so this contrast would be confirmed even if routing added nothing); H2a under the
`nll` rule; H1 against FT-L with margin 0.05 (the v1 form of H1); H1 under the `greedy` rule; H3 on the composite
S = (R + (PF6 + IF)/2)/2 − Dmg_A against CMP★ (the v1 form of H3; S mixes rates over populations of different
size and is positive for arms whose net is negative) and as a superiority test on net facts per write against
CMP★; the same-old-answer leak (SO family) of FEC-A against BP★ (v2 pilot: the tape-free rules lose selectivity,
not transfer); the capacity knee of H4b with the crossing of the measured mean (§5.4).

### 4.1 Deviations from the source proposals and their reviews (complete list)

The two methods reviews judged the proposals "fixable" on condition of numbered changes. The table says, for
every change to the experiment, whether this protocol follows it. Rows marked *restored* were deviations in the
first draft of this protocol (it tested H1 against FT-L, H2a against FEC-0 and H3 on the composite S) and were
put back by the independent review of 2026-11-14.

| Source | Required | Here |
|---|---|---|
| FEC methods R1 | 42 edits per world from CEV-11's list, paired by `edit_id`, disclosure | followed; the comparators are re-run fresh and cross-checked instead of reused |
| FEC methods R2 | H1 against FT-ALL, margin 0.04, NLL-matched installs | followed (*restored*). **Not followed:** FT on all six fc2 and FT-L on block 6 in Part A are not run |
| FEC methods R3 | H2a = FEC-A − FEC-TOP; H2b one-sided against the best exact one-step variant | followed (*restored*). The install rule of H2 is `greedy`, not NLL-matched (open point §11.1) |
| FEC methods R4 | projector-matched backprop arms: multi-step FT on fc2 and on block 6 with the RLS-projected gradient, same criterion and cap | **partly**: only the one-step arm BP-1+RLS. The multi-step projected arms (FT-fc2+RLS, FT-L6+RLS of FEC v2) are **not implemented**; they would be the strongest comparators of H3 |
| FEC methods R5 | damage on the edited relation, by shared old answer; net facts per write as the primary H3 quantity | followed (*restored*). "Usable capacity" is not computed |
| FEC methods R6 | leak families "same old object" and "shares the first token of the new object"; kernel-law prediction of top-block leak | SO is measured; the first-token family and the kernel-law prediction are **not** |
| FEC methods R7 | μ = 0, no write below a norm, a cap for every arm, capped write = failure | followed; the stream cap is one common cap (§5.2) |
| FEC methods R10 | RAND-0 seeded per edit, G over all edits, three outcomes, guards on interval bounds | followed |
| FEC v2 | path-class kernel (P1–P4), derived mixtures, margin ratios, exploratory paraphrase bases | out of scope (§10) |
| DEFTER methods 1, v2 H1(b) | a like-for-like episodic store over the same keys (KEY-kNN) next to the gated fast weights | **not run**: H4a compares the gate with a global gain only, so it says nothing about fast weights against an episodic store |
| DEFTER v2 arms | INPLACE (In-Place TTT replica), HEBB as a meta-trained arm, kNN-cache, TTT-FT, EPI-SLOT | **not run** (they need meta-training or belong to its H5; §10) |
| DEFTER v2 H2 | knee = median familiarity crossing θ, against the law | followed as worded; the law predicts the mean, so the crossing of the measured mean is recorded too |
| DEFTER v2 H4b | retraction unit = chain, 40 facts per model, 90% bounds | one read per fact (write = chain), 7 facts per world, 95% bounds |

## 5. Experiment

### 5.1 Part A — single edits (H1, H2)

Per world 42 edits: CEV-11's confirmatory item list (`edit_items(world, editable, seed = world + 2,
per_case = 10)`), the first 6 of every lived case in list order. One read of the lived sentence, the object span
as the learned positions, all six fc2 writable (FEC-TOP, HEBB-TOP: the last), no projector (a single write).
Arms: FEC-TOP, FEC-0, FEC-1, FEC-A, HEBB-TOP, HEBB-0, RAND-0, BP-1 (exact fields, global norm), BP-1b (per-block
norm), BP-pred (exact fields at the loss positions) — each under both install criteria — and FT-L, FT-ALL
(CEV-11's code and frozen selection). Step caps from the discovery selection (4 × the discovery median of the
installing step per arm and criterion; RAND-0 takes FEC-0's cap; single edits are independent, so an arm's own
median is not distorted by earlier writes); a capped search that does not install is a failed install. Families: CEV-11's WF, PF6, PF5, IF5, RN, RD, RB, RC, plus **SO** (up to 6 other probe-pool
entities whose old answer equals the edit's old answer, 7 formal frames). Prose bpb on 40 held-out texts for
FEC-A, FEC-TOP and FT-L on the first 12 edits.

### 5.2 Part B — stream (H3)

Per world 2 streams of N = 100 facts of relation A (entities sampled from the editable pool, lived case cycling
through the 7 cases, a random different town), one read each, no replay; evaluation at N = 10, 30, 100. Arms:
FEC-TOP+RLS, FEC-A+RLS, FEC-TOP (no projector; ablation), FEC-TOP+RLS+G (capacity guard on, descriptive),
HEBB-TOP+RLS, HEBB-0+RLS, BP-1+RLS (one projected exact-gradient step), FT-L-seq (CEV-11's FT-L applied fact
after fact) and RLSEdit (CEV-11's ROOT site and value search on the current model, written with the RLS gain
g = Pk/(1/ξ + kᵀPk) under the same prior; if the plain RLS write misses the criterion, larger scales from the
common grid are tried). Same install criterion (`nll`) for all, and **one common step cap** for every arm whose
step is the Frobenius norm of a unit-normalised write (all learner arms): 4 × the median installing step of
FEC-TOP+RLS and FEC-A+RLS over the discovery streams. RLSEdit's step is a multiple of its own value search and
keeps 4 × its own discovery median; FT-L-seq has no step search. The first draft gave every arm 4 × its own
median; that is no cap for an arm that collapses in the uncapped discovery run, because its steps grow with the
damage (§9), and FEC v2 asks for "the same step cap for all".

Metrics: R = retention in the lived frame; PF6, IF = the other formal and the informal frames of all written
facts; **Dmg_A** = share of 120 untouched entities' relation-A facts the base knew whose greedy answer changed
(the edited relation only: the v1 pilot's damage averaged relations A and B and was diluted by half), Dmg_B the
same for relation B, damage by the number of stream facts sharing the old answer, own-entity B/C damage; net
facts per write = (retained − Dmg_A × never-edited relation-A facts the base knows) / N; Δbpb at the last
checkpoint; composite S = (R + (PF6 + IF)/2)/2 − Dmg_A.

### 5.3 Gate (H4a), stream 0

On FEC-TOP+RLS and FEC-A+RLS after the last fact: the gated read at θ ∈ {0, 0.05, 0.1, 0.2, …, 0.9, 0.95} and
the ungated read with a global gain γ ∈ {1, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1} (DEFTER's "no gate" comparator).
recall_other = PF6 of all written facts. **Hard negatives** (up to 240 per type): *rn* entities outside the
stream sharing the last root token with a written entity (relation A, the writer's frame); *rel* the written
entities' relations B and C in the lived frame; *so* entities outside the stream whose answer is the old answer
of a written fact. A leak event is a total-variation distance > 0.10 between the next-token distributions with
the memory on and off at the first object position ("argmax changed" is not used). The gate arm of H4a is FEC★.

### 5.4 Capacity law (H4b), stream 0, top block of FEC-TOP+RLS

Top-block keys do not move under top-block writes, so the familiarity of fixed probe keys can be traced after
every write. Queries: prediction-site keys (first object position) of the stream's untouched probe-pool
entities. Prediction, from the base model before any write: reference entities = the C0 pool and the P3 entities
(every entity outside the editable pool; disjoint from the stream and from the queries), relation-A sentences in
the trained case frames with a seeded counterfactual object. **Registered form: mixture** with two stored-key
classes — the prediction site and the later object positions — with their shares and whitened second moments
S_c, S_q from the unit-normalised prediction-site reference keys, one self-energy per class:
delta_c = ξ tr(S_c R), R = (I + ξN Σ_c share_c S_c/(1+delta_c))⁻¹, mean nu = 1 − tr(S_q R)/tr S_q; N counts
stored keys. The registered measured knee is the crossing of the probes' **median** (the wording of the DEFTER
proposal) while the law predicts their **mean**; the crossing of the measured mean is recorded next to it
(`knees_mean`) and reported as a secondary, so a failure of H4b can be told apart from a skewed distribution.
The reference frames are all trained case frames, the measured probes use the genitive frame only; this
mismatch is part of what H4b tests and is not corrected. The two-population form as written in the DEFTER proposal (one class, C0-pool entities only) is
computed and reported next to it (`predicted_v2`, `knees_v2`).

**Disclosure:** the mixture form and the larger reference set were chosen after the smoke on discovery world 0
showed the two-population form over-predicting the fill by about a factor 2 (§9). This is use of a discovery
world for a design decision and must stay visible in any report.

### 5.5 Retraction (H5), stream 0, FEC-TOP+RLS

Facts at stream indices 5, 20, 35, 50, 65, 80, 95. The state before each of them is snapshotted. After the last
fact: (a) `retract` the fact's write and measure the next-token distributions at its lived frame and one other
formal frame, and the retention of the other facts; (b) restore the snapshot and replay the rest of the stream
without the fact, measure the same. KL(replay ‖ retract), total variation, P(first object token) under both.

## 6. Discovery selection (`run_select.py`, worlds 0 and 1 only)

Step caps (single arms: 4 × the median installing step per arm and criterion; stream: the common cap of §5.2);
BP★ (highest pooled G under `greedy`); FEC★ (highest pooled net facts per write at N = 100; CMP★ likewise, for
secondary tests only); for each gate arm θ★ and γ★ = the setting with the
highest pooled recall_other among those with pooled hard-negative leak ≤ 2% and Δbpb ≤ 0.01 (if none is feasible
the most conservative setting is taken and the arm is reported as infeasible at the budget). The file carries
the hashes of the discovery records, the engine and configuration hashes of their headers, and its own sha256.
The selection is refused at the freeze when the engine module is no longer the one the discovery records were
made with, or when they were not made with the registered configuration. Known limit: the discovery streams run
uncapped, so FEC★ and the gate operating points are chosen under conditions that differ from the capped
confirmatory streams; a wrong FEC★ can only hurt the FEC side of H3 and H4a.

## 7. Statistics

World-level paired differences; exact sign-flip permutation test (256 patterns, smallest one-sided p = 1/256;
with five Holm tests the smallest adjusted p is 5/256 ≈ 0.0195, so a primary test can pass only if at least
seven of the eight worlds agree in sign after the margin); Holm step-down over {H1, H2a, H2b, H3, H4a} at
α = 0.05 (an untested member keeps p = 1); H3 enters the family with the largest of its five p-values
(intersection–union: no correction is needed inside the union, and a comparator without records makes H3
untestable instead of being dropped); two-sided 95% t-intervals (df = 7) with all per-world values listed.
Functions:
`cev11.stats.signflip_pvalue`, `holm_adjusted`, `t_interval` (the frozen CEV-11 package). `analyze.py` refuses
incomplete world files and records whose header does not carry the configuration of the frozen selection or the
engine of the discovery records; it writes into `decisions.json` whether the freeze manifest still matches.
A base folder is accepted only for the world it was pre-trained for (the seed in its world file), so `--base`
cannot put a confirmatory base under a discovery or smoke run. An existing freeze manifest is never overwritten
by a different one.

## 8. Compute (estimate from the smoke timings; not measured end to end)

One thread per job. The smoke took 174 s in the author's run and 215 s in the reviewer's (same machine class,
other light jobs running), so every figure here carries about ±25%. Part A: about 7 s preparation per edit and
1.1–1.5 s per (edit, arm, criterion), i.e. about 30 minutes per world. Part B: about 0.15 s per forward-learning
write, 0.5 s per FT-L write, 1.2 s per RLSEdit write, 5–10 s per evaluation; with the gate sweep (prose bpb at
every setting in discovery) and the retraction replays about 25–30 minutes per world. Confirmatory package:
about 8 core-hours; discovery: about 2. If this must be cut: drop the second stream, then the `greedy` criterion
of the arms that are in no hypothesis.

## 9. Smoke on discovery world 0 (pipeline proof; not protocol data)

`run_world.py --run discovery --world 0 --smoke`: 6 edits and a 30-fact stream with the items, step grids and
install criterion (`greedy`) of the v1 pilots, so that the engine can be compared with the pilot scripts.

Measured on 2026-11-14: 174 s on one thread with the machine's other three cores busy (Part A 87 s: 34 s
preparation, 53 s for 6 edits x 9 arms; Part B 87 s), 76 records; a second run gave the same records apart from
the wall-clock fields. The reviewer's run of the same command measured 215 s (Part A 102 s: 41 s preparation,
61 s arms; Part B 112 s) and reproduced every number of the tables below. Small samples, one base, operating points not selected: none of the numbers below is a
finding.

**Closed forms on a trained base** (6 sentences): output error against autograd 7.5e-7 (relative), FEC-A and
FEC-0 against their stop-gradient graphs 6.7e-7 and 6.0e-7, FEC-TOP against the last fc2 gradient 6.0e-7. Cosine
of the rule's fc2 gradient with backprop, blocks 1..6: FEC-A 0.351, 0.445, 0.505, 0.642, 0.787, 1.000; FEC-0
0.142, 0.226, 0.420, 0.611, 0.751, 1.000 (the pilot's values, to the digit).

**Pilot numbers reproduced** (same items, grids and criterion as `fec_pilot_cev11.py`, `fec_stream_pilot2.py`):

| | v1 pilot | engine (smoke) |
|---|---|---|
| single edits, PE6 / IE5: FEC-A | 1.00 / 0.90 | 1.00 / 0.90 |
| FEC-1 | 1.00 / 0.90 | 1.00 / 0.90 |
| FEC-0 | 0.778 / 0.833 | 0.778 / 0.833 |
| FEC-TOP | 0.833 / 0.833 | 0.833 / 0.833 |
| BP-1 | 0.806 / 0.728 | 0.806 / 0.728 |
| BP-pred | 0.917 / 0.900 | 0.917 / 0.900 |
| RAND-0 installs | 0 of 6 | 0 of 6 |
| stream N = 30, FEC-TOP+RLS: retention / PF6 / IF / damage (A and B pooled) / Δbpb | 1.00 / 0.678 / 0.561 / 0.104 / +0.056 | 1.00 / 0.678 / 0.561 / 0.104 / +0.056 |
| stream N = 30, FEC-A+RLS | 0.867 / 0.628 / 0.529 / 0.054 / +0.008 | 0.867 / 0.628 / 0.529 / 0.054 / +0.008 |

**What the smoke adds, and why it lowers expectations.**

- *Undiluted damage.* On the edited relation alone the same 30-fact stream damages 0.200 (FEC-TOP+RLS), 0.108
  (FEC-A+RLS) and 0.417 (FEC-TOP without projector) of the untouched facts; net facts per write is negative for
  all three (−1.07, −0.25, −3.77). For FEC-TOP+RLS the damage rises with the number of stream facts that share
  the untouched entity's answer: 3 of 56 (none), 12 of 49 (one), 8 of 13 (two). In Part A, FT-L on the same six
  edits scores G = 0.908 against FEC-A 0.950 and FEC-TOP 0.833.
- *Comparators.* The runs of HEBB-TOP+RLS, BP-1+RLS, FT-L-seq (10 facts) and RLSEdit (6 facts) were cut short and
  are not comparable with the 30-fact arms. BP-1+RLS without a step cap collapsed within 10 facts (Dmg_A 0.64),
  as in the v1 pilot; the cap from the discovery selection is what makes it a fair control.
- *Gate* (FEC-TOP+RLS, 30 facts, 180 hard negatives): ungated, recall on other case forms 0.678 with hard-negative
  leak 0.45 (by type: same old answer 0.75, shared root token 0.53, other relations of the written entity 0.07);
  θ = 0.6: 0.678 with leak 0.117 and Dmg_A 0.033; θ = 0.9: 0.333 with leak 0. A global gain of 0.5
  removed every installed fact (retention 0), so with smallest-step installs the gain is not a usable knob. No
  setting on this coarse grid met the 2% budget with recall intact.
- *Capacity law.* At 154 stored keys the mean familiarity of 97 unrelated probes was 0.531. The two-population
  form as written in the DEFTER proposal predicted 0.695 and knees too early by factors 1.83 (θ = 0.3) and 1.72
  (θ = 0.5): outside ±50%. Diagnosis on this base: one self-energy for prediction-site keys and later-position
  keys together fills too fast, and 60 reference entities under-estimate the dimension of the entity-specific
  directions (the prediction falls as the reference set grows). The mixture form of §5.4 with 120 reference
  entities predicted 0.585 and knee ratios 1.32 (θ = 0.3) and 1.21 (θ = 0.5): better, still biased in the same
  direction, and at the edge of ±25%. H4b can fail.
- *Retraction.* Two facts (indices 5, 20), both below the recall precondition under the smallest-step install
  (P = 0.39 and 0.47): KL(replay ‖ retract) was 3.2–3.8 nat and 0.5–0.6 nat. The retracted state gave the
  fact's token probability 8.5e-5 where the replay gave 6.7e-3, and kept 0.931 of the other facts where the replay
  kept 0.966. Ledger retraction is exact arithmetic and, on this evidence, far from "never seen".

**Reviewer probe of the step cap (2026-11-14; discovery world 0, the smoke's 30-fact stream and `greedy`
criterion; one base, not protocol data).** This probe is the reason for the common cap of §5.2 and for testing
H3 against every comparator; it is a use of a discovery world for a design decision and must stay visible.

| arm, cap | installed of 30 | retention | Dmg_A | net facts per write |
|---|---|---|---|---|
| BP-1+RLS, no cap (first step 22.6; steps up to 128) | 15 | 0.133 | 0.983 | −10.03 |
| BP-1+RLS, cap 13.3 (4 × its own median over the first 10 facts) | 24 | 0.667 | 0.200 | −1.40 |
| BP-1+RLS, cap 8.0 (the same for 5.66 and 4.0) | 22 | 0.667 | 0.017 | **+0.49** |
| FEC-TOP+RLS, no cap / cap 8.0 | 30 / 29 | 1.000 / 0.967 | 0.200 / 0.150 | −1.07 / −0.58 |
| FEC-A+RLS, no cap / cap 5.66 | 30 / 29 | 0.867 / 0.933 | 0.108 / 0.108 | −0.25 / −0.19 |

Reading: the exact-gradient step is not a weak comparator that "collapses"; it collapses only when its first
steps are unbounded, and with the step budget of the FEC arms it was the only arm with a positive net on this
stream. Under the first draft's rules it could not have become the H3 comparator (uncapped on discovery) and
would have run the confirmatory streams with a cap of 4 × 8.0 = 32. H3 is expected to fail.

## 10. Out of scope (future work)

- **The meta-trained DEFTER arm.** Slow weights trained through the write rule (key map, feedback correction,
  write gate, uncertainty-scaled regulariser, life streams with never-seen entities) need new pre-training; the
  DEFTER proposal budgets 10.8 core-hours for it. Not part of this study.
- The path-class kernel of FEC v2 (PCK: Möbius decomposition into skip / value / FFN / query-key classes, derived
  credit mixtures). The stop-gradient graphs it needs are in the engine (`stop_gradient_graph`); the kernel
  instrument and its hypotheses are not.
- Pre-training with truncated credit; paraphrase bases with a non-adjacent object position; the Gauss–Newton
  and true-gradient value switches; a KV-cached streaming reader.
- **Comparators the source proposals asked for and this study does not run** (§4.1): multi-step fine-tuning
  with the RLS-projected gradient (FT-fc2+RLS, FT-L6+RLS), FT on all six fc2 and FT-L on block 6 for single
  edits, an episodic store over the same keys (KEY-kNN), the In-Place TTT replica, kNN-cache and TTT-FT. A
  positive H3 or H4a must be reported with this list next to it.

## 11. Open points before the freeze

1. H2's primary install rule: `greedy` (where the family differs) is proposed; the methods review preferred the
   NLL-matched rule for H1. Both are run; the choice of primary must be fixed before the discovery records are
   inspected for H2.
2. H4a's guard bound (0.05 for the upper 95% bound, with the 0.02 point limit reported) is a proposal.
3. n0 = 50 and ξ = 1 are fixed without tuning (the v1 pilot's value). If discovery shows no feasible gate
   setting, the protocol must say whether n0 may be re-chosen on discovery or the arm is reported infeasible.
4. The knee threshold of H4b when θ★ is not one of 0.3, 0.5, 0.7.
5. Whether stream 1 is kept (compute) and whether the gate arm of H4a should be fixed to the top-block arm,
   where the kernel law and the capacity trace are exact.
6. Full-text check of FPO, Backward Lens, Highway-BP, TinT, RLSEdit, In-Place TTT, FWL, MesaNet, FAAST and
   MobiEdit.
7. Whether the multi-step projected fine-tuning arm (FT-fc2+RLS) and the episodic store (KEY-kNN) are added
   before the freeze (§4.1). Without them H3 and H4a are tested against a weaker set than the proposals promised.
8. Whether the discovery streams are run a second time with the caps, so that FEC★ and the gate operating points
   are selected under the conditions of the confirmatory streams (about one more core-hour).
9. The margin of H3 (0.05 net facts per write, from FEC v2) was set before any capped stream was seen; the
   spread of net facts per write across worlds is unknown.

## 12. Commands

```bash
export CEV_BASES=/path/to/living_learning_paradigm_2026_11_13/results/bases
python research/forward_learning_2026_11_13/run_world.py --run discovery --world 0 --smoke   # pipeline proof
python research/forward_learning_2026_11_13/run_world.py --run discovery --world 0
python research/forward_learning_2026_11_13/run_world.py --run discovery --world 1
python research/forward_learning_2026_11_13/verify.py --records --results research/forward_learning_2026_11_13/results/discovery
python research/forward_learning_2026_11_13/run_select.py
# after "Status: FROZEN <date>":
python research/forward_learning_2026_11_13/verify.py --freeze
python research/forward_learning_2026_11_13/run_world.py --run confirm --world 1000        # ... 1007
python research/forward_learning_2026_11_13/analyze.py
python research/forward_learning_2026_11_13/verify.py --rerun --crosscheck
```

The job lists `jobs_discovery.json` and `jobs_confirm.json` hold the same commands as `[{id, cmd, deps, log}]`.
