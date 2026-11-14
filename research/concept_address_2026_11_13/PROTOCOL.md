# Kavram Adresi (KA) — PROTOCOL

Status: FROZEN 2026-11-14

This is a draft protocol. Nothing here has been run on a confirmatory world. The code refuses a confirmatory run
until this status line reads `Status: FROZEN <date>`, the discovery selection exists and reports no failed
blocking gate, and `verify.py --freeze` has written the freeze manifest (§8). Until then every number below is a
design value, not a result.

Türkçe özet. Soru: Bir olguyu (varlık, ilişki) dondurulmuş Cevahir çözücüsüne yazarken hangi iç kod gerçek bir
kavram adresidir? Yazım, varlığın bütün Türkçe hâl biçimlerine ve tokenizer'ın kökle eki tek tokene kaynaştırdığı
kesme işaretsiz yazımlara ulaşmalı; son kök tokenini paylaşan komşulara, önek eşlerine ve varlığın öbür
ilişkilerine sızmamalıdır. Üç birincil hipotez: (H1) dikkatin yazdığı akım, token anahtarından ve tam artık akımdan
daha iyi bir yazma adresidir; (H2) kalibre edilmiş, eniyileyicisiz bir bellek, ardışık ince ayardan iyidir ya da
kötüdür (iki yönlü); (H3) yanıt eşiği ayarsız kalibre kalır ve bellek büyüdükçe yanlış ateşleme artmaz. Birim
dünyadır (8 doğrulayıcı CEV-11 tabanı). Çalışma operasyon noktalarını yalnız keşif dünyalarında (0–1) seçer.
Model küçük (3,4 M parametre), dünya sentetik ve şablonludur; sonuç bu ölçek için geçerli olacaktır.

---------------------------------------------------------------------------------------------------------------

## 0. Scope, sources and what is not here

- Study folder: `research/concept_address_2026_11_13/`. Package `ka/`, scripts `run_world.py`, `run_select.py`,
  `analyze.py`, `verify.py`, job files `jobs_discovery.json`, `jobs_confirm.json`, tests
  `tests/evolution/test_concept_address_*.py`.
- Engine: the frozen CEV-11 study (`research/living_learning_paradigm_2026_11_13/cev11/*`) is imported and never
  edited: world generator, tokenizer utilities, key sites, `search_delta`, `rome_update`, `FFNMemory`,
  `ft_layer`, `ft_all`, the batched exact check, the statistics helpers.
- Bases: CEV-11's discovery bases (worlds 0, 1) and confirmatory bases (worlds 1000–1007), D192/L6, 3,445,824
  parameters each. The weights are local files read through the `CEV_BASES` environment variable; they are not in
  the repository.
- Source design: the research programme of 2026-11-13 (flagship AWA-3, second study KA), which merged the
  "living-memory cortex" and "concept geometry, Part A" proposals. This protocol is that design made runnable.
  §10 lists every place where the programme text had to be corrected or narrowed, with the reason.
- Not in this protocol (deferred, not tested, not claimed): joint writes at N = 52 and the ridge-peak law (the
  programme's S3), linear memories (RLS, HEBB), raw-key scope rules (RAWE), autonomous writing from running text,
  aliases, a pretrained in-network cortex (Part B). They need their own pilots.

## 1. Question and claims

A fact about (entity, relation) is written from one inflected sentence. It must reach every Turkish case form of
the entity and its tokenizer-fused informal spellings, but not root neighbours, prefix partners or the entity's
other relations. Which derived code inside the frozen decoder separates these, and does a memory built on it
beat sequential fine-tuning?

Known and not claimed: null-slot reads (pointer sentinel, softmax-1, attention sinks), GRACE's radius, kNN-LM,
RetoMaton, PLDA and score calibration, ROME's closed form and kernel, learned edit keys, the residual stream as a
sum of sublayer writes, deterministic annealing, lifelong-editing benchmarks with sequential fine-tuning.
KA-H2 is a decision for this project, not a novelty claim. The evaluation axis (Turkish case paradigm × fused
spellings × root neighbours) is CEV-11's.

Claimed only if the confirmatory data support it, and only at this scale:
(a) the attention-written stream at the subject's last name token is an entity-level rather than token-anchored
write address (KA-H1); (b) a tuning-free calibrated scope rule for a fact memory (two-covariance likelihood ratio
plus the Bayes prior P_t/N) stays calibrated and gallery-size invariant (KA-H3).

## 2. Codes

Pre-norm block k (dropout 0, eval mode): g_k = Attn_k(norm1(h_k)), f_k = FFN_k(norm2(h_k + g_k)),
h_{k+1} = h_k + g_k + f_k, h_0 = e(x). At block l and position t (`ka/codes.py`, `Tap`):

    a_pre  = h_l + g_l = e + Σ_{k<l} f_k + Σ_{k≤l} g_k      input of layers[l].norm2          ("pre")
    norm2(a_pre)                                              the FFN input                     ("res")
    native key                                                input of layers[l].ffn.fc2        ("key")
    c      = Σ_{k≤l} g_k                                      the attention-written stream      ("att")
    a_post = h_{l+1} = a_pre + f_l                            output of layers[l]               ("post")

A write added to f_l's output can be gated only by a code computed before f_l (pre, res, key, att), never by
a_post. The identities are unit-tested at every block.

Sites: STD = last token overlapping the bare name (CEV-11's key site); PRED = last prefix token.

## 3. Part A: one fact through five mid-stack addresses (KA-H1, S2)

### 3.1 The shared write

For each TEST edit and each step ∈ {0.5, 1, 2}, one residual δ comes from CEV-11's frozen `search_delta` at block
2's STD site (lived prefix plus 4 prose-prefixed copies, cap 8 × median ‖fc2 output‖, at most 120 steps, KL anchor
= relation B in the lived frame). The same δ is installed through five addresses, each alone, then removed.

| Address | Mechanism | Code u | Configurations |
|---|---|---|---|
| A-TOK | frozen `rome_update`: W' = W + δ rᵀ, r = C0⁻¹k̄ / (k̄ᵀC0⁻¹k̄), k̄ prefix-averaged | native key | 3 steps |
| A-TOKG | radius gate | native key | 3 steps × 3 q |
| A-RES | radius gate | norm2(a_pre) | 3 steps × 3 q |
| A-CTX | radius gate | c | 3 steps × 3 q |
| A-SUP | radius gate (supervised ceiling, not in H1) | Pᵀ(c − μ), LDA on c0 entities, ≤ 64 dims | 3 steps × 3 q |

Kernel law (A-TOK): every position x' moves by Δo(x') = δ·τ(x'), τ(x') = k(x')ᵀC0⁻¹k̄ / (k̄ᵀC0⁻¹k̄).

Radius gate: δ is added to block 2's FFN output at every position with cos(u(x'), u(x̄)) ≥ θ, where x̄ is the
lived prompt's STD site. A gate reads a code computed before f_2, so for a given token sequence whether it fires
at a position is the same before and after the write.

In-ball indicator. A gate is live at every position of the pass that scores a probe, the object positions
included (§3.3). The pre-edit indicator is therefore computed, in the base state, on that whole sequence: prefix
plus teacher-forced object tokens. Two flags are recorded per probe: "in the ball" (the gate fires somewhere in
the scored sequence: this is where the write lands) and "in the ball on the prefix" (it fires at the name or at
another prefix position). S2 uses the first flag. The second flag, the number of probes that fire only at object
positions, and the S2 statistics recomputed with the second flag are reported next to it (§10.10).

Thresholds at matched false-match rates. For each code, θ_u(q) is the (1 − q)-quantile of cos(u(x'), u(x̄)) over
(key, position) pairs of different c0 entities: keys = the STD sites of the 6 inflected formal relation-A frames
of every c0 entity; positions = all non-BOS token positions of 2,000 c0 prefixes (seed w+15) drawn from the
60 entities × 3 relations × 13 variants × 3 contexts. q ∈ {0.2%, 1%, 5%}. Computed per world, from c0 only,
before any write. A shared cosine grid would favour c, whose root-neighbour cosines are far lower than a_pre's.

Temperature = radius: two points on the √d_c sphere with cosine κ merge below T_c = d_c(1 − κ) (+ 2σ²), so a
temperature T is the radius θ = 1 − T/d_c. Entity e has a window iff max(T_c(FUSED_e), T_c(CTX_e)) <
min_{m ∈ RN(e)} T_c(e, m). CAM and the window share of every gate code are recorded per world before any write
(descriptive).

Reference arms on the same edits: FT-L at block 2, and FT-L and FT-ALL at the configurations CEV-11 selected on
its discovery worlds (read from the frozen study's `results/selection.json`; never re-tuned here).

### 3.2 TEST edits

16 edits per world (seed w+21), relation A, from the eligible entities: in CEV-11's editable list, at least one
FUSED informal case, at least one root neighbour in the probe partition. Lived cases rotate over the six inflected
cases (3, 3, 3, 3, 2, 2); the lived case always leaves at least one FUSED case to test. The new object differs
from the old one. A world is valid for Part A with at least 12 computable edits.

### 3.3 Probes and scoring

Per edit, built in the base state: WF (lived frame), PF6 (the other six formal cases), FUSED and IDENT (informal
forms of the other inflected cases, by CEV-11's classifier), CTX (lived case after P1 and P2); leak families RN
(≤ 8 entities sharing the last root token × 7 formal frames), RD (10 random entities × 7), C (relations B and C of
the edited entity × 7), PX (prefix partner × 7). Scoring is CEV-11's batched exact multi-token check ('.'
included): one teacher-forced pass over the prefix and the expected object tokens (the final '.' is predicted,
never fed). A leak probe counts as changed when the base model's own answer is no longer produced; probes whose
base answer is not a '.'-terminated object are dropped and counted. On 8 probes per edit the batched check is
compared with true greedy decoding (KV cache on) under an installed gate.

Per (edit, configuration): E (WF), Y (share of FUSED forms producing the new object), X = Leak_RN − Leak_RD,
R = Leak_RD, C, PE6. World-level cell values are means over the world's TEST edits (intention to treat: an edit
without probes of a family contributes 0).

### 3.4 Frontier statistic and frozen mixtures

    F*(a) = max_{w ≥ 0, Σw ≤ 1} Σ_k w_k Y_k   s.t.  Σ w_k X_k ≤ 0.10,  Σ w_k R_k ≤ 0.01,  Σ w_k E_k ≥ 0.80

over the address's configurations; infeasible ⇒ F* := 0. Solved exactly by enumerating the basic feasible
solutions of the linear programme. F*_CEV adds Σ w_k C_k ≤ 0.05 and is co-reported (subject-site writes share δ's
leak to other relations, so F*_CEV ≈ 0 is expected for all of them).

Frozen mixture (the primary statistic): each address's w is solved once on the pooled discovery TEST edits of
worlds 0 and 1 and written to `results/selection.json`. On a confirmatory world F_frz(a) = Σ_k w_k Y_k. An address
infeasible on discovery data is frozen at the empty write. The in-sample F* per world is secondary.

## 4. The output-side memory (KA-H2, KA-H3, S1)

### 4.1 Address

Code a_t at the layer l* fixed on discovery (§8): PRED-RES = a_post^(l*), PRED-ATT = c^(l*).
Statistics from the c0 partition only (never written, never probed): the PRED codes of 60 entities × 3 relations ×
13 variants × 3 contexts = 7,020 codes. ẑ = n(C^{−1/2}(a − μ)), C with ridge 10⁻³·tr(C)/D, n(v) = v/‖v‖.

Two-covariance PLDA on ẑ (concept = (entity, relation)): ẑ = m + y_c + ε, within-class covariance shrunk with
s ∈ {0.01, 0.03, 0.1, 0.3} chosen by cross-fitted c0 log-loss (5 entity folds). With TᵀΣ_wT = I,
TᵀΣ_bT = diag ψ, u = Tᵀ(ẑ − m):

    LLR(q,k) = Σ_i [ ½ log((ψ_i+1)²/(2ψ_i+1)) − ψ_i²(u_qi² + u_ki²)/(2(2ψ_i+1)(ψ_i+1)) + ψ_i u_qi u_ki/(2ψ_i+1) ]

Calibration: s = a·LLR + c by logistic regression on the cross-fitted c0 trials (enrolment = one formal inflected
form per concept; trials = every other code of the fold), c shifted by −logit of the trial target rate. All of
this is fitted per world on that world's own c0: nothing is tuned on written facts.

### 4.2 Read with a null slot

Prior P_t/N per stored episode and 1 − P_t for "no episode", P_t = 0.5 fixed in advance:

    ρ_j(q) = e^{s(q,k_j)} / ( e^{θ_N} + Σ_l e^{s(q,k_l)} ),   θ_N = log N − logit P_t,   ρ_0 = 1 − Σ_j ρ_j

With unnormalised scores and no null this is softmax attention exactly (unit-tested against the repository's
attention); with a → ∞ at fixed θ/a it is nearest-key-within-radius.

### 4.3 Values, continuation, injection

Values are base-relative and analytic. For each object token y at its teacher-forced position with final residual
x: if the base margin z_y − max_{v≠y} z_v ≥ m* = 6, r = 0; otherwise û = ∇_x[z_y − z_ŷ]/‖·‖ and
r = min(γ*, c_cap‖x‖)·û with γ* the smallest step reaching m* (bisection), c_cap = 1. A cap hit is recorded.

Multi-token objects: after an anchor at t', the value of object position k applies at t = t' + k only if the k
tokens after t' equal the episode's first k object tokens:

    w_j^(k)(t) = ρ_j(q_{t'}) M_j / ( ρ_0(q_{t'}) Π_{u=t'}^{t−1} p_base(x_{u+1}) + Σ_l ρ_l(q_{t'}) M_l )

Injection before `output_norm`: x_t ← x_t + Σ_j ρ_j(q_t) r_j^(0) + Σ_{k≥1} Σ_j w_j^(k)(t) r_j^(k).

### 4.4 Ledger and exact reduction

Episodes live outside the module tree, keyed by fact id, summed in id order. insert / correct / retract are exact:
"write A, write B, retract A" equals "write B" bitwise; correct = retract + insert. With an empty ledger no add is
executed and the logits are bit-identical to the base model, with and without the KV cache; `state_dict` and the
model hash never change. No block's keys or values depend on the memory, so writes never clear the KV cache.
All of these are unit tests.

Evaluation uses a cached evaluator (one base forward per probe, then the same `compose` function the hooked
memory runs). It skips the edited readout only where a Lipschitz bound proves the argmax cannot change. On 5% of
the probes per world the hooked float32 model and true greedy decoding re-score the same store.

### 4.5 Laws computed from c0 before any write (S-laws)

- L-Markov (stated, not tested): for a calibrated log-LR E[e^s | H0] = 1, so the expected number of episodes with
  s > θ_N per unrelated query is ≤ N e^{−θ_N} = P_t/(1 − P_t) for every N. At P_t = 0.5 this is 1: N-invariant
  but vacuous at the Leak_U ≤ 1% scale. It guarantees no small leak.
- L-Gauss (false firing): under s | H0 ~ N(−m, 2m), the expected count is N·Φ̄((θ_N + m)/√(2m)), with m the
  negative mean of the cross-fitted c0 non-target scores. Count rule, outside Holm: "holds" iff on at least 6 of
  8 worlds the observed share of U probes with any score above θ_N is ≤ the law + 1 pt at every checkpoint.
- L-gen (generalisation, descriptive): G_f(N) ≈ Φ̄((θ_N − m_f)/√(2m_f)) with m_f the c0 target-score mean of the
  form class.
- L-kernel and L-radius (§3.1) are algebraic identities, unit-tested, not data claims.

## 5. Data, partitions, streams

- Partitions per world: c0 = CEV-11's 60 calibration entities; U = 60 probe entities never written (seed w+14);
  stream pool = the other 220 probe entities.
- Stream L (seed w+12): inserts over relations A/B/C, one fact per concept, N = 400; lived case uniform over the
  six inflected cases; checkpoints 25, 50, 100, 200, 400. Checkpoints are prefixes of one order.
- Battery per live fact: lived form; two other formal cases; one FUSED informal form if the entity has one (G_F is
  over facts with a FUSED form); the lived case after P1 or P2. Negatives, scored against the base model's own
  answer: U (60 × 3 relations × formal and informal inflected: Leak_U), REL (other relations of written entities:
  Leak_C), LTOK (unwritten concepts sharing the last root token: Leak_RN), PX (prefix partner), OBJ (unwritten
  concept with the same old object), OBJ1 (U probes whose base object starts with the first token of a stored new
  object). A probe counts at checkpoint N iff its fact index is < N; U probes always count.
- FT streams: the first 100 facts of L in the same order, checkpoints 25, 50, 100. FT-L and FT-ALL at CEV-11's
  selected configurations, applied to the current weights fact after fact; the KL anchor is the model before that
  fact on the next relation of the same entity (A→B→C→A) in the lived frame. Right after each write the exact
  check is run on the lived prompt of the fact just written; the share of the facts written so far that pass is
  recorded at every checkpoint (`newest_efficacy`, used by G-KA1c, §8).
- Per-fact outcomes: every rule-read record of the memory and every FT record carries, for each live fact with a
  FUSED form, whether that form produces the new object (`fused_facts`). G_F is their mean; they are kept so that
  intervals over facts can be computed from the records (§7).
- Drift: bits per byte on CEV-11's held-out prose, for the base, for the hooked memory at N = 100 and for each FT
  arm at N = 100.
- Seeds: w+12 stream, w+14 partitions and battery, w+15 threshold positions, w+16 calibration folds, w+21 Part A
  edits, w+22 Part A neighbours.

## 6. Hypotheses

Unit: one confirmatory base (worlds 1000–1007, n = 8), each having passed CEV-11's gates. Validity per world:
streams need ≥ 15 live facts with a FUSED form at N = 100; Part A needs ≥ 12 of 16 edits. At least 6 valid worlds
are required, otherwise "inconclusive (insufficient data)".

### Primary family (Holm, α = 0.05; m = 3, or m = 2 if the discovery screen removes KA-H1)

- **KA-H1 (address).** d_w = min over b ∈ {A-TOK, A-TOKG, A-RES} of F_frz(A-CTX) − F_frz(b). SESOI 15 points.
  A-TOKG and A-RES differ from A-CTX only in the code; A-TOK is the standard editor.
  Guards: (i) A-CTX's best-efficacy configuration ≥ A-TOK's − 5 pts, else "address not editable"; (ii) PE6 at the
  frozen mixtures non-inferior to A-TOK (lower 95% bound ≥ −5 pts), else "formal transfer cost"; (iii) locality:
  the upper 95% bound of mean_w X(A-CTX) ≤ 0.10 + 0.03 and of mean_w R(A-CTX) ≤ 0.01 + 0.005 at the frozen
  mixture. When guard (iii) fails, the label is "not supported (locality cost)" if a lower 95% bound is also
  above its frontier constraint (X > 0.10 or R > 0.01: the data show the budget exceeded) and "not supported
  (locality not established)" otherwise (the interval is too wide to tell). Either way KA-H1 is not supported.
  Differences to each baseline are reported. The power of guard (iii) is an open point (§10.2).
- **KA-H2 (memory versus sequential fine-tuning).** d_w = G_F(PRED-RES at the rule, N = 100) − max{G_F(FT-L
  stream), G_F(FT-ALL stream)} on the same 100 facts. SESOI 10 points, two-sided: "memory better", "FT better",
  "equivalent" (90% interval within ±10), else inconclusive. Guards for the arm declared better: the upper 95%
  bound of its structured-leak difference (mean of Leak_C, Leak_RN, PX, OBJ, OBJ1) ≤ +2 pts; its Leak_U ≤ 1%;
  the upper 95% bound of its Δ bits-per-byte ≤ 0.01. A failed guard gives "not supported (… guard failed)".
  The answer holds at this scale only (N = 100, D192/L6).
- **KA-H3 (tuning-free calibrated scope; both parts).** (a) ECE of Σ_j ρ_j as the probability that a probe
  instantiates a stored episode, whole battery, N = 100: mean ≤ 0.05. (b) Leak_U(N = 400) − Leak_U(N = 25) ≤ 1 pt
  under the rule. Manipulation check: with the threshold fixed at θ_25 the same growth is ≥ 3 pts on average;
  otherwise (b) is uninformative and KA-H3 cannot be "supported". p = max of the two one-sided t-tests.
  Refuted if the lower 95% bound of the ECE is > 0.05 or of the growth is > 1 pt.

### Secondary family (Holm, α = 0.05; {S1a, S1b, S2}, or {S2} if PRED-ATT fails its screen)

- **S1a.** d_w = G_F@2(PRED-ATT) − G_F@2(PRED-RES), two-sided, SESOI 10 points, where @2 is the matched point
  Leak_U = 2% on the frontier of fixed thresholds at N = 100 (thresholds at the 0.6…0.999 quantiles of the U
  probes' largest score; linear interpolation).
- **S1b.** d_w = PX(PRED-RES) − PX(PRED-ATT) at the same matched point, one-sided, SESOI 10 points.
- **S2 (prediction before writing).** Intersection–union of three threshold claims, p = the largest of the three
  one-sided t-test p-values: (i) the smaller of two item-level AUCs at the rule, N = 100 (Σ_j ρ_j predicting which
  negatives change; ρ_own predicting which fused forms succeed), mean > 0.85; an AUC with an empty class is
  undefined and the other one is used; (ii) the smaller of two Pearson correlations over the world's 36
  radius-gate cells (pre-edit in-ball share of FUSED probes vs observed transfer; in-ball share of RN probes vs
  observed RN leak), Fisher-z, mean > atanh 0.8; (iii) the share of gate-matched FUSED probes that produce the
  new object over A-CTX's cells, mean > 0.5. "In the ball" and "gate-matched" mean that the gate fires somewhere
  in the scored sequence (§3.1). If KA-H1 is not confirmatory, S2 is (i) alone. Refuted if any part's upper 95%
  bound is below its threshold.

### Manipulation and consistency checks (failure = audit before interpreting anything)

Three pairs of scorers must agree on the same probes: the cached evaluator and the hooked float32 model; the
hooked model and true greedy decoding (both on the 5% sample of the stream battery of each world); the batched
exact check and true greedy decoding under an installed gate (Part A, 8 probes per edit). For each pair the
share of disagreements, pooled over the analysed worlds, must be ≤ 1%. A stream pair without any sample fails;
the Part A pair is skipped when Part A is not run. The base model hash must be unchanged after the checks on
every world (the runner also stops a world whose base hash changed at the end of a run).

The rule is enforced, not only reported. On the discovery worlds a failure blocks the freeze (§8). On the
confirmatory worlds `analyze.py` still writes decisions.json and tables.md, but every label reads "audit required
(consistency check failed)", the statistical label is kept as `label_before_audit`, and the exit code is 2.

A-TOK is CEV-11's STD write by construction (it calls the frozen `compute_write`); a unit test on a tiny model
compares the installed matrix and the logits bit for bit.

### Descriptive

NONULL (no null slot) at N ∈ {25, 100, 400}; in-sample F* and F*_CEV; the FT references' F* on the TEST edits;
CAM and window shares; retention of facts 1–25; cap-hit shares; `lived_fire_positions` (how many positions of the
lived prompt a gate fires at); the number of probes whose gate fires only at object positions and the S2
statistics under the prefix flag (§3.1); hierarchical-bootstrap intervals of the KA-H1 and KA-H2 contrasts (§7);
the laws of §4.5.

## 7. Statistics and labels

- Exact sign-flip permutation test over all 2^n sign patterns of (d_w − SESOI) (CEV-11's frozen
  `signflip_pvalue`), Holm step-down within each family, Student-t intervals. An untestable member enters Holm
  with p = 1, so the family size stays at its preregistered value.
- One-sided contrasts (KA-H1, S1b): **supported** iff Holm rejects; **below SESOI** iff the 95% interval lies
  inside (0, SESOI); **refuted** iff the upper 95% bound is < SESOI and the lower bound ≤ 0; else inconclusive.
- Two-sided contrasts (KA-H2, S1a): p₊ tests mean > +SESOI, p₋ tests mean < −SESOI, the family p is
  2·min(p₊, p₋). With n valid worlds the smallest two-sided p is 2^(1−n), so n = 6 can never pass the first Holm
  level of a family of three; n ≥ 7 is needed.
- Threshold claims (KA-H3, S2): one-sided t-test; refuted iff the 95% interval lies wholly on the wrong side. A
  constant sample has no variance: p = 0 when the constant is strictly on the claimed side, otherwise 1.
- Hierarchical bootstrap, descriptive only: worlds are resampled, then the TEST edits of each drawn world
  (KA-H1: min over the baselines of the difference of frozen-mixture FUSED transfer) or its facts with a FUSED
  form (KA-H2: G_F of the memory minus the larger G_F of the two FT streams, recomputed in every resample);
  10,000 resamples, seed 0, percentile 95% interval, the random draws of CEV-11's frozen
  `hierarchical_bootstrap`. It enters no decision and no label.
- Units. Every confirmatory world 1000–1007 must be accounted for before the analysis runs: complete (DONE.json
  with every phase the selection requires, §8) or excluded by the driver because its base failed CEV-11's gates
  (EXCLUDED.json). A world whose job did not finish is not an invalid unit: `analyze.py` refuses to run on the
  others without it. Validity (§6) is then judged among the complete worlds.
- Per-world values are always shown. `analyze.py` writes `decisions.json` and `tables.md`; `verify.py --rerun`
  recomputes them from the records.

## 8. Discovery selection, gates and blinding

Operating points are chosen on discovery worlds 0 and 1 only (`run_select.py` → `results/selection.json`, whose
body is hashed):

1. l*: the candidate in {2, 3, 4, 5} with the smallest mean cross-fitted c0 calibration log-loss of PRED-RES
   (write-free). Fallback: if its discovery G_F at the rule (N = 100) is more than 2 pts below the best
   candidate's, the best candidate is used. A candidate layer without its record on a discovery world is an
   error, not a G_F of zero.
2. G-KA1b: PRED-ATT enters S1a/S1b only if its c0 log-loss at l* is ≤ 2 × PRED-RES's.
3. The frozen mixture of every address (§3.4).
4. G-KA1a: KA-H1 enters the confirmatory family only if the mean over the two discovery worlds of the in-sample
   min_b [F*(A-CTX) − F*(b)] is ≥ 0.10 and the transfer given a gate match is ≥ 0.5 on at least one world.
   Otherwise KA-H1 is reported as a discovery negative, Part A is not run on the confirmatory worlds and the
   primary family is {KA-H2, KA-H3}.
5. G-KA1c: both FT streams run end to end on world 0, and for each arm the lived-form efficacy on the newest
   fact is ≥ 0.9 at N = 100: of the 100 facts, the share produced on their lived prompt right after their own
   write (§5). The discovery sign of KA-H2 is recorded as its predicted direction (the test is two-sided either
   way).
6. The consistency checks of §6 on the two discovery worlds (pooled).
7. For every frozen mixture, the outlook of the KA-H1 locality guard: the pooled discovery mean of X and of R at
   the mixture, their per-edit standard deviation, and the probability that guard (iii) of KA-H1 would pass on 8
   worlds of 16 edits whose true leak equals the discovery mean and which differ only by edit sampling. Real
   worlds also differ from each other, so the probability is an upper estimate. It is information for the freeze
   decision (§10.2), not a gate.

Blocking gates. G-KA1c and the consistency checks are go/no-go: when one fails, `run_select.py` still writes the
selection (exit code 2), and `verify.py --freeze`, `verify.py --preconditions` and therefore every confirmatory
run refuse until the cause is found and the discovery jobs are run again. G-KA1a and G-KA1b are screens: they
change the families, they do not block.

Never selected: m* = 6, c_cap = 1, P_t = 0.5, the continuation rule, the step and quantile grids, the SESOIs, the
FT configurations (CEV-11's).

Blinding. No KA process runs an edit, a memory or a fine-tuning on worlds 1000–1007, or reads CEV-11's
confirmatory outputs, before (1) CEV-11's confirmatory analysis is recorded by hash in its AMENDMENTS.md and
(2) KA's own freeze manifest exists and checks out. `run_world.py --run confirm` enforces both through
`verify.py --preconditions`. Condition (1) is met (CEV-11 AMENDMENTS.md, 2026-11-13 14:16 UTC entry); condition
(2) is not: this protocol is not frozen.

Freeze procedure: run the discovery jobs; inspect the selection; resolve the open points of §10; set the status
line to `Status: FROZEN <date>`; run `verify.py --freeze`; commit the manifest; only then run the confirmatory
jobs. Deviations after the freeze go to an `AMENDMENTS.md` in this folder.

What the freeze covers. The manifest lists PROTOCOL.md, every Python file of this folder, its JSON job lists,
`results/selection.json`, the KA test files and the frozen CEV-11 engine this study imports. It does not list
README.md and AMENDMENTS.md: the README has to follow the state of the study (frozen, running, results), and
AMENDMENTS.md is where the deviations go. Logs, results and files that are not code are outside it as well.
Writing either document after the freeze therefore does not block a confirmatory run; a changed protocol, a
changed or new Python file, a changed job list, selection or test does. The KA tests do not read the freeze
state of this folder (they work on a copy that they first put into the NOT FROZEN state), so they give the same
outcome before and after the freeze and never have to be edited once they are in the manifest.

The manifest is written once. `verify.py --freeze` does not overwrite an existing manifest; `--refreeze`
replaces it, and only while no confirmatory record exists under `results/confirm`. Once one exists the manifest
can be neither written nor replaced: a correction then is a deviation, recorded in AMENDMENTS.md, and
`verify.py --rerun` reports any analysis code that differs from the frozen code.

Complete worlds. A world directory counts only when it holds `DONE.json`, which the runner writes when every
phase the protocol requires of that world is complete in the directory: all four phases (Part A, episodic
family, FT streams, hooked checks) on a discovery world, and on a confirmatory world the same without Part A if
KA-H1 was screened out. DONE.json describes the directory, not the last invocation: it lists every complete
phase and hashes every record file present, so a run restricted with `--phases` neither turns a partial world
into a finished one nor shortens the list of a finished world. The selection, the analysis and `verify.py
--records` check the phases against what the protocol requires (not against what the record declares), refuse
record files that DONE.json does not hash, and refuse records of another format version.

## 9. Compute and jobs

One thread per process (`OMP_NUM_THREADS=1`). `jobs_discovery.json`: two discovery worlds, record validation,
selection. `jobs_confirm.json`: preconditions, eight worlds, record validation, analysis, recomputation. Commands
run from the repository root; `CEV_BASES` points at the folder holding the bases. Records go to
`results/<run>/w<world>/` and a restarted job resumes (Part A after the last complete edit; the episodic family,
each FT arm and the checks are written atomically). Wall-clock times are kept in DONE.json only, so every record
file is a deterministic function of the base, the configuration and the code.

Measured on discovery world 0 with the reduced `--smoke` configuration (2 TEST edits × all 42 configurations, a
stream of 120 with checkpoints 25/50/100, FT streams of 50, two candidate layers, half of c0; not protocol data),
one core, 2026-11-13, record format ka/2: 129 s wall in total — setup 20 s, Part A 48 s (≈ 16 s per edit after
the first, which also computes the block's C0), episodic family 32 s (4 addresses, 8 s of it for the battery), FT
streams 23 s, hooked checks 5 s. Two runs gave byte-identical setup, Part A, episodic, FT-stream and check
records. The cached evaluator, the hooked model and true greedy decoding agreed on every sampled probe. The full
per-world cost has not been measured; scaling the smoke phases gives roughly 15–25 min per discovery world on one
core (the programme's estimate was ≈ 33 min, hard cap 7.0 core-h for the study).

## 10. Corrections and narrowings relative to the programme text (to be accepted or changed before the freeze)

1. TEST edits. The programme asks for "16 eligible entities, 4 per inflected lived case"; six inflected cases × 4
   is 24. Here: 16 edits, lived cases rotating over the six cases (3, 3, 3, 3, 2, 2).
2. Locality guard of KA-H1. The programme compares A-CTX's leaks with each baseline's at the frozen mixtures. A
   baseline that is infeasible on discovery data is frozen at the empty write (X = R = 0), so that guard would
   compare A-CTX with "no write" and fail whenever A-CTX uses the leak budget the frontier allows. Here the guard
   is absolute (constraint + margin) and the differences to each baseline are reported.

   Open weakness of this replacement, to be settled before the freeze. A frontier optimum usually sits on a
   binding constraint, so the frozen mixture's expected X is about 0.10 or its expected R about 0.01, and the
   margins (0.03 and 0.005) are small against the spread of a mean over 8 worlds of 16 edits. As an indication of
   that spread: across CEV-11's 8 confirmatory worlds (70 edits per world, its `results/tables/edit_metrics.csv`)
   the standard deviation of Leak_RN − Leak_RD was 0.034 for its ROOT arm and 0.054 for its STD arm, and that of
   Leak_RD 0.0020 and 0.0056; scaled to 16 edits (× √(70/16) = 2.09) this is 0.071 and 0.113, and 0.0042 and
   0.0117. With a true mean exactly on the constraint the guard passes with probability 0.18 at an X spread of 0.071
   (0.57 at 0.034) and 0.18 at an R spread of 0.0117 (0.82 at 0.0042), by `ka.stats.upper_bound_pass_probability`
   (checked against simulation). A mixture that keeps to its budget would therefore be "not supported" in most
   runs, which is the failure the programme named when it rejected per-world re-checks.
   What this version does about it: the label separates a locality cost that the data show from a locality
   that the interval cannot establish (§6), and the selection records the guard's outlook for every frozen
   mixture from the discovery edits (§8, item 7). What it does not do is restore the power. Ways to do that:
   solve the frozen mixtures under constraints tightened by the half-width of the interval; widen the margins
   to what 8 worlds of 16 edits can resolve; use more TEST edits per world; or turn the burden around (fail only
   when the data show the budget exceeded). Each changes a preregistered quantity, so the choice is the
   owner's and none is applied here.
3. Frontier solver. The programme says "brute force over configuration pairs". With three constraints and the
   simplex bound the optimum can need three or four configurations (a unit test shows an instance where no pair
   is feasible and the optimum is 0.85). Here: exact enumeration of basic feasible solutions.
4. KL anchor in FT streams. CEV-11's anchor is relation B of the edited entity; for a stream fact about relation B
   that is the fact itself. Here: the next relation of the same entity.
5. Radius-gate key. The programme calls A-TOKG "the same key as A-TOK". A-TOK's key is prefix-averaged over 11
   prompts; a radius gate is keyed at the lived prompt's STD site (x̄), as the programme also says. Here every
   gate, A-TOKG included, is keyed at x̄; A-TOK keeps CEV-11's averaged key. The A-TOK vs A-TOKG contrast therefore
   changes the gate form and the key averaging; A-TOKG vs A-CTX changes only the code.
6. Secondary family. S3(i) (joint writes) and S3(ii) (ridge peak) are deferred, so the family is {S1a, S1b, S2}.
7. Leak_RD in the streams is Leak_U (never-touched entities); there is no separate random-entity class.
8. Cached evaluator. The pilots skipped the edited readout below a relative injection of 1e-7; here the skip rule
   is a Lipschitz bound that proves the argmax unchanged.
9. Open point, seen in the smoke run (2 edits, not protocol data): a radius gate on the attention stream can fire
   at several later positions of the edited entity's own sentence, not only at the name. The thresholds are
   calibrated on positions of other entities, which does not constrain this. The records carry
   `lived_fire_positions` so that discovery can show how often it happens. Whether the gate should be restricted
   (for example to the first match in a sequence) is a design decision for the owner before the freeze; it would
   change the mechanism, so it is not changed here.
10. In-ball indicator. The programme defines the pre-edit in-ball indicator on the probe prefix. Scoring is
    teacher-forced over prefix plus object tokens with the gate live at every position, so a gate can fire at an
    object position only; the write then lands although a prefix indicator says "not in the ball". Seen in the
    smoke run on discovery world 0 (2 edits, not protocol data): of 639 probe rows per (address, q), A-CTX had
    0, 3 and 138 rows firing only at object positions at q = 0.2%, 1% and 5%; at q = 5% A-TOKG had 78, A-SUP 24
    and A-RES 18. Here the indicator covers the scored sequence (§3.1), both flags are recorded, S2 reads the
    scored-sequence flag, and the prefix-flag statistics and the object-only counts are reported. Open for the
    owner before the freeze: whether S2(ii) and S2(iii) should read the prefix flag instead (the reading "the
    gate recognised the name"), or whether the gate should be confined to prefix positions (a change of
    mechanism, see 9). Both readings can be compared on the same discovery records; no rerun is needed.
11. G-KA1c. The programme asks for "lived-form efficacy ≥ 0.9 on the newest fact" of the FT streams on world 0.
    Here, per arm: the share of the 100 stream facts that are produced on their lived prompt right after their
    own write. The gate blocks the freeze (§8).
12. Consistency rule. The programme's "≤ 1% disagreement" is applied per pair of scorers to the share pooled
    over the analysed worlds. The per-world sample (5% of the battery: a few dozen probes) is too small for a
    per-world 1% rule: one disagreement would exceed it.
13. Hierarchical bootstrap. The programme asks for it for descriptive intervals in general. Here it is computed
    for the two contrasts KA-H1 (units: TEST edits) and KA-H2 (units: facts with a FUSED form), whose per-edit
    and per-fact outcomes are in the records. The other stream metrics are stored as per-world aggregates and
    get no bootstrap interval.
14. Not run or not computed, relative to the programme: the subject-site codes at the prediction position
    (SUBJ-TOK, SUBJ-ATT) as address arms of the memory, and with them manipulation check M1 (PRED-RES against
    SUBJ-TOK); manipulation check M2 (deferred with the linear memories, §0); M3 as a comparison with CEV-11's
    own STD records on a discovery base (A-TOK calls the frozen `compute_write`, and a unit test compares matrix
    and logits bit for bit on a tiny model; nothing is asserted on a trained base); the programme's tests 8
    (linear-memory limits) and 13 (regression against the pilot files), which need the linear memories, the pilot
    outputs and the base weights; cost per write (only phase and arm timings are kept, in DONE.json). The unit
    tests run on tiny random models only. The smoke run on discovery world 0 exercises the pipeline on a trained
    base, but it is a run, not a test with assertions on that base beyond the pipeline's own checks.

## 11. Limits, and what a negative result teaches

- The model is tiny, the world synthetic and templated, one tokenizer, one architecture. Nothing here is a claim
  about large models or natural text.
- A calibrated address is fitted on 60 calibration entities per world; that it transfers to written facts is
  KA-H3, not an assumption.
- KA-H1 refuted while S2(ii) holds and S2(iii) fails: δ is site-specific; reach is limited by the value, not the
  address. KA-H1 failing only against A-TOKG: the radius gate, not the attention code, buys the reach.
- KA-H2 "FT better" or "equivalent": sequential fine-tuning at CEV-11's configurations already lives well at
  N = 100 without drift, and this project needs no separate memory at this scale.
- KA-H3(a) refuted: calibration on c0 does not transfer; "tuning-free" is withdrawn. KA-H3(b) refuted: false
  firings grow with N despite the Bayes prior.

## 12. Freeze record (2026-11-14)

Frozen by the study lead after the discovery jobs (worlds 0 and 1) and before any confirmatory run.

- Open points of §10: every choice described there as "here" is accepted as implemented. No mechanism, margin,
  grid or SESOI was changed after the discovery records were seen. In particular items 2 (absolute locality guard,
  low power acknowledged), 9 (the radius gate is not restricted to the first match) and 10 (S2 reads the
  scored-sequence flag; the prefix-flag statistics are reported beside it) stay as written.
- Discovery outcome that shapes the confirmatory families (results/selection.json, hash in
  results/selection.sha256): the screens G-KA1a and G-KA1b failed. On both discovery worlds every gated address
  (A-CTX, A-TOKG, A-RES, A-SUP) and A-TOK was infeasible under the leak constraints at the frozen mixtures
  (F* = 0), while the fine-tuning references reached F* 0.39-0.64. KA-H1 (the attention-written stream is a
  better write address) is therefore NOT confirmatory: it is reported as a discovery-stage negative result and
  its confirmatory records are descriptive. The primary family is {KA-H2, KA-H3} (m = 2), the secondary family
  is {S2}; PRED-ATT failed its screen, the read layer is 5.
- G-KA1c and the consistency rule passed; no blocking gate failed.
