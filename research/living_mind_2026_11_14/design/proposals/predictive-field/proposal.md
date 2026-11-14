# Substrate B — the Predictive Field (Öngörü Alanı)

A settling, growing, precision-weighted field that learns while it runs. Proposal of 2026-11-14.

## Türkçe özet (12 satır)

1. Öneri: sabit bir ağ yerine "Öngörü Alanı": her olayda yinelemeyle yerleşen, büyüyen ve inancını kesinlik (nat cinsinden kanıt) olarak taşıyan bir dinamik sistem.
2. Öğrenme ile çalışma tek süreçtir: alan yerleşirken aynı adımda sinapslar ve kanıt değişir; eğit–dondur–üret ayrımı ve ayrı eniyileyici yoktur.
3. İlk duyuş zayıftır (0.23); çelişen bilgi eskisini silmez, aynı birime ikinci bir kip ekler ve birimin kesinliği artar (0.57).
4. Yeni kavram (koyu|açık değişkenliği) ve kategori kuralı ("bütün renkler bazen koyu bazen açıktır") gerektiğinde kurulur; kural örneğe destek verir.
5. Denklemlerin ürettiği sayılar: 0.23, sonra 0.57, kural 0.74, kırmızı 0.80 (sahibin son iki sayısı 0.77 ve 0.88). Yalnız iki sabit ilk iki sayıya ayarlandı; gerisi hesap.
6. Kırmızı 0.88'e ancak iki başka renk kuralı doğrulayınca çıkar (tavanı 0.884); "başka hangi renkler var?" sorusu en yüksek bilgi değeriyle (0.25 nat) öne çıkar, ama bu sıralama kırılgandır.
7. Hayal: kural bilinmeyen üyeye uygulanır ("mavi bazen koyu bazen açıktır"), gerçeklik puanı 0.22 ve "duyulmuş kanıt: 0" etiketiyle tutulur; hayal hiçbir zaman kanıt sayılmaz.
8. İşleten bilinç: genelleme, sorma, kurma işlemlerinin kendi sicili vardır; aynı iki cümle farklı geçmişli zihinlerde farklı kurulur (kural 0.65 / 0.74 / 0.89); alanlar arası taşınma tasarımda vardır, henüz kanıtlanmadı.
9. Dürüstlük: parçaların hiçbiri yeni değildir (öngörücü kodlama, Hopfield, aktif çıkarım, hiyerarşik Bayes, NARS, ART); yeni olabilecek tek şey birleşim ve ölçülecek davranıştır.
10. Belirleyici deney (3–4 çekirdek-saat): aynı olay akışı; eş parametreli küçük Transformer (anlık görüntü) ve aynı Transformer + test anında ince ayar ile karşılaştırma.
11. Çürütme: Transformer+ince ayar kalıcılıkta ve tümevarımda eşitlerse, işletme becerisi yeni alana taşınmazsa ya da bellek sınırsız büyürse öneri yanlıştır.
12. Riskler: akıcı dil (Transformer kazanır), eşiklerin öğrenilmesi (alanın çözemediği sorun), simgesel dünyada "alan"ın katkısının belirsizliği; 2.000 satırlık kurulum planı sondadır.

## 0. Status of this document

- **Run:** `pf_trace.py` in this folder (719 lines, numpy only, under one second on one core). Its output is `trace_output.txt`;
  its key numbers are in `trace_numbers.json`. Every number in section 3 comes from it. These numbers are arithmetic
  consequences of the equations of section 2. They are not evidence that the engine works.
- **Not run:** the prototype of section 8 and the experiment of section 5. No experimental claim is made.
- **Literature:** 41 web searches (one in extended mode); arXiv abstracts read for 21 papers; full text read in the
  cited parts for 5 (BayesPCN, CMP, AXIOM, SQHN, the hierarchical-Gaussian-filter predictive coding paper). Tags in
  section 6: **[V]** read in the source text, **[A]** abstract or search snippet only, **[M]** from memory, not checked.
  Fetched material is under `sources/` (one directory per paper: the abstract page and, where read, the extracted
  text; the PDFs were deleted to save space); the helper scripts are in `tools/`.
- **Calibration:** two constants were fitted to the owner's first two numbers (0.23, 0.57). Nothing else was fitted.
- I read `../VISION.md`, `../LEAD_DESIGN.md` and the two notes in `../landscape/` (written by other agents of this
  session) and refer to them where they bear on a claim. Section 9 says how this substrate relates to them.

## 1. The idea in five sentences

1. **One field, one step.** The core is a growing bank of concept codes and association units (a cue that predicts
   a mixture of values) under one free-energy function; every event (a statement, a question, an idle tick) is
   processed by letting the field settle by iteration, and the same settle changes the synapses, so there is no
   training phase to separate from use. *(P1, P6)*
2. **Precision is conviction.** Each unit's synaptic precision, counted in nats of accumulated evidence, is its
   belief strength: a first hearing adds one increment on a sceptical prior (0.23); a contradicting value is not
   written over the old one but opens a second mode in the same unit, whose precision grows (0.57); precision also
   sets plasticity (gain = 1 / precision). *(P2)*
3. **Structure is recruited when error cannot be explained.** A "none of the existing" option competes in every
   settle (new concept, new unit, new mode); where the fillers of a unit differ, its precision is relaxed along the
   difference, which forms a new concept (a variation); relaxing the subject slot to the category forms a rule whose
   prior comes from the operator level; cavity-corrected messages carry support from the rule to the instance and
   back, and no belief receives its own evidence back. *(P2, P3)*
4. **Imagination is the generative direction, with a reality measure.** The same units are run downward on members
   that have no evidence; the result is kept as a construction that carries zero evidence nats and a reality score
   (posterior probability plus the share of its support that was actually told); the hole in the most valuable
   imagined probe is emitted as a question. *(P4, and the question of P2)*
5. **The operating process is itself a field of graded beliefs.** How eagerly to generalise, how much to trust a
   source, when to recruit, when to ask are beliefs of a second, content-free field, updated by the same rule from
   the verified outcomes of its own constructions; so the way of constructing improves with experience and carries
   over to a domain whose words it has never met. *(P5)*

How natively each principle is met (my own reading; "partly" is not softened):

| Principle | Met | Where | Reservation |
|---|---|---|---|
| P1 one process | natively | 2.3: one settle per event, the write belongs to it; a probe writes no evidence but still updates usage and the operator level | none in the design; untested at scale |
| P2 graded belief, refinement, rule, feedback | natively | 2.4–2.6 | the size of the feedback is set by an operator prior, not by the two statements |
| P2 the question | by an innate value function | 2.9 | the ranking is fragile (section 3, table E) |
| P3 concepts formed when needed | partly | 2.5: units, variation concepts and categories are recruited | the codes of perceptual concepts come from the channel adapter; no features are learned |
| P4 imagination with a reality measure | natively as defined | 2.7 | "constructs more coherently over time" rests on the operator level and is untested (H6) |
| P5 the operating process improves | natively by design | 2.10 | only one operator belief (p_H) is exercised by code |
| P6 engine, language | tensors and equations yes; language through a hand-written adapter | 2.1–2.12, section 4 | language at scale is the weakest part |

What is specific to this substrate and what is not (said early, because it matters for choosing between
substrates): the belief calculus of 2.4–2.6 and 2.10 (nats, cavity messages, licences, operator-level track record)
does not depend on the field and could sit on a graph of units and ties. What the field adds is 2.1–2.3 and 2.5:
graded matching of noisy or inflected input, novelty decided by an energy comparison, and generalisation as
relaxed precision. In a clean symbolic world the field settles in one sweep to one-hot choices and adds nothing
measurable (section 10). Hypothesis H8 of section 5 tests whether it earns its keep.

## 2. Formal model

Notation. d: code dimension (256 in the prototype; 64 in the trace). σ(x) = 1/(1+e^(−x)), logit its inverse.
"nats" always means natural-log likelihood ratio. A *frame* is one statement reduced to role slots (S subject,
R relation, O value), produced by the channel adapter (section 4).

### 2.1 State (all of it plastic, all of the time)

| Object | Tensors | Shape | Meaning |
|---|---|---|---|
| Concept bank | codes C, code precision φ | C: N×d, φ: N | one attractor per concept; N grows |
| Association unit m | cue indices (s_m, r_m); modes o_mj; evidence Λ⁺_mj, Λ⁻_mj per provenance class | 2 ints; J_m ints; J_m×\|Π\| floats | "(S, R) predicts a mixture over values" |
| Variation concept V | centre μ_V, axes U_V, spreads s²_V | d; d×r; r (r = J−1 ≤ 3) | the value slot of a unit with ≥ 2 modes, as a blob |
| Rule unit r | (category, relation, V), compliance θ_r, licence, member table (μ_j, L_j, u_j) | O(members) scalars | the same unit with the subject slot relaxed to a category |
| Ledger window | (statement id, unit, mode, nats, provenance, source) | W×6 | last W statements; older ones are absorbed (2.11) |
| Operator level | situation prototypes, Beta counts (a, b), construction ledger | P×d_s, P×2 | beliefs about the operating process (2.10) |
| Workspace (fast) | slot states z, responsibilities a (concepts), h (units), g (modes) | 3×d, 3×(N+1), M+1, J+1 | exists only during one settle |

Provenance classes Π = {sensed, told, acted, imitated} (grounded) and {inferred, imagined} (constructed). Only
grounded classes ever enter Λ. Constructed support exists only as messages (2.6) and is recomputed at every settle.

Codes are tied: the bank C encodes input tokens and decodes output values. The prediction error of a value slot is
therefore available in closed form in code space, e = C[o_target] − ẑ_O, with no output layer to differentiate
(the first reusable local fact).

### 2.2 The free energy

For an event with observed slots x_k (sensory precision π_k; π_k = 0 for a slot that is asked or imagined):

```
F(z,a,h,g) =  Σ_k ½ π_k |x_k − z_k|²                                          sensory
            + Σ_k [ Σ_n a_kn ½ φ_n |z_k − C_n|²  +  a_k0 c_con ]               concept layer
            + Σ_m h_m [ ½ α |z_cue − w_m|²  +  Σ_j g_mj E_mj(z_O)  +  g_m0 c_mode ]  +  h_0 c_unit
            + Σ_k Σ_n a_kn ln a_kn + Σ_m h_m ln(h_m/ρ_m) + Σ_m h_m Σ_j g_mj ln(g_mj/ω_mj)     (entropies, priors)

w_m = [C[s_m]; C[r_m]]            cue code of unit m, read from the bank
E_mj(z) = ½ α |z − C[o_mj]|²      a point mode
F_m^val = −ln( Σ_j ω_mj e^(−E_mj(z_O)) + e^(−c_mode) )    free energy of the value slot (0 if the value is not observed)
E_V(z)  = ½ (z − μ_V)ᵀ P_V (z − μ_V),   P_V = α (I + α Σ_V)⁻¹ = α [ I − U diag( α s²/(1+α s²) ) Uᵀ ]   a variation
```

α is the code precision of a slot (8 in the trace). Index 0 is the option "none of the existing ones", with constant
energies c_con, c_unit, c_mode (c_unit = 5.0 and c_mode = 3.0 in the trace, which recruits concepts by exact token
match and so does not use c_con). These three constants are the vigilances of the field;
they are operator-level beliefs (2.10), not hand-set forever. ρ_m and ω_mj are usage priors. P_V is obtained from
the modes by the Woodbury identity (the second reusable local fact): precision is relaxed exactly along the
directions in which the fillers differ and stays α elsewhere.

### 2.3 Perceiving one statement

**Settle** (block coordinate descent on F; each block update is the exact minimiser, so F never increases and the
iteration converges; T ≤ 5 sweeps, one sweep when the input is clean):

```
a_k  ← softmax_n ( −½ φ_n |z_k − C_n|² ),            option 0 scores −c_con
h    ← softmax_m ( −½ α |z_cue − w_m|² − F_m^val + ln ρ_m ),   option 0 scores −c_unit
g_m  ← softmax_j ( −E_mj(z_O) + ln ω_mj ),           option 0 scores −c_mode
z_k  ← precision-weighted mean of { x_k (π_k), Σ_n a_kn C_n (φ), unit prediction (α Σ_m h_m …) }
```

**Decide** (at the fixed point): if option 0 wins in a, recruit a concept with code z_k; if it wins in h, recruit a
unit for this cue; if it wins in g of the owning unit, add a mode. Otherwise the winning concept, unit and mode own
the frame. Nothing is ever written over a stored mode.

**Write evidence** (local, at the owning unit m*, mode j*):

```
λ_e = λ(source) · χ_e           nats of this statement; χ_e = product of the winning responsibilities (≈ 1 if clean)
Λ⁺_{m*j*}[class] += λ_e          for "S is not O":  Λ⁻_{m*j*}[class] += λ_e
ledger.append(statement id, m*, j*, ±λ_e, class, source)
```

**Timing of the write.** v0 writes once, at the fixed point of the event's settle; there is no later phase and no
optimiser. Because the increments are linear in the responsibilities, writing a provisional increment at every
sweep and correcting it at the next gives the same result (the schedule of incremental predictive coding).

**Move codes** (Kalman form; a concept that is well known moves little):

```
C_n ← C_n + (π a_kn / (φ_n + π a_kn)) (x_k − C_n),      φ_n ← φ_n + π a_kn
```

Why precision can be counted in nats: a frame observed with precision π that lies at energy 0 under its unit and
at energy ½ π |C_a − C_b|² under the best alternative contributes that difference as log-likelihood ratio. For
unit-norm, near-orthogonal codes |C_a − C_b|² ≈ 2, so one unit of sensory precision is one nat. Confusable codes
(cosine 0.8) give 0.2 nats per unit of precision: evidence that cannot tell two concepts apart is weak evidence.

The trust in a source is λ(source) = ln(1 + q/((1−q) p₀)), with q the operator-level belief that a statement of
this source is reliable. The two calibrated constants of section 3 correspond to p₀ = 0.063 and q = 0.178.

### 2.4 Belief strength, content, answer

```
Λ_m      = Σ over distinct statements in the ledger of m of their nats  −  negative nats        (own evidence)
η_m      = η⁰ + Λ_m + δ_m                existence log-odds;  η⁰ = logit p₀;  δ_m = message from rules (2.6)
c_m      = σ(η_m)                         conviction that the cue is real and its value is among the modes
p_m(j)   = Λ⁺_mj / Σ_j Λ⁺_mj               content: Dirichlet mean over the modes
claim "S R o_j"  has strength  c_m · p_m(j);   grounded share  g_m = Λ_m / (Λ_m + max(δ_m, 0))
```

Each statement id is counted once per unit, whatever number of modes it touches. The probability reported to the
outside is passed through a monotone calibration map c' = σ(β₁ η + β₀), with (β₁, β₀) operator-level beliefs
fitted online on verified forecasts (start: 1, 0).

**Answering** a cue (S, R, ?): clamp z_S, z_R, leave z_O free (π_O = 0) and settle. If a unit owns the cue, the
answer is its modes with strengths c_m p_m(j) and the grounded share. If no unit owns it but a rule applies through
a membership, the answer is the rule's content with the constructed strength of 2.6 and grounded share 0, spoken
with a hedge ("I assume"). If option 0 wins everywhere, the answer is "I do not know", and a fill question becomes
a candidate (2.9). A probe writes no evidence; it updates usage priors and the operator level only.

### 2.5 Forming a concept and a category

Three recruitments, all the same event (option 0 wins, or a coincidence must be explained):

1. **Perceptual concept.** No attractor explains a token: a concept is recruited with the token's code.
2. **Variation concept (REFINE).** A unit gains a second mode. Its value slot becomes a blob: μ_V = Σ_j p_j C[o_j],
   Σ_V = Σ_j p_j (C[o_j] − μ_V)(C[o_j] − μ_V)ᵀ, precision P_V as in 2.2. For two modes, the slack axis is the
   difference of the two codes and the coverage energy of each mode is ½ α s²/(1 + α s²) with s² = |C_a − C_b|²/4.
   When a second unit needs the same blob (a rule and its instance), V is entered in the concept bank as a concept
   of its own. This is the "new concept" of the owner's example: not dark, not light, but "dark-or-light".
3. **Category.** Named: the value of a membership unit (j, isa, C) is a category. Latent: when two units with
   different subjects hold overlapping content under the same relation and no shared category explains it, the
   coincidence costs Σ ln(1/P(match by chance)) nats; if that exceeds c_unit, a category concept κ is recruited
   with membership units (j, isa, κ) whose evidence is the coincidence nats (provenance: inferred). A word heard
   later ("these are called colours") attaches to κ.

### 2.6 The rule and the support that flows through it

**GENERALISE.** Trigger T2 (eager, from one instance): a unit holds a variation V with ≥ 2 grounded modes and its
subject has a membership unit. Trigger T1 (from data): two or more members agree on a value. The rule
r = (C★, R, V) is the instance unit with the precision of its subject slot moved from the instance to the category.
Its prior p_H(s) is read from the operator level for the structural situation s (2.10); no statement supports it
directly unless someone says "all C are …". Triggers are conditions on the state, re-evaluated after every event,
so the order of arrival does not matter (checked in section 3, table E). Every rule records its **licence** (the
ledger condition that created it); a rule whose licence no longer holds stays in memory but is dormant.

**Messages.** Variables: h (the rule holds), M_j (j is a member), x_j (j's unit is real with content in V).
Generative assumption: P(x_j = 1 | h, M_j) = θ_r if both hold, b otherwise (b = p₀). For member j let
μ_j = σ(η⁰ + Λ_isa,j) (membership from its own evidence), L_j = exp(Λ_j^V) (likelihood ratio of j's own evidence
for content in V; < 1 if the evidence is against), D_j = b L_j + 1 − b.

```
up:        u_j     = ln { [ μ_j (θ L_j + 1 − θ) + (1 − μ_j) D_j ] / D_j }
rule:      η_r     = logit p_H(s) + Λ_r(direct) + Σ_j u_j ,          q_r = σ(η_r)
cavity:    q_r\j   = σ(η_r − u_j)                                   the rule as seen without member j
down:      prior_j = q_r\j μ_j θ + (1 − q_r\j μ_j) b ,              δ_j = logit(prior_j) − logit(b)
instance:  c_j     = σ( logit(prior_j) + Λ_j^V )
member:    odds(M_j) = odds(μ_j) · [ q_r\j (θ L_j + 1 − θ) + (1 − q_r\j) D_j ] / D_j
```

These are the sum-product messages of the star-shaped factor graph of one rule; on a tree they are the exact
posterior (checked against brute-force enumeration in section 3: 1e-16 on the example, 3.4e-15 at worst on 300
random trees with θ < 1 and negative evidence). With several rules sharing members the
graph has loops; the messages are then iterated with damping and are approximate (test in section 8).

Three properties follow from the form of the equations:

- **No instance receives its own evidence back.** The downward message uses q_r\j, which excludes u_j.
- **The lift of an instance by a rule built on that instance alone comes entirely from the rule's prior.** With
  p_H = 0 the lift vanishes. So "the rule feeds support back" is not evidence counted twice: it is what conceiving
  a new hypothesis that has a prior of its own does to the joint posterior. The prior is the operator level's
  track record with this kind of construction (2.10).
- **A counter-example lowers the rule and creates an exception without deleting the rule.** The member's own unit
  outvotes the rule for its own cue; θ_r is re-estimated as (a_θ + Σ_j μ_j c_j^own)/(a_θ + b_θ + Σ_j μ_j) from the
  members' own-evidence beliefs (v0 of the trace keeps θ = 1).

### 2.7 Imagining, and measuring how real an imagination is

**Imagine.** Choose a rule r (by epistemic value, 2.9). Run it in the generative direction on a member j that has
no instance unit: clamp z_S = C[j], z_R, leave z_O free, settle; the value slot settles on μ_V and decodes to the
rule's content. The construction x̂ = (j, R, V) is stored as a *virtual member* in the rule's member table with
provenance "imagined" and **zero evidence nats**. It sends the message u_j = 0 upward, so imagining changes no
other belief.

**Score.**

```
R(x̂)  = σ( logit(prior_j) + Λ_remembered(x̂) )        reality score
g(x̂)  = Λ_remembered⁺ / (Λ_remembered⁺ + max(δ_j, 0))  grounded share (0 for a pure construction)
```

Λ_remembered is the nats of grounded ledger frames that x̂ covers (coverage energy below c_mode) minus the nats
of frames against it. Scoring is therefore "letting the construction compete for remembered evidence".

**Advance by it.** When grounded evidence about x̂ arrives, the forecast R is scored (log score and Brier score),
and the outcome updates the operator level (2.10): its belief about GENERALISE in situation s, and the calibration
map. Wrong constructions are not removed: the instance keeps its own unit as an exception, the rule's θ and q
fall, and the construction's record stays in the operator level's ledger as a failed forecast.

### 2.8 Retracting

Remove a statement id from every ledger in which it occurs; drop modes left without grounded evidence; re-check
licences; settle. Because every belief is a function of the ledgers and the licensed structure, the state after
retraction equals the state that would have been reached without the statement (section 3, table E: difference
0.0). The concept recruited for a heard token stays: the token was heard. Exact retraction is available inside the
ledger window W only (2.11).

### 2.9 Generating a question

Candidates come from the holes of the units touched in this settle:

```
confirm(u)     value = I(u; answer) = H( p r + (1−p)(1−r) ) − H(r),     p = belief in u,  r = σ(λ)
fill(S,R,?)    value = H(content of the cue) − E H(content | answer)     (asked when a needed cue is unknown)
enumerate(r)   value = E[k] · ( KL_new + γ I_follow )
               P_unknown = α_cat / (α_cat + Σ_j P(M_j)),   E[k] = P_unknown / (1 − P_unknown)
               KL_new    = KL( Bern(σ(η⁰+λ)) ‖ Bern(p₀) )               surprise of one new membership fact
               I_follow  = I(h; one statement about the new member's value)
```

α_cat is the operator-level belief about how open categories are (the concentration of a Chinese-restaurant
prior); γ = 0.5 discounts the second step. The enumerate question is what imagination produces when it instantiates
a rule's relaxed slot and most of the predictive mass falls on option 0, a member the field does not know: the
unfilled slot of the most valuable imagined probe is the question. The best candidate is emitted as an act on the
"ask" channel if its value exceeds θ_ask, an operator-level threshold learned from realised against predicted gain.

### 2.10 Improving the operating process

The operating process is the set of quantities that decide *how* the field constructs, not *what* it knows:

| Operator-level belief | Governs | Updated by |
|---|---|---|
| p_H(s) | eagerness of GENERALISE in structural situation s | later evidence on the rule's *other* members |
| q(source), hence λ | trust in a source, including the source "my own construction" | later independent evidence on the same unit |
| c_con, c_unit, c_mode | when to recruit (vigilance) | later merges (a recruit was wrong) and splits (an absorb was wrong) |
| α_cat | how open a category is; drives enumerate questions | how many new members later appeared |
| θ_ask, γ | when and what to ask | realised against predicted information gain |
| (β₁, β₀) | calibration of spoken probabilities | every verified forecast |

Each belief is a Beta (or Normal-Gamma) posterior attached to a prototype in a space of **content-free** situation
descriptors s (number of supporting members, own evidence of the trigger, whether the content is a variation or a
single value, membership strength, depth, grounded share). Lookup is the same settle as in 2.3 one level up, with
recruitment of a new prototype when the situation is new. Because s contains no concept codes, what is learned
about constructing in one domain applies unchanged in the next.

The update is one rule. A construction (a rule, an imagined instance, a question, a recruit) is a forecast; when
evidence tests it, the evidence has a likelihood ratio LR (success : failure). The operator level keeps a
**ledger of constructions** (one current LR per construction) and recomputes its posterior from it by assumed-density filtering:

```
posterior ∝ Beta(p; a, b) · ( p·LR + (1−p) ) = w₁ Beta(a+1, b) + w₀ Beta(a, b+1),   w₁ ∝ a·LR,  w₀ ∝ b
(a, b) ← moment match of the mixture
```

LR = 1 returns (a, b) unchanged: a construction that was never tested changes nothing. Two cavity rules prevent
self-confirmation at this level too: the instance that triggered a rule is its premise, not a test (only other
members count), and the prior a rule receives excludes that rule's own entry in the ledger.

### 2.11 Idle processing, memory growth and its bound

When no event arrives the same step runs on internal events: (i) imagine on the rules with the highest epistemic
value; (ii) absorb ledger entries older than the window W into the per-class sums Λ of their units (the knowledge
then lives only in the units; the episode is gone); (iii) absorb an instance unit that a confident rule predicts
completely into one scalar (its L_j) in the rule's member table; (iv) merge the two closest units when the budget
is exceeded, with the loss in nats computed in closed form and logged; (v) re-check licences.

| Quantity | Growth | Bound |
|---|---|---|
| concepts N | one per distinct stem met | N_max; beyond it, least-used concepts are merged into their nearest attractor |
| units M | one per distinct cue (S, R) met | budget B; with absorption, M ≈ rules + exceptions + memberships + recent cues |
| ledger | one row per frame | hard window W; exact retraction only inside it |
| floats in total | N·d + O(M) + 6W | N_max·d + c·B + 6W, fixed by the budget the body grants |

For the regular part of a world with I instances, A attributes and C categories, the episodic content is I·A facts
and the absorbed state is about I + C·A + exceptions: a compression by roughly A. Idiosyncratic facts are not
compressed. Section 5 measures this (H7).

### 2.12 Complexity and capacity

Per event: lexical settle O(N·d) per token (O(d log N) with an index); unit retrieval O(candidates·d) through an
inverted index on concept ids; messages O(members of the touched rules); questions O(touched units). With d = 256,
N = 10⁴: about 8·10⁶ multiply-adds, a few milliseconds on one core; 10⁶ events in the order of an hour.

Capacity of the novelty decision: a wrong unit is accepted when the cosine of two unrelated codes exceeds
1 − c_unit/α = 0.375. For random codes the cosine is N(0, 1/d): at d = 256 that is a 6σ event (about 10⁻⁹ per
comparison, 10⁻⁵ per lookup at N = 10⁴); at d = 64 it is a 3σ event (1.3·10⁻³ per comparison), so d = 64 is
adequate only for the nine concepts of the trace. Codes built from character trigrams (section 4) are similar
on purpose; their false merges are a design question, not a capacity question.

## 3. Worked trace of the "red" example

All numbers: `python3 pf_trace.py` (seed 0, d = 64, α = 8, c_unit = 5, c_mode = 3, θ = 1, novice operator prior
Beta(1, 1)). The front end is symbolic: sentence 1 is given as the frames (red, isa, colour) and (red, is, dark),
sentence 2 as (red, is, light). That mapping, including the decision that "has light shades" uses the same relation
as "is a dark colour", is an assumption of v0 (section 4).

**Calibration (the only fitted numbers).** λ = logit 0.57 − logit 0.23 = 1.4902 nats per statement (likelihood
ratio 4.44); η⁰ = logit 0.23 − λ = −2.6985 (p₀ = 0.0631). So the first two numbers below are reproduced by
construction, not predicted.

### Table A. The two statements

| Step | Operator | What the field does (energies from the script) | Belief |
|---|---|---|---|
| 1 | RECRUIT | "red is a dark colour". Five concepts recruited. Cue energy to the nearest existing unit 7.19 > c_unit 5.0, so a second unit is recruited. Each unit receives 1.49 nats. | red is {dark 1.00}: **0.230**; red isa colour: 0.230 |
| 2a | REFINE | "red also has light shades". Concept "light" recruited. Cue energy 0.00: the unit (red, is, ·) owns the frame. Value energy 8.53 > c_mode 3.0: dark is not written over; light becomes a second mode. Variation concept V{dark\|light} recruited: coverage energy 0.41 nats for dark and for light, 5.71 on average for the four other concepts. | red is {dark 0.50, light 0.50}: **0.570** (own evidence, 2.98 nats) |
| 2b | GENERALISE (T2) | Subject slot relaxed from red to its category. Rule (colour★, is, V) created with operator prior 0.500. Upward message from red: +1.05 nats. | rule "a colour is sometimes dark, sometimes light": **0.740** |
| 2c | SUPPORT | Cavity belief in the rule as seen by red: 0.500 (its prior). Downward message: +1.12 nats. | red: **0.802** (told 2.98 nats, constructed 1.12 nats, grounded share 0.73); red isa colour: 0.600 |
| 2d | ASK | Epistemic values (nats): enumerate 0.251; confirm "is red a colour?" 0.208; confirm the rule 0.169; confirm red 0.141. | question: **"which other colours are there?"** |

Exact enumeration of the same generative model (8 states) gives rule 0.740183, red 0.802253, membership 0.599881;
the largest difference from the messages is 1.1·10⁻¹⁶. On 300 random trees (θ in {1, 0.95, 0.8, 0.6}, positive and
negative evidence, up to five members; part H of the script) the largest difference is 3.4·10⁻¹⁵.

| | first hearing | refined | category rule | red after feedback |
|---|---|---|---|---|
| owner's example | 0.23 | 0.57 | 0.77 | 0.88 |
| these equations | 0.230 (fitted) | 0.570 (fitted) | 0.740 | 0.802 |

Reading, including what does not match:

- The order of the owner's four numbers is reproduced; the last two values are lower (0.74 against 0.77, 0.80
  against 0.88). I did not tune anything to close the gap.
- Where the lift from 0.57 to 0.80 comes from: 1.12 nats of constructed support, all of it traceable to the
  operator prior p_H = 0.5. With p_H = 0 the red belief stays at 0.570. The field reports this itself: 27% of the
  support of "red: 0.80" is its own construction. The novice prior Beta(1, 1) is the uniform prior on an untried
  operator; it is a choice, and table C shows how much depends on it.
- Membership matters. Red's membership in "colour" rests on one statement (0.23 alone, 0.60 after the joint
  settle). If membership is taken as given, the same equations give rule 0.900 and red 0.957 for p_H = 0.5, and
  rule 0.770, red 0.901 for p_H = 0.27 (table E). The owner's four numbers are therefore close to one coherent
  posterior (membership given, p_H = 0.27; fourth number 0.90 against 0.88), but that reading fits three constants
  to four numbers and I do not claim it.
- The ceiling of the red belief when the rule is certain, with membership resting on one statement, is 0.884.

### Table B. The question is answered; imagination; verification; a counter-example

| # | Event | Rule | Red | Other |
|---|---|---|---|---|
| 3, 4 | "blue is a colour", "green is a colour" | 0.740 | 0.802 | IMAGINE "blue is sometimes dark, sometimes light": reality 0.223, told 0 nats, constructed +1.45 nats, grounded share 0. Same for green. Imagining moved no other belief. |
| 5 | "blue is a dark colour" | 0.877 | 0.845 | blue 0.790 (forecast before: 0.223); red isa colour 0.668 |
| 6 | "blue also has light shades" | 0.941 | 0.865 | blue 0.943; green's imagined score has risen from 0.223 to 0.266 without any statement about green's tone |
| 7 | "green is a dark colour" | 0.976 | 0.876 | green 0.852 |
| 8 | "green also has light shades" | 0.989 | **0.880** | green 0.962; exact enumeration with three members (128 states): rule 0.988860, difference 2.2·10⁻¹⁶ |
| 9–13 | "black is a colour", "black is a dark colour", "black is not light" ×3 | 0.976 | 0.876 | black's upward message −0.77 nats; "black is sometimes dark, sometimes light": 0.068; black keeps its own unit {dark} as an exception; the rule is lowered, not deleted |

In these equations the owner's "red: 0.88" is reached after two other colours have confirmed the rule, not at once.
After 13 statements the field holds 9 concepts, 8 units, 1 rule and 1 variation concept.

### Table C. The same two statements heard by minds with different operating histories (P5)

Each history is a previous life in other domains (animals and sizes, tools and weights, foods and tastes, stones
and surfaces), run through the same code: the two-statement pattern on a first member, then two further members
that confirm the rule or refute it (membership stated twice; a refuting member is told three times that it lacks
the second value). The track record counts only members other than the one that triggered the rule. The words
share nothing with the red example.

| History | Track record of GENERALISE (test LR on new members) | Operator belief p_H | Rule | Red after feedback |
|---|---|---|---|---|
| burnt: 1 of 4 earlier rules held | 31.2, 0.21, 0.21, 0.21 | 0.399 (Beta 1.45, 2.18) | 0.654 | 0.775 |
| novice: no history | none | 0.500 (Beta 1, 1) | 0.740 | 0.802 |
| seasoned: 3 of 4 earlier rules held | 31.2, 31.2, 31.2, 0.21 | 0.733 (Beta 3.36, 1.22) | 0.887 | 0.848 |

First hearing (0.23) and refinement (0.57) are identical in the three minds; what differs is how the mind
constructs on top of them. This demonstrates the mechanism only: I wrote the previous lives, so the table is not
evidence that the transfer helps in a world I did not write (that is H5).

### Table D. What the two safeguards prevent (same example, safeguards removed)

| Variant | Rule | Red |
|---|---|---|
| as specified (cavity, constructions carry no evidence) | 0.740 | 0.802 |
| no cavity: posteriors fed back as if they were evidence, iterated to the fixed point | 0.796 | 0.858 |
| imagined instances written back as evidence, 1 imagined member | 0.824 | – |
| the same, 3 imagined members | 0.937 | – |
| the same, 10 imagined members | 0.999 | – |

Reading: in this smallest case the cavity matters little (+0.06). The separation of grounded from
constructed evidence is what prevents a rule from becoming certain by being imagined over ten colours about which
nothing is known.

### Table E. Invariances and sensitivities

| Check | Result |
|---|---|
| Retract statement 2, compare with the state after statement 1 | largest belief difference 0.0; rule kept but dormant; the concept "light" stays |
| Order of arrival (S1 then S2, S2 then S1, frames of S1 swapped) | rule 0.740183, red 0.802253, membership 0.599881 in all four orders |
| 200 random code seeds, d = 64 and d = 256 | identical structure and beliefs in 200 of 200 for both |
| Question ranking against category openness α_cat | enumerate is first for α_cat ≥ 0.82 (0.251 nats at α_cat = 1); below that "is red a colour?" (0.208) is first |
| Membership given, p_H = 0.50 | rule 0.900, red 0.957 |
| Membership given, p_H = 0.27 (fitted to 0.77) | rule 0.770, red 0.901 |

The ranking of the question is not robust: it depends on one operator-level belief (α_cat) and flips at 0.82.
The owner's question is the top one at the default α_cat = 1, and would not be for a mind that has learned that
categories are small.

Approximations in the trace: (A1) a unit with two modes gets the same prior p₀ and the same nats per statement as
a unit with one; a sharper model would raise the prior and lower the diagnosticity of each statement. (A2) one
statement of one mode ("blue is dark") counts as full evidence that blue's content lies in V. (A3) sentence 1
supports both the membership and the tone; the messages treat the two as independent evidence although they come
from one witness. Each needs a test against a witness-level reference (section 8).

## 4. How language enters and leaves

The core never sees text. A channel adapter turns a channel into events (time, channel, code, provenance, source);
a second adapter turns act events into output. Text is the first channel.

**In, v0 (inside the 2,000 lines).**
1. Token code: sum of fixed random codes of the token's character trigrams, normalised (d = 256).
2. Lexical settle: the token code falls into a concept attractor or recruits one (2.3). Inflected Turkish forms
   fall into the stem's attractor with the same energy and threshold as everything else (`pf_trace.py`, part F):

   | form | against | cosine | energy | verdict |
   |---|---|---|---|---|
   | kırmızının | kırmızı | 0.695 | 2.44 | same concept |
   | kırmızıdır | kırmızı | 0.710 | 2.32 | same concept |
   | kırmızıya | kırmızı | 0.768 | 1.86 | same concept |
   | kırıcı | kırmızı | 0.234 | 6.13 | new concept (correct) |
   | mavi | kırmızı | −0.046 | 8.37 | new concept (correct) |
   | rengi | renk | 0.422 | 4.63 | new concept (**wrong**: consonant softening on a short stem) |

   The residual between token and attractor (the suffix) is kept as a cue for the role of the token.
3. Frames: about 14 sentence templates per language (Turkish and English) map token sequences to frames with a
   mood (assert, also, not, question) and a source. This parser is hand-written. It is the part of v0 that is not
   learned, and section 5 keeps it out of the comparison by giving every system the same frames.

**Out, v0.** An answer is a frame with a strength, a grounded share and a content distribution. The value code is
decoded by the tied bank (nearest attractor, then its surface form). A template renders it, and the hedge is chosen
from the numbers, not from a vocabulary softmax: below 0.3 "I heard once that …" (bir kez duydum), 0.3–0.6
"possibly" (belki), 0.6–0.85 "probably" (büyük olasılıkla), above 0.85 plain; grounded share 0 adds "I assume"
(sanırım, ben kurdum); two modes render as "sometimes … sometimes …". Silence and "I do not know" are legal outputs.

**How it scales (v1, outside the 2,000 lines, uncertain).**
- Templates become learned *constructions*: a construction is a unit over token positions whose varying positions
  have relaxed precision. It is formed by the same REFINE and GENERALISE operators applied to sentences: two
  sentences that differ in one position give a slot. Which slot fills which frame role is learned first by
  imitation of the hand-written parser's output, then kept or revised by the reality measure.
- Sentences that no construction parses are stored as raw sequence chunks with provenance "imitated", to be
  explained later. Coverage (share of sentences parsed) is the scale metric, not perplexity.
- Cost: O(N·d) per token without an index. Vocabulary of 10⁴–10⁵ forms at d = 256 fits in 10–100 MB.
- A realistic phase-2 target is child-directed text of 5–20 thousand word forms and 50–200 constructions.
- What I expect to fail: fluent open text and long-range syntax. A Transformer of equal size will have lower
  next-word loss; the CMP paper (2026) reports exactly such a gap for its local-learning architecture and notes
  that recent non-attention models have not closed it either. Stem alternation needs alternation-aware codes or a
  learned merge of two concepts that behave identically.

## 5. Falsifiable hypotheses and a decisive experiment

### 5.1 World and stream ("Living Worlds")

A generator W(seed) produces a life of 6 domains met one after another. Per domain: 3 categories × 8 instances;
4 attributes with 4 values each; each (category, attribute) pair is of kind RULE-VAR (all members share a two-value
variation), RULE-VAL (all share one value) or IDIO (every member has its own values), with a mixture that is hidden
and differs between domains (a homogeneity regime: 70% rules in some domains, 20% in others); 10% of the members of
a rule are exceptions; 10% of statements are false; some are repeated; some are later corrected. All tokens are
random strings drawn anew per life, so nothing about any domain (colours included) is in the core or the tests.
A domain contributes about 300 statements and 300 probes, interleaved; probes give no feedback. A life is about
1,800 statements and 1,800 probes. Probe types: recall of a told fact at delays 1, 10, 100 and end of life; both
values after an "also"; the value after a correction; induction for a member whose attribute was never stated
(RULE pairs: predict; IDIO pairs: abstain); an entity never met (must say unknown). Every answer is a set of
values with probabilities, scored by accuracy or F1 and by Brier score and expected calibration error. A system
may put up to 20 questions per domain to an oracle.

All systems receive the same frames as symbol sequences and answer in the same form, so the comparison is about
the core, not about parsing. Two discovery lives fix every operating constant and every baseline hyper-parameter;
eight confirmatory lives are run once (the convention of this repository: discovery worlds, freeze, confirmation).

### 5.2 Systems

| System | Description |
|---|---|
| PF | this proposal; memory counted in floats at the end of life (expected 50–80 thousand) |
| TF-snapshot | a small Transformer decoder with the **same number of parameters as PF's floats**, trained the standard way: at each domain boundary, trained to convergence by backpropagation on the whole stream so far, then frozen until the next boundary |
| TF-snapshot+TTT | the same Transformer with test-time fine-tuning: k optimiser steps on every incoming statement, with a small replay buffer; learning rate, k, layers and buffer tuned on the discovery lives |
| TF-online | the same Transformer trained in one online pass, no snapshot |
| TF-8× | TF-snapshot+TTT with eight times the parameters (robustness of the matched-size comparison) |
| TF-meta (extra) | the 8× Transformer pre-trained on 2,000 other lives of the generator, frozen, the life in its context window (512 tokens) |
| Table | every statement stored, answers by exact lookup with counts; no rules |
| HB-oracle | hierarchical Bayes given the true structure of the world, exact inference: the ceiling |
| PF ablations | no operator level; no rules; no cavity; imagination written as evidence; random questions; overwrite instead of REFINE; hash keys instead of the field (for H8) |

A Transformer's graded belief is its next-token probability for the value position; its question is the
highest-entropy cue among sampled candidates.

### 5.3 Hypotheses (pre-registered thresholds; all on the confirmatory lives)

Each row has a support threshold and a refutation threshold; a result between the two is reported as undecided.

| # | Claim | Refuted if |
|---|---|---|
| H1 (P1) | Facts told once in domain 1 are recalled at the end of life with accuracy ≥ 0.90, and PF − TF-snapshot+TTT ≥ 0.10. | PF < 0.90, or the difference < 0.05 |
| H2 (P2) | After an "also" statement both values are returned (F1 ≥ 0.85) and PF − TF-snapshot+TTT ≥ 0.10; after a correction the old value stays retrievable with lower strength. | F1 < 0.85, or difference < 0.05 |
| H3 (P2, P3) | Induction on RULE pairs: PF ≥ Table + 0.25 accuracy and within 0.05 of HB-oracle; on IDIO pairs PF abstains (false assertion rate ≤ 0.2). Brier score over all probes: PF ≤ TF-snapshot+TTT − 0.02; calibration error ≤ 0.08. | any of the three fails |
| H4 (question) | With the same budget, PF's chosen questions reach a given induction Brier score with ≤ 60% of the questions that random choice needs. | ratio > 0.8 |
| H5 (P5) | Statements needed to reach a fixed induction Brier score fall from domain 1 to domain 6 by ≥ 30% for PF and by < 10% for PF without the operator level; after a regime flip at domain 4 PF needs ≤ 1.3 × its domain-1 count. | the with/without difference is < 10 points, or the flip costs more than 1.3 × |
| H6 (P4) | Reality scores of constructions are calibrated by the last domain (error ≤ 0.10) and their Brier score falls by ≥ 20% from domain 1 to 6; the ablation "imagination as evidence" shows rule over-confidence (calibration error ≥ 0.2). | calibration error > 0.10 or no fall |
| H7 (growth, stability) | Floats at the end ≤ 0.5 × the Table's with retention loss ≤ 0.02; every settle converges within 5 sweeps in ≥ 99.9% of events; no belief leaves (0, 1). | any fails |
| H8 (the field earns its keep) | With 15% of tokens corrupted (one character changed or an inflected form), PF's recall falls by ≤ 0.05 while the hash-key ablation falls by ≥ 0.20. | the hash-key ablation is within 0.05 of PF |

### 5.4 What result refutes the proposal

- **As a better living core:** if TF-snapshot+TTT at equal parameters is within the margins on H1, H2 and H3, the
  properties the owner asks for are available from test-time fine-tuning at this scale, and leaving the Transformer
  is not justified by this experiment. If only TF-8× closes the gap, I report "holds at equal size, not at 8×".
- **As a native account of P5:** if H5 fails, the operating process does not improve in this design.
- **As an engine:** if H7 fails (memory grows like the Table's, or settling fails).
- **As substrate B in particular:** if H8 fails, the field adds nothing over a symbolic store and the belief
  calculus should be carried by the simpler substrate.
- Expected losses, stated in advance: TF-meta will be at or near the HB-oracle inside its context window on
  lives that resemble its pre-training lives; any fluent-language measure goes to the Transformer.

### 5.5 Cost

PF: about 3,600 events per life at a few milliseconds each, under a minute per life; ten lives and seven ablations
in about ten minutes. Transformers of 50–600 thousand parameters on 4 cores: six snapshots per life (the stream is
at most about 10⁴ tokens; a few hundred steps each) about 3 minutes per life; test-time fine-tuning about 1 minute
per life; tuning 6 configurations on 2 discovery lives about 20 minutes; TF-meta pre-training about 1 hour.
Total about 3–4 core-hours.

## 6. Prior art (from the searches of this session)

### 6.1 The eight closest works

| # | Work | What it already does | What it does not do (as far as I read) | Tag |
|---|---|---|---|---|
| 1 | BayesPCN (Yoo & Wood, NeurIPS 2022, arXiv:2205.09930) | A predictive coding associative memory whose write is a "local conjugate normal-normal Bayesian update" of the weights; continual one-shot writes without meta-learning; a diffusion-based forget that returns the memory toward its prior. | Fixed architecture (no recruitment); stores and recalls patterns; no categories, rules, questions or operator level. | [V] |
| 2 | Closed-form predictive coding via hierarchical Gaussian filters (Baskakovs … Mathys, Legrand, arXiv:2605.20293, May 2026) | Says current predictive coding networks "fix the precision matrix to the identity"; restores precision-weighted message passing; learns activations, weights and precisions under one free energy, without iterative relaxation; better than backpropagation on online and concept-drift tasks on FashionMNIST. | Fixed architecture; classification; precision is a learned gain, not the stated strength of a belief; no growth, rules or questions. | [V] for the introduction, [A] for results |
| 3 | AXIOM (Heins … Friston … Buckley, arXiv:2505.24784, 2025) | Active inference with mixture models that grow online: a new component is created when the best posterior-predictive log-density falls below τ = log p₀(y) + log α, "a deterministic, maximum a posteriori version of the CRP assignment rule"; merging by Bayesian model reduction; no gradients; variational Bayesian updates. | Object-centric game playing; no language; no graded statements about rules that feed back to instances; structure priors are designed. | [V] |
| 4 | Active inference structure learning: Smith, Schwartenbeck, Parr & Friston (Front. Comput. Neurosci. 2020); Friston et al., "Active inference, curiosity and insight" (Neural Comput. 2017) | Concept learning by engaging spare hidden-state "slots", with Bayesian model reduction to undo it; curiosity as expected information gain; rule learning with "aha" moments. | Small discrete state spaces with a designed model; slots exist in advance; no engine for an open stream of statements. | [A] |
| 5 | Hierarchical Bayesian overhypotheses (Kemp, Perfors & Tenenbaum, Dev. Sci. 2007) | Induces a category-level regularity from instances and sends graded support back to an instance; learns how homogeneous categories are, which is what p_H(s) is. | Batch posterior over a fixed hypothesis space; no reliability of reports (first belief is strong); no recruitment, no questions. The landscape note `../landscape/art-growth.md` §6.1 ran it on the red stream and found that the owner's shape appears only after adding a reliability parameter. | [A] for the paper; the run is another agent's |
| 6 | NARS (Wang; OpenNARS, ONA) | Truth value (frequency, confidence) with c = w/(w+k); revision pools evidence; stamps prevent counting evidence twice; questions and goals in one reasoner. The landscape note `../landscape/nars-logic.md` verified in the ONA source that revision is refused when stamps overlap, so a rule cannot lift the instance it was built on. | No prior for a newly conceived rule from a learned track record; no vector substrate; first-hearing confidence is a fixed parameter. | [A] from my searches; the source reading is another agent's |
| 7 | Growing associative memories: SQHN (Alonso & Krichmar, Nat. Commun. 2024, arXiv:2307.15040); ART (Carpenter & Grossberg); SUSTAIN (Love, Medin & Gureckis 2004); IGMN (Engel & Heinen) | SQHN: an energy-based model whose "learning algorithm … combines neuro-genesis (neuron growth) and local learning rules" for the online-continual setting, with a decaying growth threshold. ART: recruit a category when no stored one passes vigilance. SUSTAIN: clusters "recruited in response to surprising events". IGMN: one-pass Gaussian mixture with a novelty criterion and rank-one precision updates. | Pattern memory or classification; no rule-to-instance support, no imagination score, no questions, no learned operators. | SQHN [V]; ART, SUSTAIN, IGMN [A] |
| 8 | CMP, "The Art of Not Forgetting" (Atmuri et al., arXiv:2607.17944 and 2607.26523, July 2026) | The nearest recent *experiment*: a local-learning, non-backpropagation architecture (sparse relational codes, two-tier competitive memory, a predictive-coding stage) against a parameter-matched Transformer with online EWC on 15 text domains; much less forgetting. The authors state that the components are not new. | Reports "a real, substantial accuracy gap versus the Transformer baseline", a null result on vision, and a failed attempt to add an accuracy mechanism. No beliefs, rules, questions. | [V] |

### 6.2 Further works that own a part of this proposal

- **Synapses change during settling.** Equilibrium propagation (Scellier & Bengio 2017) has two phases; C-EP
  (Ernoult et al. 2020) makes the update local in time during the second phase; incremental predictive coding
  (Salvatori et al., ICLR 2024) updates activities and weights simultaneously. [A]
- **Precision as plasticity.** MESU (Bonnet et al., Nat. Commun. 2025): "updates network parameters according
  [to] their uncertainty", streaming, no task boundaries. The hierarchical Gaussian filter (Mathys et al.):
  precision-weighted prediction errors with volatility. EWC, variational continual learning. [A] / [M]
- **Predictive coding that learns without forgetting, with a controller.** Lifelong neural predictive coding
  (Ororbia et al., NeurIPS 2022): "another neural system learns to direct and control this cortex-like structure".
  This is the nearest precedent for a second system steering a predictive substrate. [A]
- **Energy and attention.** Modern Hopfield networks (Ramsauer et al. 2020): the update rule "is equivalent to the
  attention mechanism used in transformers". Energy Transformer (Hoover et al., NeurIPS 2023). The settle of 2.3 is
  a modern Hopfield retrieval over two memory banks with a novelty option. [A]
- **Learning at test time inside the Transformer family.** Titans (Behrouz et al. 2024): a memory module updated
  at test time, driven by surprise; Nested Learning / Hope (Behrouz et al., NeurIPS 2025): a self-modifying
  sequence model with a continuum memory. Both are gradient-based and meta-trained. Test-time training (Hardt &
  Sun, ICLR 2024; Akyürek et al. 2024) and dynamic evaluation are the baseline of section 5. [A]
- **Online sequence memory with local rules.** HTM (Cui, Ahmad & Hawkins 2016); the Thousand Brains Project's
  Monty (Leadholm et al. 2025): "associative, Hebbian-like binding", sparse local updates, 3D objects. [A]
- **Questions as expected information gain.** Lindley's design of experiments [M]; Rothe, Lake & Gureckis (NIPS
  2017), questions as programs scored by expected information gain. [A]
- **Reality monitoring.** Dijkstra & Fleming (Nat. Commun. 2023): reality judged by comparing signal strength with
  a threshold; wake-sleep, Dyna, generative replay generate without scoring [M].
- **A level that models and controls the level below.** Deep parametric active inference (Sandved-Smith et al.
  2021): a higher level sets the precision of the lower one (meta-awareness, attention control); strategy selection
  as rational metareasoning (Lieder & Griffiths 2017). [A]
- **Why a rule may lift its only instance.** Awareness growth / reverse Bayesianism (Karni & Vierø 2013): how
  probabilities change when a new hypothesis is conceived; the witness model of partially reliable sources
  (Bovens & Hartmann 2003). [A]
- **Concept formation during revision, symbolic.** MOBAL (Wrobel 1994) forms a concept when no existing one
  captures a rule's instances and exceptions; truth maintenance (Doyle; de Kleer) is the source of licences. [A] / [M]
- **In-context Bayes.** Prior-data fitted networks (Müller et al., ICLR 2022): a Transformer meta-trained on a
  prior approximates the posterior in one forward pass. This is why TF-meta is a baseline and may win. [A]

### 6.3 What overlaps, and what, if anything, remains

Every mechanism above has an owner:

| Part of this proposal | Owner |
|---|---|
| settle by iteration on an energy; local conjugate writes | predictive coding, modern Hopfield, BayesPCN |
| precision learned and used as gain | HGF predictive coding, MESU |
| recruit when prediction error cannot be explained; merge afterwards | ART, SUSTAIN, IGMN, SQHN, AXIOM, latent-cause models |
| rule from instances, support back to the instance, learned homogeneity | hierarchical Bayes (overhypotheses) |
| evidence counted once; revision without deletion | NARS stamps, truth maintenance, belief propagation |
| questions by expected information gain; curiosity | active inference, Bayesian experimental design |
| a higher level that learns to steer the lower one | deep parametric active inference, metareasoning, Ororbia's controller |
| local learning against a matched Transformer on a stream | CMP (2026) |

What I did **not** find in 41 searches and the texts read (absence of evidence, not proof of absence; these are
candidates, not claims):

1. **Rule-to-instance support inside an online, growing field, with the lift attributed to a learned prior.** The
   pieces exist separately (hierarchical Bayes offline; NARS forbids the loop; ART has no rule level). The
   observation that the owner's "self-support" is model expansion with a prior, and not double counting, is an
   application of known probability, not a discovery.
2. **Two kinds of precision on every belief (told against constructed), with the reality score of an imagination
   defined as its posterior plus its grounded share, and with constructions barred from ever becoming evidence.**
   Source tags exist in NARS and truth maintenance; a calibrated reality score on a neural memory's own
   constructions was not found.
3. **The operator level as the same update rule applied to the track record of the system's own constructions,
   indexed by content-free situations, with cavity at that level.** Overhypotheses, metareasoning and deep
   parametric active inference each cover a part; the measured cross-domain transfer of H5 was not found for a
   system of this kind.
4. **The head-to-head experiment**: these properties, on one stream, against a parameter-matched Transformer with
   test-time fine-tuning. CMP (2026) did the forgetting half and reported an accuracy gap.

### 6.4 Relation to the named families, one paragraph each

- **Predictive coding networks.** Same descent on a free energy with local errors. The difference is the hidden
  layer: here it is a mixture with conjugate writes, not a stack of continuous feature layers trained by local
  gradient steps. The price is that this field learns no feature hierarchy; its codes come from the channel
  adapter. BayesPCN is the nearest relative. The hierarchical-Gaussian-filter paper shows how precision is learned
  in deep predictive coding, which this proposal does not attempt.
- **Equilibrium propagation.** It also settles on an energy, but it learns from the difference between a free and
  a nudged equilibrium and needs small learning rates to follow the gradient. Here there is one equilibrium per
  event and no gradient. P1 is met in a simpler and less general way: deep features cannot be trained like this.
- **Modern Hopfield networks and the Energy Transformer.** The settle of 2.3 is a modern Hopfield retrieval. The
  Energy Transformer uses energy descent as the forward pass of a network trained by backpropagation; nothing here
  is trained by backpropagation.
- **HTM.** Both learn online with local rules. HTM's graded synapse is a permanence with a connection threshold,
  and its strength is variable-order sequence prediction, which v0 of this proposal lacks. HTM attaches no belief
  strength to a statement and forms no rule that supports an instance.
- **Continual learning.** Interference is avoided the way SQHN and ART avoid it: by writing to recruited local
  units, which costs growth. The growth literature finds that forgetting returns as soon as learned features are
  shared (`../landscape/art-growth.md` §5.6). This proposal does not solve that; it avoids it by sharing no
  learned features. MESU is the weight-level counterpart of "precision sets plasticity".

### 6.5 Judgement

Nothing here should be called new as a mechanism. If anything survives, it is the closed combination
and its measured behaviour, and that can only be said after section 5 has been run. The owner's earlier verdict
"incremental" on the two add-on proposals would apply to each part of this one taken alone.

## 7. Failure modes and the mechanism against each

| Failure | How it would show | Mechanism | How it is tested |
|---|---|---|---|
| **Runaway growth** | a unit per statement; memory like the Table's. The field's literature reports it: fuzzy ARTMAP keeps about 0.9 categories per training image (`../landscape/art-growth.md` §5.2). | Recruitment only when option 0 wins; absorption of instances that a rule predicts into one scalar; merge under a hard budget with the loss logged; vigilances learned at the operator level. | H7; growth curve per life; failure criterion: slope near 1 unit per statement in a regular world |
| **The threshold nobody can set** | c_unit, c_mode wrong: over-splitting or false merges. The ART survey calls vigilance setting an open problem (same note, §5.3). | (i) thresholds are in nats and compared with likelihood ratios, not arbitrary similarities; (ii) reuse of this repository's calibrated "do I know this context" score (study `research/concept_address_2026_11_13`, hypothesis KA-H3 supported on 8 worlds: a likelihood ratio plus a count prior, read with a null slot): the same statistic sets option 0 here; (iii) operator-level learning from later merges and splits. | Untested in this substrate. A dedicated ablation fixes the thresholds at 0.5× and 2× and reports the damage. |
| **Self-reinforcing false beliefs** | a rule becomes certain by being imagined (table D: 0.999 with ten imagined members); a false statement repeated by one source becomes true. | Constructed support never enters Λ; cavity messages; cavity at the operator level; per-source saturation of repeated statements (the n-th repetition from one source adds λ ρⁿ⁻¹, ρ = 0.5); every answer reports its grounded share. | H6 with the ablation "imagination as evidence"; a liar condition (one source repeats a falsehood 20 times) |
| **Instability of settling** | oscillation between units; loops between rules that share members. | F is a Lyapunov function of the block updates; messages are exact on trees; damping and a 5-sweep cap otherwise, with fallback to the lowest-F iterate and a flag on the units involved. | H7; brute-force comparison on small loopy graphs; count of flagged events |
| **Forgetting** | early facts lost after many later ones. | Writes are local to the owning unit; no shared weights are overwritten; the only lossy operations are merge and absorb, both logged with their cost in nats. | H1 at end of life; retention against budget B |
| **Rigidity** (the opposite failure) | an old strong belief cannot follow a changed world. | Negative evidence subtracts nats; a per-unit volatility estimate discounts old nats when recent predictions fail (hierarchical-Gaussian-filter style; v1). | a change-point condition in the world generator |
| **Order dependence** | a different life order gives a different mind. | Beliefs are functions of the ledgers and the licensed structure; triggers are state conditions (table E: four orders, identical beliefs). Recruitment thresholds near the boundary remain order-dependent. | permutation test: same statements, 20 orders, spread of final beliefs |
| **Correlated evidence** | one sentence that asserts two things counted as two witnesses (A3). | statement ids are shared across ledgers, so the correlation is known; v1 treats a sentence as one witness in a joint factor. | witness-level brute-force reference |
| **Wrong structure** | a variation concept merges unrelated values; an eager rule from one instance. | structure is licensed and dormant when unlicensed; p_H(s) falls where eager rules fail; exceptions are kept as units. | regime-flip condition of H5 |
| **The parser** | sentences outside the templates are lost. | stored as raw chunks with provenance "imitated"; coverage reported. | coverage on a held-out template set |
| **The match score is not a safe reality measure** | an input crafted to score well (reported for fuzzy ARTMAP in 2026; `../landscape/art-growth.md` §5.9). | the reality score is not a match score: it requires grounded evidence, and a construction with none has grounded share 0 whatever its probability. | not tested against adversaries in phase 1 |

## 8. Build plan (2,000 lines of core) with tests

| File | Lines | Content | Tests (each is one or more assertions with numbers) |
|---|---|---|---|
| `pf/codes.py` | 160 | trigram token codes; concept bank; recruitment; Kalman move of codes; capacity check | cosines of the table in section 4; false-match rate at d = 64 and 256 against the normal approximation |
| `pf/field.py` | 300 | units and modes as tensors; energies; block-coordinate settle; three novelty options; inverted index | F never increases over 10⁴ random settles; one-sweep convergence on clean input; recruit / same-unit / new-mode decisions of table A |
| `pf/ledger.py` | 150 | statement ledger with provenance and source; window; absorption into sums; exact retraction | retract(k) equals never-heard(k) to 1e-12 for every k in random streams; no retraction outside the window |
| `pf/belief.py` | 220 | nats to log-odds; content; calibration map; messages of 2.6; brute-force enumerator | messages equal enumeration to 1e-12 on random trees up to 4 members; the golden numbers of table A (0.230, 0.570, 0.740183, 0.802253, 0.599881); error bound on small loopy graphs |
| `pf/structure.py` | 260 | REFINE with Woodbury precision; GENERALISE T1 and T2; latent category; licences; absorb and merge under budget | coverage energies 0.41 and 5.71; order invariance (table E); a dormant rule sends no message; merge loss equals the logged nats |
| `pf/imagine.py` | 120 | generative direction; virtual members; reality score; outcome scoring | imagining changes no other belief (table B rows 3–4); reality 0.223; table D reproduces when the safeguard is switched off |
| `pf/ask.py` | 130 | candidate probes; three value functions; threshold | the ranking of table A; the flip at α_cat = 0.82 |
| `pf/operator.py` | 200 | situation descriptors; Beta and Normal-Gamma beliefs; assumed-density filter with cavity; ledger of constructions | LR = 1 leaves (a, b) unchanged; table C (0.399, 0.500, 0.733); a rule never raises its own prior |
| `pf/lang.py` | 180 | tokeniser; templates to frames (Turkish, English); frames to sentences with hedges | round trip on the template set; hedge chosen from strength and grounded share |
| `pf/engine.py` | 160 | the single `step(event)`: perceive, answer, idle; channels and provenance; save and load | determinism (same stream, same state hash); save/load round trip; a probe writes no evidence |
| `pf/audit.py` | 120 | invariants checked after every step in test mode: belief bounds, ledger sums, memory accounting | used by all tests |
| **core** | **2,000** | | |
| `tests/` | about 600 | the tests above | |
| `exp/` | about 1,000 | world generator, Transformer baselines with test-time fine-tuning, life runner, analysis | generator invariants; baseline parameter count equals PF's floats |

Order and gates:

1. **M0** codes, field, ledger, belief: the golden red trace must reproduce `pf_trace.py` to 1e-9. Gate: if the
   field's settle and the script disagree, stop and find out which is wrong.
2. **M1** structure, operator: tables C and E. Gate: growth bound on a 2,000-statement regular stream.
3. **M2** imagine, ask: tables B and D.
4. **M3** lang, engine, idle processing.
5. **M4** world and baselines; two discovery lives; freeze constants, hyper-parameters and thresholds of 5.3.
   Gate: if PF fails H7 or H1 on discovery, write the negative report and stop.
6. **M5** eight confirmatory lives, once; analysis; report including every hypothesis that failed.

## 9. Relation to the wider vision and to the lead design

`../VISION.md` widens the six principles to fifteen points; `../LEAD_DESIGN.md` proposes units and ties with
evidence per provenance class. This substrate is compatible with both and differs in three places.

| Vision point | In this substrate | Gap |
|---|---|---|
| 1, 2 life, process | one `step` for external and idle events; biography = ledgers and the ledger of constructions | idle scheduling is a fixed list in v0 |
| 3, 3b concepts built by encounters | a concept is its attractor plus every unit it takes part in; codes move with use | one code per concept; facets per channel are not built |
| 4 conviction | precision in nats, per provenance class | – |
| 5, 6 imagination, aims | 2.7 and 2.9 | the value functions of questions are innate, not acquired |
| 7, 8 operating self, development | 2.10 | only six operator beliefs; richer control (focus, horizon) is not modelled |
| 9, 9b imitation, detaching | provenance "imitated" for unparsed chunks; trust per source | no act imitation |
| 10–12 any sense, acting, body | events with channels; the budget of 2.11 is a body signal | no act channel beyond "say" and "ask"; no interoceptive channel |
| 13 nothing as an algorithm | six innate operators, all domain-free (recruit, refine, generalise, imagine, ask, consolidate) | the parser of v0 is an algorithm |
| 14 learned into the core | knowledge lives in units; the ledger is a short window whose rows are absorbed | exact retraction and absorption pull in opposite directions; W sets the trade |
| 15 no special domain | random vocabularies in section 5 | – |

Differences from the lead design that the lead may want to take or reject:

- **Support.** As written in §4 of the lead design, inheritance adds λ·e(i→g)·W(g→v) to an instance, and g's
  evidence includes that instance's own. With one instance the instance receives its own evidence back, which is
  the loop NARS forbids. (The lead's §6 already uses a leave-path-out rule for imagined conclusions; applied to §4
  it would be the cavity message of 2.6.) With the cavity the lift comes from an explicit prior. In this
  substrate's own equations, removing the cavity gives red 0.858 instead of 0.802.
- **First hearing.** The lead uses c = W/(W + k); with trust 0.3 for both hearings that gives 0.23, then 0.375.
  Here log-odds add (0.23, then 0.57). The two differ from the second hearing on; which is better calibrated is an
  empirical question that H3 can settle if both are implemented.
- **Questions.** The lead has no question procedure (asking is an imitated act reinforced by relief). Here the
  value of a question is computed. The two can be combined: the computed value as the innate drive, the form of
  the act acquired by imitation.

Recommendation, as the architect of this substrate. If one substrate is built first, I would not choose this one
for phase 1: its distinctive part, the field, is idle in a clean symbolic world. I would carry four things from it
into whichever substrate is built: the cavity message, the separation of told from constructed evidence with a
grounded share on every answer, licences for constructions, and the operator-level filter with its own cavity.
H8 (corrupted and inflected tokens) is the test that decides whether the field itself is added later.

## 10. Limits, and negative findings of this session

- Nothing was tested beyond `pf_trace.py`. The prototype may fail at the first gate.
- The equations do not reproduce the owner's last two numbers (0.74 and 0.80 against 0.77 and 0.88).
- 0.57 is fitted, not predicted. A count-based calculus with the same first number gives 0.37–0.41 for the second.
- The question ranking is fragile (flips at α_cat = 0.82), and its follow-up term is negligible (0.009 nats).
- Removing the cavity changes little in the smallest example (0.06); the claim that it matters at scale is untested.
- In the clean symbolic world the field settles in one sweep to one-hot choices: its continuous dynamics are idle.
  Only part F (inflected forms) shows them doing work, and one of six cases there is wrong.
- The parser is hand-written; language at scale is the weakest part, and I expect to lose to a Transformer on any
  fluency measure.
- The field learns no feature hierarchy. Codes come from the channel adapter (character trigrams for text). A
  channel richer than text would need a front end that this proposal does not provide.
- What `pf_trace.py` exercises: recruitment of concepts, units and modes by energy comparison; REFINE with the
  variation concept; GENERALISE (T2); cavity messages; confirm and enumerate questions; imagination with virtual
  members; the operator-level filter for p_H with both cavity rules; retraction with licences; order invariance;
  trigram codes. What is specified but not exercised by any code: the iterative z-update and the Kalman move of
  codes; trigger T1 and latent categories; re-estimation of θ; fill questions; the calibration map; learning of
  source trust, of the vigilances, of α_cat and θ_ask; absorption, merge and the budget; loopy graphs; per-source
  saturation; the volatility discount; the idle cycle.
- Web access: `WebFetch` could not resolve arxiv.org in this session; papers were fetched with curl from arXiv
  hosts only. Works that are not on arXiv (Kemp et al., SUSTAIN, ART, Smith et al., Friston et al. 2017, NARS)
  rest on search snippets or on the other agents' notes, as tagged.

## Sources

Read in the cited parts ([V]):
- [BayesPCN: A Continually Learnable Predictive Coding Associative Memory](https://arxiv.org/abs/2205.09930)
- [The Art of Not Forgetting (CMP)](https://arxiv.org/abs/2607.17944) and [its later version](https://arxiv.org/abs/2607.26523)
- [Closed-form predictive coding via hierarchical Gaussian filters](https://arxiv.org/abs/2605.20293)
- [AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models](https://arxiv.org/abs/2505.24784)
- [A Sparse Quantized Hopfield Network for Online-Continual Memory](https://arxiv.org/abs/2307.15040)

Abstract read on arXiv ([A]):
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
- [Nested Learning: The Illusion of Deep Learning Architectures](https://arxiv.org/abs/2512.24695)
- [A Stable, Fast, and Fully Automatic Learning Algorithm for Predictive Coding Networks](https://arxiv.org/abs/2212.00720)
- [Equilibrium Propagation with Continual Weight Updates](https://arxiv.org/abs/2005.04168)
- [Equilibrium Propagation](https://arxiv.org/abs/1602.05179)
- [From pixels to planning: scale-free active inference](https://arxiv.org/abs/2407.20292)
- [Thousand-Brains Systems](https://arxiv.org/abs/2507.04494)
- [Bayesian continual learning and forgetting in neural networks (MESU)](https://arxiv.org/abs/2504.13569)
- [Lifelong Neural Predictive Coding](https://arxiv.org/abs/1905.10696)
- [Energy Transformer](https://arxiv.org/abs/2302.07253)
- [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)
- [Continuous online sequence learning with an unsupervised neural network model (HTM)](https://arxiv.org/abs/1512.05463)
- [Test-Time Training on Nearest Neighbors for Large Language Models](https://arxiv.org/abs/2305.18466)
- [The Surprising Effectiveness of Test-Time Training for Few-Shot Learning](https://arxiv.org/abs/2411.07279)
- [Transformers Can Do Bayesian Inference](https://arxiv.org/abs/2112.10510)

Search result or snippet only ([A]):
- [An active inference approach to modeling structure learning: concept learning as an example case](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2020.00041/pdf)
- [Active inference, curiosity and insight](https://www.pubmed.gov/28777724/)
- [Learning overhypotheses with hierarchical Bayesian models](https://sites.socsci.uci.edu/~lpearl/courses/readings/KempPerforsTenenbaum2007_LearningOverhypotheses.pdf)
- [NARS: uncertainty and evidence (course notes)](https://cis.temple.edu/~pwang/AGI-course/04-Uncertainty.html)
- [SUSTAIN: a network model of category learning](https://groups.psych.northwestern.edu/medin/publications/Love,%20Medin,%20Gureckis%202004%20sustain.pdf)
- [A Fast Incremental Gaussian Mixture Model](https://pmc.ncbi.nlm.nih.gov/articles/PMC4596621/)
- [Adaptive resonance theory (overview)](https://adaptiveresonancelib.readthedocs.io/en/latest/what_is_art.html)
- [Towards a computational phenomenology of mental action](https://pmc.ncbi.nlm.nih.gov/articles/PMC8396119)
- [Strategy selection as rational metareasoning](https://cocosci.princeton.edu/falk/Strategy%20selection%20as%20rational%20metareasoning.pdf)
- [Question Asking as Program Generation](https://papers.nips.cc/paper/6705-question-asking-as-program-generation)
- [Subjective signal strength distinguishes reality from imagination](https://www.nature.com/articles/s41467-023-37322-1.pdf)
- [The generalized Hierarchical Gaussian Filter](https://arxiv.org/html/2305.10937v3)
- [Reverse Bayesianism (Karni & Vierø)](https://econpapers.repec.org/RePEc:qed:wpaper:1258)
- [Coherence, Belief Expansion and Bayesian Networks (Bovens & Hartmann)](https://arxiv.org/pdf/cs/0003041)
- [The computational nature of memory modification (latent causes)](https://elifesciences.org/articles/23763)
- [Concept Formation During Interactive Theory Revision (MOBAL)](https://link.springer.com/article/10.1023/A:1022674116380)

Files of this proposal (all under `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/predictive-field/`):
`proposal.md`, `pf_trace.py`, `trace_output.txt`, `trace_numbers.json`, `tools/` (fetch and read helpers),
`sources/` (fetched arXiv pages and papers).
