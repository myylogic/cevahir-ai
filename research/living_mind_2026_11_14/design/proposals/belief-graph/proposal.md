# Substrate A — the belief graph

A growing graph of concept units whose links carry evidence, operated by one loop in which perceiving,
learning, constructing and asking are the same process.

**Türkçe özet (12 satır)**

1. Öneri: çekirdek sabit bir ağ değil, büyüyen bir kavram çizgesi ve onu işleten tek döngüdür: duy → tahmin et → kanıtı yaz → yeniden çöz → kavram kur → hayal et → sor.
2. Ağırlık = dereceli inanç: her bağ (olumlu kanıt, olumsuz kanıt) taşır; sıklık f = w⁺/w, güven c = w/(w+2). Hiçbir şey silinmez; kanıt eklenir ve her yazı tam olarak geri alınabilir.
3. Öğrenme ile çalışma tek süreçtir: geri yayılım, optimizer, eğit/dondur yoktur; tek birincil durum kanıt defteridir, gerisi ondan kapalı biçimli tensör işlemleriyle türetilir.
4. Kavram gerektiğinde kurulur: aynı öznede birbirini dışlayan iki özellik (koyu/açık) yeni bir birim açar ("bazen koyu, bazen açık"); deneyimli zihin bu birimi söylenmeden, öngörerek açar.
5. Kategori kuralı örneklerden yukarı toplanır ve örneğe destek olarak geri verilir; örneğin kendi kanıtı kendine dönmez (biri-dışarıda), böylece inanç kendini şişirmez.
6. Hayal: doğrudan kanıtı olmayan inanç "kurgu"dur; deftere yazılmaz, günlüğe yazılır, dünya konuşunca puanlanır; ölçülen gerçeklik oranı bir sonraki kurgunun ön güvenidir.
7. İşleten süreç de öğrenir: kategori içinde ne kadar genelleyeceği (k), bir konudan ötekine ne kadar taşıyacağı (K_S), kendi kurgularına ve kaynaklarına ne kadar güveneceği deneyimle güncellenen inançlardır.
8. Kırmızı örneği (sayılar betikten): ilk söz c = 0,20 → incelmiş inanç 0,33 → kategori kuralı: hiç yaşamamış zihinde 0,03, üç alan yaşamışta 0,37, altı alan yaşamışta 0,51 → kırmızıya geri besleme: 0,33 / 0,47 / 0,56.
9. "Başka hangi renkler var?" sorusu yeni doğmuş zihinde doğmaz (kazanç 0,002 < 0,03); üç alan yaşamışta doğar ama ikinci sıradadır (0,033); altı alan yaşamışta birinci sorudur (0,049 > 0,033).
10. Olumsuz bulgular: örnekte örneklerden yeni kategori doğmadı (ölçüt ya yanlış bölüyor ya hiç bölmüyor); aktarım pilotunda kazanç küçük (saf dünyada ideal gözlemciye olan açığın %17'si, karışık dünyada ölçülemez); ilk formül terk edildi.
11. Önceki sanat: parçaların hepsi var (NARS, hiyerarşik Bayes/üst-hipotezler, ART, PLN, öznel mantık, LNN, NELL). Hiçbir şeye "yeni" demiyorum; tek aday, NARS'ın sabit k'sinin bağlam başına öğrenilen inanç olmasıdır ve tam metin denetimi ister.
12. Karar deneyi (yaklaşık 3,5 çekirdek-saat): eşit parametreli Transformer (standart eğitim), aynısı + test-zamanı ince ayar, 10 kat büyüğü ve ideal gözlemci; H1–H3 tutmazsa bu çekirdek kurulmamalı. En büyük risk dildir.

**Status.** A design proposal with executable reference equations. `bg_core.py` (numpy, about 600 lines) runs every
equation of section 2; the numbers of section 3 are printed by `trace_red.py`, not typed by hand. Four small checks
were run (comparison with exact hierarchical Bayes, a dose-response of the red episode, an internal transfer pilot,
a continuation of the episode to see whether a category would form).
**No experiment against a Transformer has been run.** Nothing below is a result about being better than anything.
Two hosts needed for full-text literature checks (arxiv.org, cis.temple.edu) were not reachable from this session;
the prior-art section therefore rests on search-result summaries and is weaker than it should be (section 6).

---

## 0. Verdict first

Substrate A satisfies P1 (one process) and P2 (graded, revisable beliefs) natively and exactly. P3 it satisfies in
part: symbols and refined concepts are formed when needed, but in the final equations no category emerged from
instances. P4 (constructions that are measured) holds in a narrow but real sense. P5 (the operating process improves)
holds only parametrically: the mind learns *how strongly* to generalise, to carry over and to trust, and it applies
an acquired lens to new subjects, but it does not invent new kinds of inference. P6 is the weak point: this family
of systems has a long record of not scaling to open language.

Every mechanism has prior art (NARS, hierarchical Bayes with overhypotheses, ART, PLN, subjective logic). I claim
nothing as new. What the substrate offers is a specific, testable combination, and three properties a trained
Transformer does not have by construction: every experience changes the state at once and can be taken back exactly;
every belief carries how much evidence stands behind it; and the state says what it does not know.

The worked example gives a mixed picture, reported as it came out (section 3): a mind that has lived nothing
stores the first statement weakly (c = 0.20) and refines it without deleting anything (0.33), but its rule about
the category is negligible (0.03), nothing is fed back to the instance, and the question about the other colours
does not arise. After three lived subjects the rule is 0.37, red is lifted to 0.47, and the question arises as the
second candidate; after six, 0.51 and 0.56, and it is the first question. The closed forms agree with exact
hierarchical Bayes to 0.016 on average. Against that: in the final version **no category emerged from instances**,
the internal transfer pilot showed a **small** effect (17% of the gap to an ideal observer closed in one world,
nothing measurable in the other), and the first formulation, which gave the nicest numbers, had to be abandoned.

My recommendation as an architect: build it only as the belief-and-memory core of the new engine, behind the decisive
experiment of section 5, and with the language risk stated to the owner in plain words.

---

## 1. The idea in five sentences

1. The whole primary state is a **ledger of direct evidence**: each experience appends one line `(source, kind,
   subject, object, polarity, amount)`, and every link of the graph is the pair (positive evidence, negative evidence)
   obtained by summing its lines, each weighted by the current trust in its source.
2. Everything else — category rules, how strongly members follow their category, what a rule lends to a member that
   was never observed, how real the mind's own constructions have been — is a **closed-form function of the ledger**,
   computed by a handful of tensor operations; there is no optimiser, no gradient, no train/freeze split.
3. A **new unit is allocated when the evidence cannot be explained** by the existing ones: a refined concept when one
   subject carries two properties believed to exclude each other, a sub-category when the members of a category are
   better explained as two groups than as one.
4. A belief held without direct evidence is a **construction**; it is kept out of the ledger, journalled with its
   predicted value and predicted realism, and scored when the world speaks; the measured share of constructions that
   came true becomes the prior confidence of the next construction.
5. The quantities that govern the loop itself — how strongly to generalise inside a category, how strongly to carry a
   regularity from one subject to another, how much to trust each source and the mind's own projections — are
   themselves **beliefs updated by the same evidence**, so the way of constructing improves with experience and
   transfers to new subjects.

| Part | Serves | How |
|---|---|---|
| Ledger; beliefs are functions of it | P1, P6 | one write per experience is the whole learning step; exact retraction |
| Evidence pair, frequency, confidence | P2 | weak first storage, revision by addition, "sometimes" = stable middle frequency |
| Trust-weighted evidence, per source | P2, P5 | a statement is worth what its source has proved to be worth |
| Refined lens (MIX unit), sub-category | P3 | allocated on unexplained evidence; a sub-category is meant to form from its instances (it did not, 3.4) |
| Votes upward, hand-down, leave-one-out | P2 | the category rule, and the support it feeds back without self-support |
| Imagination journal, realism `v` | P4 | constructions are measured against later experience; realism gates later constructions |
| Question gains | P2 | questions arise from an explicit objective, not from a template |
| Learned couplings `k`, `K_S`, `v`, trust; anticipated lens | P5 | the operating process is made of beliefs; a lived way of constructing is applied elsewhere |
| Tensors, closed forms, tiny language | P6 | implementable; language enters as statements (section 4) |

---

## 2. The formal model

Everything in this section is implemented in `bg_core.py` under the same names. Constants are listed in 2.10.

### 2.1 State

| Object | Shape | Meaning |
|---|---|---|
| units | `N` (grows) | one index per concept; relations, categories, lenses and sub-categories are units too |
| ledger | `T` lines | `(t, q, kind, s, o, y, amt, event)`; `kind ∈ {ISA, HAS, EXC}`, `y ∈ [0,1]`, append-only |
| trust | `Q × 2` | per source `q`: agreements `A_q`, disagreements `D_q`; `v_q = (1 + A_q) / (2 + A_q + D_q)` |
| evidence | `W⁺, W⁻ : 3 × N × N` (sparse) | `W⁺[kind,s,o] = Σ_lines v_q · amt · y`, `W⁻ = Σ_lines v_q · amt · (1 − y)` |
| lenses | set | `MIX(o,o′)` units over pairs of properties believed exclusive |
| sub-categories | set | `(parent category, lens) → unit`, with assigned members |
| journal | list | resolved constructions (reporting only; never read by inference) |

`W±` is one `einsum` over the source mode: `W± = Σ_q v_q · D±_q`, where `D±_q` are the raw per-source counts. When
the trust in a source changes, everything it ever said is re-weighted at once, without touching the ledger.

Derived, recomputed, never stored as truth: exclusion strengths `X`, membership weights `m`, contexts, couplings,
responsibilities, rule readouts, question gains.

### 2.2 Truth values and the two evidence operations

For any link with evidence `(w⁺, w⁻)`, `w = w⁺ + w⁻`:

    frequency    f = w⁺ / w
    confidence   c = w / (w + k_ref)                       k_ref = 2
    expectation  e = (w⁺ + P⁺) / (w + P)                   (P⁺, P) = prior pseudo-counts; (1, 2) if nothing is known

`f` says what the belief says, `c` how much evidence stands behind it; "sometimes" is a middle `f` with a high `c`,
"I do not know" is a low `c`. With the uniform prior `(1, 2)` the expectation is Laplace's rule and equals the NARS
expectation `c (f − ½) + ½` at `k = 2`.

Two operations move evidence through the graph:

    vote(n, k)     = (n ∥ k) · (k + 1) / k,   n ∥ k = n k / (n + k)     upward: what a member with n units says about its category mean
    handdown(k, E) = k (E + 2) / (E + k + 3)                             downward: prior strength a mean known with E pseudo-votes lends to one member

`k` is the **coupling** of a context: how closely members follow the category mean. Limits: `k → ∞` gives `vote = n`
(every observation of any member is an observation of the category) and `handdown = E + 2` (pooled counts with
Laplace); `k → 0` gives `vote = 1` (one member, one vote, however often it was seen) and `handdown = 0` (a member's
first observation is believed outright: members are "pure"). At `E = 0`, `k → ∞` it is the uniform prior.
Both are the moment-matched messages of a Beta-binomial hierarchy `θ_member ~ Beta(k a, k(1 − a))`,
`a ~ Uniform` (derivation in 2.12).

### 2.3 Lenses: the evidence of a subject under a property unit

    plain property p:   w⁺_p(s) = W⁺[HAS,s,p]
                        w⁻_p(s) = W⁻[HAS,s,p] + Σ_o′ X[p,o′] · W⁺[HAS,s,o′]          (what excludes p counts against p)
    lens u = MIX(o,o′): a = W⁺[HAS,s,o], b = W⁺[HAS,s,o′]
                        w⁺_u(s) = 2 · min(a, b)                                       (both poles occur)
                        w⁻_u(s) = max(0, |a − b| − w_free) + W⁻[HAS,s,o] + W⁻[HAS,s,o′]   (one-sided beyond the first unit, or told "never")

    X[o,o′]  = w⁺_EXC / (w_EXC + k_ref)                 exclusion strength (zero base rate), symmetric
    m[s,c]   = e_ISA(s,c) if e_ISA(s,c) > ½ else 0      membership weight

A lens is not new evidence. It is a second reading of the same ledger lines, in which two statements that conflict
under the old reading ("dark", "light") support one statement under the new one ("sometimes dark, sometimes
light"). A query is always answered through one lens, so the two readings are never added.

### 2.4 Contexts, the category rule and the support fed back

A **context** `g = (c, p)` is a category seen through one lens; its members are the units with `m[s,c] > 0`.
For member `s`, grid value `k_j` (`J = 8` values), prior on the context mean `(E0, A0)` from 2.6:

    V_js = r_s · vote(w_s, k_j)                                   vote of member s (r_s: responsibility, below)
    E_js = E0 + Σ_{i≠s} V_ji                                      evidence about the mean WITHOUT s
    a_js = (E0·A0 + Σ_{i≠s} V_ji f_i) / E_js                      the mean WITHOUT s
    ā_js = (E_js a_js + 1) / (E_js + 2)                           Laplace on the mean
    κ_js = handdown(k_j, E_js)                                    prior strength handed to s

    in :  θ_s ~ Beta(κ_js ā_js, κ_js (1 − ā_js))                  s follows its category
    out:  θ_s ~ Beta(1, 1)                                        s is not governed by this rule
    ρ_s  = v_σ · m_s                                              prior that s is a member AND conforms
    L_g(j) = Π_s [ ρ_s · B(κā + w⁺_s, κ(1−ā) + w⁻_s) / B(κā, κ(1−ā)) + (1 − ρ_s) · B(1 + w⁺_s, 1 + w⁻_s) ]
    r_js = responsibility of the first term;   r_s = Σ_j π_g(j) r_js

The sums exclude `s` itself (**leave-one-out**): a member never receives its own evidence back, which is what keeps
a rule from confirming the instance it was built from. Shapes: `V, E, a, κ, r : J × M_g`. The coupling of the
context is a belief on the grid:

    q_g(j) ∝ π0(j) L_g(j)                                                      what this context alone says
    ψ_g(j) = (ν0 π0(j) + Σ_{g′: same kind, other category} q_g′(j)) / (ν0 + n)   what sibling contexts say (2.6)
    π_g(j) ∝ ψ_g(j) L_g(j)

Readouts (all evidence in pseudo-counts; `κ⁰_j = handdown(k_j, 0)` is what a member gets when nothing is known):

    support fed back to s      K_s  = Σ_j π_j r_js (κ_js − κ⁰_j),    K⁺_s = Σ_j π_j r_js (κ_js − κ⁰_j) ā_js
    prior pseudo-counts of s   P_s  = Σ_j π_j [ r_js κ_js + (1 − r_js)·2 ],   P⁺_s likewise with ā_js and 1
    belief of s under lens p   e = (w⁺ + P⁺)/(w + P),   f = (w⁺ + K⁺)/(w + K),   c = (w + K)/(w + K + k_ref)
    the rule (c → p)           a_rule = Σ_j π_j (E0·A0 + S1_j + 1)/(E0 + S0_j + 2),   S0_j = Σ_s V_js, S1_j = Σ_s V_js f_s
    what the rule lends to a   K_new = Σ_j π_j v_σ (handdown(k_j, E0 + S0_j) − κ⁰_j),   c_rule = K_new/(K_new + k_ref)
      new, certain member

`c_rule` is the single number I report as "the weight of the category rule": the confidence it would lend to a
member about which nothing else is known. Feeding support back to an instance is the term `K_s`: in NARS words, a
revision of the direct belief with the belief deduced from the rule, where the deduced evidence comes only from
other members and other subjects.

### 2.5 Perceiving one statement

`perceive(kind, s, o, y, q)` is the only learning step:

1. **Address.** Map `s`, `o` to units; allocate a unit for an unknown symbol.
2. **Predict.** `(e, f, c) = answer(kind, s, o)`; surprise `= −[y ln e + (1 − y) ln(1 − e)]`.
3. **Judge the source.** Recompute the belief without any line of `q` (`W± − v_q D±_q`), giving `(f_ind, c_ind)`;
   then `A_q += c_ind (1 − |y − f_ind|)`, `D_q += c_ind |y − f_ind|`. A source is judged only against evidence that did
   not come from it; with no independent evidence its trust does not move.
4. **Write.** Append one ledger line with `amt = ρ_rep^r`, `r` = number of earlier identical generic statements by the
   same source (observation events are never discounted). This is the weight change; it takes effect at once.
5. **Re-solve** the affected contexts (2.4, 2.6); **allocate** if evidence is unexplained (2.7); **imagine** (2.8);
   update the **question** agenda (2.9).

### 2.6 Improving the operating process

Four families of quantities govern the loop. None is a constant, none is trained by gradient; each is a belief with
a closed-form update.

**(a) Coupling of a context, `π_g` over the grid** (2.4): how strongly to generalise inside this category.
A new context starts from `ψ_g`, the pooled posterior of contexts of the same kind in *other* categories. Kinds are
structural: `MIX` (a lens), `pole` (a property that has an exclusive partner), `plain`.

**(b) Carry-over between subjects.** For kinds whose lens means the same thing everywhere (`MIX`, `pole`), the mean
of a context gets a prior from the means of the same kind in other categories, with the same two operations one
level up:

    vote_S(g′) = vote(S̄0_g′, K_S)                         S̄0 = Σ_j π_j S0_j
    rest_g     = Σ_{g′: same kind, other category} vote_S(g′)
    E0_g(K_S)  = handdown(K_S, rest_g) − handdown(K_S, 0)
    A0_g(K_S)  = (Σ vote_S(g′) ā_g′ + 1) / (rest_g + 2)
    score(K_S) = Σ_g ln Σ_j ψ_g(j) L_g(j | E0_g(K_S), A0_g(K_S))
    ω(K_S) ∝ π0(K_S) exp score(K_S);     E0_g = Σ ω E0_g(K_S),   A0_g = evidence-weighted mean

`K_S` is how strongly a regularity lived in some subjects carries to a new one. It is scored by how well it predicts
the members of every context it touches, each left out in turn. With `E0 > 0` a rule exists for a category after a
single member, and that member receives support from what was lived elsewhere.

**(c) Realism of the mind's own projections, `v_σ`** per kind: the share of informed members that turned out to
conform to their category rule,

    v_σ = (1 + Σ_{informed (s,g)} inl_s) / (2 + #informed),     inl_s = r_s / (r_s + exc_s),
    exc_s = (1 − r_s) m_s (1 − v_σ) / (1 − v_σ m_s).

It enters `ρ_s = v_σ m_s`, so the measured realism of past constructions is the prior confidence of the next one.

**(d) Trust in each source, `v_q`** (2.5, step 3). The mind's own projection channel is treated the same way as an
informant: it is believed in proportion to its record.

A fifth piece is a rule, not a number: **anticipation**. Once a lens of some kind has proved informative in a lived
context, the same lens is allocated for a new exclusive pair as soon as the pair appears in a category, before any
conflict has been observed (2.7). This is the one place where an acquired *way of constructing* is applied to a new
subject, and it is what makes an experienced mind imagine the refinement before being told.

### 2.7 Forming a concept and a category

**A refined concept (lens).** For a pair `(o, o′)` with `X[o,o′] > 0` and no lens yet:

    conflict mass  T = X[o,o′] · max( max_s min(W⁺[s,o], W⁺[s,o′]),  max_c min(Σ_s m_sc W⁺[s,o], Σ_s m_sc W⁺[s,o′]) )
    allocate MIX(o,o′) if T ≥ θ_alloc                          (one subject, or the members of one category, carry both)
    or by anticipation:  a lived MIX context has an informed member,  X[o,o′] ≥ ½,  and the pair occurs in a category

The unit is defined by its lens (2.3); it stores nothing of its own.

**A category from instances (split).** For a context with at least four informed members, order them by frequency,
cut at the largest gap into groups `A`, `B` (each at least two) and compare

    one = pml(A ∪ B),    two = pml(A) + pml(B) + ln[ Γ(|A|) Γ(|B|) / Γ(|A| + |B|) ]
    split if  two − one > ln 10 + ln(n − 1)

`pml` is the pseudo-marginal likelihood of a group as one cluster: the product of leave-one-out predictives of 2.4
with the coupling integrated over the grid. The second term is the Chinese-restaurant prior of the partition, the
threshold is "strong evidence" corrected for having chosen the cut among `n − 1`. The smaller group becomes a
sub-category unit with its own context, rule and coupling; its members leave the parent's context for that lens.
A later informed member joins the sub-category when its predictive ratio (sub-category against parent) exceeds
`ln 10`. A member not yet informed belongs to both, in proportion to their sizes, so what is imagined about it is a
mixture.

### 2.8 Answering, imagining and scoring an imagination

**Answering** `answer(HAS, s, p)`: the belief of `s` under lens `p` from 2.4, with the prior pseudo-counts of every
context that holds `s` (parent and sub-category weights add). `ISA` and `EXC` are answered from their own evidence
with the uniform prior. The output is `(e, f, c)` and, in words, a frequency word from `f` and a hedge from `c`
(section 4).

**Imagining.** Every `(s, g)` with no direct evidence under the lens (`w_s = 0`) and support fed back (`K_s > 0`)
is a construction. On creation the journal records its expectation `e`, its confidence `c`, and its **predicted
realism** `R = inl_s` (the prior that the construction will prove real, i.e. `v_σ` adjusted by membership).
It is never written to the ledger and never counted as evidence for anything.

**Observing one's own output from outside.** In this substrate the phrase has a modest, concrete meaning and no
more. What the mind has constructed is available to it only through the same `answer` path by which it would judge
a stranger's statement, with the confidence that the evidence behind it deserves; and the channel that produced it
("my own projection of kind σ") is treated like an informant with a record, `v_σ`. There is no second observer.

**Scoring.** When direct evidence under the lens reaches `θ_res`, the construction is resolved: the journal gets the
world's frequency `f_world`, the squared error `(e − f_world)²`, and the **realism after the fact** `r_post = inl_s`
(0 if the member had to be moved out of the rule). Resolution changes `v_σ` (2.6c), the coupling `π_g` and the
carry-over `K_S`; that is how the mind advances by the measured reality of what it constructed. A construction that
proves wrong is not deleted and not forbidden: it lowers `v_σ` a little, and if enough of them cohere they become a
new category (2.7).

### 2.9 Generating a question

The mind's objective for a category `c` is its expected confidence about the next statement it will have to deal
with. The next statement concerns a known member `i` with probability `P(i)`, an unseen one with `P_unseen`:

    P_unseen(c) = γ / (n_c + γ)                       n_c = mentions of members of c;  Chinese-restaurant estimate
    P(i)        = (1 − P_unseen) n_i / n_c

    "tell me more about i under lens p":   gain = P(i) · [ c_i(w + δ) − c_i(w) ],     c_i(w) = (w + K_i)/(w + K_i + k_ref)
    "which other c are there?":            gain = P_unseen · φ · K_proj/(K_proj + k_ref)
                                           K_proj = Σ_j π_j v_σ m_new (handdown(k_j, E0 + S0_j) − κ⁰_j)

`δ` is the evidence of one more statement from the current source, `m_new` the membership it would establish, `φ`
the share of the unseen mass carried by the first member revealed. The objective is read through the most refined
lens available. A question **arises** when its gain exceeds `θ_Q`. The second gain is the point of the construction:
knowing that something is a colour is worth exactly as much as the colour rule can lend to it. Before a rule exists,
"which other colours are there?" has no value; when the rule is strong, it has.

### 2.10 Constants, and what is learned

| Constant | Value | Role | Learned? |
|---|---|---|---|
| `k_ref` | 2 | readout of confidence; uniform prior | no (a unit convention) |
| grid | 0.25 … 32 (8 values, ×2) | support of every coupling belief | no; the weights on it are learned |
| `π0`, `ν0` | uniform, 1 | prior over the grid, weight of that prior in pooling | no |
| `w_free` | 1 | one-sided evidence that says nothing about mixedness | no (candidate for learning) |
| `θ_alloc` | 0.25 | conflict mass that opens a lens | no (rate controller in the build, section 7) |
| `θ_res` | 0.75 | evidence at which a construction is resolved | no |
| split threshold | `ln 10 + ln(n − 1)` | evidence ratio for a new category | no |
| `ρ_rep` | 0.5 | a source repeating itself | no |
| `γ`, `φ` | 1, 0.5 | unseen-member estimate | no in v0 (should be learned from lived category sizes) |
| `θ_Q` | 0.03 | gain at which a question is uttered | no |
| couplings `π_g`, carry-over `ω(K_S)`, realism `v_σ`, trust `v_q` | — | the operating process | **yes, from the ledger** |

All constants were fixed before the trace was read, with the exceptions recorded in the design log of section 3.4.

### 2.11 Shapes, complexity, memory

Let `N` units, `L` non-zero links, `Q` sources, `C` contexts, `M_g` members of context `g`, `J = 8`.

- **Write:** `O(1)` (one ledger line, one `index_add` into `W±`).
- **Re-solve after a statement about `(s, p)`:** only the contexts of the categories of `s` under the lenses of `p`
  change their own sums: `O(J · M_g)` each for the leave-one-out terms (one batched `lgamma` over a `J × M_g`
  tensor). The schema level touches the contexts of the same kind: `O(J · J · Σ M_g)` if recomputed in full, to be
  amortised (recompute when a context mean moved by more than a tolerance). Nothing depends on `N`.
- **Answer:** `O(J)` per context holding the subject, from cached `S0, S1, π`.
- **Reference script:** recomputes everything from the ledger at every statement (`O(state)`), six sweeps. That is
  what makes it a specification rather than an engine. On the trace it costs about 0.3 s per statement (numpy, one
  core, a state of 150 ledger lines).
- **Memory:** `2L` floats of evidence (`× Q̄` if per-source slices are kept), `2 C J` cached sums, `C J` coupling
  weights, the ledger. The ledger compacts to per-`(source, link)` counts; the order of lines is needed only for
  replay.
- **Growth:** `L ≤ T`; units `≤` symbols `+` exclusive pairs `+` contexts; one lens per exclusive pair, one
  sub-category per (category, lens) in v0.
- **Bound:** hard caps `N_max, L_max`. When a cap is reached, the instance links of the member that its category
  rule predicts best are folded into the context's sums (`S0, S1` keep their votes as "ghost votes") and dropped.
  The category rule is unchanged by the eviction; what is lost is the detail of a member that was typical anyway.
  **This eviction is specified, not implemented, and not tested** (section 7).

### 2.12 What these equations are

They are an online, closed-form approximation of a hierarchical Beta-binomial model with outliers:

    a_g ~ prior from the schema (uniform if none);   k_g ~ grid;   θ_s ~ Beta(k_g a_g, k_g (1 − a_g));   evidence ~ Binomial(θ_s)

`vote` follows from the variance of a member's observed frequency given the mean,
`Var(f_s | a) = a(1−a)(n+k) / (n(k+1))`, so one member contributes `n(k+1)/(n+k)` pseudo-counts about `a`.
`handdown` follows from matching the variance of `θ_s` when `a` is known only with `E + 2` pseudo-counts:
`1/(κ+1) = (1 − 1/(E+3))/(k+1) + 1/(E+3)`. Both are exact in the limits listed in 2.2.

**Measured gap to exact hierarchical Bayes** (`check_exact_hbm.py`; reference by numerical integration over the mean
on 4,001 points and the same grid; 144 random scenarios, 816 queries, 2 to 8 members with 1 to 6 observations each,
four regimes): the mean absolute error of the expectation is **0.016** (maximum 0.153). For comparison, each member's
own counts with Laplace's rule: 0.099; pooled counts: 0.143. By regime, for members with data and for a new member:
mixed 0.018 and 0.012, skewed 0.015 and 0.021, pure 0.010 and 0.018, heterogeneous 0.020 and 0.022. The reference
has certain memberships and no outlier component; the model was run with both set to match. With sub-categories
allowed (final threshold) the numbers are identical, because no split occurred in any scenario.

**Relation to the named systems** (details and sources in section 6):

- *NARS.* The evidence pair, `f`, `c = w/(w+k)`, revision by adding evidence, and the expectation are NARS's. NARS
  induction lets one instance contribute at most one unit of evidence to a general statement (a product of
  confidences of the form `n/(n+k)`): that is the shape of `vote(n, k)` at `k = 1`, up to the factor `(k+1)/k`.
  (Stated from the NARS literature as I know it; the formulas could not be re-read in this session.) The difference
  is that NARS's `k` is one global constant of the system ("evidential horizon", usually 1), while here the coupling
  is a belief per context, learned, and pooled across subjects; and that overlapping evidence is excluded by
  leave-one-out algebra instead of evidential-base stamps. NARS has far more than this substrate: compound terms,
  temporal and procedural inference, self-monitoring, budgeted attention.
- *Hierarchical Bayes, overhypotheses.* The coupling `k` is exactly the `α` of Kemp, Perfors and Tenenbaum (2007),
  the pooled mean their `β`, and the owner's example is their bags of marbles ("all bags are mixed" against "each bag
  is one colour"). What is added is only that the inference is online, closed-form and retractable.
- *ART.* Allocation on mismatch instead of overwriting is ART's answer to the stability–plasticity dilemma. Here the
  mismatch test is a Bayesian evidence ratio and learning inside a category is evidence addition, not weight decay.
- *PLN.* PLN reasons with strength and confidence over a hypergraph and has a differentiable variant with tensor
  truth values. PLN uses term probabilities and a much larger rule set; this substrate uses neither.

### 2.13 What a learned, a differentiable and a self-modifying version add

- **Learned (this proposal).** With the operating process frozen the system is a fixed-constant evidence logic: it
  cannot change how strongly it generalises, and nothing lived in one subject changes how the next is met.
  Section 3.3 shows the difference on the red episode (frozen: identical to a mind that has lived nothing).
- **Differentiable (an ablation, not the design).** Every readout is differentiable in the continuous parameters
  (`ln k`, `logit v`, `w_free`, thresholds), so one can take an online gradient step on the prequential log-loss per
  statement. It would tune shapes the grid cannot reach. It also brings a step size, optimiser state and a state that
  is no longer a function of the ledger, so exact retraction is lost. I propose it as arm `A-grad` of the
  experiment, not as the core. The batch version exists for PLN (Potapov et al. 2019).
- **Self-modifying (partly here).** The set of units, contexts and schema nodes — which messages exist at all —
  grows with experience, and an acquired lens is applied to new subjects by anticipation. What is absent is the
  invention of new *kinds* of lens or rule. A research extension would enumerate a tiny language of evidence
  re-codings (`min`, `|difference|`, conjunction, split by context) and keep those that reduce prequential loss; this
  is EURISKO and DreamCoder territory, it is risky, and it is not in the 2,000 lines.

---

## 3. Worked trace of the "red" example

Produced by `python3 trace_red.py` (about 95 s on one core; full output in `trace_red.out`, numbers in
`trace_red.json`). Nothing in this section was computed by hand.

### 3.1 Setting

Two minds hear exactly the same episode from a **stranger** (a source never heard before, trust 0.50).

- **NEWBORN** knows only that dark and light exclude each other (8 own observations; exclusion strength 0.792).
- **EXPERIENCED** has in addition lived three unrelated subjects — instruments (high/low), fruits (sweet/sour),
  animals (big/small): four members each, named by a teacher, each member observed by the mind itself twice on each
  pole. No word of these subjects recurs in the colour episode. Its state before the episode: 92 ledger lines,
  26 units, three lenses it formed itself, measured realism of its lens projections 0.904, carry-over coupling
  `K_S ≈ 4.3`.

One statement by the stranger is worth `v_q · 1 = 0.5` units of evidence.

### 3.2 The trace

`c` is the confidence (evidence behind the belief), `f` the frequency the belief asserts, `e` the expectation.
The owner's illustrative figures are in the last column; they are weights, so they are compared with `c`.

| Stage | NEWBORN | EXPERIENCED (three lived subjects) | Owner |
|---|---|---|---|
| **1. "red is a dark colour"** — first hearing | "red is dark": f = 1.00, **c = 0.200**, e = 0.613. Says: *"I have only heard that red is always dark."* By exclusion: red is light f = 0.00, c = 0.165 (*"never light"*). | Own evidence the same (**c = 0.200**). The lens `MIX(dark\|light)` is allocated **by anticipation**; the mind constructs *"red is sometimes dark, sometimes light"* (e = 0.659, c = 0.226, predicted realism 0.902) before being told. With the support of what it lived in other subjects: red is dark f = 0.696, c = 0.425 (*"usually dark"*); red is light f = 0.379 (*"sometimes"*, not "never"). | 0.23 |
| **2. "red also has light shades"** — refine, do not delete | Surprise 0.874 nats. Conflict mass 0.40 ≥ 0.25: the lens **`MIX(dark\|light)` is allocated**. "red is dark" moves to f = 0.558, c = 0.309 (*"I think red is sometimes dark"*). Under the lens: f = 1.00, **c = 0.333**, e = 0.686. All three ledger lines about red are still there. | Surprise 0.789 nats. The construction of stage 1 is **resolved**: imagined e = 0.659, world f = 1.00, squared error 0.117, realism after the fact 0.939. Own evidence under the lens: **c = 0.333**. | 0.57 |
| **3. the rule about the category** `colour → MIX(dark\|light)` | mean a = 0.565, **c_rule = 0.027** (one instance, nothing lived). | mean a = 0.823, **c_rule = 0.372**; of its 5.08 pseudo-votes, 4.42 come from the three lived subjects (prior mean 0.94). | 0.77 |
| **4. support fed back to red** | K = 0: one instance cannot confirm itself. **c stays 0.333.** | K = 0.773 pseudo-counts fed back: **c = 0.470**, e = 0.800. | 0.88 |
| **5. the question** (arises if gain ≥ 0.03) | "tell me more about red": 0.0714 — arises. "which other colours are there?": 0.0021 — **does not arise**. | "tell me more about red": 0.0465 — arises. **"which other colours are there?": 0.0328 — arises**, ranked second. | arises |
| **6. "blue is a colour", "green is a colour"** — the mind expands over the set | Constructs "blue is sometimes dark, sometimes light": e = 0.512, c = 0.017, predicted realism 0.500. About "is blue dark?": c = 0.009 (*"I do not know"*). | Constructs the same with e = 0.684, c = 0.262, predicted realism 0.905. "Is blue dark?": f = 0.541, c = 0.338 (*"I think blue is sometimes dark"*). | — |
| **7. the world speaks about blue and green** (both shades each) | Constructions resolved: squared error 0.238 each, realism after 0.567 / 0.641. Now red gets support from the other colours: c = 0.359 (K = 0.118); rule 0.110; realism `v` 0.500 → 0.585. | Resolved: squared error 0.100 each, realism after 0.942 / 0.945. Red: **c = 0.500** (K = 0.996); rule 0.422; `v` = 0.913. | — |
| **8. a wrong construction**: "black is a colour"; then the mind itself observes black six times dark, six times not light | Before evidence: *"I have only heard that black is sometimes light"* (c = 0.02). After: black is dark f = 0.995; black is light f = 0.003, c = 0.836 (*"I know black is never light"*). Journal: imagined 0.551, world 0.00, squared error 0.304. | Before evidence: *"I think black is sometimes light"* (f = 0.54, c = 0.38) — a confident error. After: black is light f = 0.015, c = 0.839. Journal: imagined 0.714, world 0.00, **squared error 0.510**. The rule is revised, not dropped: mean 0.855 → 0.754, coupling k 2.5 → 1.2, `v` 0.913 → 0.896, support to red 0.996 → 0.572. | — |
| **9. a second exception**: white | Rule mean 0.304. Split test: log evidence ratio 1.81 against the threshold 3.69: **no sub-category**. | Rule mean 0.554, coupling k ≈ 0.6, c_rule 0.192; red c = 0.379. Split test 1.81 against 3.69: **no sub-category**. | — |
| **10. the mind looks at red, blue and green itself** (4 dark, 4 light each) | Red under the lens: c = 0.812. Rule: mean 0.595, coupling k ≈ 0.3, c_rule 0.050. Split test 2.54 against 3.69: none. | Red: c = 0.812. Rule: mean 0.724, coupling k ≈ 0.3, c_rule 0.060. Split test 2.54 against 3.69: none. About a colour never described (grey) under the lens: e = 0.540, **c = 0.037**. | — |

Checks printed by the same run:

- **Retraction.** Removing the second statement by subtracting its ledger line restores the evidence tensors with a
  maximal difference of 0.0 and the belief "red is dark" with a difference of 0.0, in both minds; the lens allocated
  because of the line stays as an empty unit (5 units against 4 in the newborn). Living the life again without the
  line (`replay`) restores everything, unit inventory included, with difference 0.0.
- **Imagination never wrote to the ledger**: 68 and 152 lines at the end, all of them statements heard or observed.
- **Sources.** A friend who says what the mind has itself observed goes from trust 0.500 to 0.806 in six statements;
  a liar who says the opposite falls to 0.168; the beliefs barely move (coal is dark: f = 0.96, c = 0.73).
- **The journal of the colour episode** (12 constructions per mind). On blue and green (which conformed) the mean
  squared error is 0.034 for the experienced mind and 0.081 for the newborn. On black and white (which did not) it is
  0.324 against 0.272: a confident mind is punished harder when its construction is wrong. Overall 0.179 against
  0.176 — in this short episode, with two exceptions among five colours, experience did not make the constructions
  more accurate on average; it made them sharper in both directions.

### 3.3 Dose-response: how the same two statements land in minds that have lived more or less

`python3 dose_response.py`; readouts after stage 2, under the lens.

| Lived subjects | Refined belief, own c | Rule c_rule | Red after feedback c | Rule mean a | Carry-over `K_S` | Realism `v` | Surprise at statement 2 | Gain "which other colours?" | Gain "more about red" |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.333 | 0.027 | 0.333 | 0.565 | 2.8 | 0.500 | 0.874 | 0.0021 | 0.0714 |
| 1 | 0.333 | 0.196 | 0.384 | 0.690 | 3.0 | 0.789 | 0.872 | 0.0160 | 0.0616 |
| 2 | 0.333 | 0.290 | 0.426 | 0.765 | 3.5 | 0.870 | 0.830 | 0.0246 | 0.0540 |
| 3 | 0.333 | 0.372 | 0.470 | 0.823 | 5.5 | 0.907 | 0.789 | **0.0328** | 0.0465 |
| 6 | 0.333 | **0.514** | **0.557** | 0.904 | 16.4 | 0.950 | 0.727 | **0.0485** | 0.0332 |
| 3, `K_S` fixed at 4 | 0.333 | 0.241 | 0.399 | 0.732 | 4 | 0.904 | 0.849 | 0.0200 | 0.0588 |
| 6, `K_S` fixed at 4 | 0.333 | 0.279 | 0.416 | 0.769 | 4 | 0.948 | 0.834 | 0.0236 | 0.0557 |
| 3, `K_S` fixed at 32 | 0.333 | 0.489 | 0.542 | 0.881 | 32 | 0.908 | 0.740 | 0.0456 | 0.0353 |
| 3, operating process frozen | 0.333 | 0.027 | 0.333 | 0.565 | — | 0.500 | 0.874 | 0.0021 | 0.0714 |

Three things can be read off. The order the owner described — weak, refined, a rule stronger than the instance, the
instance lifted, then the question about the other colours — appears in full only in the mind that has lived six
subjects: there the rule (0.514) is stronger than the refined belief (0.333), the instance is lifted to 0.557, and
"which other colours are there?" is the top question (0.0485 against 0.0332). With the operating process frozen,
three lived subjects change nothing at all: every transfer effect in the table comes from the learned quantities of
2.6. And the single constant that I first fixed by hand (`K_S = 4`) capped the effect; when it is learned, the
effect grows with what was lived.

### 3.4 Design log, including what failed

The trace was the instrument with which the equations were debugged; that is a garden of forking paths and the
trace is an illustration, not evidence. In order:

1. **First formulation abandoned.** Messages `n ∥ k` in both directions and a fixed prior of strength 2 on every
   member. It produced the nicest table (0.200 → 0.333 → rule 0.472 → red 0.505; the question about other colours
   top-ranked by a factor 1.09). It was dropped before any numerical check because a learned `k` could only change
   how much a member inherits, never how sharply its own first observation is believed, so the "each bag has one
   colour" regime could not be represented at all.
2. **Moment-matched messages (2.2, 2.12), `K_S` fixed at 4.** Agreement with exact hierarchical Bayes was measured
   (2.12). The table became weaker: rule 0.241, red 0.399, question ratio 0.34.
3. **`K_S` learned.** Rule 0.372, red 0.470, ratio 0.71 (the table of 3.2).
4. **Anticipated lens** added after noticing that a lived lens could not help in a new subject until a conflict had
   been observed there.
5. **Negative pilot.** With the lens shadowing its poles and handing evidence down to them, the transfer pilot
   (3.5) showed no transfer in the mixed world and little in the pure one. The hand-down was removed; poles keep
   their own contexts, pooled across subjects like the lens.
6. **Split threshold.** The first threshold (`ln 3`) formed a sub-category {black, white} at stage 9, but in the
   comparison scenarios it also split single populations: 3 of 12 homogeneous and 6 of 12 heterogeneous
   eight-member categories. With `ln 10 + ln(n − 1)` there are no false splits in 144 scenarios — and no split in the
   trace either, not even after eleven well-observed colours (`split_continuation.py`: ratio 3.88 against a threshold
   that has grown to 4.61). A low coupling explains "most are mixed, some are pure" almost as well as two
   categories do. **In the final version no category emerged from instances in the worked example.**

### 3.5 Internal transfer pilot (not the decisive experiment)

`python3 pilot_transfer.py`: five subjects in sequence, each with its own category, four members and its own
contrast pair; four observations per member; before each observation the mind predicts which pole will be seen.
Loss in nats on observations 2–4 of every member, mean over 5 seeds. In the mixed world every member shows either
pole with probability ½ (ideal loss 0.693 throughout); in the pure world each member shows one pole only (ideal
loss about 0 once the first observation is in).

| Arm | Mixed world, subjects 1 → 5 | Pure world, subjects 1 → 5 | Pure world, first member of subjects 2–5 |
|---|---|---|---|
| FULL (everything of 2.6 learned) | 0.784 0.762 0.725 0.734 0.751 | 0.318 0.294 0.292 0.277 **0.270** | 0.299 |
| FROZEN (operating process frozen) | 0.783 0.770 0.734 0.746 0.772 | 0.321 0.327 0.323 0.320 0.327 | 0.332 |
| FIXED-K1 (one coupling `k = 1`, frozen) | 0.799 0.781 0.743 0.758 0.786 | 0.307 0.316 0.312 0.312 0.319 | 0.327 |
| FLAT (counts and exclusion, no categories) | 0.778 0.765 0.729 0.745 0.766 | 0.340 0.340 0.340 0.340 0.340 | 0.340 |
| FULL+M (FULL, stated memberships taken as certain) | 0.778 0.749 0.715 0.716 0.729 | 0.304 0.237 0.220 0.197 **0.192** | 0.247 |
| FROZEN+M | 0.776 0.767 0.733 0.740 0.769 | 0.308 0.319 0.312 0.306 0.320 | 0.325 |
| ideal observer | 0.693 | about 0 | about 0 |

In the pure world the learned process gets better from subject to subject (0.318 → 0.270) while the frozen and the
fixed-coupling arms do not move and the count-only arm stays at 0.340; with memberships taken as certain the gain is
larger (0.304 → 0.192). That is learning to learn, in sign. In size it is small: by the fifth subject FULL has closed
17% of the gap between FROZEN and the ideal observer (40% with certain memberships). In the mixed world all arms lie
between 0.72 and 0.80 with standard errors of 0.02–0.03 per cell; FULL is 0.015 below FLAT at the fifth subject and
FULL+M 0.04 below FROZEN+M. I do not count that as an effect. Reasons I can see: a membership stated once is believed
at 0.6, the realism prior starts at 0.5, and four observations per member leave the coupling beliefs broad. The
earlier design, in which the lens handed evidence down to its poles, did no better (`pilot_transfer_v0c.out`).

### 3.6 What the example does and does not reproduce

| The owner's step | Reproduced? |
|---|---|
| weak first storage (0.23) | Yes: c = 0.200, from "one statement of an unknown source is half an observation" and the uniform prior. |
| refine, do not delete (0.57) | Yes in kind: nothing is deleted, the belief becomes "sometimes" (f = 0.56) and its evidence grows; under the new concept c = 0.333. The figure 0.57 is not reached with two statements from a stranger. |
| a new concept | Yes: the lens unit, allocated on conflict (newborn) or by anticipation (experienced). |
| a rule about the category, stronger than the instance (0.77) | Only with experience: 0.027 with nothing lived, 0.372 after three subjects, 0.514 after six. A rule that rests on one instance and nothing else is weaker than the instance in these equations, and I think it should be. |
| support fed back to the instance (0.88) | Only with experience or other instances: 0 in the newborn, 0.470 after three subjects, 0.557 after six, growing as other colours are confirmed. The owner's 0.88 is not reached. |
| the question "which other colours are there?" arises by itself | It crosses the threshold only in the experienced mind (0.0328 after three subjects) and is top-ranked only after six. The margin at three is thin (0.0328 against 0.03), and it depends on two constants: with `γ = 0.5` it would not arise. |
| the mind expands over the set of colours | Yes: each new colour is immediately constructed under the rule, journalled, and later scored. |
| categories emerge from instances | **No** in the final version (3.4, item 6). |

---

## 4. Language in and out

**L0 — statements (the reference script).** Input is a triple with polarity and source; output is `(e, f, c)`.

**L1 — a tiny tokenised world (the prototype).** A finite-state parser maps a few templates to ledger lines and
back, in Turkish and English, with nonce vocabularies:

| Sentence | Ledger lines |
|---|---|
| *kırmızı koyu bir renktir* / "red is a dark colour" | `ISA(red, colour) y=1`, `HAS(red, dark) y=1` |
| *kırmızının açık tonları da vardır* / "red also has light shades" | `HAS(red, light) y=1` |
| *siyah hiç açık değildir* / "black is never light" | `HAS(black, light) y=0` |
| *koyu açığın zıddıdır* / "dark is the opposite of light" | `EXC(dark, light) y=1` |
| *Ayşe'ye göre …* / "according to A, …" | the same lines with source `A` |
| *mavi koyu mudur?* / "is blue dark?" | query `HAS(blue, dark)` |

Output is produced from `(f, c)`, never from a stored sentence: a frequency word from `f` (never < 0.15 ≤ rarely
< 0.35 ≤ sometimes < 0.65 ≤ usually < 0.85 ≤ always; *hiç, nadiren, bazen, çoğunlukla, hep*) and a hedge from `c`
(*duyduğuma göre* < 0.25 ≤ *sanırım* < 0.5 ≤ *oldukça eminim* < 0.75 ≤ *biliyorum*); below `c = 0.02` the answer is
*bilmiyorum*. Constructions are spoken with a marker (*kurduğuma göre …*), questions from the agenda of 2.9
(*başka hangi renkler var?*). The English sentences quoted in section 3 are produced this way by `say()` in the
reference script; the Turkish words are the same table and are not implemented there.

**L2 — words and constructions as units (research stage, outside the 2,000 lines).** A word is a unit; "this word
denotes this concept" is a link with evidence gathered across situations in which the word and the concept occur
together; a construction is a unit allocated when a token sequence is not explained by known constructions; a word
class is a category of the items that fill the same slot, formed like any other category. All of this exists in the
literature as distributional and construction-grammar induction, and none of it is easy.

**How it scales, and where it does not.** Evidence, rules and questions scale with the number of links, not with a
context window, and what was learned ten thousand statements ago is as available as what was learned now. What does
not scale by itself is the mapping from open text to statements: relations with more than two arguments, events,
time, negation scope, quantifiers, metaphor. This family of systems (Cyc, NELL, NARS, PLN) has not solved open
language in thirty years, and I have no argument that this substrate would. The honest position is that Substrate A
is a candidate for the belief-and-memory core, and that it needs a perceptual front-end that turns text into
candidate statements with weights. That front-end need not be a frozen Transformer — it can be a learner whose
proposals enter the ledger as lines from a source called "my parser", trusted according to its record like any other
source — but it has to exist, and it is not designed here.

Two local facts of the project connect at this point. The exact, retractable write that the project obtained with a
Woodbury update has its counterpart here at the level of the ledger (3.2, retraction). And the calibrated "do I know
this context" score that the lead reports as confirmed on eight worlds (study `concept_address_2026_11_13`; I did not
re-verify it) is what a front-end would need in order to decide whether a piece of text addresses an existing unit
or opens a new one; inside the graph that role is played by `c`.

---

## 5. Falsifiable hypotheses and the decisive experiment

### 5.1 World

A tokenised stream ("shade world"), about 400 tokens: nonce names for categories, members, property poles and
informants, and a dozen function tokens (`is`, `a`, `not`, `excludes`, `says`, `?`, `.`, `yes`, `no`). Every life
draws fresh names, so nothing lexical can be carried from one life to the next.

A **life** is about 3,000 statements (about 18,000 tokens) over 40 subjects that arrive in sequence, 30% of the
statements returning to earlier subjects. A subject has one category, 6–10 members, two contrast pairs and two plain
properties. Each (subject, pair) has a regime — mixed (θ = 0.5), skewed (θ = 0.8) or pure — drawn with probabilities
that are fixed within a life and differ between lives, so that there is something to learn across subjects. A
quarter of the categories contain two or three exceptional members. Statements come from the mind's own observation
and from four informants of reliability 0.95, 0.9, 0.7 and 0.3. For a fifth of the members the first statement is a
generic one that later observations contradict (the "red" pattern).

**Probes** are questions `is X A ?` answered with a probability, never followed by feedback, scored by expected
log-loss under the generating parameters. Classes: R1 retention (the fact is at least 1,000 statements old), R2
refinement (frequency after a contradiction), G1 a new member of a known category, G2 the first member of a new
subject after one observation, X1 exceptions, S1 what the unreliable informant said. An **ideal observer** (exact
Bayes for the generating family, by enumeration and quadrature) gives the floor; results are reported as **regret**,
loss minus ideal loss, in nats per probe, with calibration error alongside.

### 5.2 Arms

| Arm | What it is |
|---|---|
| A-full | the 2,000-line prototype |
| A-frozen | the same with the operating process frozen (no pooling across subjects, realism and trust fixed) |
| A-fixed-k | one global coupling `k = 1`, frozen: an evidence logic with NARS's constant. It is not NARS and will not be called NARS |
| A-flat | counts and exclusions only, no categories |
| T-eq | a Transformer decoder whose parameter count equals the number of floats in A-full's state at the end of a life (evidence, cached sums, coupling weights; I expect 30–60 thousand), **trained the standard way**: next-token prediction on 30 million tokens of lives from the same generator, context 512 tokens, then frozen. At test it learns only in context |
| T-10x | the same with ten times the parameters |
| T-eq + TTT, T-10x + TTT | the same weights with **test-time fine-tuning**: one to three SGD steps on each incoming statement, learning rate in {1e-4, 3e-4, 1e-3, 3e-3}, with and without replay from a buffer whose size in floats equals A-full's state. The configuration is chosen on three validation lives and frozen before the ten test lives |
| Ideal | exact Bayes for the generator (floor) |

Reported for every arm: floats and bytes of state, wall time, statements per second. "Equal parameter count" is a
generous reading for A (its state is memory of this life; the Transformer's parameters are general), which is why
T-10x is included: if A only beats a Transformer of its own size, that is weak.

**Budget.** The project measured 2,729 training tokens per second on one thread for a 3.45-million-parameter model.
At about 0.4 million parameters (T-10x) I estimate 15,000 tokens per second: 30 million tokens in about 35 minutes
per core; T-eq a few minutes; test-time fine-tuning over validation and test lives about 1.5 core-hours; the A arms
and the ideal observer minutes. Total about 3.5 core-hours. If throughput is lower, lives are cut to 1,500
statements and pre-training to 15 million tokens; this fallback is fixed now.

### 5.3 Hypotheses and what refutes them

| | Claim | Refuted if |
|---|---|---|
| **H1** persistence and revision in one process (P1, P2) | On R1 and R2, A-full's regret is at most 0.03 nats, and every TTT arm is worse than A-full by at least 0.03 nats on at least one of R1, R2 (paired over ten lives, 95% interval excluding 0). | A-full's own regret exceeds 0.03 nats on R1 or R2; or any TTT arm, of either size, is within 0.03 nats of A-full on **both**. In the second case writing evidence into a ledger buys nothing measurable over fine-tuning a network at test time in this world. |
| **H2** the closed forms are good enough (P2, P3) | A-full's regret on G1 is at most 0.05 nats in each regime. | Regret above 0.10 in any regime; or A-flat within 0.02 nats of A-full on G1 (then categories do nothing). |
| **H3** the operating process improves and transfers (P5) | On G2, in subjects 21–40, A-full closes at least half of the gap between A-frozen and the ideal observer. | It closes less than a quarter. Between a quarter and a half: inconclusive, reported as such. The pilot of 3.5 is a warning: after five subjects A-full had closed 17% of that gap in the pure world (40% with certain memberships) and nothing measurable in the mixed one. |
| **H4** constructions are measured and become more real (P4) | Predicted realism is calibrated against realism after the fact (calibration error at most 0.10 over at least 300 resolved constructions per life); the squared-error skill of constructions against the uniform guess is positive and larger in the last third of a life than in the first. | Calibration error above 0.20, or no increase in skill (interval including 0). |
| **H5** questions matter (P2) | In an interactive condition (every fifth statement answers one of the mind's questions), A-full with its own questions reaches regret 0.05 on G1 with at least 15% fewer statements than with random questions from the same candidates, and no more than with lowest-confidence-first questions. | No better than random; or worse than the lowest-confidence heuristic (then 2.9 adds nothing to a textbook rule). |
| **H6** no self-support | Exact invariants: constructions never appear in the ledger; a rule never raises the confidence of the only instance it was built from; an informant that repeats one statement any number of times adds at most 2 units. | Any violation. |

Reported but not counted for A: (i) inside its 512-token window I expect the Transformer arms to match A-full and
the ideal observer on G1, because a Transformer meta-trained on such a family approximates Bayesian prediction in
context (prior-data fitted networks; meta-learned models of cognition). That outcome refutes nothing; H1 is about
what lies beyond the window and about revision. (ii) A **structure-shift** test: one subject whose regularity is a
transitive order and one whose rule is a conjunction of two properties. A has no unit that can express either and
should fail; if a Transformer arm learns them in context, that is a point for the baseline and will be said so.
(iii) A known hole: several informants that copy each other are counted as independent.

**Decision rule.** Substrate A is worth building beyond the prototype only if H1, H2 and H3 all hold. If H1 fails,
the argument for leaving the fixed network has to come from elsewhere. If H3 fails, P5 is not delivered and what
remains is a fixed evidence logic, which is prior art. H4–H6 qualify the result; they do not decide it. The
protocol (generator configuration, arms, grids, thresholds, analysis script) is frozen before the first test life,
as in the project's earlier studies.

---

## 6. Prior art

Searched on 2026-11-14 with 36 web-search queries; none contained the name of the project or its owner. Only
search-result summaries were available: arxiv.org and cis.temple.edu could not be fetched from this session, so no
full text was read. The eight closest works, then what remains.

| # | Work | What overlaps | What differs |
|---|---|---|---|
| 1 | **NARS / Non-Axiomatic Logic** (Wang; OpenNARS; "OpenNARS for Applications", Hammer & Lofthouse 2020) [1][2][3] | Almost everything in 2.2 and 2.5: truth value as (frequency, confidence) from positive and total evidence, `c = w/(w+k)`, revision by pooling evidence, induction and abduction as weak single-unit evidence, deduction, expectation, questions derived by backward inference, no separate training phase, evidence never counted twice, failed anticipations as negative evidence. | `k` is one user-set constant of the system ("evidential horizon", usually 1); the inference rules and control are fixed at the meta-level by design; overlap is prevented by evidential bases, not by leave-one-out; NARS is much richer (compound terms, time, procedures, attention under bounded resources). |
| 2 | **Hierarchical Bayes, overhypotheses** (Kemp, Perfors & Tenenbaum 2007, after Goodman) [4] | The model of 2.12 is theirs: `α` (how uniform each bag is) is the coupling `k`, `β` the category mean; the owner's example is their bags of marbles; learning the higher level speeds learning in a new category. | Batch inference by integration or sampling; no growing graph, no revision ledger, no questions, no language. |
| 3 | **Probabilistic Logic Networks** (Goertzel et al. 2008) and **Differentiable PLN** (Potapov et al. 2019) [5][6][7] | Strength and confidence on a hypergraph; in the differentiable version truth values are tensors and both premise truth values and rule formulas are learned by backpropagation; inference control learning is an acknowledged open problem. Goertzel (2024) shows PLN and NARS give similar strength × confidence when term probabilities are uncertain. | Term probabilities and a large rule set; learning there is gradient-based and offline. |
| 4 | **Adaptive Resonance Theory**, fuzzy ARTMAP, fusion ART (Carpenter, Grossberg; Tan) [8][9][10] | A new category node is recruited when no existing one matches (vigilance, match tracking) instead of overwriting: the classical answer to stability versus plasticity; incremental, one pass; rules can be read off. | Feature vectors and fuzzy AND; no evidence counts, no hierarchy of couplings; the mismatch test here is an evidence ratio. |
| 5 | **Subjective logic** (Jøsang) [11][12] | An opinion is belief, disbelief, uncertainty and base rate, equivalent to evidence counts; cumulative fusion is revision; trust discounting of sources; deduction and abduction operators. | No concept formation, no learned operating parameters. (The prior weight W = 2 of that literature is from my memory; the search did not confirm it.) |
| 6 | **Incremental concept formation**: COBWEB (Fisher 1987; revisited 2024), rational models of categorisation, structure learning in active inference (Smith, Schwartenbeck, Parr & Friston 2020; Friston et al. 2023) [13][14][15][16] | A new concept or latent slot is added when the evidence warrants, and removed by model reduction; incremental; category utility or model evidence as the criterion. | No graded links with revision; in active inference the slots are pre-allocated. |
| 7 | **Weighted logics trained by gradient**: Logical Neural Networks (Riegel et al. 2020), ∂ILP (Evans & Grefenstette 2018), Logic Tensor Networks, Neural Theorem Provers [17][18] | Truth values in tensors; in LNN truth *bounds*, an upward and a downward pass (the downward pass is the analogue of feeding support back), and the open-world distinction between unknown and contradictory. | Trained offline by backpropagation over a fixed set of formulas or templates. |
| 8 | **NELL, never-ending learning** (Carlson et al. 2010; Mitchell et al. 2018) [19] | Beliefs with confidence accumulated without end, learned rules as coupling constraints, promotion when independent sources agree. Its documented failure is **semantic drift**: errors that reinforce themselves. | Web-scale extraction with periodic retraining; not a single revision algebra. |

Also relevant, more briefly: **ACT-R and Soar** (utility learning, chunking: an operating process that improves with
use; both suffer a "utility problem" under continuous learning) [20]; **AM and EURISKO** (an agenda of
self-generated tasks, heuristics that modify heuristics; Lenat later analysed why learning new heuristics was much
harder than expected) [21]; **AERA / Replicode** (self-programming, models kept or dropped by their record) [22];
**Sigma** (a cognitive architecture as a factor graph with message passing) [23]; **a NAL-based sequence learner**
(Xu 2023/2025: hypothesise, revise, recycle; claims freedom from catastrophic forgetting) — the closest cousin on the
language path [24]; **NARS-GPT** and a 2026 Narsese benchmark (a language model as the language channel of NARS) [25];
**BeliefBank** (a belief graph added to a frozen language model) [26]; the **Thousand Brains Project** (evidence-based
learning modules without backpropagation, for sensorimotor objects) [27]; **prior-data fitted networks** and
**meta-learned models of cognition** (why the Transformer baseline may be Bayes-like inside its window) [28][29];
**dynamic evaluation** (the test-time fine-tuning baseline) [30]; **question asking by expected information gain**
(Rothe, Lake & Gureckis 2017) [31]; **reality monitoring** as the discrimination of imagined from perceived [32].
Truth discovery among unreliable sources (Dawid & Skene 1979) is cited from memory and was not searched.

**What remains new — if anything.** Stated as candidates with their status, not as claims:

- (a) *The coupling as a learned belief per context, pooled across subjects, inside an evidence logic of the NARS
  kind* — i.e. NARS's constant `k` read as the overhypothesis `α` and learned. Two queries aimed at exactly this
  (a learned or adaptive evidential horizon; NAL induction read as hierarchical-Bayes shrinkage) returned nothing,
  and the NARS sources describe `k` as a system constant and the meta-level as fixed. This is the one candidate worth
  a full-text check of the NAL book and of ONA before anyone says "new".
- (b) *One number in three roles*: the measured realism of past constructions is the mixing weight of a robust
  mixture and the prior confidence of the next construction. No source found; it is a small design idea next to
  NARS's anticipation and to textbook robust mixtures.
- (c) *The question gain of 2.9*: unseen-member mass times what the rule would lend a new member. Question asking
  by expected information gain is prior art; this particular objective was not found. Small.
- (d) *The two message formulas of 2.2.* A derivation from textbook moments; the statistics literature on shrinkage
  was not searched; not claimed.
- (e) *The combination*: one ledger, every belief a closed-form function of it, all couplings learned from it,
  exact retraction. Combination only, and unproven.

So nothing here is called new. If the experiment supports H3, candidate (a) is the thing to check properly.

Sources:
[1] https://cis.temple.edu/~pwang/GTI-book/GTI-CH3/GTI-3-4.html ·
[2] https://cis.temple.edu/~pwang/AGI-NARS/2015-KFU/02-Evidence.html ·
[3] https://cis.temple.edu/tagit/publications/ONA.pdf ·
[4] https://sites.socsci.uci.edu/~lpearl/courses/readings/KempPerforsTenenbaum2007_LearningOverhypotheses.pdf ·
[5] https://people.csail.mit.edu/zucker/PLN_BOOK_6_27_08.pdf ·
[6] https://arxiv.org/abs/1907.04592 ·
[7] https://arxiv.org/abs/2412.19524 ·
[8] https://en.wikipedia.org/wiki/Adaptive_resonance_theory ·
[9] https://open.bu.edu/items/f04eeac2-577c-4309-bf3d-720170e856eb ·
[10] https://ink.library.smu.edu.sg/sol_research/3587 ·
[11] https://www.openphilanthropy.org/wp-content/uploads/Josang2013.pdf ·
[12] https://mn.uio.no/ifi/english/people/aca/josang/publications/jos2016-fusion.pdf ·
[13] https://cdn.aaai.org/AAAI/1987/AAAI87-082.pdf ·
[14] https://arxiv.org/abs/2403.03835 ·
[15] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7250191/ ·
[16] https://arxiv.org/abs/2311.10300 ·
[17] https://arxiv.org/abs/2006.13155 ·
[18] https://arxiv.org/abs/1711.04574 ·
[19] https://papers.baulab.info/papers/knowledge/Mitchell-2015.pdf ·
[20] https://arxiv.org/abs/2201.09305 ·
[21] https://aaai.org/papers/00236-aaai83-059-why-am-and-eurisko-appear-to-work/ ·
[22] https://openaera.org/ ·
[23] https://arxiv.org/abs/2101.02231 ·
[24] https://arxiv.org/abs/2308.12486 ·
[25] https://github.com/opennars/NARS-GPT and https://arxiv.org/abs/2604.18873 ·
[26] https://aclanthology.org/2021.emnlp-main.697/ ·
[27] https://arxiv.org/abs/2412.18354 ·
[28] https://arxiv.org/abs/2112.10510 ·
[29] https://arxiv.org/abs/2304.06729 ·
[30] https://proceedings.mlr.press/v80/krause18a.html ·
[31] https://arxiv.org/abs/1711.06351 ·
[32] https://www.nature.com/articles/s41467-023-37322-1

---

## 7. Failure modes and the mechanism against each

"Status" says whether the mechanism exists in the reference code, and whether anything was observed.

| Failure | Mechanism | Status |
|---|---|---|
| **Runaway growth** of units, links, contexts | Allocation only on unexplained evidence (conflict mass; strong-evidence split); one lens per exclusive pair; a controller that raises `θ_alloc` when allocations per thousand statements exceed a budget; a merge test (the split test read backwards); hard caps with ghost-vote eviction (2.11). | Thresholds implemented. Controller, merge and eviction are **specified only**. In the trace: 34 units for 152 ledger lines, no runaway — on a tiny example. |
| **Self-reinforcing false beliefs** | Constructions never enter the ledger. Leave-one-out inside a category and leave-one-category-out across subjects, so a rule cannot support the instance it came from. Repetition discount: one source repeating one statement adds at most 2 units. Trust is earned only against evidence that did not come from the source. | Implemented; in the trace the newborn's fed-back support is exactly 0, and the liar falls to 0.168. **Not covered:** informants that copy each other count as independent; evidence that reaches a member along two category paths would be counted twice (needs provenance sets, NARS-style, on fed-back evidence); a biased parser is a biased source. NELL's semantic drift is the cautionary tale. |
| **Instability** | The solver is a fixed-point iteration (six sweeps) among responsibilities, couplings, realism and carry-over; damping and a per-sweep change monitor in the build; hysteresis between split and merge. | No convergence proof. Stable in every run here (trace, 144 comparison scenarios, pilot). Two weaknesses seen: hard assignments (split, join) depend on order and cannot be undone; and two lenses on the same evidence can disagree — at stage 10 the mind says of an undescribed colour "sometimes dark" with c = 0.44 through the pole lens while holding c = 0.037 through the refined lens. Consistency between lenses is not enforced. |
| **Forgetting** | None by construction for what is in the ledger: evidence is additive and local, so learning one thing cannot erase another. Under a cap, eviction loses the detail of typical members and keeps category statistics. | Additivity is exact. The opposite failure is open: when the world changes, old evidence outweighs new. Planned: a volatility belief per context on a grid of discount factors, scored like the couplings. **Not implemented.** |
| **Category formation that is over-eager or powerless** (found here) | A proper two-cluster evidence (each cluster with its own coupling), the merge test, and acceptance tests before the module ships (section 8). | With `ln 3`: false splits in 3 of 12 homogeneous and 6 of 12 heterogeneous eight-member categories. With `ln 10 + ln(n−1)`: none, and no split at all up to eleven colours. **Unsolved in this proposal.** |
| **Weak transfer from little data** (found here) | Longer lives; memberships confirmed more than once; learned unseen-member and repetition constants. | Pilot: the learned process closes 17% of the gap to the ideal observer after five subjects in the pure world (40% with certain memberships), nothing measurable in the mixed world. H3 exists to decide whether this is a matter of data or of design. |
| **Parser errors become evidence** | The parser is a source with learned trust; its lines can be retracted exactly. | Design only. |
| **Cost** | Incremental solver; test that it equals the from-scratch solver to 1e-9. | Reference script: about 0.3 s per statement on the trace (full recomputation, numpy). |
| **Over-reading** | — | Nothing here is consciousness. It is a bookkeeping loop whose parameters are beliefs. The owner should not be told otherwise. |

---

## 8. Build plan: 2,000 lines, with tests

`bg_core.py` is the oracle: the prototype must print the numbers of section 3 to six decimals before anything is
optimised.

| Module | Lines | Content | Tests that gate it |
|---|---|---|---|
| `ledger.py` | 160 | append-only lines, per-source sparse counts, compaction, retraction, replay from checkpoints | retraction equals "never heard" on evidence (exact); replay equals the state (exact) at random positions |
| `store.py` | 200 | units, sparse `W±` (index maps + `index_add_`), growth without re-indexing, caps | growth keeps indices; caps respected |
| `truth.py` | 120 | `f, c, e`, `vote`, `handdown`, batched `lbeta` on `torch.lgamma`, grids | the limits of 2.2; agreement with `math.lgamma` |
| `context.py` | 320 | contexts, incremental `S0, S1`, leave-one-out tensors, coupling posteriors, responsibilities | incremental equals from-scratch (1e-9) on random lives; mean gap to exact hierarchical Bayes at most 0.02 (the check of 2.12 as a regression test) |
| `schema.py` | 180 | kinds, pooling of couplings, carry-over grid, leave-one-category-out | property test: no context ever receives evidence that originated in its own category |
| `allocate.py` | 220 | lens allocation (conflict, anticipation), split and join with a two-cluster evidence, merge, rate controller | at most 5% false splits on 200 single-population categories; at least 80% detection on two groups of 8; a forced wrong split is merged back |
| `operate.py` | 140 | trust by leave-one-source-out, realism, journal | friend and liar; repetition and echo invariants (H6) |
| `imagine.py` | 110 | constructions, predicted realism, resolution, calibration report | constructions never in the ledger; calibration on synthetic lives |
| `question.py` | 110 | candidates, gains, agenda | gain increases with unseen mass and with rule strength; no question when nothing can be gained |
| `language.py` | 200 | Turkish and English templates to and from ledger lines; hedged output; question wording | round trip of every template; the red trace as a golden test |
| `mind.py` | 140 | the loop of 2.5, budgets, statistics | end to end: the trace of 3.2; the dose-response is monotone in lived subjects |
| **core** | **1,900** | | |
| tests | about 600 | the gates above | |
| harness (separate) | about 700 | world generator, ideal observer, Transformer arms, test-time fine-tuning, analysis | generator determinism; ideal observer against quadrature |

Order: (1) ledger, store, truth; (2) context, against the oracle; (3) schema, operate; (4) allocate — the module
most likely to need a redesign; (5) imagine, question, language, mind; (6) harness; protocol frozen before the first
test life. Stop rules: if the acceptance tests of `allocate.py` cannot be met in two iterations, ship without
sub-categories and say so in the report; if the incremental solver cannot be made equal to the from-scratch one, run
the experiment on the from-scratch solver, ported to batched tensors (the numpy reference needs about 0.3 s per
statement at 150 ledger lines and grows with the state, so it would not fit a 3,000-statement life as it stands).

---

## 9. What is not claimed, and how to reproduce

Not claimed: novelty of any mechanism; superiority over a Transformer of any size; that categories emerge from
instances in the present equations (they did not); that questions arise by themselves in a mind that has lived
nothing (they do not); that language is solved; that this is consciousness.

Files, all in
`/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/belief-graph/`:

| File | What it is | Run time |
|---|---|---|
| `bg_core.py` | the reference equations (section 2) | — |
| `trace_red.py` → `trace_red.out`, `trace_red.json` | the worked trace (3.2) | 95 s |
| `dose_response.py` → `dose_response.out`, `.json` | lived subjects against readouts (3.3) | about 6 min |
| `check_exact_hbm.py [--split]` → `check_exact_hbm.out`, `.json` | gap to exact hierarchical Bayes (2.12) | about 1 min |
| `pilot_transfer.py` → `pilot_transfer.out`, `.json` | internal transfer pilot (3.5) | about 8 min on 3 cores |
| `split_continuation.py` → `split_continuation.json` | how much more must be lived before a sub-category forms (3.4) | about 3 min |
| `*_v0c.*`, `bg_core_v0a.py.bak`, `bg_core_v0c.py.bak` | earlier formulations of the design log (3.4), kept for audit | — |

All runs: `OMP_NUM_THREADS=1 python3 <file>` from this folder; numpy only; seeds fixed in the files.
