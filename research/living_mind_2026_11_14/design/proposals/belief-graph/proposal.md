# Substrate A — the belief graph

A growing graph of concept units whose links carry evidence, operated by one loop in which perceiving,
learning, constructing and asking are the same process.

⟦TR_SUMMARY⟧

**Status.** A design proposal with executable reference equations. `bg_core.py` (numpy, about 600 lines) runs every
equation of section 2; the numbers of section 3 are printed by `trace_red.py`, not typed by hand. Three small checks
were run (comparison with exact hierarchical Bayes, a dose-response of the red episode, an internal transfer pilot).
**No experiment against a Transformer has been run.** Nothing below is a result about being better than anything.
Two hosts needed for full-text literature checks (arxiv.org, cis.temple.edu) were not reachable from this session;
the prior-art section therefore rests on search-result summaries and is weaker than it should be (section 6).

---

## 0. Verdict first

Substrate A satisfies P1 (one process), P2 (graded, revisable beliefs) and P3 (units formed when needed) natively and
exactly, P4 (constructions that are measured) in a narrow but real sense, and P5 (the operating process improves)
only parametrically: the mind learns *how strongly* to generalise, carry over, trust and split, and it applies an
acquired lens to new subjects, but it does not invent new kinds of inference. P6 is the weak point: this family of
systems has a long record of not scaling to open language.

Every mechanism has prior art (NARS, hierarchical Bayes with overhypotheses, ART, PLN, subjective logic). I claim
nothing as new. What the substrate offers is a specific, testable combination, and three properties a trained
Transformer does not have by construction: every experience changes the state at once and can be taken back exactly;
every belief carries how much evidence stands behind it; and the state says what it does not know.

The worked example gives a mixed picture, reported as it came out (section 3): ⟦VERDICT_NUMBERS⟧

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
   regularity from one subject to another, how much to trust each source and the mind's own projections, when to
   split — are themselves **beliefs updated by the same evidence**, so the way of constructing improves with
   experience and transfers to new subjects.

| Part | Serves | How |
|---|---|---|
| Ledger; beliefs are functions of it | P1, P6 | one write per experience is the whole learning step; exact retraction |
| Evidence pair, frequency, confidence | P2 | weak first storage, revision by addition, "sometimes" = stable middle frequency |
| Trust-weighted evidence, per source | P2, P5 | a statement is worth what its source has proved to be worth |
| Refined lens (MIX unit), sub-category | P3 | allocated on unexplained evidence; a category is formed from its instances |
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

**Observing one's own output from outside.** The constructed statement is passed through the same `answer` path as
a statement heard from a stranger; what the mind would say about it is exactly what it constructed, with the
confidence it deserves, and the source "my own projection of kind σ" carries the trust `v_σ` it has earned.

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
  what makes it a specification rather than an engine. ⟦TIMING⟧
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
four regimes): ⟦HBM_CHECK⟧

**Relation to the named systems** (details and sources in section 6):

- *NARS.* The evidence pair, `f`, `c = w/(w+k)`, revision by adding evidence, and the expectation are NARS's. NARS
  induction lets each instance contribute at most one unit, `w = n/(n+k)`: that is `vote(n, k)` at `k = 1` up to the
  factor `(k+1)/k`. The difference is that NARS's `k` is one global constant of the system ("evidential horizon",
  usually 1), while here the coupling is a belief per context, learned, and pooled across subjects; and that
  overlapping evidence is excluded by leave-one-out algebra instead of evidential-base stamps. NARS has far more than
  this substrate: compound terms, temporal and procedural inference, self-monitoring, budgeted attention.
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
| **1. "red is a dark colour"** — first hearing | "red is dark": f = 1.00, **c = 0.200**, e = 0.613. Says: *"I have only heard that red is always dark."* By exclusion: red is light f = 0.00, c = 0.165 (*"never light"*). | Own evidence the same (**c = 0.200**). The lens `MIX(dark\|light)` is allocated **by anticipation**; the mind constructs *"red is sometimes dark, sometimes light"* (e = 0.659, c = 0.226, predicted realism 0.902) before being told. With that support: red is dark f = 0.696, c = 0.425 (*"usually dark"*); red is light f = 0.379 (*"sometimes"*, not "never"). | 0.23 |
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

⟦PILOT⟧

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
(*başka hangi renkler var?*). The examples of section 3 are produced this way.

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
