# Evidence arithmetic and inference: attack on LEAD_DESIGN.md (Zihin v0)

Lens: the equations of §1, §2.5, §4 and §6, worked by hand and in simulation.
Reviewed text: `newcore/LEAD_DESIGN.md` against `newcore/VISION.md`. Date: 2026-11-14.

## 0. Result in one paragraph

The readout of a belief (§1: f, c, e from additive evidence with horizon k) is sound, and the provenance
vector is the asset that makes every repair below possible. The three rules that move evidence are not sound
as written. (1) §2.5 counts silence as evidence against a belief and counts a denial as evidence for it.
(2) §6 lets imagination write evidence, and §5 rewards it for doing so. (3) §4 sends an instance's evidence up
into the rule and back down into the instance, without a bound. In one generated life of 6000 experiences
(two tellers, a closed sense, categories, idle steps) the design as written ends with expectations about
unobserved pairs that are worse than answering with the base rate every time (Brier 0.238 against 0.217; "always
1/2" gives 0.250) and that fall on the right side of 1/2 for 68.6% of the pairs, which is the share of false
pairs, while the conviction it reads has risen to 0.87. The corrected arithmetic of section 2, on the identical
stream, ends at Brier 0.048 with 94.6% of expectations on the right side of 1/2 (0.038 and 97.1% with the
completeness estimator that section 2 recommends; `evm_s8`). The corrections add no table, no frozen part, no
training phase, no vocabulary and no domain knowledge: they change what is counted, where it is stored, and how
it is read.

How to read the evidence. Every number below was produced by a script in this folder (`evm_*.py`, output saved
next to it as `evm_*.out`; pure NumPy, one thread, each under a minute). Where the design text is silent I
state the reading I used; where two readings are possible I ran both. I did not run the prototype. The full
settling dynamics of §2.2 (kWTA, decay) are not simulated here: all units of a moment have a = 1, g = 1.
The bias of the §2.5 rule on a single tie was found independently by another reviewer
(`sim_rule_bias.py`); my `evm_s1` A2 reproduces the same fixed point.

Notation. Quotations keep the design's symbols (w⁺, τ, λ, ι). My own text writes them in plain letters (W+,
tau, lambda). For the corrections: n+ and n- are raw counts on a tie (per source), N is a raw count of
opportunities on a unit (per source), chi is the completeness of a source, rho its reliability,
K = k rho/(1 - rho) its coupling, b a base rate, k_g the horizon of a category rule. `par(a, b) = ab/(a+b)`.

---

## 1. Findings, ordered by severity

### FATAL

#### E1. Silence is counted as evidence against a belief; a denial is counted as evidence for it

Attacked text. §2.5: "v occurred: w⁺[π] += g·a_u (π = provenance of the confirming event)"; "v was predicted,
did not occur: w⁻[π] += g·a_u·e_{u→v}". §1: "frequency f = W⁺ / W (how often it held)"; "Contradiction does not
delete: it adds W⁻, f moves, c rises (more has been lived)."

Evidence.
1. Derivation. Let the target of a tie occur with probability p when its source is present. Positive evidence
   grows at rate p, negative at rate (1 - p)·e. With much evidence e = f, so the stationary frequency solves
   (1 - p) f² + p f - p = 0. It is not p: f*(0.5) = 0.618, f*(0.2) = 0.390, f*(0.1) = 0.282, f*(0.02) = 0.133,
   and f* = 1/2 exactly at p = 1/3. After a single coincidence followed by n misses, f falls like 1/sqrt(3n),
   not 1/(n+1): 0.061 after 100 misses, where a count gives 0.010. The weighting by e does not make weak
   expectations cost little either: a fresh tie has e ≥ 1/2, so its first misses weigh at least half.
2. A true fact that is never contradicted. X has K permanent true properties and every statement about X names
   one of them. Each statement is a miss for the other K - 1 ties. Told once (tau_told = 0.3), then two other
   true statements about X: f = 0.464, e = 0.486. After 4000 statements every tie ends at f*(1/K) with c = 0.998:
   0.390 for K = 5, 0.282 for K = 10. The mind is certain that each true fact holds less than half the time.
   If only e > 1/2 counts as "predicted", every tie ends at e = 0.50 with c between 0.98 and 1.00, which §2.6
   reports as conflict (`evm_s1` A1, A2).
3. A denial. "X not v1" contains v1. The only route to w- is non-occurrence, so the denial is booked as an
   occurrence: ten denials raise e(X → v1) from 0.615 to 0.884 (`evm_s1` B). If instead only lag-1 ties
   predict (§2.3 names only those), lag-0 ties never receive negative evidence at all and stay at f = 1.
4. Trust inherits the error. An honest teller is "refuted" every time the senses show another aspect of the
   same object. A 100% honest teller ends at tau = 0.390 for K = 5 and 0.133 for K = 50; the number measures
   how many properties objects have (`evm_s1` C).
5. World test. 200 objects, 20 properties, 5 true per object, 6000 told statements by an honest teller, 15% of
   pairs also looked at. Design: Brier 0.315 on pairs never looked at, 0.4% of the told true facts on the right
   side of 1/2, learned tau_told = 0.201 for a teller who is never wrong (`evm_s1` D).
6. Refinement. "X a0" told once (e = 0.615). A compatible statement about X ("X b1", a property on another
   channel), an incompatible one ("X a3", the same channel) and the same incompatible one from the same teller all
   move the design's belief to exactly the same place (f = 0.62, e = 0.539). It reacts to the absence of the
   word, not to the contradiction (`evm_s9`).

Severity: fatal. T1, T3 and T7 depend on this rule. T1 can pass for the wrong reason: any statement about red
that does not mention "dark" weakens "red is dark", so a test that only checks "f moved after the contradiction"
is satisfied by an unrelated sentence.

Smallest fix. Three changes; the third is the one that carries the information.
- Store raw counts and stop weighting by belief. The weight of an observation may depend on attention (a_u,
  a property of the observation), never on the expectation e (a property of the believer). A tie keeps n+ per
  source. Silence is not written on ties at all: a unit keeps N[u, s], the number of moments it was present
  while source s (a teller, a sense channel) reported. This is fewer scatter-adds than the design, not more.
- Give explicit absence a route: an absence event on a channel, or the occurrence of v' weighted by a learned
  exclusion x(v', v), adds to n-. The exclusion is read from the ordinary tie between the two property units:
  x(v', v) = c·max(0, 1 - f_up/b_v), how much less often than by chance v holds when v' holds (f_up is the
  frequency two standard errors up, b_v the base rate of v). Learned from 2000 looks through two state channels
  it is 0.97 between symbols of one channel and 0.02 across channels; the compatible statement of point 6 then
  leaves the belief at 0.615, the incompatible one moves it to f = 0.50 with c up from 0.231 to 0.374, and an own
  look moves it to f = 0.23, c = 0.565 (`evm_s9`). "Not" is learnable the same way, as a unit whose frames are
  followed by absence; nothing is parsed in the core (this last step is specified, not simulated).
- Read silence through the completeness chi_s of the source: the effective number of observations a source has
  made of "does v hold of u" is E = max(n+ + n-, chi_s·N), and its frequency is f = n+/E. chi = 1 is a closed
  world, chi = 0 an open one, and chi is learned per source as the source's own repetition rate
  (after it has said something about a subject once, how often it says it again when it speaks about that
  subject); a channel whose adapter declares that it reports the whole state of what it measures has chi = 1.
  Predicting what a channel will show next (PREDICT, surprise, chunking) keeps chi = 1: f_occ = n+/N.
Do not fix E1 by simply dropping negative evidence. In the integrated life that variant is worse than the
design (Brier 0.354, 38.7% on the right side): with no negative information every generalisation is believed.
With chi taken from the tellers' own repetition rate (no closed referee is needed for chi), the same life gives
0.038 and 97.1% (`evm_s8`, ablation table). On the world of point 5 the corrected arithmetic gives Brier 0.023
and learns chi = 0.199 (true 0.200) and rho = 0.989.

#### E2. Imagination writes evidence, the reality score reads its own deposits, and relief rewards the loop

Attacked text. §6: "the mind runs §2 on its own predictions: the top prediction is fed back as an event with
provenance "imagined" on an internal channel"; "R = e(i → v) computed from the evidence that does NOT lie on the
chain that produced it (leave-path-out)."; "It is written with provenance "imagined"/"inferred" and weight ι·R."
§2.5: "LEARN (always on)". §5: "D(u) = Σ_{ties t at u} use(t) · (1 − c_t)"; "Relief = −ΔD is felt (§2.6)."

Evidence.
1. LEARN on imagined events locks the mind onto its first guess. World: u is followed by v1 with p = 0.6 and by
   v2 with 0.4; 300 lived moments; r idle steps after each. With r = 10 the mind does not converge on 60/40: it
   drives whichever successor was ahead early to f = 0.95 and the other to 0.20. In 40% of 40 lives it locks on
   the wrong one (the 40% successor at f = 0.918, the 60% one at 0.267), for good: with a life ten times longer
   all 40 lives keep the direction they locked at. The evidence it reads is 6.07 times the evidence it lived.
   tau_imagined settles at 0.64 because the top prediction is usually confirmed. A sceptical birth prior
   (tau_imagined = 0.05) only delays this: 32.5% wrong, same lock-in. Sampling the imagined event instead of
   taking the top removes the lock-in and keeps the inflation (W read / W lived = 7.0): a Pólya urn adds weight,
   not information (`evm_s3` A).
2. If C3 is read literally for the trust tie (an imagined event confirms the imagined prediction), the fixed
   point after n self-confirmations is tau = [(n-1) + sqrt((n-1)² + 2n)]/(2n): 0.95 after ten idle steps.
3. The deposit. For a conclusion about an instance that was never observed, the evidence "not on the chain" is
   empty, so R = 1/2 whatever the chain says. The deposit ι·R is written on the tie i → v, which is not on the
   chain, so the next idle cycle reads it back: with tau_im = 0.5, ι = 1 the conviction about something never
   met is 0.62 after 5 cycles, 0.89 after 20, 0.98 after 100 (`evm_s3` B).
4. Sign. R < 1/2 still deposits on the positive side. An instance seen twice not to be v has R = 0.167; after
   13 idle cycles e(i → v) > 1/2: the mind expects what it has twice seen to be false.
5. Every such cycle raises c, lowers D and is felt as relief, so by C3 the tie (state, imagine) → relief is
   reinforced. The design's own drive pays for the loop.
6. T8 as written cannot be met: for a new construction the design's R is 1/2 always (Brier 0.250 at the start
   and at the end of 1500 constructions); with five rehearsals it is between 0.81 and 0.86 in every domain
   although the true hit rates are 0.905, 0.625 and 0.505 (`evm_s3` C).

Severity: fatal. This is the opposite of vision point 5 (a reality sense), and T8's ablation "without reality
scoring drifts" does not separate the design from its ablation, because the design drifts with it.

Smallest fix. Events of provenance imagined or inferred activate and predict; they write no counter. Delete
the sentence "It is written with provenance "imagined"/"inferred" and weight ι·R." R becomes a readout: the
tie's ordinary expectation under the leave-one-out discipline of E3/E4, which for a new conclusion is the base
rate supplied by its rule, with the horizon the mind has learned for such rules. The record "this was
constructed" is the tie itself (it has structure and no primary evidence). Self-trust keeps its job and gets a
signal that is independent of it: when primary evidence later arrives on the tie, it referees the rule that
predicted it (CA6, CA7 below). On the three-domain test the corrected R goes from Brier 0.209 (first 100
constructions) to 0.187 (last 500), which is the best possible value, and reads 0.905 / 0.630 / 0.510 against
true 0.905 / 0.625 / 0.505. Relief then cannot be earned by idling, because idling changes no conviction.

#### E3. Induction and support form a loop with gain m·tau·lambda·e²; the cap is bypassed; twin categories multiply

Attacked text. §4: "g acquires EXPECT g → v with evidence = Σ_i min(W_i, cap) discounted by the ABSTRACT
expectations"; "W_eff⁺ = W_own⁺ + λ · e(i→g) · W⁺(g→v), same for W⁻, λ = 0.5, provenance "inferred"".
§2.2: "up through ABSTRACT (instance activates its categories)". §2.5: "for every tie u → v with u in the last
focus". §3: "the fillers seen in that position get ABSTRACT ties to a new category unit for the slot".

Evidence.
1. The loop. The Up rule reads W_i, which by §1 is the trust-weighted sum over all provenance classes,
   including "inferred", and the Down rule produces evidence of provenance "inferred" on the instance tie.
   For m instances with own evidence w and membership e: W_g = m·e·(w + tau_inf·λ·e·W_g), so
   W_g = m·e·w / (1 - G) with G = m·tau_inf·λ·e². For G ≥ 1 there is no fixed point. With lambda = 0.5 and e = 1
   that is m ≥ 2 if tau_inf = 1 and m ≥ 4 if tau_inf = 1/2 (the value of an untrained trust tie). With the cap
   the fixed point is the cap itself, W_g = m·cap, whatever the instances know: five hearsays (c = 0.23 each)
   become c_i = 0.74 and c_g = 0.83, against 0.47 without the loop. If the inherited evidence is added at every
   moment, as C3 says evidence is, three hearsays reach c_i = 0.94 in 20 moments (`evm_s2` a).
2. Without the loop (the Up rule reads own evidence only) the instance's own evidence still returns to it
   (+50% for e = 1), and every twin category adds the whole rule again: ten instances with one observation each
   read W_eff = 6, 11, 26, 51 for 1, 2, 5, 10 categories with the same members. §3 creates such twins by
   construction, one per frame slot (`evm_s2` b).
3. The cap is bypassed. The category unit is activated whenever an instance is (§2.2), and while it is in the
   focus LEARN writes g → v directly and uncapped. One member seen 1000 times gives W(g → v) = 1001 and a member
   never observed inherits 500.5: c = 0.998 (`evm_s2` c).

Comparison. Evidential reasoning systems forbid exactly this. In NARS a conclusion is merged into a belief
only if their evidential bases (stamps) do not overlap; otherwise the more confident of the two is kept
("choice"), so "that evidence is only counted once in the conclusions the system makes"
(`landscape/nars-logic.md` §5.5, verified there in the ONA source). The same note asked the design for "an
explicit rule for how much coherence may add and a proof or test that the loop instance → rule → instance
converges below certainty". The design supplies neither; the recursion above is the missing analysis and it
diverges. Stamps also leak in practice (ONA cuts them at ten ids and gives every repeated sentence a fresh id,
same note §4), so they are not the repair proposed here.

Severity: fatal under the literal reading (loop), serious under the charitable one (finite inflation that grows
with category size and with the number of twin categories). T3 passes only for small categories.

Smallest fix. (a) The Up rule reads the instance's own primary evidence, never its effective evidence. (b) The
product of the Down rule is never stored; it is a readout computed from the rule built without the member
(leave-one-out; exact, because the sums are additive). (c) A unit that is in the focus without an event of its
own (a category reached through ABSTRACT, a recalled unit) is not a source end of any evidence increment (E9).
(d) Several categories are combined by choice (the best-established rule), not by sum. With (a) to (d) no
derived number is ever an input to its own derivation, so the loop gain is zero by construction, and stamps
are not needed. Proof sketch in section 2 (I1).

### SERIOUS

#### E4. Inherited support is unbounded: exceptions do not stay exceptions, and a mixed category overwrites what its members know

Attacked text. §4: "an exception stays an exception (own negative evidence outweighs inherited support)".

Evidence. With cap = 1 the support is λ·m. A member needs n- > λ·(m - 1) own contrary observations before
e_eff < 1/2: 3 for m = 5, 10 for m = 20, 50 for m = 100, 500 for m = 1000. With m = 20, after one sighting that
x is not v: W+ = 10, W- = 1.5, e = 0.84 (`evm_s2` d). In generated worlds (rule 0.95/0.05, a member conforms
with probability q, 0 to 10 own observations) the mean squared error of e_eff against the member's true
frequency is, for categories of 100:

| q | design (own-evidence reading) | corrected, base rate only | exact Bayes of the generator | own evidence only |
|---|---|---|---|---|
| 0.98 | 0.014 | 0.011 | 0.006 | 0.055 |
| 0.90 | 0.064 | 0.025 | 0.019 | 0.055 |
| 0.50 | 0.176 | 0.056 | 0.043 | 0.055 |

Exceptions with ten own observations are still overruled by the design (0.416 at q = 0.9; own evidence alone
gives 0.006). At q = 0.5, where no usable rule exists, the design is three times worse than ignoring categories
altogether: unbounded 50/50 evidence overwrites each member's own knowledge. The design's error grows with the
size of the category (0.039, 0.050, 0.064 for m = 5, 20, 100 at q = 0.9); the corrected error falls
(`evm_s2` e). Where the design is right is the one case of a homogeneous category and a conforming member
(q = 0.98, conformers: 0.001 against 0.009 for the base-rate form).

Severity: serious.

Smallest fix. The category does not add evidence to the instance tie; it sets the base rate of its
expectation. With S± the rule's evidence without the member: b_i = (S+ + k_g·b0)/(S + k_g) and
e_i = (W_own+ + k·b_i)/(W_own + k). A member never observed gets e = b_i (calibrated: 0.074 against 0.073 for
exact Bayes at q = 0.9). One own closed observation against, with no other own evidence, brings e to
k·b_i/(1 + k) ≤ 1/2 whatever the size of the category; with one earlier hearsay in favour it takes two
(`evm_s7` A, rows i and j: 0.534 then 0.372 for six members, 0.565 then 0.394 for six hundred). The horizon
k_g = k(1 - rho_g)/rho_g is learned from how well members agree with rules built without them (rho_g is the
intraclass agreement, chance-corrected); it is what makes one example enough in a homogeneous domain and many
examples necessary in a mixed one. If the design's sentence "a category rule raises the conviction of the
instance belief" is to hold for c as well, add a bounded amount h_i = par(S, K(rho_i)), where rho_i is the
member's own conformity. This "gated" variant is within 0.001 of the best corrected variant in all nine worlds
(0.009 to 0.057), and the variants with a learned strength recover most of the homogeneous case (conformers
0.003 to 0.006 against 0.009 for the base rate alone).

#### E5. The reality score R is not well defined on a graph with shared ties

Attacked text. §6: "R = e(i → v) computed from the evidence that does NOT lie on the chain that produced it
(leave-path-out)".

Evidence (`evm_s5`).
1. Novel conclusions: the tie i → v is never on the chain, and for a new conclusion it has no evidence, so
   R = 1/2 exactly. R differs from 1/2 only when the conclusion was already known by other means.
2. No shared tie, all evidence shared. If i belongs to g and to a twin g' with the same six members, the
   conclusion through g has "outside" evidence through g' of λ·6 = 3 units: R = 0.875 (0.929 and 0.969 with two
   and five twins). The six observations behind the check are the six observations behind the chain. Leaving
   out ties does not leave out evidence, because the Up rule has already copied it onto other ties.
3. "The chain that produced it" does not exist in §2.2, where settling sums all incoming ties in parallel.
   With one route for (six members with v) and one against (four without), R is 0.167 if the first is called
   the producing chain, 0.875 if the second is, and settling itself computes 0.583.
4. Shared prefix. Two chains i → a → b → v and i → a → c → v: removing the first removes i → a, the second
   route is cut, R = 1/2, although a → c → v is independent evidence for half of the conclusion.
5. The conclusion's own evidence sits on both sides (it is in the rule through Up and it is the "outside"
   evidence). In a world with no regularity at all (members independent, p = 1/2) the hit rate of constructions
   refereed by later sensing is 1.000, 0.750, 0.688, 0.664, 0.584 for categories of 2, 3, 5, 10, 30 members;
   with the rule built without the member it is 0.500. The design's self-trust therefore rises above chance
   exactly where nothing can be constructed, most for small categories.
6. A chain of L lag-1 ties concludes something at lag L; the design's ties have lag 0 or 1 only, so the tie
   "i → v" on which R is read and the deposit written means something else (not simulated).

Severity: serious (it becomes fatal through the deposit, E2).

Smallest fix. Replace leave-path-out by leave-one-out on evidence, which costs nothing once E3 is fixed: the
rule seen by a member never contains that member's primary evidence, and routes with possibly overlapping
members are combined by choice. Then R is simply the tie's expectation (CA8) and the referee signal for
self-trust is independent by construction.

#### E6. "Trust in the teller" cannot be represented or learned as specified

Attacked text. §1: "Evidence on a tie is a vector over provenance classes Π = {sensed, told, imitated(observed
in others), acted(own outcome), inferred, imagined}"; §10: "w⁺[6], w⁻[6]"; §1: "τ for a source is the
expectation of the tie (source unit → "holds"), learned by C3 (told statements later confirmed or refuted by
sensed evidence change the trust in the teller; inferred/imagined conclusions later confirmed or refuted change
the trust in the mind's own construction)."

Evidence.
1. Six class slots hold one tau_told. An honest teller (95% right) and a liar (5% right) who talk equally much
   give tau_told = 0.495; on 90.8% of the propositions the two cancel exactly (e = 1/2), and 4.5% end on the
   right side. The mind learns nothing from being told. With evidence indexed by source the same stream gives
   95.0% (`evm_s4` b). Evidence added into a shared slot cannot be separated or re-weighted later, so the
   retroactive re-weighting that readout-time trust would allow (learning that a source lied discounts
   everything it said) is lost.
2. No equation says how the trust tie is updated. At confirmation time the source unit is not in the focus,
   so §2.5 ("u in the last focus") does not apply. The update I had to assume for every simulation is: when
   evidence of a more primary class arrives on a tie, score every other source that has evidence on that tie.
3. If the referee evidence is not excluded from what it referees, trust is not identifiable (E5 point 5).

Severity: serious.

Smallest fix. Index evidence by source unit (sparse: a tie has few sources). Classes are ordinary categories of
source units, so a new teller inherits the trust of its class as a base rate through the corrected Down rule
(one substance, one rule). Write the trust update as the referee rule CA6: sources are ordered (own closed
senses and acts > told and imitated > derived), a source is judged only by evidence of strictly more primary
sources on the same tie, each tie contributes at most one unit, and the judged source's own evidence is never
part of the referee.

#### E7. Trust as a hit rate counts noise as evidence, counts the base rate twice and lets repetition make certainty

Attacked text. §1: "W⁺ = Σ_π τ_π w⁺[π], W⁻ = Σ_π τ_π w⁻[π]", with τ "the expectation of the tie
(source unit → "holds")".

Evidence.
1. A source that is right half the time has tau = 1/2 and is counted as half an observation per claim. One
   90% source plus M noise sources: 90.1% of beliefs on the right side for M = 0, 80.4% for M = 3, 59.3% for
   M = 10, 57.1% for M = 30, while the conviction read on those beliefs rises from 0.48 to 0.94. With
   rho = 2A - 1 the accuracy stays at 89 to 90% (`evm_s4` a).
2. A hit rate is not a reliability. In a world where 90% of all ties hold anyway, constructions drawn at random
   are confirmed 90.7% of the time; the design then writes each construction at 0.91 of a sensed observation.
   Relative to the base rate the reliability is (0.907 - 0.9)/(1 - 0.9) = 0.07 (`evm_s4` e).
3. Repetition. One teller repeating one statement gives W = 0.3 n: c = 0.55 after 4 repetitions, 0.75 after 10,
   0.97 after 100. Four repetitions outweigh one own contrary sighting (f = 0.55 for the told side), and every
   repetition is felt as relief, so asking the same source again is the cheapest way to feel better
   (`evm_s4` d).

Severity: serious.

Smallest fix. Two lines. Reliability is chance-corrected: rho_s = max(0, (A_s - b)/(1 - b)), where A_s is the
expectation of the trust tie and b the base rate of ties of that kind (2A - 1 when b = 1/2). The evidence of
one source on one tie passes a series coupling whose strength is the odds of its reliability:
W_s = par(E_s, K_s), K_s = k·rho_s/(1 - rho_s). Then par(1, K) = rho: the first hearing is unchanged
(c = 0.231 for rho = 0.3); n repetitions saturate at K, so a belief known from one source alone can never
exceed c = rho_s (0.291 after 10, 0.300 after a million); own closed senses have rho = 1 and accumulate
without bound.

#### E8. "Exact retraction" does not hold, and evidence is order-dependent

Attacked text. §9 (iv): "exact retraction of any single experience (evidence is additive)"; §10: "Every
experience's evidence increments are logged (moment id → tie increments), so one experience can be retracted
exactly by subtracting them."

Evidence.
1. An increment depends on the expectation at the time, which depends on earlier experiences. Experience x:
   told once that u → v. Then u occurs five times without v. Subtracting x's logged increment leaves W+ = 0,
   W- = 1.94: c = 0.66 that u → v is false. In the life without x the tie does not exist (`evm_s6` A).
   The same holds for trust-tie increments caused by x, for caps, for threshold decisions (chunking,
   recruitment) and for every inferred or imagined deposit computed while x was in.
2. Order. The same ten moments (five with v, five without) give f = 0.576, 0.666 or 0.600 depending on their
   order (`evm_s7` B1). Order independence is the first test the landscape note derived from this field.
3. Float64. Subtracting 999 of 1000 added increments in another order reproduced the remaining one exactly in
   0.0% of 2000 trials (max error 1.8e-12), and after retracting everything the slot was negative in 50.7% of
   trials, which violates w ≥ 0 (`evm_s6` B).

Severity: serious (T9 fails; claim (iv) of §9 is false as stated).

Smallest fix. Store only raw counts of primary events (integers or fixed point): n+, n- per (tie, source) and
N per (unit, source). Apply every weight (trust, completeness, cap, base rate) at readout, and keep every
learned source parameter as a function of the raw counts. Then retraction is a decrement and is identical to
the life without the experience, and the same experiences in any order give the same beliefs (`evm_s7` B2:
4875 experiences in a random order, maximum difference 0.0). T9 must compare with the life in which the
experience never happened, not with a snapshot.

#### E9. An evidence increment takes its provenance from one end and its size from a settled activation

Attacked text. §2.5: "w⁺[π] += g·a_u (π = provenance of the confirming event)".

Evidence. (a) While idle the mind imagines u; in the next moment the real event v arrives. The tie u → v
receives sensed evidence at full trust: after five such coincidences c = 0.83 and e = 0.92 that u is followed
by v, although u was never sensed (`evm_s6` D). The same rule books evidence for a category unit that is active
only through ABSTRACT (E3 point 3), for an antecedent that settling ignited, and for the mind's own utterance
heard back on a channel. (b) For lag-0 ties the weight a_u is the settled activation, and settling lets the
presence of v raise u. If a_u is a+ when v is present and a- when it is absent, the frequency converges to
p·a+/(p·a+ + (1-p)·a-): 0.667 instead of 0.5 for a+ = 1, a- = 0.5 (`evm_s6` E). Belief raises activation,
activation raises the weight of the confirming evidence, evidence raises belief.

Severity: serious.

Smallest fix. An increment has two ends and is as real as the less real of them: book it under the less
trusted of the two sources, and write nothing if either end is present only by inference or imagination. Use
the event-driven presence of each end as the weight, not the settled activation.

#### E10. The evidence of ABSTRACT ties is undefined, and under C3 it is degenerate

Attacked text. C3: "Every tie carries EVIDENCE, and evidence changes at every moment by the same local rule";
§4: "discounted by the ABSTRACT expectations".

Evidence. A category unit is never carried by an event. If "occurred" means carried by an event, every
perception of an instance is a failed prediction of its category and membership erodes with use (the
single-coincidence curve of E1: e about 0.09 after 100 perceptions). If "occurred" means active in the focus,
the category is activated through the very tie being scored and every ABSTRACT tie goes to e = 1 regardless of
whether the instance conforms. In neither reading can a membership be revised by what the instance turns out
to be, and §4 has no defined value for its discount. An evidence-free membership has e = 1/2 and passes half
of the evidence; a disbelieved one (e = 0.2) still passes a fifth.

Severity: serious (the prototype has to invent this rule; §4 depends on it).

Smallest fix. The evidence of ABSTRACT i → g is (a) the occupancy counts of the slot that created it and (b)
the agreement of i's own primary evidence with g's rules built without i, by the same referee rule as for any
source. Membership weight is m_i = max(0, 2·e(i→g) - 1), zero for unknown or disbelieved membership. Pooled
over a category this is rho_g, the quantity that sets the rule's horizon (E4).

### MINOR

#### E11. W = 0 gives NaN, and NaN wins kWTA
§1: "frequency f = W⁺ / W", "expectation e = c·(f − ½) + ½". With W = 0 (a tie after retraction of its only
experience, a PART or ABSTRACT tie at creation, a candidate created while g = 0), f = 0/0 and 0·NaN = NaN.
`np.argmax` and `torch.topk` both return the NaN unit first (`evm_s6` C). Fix: e = (W+ + k·b)/(W + k), the
same quantity, defined everywhere.

#### E12. Birth values and gauge
§1: "one told piece of evidence with τ_told = 0.3 gives c = 0.3/1.3 = 0.23". By the same section tau is the
expectation of a trust tie, which is 1/2 without evidence, for every class: a first hearing and a first
sighting both give c = 0.333, and imagination starts at the same trust as the senses (`evm_s4` c). The trust
tie of "sensed" has no referee. f is unchanged when all tau are scaled together and c is not, so the unit of
evidence is undefined until one class is fixed. Fix: declare birth pseudo-counts per class (told: 1.3 positive,
0.7 negative gives rho = 0.3 at base rate 1/2) and fix rho = 1 for the mind's own closed senses and acts.

#### E13. The base rate 1/2 is hard-wired into e
§1: "expectation e = c·(f − ½) + ½ (what the mind acts on)". A tie with no evidence is expected at 50%, in an open
alphabet where almost no tie holds. Fix: e = (W+ + k·b)/(W + k) with b supplied by the category (E4) and, at
the root, by the occurrence rate of the target on its channel. In the integrated life this matters little once
silence is read through chi (0.047 against 0.048); it matters for unknown pairs and for chance correction
(E7). The settling weight (e - 1/2)·2 has the same issue and is the dynamics reviewer's subject.

#### E14. Ambiguities in the Up rule
§4: "Σ_i min(W_i, cap)". Capping each side separately turns an instance with W+ = 100, W- = 1 into a 50/50
vote; cap the total and split by f. "each hold" reads as if one member without the tie blocked the rule; say
"those that hold". The words "provenance "inferred"" on the Down product suggest a second discount by
tau_inferred on top of lambda; say which.

#### E15. Acting on evidence that is counted once per settling iteration
§2.4: "Act units accumulate their incoming expectation over the settling iterations; an act is emitted when its
accumulated evidence crosses the threshold θ". The same expectation is added T ≤ 6 times; the effective
threshold is θ/T and depends on how long settling ran. Accumulation is a decision rule for independent
samples. Fix: compare the settled expectation with θ, or accumulate across moments.

#### E16. Creation weight and lag
§2.5 creates a candidate tie with g·a_u·a_v and later confirms it with g·a_u. Use the same product in both
places (it also keeps n+ ≤ N). A conclusion reached through L lag-1 ties is not a lag-1 tie (E5 point 6).

#### E17. §7 has no equation for "outweigh"
§7: "As its own outcome evidence accumulates it can outweigh them". The imitated tie ("in situation S, others
do X") and the own tie ("(S, X) → relief/harm") are different ties, and §2.4 does not say how the two are
compared. If they are summed as evidence, detaching needs more own bad outcomes than tau_imitated times the
number of times others were seen doing X (50 for 100 observations at tau = 0.5): the unbounded support of E4
again. Fix, with the readout of CA4: the imitated expectation is the base rate of the own (state, act) tie,
e_act = (W_own+ + k·b_imit)/(W_own + k). With no own evidence the mind copies (e_act = b_imit); one or two own
bad outcomes bring e_act below 1/2 however often others were watched (b_imit/2 after one); the imitated tie
keeps all its evidence, so "others do X" stays known. T7 depends on this. Worked by hand, not simulated.

---

## 2. Corrected arithmetic

One principle: **what is stored is a raw count of primary events; every other number is a readout; no readout
is ever an input to itself.** The alternative proposal in this folder (`belief-graph/bg_core.py`) already had
the batch form of this (a ledger, a full re-solve, leave-one-out messages, a series coupling `par`, a discount
for a source repeating itself). What follows is the incremental form: sums maintained by delta-updates.

The rules are numbered CA1 to CA9 (corrected arithmetic) to keep them apart from the design's commitments
C1 to C3.

**CA1. State (stored, additive, logged per moment, integer or fixed point).**
- per tie t = (u → v, lag) and source unit s: n+[t, s], n-[t, s];
- per unit u and source s (a teller, a sense channel): N[u, s], the presence-weighted number of moments in
  which u was present by an event while s reported at that lag;
- nothing else is evidence. Structure (units, ties) is created as in §3.

**CA2. LEARN (replaces §2.5 step 5).** For a unit u carried by an event (presence p_u, source s_u) and an event
v (presence p_v, source s_v):
- N[u, s_v] += g·p_u;
- n+[u → v, s*] += g·p_u·p_v, where s* is the less trusted of s_u and s_v;
- an absence report of v (x = 1), or an occurring v' with learned exclusion
  x(v', v) = c(v'→v)·max(0, 1 - f_up(v'→v)/b_v) > 0, gives n-[u → v, s*] += g·p_u·x;
- units that are active without an event of their own, and events of provenance imagined or inferred, write
  nothing.
No increment depends on e. There are no negative scatter-adds for silence.

**CA3. Evidence of one source on one tie.**
- E_s = max(n+ + n-, chi_s·N[u, s]) and f_s = n+/E_s;
- W_s = par(E_s, K_s) with K_s = k·rho_s/(1 - rho_s) (infinite for own closed senses and acts).

**CA4. Readout of a tie.** W+ = Σ_s W_s f_s, W- = Σ_s W_s (1 - f_s), W = W+ + W-;
c = W/(W + k); e = (W+ + k·b)/(W + k), with b from CA7. For predicting what a channel shows next (PREDICT,
surprise, chunking) the same counters are read with chi = 1: f_occ = n+/N. On a closed channel the two
readouts coincide; elsewhere "holds" and "is mentioned" differ exactly by the completeness.

**CA5. Completeness of a source** (how much its silence means):
chi_s = Σ (n+ - 1) / Σ (N - 1) over the ties s has asserted; chi = 1 for a channel declared by its adapter to
report the whole state of what it measures; where a closed referee exists, the recall of s on ties the referee
holds true is the cross-check. Two running sums per source.

**CA6. Reliability of a source** (how much its assertion means). Trust tie (s → holds) with counts (a_s, d_s):
for every tie on which s has a standing claim (f_s, c_s = E_s/(E_s + k)) and strictly more primary sources
have evidence (f_ref, c_ref) that does not contain s,
a_s gets c_ref·c_s·f_s·f_ref and d_s gets c_ref·c_s·f_s·(1 - f_ref), at most one unit per tie;
A_s = (a_s + α0)/(a_s + d_s + α0 + β0); rho_s = max(0, (A_s - b)/(1 - b)).
Birth: (α0, β0) per class, declared; rho = 1 for own closed senses and acts (gauge).

**CA7. Up and Down (replace §4).**
- Up: S_g± = D_g± + Σ_j m_j·min(W_j, k)·f_j±, from the members' own evidence (CA4 without base rate); D_g is
  evidence about the category itself ("all g are v"), read by CA3; one member is at most one vote.
- For member i: S_{g\i} = S_g minus i's own term (exact).
- Base rate handed down: b_i = (S+_{g\i} + k_g·b_g)/(S_{g\i} + k_g), where b_g is the base rate of the rule
  itself (from its parent, or the occurrence rate of v, or 1/2).
- Horizon: k_g = k·(1 - rho_g)/rho_g, where rho_g is the chance-corrected agreement of members' own evidence
  with rules built without them (the pooled evidence of the ABSTRACT ties, by CA6). For members that either
  have v or do not, this is exact: the chance-corrected agreement of two members is the intraclass correlation
  of a Beta-Bernoulli model, 1/(k_g/k + 1), at any base rate.
- Several categories: use the one with the largest S_{g\i}/(S_{g\i} + k_g).
- Optional, for conviction: h_i = par(S_{g\i}, K(rho_i)) added as evidence at frequency f_{g\i}.

**CA8. Imagination and R (replace the last four sentences of §6).** Imagining creates structure and activation.
R(i → v) is e(i → v) of CA4 with the base rate of CA7; for a new conclusion R = b_i. Nothing is written. Primary
evidence that arrives later referees the rule by CA6; that is the self-trust "per domain" of §6, and it sets
k_g. A chain of EXPECT ties is a source like a teller: it claims a frequency (the product along the chain) with
evidence at most K(rho_L), rho_L learned by CA6 for chains of that kind (this last sentence is specified here,
not simulated).

**CA9. Retraction.** Decrement the raw counters of the moment; recompute the cached sums that contain them.

**Invariants** (each is a test; `evm_s7` B checks I2 to I4 numerically).
- I1 No runaway. The evidence W of a tie is a function of the raw counts on that tie and of source parameters.
  Every base rate is a convex combination of observed frequencies and the root prior, so it stays in [0, 1],
  and it never enters W. Hence idle time changes no conviction, and the recursion instance → rule → instance
  has gain 0. With the optional h_i the bound is K(rho_i), and h_i is computed from primary counts only.
- I2 Hearsay is bounded: evidence from one source on one tie is below K(rho_s), so c ≤ rho_s.
- I3 The same experiences in any order give the same state; retraction equals the life without.
- I4 One own closed observation against, with no other own evidence, gives e = k·b/(1 + k) ≤ 1/2 for any
  category size.
- I5 A source whose assertions hold no more often than the base rate contributes nothing.

**What the arithmetic keeps from the design.** First hearing c = 0.231, e = 0.615 (identical); a second
independent teller 0.375; an own sighting 0.565; a contradiction by an own sighting moves f to 0.23 and raises
c to 0.565 (identical to the design: "Contradiction does not delete"); a rule from six other members lifts the
expectation of an unseen instance to 0.945 at once (`evm_s7` A).

**What each part buys** (the integrated life of `evm_s8`: 6000 experiences, pairs never looked at, mean of 5
lives; "always 1/2" = 0.250, "always the base rate" = 0.217):

| arithmetic | Brier | right side of 1/2 | conviction read |
|---|---|---|---|
| design as written | 0.238 | 68.6% | 0.873 |
| design without imagination | 0.222 | 64.5% | 0.841 |
| design, silence simply not counted | 0.354 | 38.7% | 0.818 |
| design, silence not counted, base-rate support | 0.410 | 38.2% | 0.455 |
| corrected, chi refereed by the closed sense | 0.048 | 94.6% | 0.755 |
| corrected, chi = the tellers' own repetition rate (CA5; no closed referee for chi) | 0.038 | 97.1% | 0.707 |
| first corrected row without completeness (chi = 0) | 0.271 | 51.6% | 0.324 |
| first corrected row, one slot for both tellers | 0.064 | 94.9% | 0.560 |
| first corrected row without category base rate | 0.064 | 93.0% | 0.755 |
| first corrected row, reliability = hit rate | 0.050 | 94.8% | 0.802 |
| first corrected row, base rate fixed at 1/2 | 0.047 | 94.7% | 0.708 |

The design's 68.6% is about the share of false pairs (68%): it has learned to say "no". Its curve over the life
is flat (0.246 at 250 experiences, 0.238 at 6000) while its conviction climbs from 0.28 to 0.87; the corrected
curve goes from 0.178 to 0.048 with conviction from 0.20 to 0.76. Completeness is the part that cannot be left
out. The other parts cost little in this particular life and are decisive in the lives built to test them
(noise sources and a liar in E6/E7, exceptions and mixed categories in E4).

**Cost.** Evidence per (tie, source) instead of six class slots (sparse, a few sources per tie); one counter
per (unit, source); two or three numbers per source; cached category sums updated by differences. Per moment:
O(F) opportunity updates per reporting source and O(F²) positive increments, with no negative updates.

**Order of adoption for the prototype.** (1) Raw counts and the completeness readout (CA1 to CA5): this repairs
E1 and E8 and is the largest single gain. (2) No writes from imagined or inferred activity (CA2, CA8): E2, E9.
(3) Down as a leave-one-out base rate (CA7): E3, E4, E5, E10. (4) Evidence per source with the series coupling
and chance-corrected reliability (CA3, CA6): E6, E7, E12.

**Relation to the dynamics review (`dynamics-scale.md`, same folder).** The two reviews agree where they
overlap and can be merged; four points need a decision.
- Counters. Its "exposure" (fix 4: non-occurrence is not written, W = exposure of u while v's channel was live)
  is the counter N of CA1, and its e measured against the base rate b_v is the occurrence readout f_occ of CA4.
  That repairs relating and predicting. It is not yet a belief: on one text channel a true fact that shares its
  subject with four others reads 0.2 for ever. Its D15 expects the silence problem only "as soon as a second
  channel exists"; E1 point 2 shows it on a single text channel. The completeness factor of CA3/CA5 is the
  missing piece, and the two fixes use the same counters.
- Imagination. Its D8 fix keeps one write per distinct conclusion ("set, not add") with weight ι·R. That stops
  the runaway of E2 point 3, not the rest: R is 1/2 for every new conclusion, and a conclusion with R < 1/2 is
  still written on the positive side (E2 points 3, 4 and 6). CA8 writes nothing.
- Support. Its D18 fix leaves the member's own share out of the rule and keeps λ·e·W(g→v) otherwise. That
  removes the loop, not the size effect: an exception still needs more than λ(m - 1) own observations (E4).
- Horizon. Its "k is the learned trust in young evidence" and the rule horizon k_g of CA7 are the same idea in
  two places; one mechanism should serve both.

---

## 3. Tests to add or change

- T1: control sentences about the same subject that do not mention the property must not move the belief
  while chi·N ≤ n+ (the first few always qualify), and must always move it less than an explicit contradiction
  does; an explicit denial must lower e; the same teller repeating must add less than a second teller.
- T3: vary the category size (3, 30, 300); an exception must flip after at most two own observations at every
  size; add a mixed category (no rule) and require that members' own knowledge is not degraded.
- T4, T5: asking the same source again must give diminishing relief; N idle steps must change no stored
  counter; report (conviction read) against (evidence lived).
- T7: one liar and one honest teller in the same class; the honest one must stay usable.
- T8: calibration of R on conclusions that are new at the time they are constructed, per domain, over a life;
  the ablation to compare against is "imagination that writes evidence" (lock-in).
- T9: compare with the life in which the experience never happened; shuffle the order of a stream and require
  identical beliefs.
- From the landscape note: no runaway (feed the rule back 1000 times), noise sources (ten at chance must not
  move a belief held from one good source).

---

## 4. Limits of this review

Settling and kWTA were not simulated (a = 1, g = 1 throughout), so interactions between evidence and attention
beyond E9 are not covered. Chains of EXPECT ties in imagination, multi-level category hierarchies, collusion
between tellers and non-stationary worlds were not simulated. Five rules are specified and only worked by
hand: choice among several categories (`evm_s2` b), the learning of negation as a unit (CA2), booking an
increment under the less trusted end (CA2), a new source inheriting the trust of its class as a base rate (E6),
and imitation as the base rate of own action (E17). Neither arithmetic lets evidence decay, so a
well-evidenced belief moves slowly when the world changes. The constants of the corrected arithmetic (birth
priors 1.3/0.7 and 1.5/0.5, pooling strength 2, one vote per member) were set once and not tuned; each regime
was run on generated worlds with a handful of seeds, not swept. The per-source completeness is a single
number; a source that mentions some true things rarely will have them under-believed (bounded by rho_s).
An object whose state alternates on a channel cannot be told apart from an object with two co-present
properties shown alternately unless the adapter declares the channel a state channel: that one bit per channel
is the only declaration the corrected arithmetic needs.

## 5. Files

All under `newcore/attack/`. Run with `python3 <file>`; outputs are saved as `<file>.out`.

| script | subject |
|---|---|
| `evm_common.py` | both arithmetics' primitives |
| `evm_s1_closed_world.py` | E1: silence, denial, trust in an honest teller, world test |
| `evm_s2_updown.py` | E3, E4: loop gain, twins, cap bypass, exceptions, calibration against exact Bayes |
| `evm_s3_imagination.py` | E2: lock-in, deposit, sign, calibration of R |
| `evm_s4_trust.py` | E6, E7, E12: noise sources, class slots, gauge, repetition, base rate |
| `evm_s5_reality.py` | E5: R on graphs with shared ties, self-inclusion |
| `evm_s6_boundary.py` | E8, E9, E11: retraction, float, W = 0, provenance, activation weight |
| `evm_s7_corrected.py` | corrected arithmetic on the design's claims; invariants |
| `evm_s8_integrated.py` | one life with all pieces, both arithmetics, ablations (about 45 s) |
| `evm_s9_exclusion.py` | E1: refinement by a compatible and an incompatible statement; learned exclusion |

## 6. What survives

- The readout of §1: additive evidence, c = W/(W + k) with k = 1, e as the quantity acted on. In the form
  e = (W+ + k·b)/(W + k) it is defined everywhere and takes a base rate.
- "First hearing is weak by construction" and "Contradiction does not delete: it adds W⁻, f moves, c rises":
  both hold in the corrected arithmetic with the design's own numbers, once the first has a declared birth prior
  and the second a route for explicit contradiction.
- Provenance-resolved evidence (§1 of the design) and trust applied at readout rather than at write time.
  This is what allows leave-source-out, retroactive re-weighting when a source is found unreliable, and the
  detachment of §7. It survives with a finer index (source unit instead of class).
- Trust as an expectation learned from confirmation by more primary evidence, and self-trust per domain
  (§6). The principle is right; the weight derived from it (hit rate, linear) and the place it is used
  (a deposit) are not.
- The cap in the Up rule: one instance is at most one vote. That is the correct idea and is kept (CA7).
- The intention behind leave-path-out: judge a construction by evidence it did not use. It becomes
  leave-one-out on evidence.
- Logging increments for retraction (§10): right once the increments are raw counts.
- §7's separation of "in situation S, others do X" (imitated) from "(S, X) → relief/harm" (acted): these are
  different ties with different provenance, and nothing in this review touches them. The sentence "it can
  outweigh them" still needs an equation (E17).
- Induction from instances to a category and immediate transfer to unseen instances (§4 Up; base rate down).
