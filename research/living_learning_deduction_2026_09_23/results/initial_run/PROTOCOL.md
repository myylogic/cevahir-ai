# Round 21 — internal deduction, reusable computation and validity

Muhammed Yasin Yılmaz · 2026-09-23 · Local preperformance protocol

This protocol is written before running the new engine. Preserve rounds 1–20 and the book byte-for-byte. No commit, push, publication or main-system integration. This is not a proposed solution of general living-learning.

## Questions and hypotheses

Can runtime internally checked deductions alter future problem-solving cost without adding information about the environment? Does their construction pay for itself over the whole life? What happens when the premises change? Learned rules, DPLL, explanation-based learning and knowledge compilation are prior art, not claimed inventions.

H21.1: a retained entailed clause can reduce service work on a previously unseen assumption combination. H21.2 (stronger, rejectable): total life work including acquisition/checking is below a fresh solver. H21.3: checking proof dependencies prevents incorrect reuse after premises are removed, while blind persistence can fail. H21.4: scoped persistence reproduces recomputation from the same chronological history, at different cost. H21.5 (rejectable): it beats exact upfront compilation of the complete query-interface truth table. Hypotheses are distinct; one success does not entail the others.

## Worlds and queries

24 deterministic seeds, six public Boolean query variables and four independent guarded gadgets, each with six private variables: 30 variables total. Each gadget consists of the standard unsatisfiable three-pigeons/two-holes clauses, with a two-literal guard appended to every clause. Its projected meaning is that guard. Distinct guard variable pairs and signs are generated with a common satisfying public assignment. Literals use canonical variable order and clauses use content-hash order, so process reload preserves the solver order. The learner receives only flat clauses with immutable content identities, not guards, pigeonholes or target answers. The public query vocabulary is given to all methods.

One life has four blocks of 24 queries: initial base, unchanged base, a base in which one entire gadget is replaced by its guard with both signs reversed, and return to the original base. All queries independently choose three distinct public variables and signs uniformly; no target-generation hints or teacher labels. Queries can repeat; report novel combinations separately. The replacement can change satisfiability structure, including making a base globally inconsistent; report this rather than silently removing cases. Every method sees identical bases and queries. 24 seeds are the independent paired units, not the number of queries/methods.

## Methods

1. `fresh`: solve each query against the current base from scratch.
2. `scoped`: retain checked clauses negating subset-minimized unsatisfiable query assumptions. Minimize by deterministic single-assumption deletion. A separately implemented proof checker checks an UNSAT search tree; discharge all surviving assumptions. Recursively flatten proof dependencies to immutable original clauses. A lemma is active only when all of those clauses remain in the current base. Retain dormant lemmas for a later return.
3. `blind`: same acquisition, but retain all lemmas regardless of current dependencies. This deliberately unsound control is evaluated against an independent oracle, whose labels are never fed back. It is not a deployable candidate.
4. `erase`: same learner as scoped, but delete all lemmas whenever the base changes.
5. `replay`: recompute scoped learning from the complete chronological base/query history before every answer, including actual changes and returns. Count all repeated work. Answers, newly learned clauses, and final logical state must match scoped exactly.
6. `full_table`: whenever the base changes, solve all 64 complete public assignments against the flat base, then answer by scanning compatible satisfiable rows. Include all construction and query scans. This control already has the full declared public query interface, just as the other methods do. It does not receive the hidden guards.

Learners answer first, then may construct a lemma from the same query. No oracle feedback trains them. Fresh solving with the same base and retained history but lemma use disabled is a paired causal probe; its answer/service cost equals `fresh` because this deterministic solver otherwise does not inspect that history. This is not a seventh supposedly stronger competitor with free acquisition.

## Costs, limits, validation

Report virtual work by stage: service solving, lemma construction, proof checking, dependency checks, state writes and full compilation/replay. Unit: inspected literal/header, assignment/branch/proof-node operation, checked dependency, or written certificate/history byte (table construction charges seven scalar writes per row). These heterogeneous counts form an explicit virtual proxy, not complete CPU, memory bandwidth or Python overhead. Do not infer wall-time superiority. Store actual serialized state size separately; proof/history bytes matter. The unbounded solver gives answer correctness and cost. Predeclared service thresholds 500, 2,000 and 10,000 virtual operations are retrospective completion counts, **not** a trained system actually learning under interrupted budgets. Also report whole-life work, without hiding acquisition after prediction. JSON diagnostic serialization and general Python bookkeeping are excluded. Base storage is common; reported additional state excludes the common current base except the replay history copies.

Independent truth reference: enumerate each gadget's 64 private assignments for each public assignment, then combine disjoint private blocks; enumerate all 64 public assignments for each query. Verify the gadget projection rather than assuming it. Independent proof checker validates unit reasons, conflict leaves and both branches. Check all accepted scoped/erase lemmas against independent public truth sets and check active scoped lemmas at every use. Blind invalid lemmas must be measured, not asserted sound.

Additional checks: random small CNFs against full enumeration (including SAT witnesses and UNSAT proofs), query-scope counterexample, proof tampering rejection, exact deterministic rerun, fresh-process snapshot/continuation for learned methods and preservation manifest. Record failures and implementation repairs; do not retune world generation or budgets after seeing performance.

## Scope

SAT syntax, solver, checker, implication-based lemma language, query interface, minimizer, retention policy and resource meter are supplied by the researcher. The system does not invent these. No new environmental labels arise in stationary blocks; changed base clauses ARE external information. These are small constructed propositional worlds, not concepts, emotion, neurobiology, language or AGI. Positive results would establish a component of computational consolidation; negative lifetime/strong-control results narrow that claim. The wider living-learning question stays open.
