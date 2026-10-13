# Round 22 — cross-domain mapping, composition and crossing partitions

Muhammed Yasin Yılmaz · 2026-09-23 · Local preperformance protocol

This is a new direction after rounds 1–21. Preserve all earlier files and results. No commit, push, publication or main-system integration. The protocol is written before the engine and results.

## Main question

Can independently learned relational domains be joined by experience so that a later task, never directly taught, becomes computable? Does the same stored event need multiple crossing organizations rather than one nested semantic tree? A success is not a general living-learning principle: symbols, finite relations, candidate language and task format are supplied.

## Exact worlds

Twenty-four deterministic seeds generate six latent objects. The X domain exposes a random permutation of object names and a six-cycle successor relation. The Y domain exposes an independent random permutation and three disjoint two-cycles. Two public labels are supplied in each domain, but their underlying partitions are crossing: X groups are `{0,1,2}/{3,4,5}` and Y groups are `{0,2,4}/{1,3,5}` in latent object IDs. A single rooted hierarchy cannot represent both partitions as unique nested nodes; the experiment measures whether a learner retains both views.

The learner receives source transitions in X and Y separately, then bridge episodes pairing X and Y symbols for a growing subset of latent objects. No common target interface, object identity or mapping is given. Bridge pairs are presented in a fixed deterministic order selected from the seed. Source and bridge histories have the same total count across methods. Four blocks each contain six held-out queries: (1) cross-domain conjugate successor `X_i -> X_f -> Y_map`, (2) reverse composition `Y_i -> Y_g -> X_map`, (3) crossing-view category pair `(X_group,Y_group)`, and (4) repeated original blocks after one new bridge pair. Target query answers are withheld until scoring; target labels are not used by the mapping methods. Each block has six queries with no duplicate exact input.

The world generator independently enumerates the 720 bijections from X to Y symbols and checks every answer by latent object semantics. It also independently checks the two partition outputs and the inability of a single nested tree to contain both crossing subsets.

## Predeclared hypotheses

H22.1: source relations plus bridge experience can lower held-out composition error relative to separate-domain lookup. H22.2: a persistent candidate mapping can reduce later service work relative to recomputing all mappings from the complete history, with identical posterior predictions. H22.3: a wrong mapping can create confident systematic errors; disabling the learned mapping must remove those errors. H22.4: both crossing partitions must be retained as separate views or an equivalent non-tree representation; one fixed hierarchy loses at least one task. H22.5 (strong): learned mapping beats a one-shot full-history synthesizer with the same evidence and resource accounting. This is expected to be difficult and is explicitly falsifiable.

## Methods

1. `separate`: use only each domain's local transition/category records; no bridge composition.
2. `persistent`: maintain the set of all bijections consistent with bridge records and update it once per new episode. Enumerate candidate mappings only when needed, then cache the exact candidate set keyed by bridge history.
3. `recompute`: reconstruct source relations and all bridge-consistent mappings from the full chronological history before every query. It is a same-evidence causal control.
4. `oracle`: receive the true X-to-Y mapping only as a ceiling; its information cost is reported as an external gift.
5. `wrong`: receive a deterministic bridge-consistent mapping selected by the first candidate but use it as certain, to test confident error under partial evidence.
6. `disabled`: run the persistent state but disable cross-domain composition and answer only from local records; stored evidence remains and can be re-enabled in the final block.
7. `tree`: force a single rooted hierarchy over the six objects for both category views; when a crossing partition cannot be represented, it must abstain or copy objects. Copy count and abstentions are charged.

All methods receive identical raw source and bridge experiences in identical order. `oracle` and `wrong` are explicit controls, not fair information-matched competitors. No method receives latent IDs, relation names, target answers, or the true mapping except `oracle`. A full-history synthesizer is allowed to enumerate all 720 maps; this prevents a history-blind baseline from manufacturing a gain.

## Metrics and costs

Report exact correctness, abstention, confident error, candidate-set size, and mean proper score over the four blocks. Count source parsing, bridge filtering, permutation checks, cache writes, state serialization bytes, candidate scans and query composition operations. Report service-only and whole-life totals separately. A virtual operation is not CPU time. Include bridge acquisition and tree-copy cost. Count independent seeds, not queries, as experimental units.

The primary comparisons are persistent versus recompute on identical answers and persistent versus separate on block-1/2 held-out composition. Strong-control comparisons are oracle and recompute; do not claim a win against an oracle as learning. H22.5 is rejected if `persistent` is not cheaper than `recompute` after all cache/filter costs, or if `oracle` is materially better on held-out queries.

## Verification and limits

Independent latent enumeration checks all 720 maps and every query. A fresh-process snapshot at cuts 6, 12, 18 and 24 must reproduce exact candidate sets, answers and costs. Permutation-renaming invariance uses 12 additional transformations. Tamper tests alter one bridge pair and one cached candidate; the verifier must detect both. Full replay and results are deterministic. No significance test across 24 constructed seeds is claimed. The task family has six objects, finite relations, noiseless observations and a supplied symbol grammar; it does not test perception, open-ended concepts, biological chemistry, consciousness or AGI.
