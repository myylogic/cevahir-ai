# Zihin v0 under the lens of DYNAMICS, STABILITY, GROWTH AND COST

Object: `../LEAD_DESIGN.md` (v0, 2026-11-14), judged against `../VISION.md`.
Method: every claim below was run (scripts and raw outputs in this folder, list in §6) unless it is marked
"derivation". No product code was written; `zsim.py` is an instrumented attack tool.

## 0. Verdict

The substance of the design holds (units and ties that grow, additive evidence, a small focus, composed units).
The rule (§2.5) and the settling equation (§2.2) as written do not: in a 10^5-moment life the mind stops perceiving
(0.8 % of its input symbols reach the focus), recalls nothing (association recall 0.00, true associates rank
below units it never met), feels relief in 3 of 100 000 moments, and writes a 145 GB retraction log around a 51 MB
core. Five findings are fatal as written, ten serious, four minor. Every fatal one has a small repair that needs
no embedding table, no frozen part, no softmax, no external memory and nothing domain specific; with the repairs
the same engine reaches association recall 0.90–0.96 at 10^5 moments (0.82–0.89 with bounded bags) and runs at
160–320 tie reads and 0.5–0.8 ms per moment (pure Python, one thread), bounded by F·K and not by the number of units,
up to 1.1·10^6 units.

## 1. What was simulated, and how the text was read where it is silent

Toy world (`zsim.World`, random ids only): 3 000 entities (Zipf), 30 categories, 12 relations, 400 values; a fact is
(entity, relation) → value; 90 % of facts are the category default, 10 % are exceptions of one entity; a moment is
one statement whose three symbols arrive together; 5 % of statements carry a random value; the next statement stays
on the same entity with p = 0.5. Read-outs: (i) association recall: present the entity alone, how many of the
values it was met with are among the 5 best supported value units; (ii) relation recall: present (entity,
relation), is the true value the best supported; (iii) share of the moment's external symbols that are in the
settled focus.

Engine (`zsim.py`): §1–§2 literally: units, EXPECT ties lag 0/1 with (w⁺, w⁻), f, c, e, settle with kWTA_F
(F = 7, T ≤ 6), LEARN always on, candidate ties for focus pairs, one composed unit per distinct event tuple,
interoceptive events of §2.6 (surprise, relief, load, conflict, each quantised to a few level symbols). One
provenance class (sensed, τ = 1). Not in the engine: ABSTRACT up/down and acts (slots, chunks, idle, drive are
simulated in separate scripts).

Readings (all tested for sensitivity where it could matter):
- `in` = 1 for every evented unit at every settling iteration; a(0) = decay × previous focus; decay = 0.5
  (0.2 and 0.9 give the same picture, table in D1); kWTA keeps the F largest positive values, then a is clipped to
  [0, 1]; settling stops early when the focus and its values stop changing.
- "v occurred" = v is carried by an event of the moment (or is a composed unit all of whose parts are).
- lag-1 candidate ties are created for (u in focus(t−1), v in focus(t)); the text only says "pairs in the focus",
  and without this reading no lag-1 tie can exist (prediction, chunking and imitation would have nothing).
- a tie's `use` is incremented when its source is in the focus.

The two columns "design" in the tables are the design as written with and without the interoceptive events of §2.6,
so that no result depends on how interoception is quantised.

## 2. Findings, ordered by severity

Format: **Attacks** (exact text) / **Evidence** / **Severity** / **Smallest fix** (and whether the fix was run).

### FATAL

#### D1. The reference ½ and the e-weighted negative update turn real associations into inhibition

**Attacks.** §1 "expectation e = c·(f − ½) + ½ (what the mind acts on)"; §2.2 "Σ_u a_u·(e_{u→v} − ½)·2 through
EXPECT(lag 0)"; §2.5 "v was predicted, did not occur: w⁻[π] += g·a_u·e_{u→v}"; §1 "frequency f = W⁺/W (how often
it held)".

**Evidence.**
1. Derivation. If v accompanies u with probability p, the rule's fixed point is f = p / (p + (1 − p)·f), i.e.
   (1 − p)f² + pf − p = 0. So f is not "how often it held": p = 0.01 → f = 0.096, p = 0.2 → 0.39, p = 0.5 → 0.618
   (a coin is believed at 0.62). f crosses ½ exactly at p = 1/3, hence the settling weight (e − ½)·2 is negative for
   every tie with p < 1/3. Monte Carlo agrees to 3 decimals (`sim_rule_bias.py`):

   | p | 0.001 | 0.01 | 0.05 | 0.1 | 0.2 | 1/3 | 0.5 | 0.9 |
   |---|---|---|---|---|---|---|---|---|
   | f (design rule) | 0.031 | 0.095 | 0.205 | 0.283 | 0.391 | 0.500 | 0.618 | 0.908 |
   | settling weight | −0.94 | −0.81 | −0.59 | −0.44 | −0.22 | 0.00 | +0.24 | +0.82 |

2. A concept with m equally likely facets (VISION 3b: "each encounter adds a facet"): weight concept → facet is
   +1.00 (m = 1), +0.24 (2), 0.00 (3), −0.13 (4), −0.28 (6), −0.44 (10), −0.66 (30). The richer the concept, the
   harder it suppresses each of its own facets. With the base rate as reference the same counts give
   +0.23 (m = 4), +0.09 (10), +0.03 (30).
3. Engine, 10^5 moments (`run_stream.py`, `stream_literal.jsonl`). Association recall is 0.00/0.00/0.00
   (head/mid/tail entities) at every checkpoint from 10^3 to 10^5; the share of an entity's true values whose
   support is below that of a value never met with it is 0.72–0.89 at 10^3 and 1.00 at 10^5; mean support of a true
   value −2.3. Without interoception: recall 0.09–0.13 at 10^3, 0.00–0.04 at 10^5, "below unrelated" 0.96–1.00. What the
   mind has learned about a thing ranks under what it never saw with it.
4. Which single change repairs it (`run_ablate.py`, 10^4 moments, `ablate_final.jsonl`):

   | variant | input symbols in focus | association recall h/m/t | true below unrelated |
   |---|---|---|---|
   | design | 0.08 | 0.00 / 0.00 / 0.00 | 0.96 |
   | design, no interoception | 0.67 | 0.02 / 0.01 / 0.01 | 0.94 |
   | design, decay 0.2 / 0.9 | 0.08 / 0.07 | 0.00 / 0.00 / 0.00 | 0.92 / 0.85 |
   | only inputs clamped | 1.00 | 0.00 / 0.00 / 0.00 | 1.00 |
   | only unweighted negatives | 0.08 | 0.00 / 0.00 / 0.00 | 0.99 |
   | only base rate instead of ½ | 0.80 | 0.45 / 0.10 / 0.15 | 0.00 |
   | base rate + unweighted negatives | 0.82 | 0.60 / 0.56 / 0.56 | 0.09 |
   | base rate + unweighted negatives + clamped inputs | 1.00 | 0.80 / 0.94 / 0.98 | 0.00 |
   | all fixes of this report | 1.00 | 0.91 / 0.98 / 1.00 | 0.00 |

**Severity.** Fatal. Relating and recalling, the core of §2, give the opposite of knowledge, and it gets worse with
experience (c → 1 with exposure, so every such weight reaches its full negative value 2f − 1).

**Smallest fix (run).** Two symbols and one factor:
- replace ½ by the target's own base rate b_v = (occurrences of v)/(age of v), both already in the unit table
  ("use count, birth time"): e = c·(f − b_v) + b_v, and settle with a_u·(e_{u→v} − b_v);
- drop the factor e in the negative update (count events, do not weight them): then f is "how often it held"
  (0.2003 for p = 0.2 in the same Monte Carlo).

#### D2. The focus is taken over by hubs; the mind stops perceiving its input

**Attacks.** §2.2 "a ← kWTA_F( decay·a + in + Σ_u a_u·(e_{u→v} − ½)·2 … )" (`in` is only one term of a sum that
expectation can outweigh); §2.6 "Internal events are emitted on the interoceptive channels and enter the next moment
like any event"; §2.2 "F … default 7".

**Evidence.** Design as written, same run:

| moments | 10^3 | 3·10^3 | 10^4 | 3·10^4 | 10^5 |
|---|---|---|---|---|---|
| share of external input symbols in the settled focus | 0.22 | 0.11 | 0.065 | 0.031 | 0.008 |
| interoceptive units in the focus (of 4 emitted) | 3.87 | 3.94 | 3.89 | 3.90 | 3.92 |
| focus size | 6.9 | 6.7 | 6.3 | 5.3 | 5.1 |

Trace at moment 5001 (`dbg1.py`): the input is the most frequent entity of the world, its relation and value;
support from the context is −0.75, −0.97, −1.71, so their net input is 0.25, 0.03, −0.71; the settled focus is
five interoceptive level units and nothing else; the same statement presented three times in a row never enters.
Mechanism: a unit present in most moments holds a tie to every unit with P(v|hub) ≈ 0, weight → −1; two such hubs
outweigh `in` = 1. Hubs themselves are supported by everything (P(hub|u) ≈ 1), raw activation reaches 7.4 before
clipping, 18–22 unit-iterations per moment sit at the clip.
Independent of D1: with the weights repaired and inputs clamped, 3 symbols + 1 composed unit + 4 interoceptive
events are 8 evented units for 7 slots, so the number of recalled units in the focus is 0.00 per moment (row
"base rate + unweighted negatives + clamped inputs" of `ablate_final.jsonl`): there is never room to recall.

**Severity.** Fatal as written (perception → 0.8 %); serious on its own after D1.

**Smallest fix (run).** (a) an evented unit is in the focus with a = 1 and cannot be removed in that moment
(expectation may not veto perception); (b) an interoceptive event is emitted when its level changes, not every
moment (an event is a change; the held level stays available as decaying context and by recall; 0.4–0.9 interoceptive events per moment remain). With both, input
survival is 1.00 and 1.3–2.5 recalled units enter the focus per moment.

#### D3. Cost per moment is proportional to the number of units, and the retraction log outgrows the core 2 800-fold

**Attacks.** §2.5 "for every tie u → v with u in the last focus: … v was predicted, did not occur: w⁻[π] += …" (every
out-tie of every focus unit is written at every moment); §2.5 "create EXPECT u → v" for all focus pairs; §10 "Every
experience's evidence increments are logged (moment id → tie increments)"; §10 "Complexity per moment
O(F² + F·deg)".

**Evidence.** The 10^5-moment stream, design as written (`stream_literal.jsonl`; tie bytes at the §10 row of 114 B,
log at 16 B per increment; ms are Python/NumPy, one thread; the portable numbers are the counts):

| moments | units | ties | ties MB | tie reads / moment | evidence writes / moment | log entries (cum.) | log GB | ms / moment | max degree |
|---|---|---|---|---|---|---|---|---|---|
| 10^3 | 1 071 | 18 330 | 2 | 4 230 | 2 953 | 2.95·10^6 | 0.05 | 0.75 | 1 423 |
| 3·10^3 | 2 275 | 43 806 | 5 | 19 354 | 10 015 | 2.30·10^7 | 0.37 | 1.16 | 3 373 |
| 10^4 | 5 130 | 108 856 | 12 | 45 673 | 24 695 | 1.96·10^8 | 3.1 | 1.74 | 8 421 |
| 3·10^4 | 10 137 | 230 690 | 26 | 99 079 | 54 005 | 1.28·10^9 | 20.4 | 2.91 | 18 699 |
| 10^5 | 19 209 | 450 887 | 51 | 239 174 | 111 403 | 9.07·10^9 | 145 | 6.75 | 37 311 |

Same without interoception: 3 349 471 ties (382 MB), 92 395 reads and 29 611 writes per moment, log 2.33·10^9
entries (37 GB), 2.87 ms/moment, max degree 30 805.
- Writes per moment / units = 2.8, 4.4, 4.8, 5.3, 5.8: the cost is Θ(n) per moment and the coefficient still rises.
  "O(F² + F·deg)" is true and empty: deg is the hub degree, and the hub degree is 1.94·n (with interoception) or
  1.61·n (relation words and the most frequent entity, without).
- Degree distribution. With interoception: median 14, p99 22, max 37 311; the ten largest units (all interoceptive
  levels) hold 42 % of all ties. Without: median 119, p90 194, p99 1 388, max 30 805, the top 1 % of units hold 30 %
  of the ties, Hill tail exponent ≈ 1.2.
- The log is what breaks first on any machine: 3 GB after 10^4 moments, 145 GB after 10^5, growing like t·n. The
  core it protects is 51 MB. Knowledge would effectively live in the ledger (VISION 14 forbids exactly that).
- The claim that did hold: tie creation is bounded by the focus, measured 3–43 new ties per moment (bound 91).

**Severity.** Fatal for §10 (an engine whose moment costs Θ(n) and whose log grows quadratically cannot live).

**Smallest fix (run).**
- Non-occurrence is not written. Each unit keeps an exposure counter X_u (per provenance class, per lag);
  a tie stores W⁺ and the value of X_u at its birth; W = X_u − X_u(birth), W⁻ = W − W⁺. This is the same evidence
  as the unweighted rule of D1, at O(1) per focus unit. Measured: 61 writes per moment instead of 21 818 at 10^4
  moments, 63 instead of 111 403 at 10^5; log 6.3·10^6 entries instead of 9.07·10^9.
- Bounded bags (D4 fix) make deg ≤ K: tie reads per moment 292 (flat) instead of 239 174 (growing).
- The log becomes "events and focus of the moment" in a ring buffer (D16).

#### D4. The pruning rule deletes what is rare or new and keeps what is frequent; once the budget binds the mind stops learning

**Attacks.** §8 "pruning (dropping ties with low c and low use, oldest first)"; §10 "Memory grows with ties; bounded
by pruning under a budget given by the body."

**Evidence.** `run_prune.py`: 30 000 moments, the settling repairs switched on (otherwise nothing can be read out),
a second disjoint domain starts at moment 15 000; unbounded reference 447 365 ties. Association recall
head/mid/tail and survival of the ties that carry the facts:

| budget | rule | first domain | second domain | fact ties alive h/m/t | exceptions alive |
|---|---|---|---|---|---|
| none | – | 0.92 / 0.95 / 0.99 | 0.84 / 0.97 / 0.98 | 1.00 / 1.00 / 1.00 | 1.00 |
| 150 000 | design | 0.59 / 0.02 / 0.03 | 0.27 / 0.07 / 0.07 | 0.60 / 0.03 / 0.02 | 0.08 |
| 150 000 | significance z | 0.92 / 0.83 / 0.93 | 0.80 / 0.89 / 0.93 | 0.99 / 0.79 / 0.83 | 0.80 |
| 150 000 | z, no EXPECT tie between a composed unit and its parts | 0.93 / 0.91 / 0.97 | 0.87 / 0.96 / 0.97 | 0.99 / 0.90 / 0.92 | 0.89 |
| 60 000 | design | 0.27 / 0.01 / 0.00 | 0.07 / 0.04 / 0.02 | 0.28 / 0.01 / 0.00 | 0.03 |
| 60 000 | significance z | 0.83 / 0.58 / 0.59 | 0.61 / 0.59 / 0.61 | 0.88 / 0.46 / 0.46 | 0.43 |

Why: c = W/(W + k) and `use` both grow with the exposure of the source, whether or not the tie says anything.
"Low c, low use" therefore selects the ties of rarely met units and every newly made tie; "high c, high use" protects
the well-confirmed non-associations of hubs (hub → rare thing, seen once in 10 000 exposures, c ≈ 1). One third of the
unbounded size is enough for the design's rule to erase 97 % of mid/tail facts and 92 % of the exceptions, and the
second domain is not learnable (0.07 for its mid and tail entities): the mind has become a snapshot of its head. This is the stability–plasticity
failure VISION 14 asks the design to avoid, reintroduced by the forgetting rule.

**Severity.** Fatal once the budget binds (T5, T6 and "an exception stays an exception" fail).

**Smallest fix (run).** Prune by how much a tie says, not by how old or how used it is:
z = |W⁺ − b_v·W| / sqrt(W + k) (observed minus expected co-occurrence under independence, in standard units);
least z first. It is local, uses only quantities already stored, and is domain neutral. Apply it locally as a bag:
each unit keeps at most K ties per form and lag; a newcomer displaces the least significant tie of the same source if
it is more significant (measured K = 64: max degree 128, association recall 0.89 / 0.82 / 0.88 at 10^5 moments
against 0.93 / 0.90 / 0.96 unbounded). No global sort is ever needed. Do not create EXPECT ties between a composed
unit and its own parts (PART says it; they are 58–70 % of the lag-0 ties in this world).

#### D5. The drive rises with knowledge, relief is never felt, and the knobs that earn relief are "forget", "believe harder", "attend less"

**Attacks.** §5 "D(u) = Σ_{ties t at u} use(t)·(1 − c_t) … Relief = −ΔD is felt"; §8 "The same ties and the same rule
learn (internal state shape) + (knob act) → relief"; §8 "pruning … is an internal act whose outcome is felt"; §5 "A
question is therefore an act that the mind has found to be followed by relief".

**Evidence** (`sim_drive.py`, `stream_literal.jsonl`).
1. One tie, p = P(v|u): its contribution use·(1 − c) starts at 0.50 and converges to f/p ≥ 1: 1.00 (p = 1),
   1.24 (0.5), 1.95 (0.2), 4.09 (0.05), 9.56 (0.01). A confirmed belief carries more drive than a fresh guess.
2. Engine: D/tie rises 1.12 → 2.39 over 3 000 moments, D fell in 0 of 30 windows; in the 10^5-moment run ΔD < 0 in
   3 moments of 100 000 (relief is never felt; the interoceptive channel shows the same "no relief, high conflict"
   symbols for the whole life).
3. Knob experiment: one state (2 233 units, 124 165 ties, D = 308 973), the same next 300 moments under each
   internal act; relief = D(before) − D(after):

   | internal act | relief under the design's D |
   |---|---|
   | prune 50 % of the ties | +37 238 |
   | plasticity g: 1 → 4 | +11 315 |
   | focus F: 7 → 3 | −16 446 |
   | prune 10 % | −28 038 |
   | no act (keep living) | −40 614 |
   | focus F: 7 → 12 | −91 682 |
   | plasticity g: 1 → 0.25 | −92 838 |

   The only acts that produce relief are deleting half of memory and counting each observation four times; attending
   to less hurts least. This is what "(internal state) + (knob act) → relief" will learn, and by §8 it transfers to
   every domain.
4. Not sufficient: weighting by a bounded need and counting a pruned tie as unknown makes pruning the worst act
   (−330) but leaves "g: 1 → 4" the best (+121), because any drive that is a function of c can be lowered by
   inflating evidence.

**Severity.** Fatal for §5 and §8 as designed (the act that best "reduces the drive", T4's criterion, is forgetting;
what would carry over to a new domain in T6 is that preference).

**Smallest fix (partly run).** (a) drive = Σ need_t·(1 − c_t) with need a recency-weighted rate bounded by 1, not a
count; (b) relief of a moment is computed on the ties that existed before it, and a pruned tie counts as c = 0 for as
long as it is needed (run: pruning 50 % then gives −330); (c) g must not multiply evidence: one event is one unit
of evidence, otherwise conviction is a knob (run: needed, see 4.); what plasticity may control is which candidates
are admitted; (d) better anchor (derivation, cited by §9 itself as "learning progress"): relief = decrease of the
need-weighted proper score of the mind's own predictions on real events, which neither forgetting nor believing
harder can lower. It only works after D9: with today's expectations the mind's predictions score worse than base
rates, so predicting less is truly better (measured: F 7 → 3 lowers the log loss from 141 to 90 nats per moment).

### SERIOUS

#### D6. The settling recurrence does not converge; T ≤ 6 returns a phase of an oscillation

**Attacks.** §2.2 "RELATE (settle, T ≤ 6 iterations): a ← kWTA_F( decay·a + in + Σ_u a_u·(e_{u→v} − ½)·2 …)".

**Evidence** (`sim_settle.py`, `stream_literal.jsonl`, `ablate_final.jsonl`).
- Derivation. Inside a fixed focus the map is a ← (decay·I + Mᵀ)a + in. For an input x that is a rare special case of
  a frequent unit y (P(y|x) = 1, P(x|y) = q) the learned weights are +1 and 2f(q) − 1 < 0, the eigenvalues are
  decay ± i·sqrt(|w_xy·w_yx|), modulus sqrt(decay² + |w_xy·w_yx|): 1.03 (q = 0.01, decay 0.5), 1.09 (q = 0.001), 1.27
  (q = 0.01, decay 0.9). An unstable spiral for every familiar "rare thing of a common kind".
- Counterexample with two frequent associates y, z of the input x (q = 0.01, weights from the design's own rule):
  focus per iteration = x, xyz, yz, yz, xyz, xyz, xyz, xyz, yz, xyz, … with a_x = 1.00 1.00 0.00 0.00 0.19 0.69
  0.83 0.04 0.00 0.12 0.62 0.89. Whether the input is in the focus depends on the iteration at which settling stops.
- Latch (derivation + run): two units with mutual weight w keep each other at 1 without input when
  decay + sqrt(w_uv·w_vu) ≥ 1 (decay 0.5: w ≥ 0.5, i.e. f ≥ 0.75): a_u stays 1.000 after 40 iterations.
- Engine, design without interoception: mean 5.82 of 6 iterations at 10^5 moments; values still moving at T in 89 %
  of moments; the focus SET itself still changed at the last iteration in 22 % (16 % at 10^4); period-2 in 2.4 %.
  With all other fixes but the recurrence kept: set still changing in 8.4 %, values moving in 51 %.

**Severity.** Serious (the focus decides what is learned; an unconverged focus makes learning depend on T).

**Smallest fix (run).** Monotone admission instead of a recurrence: evented units are in the focus (a = 1); free
slots are filled best first by the unit with the largest support s_v = decay·a_v(prev) + Σ_{u∈focus} a_u·(e_uv − b_v),
if s_v > θ_r, with a = min(1, s_v), then frozen; nothing leaves within a moment. Terminates after at most
F − |events| admissions by construction, 0 % set changes, each focus unit's ties are read once (1 668 reads per
moment instead of 8 746 at 10^4 moments), association recall 0.91 / 0.98 / 1.00 against 0.80 / 0.93 / 0.97 with
the recurrence. Changing one's mind happens across moments, where the process (VISION 2) lives.

#### D7. Ties are founded by recall, not by experience: 80–92 % of new ties have an end that did not occur

**Attacks.** §2.5 "For pairs (u, v) in the focus with no tie: create EXPECT u → v with evidence g·a_u·a_v (candidate
tie) … Ties are only ever created between units that were in the focus together".

**Evidence.** Design without interoception: share of newly created ties with at least one non-evented end = 0.80
(10^3 moments), 0.87 (10^4), 0.92 (10^5); 3.35 M ties against 0.85 M when only evented pairs found ties.
Perseveration test (`sim_settle.py` (4), unstructured background of 2 000 symbols, a pair met 100 times and then
never again): the pair is back in the focus in 16.3 % of the next 1 000 moments (12.2 % with a > 0.5, last time at
moment +991), its out-ties grow 620 → 1 286 and 606 units acquire a tie into it; a pair met only 10 times: 3 %,
76 → 244 ties. An activation floor of 0.05 changes nothing (12.3 %). With evented-only ties: 0.0 %, 281 → 281,
1 unit. The design stores its own recollections as co-occurrence and is then reminded by them.

**Severity.** Serious (unbounded junk growth and a positive feedback between recalling and believing).

**Smallest fix (run).** A candidate tie is created only between two units carried by events (same moment for lag 0,
consecutive moments for lag 1). Co-presence with a merely recalled unit may at most add evidence of provenance
"inferred" to a tie that already exists.

#### D8. Idle imagination confirms itself: a 12-step loop is believed at 0.99 while it is true half of the time

**Attacks.** §6 "the top prediction is fed back as an event with provenance 'imagined' … for up to H steps, starting
from high-drive units" together with §2.5 "LEARN (always on) … v occurred: w⁺[π] += g·a_u (π = provenance of the
confirming event)".

**Evidence** (`sim_idle.py`: Markov world of 300 symbols, 30 000 real steps, then idle episodes of H = 20):

| idle episodes (τ_imagined = 0.3) | distinct units visited | cycle | mean e on the path (true p) | log loss on 10 000 real steps |
|---|---|---|---|---|
| 0 | – | – | 0.635 (0.52) | 2.095 |
| 100 | 14 of 300 | 12 | 0.726 (0.52) | 2.093 |
| 1 000 | 14 | 12 | 0.919 (0.51) | 2.118 |
| 10 000 | 14 | 12 | 0.989 (0.51) | 2.204 |

τ_imagined = 0.1 only delays it (0.970 after 10 000 episodes). Sampling the prediction in proportion to e instead of
taking the top one visits 294 units but drifts everywhere (calibration error 0.168 → 0.205, log loss 2.095 → 2.316).
Nothing bounds it: trust in "my own construction" is only corrected by sensed evidence, which an idle mind does not
get. Also, "starting from high-drive units" means starting from hubs, since D(u) grows with degree (D5).

**Severity.** Serious (the more the mind thinks, the worse it predicts; VISION 5 asks for the opposite).

**Smallest fix (run).** An imagined event is not evidence for the tie that produced it: LEARN skips ties on the
generating path; a construction is written once per distinct conclusion with weight ι·R (set, not add). With it the
table stays at 0.635 / 2.095 for any amount of idling.

#### D9. Expectations are not usable as probabilities: issued 0.10, realised 0.005, negative skill; k = 1 is off by two orders of magnitude

**Attacks.** §1 "conviction c = W/(W + k) (how much has been lived; k = 1)", "expectation e … (what the mind acts on)",
"First hearing is weak by construction".

**Evidence** (`sim_calib.py`: 5 000 moments of life, then 6.76 M lag-1 predictions scored over 3 000 moments).
- One sensed co-occurrence: W = 1, c = 0.5, f = 1, e = 0.75 (weak first hearing holds only for told evidence).
- Design: mean expectation issued 0.1017, share that came true 0.0049; log loss per prediction 0.122 against 0.032
  for "predict every target at its base rate": skill −0.090. The ties make the mind predict worse than no ties,
  in every evidence bucket (ties with W⁺ > 20: e 0.266, observed 0.094).
- Base rate as prior mean and unweighted counts, k = 1: 0.0207 issued against 0.0072, skill −0.007 (negative below
  W⁺ ≈ 5, positive above). With k fitted on the first half of the predictions (k = 128, the top of the grid) and
  scored on the second half: 0.0097 against 0.0066, skill +0.001, positive from W⁺ ≥ 3.

**Severity.** Serious (conviction is VISION 4; surprise, the act threshold and T8 all read e).

**Smallest fix (run post hoc).** Prior mean = base rate (D1). k is not a constant: it is the mind's trust in young
evidence, learned by the one rule from the hit rate of its own young ties, per tie form and lag (increase k when
young ties' predictions fail more often than e says, decrease it otherwise; sampling one tie per focus unit per
moment suffices).

#### D10. Slot induction makes one category per frame: 21 526 categories for 6 word classes

**Attacks.** §3 "two composed units of equal length that differ in exactly one position give a slot-bearing unit
(frame) with a variable there; the fillers seen in that position get ABSTRACT ties to a new category unit for the
slot"; §1 "slot-bearing = a composed unit some of whose positions are variables" (so the rule applies to frames
again); §2.2 "up through ABSTRACT (instance activates its categories)".

**Evidence** (`sim_slots.py`, grammar with 6 word classes, tuples of length 5, 30 000 sentences): 21 352 composed
units → frames with 1/2/3/4 variables: 21 526 / 22 937 / 3 752 / 137; 21 526 categories; 71 454 ABSTRACT ties; one
word belongs to up to 3 340 categories, all of which "up" would activate when the word occurs; done over pairs as
written: 227 943 276 comparisons. Length 3: 3 684 composed, 1 261 categories for 5 classes, 384 per word. Upper
bound of the rule: 2^m − 2 tuples-with-holes per composed unit, 126 at m = 7.

**Severity.** Serious (blow-up of units and of per-moment work; categories that are 3 000 overlapping fragments
cannot carry T2/T3).

**Smallest fix (run).** Use the recruitment-by-vigilance the design already has (§2.2) for categories: a slot's
filler set is first matched against existing categories; it joins the best one if the overlap is ≥ ρ, else a
category is recruited; categories whose member sets overlap ≥ ρ merge. Result on the same data: 6 categories
(5 for length 3), purity 1.00. Find frames by hashing the m one-hole signatures of a new composed unit (O(m)), not
by comparing pairs; do not build frames of frames until a first-order frame is itself used.

#### D11. Chunking by chains of pairwise ties yields no word at e_chunk ≥ 0.5 and 93 % never-seen chains at 0.3

**Attacks.** §3 "an EXPECT(lag 1) chain u₁→u₂→…→u_m whose ties all have e ≥ e_chunk and use ≥ m_chunk is compiled into
one composed unit (a word from characters, a phrase from words, a habit from acts). Imitation is this".

**Evidence** (`sim_chunk.py`: 220 026 characters of real English, 27 symbols, first-order ties, design rule).
- e_chunk ≥ 0.5: 12 ties pass (`he`, `qu`, `ve`, `e␣`, …), 3 chains of length 3, none longer, 0 of the 100 most
  frequent words. First-order transitions are rarely that reliable: P(h|t) = 0.31, P(e|h) = 0.49.
- e_chunk = 0.3, the first value at which words appear (9 of 100): 78 ties, chains of length 3…8: 192, 435, 929,
  1 751, 2 813, 3 704; of these 9 824 chains 9 159 never occurred in the stream. A chain of strong pairs is not a
  sequence that happened, and the number of chains grows geometrically with length.

**Severity.** Serious (imitation, the first developmental stage of VISION 9, rests on it).

**Smallest fix (run).** Chunk only what was experienced, one step at a time: when the unit that just occurred is
followed by a symbol and that tie is strong relative to the symbol's base rate (e − b ≥ δ) and confirmed ≥ m_chunk
times, recruit (unit, symbol) as a composed unit; it then owns its own ties, so P(e|th) is measured where P(e|h) was
too weak. Same stream, δ = 0.25, m_chunk = 10: 53 / 93 / 179 / 297 chunks after 10k / 30k / 100k / 220k characters,
0 that never occurred, 17 of the top 100 words as single units, longest `distribute the␣`. At most one new unit per
moment.

#### D12. §10 as written: one settling iteration reads every tie; 114 bytes per tie

**Attacks.** §10 "sparse tie tables (COO: src, dst, form, lag, w⁺[6], w⁻[6], use, last), settling = sparse
matrix–vector products + top-F selection … Float64 evidence."

**Evidence** (`bench_scale.py`, 10^6 units, 2·10^7 ties, this machine: 4-core Xeon 2.1 GHz, 15 GB).
- One settling iteration as an SpMV over the COO table: 1.02 s (51 ns/tie; 0.37 s with the weights cached once per
  moment; in torch 1.15 s on one thread and 0.86 s on four, it is memory-bound) → 6.1 s per moment at T = 6, 30.6 s per moment at 10^8 ties.
  The same iteration through an index by source: 75 µs (13 600 times less).
- With an index but with hubs (D3): a unit of degree 10^6 in the focus costs 64 ms per settling iteration and 38 ms
  per LEARN, and writes 16 MB of log per moment.
- 114 B per tie → 11.4 GB at 10^8 ties (10^6 units × degree 100) before any index; a lean row is 13 B.

**Severity.** Serious (the stated engine form cannot run at the stated size; see §3 for one that can).

**Smallest fix.** The structures of §3 below (measured in the Python tool: flat cost up to 1.1·10^6 units).

#### D13. Recruitment: one unit per moment under one reading, units are never forgotten, and the composed units that hold the facts are never used to complete a pattern

**Attacks.** §2.2 "if the focus holds a tuple matched by no composed unit above vigilance ρ, RECRUIT a new composed
unit for it"; §2.2 "Composed units whose parts are all active become active (recognition)"; §10 "Memory grows with
ties; bounded by pruning".

**Evidence** (`run_recruit.py`, 20 000 moments, 4 780 distinct facts met; `ablate_final.jsonl`).
- If "a tuple in the focus" is the focus content: 16 039 composed units (0.80 per moment), matched again in 19.8 % of
  moments; with the fixes of this report 18 507 (0.93 per moment, 7.5 % re-matched): episodic storage, one unit per
  moment. If it is the tuple of the moment's events: 5 739 units (0.29 per moment), re-matched in 71 %.
- No rule ever removes a unit or a PART tie. The 5 % random statements of the toy world each leave a permanent unit;
  true facts saturate, noise units grow linearly for ever.
- Pairwise EXPECT ties cannot carry a fact that depends on two units: relation recall (entity, relation → value) is
  0.27 / 0.43 / 0.49 with all fixes and falls with experience (0.31 / 0.17 / 0.26 at 10^5 moments). The composed
  units do hold the facts: letting a composed unit whose other parts are active offer its missing part gives
  0.96–1.00 on the same states. §2.2 only activates composed units whose parts are ALL active, so this knowledge is
  in the core and unreadable.

**Severity.** Serious.

**Smallest fix (completion read-out run, the rest derivation).** (a) the recruited tuple is the tuple of evented
units of the moment; (b) a composed unit with at least a fraction ρ of its parts active becomes active to that
degree and offers its missing parts through PART (the vigilance the design already has), found through an index
from the rarest active part (§3); (c) a unit whose ties have all been displaced and which is part of no other unit
is collected.

#### D14. The act threshold integrates over settling iterations, so confusion acts

**Attacks.** §2.4 "Act units accumulate their incoming expectation over the settling iterations; an act is emitted
when its accumulated evidence crosses the threshold θ".

**Evidence (derivation + measured iteration counts).** With a settled focus an act unit receives the same support
s at every iteration, so the accumulated value is s·T′, T′ = number of iterations actually run. The act fires iff
s ≥ θ/T′. T′ is not evidence: it is 2–3 when settling converges at once and 6 when it does not (measured mean 2.8–4.2
with interoception, 4.1–5.8 without; values still moving at T in 8–34 % and 39–89 % of moments). An act with s = 0.25 and θ = 1 is silent in the
clear moments and is emitted exactly in the moments in which the focus did not settle.

**Severity.** Serious (acts leave the mind through this rule; VISION oversight).

**Smallest fix (derivation).** Threshold the settled support of the act unit; accumulate across moments with a leak
if hesitation is wanted (evidence then arrives with events, not with iterations).

#### D15. Absence on a silent channel is counted as negative evidence

**Attacks.** §2.5 "v was predicted, did not occur: w⁻[π] += …" under C1 (events from any channel).

**Evidence (derivation).** The word is read 1 000 times while the camera is off and 10 times while the thing is
seen. The tie word → seen-colour gets 10 confirmations and 1 000 "did not occur": P = 0.0099, design f = 0.095,
weight −0.81; with the base rate as reference the tie is still inhibitory whenever that colour is seen in more than
1 % of all moments. Reading the word suppresses the seen quality it belongs to (the owner's own example, VISION 3b
and 10). The provenance π of an
absence is also undefined in the rule.

**Severity.** Serious as soon as a second channel exists (Phase 1 has only text, so it will not show in T1–T9).

**Smallest fix (derivation).** Exposure counters of D3 are kept per channel of the target: a tie's exposure advances
only in moments in which the target's channel delivered events. No new code per channel.

### MINOR

#### D16. "Retracted exactly" is not exact in float64, not counterfactual under the design's rule, and the log has no end

**Attacks.** §10 "Float64 evidence. Deterministic … one experience can be retracted exactly by subtracting them";
§9 (iv).

**Evidence** (`sim_retract.py`). (a) Subtracting a logged float64 increment does not restore the bit pattern of the
sum without it in 29–36 % of trials (error up to 7·10⁻¹² at 10^5 increments). (b) With the e-weighted negative rule
later increments depend on what the retracted experience left: after retraction the tie's e differs from that of a
life without the experience by 0.029 on average (max 0.065) for a 20-step tie, 0.0004 for 2 000 steps; the
retracted experience also decided which ties and units exist and what was in every later focus. (c) With unweighted
integer counts both differences are exactly 0 (0 of 300 trials). (d) The log grows for ever (D3).

**Smallest fix (run for a–c).** Evidence as integer counts of events (fixed point), negatives implicit; then
retraction of a moment = replaying "minus the events of that moment" and is exact for evidence. Keep the log as a
ring buffer of the last M moments (events and focus, ~100–200 B per moment); beyond it an experience is absorbed and
can only be answered by new evidence. State (iv) as "exact within the retention window".

#### D17. Activation is not kept in [0, 1] by the equation, there is no floor, and ties at the F-th place are broken by index

**Attacks.** §1 "activation a_u ∈ [0, 1]" against §2.2's sum; "kWTA_F keeps the F most active units".

**Evidence.** Raw values reach 7.4 (5.8 without interoception); 18–22 unit-iterations per moment are clipped at 1;
after clipping the F-th and (F+1)-th candidates are exactly equal in 46 % of moments at 10^3 and 4 % at 10^5
(design without interoception), decided by unit index, i.e. in favour of older units. Without a floor a unit with
any positive activation keeps a focus slot while there is room.

**Smallest fix.** Follows from D6 (admission with a threshold θ_r and a = min(1, s)); break exact ties by recency.

#### D18. Up and down count the same evidence twice, and one reading diverges (derivation)

**Attacks.** §4 "g acquires EXPECT g → v with evidence = Σ_i min(W_i, cap) discounted by the ABSTRACT expectations";
"W_eff⁺ = W_own⁺ + λ·e(i→g)·W⁺(g→v)".

**Evidence.** W(g→v) contains instance i's own evidence, which returns to i: W_eff(i) = W_i + λ·e_i·(e_i·W_i + …):
a single-member category raises its member's conviction by the factor 1 + λe² with no new experience. If "W_i" in
"up" is read as the effective evidence, W_g = Σ_i e_i·(W_i + λ·e_i·W_g), so W_g = Σ e_i W_i / (1 − λ Σ e_i²): infinite
when Σ e_i² ≥ 1/λ = 2, i.e. three members with e ≥ 0.82. By §2.2 a category activated through "up" is a focus member,
so by §2.5 it additionally collects direct evidence for the same events.

**Smallest fix.** Up sums own evidence only; down leaves the instance's own share out:
W_eff(i) = W_i + λ·e_i·[W_g − e_i·min(W_i, cap)]; a category active only through "up" does not also count the event
directly.

#### D19. F is a knob the mind sets itself, and cost is quadratic in it (derivation)

**Attacks.** §8 "Control knobs κ = (focus size F, …). Setting a knob is an internal act"; §2.5 "growth O(F²) per
moment".

**Evidence.** Candidate ties per moment ≤ 2F² − F: 91 at F = 7, 276 at F = 12, 1 128 at F = 24; the knob experiment
of D5 shows what moves it. Nothing in §8 bounds it, and "changing its own configuration" is listed in VISION's
oversight as an act that needs approval.

**Smallest fix.** F, K, M live inside a budget granted through the same gate as external acts; the body channel
reports the cost.

## 3. Data structures and bounds that make §10 real

Everything here is what the repaired engine of `zsim.py` does (in Python dictionaries and NumPy arrays); the sizes
are for a compiled struct-of-arrays layout.

| structure | content | bound |
|---|---|---|
| unit table (n rows) | kind flags, channel, occurrence count, birth, exposure counters X[lag][provenance][channel group], bag heads, parts pointer | ~100 B per unit |
| tie bags, one per unit, tie form and lag | K contiguous slots: dst u32, W⁺ u32 (integer count), exposure-at-birth u32, flags u8; size classes 8…K_max from a slab allocator; ties with mixed provenance point into a small side table with the 6-vector | 13 B per tie; ties ≤ n · forms · K_max; no global hash: "is there a tie u → v" is a scan of one bag |
| base rate | occurrence count / age of the unit | derived, O(1) |
| settle scratch | dense accumulator of n floats + list of touched ids, reset after use | reads ≤ F · K per moment (each focus unit's bag once, D6), top selection over ≤ F · K candidates |
| LEARN | per focus unit: one exposure increment; per (focus unit, event of the moment): one bag scan and one W⁺ += 1 | ≤ F + F·|events| writes; O(F²) candidate insertions, each ≤ one bag scan and one displacement |
| composed units | hash from the exact tuple of part ids to the unit (recognition O(1)); posting bags part → composed units, intersected from the rarest active part; a part whose base rate is high is not indexed (it selects nothing) | O(m) per new unit; partial match ≤ shortest posting bag |
| frames | hash from one-hole signature to frame unit | m insertions per new composed unit |
| forgetting | displacement on insertion (local, by z); a clock sweep over bags when idle; unit collection when a unit has no ties and no parent | no global sort ever; memory ≤ n·(100 + 13·forms·K_max) B |
| retraction | ring buffer of the last M moments: event ids, focus ids, activations | ~100–200 B per moment, exact for integer evidence |
| determinism | integer evidence: sums are order independent, so results do not depend on thread count | – |

Per moment, F = 7, K = 64: ≤ 448 tie reads for settling, ≤ 448 for prediction, ≤ 91 candidate insertions, ≤ ~60
evidence writes; none of it depends on n.

Measured with the Python tool (`run_stream.py`, `run_big.py`, `run_big2.py`; one thread):

| run | moments | units | ties | tie reads / moment | evidence writes / moment | ms / moment | max degree | association recall h/m/t |
|---|---|---|---|---|---|---|---|---|
| design as written | 10^5 | 19 209 | 450 887 | 239 174 | 111 403 | 6.75 | 37 311 | 0.00 / 0.00 / 0.00 |
| all fixes, no bags | 10^5 | 19 206 | 853 135 | 9 653 | 63 | 1.60 | 19 985 | 0.93 / 0.90 / 0.96 |
| all fixes, bags K = 64 | 10^3 | 1 068 | 18 813 | 160 | 51 | 0.44 | 128 | 0.95 / 0.99 / 0.99 |
| all fixes, bags K = 64 | 10^5 | 19 203 | 437 479 | 292 | 55 | 0.82 | 128 | 0.89 / 0.82 / 0.88 |
| same, open world (300k entities) | 10^5 | 67 657 | 846 751 | 243 | 48 | 0.85 | 128 | 0.95 / 0.89 / 0.99 |
| same, open world | 10^6 | 394 552 | 5 183 763 | 314 | 46 | 0.76 | 128 | 0.96 / 0.93 / 0.98 |
| same, open world | 1.3·10^6 | 473 502 | 6 362 766 | 318 | 46 | 0.78 | 128 | 0.96 / 0.92 / 0.97 |
| same, larger and flatter world (10^6 entities) | 10^5 | 167 279 | 1 608 918 | 193 | 42 | 0.54 | 128 | 0.77 / 0.94 / 1.00 |
| same | 3·10^5 | 452 092 | 4 188 013 | 235 | 43 | 0.66 | 128 | 0.82 / 0.79 / 0.99 |
| same | 6·10^5 | 818 445 | 7 766 745 | 279 | 43 | 0.73 | 128 | 0.90 / 0.81 / 0.99 |
| same | 9·10^5 | 1 135 850 | 11 101 850 | 283 | 42 | 0.78 | 128 | 0.88 / 0.77 / 0.98 |

Row 2 shows that the bags are necessary: with every other fix a hub still reaches degree ≈ n (an interoceptive level
unit, 19 985) and reads per moment grow with n (285 → 9 653). The price of the bags is visible in row 4 (head
entities lose 4–8 points of association recall against the unbounded row 2); K is the dial.

**10^6 units on a 4-core CPU.**
- Design as written: tie reads ≈ 12.4·n and evidence writes ≈ 5.8·n per moment (measured coefficients at n = 19 209,
  both still rising) → 1.2·10^7 reads and 5.8·10^6 writes per moment; at the measured 64 ns and 38 ns per tie in the
  §10 layout that is ≈ 1 s per moment on one core (the work is memory-bound: four threads gained 25 % on the SpMV), and
  93 MB of retraction log per moment, i.e. a terabyte every ~11 000 moments. Done as "sparse matrix–vector products
  over COO" it is 6–30 s per moment. A character stream of 10^6 symbols would take 12 days to 1 year.
- With the structures above: ≈ 300 tie reads and ≈ 50 writes per moment regardless of n; measured at 1.14·10^6 units:
  283 reads, 42 writes, 0.78 ms per moment in the Python tool (reads saturate, 193 → 235 → 279 → 283, as the bags of
  frequent units fill to K; the tool itself holds 4 GB of interpreter objects at that size); memory
  ≤ 10^6 × 2 × 64 × 13 B = 1.7 GB as a hard bound (measured occupancy 10–13 ties per unit → 0.13–0.17 GB), plus 0.1 GB
  of units. In compiled code the per-moment work is of the order of 10–50 µs (estimate from the measured 41 ns per
  scattered tie read; not measured as a whole). The four cores are then free for channel adapters and for idle
  consolidation.

**What breaks first** (design as written, in the order in which a builder will meet it):
1. Function, within the first 10^3 moments: 22 % of the input reaches the focus, association recall 0.00 (D1, D2).
2. The retraction log: 3 GB at 10^4 moments, 145 GB at 10^5 (D3).
3. Time per moment, linear in the number of units: 0.75 ms at 1.1·10^3 units, 6.75 ms at 1.9·10^4 (Python), ≈ 1 s at
   10^6 in optimised code (D3, D12).
4. Knowledge, the day a memory budget is set: the pruning rule keeps the head and nothing else (D4).
5. RAM last: 114 B per tie is wasteful but the tie table is 51–382 MB at 10^5 moments.

After the fixes the next limits are: units are never collected (D13), the bag size trades memory against recall of
frequent units (D4), relation-specific facts need the completion read-out (D13), and the Python prototype is
interpreter-bound at ~0.8 ms per moment.

## 4. The fixes as replacement text (smallest set that removes the fatal findings)

1. §1: e = c·(f − b_v) + b_v with b_v = occurrences of v / age of v. k is the learned trust in young evidence (per
   tie form and lag), not 1. [D1, D9]
2. §2.2: evented units are in the focus with a = 1. Free slots are filled best first by
   s_v = decay·a_v(prev) + Σ_{u∈focus} a_u·(e_{u→v} − b_v) if s_v > θ_r, a = min(1, s_v), frozen; at most T
   admissions. A composed unit with a fraction ≥ ρ of its parts active is active to that degree and offers its
   missing parts. [D2, D6, D13, D17]
3. §2.4: an act is emitted when its settled support crosses θ. [D14]
4. §2.5: "v occurred: W⁺ += 1" (one event, one unit, integer). Non-occurrence is not written: W = exposure of u (in
   moments in which v's channel was live) since the tie's birth. Candidate ties only between two evented units. A
   unit keeps at most K ties per form and lag; a newcomer displaces the least significant one. [D1, D3, D4, D7, D15]
5. §2.6: an interoceptive event is emitted when its level changes. [D2]
6. §3: chunk = extend the unit that just occurred by what just followed, when that tie is strong against the base
   rate; a slot's fillers join an existing category when they overlap it above ρ, categories that overlap merge.
   [D10, D11]
7. §5: drive = Σ need·(1 − c) over ties that existed before the moment, need a bounded rate, a pruned tie counts as
   unknown, g does not multiply evidence; preferably relief = progress of the proper score of own predictions on
   real events. [D5]
8. §6: an imagined event is not evidence for the tie that produced it. [D8]
9. §8/§10: forget by significance z, locally (bags), units without ties are collected; the structures of §3; the
   retraction log is a ring buffer and retraction is exact within it. [D4, D12, D16]

## 5. What survives

- C2, a growing substance instead of an overwritten tensor. In every run without a budget, adding a second, disjoint
  domain cost the first one nothing (first domain 0.92 / 0.95 / 0.99, second 0.84 / 0.97 / 0.98, `run_prune.py`);
  nothing collapsed in 1.3·10^6 moments. The "stability of a store with the integration of weights" asked for in
  VISION 14 is real here, as long as forgetting is done by significance (D4).
- The focus as the bottleneck of growth: tie creation stayed under the stated O(F²) in every run (3–43 new ties per
  moment, bound 91). What failed was the F·deg term, not the F² term.
- Additive evidence as counts. With unweighted integer counts f is the observed frequency (0.2003 for p = 0.2),
  retraction of a moment is exact (D16 c), and negative evidence can be implicit, which is what makes the engine
  O(F²) per moment.
- The form c = W/(W + k): weak first hearing and conviction that rises with lived evidence survive once the prior mean
  is the base rate and k is learned.
- Composed units recruited from the tuple of a moment's events: sublinear growth in a finite world (0.29 per moment,
  71 % re-matched), and they do hold the conjunctive facts (0.96–1.00 by completion).
- A small focus with k-winners as attention survives once inputs cannot be vetoed and admission is monotone
  (0 % unsettled moments, recall 0.91–1.00).
- Categories from "instances that occupy the same place" and chunks from repeated sequences survive in the repaired
  forms: 6 categories with purity 1.00 for 6 classes; 297 chunks, none of which never occurred.
- The engine form as tensors, deterministic, on a CPU: with the structures of §3 the cost per moment is independent
  of the number of units (measured flat from 10^3 to 1.1·10^6 units).
- C1 (one event interface) is not touched by any finding; D15 is repaired inside it (exposure per channel), with
  no code per channel.
- Not attacked here (outside this lens): provenance-resolved trust τ, leave-path-out reality scoring R, detachment
  from imitated behaviour, the governance gate.

## 6. Scripts and raw results (all in this folder)

| script | output | what it tests |
|---|---|---|
| `zsim.py` | – | the engine (design as written + switches for each fix) and the toy world |
| `run_stream.py` | `stream_literal.jsonl`, `stream_patched.jsonl`, `stream_literal.log` | the 10^5-moment stream: units, ties, bytes, log, reads/writes/ms per moment, degree distribution, focus content, convergence, recall |
| `run_ablate.py` | `ablate_final.jsonl`, `ablate_final.log` | which single change repairs perception and recall |
| `run_prune.py` | `prune_results.jsonl` | pruning rules under a tie budget, second domain |
| `run_recruit.py` | (stdout, quoted in D13) | unit growth under the two readings of "tuple in the focus" |
| `run_big.py`, `run_big2.py` | `stream_big.jsonl`, `stream_big2.jsonl` | the repaired engine at 4.7·10^5 and ~10^6 units |
| `sim_rule_bias.py` | `sim_rule_bias.json` | fixed point of the learning rule, facets |
| `sim_settle.py` | `sim_settle.json` | spiral, latch, stability region, perseveration |
| `sim_idle.py` | `sim_idle.json` | idle self-confirmation |
| `sim_drive.py` | `sim_drive.json` | drive, relief, knob experiment |
| `sim_calib.py` | `sim_calib.json` | calibration and skill of issued predictions, fitted k |
| `sim_slots.py` | `sim_slots.json` | slot and category blow-up, vigilance merge |
| `sim_chunk.py` | `sim_chunk.json` | chain chunking on real characters, experienced chunking |
| `sim_retract.py` | `sim_retract.json` | exactness of retraction |
| `bench_scale.py` | `bench_scale.json` | §10 at 10^6 units on this machine |
| `dbg1.py` | (stdout, quoted in D2) | trace of three moments |

Limits of this review: one family of toy worlds; one provenance class; ABSTRACT up/down and acts are not in the
engine (D14, D15, D18, D19 are derivations); timings are from a Python tool and only the operation counts carry
over to another implementation; the fixes were tested for the failure they repair and for recall on this world,
not on T1–T9.
