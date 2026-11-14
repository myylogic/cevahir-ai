# Zihin prototype — first life measurements (2026-11-14)

Runner: `run.py --n 600` (600 experiences = 2400 moments per life). Records: `results/life_<domain>_<seed>_<variant>.jsonl`.
Lives that exist: nitelik × seeds 0–4 × changed, nitelik × seeds 0–3 × as_written, iliski × seeds 0–4 × changed
(iliski as_written and nitelik seed 4 as_written were not run). "changed" = the six fixes F1–F6 of NOTES.md.
Numbers below are means over seeds at the end of the life (at = 600), with [min, max] over seeds. Words are random
per (seed, domain); the core never sees the world's truth tables. These are tests of the design, not a claim.

## T1 — weak first storage, refinement without deletion (white box)

Own conviction c of a told fact by number of hearings (changed):

| hearings | nitelik c | nitelik e | iliski c | iliski e |
|---|---|---|---|---|
| 1 | 0.399 [0.396, 0.403] | 0.700 | 0.410 [0.405, 0.414] | 0.705 |
| 2 | 0.571 | 0.785 | — | — |
| 3 | 0.666 | 0.833 | — | — |
| ≥4 | 0.762 | 0.881 | 0.784 | 0.892 |

Monotone in hearings, as designed. Learned trust in "told" rises from 0.53 (at 60) to 0.67 (nitelik) / 0.70 (iliski).
Lies and corrections (40 per domain, changed): after the lie the false value has e 0.68, c 0.36; after a correction
the false value falls to e 0.45 (sensed correction) / 0.51 (told correction) while its c RISES to 0.55–0.62 (more has
been lived) and the true value reaches e 0.79 / 0.74. In 120 of 120 corrections the positive evidence of the lie is
bit-identical before and after: refinement adds negative evidence, it deletes nothing.
Honest weakness: a told correction leaves the false value at e ≈ 0.51, i.e. not yet below ½.

## T2 — categories formed from instances (against the world's hidden classes)

Best Jaccard between an emergent category and each hidden class (end of life):
entity classes 1.000 in both domains and both variants (at 60 experiences: 0.43 nitelik, 0.55 iliski);
the set of all entities 1.000; dimensions 0.80; value sets 0.41–0.47 (values of a dimension are NOT grouped into
one category — open). Changed variant makes many more frames (nitelik 1047 vs 61) — growth to watch.

## T3 — support from category to instance; exceptions

| (changed, end) | nitelik | iliski |
|---|---|---|
| members told, top & > ½ | 1.000 (e 0.90) | 0.991 (e 0.91) |
| exceptions kept over the rule | 1.000 (e 0.90 vs rule 0.50) | 1.000 (e 0.94 vs rule 0.46) |
| held-out instance, told rule | 0.667 [0.33, 1.0], e 0.57 vs other 0.53 | 0.550, e 0.57 vs 0.57 |
| held-out instance, induced rule | 0.467, e 0.54 vs 0.54 | 0.650, e 0.59 vs 0.58 |

As written (no fixes) T3 fails completely: every probe reads 0 (the summed votes swamp the specific value) — this
was finding F2. With the fixes, members and exceptions are right. **The decisive negative: support from a category
to an instance never seen with that dimension does not work** — the right value is barely above the best wrong one
(≈ chance among 4–5 values). White box: the held-out instance has no own tie and its effective expectation stays
at ½, i.e. the category rule does not reach it. This is the next thing to fix (§ next).

## T5 — no collapse, no forgetting, no lookup outside the core

Changed, end of life: facts told once in the first fifth and never again: recalled 0.98 (nitelik) / 0.96 (iliski);
their evidence is bit-identical to the moment they were learned in 100% of cases; all evidence finite; exact
(ledger sum = state); units ≈ 1460 / 1010, ties ≈ 9770 / 6470, core 2.4 / 1.6 MB; 8.4 / 7.7 ms per moment.
As written: early recall 0.00 and 20 ms per moment — F1–F6 are what make it work.
Open: the log of increments (needed for exact retraction) is 3.7–6.7× the core and grows linearly.

## T9 — exact retraction of one experience

Changed, one random early fact per life: after subtracting its logged increments its own evidence is exactly 0,
every untouched tie is bit-identical, the state equals the sum of the other moments' ledgers, other early facts
unchanged (5/5 lives, both domains). Against the counterfactual life lived without that experience: the retracted
fact's own expectation matches exactly (gap 0.0000); the rest of the mind differs a little (mean gap 0.005, max
0.07–0.14) and has 7 more units / 46–54 more ties — structure created by that experience remains (units are not
removed). Retraction is exact for evidence, not for structure (same law as "same predictions ≠ same future").

## What is not yet measured

T4 (questions), T6 (transfer to an unseen domain), T7 (imitation and detaching), T8 (imagination with reality
score), and the Transformer baselines on the identical stream. These remain the decisive tests of the design.

## Follow-up on T3 held-out (exploratory, seeds 0-2, `probe_reading.py`, `results/probe_reading.jsonl`)

Share of held-out probes answered right (right value top and > ½; chance of being top: 1/4 nitelik, 1/5 iliski),
three seeds each, 6-8 held-out probes per life — small samples:

| learning / reading | nitelik told rule | nitelik induced | iliski told rule | iliski induced | members |
|---|---|---|---|---|---|
| changed / nested (as in T3 above) | 0.78 | 0.33 | 0.50 | 0.67 | 1.00 |
| changed / sharpest (F8) | 0.83 | 0.50 | 0.54 | 0.62 | 1.00 |
| symmetric (F7) / nested | 0.22 | 0.00 | 0.08 | 0.17 | 1.00 |
| symmetric (F7) / sharpest (F8) | 0.83 | 0.39 | 0.67 | 0.50 | 1.00 |

Reading: F7 alone is harmful (and 3-4× slower); F8 repairs it, and on the changed learning F8 gives a small,
not significant gain. Generalisation to an instance never told the dimension is still only partly solved
(0.4-0.8, far from the members' 1.00). The root cause recorded in NOTES.md (many overlapping classes; the mind
does not yet learn WHICH classification predicts WHICH dimension) is the open problem, not a reading detail.

## T7 — imitation first, then detaching (`t7_imitation.py`, `results/t7_imitation.jsonl`, 5 seeds)

The mind first only watches others: in situation S1 they do X, in S2 they do Y (provenance imitated). Then it
lives: the world answers Y with something new and answers X with nothing. Share of trials (blocks of 5 per act):

| condition | does X in S1 | does Y in S2 | "others do X" evidence unchanged |
|---|---|---|---|
| changed / reliable | 0.7, 0.1, 0.0, 0.0, 0.0, 0.0 | 1.0 throughout | 5/5 (bit-identical, imitated w+ 60.3 w- 17.1) |
| control: the world answers X too | 1.0 throughout | 1.0 | 5/5 |
| F6 ablation (no relief expectation for own acts) | same as changed | 1.0 | 5/5 |

Copying comes first; the copied act that brings nothing is given up within about ten own trials; the useful copied
act is adopted; the knowledge that others do X stays exactly as it was. The control shows it is the outcome that
decides. Honest note: the F6 ablation also detaches here, because while watching the mind already learned to
expect relief after X; F6 is needed only where no such expectation exists.

## T3 diagnosis — acquisition versus meaning (`oracle_t3.py`, `results/oracle_t3.jsonl`, 600 experiences, 5 seeds)

For every held-out probe: a frame whose category is exactly the hidden class exists in 100% of probes (140/140)
and holds the right value on top in 80% (112/140). The knowledge is in the core. The reading chooses it in only
30-53% of probes and answers right in 51-63%. Attempts to close this gap by learned reliability of generalisations
(F9, F10 reflection in idle moments, trust-weighted pooling, opinion-weighted verdicts) move the right-answer rate
to 58-73% depending on domain and reading, none uniformly better: iliski reflective 46/80, nitelik trusted 44/60.
Recorded in NOTES.md (F9, F10). The owner's sentence names the open part: the deciding mind making sense of what it
has acquired.

## F11 — a category speaks with what its members have lived (support = "members")

Same runner and records as above (`results/life_*_members.jsonl`, `results_long/life_*_members.jsonl`), 5 seeds at
600 experiences, 3 seeds at 1500. Held-out = instance never told that dimension (right value on top and > ½).

| | held-out, told rule | held-out, induced rule | exceptions kept | members | early facts recalled | ms/moment |
|---|---|---|---|---|---|---|
| nitelik changed, 600 | 0.67 | 0.47 | 1.00 | 1.00 | 0.98 | 8.4 |
| nitelik members, 600 | **0.90** | **0.97** | 1.00 | 1.00 | 1.00 | 9.1 |
| nitelik changed, 1500 | 0.56 | 0.22 | 1.00 | 0.99 | 0.97 | 5.8 |
| nitelik members, 1500 | **1.00** | **1.00** | 1.00 | 1.00 | 1.00 | 9.3 |
| iliski changed, 600 | 0.55 | 0.65 | 1.00 | 0.99 | 0.96 | 7.7 |
| iliski members, 600 | **0.72** | **0.90** | 1.00 | 1.00 | 1.00 | 8.2 |
| iliski changed, 1500 | 0.67 | 0.54 | 1.00 | 0.99 | 0.80 | 5.6 |
| iliski members, 1500 | **0.88** | **0.92** | 1.00 | 1.00 | 1.00 | 6.7 |

With F11 generalisation to unseen instances improves as the life goes on; with the earlier reading it decayed with
age. T9 exact retraction holds in 10/10 lives; T1 conviction by hearings unchanged (c 0.41 at one hearing, 0.78 at
four or more); categories recovered 1.00. Limits: two small generated domains, 6-8 held-out probes per life; the
mechanism is close to exemplar models of categories in cognitive science; what is new here is that it runs inside
a core that learns while living under one evidence rule and stays exactly retractable.
