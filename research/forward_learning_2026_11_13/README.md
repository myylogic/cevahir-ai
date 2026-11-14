# FWL — learning in the forward pass (study folder)

A study of the 2026-11-13 research programme. It measures the engine's optional forward-learning layer
(`src/neural_network_module/ortak_katman_module/forward_learning.py`) on the CEV-11 bases. The frozen CEV-11
package (`research/living_learning_paradigm_2026_11_13/cev11/`) is imported and never edited; no model is
pre-trained here.

**State: engine code, study code and tests only. The protocol is NOT FROZEN, the discovery jobs have not been
run, no operating point is selected and there is no result.** A reduced smoke run on discovery world 0 proved the
pipeline end to end and reproduced the numbers of the two v1 pilots; its numbers are not protocol data
(`PROTOCOL.md` §9). An independent review on 2026-11-14 put three primary tests back to what the methods review
had required (H1 against FT-ALL, H2a against FEC-TOP, H3 on net facts per write against every comparator),
replaced the stream's step-cap rule and tightened the integrity checks; `PROTOCOL.md` §4.1 lists what still
deviates from the source proposals, including the comparators that are not run.

## What the mechanism is, and what it is not

A model reads a sentence once in eval mode. The next-token error is pulled back to the residual stream in closed
form, turned into error fields for the `ffn.fc2` outputs with quantities the forward pass already computed
(frozen attention pattern, value path only), and written as rank-one fast weights with a recursive-least-squares
key projector and a forward-only step search. A familiarity gate decides where the fast weights are read; every
write is a float64 ledger entry that can be subtracted again. No backward pass and no optimizer are involved.

Both source proposals (FEC and DEFTER) were judged **incremental** by their novelty reviews. The closed-form
output error applied to MLP weights is FPO; the target-embedding write is Backward Lens; the path-order family is
Highway-BP; stop-gradient through attention scores is TinT; the RLS write is RLSEdit / AlphaEdit / OWM; fast
weights in the MLP projection are In-Place TTT and Fast Weight Layers; the in-layer least-squares solve is
MesaNet; further close work (FAAST, MobiEdit, LoCA, Titans, RETS) is in the table of `PROTOCOL.md` §1.1. What
the study may add is narrow and listed in `PROTOCOL.md` §1.2. The meta-trained DEFTER arm (learned
key map, write gate and feedback; new pre-training) is out of scope and listed as future work.

## Files

| File | Content |
|---|---|
| `PROTOCOL.md` | hypotheses H1–H5, decision rules, parts A/B, gate, capacity law, retraction, discovery selection, blinding, the smoke, open points before the freeze |
| `fwl/common.py` | paths, run configuration (`protocol_config`, `smoke_config`), record schema, resumable JSONL writer |
| `fwl/single.py` | Part A: one-step arms (FEC-TOP/0/1/A, HEBB, RAND-0, BP-1/1b/pred), FT-L / FT-ALL through CEV-11's code, the SO leak family |
| `fwl/stream.py` | Part B: stream, learner arms, sequential FT-L, RLSEdit-style editor, gate sweep, capacity trace against the law, retraction against replay |
| `fwl/world_run.py` | one world end to end, header record, summaries, the closed-form check on a trained base |
| `fwl/analysis.py` | tables, discovery selection with hash, confirmatory decisions (unit = world, exact sign-flip, Holm) |
| `fwl/verification.py` | freeze manifest, preconditions, record checks, re-analysis, comparator cross-check |
| `fwl/cli.py` | the command-line entry points |
| `run_world.py`, `run_select.py`, `analyze.py`, `verify.py` | thin wrappers around `fwl/cli.py` and `fwl/verification.py` (the logic sits in the package because other study folders use the same script names) |
| `jobs_discovery.json`, `jobs_confirm.json` | job lists `[{id, cmd, deps, log}]`; commands run from the repository root |

Tests (tiny random models, no trained weights): `python -m pytest -q tests/evolution/test_forward_learning_*.py`.

## Running

The base weights are not in the repository. Point `CEV_BASES` at the folder that holds
`discovery_w0_K20/model.pt`, `confirm_w1000_K20/model.pt`, …:

```bash
export CEV_BASES=/path/to/living_learning_paradigm_2026_11_13/results/bases
# smoke: 6 edits and a 30-fact stream; measured 174-215 s on one thread; writes to results/smoke unless --out is given
OMP_NUM_THREADS=1 python research/forward_learning_2026_11_13/run_world.py --run discovery --world 0 --smoke
# discovery (worlds 0 and 1), record check, selection
OMP_NUM_THREADS=1 python research/forward_learning_2026_11_13/run_world.py --run discovery --world 0
OMP_NUM_THREADS=1 python research/forward_learning_2026_11_13/run_world.py --run discovery --world 1
python research/forward_learning_2026_11_13/verify.py --records --results research/forward_learning_2026_11_13/results/discovery
python research/forward_learning_2026_11_13/run_select.py
```

Confirmatory worlds (1000–1007) are refused until the protocol is frozen and the manifest matches:

```bash
python research/forward_learning_2026_11_13/verify.py --freeze          # after PROTOCOL.md says "Status: FROZEN <date>"
python research/forward_learning_2026_11_13/verify.py --preconditions
OMP_NUM_THREADS=1 python research/forward_learning_2026_11_13/run_world.py --run confirm --world 1000
python research/forward_learning_2026_11_13/analyze.py
python research/forward_learning_2026_11_13/verify.py --rerun --crosscheck
```

Estimated compute (from the smoke timings, not measured end to end): about one core-hour per world, ±25%.
A run may only use the base of its own world (the seed in the base's world file is checked), a record file cut
by a kill is repaired on resume, and the selection is tied to the engine module the discovery records were made
with: change the engine after discovery and the discovery jobs have to be run again.

## Limits

A 3.4M-parameter model, a synthetic templated world with the relation phrase directly before the object, 48
possible answers, one tokenizer. The forward-learning layer is an optional module that attaches through hooks; it
is not wired into the product's generation, training or serving paths. The smoke already shows where the idea is
weak: in a stream the tape-free writes damage untouched facts that share the old answer, a capped
exact-gradient step had a better net than both forward-learning arms on the one base probed, the capacity law as
first proposed missed by about a factor two on one base, and ledger retraction was far from a replay without the
fact. These are expectations to be tested, not results.
