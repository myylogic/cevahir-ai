# Kavram Adresi (KA) — which internal code is a concept address for writing a fact?

Second study of the 2026-11-13 research programme. It reuses the frozen CEV-11 engine
(`research/living_learning_paradigm_2026_11_13/cev11/`, imported, never edited) and CEV-11's trained bases.

**State: code and tests only. The protocol is NOT FROZEN, the discovery jobs have not been run, and there is no
result.** A reduced smoke run on discovery world 0 proved the pipeline end to end; its numbers are not protocol
data and are not reported as findings.

| File | Content |
|---|---|
| `PROTOCOL.md` | hypotheses KA-H1–H3, secondary S1a/S1b/S2, the laws, decision rules, discovery selection, blinding, freeze procedure; §10 lists the corrections to the programme text and what must be decided before the freeze |
| `ka/codes.py` | read-only taps for the address codes (pre, res, key, att, post), whitening, two-covariance PLDA, calibration, the null-slot read |
| `ka/gates.py` | A-TOK kernel law, radius gates A-TOKG / A-RES / A-CTX / A-SUP (`GateMemory`), thresholds at matched c0 false-match rates |
| `ka/episodic.py` | output-side memory: analytic values, exact insert / correct / retract ledger, Bayesian continuation, hooked `EpisodicCortex` |
| `ka/streams.py` | fact stream, probe battery, cached evaluator, metrics, the calibrated-read laws |
| `ka/ft.py` | sequential fine-tuning comparators (FT-L, FT-ALL at CEV-11's frozen selection) |
| `ka/part_a.py`, `ka/world_run.py` | Part A edits and the per-world runner with resume, JSONL records and the completeness rules behind `DONE.json` |
| `ka/stats.py`, `ka/analysis.py` | frontier LP, sign-flip / Holm / labels, hierarchical bootstrap, discovery selection with its blocking gates, confirmatory decisions with the consistency checks |
| `run_world.py`, `run_select.py`, `analyze.py`, `verify.py` | drivers, selection file with hash, analysis, freeze manifest and record verification |
| `jobs_discovery.json`, `jobs_confirm.json` | job lists `[{id, cmd, deps, log}]`, commands run from the repository root |

## Running

The base weights are not in the repository. Point `CEV_BASES` at the folder that holds
`discovery_w0_K20/model.pt`, `confirm_w1000_K20/model.pt`, …:

```bash
export CEV_BASES=/path/to/living_learning_paradigm_2026_11_13/results/bases
# smoke (measured 129 s on one core; writes to results/smoke unless --out is given)
OMP_NUM_THREADS=1 python research/concept_address_2026_11_13/run_world.py --run discovery --world 0 --smoke
# discovery (worlds 0 and 1), then the selection
OMP_NUM_THREADS=1 python research/concept_address_2026_11_13/run_world.py --run discovery --world 0
OMP_NUM_THREADS=1 python research/concept_address_2026_11_13/run_world.py --run discovery --world 1
python research/concept_address_2026_11_13/verify.py --records --results research/concept_address_2026_11_13/results/discovery
python research/concept_address_2026_11_13/run_select.py          # exit code 2: a blocking gate failed, no freeze
```

Confirmatory worlds (1000–1007) are refused until the protocol is frozen:

```bash
python research/concept_address_2026_11_13/verify.py --freeze          # after PROTOCOL.md says "Status: FROZEN <date>"
python research/concept_address_2026_11_13/verify.py --preconditions
OMP_NUM_THREADS=1 python research/concept_address_2026_11_13/run_world.py --run confirm --world 1000
python research/concept_address_2026_11_13/verify.py --records         # every world complete, every phase the selection requires
python research/concept_address_2026_11_13/analyze.py                  # exit code 2: a consistency check failed, audit first
python research/concept_address_2026_11_13/verify.py --rerun
```

What the freeze and the records guarantee (PROTOCOL §8):

- The freeze manifest covers `PROTOCOL.md`, the Python code, the job lists, the selection, the KA tests and the
  CEV-11 engine. This README and `AMENDMENTS.md` are outside it on purpose, so recording a deviation or updating
  the status here never blocks a run. The manifest is not overwritten (`--refreeze` replaces it while no
  confirmatory record exists; afterwards it is final).
- A world counts only with `DONE.json`, written when every phase the protocol requires of it is complete. A run
  restricted with `--phases` cannot produce a finished world, and the analysis does not start unless every
  confirmatory world is complete or explicitly excluded.
- The selection blocks the freeze when the FT comparators fail G-KA1c or when the scorers disagree.

Still open before the freeze (PROTOCOL §10): the power of KA-H1's locality guard (item 2), a gate that fires at
later positions of the edited sentence (item 9), and which in-ball flag S2 should read (item 10).

Tests (tiny random models, no trained weights): `python -m pytest -q tests/evolution/test_concept_address_*.py`.

## Limits

A 3.4M-parameter model, a synthetic templated world, one tokenizer. The memory and the gates are study-local
hooks; nothing in `src/` changed, and nothing here is integrated into the product path.
