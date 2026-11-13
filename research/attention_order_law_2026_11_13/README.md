# AWA-3: the order law in attention and prefix-closed absorbing walks

Status on 2026-11-13: code and tests exist; the protocol is a DRAFT (NOT FROZEN); no Phase-0 and no Phase-1 run
has been made. This folder contains no result. The only thing that has been executed is a pipeline smoke with
40 training steps per arm (below), which tests the code path and nothing else.

The mechanism lives in the engine, not here:
`src/neural_network_module/ortak_katman_module/attention_manager_module/absorbing_walk.py` implements
Absorbing-Walk Attention (AWA) and its two comparison operators (own-window walk, ChaCAL-style resolvent) as an
optional attention operator of `MultiHeadAttention` / `CevahirNeuralNetwork`. It is off by default; with it off
the engine is bit-identical to the engine before it existed (`tests/evolution/test_awa_attention.py`: a
differential run against the engine exported from git at 0f9be09, which needs that commit in the local history,
plus digests recorded on the development machine).

## What is here

| Path | Content |
|---|---|
| `PROTOCOL.md` | draft protocol: claims, mechanism, hypotheses H1–H6 with decision rules, task, arms, statistics, open decisions |
| `AMENDMENTS.md` | log of deviations after the freeze (empty) |
| `awa3/task.py` | VarDeref v2 generator: four paired order strata with equal chance floors, frozen test / validation streams |
| `awa3/arms.py` | the six arms (parameter counts asserted), answer-row forwards, analytic cost model |
| `awa3/handset.py` | the order law with hand-set weights (1/m! under shuffling), using the engine operators |
| `awa3/train.py` | deterministic single-thread trainer with resume and JSONL records |
| `awa3/evaluate.py` | depth × stratum × budget grids, ablation, hop traces, gate statistics |
| `awa3/stats.py`, `awa3/analysis.py` | exact tests (T4, Mann–Whitney), Hodges–Lehmann, t-bounds, the decisions |
| `awa3/phases.py` | Phase-0 learning-rate selection and GO/NO-GO, Phase-1 extension ledger (restart-safe) |
| `awa3/manifest.py` | freeze manifest, results manifest, recomputation of the decisions |
| `awa3/jobs.py`, `jobs_*.json` | job lists `{id, cmd, deps, log}` for an external scheduler |
| `run_train.py`, `run_eval.py`, `run_phase0_select.py`, `run_extend.py`, `run_analyze.py`, `run_verify.py`, `make_jobs.py` | command-line entry points (no name is shared with another study folder: tests import some of those by bare name) |
| `results/` | outputs (not committed, except the empty `results/job_logs/` the scheduler writes into) |

Tests: `tests/evolution/test_awa_attention.py` (engine operator) and `tests/evolution/test_awa_study.py` (this
pipeline).

## Running it

Every command runs from the repository root, single-threaded (`OMP_NUM_THREADS=1`; the drivers also set the
thread count themselves) and can be restarted after a kill.

```bash
# tests
OMP_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests/evolution/test_awa_attention.py tests/evolution/test_awa_study.py

# Phase 0 (15 pilots + selection): hand jobs_phase0.json to the scheduler, or run the commands by hand
python research/attention_order_law_2026_11_13/run_train.py --phase 0 --arm AWA-2L --lr 0.001
python research/attention_order_law_2026_11_13/run_phase0_select.py

# freeze (a person does this after Phase 0: close PROTOCOL section 12, remove "NOT FROZEN", then)
python research/attention_order_law_2026_11_13/run_verify.py --freeze

# Phase 1: jobs_phase1.json (GO) or jobs_phase1_nogo.json (NO-GO)
python research/attention_order_law_2026_11_13/run_verify.py --check-freeze
python research/attention_order_law_2026_11_13/run_train.py --phase 1 --arm AWA-2L --seed 0
python research/attention_order_law_2026_11_13/run_extend.py
python research/attention_order_law_2026_11_13/run_eval.py --phase 1 --arm AWA-2L --seed 0
python research/attention_order_law_2026_11_13/run_analyze.py --phase 1
python research/attention_order_law_2026_11_13/run_verify.py --manifest
python research/attention_order_law_2026_11_13/run_verify.py --rerun --phase 1
```

Job files. Each entry is `{"id", "cmd", "deps", "log"}`; `cmd` runs from the repository root, `log` is relative
to the repository root, and a job may start when all its `deps` have finished with exit code 0. The CPU budget
(hard cap 12.0 core-hours) and the drop order of PROTOCOL §11.2 are the scheduler's responsibility; this code
enforces only the 1.0 core-hour extension reserve. `python research/attention_order_law_2026_11_13/make_jobs.py
--check` verifies that the committed files equal the generator.

Guards.
- `run_train.py --phase 1` refuses to start unless `freeze_manifest.sha256` exists and still matches the code,
  the evaluation sets and the Phase-0 selection. The other Phase-1 drivers do not check the freeze themselves;
  in the job list they depend on `p1_freeze_check` and on the training runs.
- `run_verify.py --freeze` refuses while PROTOCOL.md says NOT FROZEN, and refuses to overwrite a freeze.
- `run_analyze.py` refuses truncated (smoke) evaluation records unless `--allow-truncated` is given.

## Smoke run (2026-11-13, not an experiment)

Six arms, seed 900, lr 1e-3, 40 steps of batch 64 (n = 67), validation on 100 items per cell at steps 0 / 20 /
40, evaluation truncated to the first 24 items of every set, then analysis, manifest and recomputation. One
thread, torch 2.15.0, 4-core machine shared with other jobs. Run twice before the review: 113 s wall at load
average ≈ 1, and 127 s wall at load ≈ 2 (the table). Outputs were written to a scratch directory and are not
part of the repository.

| Arm | Params | CPU-s / training step (load ≈ 2) | same, first run (load ≈ 1) | Training loss, step 10 → 40 | Evaluation CPU-s (24 items per set) |
|---|---|---|---|---|---|
| AWA-2L | 551,308 | 0.160 | 0.150 | 4.293 → 3.854 | 3.7 |
| ChaCAL-2L | 551,300 | 0.181 | 0.172 | 4.295 → 3.857 | 2.3 |
| AWA-own-2L | 551,308 | 0.157 | 0.166 | 4.293 → 3.857 | 4.0 |
| LOOPED-2L | 551,296 | 0.484 | 0.438 | 4.272 → 3.589 | 17.4 |
| ARLOOP-2L | 551,296 | 0.218 | 0.202 | 4.274 → 3.599 | 6.2 |
| STD-2L-pm | 551,296 | 0.147 | 0.131 | 4.294 → 3.870 | 1.2 |

The losses are the first and the last record of the curve, which is written every 10 steps (step 10 and
step 40). The two runs of the table used the scripts under their earlier names (`analyze.py`, `verify.py`). A
third run after the review fixes, with the renamed scripts (182 s wall at load ≈ 5), gave the same six loss
pairs to the printed digits; its step times (0.13–0.50 CPU-s) are not in the table.

What it shows: every arm trains, checkpoints, evaluates along its whole budget axis and passes through the
analysis; `run_verify.py --rerun` reproduces the stored decisions; all three runs gave the same losses to the
printed digits (the trainer is deterministic). What it does not show: anything about the hypotheses. After 40 steps
every arm is at the chance floor (in-distribution competence 0.04–0.05), and the analysis correctly reports
P-M and P-T as INCOMPLETE. The step times are from a lightly loaded machine; under load they will be longer
(the program measured 0.22–0.66 s/step at load ≈ 6), so the compute plan of PROTOCOL §11.1 keeps the slower
budget figures. Extrapolated from these step times alone, 3,000 steps take about 8 CPU-minutes for AWA and about
24 for LOOPED, before validation.

## Limits

- The comparator is "ChaCAL-style": the published parametrisation has not been checked against the paper.
- All-row AWA costs O(K n³) per head; the study uses the rows-only final layer. Long contexts would need sparse
  (top-r) transitions, which are not implemented.
- AWA decodes with the KV cache, recomputing the transition matrix every step (O(n² d) per token and head);
  the own-window and ChaCAL-style operators have no cache path.
- Under half-precision autocast the operators stay in float32 (tested on CPU autocast, fp16 and bf16; not run
  on a GPU).
- The digests that pin the operator-off engine are compared only on the machine that recorded them. Elsewhere
  the differential test needs commit 0f9be09 in the local git history and skips without it; what then remains
  is a check of recorded values to a relative 1e-4 (torch 2.15.0 only), which cannot see last-bit changes.
- The task is synthetic and tiny (551k parameters, 67 tokens). Nothing here speaks about language modelling.
