# -*- coding: utf-8 -*-
"""AWA-3 Phase 0: select the learning rate of every piloted arm and decide GO / NO-GO for AWA.

Usage (from the repository root):
    python research/attention_order_law_2026_11_13/run_phase0_select.py [--results DIR] [--no-extend]

Rule (PROTOCOL section 7): per arm, the lowest in-distribution validation loss on the arm's competence cells at
step 800; losses within 2% of the best go to the smaller learning rate. GO for AWA if, at some learning rate,
k = 2 accuracy is >= 0.30 at step 800 or the k = 2 validation loss at step 800 is <= 0.8 x its step-200 value.
Otherwise the best-lr AWA pilot is extended once by 800 steps (this job does it, unless --no-extend) and the rule
is applied to it at step 1,600. NO-GO: Phase 1 runs only ChaCAL-2L, AWA-own-2L and STD-2L-pm.

Writes <results>/phase0/selection.json.
"""
from __future__ import annotations

import os
import sys
import pathlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
STUDY = pathlib.Path(__file__).resolve().parent
REPO = STUDY.parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(STUDY), str(REPO)]

import argparse  # noqa: E402
import json  # noqa: E402
from typing import Optional, Sequence  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    import torch

    torch.set_num_threads(1)
    from awa3 import layout, phases, train

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(layout.DEFAULT_RESULTS))
    ap.add_argument("--no-extend", action="store_true", help="report 'extension pending' instead of running it")
    args = ap.parse_args(argv)
    results = pathlib.Path(args.results)

    def extend(lr: float) -> None:
        run_dir = layout.run_dir(results, "0", "AWA-2L", layout.PILOT_SEED, lr)
        train.train(run_dir, "AWA-2L", layout.PILOT_SEED, phases.recipe_for("0", lr),
                    total_steps=layout.PILOT_STEPS + layout.PILOT_EXTENSION_STEPS,
                    progress=lambda msg: print(msg, flush=True))

    try:
        selection = phases.select_phase0(results, extend=None if args.no_extend else extend)
    except ValueError as exc:
        print(f"selection not possible: {exc}", file=sys.stderr)
        return 1
    train.write_json(layout.selection_path(results), selection)
    print(json.dumps({k: selection[k] for k in ("lrs", "go", "awa_extended", "extension_pending", "phase1_arms")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
