# -*- coding: utf-8 -*-
"""AWA-3 Phase 1: the one funded extension of runs that fail their competence check at step 3,000.

Usage (from the repository root):
    python research/attention_order_law_2026_11_13/run_extend.py [--results DIR]

Failing runs are extended by 1,500 steps in the priority order AWA > ChaCAL > AWA-own > LOOPED > ARLOOP (then by
seed) while a fixed reserve of 1.0 core-hour lasts; STD-2L-pm is exempt. The decision depends only on the
in-distribution check, never on test results (this job runs before any evaluation job). The ledger is
<results>/phase1/extensions.json; the job can be restarted.
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
    from awa3 import layout, phases

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(layout.DEFAULT_RESULTS))
    args = ap.parse_args(argv)
    ledger = phases.run_extensions(pathlib.Path(args.results), progress=lambda msg: print(msg, flush=True))
    print(json.dumps(ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
