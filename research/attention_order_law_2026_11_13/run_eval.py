# -*- coding: utf-8 -*-
"""AWA-3: evaluate one finished run on the frozen test and validation sets (whole budget axis, traces, costs).

Usage (from the repository root):
    python research/attention_order_law_2026_11_13/run_eval.py --phase 1 --arm AWA-2L --seed 0
    python research/attention_order_law_2026_11_13/run_eval.py --phase smoke --arm AWA-2L --lr 1e-3 --limit 16 \
        --results /some/scratch/dir

Writes <run>/eval.json. ``--limit`` (smoke runs only) truncates every set; such records are marked and the
analysis refuses them unless asked explicitly.
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
    from awa3 import arms, evaluate, layout, phases, train

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", required=True, choices=layout.PHASES)
    ap.add_argument("--arm", required=True, choices=sorted(arms.ARMS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="phase 0 and smoke runs (part of the run name)")
    ap.add_argument("--limit", type=int, default=None, help="smoke runs only: first N items of every set")
    ap.add_argument("--results", default=str(layout.DEFAULT_RESULTS))
    args = ap.parse_args(argv)
    results = pathlib.Path(args.results)
    if args.phase == "1":
        if args.limit is not None or args.lr is not None:
            ap.error("phase 1 evaluates the full frozen sets of the selected-lr run")
        if args.seed is None:
            ap.error("phase 1 needs --seed")
        lr, seed = phases.selected_lr(results, args.arm), args.seed
    else:
        if args.lr is None:
            ap.error("--lr is part of the run name outside phase 1")
        if args.phase == "0" and args.limit is not None:
            ap.error("--limit is for smoke runs")
        lr, seed = args.lr, layout.PILOT_SEED if args.seed is None else args.seed
    run_dir = layout.run_dir(results, args.phase, args.arm, seed, lr)
    model, meta = train.load_run(run_dir)
    record = evaluate.evaluate_run(model, args.arm, limit=args.limit, progress=lambda msg: print(msg, flush=True))
    record.update(seed=meta["seed"], steps=meta["steps"], model_sha256=meta["model_sha256"])
    train.write_json(run_dir / "eval.json", record)
    print(json.dumps({"eval": str(run_dir / "eval.json"), "cpu_s": record["timing"]["cpu_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
