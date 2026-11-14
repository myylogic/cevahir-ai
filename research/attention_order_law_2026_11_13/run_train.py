# -*- coding: utf-8 -*-
"""AWA-3: train (or resume) one run. Single thread, deterministic, resumable.

Usage (from the repository root):
    python research/attention_order_law_2026_11_13/run_train.py --phase 0 --arm AWA-2L --lr 1e-3
        Phase-0 pilot: seed 900, the first 800 steps of the main schedule.
    python research/attention_order_law_2026_11_13/run_train.py --phase 1 --arm AWA-2L --seed 0
        Phase-1 run: S = 3,000 steps at the learning rate selected in Phase 0. Refuses to start unless the freeze
        manifest exists and still matches the code, the evaluation sets and the Phase-0 selection.
    python research/attention_order_law_2026_11_13/run_train.py --phase smoke --arm AWA-2L --lr 1e-3 --steps 20 \
        --results /some/scratch/dir
        Smoke run: any step count; never part of a result.

A killed run continues from its last checkpoint when the same command is started again. Outputs:
<results>/phase<P>/runs/<run>/{curve.jsonl, ckpt.pt, meta.json}.
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
    from awa3 import arms, layout, phases, train

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", required=True, choices=layout.PHASES)
    ap.add_argument("--arm", required=True, choices=sorted(arms.ARMS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None, help="smoke runs only")
    ap.add_argument("--val-every", type=int, default=None, help="smoke runs only")
    ap.add_argument("--ckpt-every", type=int, default=None, help="smoke runs only")
    ap.add_argument("--val-items", type=int, default=None, help="smoke runs only")
    ap.add_argument("--signal", default=None, help="smoke runs only: query rows 'forward,inverse,prefix_forward,"
                                                   "prefix_inverse,chain' (default: the protocol's training signal)")
    ap.add_argument("--results", default=str(layout.DEFAULT_RESULTS))
    args = ap.parse_args(argv)
    results = pathlib.Path(args.results)

    if args.phase == "1":
        if args.lr is not None:
            ap.error("phase 1 takes its learning rate from the Phase-0 selection")
        if args.seed is None or args.seed not in layout.SEEDS[args.arm]:
            ap.error(f"phase 1 seeds of {args.arm}: {list(layout.SEEDS[args.arm])}")
        from awa3 import manifest

        report = manifest.check_freeze(results)
        if not report["ok"]:
            print("refusing to start a Phase-1 run: the freeze manifest is missing or no longer matches\n"
                  + json.dumps(report, indent=1), file=sys.stderr)
            return 2
        lr, seed = phases.selected_lr(results, args.arm), args.seed
    elif args.phase == "0":
        if args.arm not in arms.PILOT_ARMS or args.lr not in layout.PILOT_LRS:
            ap.error(f"phase 0 pilots: arms {list(arms.PILOT_ARMS)}, learning rates {list(layout.PILOT_LRS)}")
        if args.seed not in (None, layout.PILOT_SEED):
            ap.error(f"phase 0 uses seed {layout.PILOT_SEED}")
        lr, seed = args.lr, layout.PILOT_SEED
    else:
        if args.lr is None:
            ap.error("smoke runs need --lr")
        lr, seed = args.lr, 900 if args.seed is None else args.seed
    signal = None if args.signal is None else tuple(int(v) for v in args.signal.split(","))
    recipe = phases.recipe_for(args.phase, lr, args.steps, val_every=args.val_every, ckpt_every=args.ckpt_every,
                               val_items=args.val_items, signal=signal)
    run_dir = layout.run_dir(results, args.phase, args.arm, seed, lr)
    meta = train.train(run_dir, args.arm, seed, recipe, progress=lambda msg: print(msg, flush=True))
    print(json.dumps({"run_dir": str(run_dir), "steps": meta["steps"], "competence": meta["competence"],
                      "s_per_step": meta["timing"]["s_per_step"], "model_sha256": meta["model_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
