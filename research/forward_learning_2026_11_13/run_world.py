# -*- coding: utf-8 -*-
"""Forward-learning study: run (or resume) one world.

    python research/forward_learning_2026_11_13/run_world.py --run discovery --world 0 [--smoke]
    python research/forward_learning_2026_11_13/run_world.py --run confirm --world 1000

Discovery worlds are 0 and 1, confirmatory worlds 1000-1007. The base weights are read from $CEV_BASES (the
folder that holds discovery_w0_K20/model.pt, ...); nothing in the CEV-11 study folder is written. Records go to
results/<run>/w<world>.jsonl (append-only, resumable). Confirmatory worlds are refused until the protocol is
frozen and the freeze manifest matches (verify.py --freeze / --preconditions).

--smoke runs a reduced setting that mirrors the v1 pilots (not protocol data) and prints timings and a
comparison with the pilot numbers; it writes to results/smoke unless --out is given.
"""
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fwl.cli import world_main as main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
