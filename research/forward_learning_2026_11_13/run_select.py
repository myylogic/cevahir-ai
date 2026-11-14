# -*- coding: utf-8 -*-
"""Forward-learning study: operating points from the discovery worlds (PROTOCOL §6).

    python research/forward_learning_2026_11_13/run_select.py [--results DIR] [--out FILE]

Reads results/discovery/w0.jsonl and w1.jsonl (both must be complete) and writes results/selection.json: step
caps (4 x the discovery median per arm; one common cap for the unit-normalised stream arms), the best exact
one-step variant BP*, the stream learner FEC* (by net facts per write; CMP* is descriptive, H3 is tested against
every comparator), and the gate operating points (theta*, gain*) at hard-negative leak <= 2% and prose bpb
change <= 0.01. The file carries its own sha256, the engine and configuration hashes of the discovery records,
and is part of the freeze manifest.
"""
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fwl.cli import select_main as main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
