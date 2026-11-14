# -*- coding: utf-8 -*-
"""Forward-learning study: confirmatory analysis (PROTOCOL §4, §7). Unit = world, exact sign-flip test, Holm.

    python research/forward_learning_2026_11_13/analyze.py [--results DIR] [--selection FILE] [--out FILE]

Reads the confirmatory records (results/confirm/w1000..w1007.jsonl) and the frozen discovery selection, writes
results/decisions.json and prints the labels. Incomplete world files are refused; fewer than six worlds give
"not testable". ``--describe DIR`` prints descriptive tables of any results folder (no test).
"""
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fwl.cli import analyze_main as main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
