# -*- coding: utf-8 -*-
"""AWA-3 analysis: the preregistered decisions of PROTOCOL sections 3 and 10, from the per-run eval records.

Usage (from the repository root, single thread):
    python research/attention_order_law_2026_11_13/run_analyze.py [--results DIR] [--phase 1] [--allow-truncated]

Reads  <results>/phase<P>/runs/<arm>_s<seed>/{meta.json, eval.json}
Writes <results>/phase<P>/decisions.json and <results>/phase<P>/tables/summary.md

``compute_decisions`` is a pure function of the loaded records (no file access, no randomness), so run_verify.py
can recompute the stored decisions. Nothing here looks at training curves; the unit is the training run (seed).
Every rule below quotes its PROTOCOL section; where PROGRAM.md left a detail open, the choice made here is
listed in PROTOCOL section 12 ("Open decisions to close before the freeze").
"""
from __future__ import annotations

import sys
import pathlib

STUDY = pathlib.Path(__file__).resolve().parent
REPO = STUDY.parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(STUDY), str(REPO)]

from awa3.analysis import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
