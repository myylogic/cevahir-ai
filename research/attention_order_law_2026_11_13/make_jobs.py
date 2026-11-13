# -*- coding: utf-8 -*-
"""AWA-3: job lists for an external scheduler. Each job is {"id", "cmd", "deps", "log"}.

Usage (from the repository root):
    python research/attention_order_law_2026_11_13/make_jobs.py --check     compare the committed files
    python research/attention_order_law_2026_11_13/make_jobs.py --write     regenerate them

Files (next to this script):
    jobs_phase0.json        15 pilots (5 arms x 3 learning rates, seed 900, 800 steps) and the selection job.
    jobs_phase1.json        GO branch: freeze check, 26 training runs (seeds round-robin across arms), the
                            extension job, 26 evaluations, analysis, manifest, verification.
    jobs_phase1_nogo.json   NO-GO branch: the same pipeline for ChaCAL-2L x 5, AWA-own-2L x 5 and STD-2L-pm x 3.

Every ``cmd`` runs from the repository root; ``log`` is a path relative to the repository root (the folder
results/job_logs/ exists in the checkout). A job may start when all its ``deps`` have finished successfully. Every
driver is single-threaded and resumable, so a failed or killed job can simply be started again. Phase 1 must not
be scheduled before the study is frozen: its first job fails otherwise, and every other job depends on it.
"""
from __future__ import annotations

import sys
import pathlib

STUDY = pathlib.Path(__file__).resolve().parent
REPO = STUDY.parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(STUDY), str(REPO)]

from awa3.jobs import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
