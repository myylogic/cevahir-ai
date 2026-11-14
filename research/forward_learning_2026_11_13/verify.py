# -*- coding: utf-8 -*-
"""Forward-learning study: freeze manifest, preconditions, record checks, re-analysis, comparator cross-check.

    python research/forward_learning_2026_11_13/verify.py --records [--results DIR]   validate records, completeness
    python research/forward_learning_2026_11_13/verify.py --freeze                    write freeze_manifest.sha256
    python research/forward_learning_2026_11_13/verify.py --preconditions             exit 1 unless confirm may run
    python research/forward_learning_2026_11_13/verify.py --rerun                     recompute decisions, compare
    python research/forward_learning_2026_11_13/verify.py --crosscheck                FT-L / FT-ALL vs CEV-11 records

The freeze is refused unless PROTOCOL.md carries a line "Status: FROZEN <YYYY-MM-DD>" and the discovery selection
(results/selection.json) exists with a valid hash. The manifest covers the protocol, the selection, the study
code and the engine module; any later change makes --preconditions fail, and run_world.py then refuses
confirmatory worlds.
"""
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fwl.verification import main as main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
