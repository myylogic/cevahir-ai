# -*- coding: utf-8 -*-
"""AWA-3 verification: freeze manifest, results manifest and recomputation of the decisions (no git involved).

Usage (from the repository root):
    python research/attention_order_law_2026_11_13/run_verify.py --freeze [--results DIR]
        After Phase 0, once PROTOCOL.md no longer says "NOT FROZEN": write ``frozen_sets.json`` (digests of every
        test and validation set, floors h_k and followable fractions P_f) and ``freeze_manifest.sha256``: the
        study code tree, the two test files, the engine files the arms are built from, frozen_sets.json and the
        Phase-0 selection (learning rates, GO/NO-GO). Refuses to overwrite an existing freeze.
    python research/attention_order_law_2026_11_13/run_verify.py --check-freeze [--results DIR]
        Re-hash everything the freeze lists and regenerate the frozen sets; exit 1 on any difference. Every
        Phase-1 job depends on this check.
    python research/attention_order_law_2026_11_13/run_verify.py --manifest [--results DIR]
        Hash the code tree and every file under the results directory into <results>/manifest.sha256.
    python research/attention_order_law_2026_11_13/run_verify.py --check [--results DIR]
        Re-hash the files of <results>/manifest.sha256: missing or changed files fail, unlisted ones are reported.
    python research/attention_order_law_2026_11_13/run_verify.py --rerun [--results DIR] [--phase 1]
        --check, then recompute the decisions from the run records with analysis.compute_decisions and compare
        them with the stored decisions.json (numbers within 1e-9); the analysis code must have the hash recorded
        in decisions.json and, when a freeze exists, the hash listed in the freeze manifest.

Manifests are ``sha256sum``-compatible; labels are relative to the study folder, so
``cd research/attention_order_law_2026_11_13 && sha256sum -c freeze_manifest.sha256`` works.
"""
from __future__ import annotations

import sys
import pathlib

STUDY = pathlib.Path(__file__).resolve().parent
REPO = STUDY.parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(STUDY), str(REPO)]

from awa3.manifest import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
