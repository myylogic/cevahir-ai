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

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

STUDY = pathlib.Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]

PREFIX = "research/attention_order_law_2026_11_13"
FILES = {"jobs_phase0.json": "0", "jobs_phase1.json": "1", "jobs_phase1_nogo.json": "1-nogo"}


def _job(job_id: str, script: str, args: str, deps: Sequence[str]) -> dict:
    cmd = f"python {PREFIX}/{script}" + (f" {args}" if args else "")
    return {"id": job_id, "cmd": cmd, "deps": list(deps), "log": f"{PREFIX}/results/job_logs/{job_id}.log"}


def phase0_jobs() -> List[dict]:
    from . import arms, layout

    jobs = []
    for lr in layout.PILOT_LRS:
        for arm in arms.PILOT_ARMS:
            jobs.append(_job(f"p0_{arm}_lr{layout.lr_tag(lr)}", "run_train.py",
                             f"--phase 0 --arm {arm} --lr {layout.lr_tag(lr)}", []))
    jobs.append(_job("p0_select", "run_phase0_select.py", "", [j["id"] for j in jobs]))
    return jobs


def phase1_jobs(arm_names: Optional[Sequence[str]] = None) -> List[dict]:
    from . import arms, layout

    arm_names = list(arms.ARM_ORDER if arm_names is None else arm_names)
    jobs = [_job("p1_freeze_check", "run_verify.py", "--check-freeze", [])]
    runs = [(arm, seed) for seed in range(max(len(layout.SEEDS[a]) for a in arm_names))     # seeds round-robin
            for arm in arm_names if seed in layout.SEEDS[arm]]
    for arm, seed in runs:
        jobs.append(_job(f"p1_train_{arm}_s{seed}", "run_train.py", f"--phase 1 --arm {arm} --seed {seed}",
                         ["p1_freeze_check"]))
    train_ids = [f"p1_train_{arm}_s{seed}" for arm, seed in runs]
    jobs.append(_job("p1_extend", "run_extend.py", "", train_ids))
    for arm, seed in runs:
        jobs.append(_job(f"p1_eval_{arm}_s{seed}", "run_eval.py", f"--phase 1 --arm {arm} --seed {seed}",
                         ["p1_extend"]))
    jobs.append(_job("p1_analyze", "run_analyze.py", "--phase 1", [f"p1_eval_{arm}_s{seed}" for arm, seed in runs]))
    jobs.append(_job("p1_manifest", "run_verify.py", "--manifest", ["p1_analyze"]))
    jobs.append(_job("p1_verify", "run_verify.py", "--rerun --phase 1", ["p1_manifest"]))
    return jobs


def build() -> Dict[str, List[dict]]:
    from . import layout

    return {"jobs_phase0.json": phase0_jobs(), "jobs_phase1.json": phase1_jobs(),
            "jobs_phase1_nogo.json": phase1_jobs(layout.NOGO_ARMS)}


def render(jobs: List[dict]) -> str:
    return json.dumps(jobs, indent=1) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    status = 0
    for name, jobs in build().items():
        path = STUDY / name
        if args.write:
            path.write_text(render(jobs), encoding="utf-8")
            print(f"wrote {path} ({len(jobs)} jobs)")
        elif not path.is_file() or path.read_text(encoding="utf-8") != render(jobs):
            print(f"{path} is missing or differs from the generator", file=sys.stderr)
            status = 1
    return status

