# -*- coding: utf-8 -*-
"""Plan constants and the results layout of the AWA-3 study (PROTOCOL sections 7 and 11)."""
from __future__ import annotations

import pathlib

from . import arms

STUDY = pathlib.Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
DEFAULT_RESULTS = STUDY / "results"

MAIN_STEPS = 3000                    # S, fixed in advance
EXTENSION_STEPS = 1500               # the one funded extension of a run that fails its competence check
EXTENSION_RESERVE_CORE_H = 1.0
PILOT_STEPS = 800
PILOT_EXTENSION_STEPS = 800          # AWA only, once, if the GO rule fails at step 800
PILOT_SEED = 900
PILOT_LRS = (5e-4, 1e-3, 2e-3)
STD_LR = 1e-3                        # STD-2L-pm is not piloted
SEEDS = {arm: tuple(range(spec["seeds"])) for arm, spec in arms.ARMS.items()}
EXTENSION_PRIORITY = ("AWA-2L", "ChaCAL-2L", "AWA-own-2L", "LOOPED-2L", "ARLOOP-2L")    # STD-2L-pm: no extension
NOGO_ARMS = ("ChaCAL-2L", "AWA-own-2L", "STD-2L-pm")                                     # Phase 1 under NO-GO
PHASES = ("0", "1", "smoke")


def phase_dir(results, phase: str) -> pathlib.Path:
    if str(phase) not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; phases: {PHASES}")
    return pathlib.Path(results) / f"phase{phase}"


def lr_tag(lr: float) -> str:
    return f"{lr:g}"


def run_name(arm: str, seed: int, phase: str, lr: float) -> str:
    """Run folder name: pilots and smoke runs carry their learning rate, main runs do not."""
    if str(phase) == "1":
        return f"{arm}_s{seed}"
    return f"{arm}_lr{lr_tag(lr)}_s{seed}"


def run_dir(results, phase: str, arm: str, seed: int, lr: float) -> pathlib.Path:
    return phase_dir(results, phase) / "runs" / run_name(arm, seed, phase, lr)


def selection_path(results) -> pathlib.Path:
    return phase_dir(results, "0") / "selection.json"
