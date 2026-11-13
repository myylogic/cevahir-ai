# -*- coding: utf-8 -*-
"""Phase logic of the AWA-3 study: Phase-0 learning-rate selection and GO/NO-GO, Phase-1 extensions.

Everything here is a deterministic function of the run records; the command-line drivers in the study folder
only parse arguments and call these functions.
"""
from __future__ import annotations

import json
import pathlib
from typing import Callable, Dict, List, Optional

from . import arms, layout, train

SELECTION_FORMAT = "awa3-phase0-selection"
TIE_TOLERANCE = 0.02                 # losses within 2% of the best count as ties -> the smaller lr
GO_ACCURACY = 0.30                   # k = 2 accuracy at the decision step
GO_LOSS_RATIO = 0.80                 # or: k = 2 validation loss <= 0.8 x its step-200 value


def recipe_for(phase: str, lr: float, steps: Optional[int] = None, *, val_every: Optional[int] = None,
               ckpt_every: Optional[int] = None, val_items: Optional[int] = None) -> train.Recipe:
    """The recipe of a phase. Phase 0 runs the first 800 steps of the main schedule; phase 1 runs S = 3,000 steps;
    only smoke runs may change the step count or the validation / checkpoint cadence."""
    phase = str(phase)
    overrides = {k: v for k, v in dict(val_every=val_every, ckpt_every=ckpt_every, val_items=val_items).items()
                 if v is not None}
    if phase == "1":
        if steps is not None or overrides:
            raise ValueError("phase 1 runs the frozen recipe: S = 3,000 steps, no overrides")
        return train.Recipe(lr=lr, steps=layout.MAIN_STEPS, schedule_steps=layout.MAIN_STEPS)
    if phase == "0":
        if steps is not None or overrides:
            raise ValueError("phase 0 runs the fixed pilot: 800 steps of the main schedule, no overrides")
        return train.Recipe(lr=lr, steps=layout.PILOT_STEPS, schedule_steps=layout.MAIN_STEPS)
    if phase == "smoke":
        n = int(steps if steps is not None else 20)
        return train.Recipe(lr=lr, steps=n, schedule_steps=max(n, layout.MAIN_STEPS), **overrides)
    raise ValueError(f"unknown phase {phase!r}")


# ------------------------------------------------------------------------------------------ phase 0
def _val_at(curve: List[dict], step: int) -> Optional[dict]:
    records = [r for r in curve if r["type"] == "val" and r["step"] == step]
    return records[-1] if records else None


def pilot_table(results) -> Dict[str, Dict[str, dict]]:
    """{arm: {lr tag: row}} for every finished pilot run. A row always describes step 800 ("loss" = validation
    loss on the arm's competence cells, "k2_acc", "k2_loss", "k2_loss_200"); a run that was extended also carries
    "extension" = {"steps", "k2_acc", "k2_loss"} at its last step."""
    table: Dict[str, Dict[str, dict]] = {}
    for arm in arms.PILOT_ARMS:
        table[arm] = {}
        for lr in layout.PILOT_LRS:
            rd = layout.run_dir(results, "0", arm, layout.PILOT_SEED, lr)
            meta_path = rd / "meta.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not meta.get("finished"):
                continue
            curve = train.read_jsonl(rd / "curve.jsonl")
            at_800, early = _val_at(curve, layout.PILOT_STEPS), _val_at(curve, 200)
            row = {
                "lr": lr, "loss": at_800["competence_loss"], "competence": at_800["competence"],
                "k2_acc": at_800["cells"]["2"]["shuffled"]["acc"],
                "k2_loss": at_800["cells"]["2"]["shuffled"]["loss"],
                "k2_loss_200": early["cells"]["2"]["shuffled"]["loss"] if early else None,
                "s_per_step": meta["timing"]["s_per_step"],
            }
            if meta["steps"] > layout.PILOT_STEPS:
                last = _val_at(curve, meta["steps"])
                row["extension"] = {"steps": meta["steps"], "k2_acc": last["cells"]["2"]["shuffled"]["acc"],
                                    "k2_loss": last["cells"]["2"]["shuffled"]["loss"]}
            table[arm][layout.lr_tag(lr)] = row
    return table


def select_lr(rows: Dict[str, dict]) -> float:
    """Lowest in-distribution validation loss on the arm's competence cells; ties within 2% -> the smaller lr."""
    if not rows:
        raise ValueError("no finished pilot run for this arm")
    best = min(r["loss"] for r in rows.values())
    return min(r["lr"] for r in rows.values() if r["loss"] <= best * (1.0 + TIE_TOLERANCE))


def go_rule(row: dict, extended: bool = False) -> bool:
    """AWA GO rule: k = 2 accuracy >= 0.30, or k = 2 validation loss <= 0.8 x its step-200 value.

    Applied at step 800, or (``extended``) at the last step of the one allowed 800-step extension."""
    at = row["extension"] if extended else row
    if at["k2_acc"] >= GO_ACCURACY:
        return True
    return row["k2_loss_200"] is not None and at["k2_loss"] <= GO_LOSS_RATIO * row["k2_loss_200"]


def select_phase0(results, *, extend: Optional[Callable[[float], None]] = None) -> dict:
    """Learning rates of every piloted arm and the AWA GO/NO-GO decision.

    ``extend(lr)`` is called at most once, when no AWA pilot passes the GO rule at step 800: it must extend the
    best-lr AWA pilot by 800 steps; the rule is then applied to that run at step 1,600. Without ``extend`` the
    decision is reported as NO-GO with "extension_pending".
    """
    table = pilot_table(results)
    missing = [f"{arm} lr {layout.lr_tag(lr)}" for arm in arms.PILOT_ARMS for lr in layout.PILOT_LRS
               if layout.lr_tag(lr) not in table[arm]]
    if missing:
        raise ValueError(f"unfinished pilot runs: {missing}")
    lrs = {arm: select_lr(table[arm]) for arm in arms.PILOT_ARMS}
    lrs["STD-2L-pm"] = layout.STD_LR
    awa = table["AWA-2L"]
    go = any(go_rule(r) for r in awa.values())
    extended, pending = False, False
    best_tag = layout.lr_tag(lrs["AWA-2L"])
    if not go:
        if "extension" in awa[best_tag]:
            extended = True                                        # the extension already ran
        elif extend is not None:
            extend(lrs["AWA-2L"])
            extended = True
            table = pilot_table(results)
            awa = table["AWA-2L"]
        else:
            pending = True
        if extended:
            go = go_rule(awa[best_tag], extended=True)
    return {"format": SELECTION_FORMAT, "lrs": lrs, "go": bool(go), "awa_extended": extended,
            "extension_pending": pending, "pilot": table,
            "phase1_arms": list(arms.ARM_ORDER) if go else list(layout.NOGO_ARMS)}


def selected_lr(results, arm: str) -> float:
    path = layout.selection_path(results)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: Phase 0 must finish (run_phase0_select.py) before Phase 1")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("format") != SELECTION_FORMAT or selection.get("extension_pending"):
        raise ValueError(f"{path} is not a finished Phase-0 selection")
    if arm not in selection["phase1_arms"]:
        raise ValueError(f"{arm} is not part of Phase 1 under the Phase-0 decision (go = {selection['go']})")
    return float(selection["lrs"][arm])


# ------------------------------------------------------------------------------------------ phase 1
def extension_queue(results) -> List[dict]:
    """Finished Phase-1 runs that fail their competence check, in the protocol's priority order."""
    queue = []
    for arm in layout.EXTENSION_PRIORITY:
        for seed in layout.SEEDS[arm]:
            rd = layout.phase_dir(results, "1") / "runs" / f"{arm}_s{seed}"
            meta_path = rd / "meta.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("finished") and not meta["competence"]["competent"] and not meta["extended"]:
                queue.append({"arm": arm, "seed": seed, "run_dir": str(rd), "lr": meta["recipe"]["lr"],
                              "s_per_step": meta["timing"]["s_per_step"]})
    return queue


def extended_runs(results) -> List[dict]:
    """Finished Phase-1 runs that carry an extension, in the protocol's priority order."""
    found = []
    for arm in layout.EXTENSION_PRIORITY:
        for seed in layout.SEEDS[arm]:
            rd = layout.phase_dir(results, "1") / "runs" / f"{arm}_s{seed}"
            meta_path = rd / "meta.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("finished") and meta.get("extended"):
                found.append({"arm": arm, "seed": seed, "run_dir": str(rd), "meta": meta})
    return found


def extension_cost(run_dir, meta: dict) -> float:
    """CPU seconds of the extension steps of a finished, extended run, from its own records: the training time
    in ``meta.json`` minus the cumulative training time the curve records at the last planned step."""
    planned = int(meta["planned_steps"])
    at_planned = [r for r in train.read_jsonl(pathlib.Path(run_dir) / "curve.jsonl")
                  if r["type"] == "train" and r["step"] == planned]
    if not at_planned:
        raise RuntimeError(f"{run_dir}: the curve has no training record at step {planned}; the cost of the "
                           f"extension cannot be booked")
    return float(meta["timing"]["cpu_train_s"]) - float(at_planned[-1]["cpu_s"])


def run_extensions(results, reserve_core_h: float = layout.EXTENSION_RESERVE_CORE_H, progress=None) -> dict:
    """Extend failing runs by 1,500 steps, in priority order, while the fixed CPU reserve lasts.

    A run is funded when the reserve still covers its projected cost (1,500 x its measured CPU-s/step). The ledger
    (``extensions.json``) is rewritten after every run, so the job can be restarted.

    Restart safety. A finished extension rewrites the run's ``meta.json`` ("extended": true) before the ledger
    is rewritten. If the job is killed between the two, the run is no longer in the queue; every start therefore
    first books each extended run that the ledger does not list, with its cost taken from the run's own records
    (``extension_cost``), so the reserve cannot be under-counted. Such entries carry "recovered": true.
    """
    ledger_path = layout.phase_dir(results, "1") / "extensions.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {
        "format": "awa3-extensions", "reserve_cpu_s": reserve_core_h * 3600.0, "spent_cpu_s": 0.0, "runs": {}}
    for item in extended_runs(results):
        key = f"{item['arm']}_s{item['seed']}"
        if ledger["runs"].get(key, {}).get("funded"):
            continue
        cost = extension_cost(item["run_dir"], item["meta"])
        ledger["spent_cpu_s"] += cost
        ledger["runs"][key] = {"funded": True, "cpu_s": cost, "recovered": True,
                               "competent_after": item["meta"]["competence"]["competent"]}
        train.write_json(ledger_path, ledger)
    for item in extension_queue(results):
        key = f"{item['arm']}_s{item['seed']}"
        if key in ledger["runs"]:
            continue
        projected = layout.EXTENSION_STEPS * item["s_per_step"]
        if ledger["spent_cpu_s"] + projected > ledger["reserve_cpu_s"]:
            ledger["runs"][key] = {"funded": False, "projected_cpu_s": projected}
        else:
            before = json.loads((pathlib.Path(item["run_dir"]) / "meta.json").read_text(encoding="utf-8"))
            recipe = train.Recipe(lr=item["lr"], steps=layout.MAIN_STEPS, schedule_steps=layout.MAIN_STEPS)
            meta = train.train(item["run_dir"], item["arm"], item["seed"], recipe,
                               total_steps=layout.MAIN_STEPS + layout.EXTENSION_STEPS, progress=progress)
            cost = meta["timing"]["cpu_train_s"] - before["timing"]["cpu_train_s"]
            ledger["spent_cpu_s"] += cost
            ledger["runs"][key] = {"funded": True, "cpu_s": cost, "competent_after": meta["competence"]["competent"]}
        train.write_json(ledger_path, ledger)
    train.write_json(ledger_path, ledger)
    return ledger
