# -*- coding: utf-8 -*-
"""Deterministic single-thread trainer of the AWA-3 study (PROTOCOL section 7).

Recipe: batch 64; AdamW betas (0.9, 0.98), eps 1e-8, weight decay 0.01 on matrices only (none on norms, biases,
rho, tau, gate biases); 100 warm-up steps, cosine decay to 0.1x at step S, then constant 0.1x (extension steps);
gradient clipping at 1.0; single-token cross-entropy at the answer position with a rows-only final layer.

Streams: data from ``random.Random(f"train|{seed}")`` (shared by every arm with that seed: common random
numbers); budgets (K for walks, T for loops) from ``random.Random(f"budget|{seed}")``, drawn once per step in
every arm so that the stream position never depends on the arm; ``torch.manual_seed(seed)`` before the build.

A run directory holds ``curve.jsonl`` (one JSON record per line: "train" every ``log_every`` steps, "val" at the
validation steps), ``ckpt.pt`` (model, optimiser, both stream states, step, curve length) and ``meta.json``
(written when the planned steps are complete). A killed run resumes from ``ckpt.pt``: the curve is truncated to
the checkpointed length and the continuation is bit-identical to an uninterrupted run. CPU seconds are recorded
next to, not inside, the deterministic fields ("cpu_s").
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from . import arms, task

CKPT_FORMAT = "awa3-run-checkpoint"
CKPT_VERSION = 1
NONDETERMINISTIC_KEYS = ("cpu_s", "cpu_train_s", "cpu_val_s", "s_per_step", "timing")


@dataclasses.dataclass(frozen=True)
class Recipe:
    lr: float
    steps: int = 3000                 # planned number of updates (S for a main run, 800 for a pilot)
    schedule_steps: int = 3000        # S of the learning-rate schedule; a pilot is the start of a main run
    batch: int = 64
    warmup: int = 100
    lr_floor: float = 0.1             # cosine decays to lr_floor * lr at step S and stays there
    betas: Sequence[float] = (0.9, 0.98)
    eps: float = 1e-8
    weight_decay: float = 0.01
    clip: float = 1.0
    depths: Sequence[int] = task.TRAIN_DEPTHS
    budgets: Sequence[int] = arms.TRAIN_BUDGETS
    val_every: int = 200
    val_items: int = task.VAL_ID_ITEMS
    ckpt_every: int = 200
    log_every: int = 10

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def lr_at(step: int, recipe: Recipe) -> float:
    """Learning rate of update ``step`` (1-based): linear warm-up, cosine to the floor at S = schedule_steps,
    floor afterwards (extension steps)."""
    if step <= recipe.warmup:
        return recipe.lr * step / recipe.warmup
    progress = min(1.0, (step - recipe.warmup) / max(1, recipe.schedule_steps - recipe.warmup))
    return recipe.lr * (recipe.lr_floor + (1.0 - recipe.lr_floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def param_groups(model, weight_decay: float) -> List[dict]:
    """Weight decay on matrices only: vectors and scalars (norm scales, biases, rho, tau, gate bias) get none.

    Built from the live model, i.e. after the operator parameters exist. Tied weights appear once.
    """
    decay, no_decay, seen = [], [], set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def model_sha256(model) -> str:
    """Canonical weight digest: sorted state_dict keys; name, shape and fp32 bytes of every tensor."""
    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        t = state[key].detach().to(device="cpu", dtype=torch.float32).contiguous()
        h.update(key.encode("utf-8") + b"\0" + repr(tuple(t.shape)).encode("ascii") + b"\0")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def code_sha256() -> str:
    """Digest of this package and of the engine operator file (names and bytes, sorted)."""
    here = pathlib.Path(__file__).resolve().parent
    repo = here.parents[2]
    files = sorted(here.glob("*.py")) + [
        repo / "src/neural_network_module/ortak_katman_module/attention_manager_module/absorbing_walk.py"]
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- files
def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: pathlib.Path) -> List[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: pathlib.Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _truncate_jsonl(path: pathlib.Path, n_lines: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    if len(lines) < n_lines:
        raise RuntimeError(f"{path} has {len(lines)} records but the checkpoint expects {n_lines}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(line + "\n" for line in lines[:n_lines]), encoding="utf-8")
    os.replace(tmp, path)


def _rng_state_to_json(state) -> list:
    return [state[0], list(state[1]), state[2]]


def _rng_state_from_json(state) -> tuple:
    return (state[0], tuple(state[1]), state[2])


# -------------------------------------------------------------------------------------- validation
_VAL_CACHE: Dict[tuple, tuple] = {}


def _val_tensors(k: int, stratum: str, n_items: int):
    key = (k, stratum, n_items)
    if key not in _VAL_CACHE:
        _VAL_CACHE[key] = task.tensors(task.val_set(k)[:n_items], stratum)
    return _VAL_CACHE[key]


@torch.no_grad()
def validate(model, arm: str, n_items: int = task.VAL_ID_ITEMS, budget: int = arms.COMPETENCE_BUDGET,
             chunk: int = 300) -> dict:
    """In-distribution validation (k = 1, 2, 3; shuffled and sorted strata) at the competence budget.

    Returns {"cells": {k: {stratum: {"loss", "acc"}}}, "competence": mean accuracy over k of the arm's competence
    stratum, "competence_loss": the same for the loss, "k1_acc": accuracy at k = 1}.
    """
    was_training = model.training
    model.eval()
    use_budget = budget if arms.ARMS[arm]["forward"] in ("walk", "looped", "arloop") else None
    cells: Dict[str, dict] = {}
    for k in task.TRAIN_DEPTHS:
        cells[str(k)] = {}
        for stratum in ("shuffled", "sorted"):
            if k <= 2 and stratum == "sorted":
                cells[str(k)][stratum] = dict(cells[str(k)]["shuffled"])
                continue
            X, Y = _val_tensors(k, stratum, n_items)
            loss_sum, correct = 0.0, 0
            for i in range(0, X.shape[0], chunk):
                logits = arms.answer_logits(model, arm, X[i:i + chunk], use_budget)
                loss_sum += float(F.cross_entropy(logits, Y[i:i + chunk], reduction="sum"))
                correct += int((logits.argmax(-1) == Y[i:i + chunk]).sum())
            cells[str(k)][stratum] = {"loss": loss_sum / X.shape[0], "acc": correct / X.shape[0]}
    stratum = arms.competence_stratum(arm)
    model.train(was_training)
    return {
        "cells": cells,
        "competence_stratum": stratum,
        "competence": sum(cells[str(k)][stratum]["acc"] for k in task.TRAIN_DEPTHS) / len(task.TRAIN_DEPTHS),
        "competence_loss": sum(cells[str(k)][stratum]["loss"] for k in task.TRAIN_DEPTHS) / len(task.TRAIN_DEPTHS),
        "k1_acc": cells["1"]["shuffled"]["acc"],
    }


def competence_verdict(arm: str, val: dict) -> dict:
    """PROTOCOL section 7 competence check: walks, loops and lower-triangular arms need mean k <= 3 accuracy
    >= 0.90 on their competence stratum at budget 5; STD-2L-pm needs k = 1 accuracy >= 0.90 (binding)."""
    if arm == "STD-2L-pm":
        return {"metric": "k1_acc", "value": val["k1_acc"], "threshold": 0.90, "competent": val["k1_acc"] >= 0.90,
                "extendable": False}
    return {"metric": f"mean k<=3 accuracy, {val['competence_stratum']}, budget {arms.COMPETENCE_BUDGET}",
            "value": val["competence"], "threshold": 0.90, "competent": val["competence"] >= 0.90,
            "extendable": True}


# ----------------------------------------------------------------------------------------- training
def train(run_dir, arm: str, seed: int, recipe: Recipe, *, total_steps: Optional[int] = None,
          stop_after: Optional[int] = None, progress=None) -> dict:
    """Train (or resume) one run until ``total_steps`` (default: recipe.steps) and write its meta.

    ``total_steps`` > recipe.steps is the one allowed extension (it continues the same schedule; past
    schedule_steps the learning rate stays at the floor).
    ``stop_after`` ends the process state at that step WITHOUT finishing (used to test resumption); the run is
    then continued by calling train() again with the same arguments.
    Returns the meta dict (with "finished": False when stopped early).
    """
    torch.set_num_threads(1)
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    curve_path, ckpt_path, meta_path = run_dir / "curve.jsonl", run_dir / "ckpt.pt", run_dir / "meta.json"
    total = int(recipe.steps if total_steps is None else total_steps)
    if total < recipe.steps:
        raise ValueError("total_steps cannot be smaller than the recipe's planned steps")
    forward = arms.ARMS[arm]["forward"]

    model, config = arms.build(arm, seed)
    model.train()
    optim = torch.optim.AdamW(param_groups(model, recipe.weight_decay), lr=recipe.lr,
                              betas=tuple(recipe.betas), eps=recipe.eps)
    data_rng = random.Random(f"train|{seed}")
    budget_rng = random.Random(f"budget|{seed}")
    step, curve_len, cpu_train, cpu_val = 0, 0, 0.0, 0.0
    identity = {"arm": arm, "seed": int(seed), "recipe": recipe.to_json()}

    if ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if ck.get("format") != CKPT_FORMAT or ck.get("version") != CKPT_VERSION:
            raise ValueError(f"{ckpt_path} is not a {CKPT_FORMAT} v{CKPT_VERSION} file")
        if ck["identity"] != identity:
            raise ValueError(f"{ckpt_path} belongs to another run: {ck['identity']} != {identity}")
        model.load_state_dict(ck["model"], strict=True)
        if model_sha256(model) != ck["model_sha256"]:
            raise ValueError("weight digest mismatch after loading the checkpoint")
        optim.load_state_dict(ck["optim"])
        data_rng.setstate(_rng_state_from_json(ck["data_rng"]))
        budget_rng.setstate(_rng_state_from_json(ck["budget_rng"]))
        step, curve_len = int(ck["step"]), int(ck["curve_len"])
        cpu_train, cpu_val = float(ck["cpu_train_s"]), float(ck["cpu_val_s"])
        _truncate_jsonl(curve_path, curve_len)
    elif curve_path.exists():
        curve_path.unlink()                                       # records of a run that never checkpointed

    def save_checkpoint() -> None:
        payload = {"format": CKPT_FORMAT, "version": CKPT_VERSION, "identity": identity, "step": step,
                   "curve_len": curve_len, "model": model.state_dict(), "model_sha256": model_sha256(model),
                   "optim": optim.state_dict(), "data_rng": _rng_state_to_json(data_rng.getstate()),
                   "budget_rng": _rng_state_to_json(budget_rng.getstate()),
                   "cpu_train_s": cpu_train, "cpu_val_s": cpu_val}
        tmp = ckpt_path.with_name(ckpt_path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, ckpt_path)

    def run_validation() -> dict:
        nonlocal cpu_val, curve_len
        t0 = time.process_time()
        val = validate(model, arm, recipe.val_items)
        cpu_val += time.process_time() - t0
        record = {"type": "val", "step": step, **val, **arms.operator_readout(model, arm), "cpu_s": cpu_val}
        _append_jsonl(curve_path, record)
        curve_len += 1
        return val

    last_val = None
    if step == 0 and stop_after != 0:
        last_val = run_validation()                               # step-0 baseline
        save_checkpoint()
    while step < total and (stop_after is None or step < stop_after):
        t0 = time.process_time()
        step += 1
        x, y = task.make_batch(data_rng, recipe.batch, recipe.depths)
        budget = budget_rng.choice(tuple(recipe.budgets))         # drawn in every arm (stream alignment)
        use_budget = budget if forward in ("walk", "looped", "arloop") else None
        logits = arms.answer_logits(model, arm, x, use_budget)
        loss = F.cross_entropy(logits, y)
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"{arm} seed {seed}: non-finite loss at step {step}")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.clip))
        lr = lr_at(step, recipe)
        for group in optim.param_groups:
            group["lr"] = lr
        optim.step()
        cpu_train += time.process_time() - t0
        # the last planned step is always recorded (like validation and the checkpoint below): an extended run
        # then has the same curve as one that ran straight through, and its "cpu_s" there prices the extension
        if step % recipe.log_every == 0 or step == total or step == recipe.steps:
            _append_jsonl(curve_path, {"type": "train", "step": step, "loss": loss_value, "lr": lr,
                                       "budget": use_budget, "grad_norm": grad_norm, "cpu_s": cpu_train})
            curve_len += 1
        if step % recipe.val_every == 0 or step == total or step == recipe.steps:
            last_val = run_validation()
        if step % recipe.ckpt_every == 0 or step == total or step == recipe.steps:
            save_checkpoint()
        if progress is not None and (step % 50 == 0 or step == total):
            progress(f"{arm} seed {seed}: step {step}/{total} loss {loss_value:.4f} "
                     f"({cpu_train / max(1, step):.3f} cpu-s/step)")

    finished = step >= total
    if not finished:
        save_checkpoint()
        return {"finished": False, "step": step}
    if last_val is None:                                          # resumed exactly at the end
        vals = [r for r in read_jsonl(curve_path) if r["type"] == "val" and r["step"] == step]
        last_val = vals[-1] if vals else run_validation()
    meta = {
        "format": "awa3-run-meta", "finished": True, **identity, "steps": step, "planned_steps": recipe.steps,
        "extended": step > recipe.steps, "n_params": arms.count_parameters(model), "config": config,
        "model_sha256": model_sha256(model), "code_sha256": code_sha256(),
        "final_val": {k: last_val[k] for k in ("cells", "competence", "competence_loss", "competence_stratum",
                                                 "k1_acc")},
        "competence": competence_verdict(arm, last_val), **arms.operator_readout(model, arm),
        "timing": {"cpu_train_s": cpu_train, "cpu_val_s": cpu_val, "s_per_step": cpu_train / max(1, step)},
        "env": {"torch": torch.__version__, "python": platform.python_version(), "machine": platform.machine(),
                "threads": torch.get_num_threads()},
    }
    write_json(meta_path, meta)
    return meta


def load_run(run_dir):
    """(model in eval mode, meta) of a finished run; the weights are checked against the recorded digest."""
    run_dir = pathlib.Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("finished"):
        raise ValueError(f"{run_dir} is not a finished run")
    ck = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=True)
    model, _ = arms.build(meta["arm"], meta["seed"])
    model.load_state_dict(ck["model"], strict=True)
    if int(ck["step"]) != int(meta["steps"]) or model_sha256(model) != meta["model_sha256"]:
        raise ValueError(f"{run_dir}: checkpoint does not match meta.json (step or weight digest)")
    model.eval()
    return model, meta


def strip_nondeterministic(obj: Any) -> Any:
    """A copy without the timing fields (everything else must be reproducible bit for bit)."""
    if isinstance(obj, dict):
        return {k: strip_nondeterministic(v) for k, v in obj.items() if k not in NONDETERMINISTIC_KEYS and k != "env"}
    if isinstance(obj, list):
        return [strip_nondeterministic(v) for v in obj]
    return obj
