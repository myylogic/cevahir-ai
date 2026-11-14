# -*- coding: utf-8 -*-
"""Part A: single edits. One read of one sentence, one write, CEV-11's probe families (PROTOCOL §5.1).

Every one-step arm reads the lived sentence once and writes all six fc2 matrices (or the top one) with one step
along its direction; the step is found by the engine's forward-only search under an install criterion. The
exact-gradient arms (BP-1, BP-1b, BP-pred) take their fields from one backward pass and then use the same
write, so only the credit rule differs. FT-L and FT-ALL run CEV-11's own code at its frozen selection.
"""
from __future__ import annotations

import hashlib
import random
import time
from typing import Dict, List, Optional, Tuple

import torch

from . import common as C

ENGINE_RULES = ("FEC-TOP", "FEC-0", "FEC-1", "FEC-A", "HEBB-TOP", "HEBB-0", "RAND-0")
BP_RULES = ("BP-1", "BP-1b", "BP-pred")


def protocol_items(base, editable, world: int, cfg: C.RunConfig) -> List[dict]:
    """CEV-11's confirmatory item list for this world (same seed and ``per_case``), cut to the first
    ``edits_per_case`` items of every lived case in list order."""
    from cev11 import probes as P
    items = P.edit_items(base.world, editable, seed=world + 2, per_case=cfg.per_case)
    count: Dict[str, int] = {}
    out = []
    for it in items:
        c = it["lived_case"]
        if count.get(c, 0) < cfg.edits_per_case:
            out.append(it)
            count[c] = count.get(c, 0) + 1
    return out[:cfg.max_edits] if cfg.max_edits else out


def fact_tokens(base, item: dict) -> Tuple[torch.Tensor, List[int]]:
    """Tokens of the lived sentence and the positions whose next token is the (external) object span."""
    from cev11.world import frame_prefix
    from tokenizer_management.byte_bpe import BOS_ID
    pid = [BOS_ID] + base.tok.encode(frame_prefix(item["name"], item["rel"], item["lived_case"]))
    tid = base.tok.encode(" " + item["new"] + ".")
    return torch.tensor(pid + tid, dtype=torch.long), list(range(len(pid) - 1, len(pid) + len(tid) - 1))


def edit_seed(edit_id: str) -> int:
    """Seed of the random-feedback control: a function of the edit id only."""
    return int(hashlib.sha256(edit_id.encode("utf-8")).hexdigest()[:8], 16)


def so_family(base, item: dict, answers, world: int, k: int) -> Tuple[List[dict], int]:
    """Leak family SO: up to ``k`` other probe-pool entities whose old answer equals the edit's old answer,
    seven formal frames each. Built on the BASE model, like CEV-11's leak families."""
    from cev11 import experiment as E
    from cev11.decode import greedy_text
    from cev11.morph_tr import CASES
    from cev11.world import frame_prefix
    if answers.locked:
        raise RuntimeError("so_family must run on the BASE model")
    w = base.world
    pool = sorted(n for n, p in w.partition.items()
                  if p == "probe" and n != item["name"] and w.facts[(n, item["rel"])] == item["old"])
    rnd = random.Random(f"{world}|{item['edit_id']}|SO")
    chosen = rnd.sample(pool, min(k, len(pool)))
    probes = [{"prefix": frame_prefix(m, item["rel"], c), "name": m, "rel": item["rel"], "case": c}
              for m in chosen for c in CASES]
    for d in probes:
        d["base_answer"] = answers(d["prefix"])
        if d["base_answer"] is None:
            d["base_text"] = greedy_text(base.model, base.tok, d["prefix"])
    E.mark_base_consistency(base, probes)
    return probes, len(chosen)


def measure_installed(base, prep: dict, so: Tuple[List[dict], int], bpb_texts: Optional[List[str]]) -> dict:
    """All probe families under whatever write is installed now (standard forward path)."""
    from cev11 import experiment as E
    item = prep["item"]
    res = E.measure(base, prep, item["new"], item["old"], families=C.FAMILIES)
    probes, n_ent = so
    if probes:
        changed, n, _ = E._changed(base, probes)
        adopted = int(sum(E._exact(base, [d["prefix"] for d in probes], [item["new"]] * len(probes))))
        res["leak"]["SO"] = {"changed": changed, "n": n, "entities": n_ent, "adopted": adopted}
    if bpb_texts is not None:
        res["bpb"] = {"base": base.layer_cache["fwl_bpb_base"], "edited": E.bits_per_byte(base, bpb_texts)}
    return res


def make_reading(base, learner, item: dict, arm: str):
    """The reading of ``arm`` for the lived sentence of ``item`` (one forward pass; BP arms add one backward)."""
    FL = C.engine()
    tok, pos = fact_tokens(base, item)
    if arm in ENGINE_RULES:
        return learner.read(tok, pos, rule=arm, seed=edit_seed(item.get("edit_id", item["name"])))
    if arm not in BP_RULES:
        raise ValueError(f"unknown one-step arm {arm!r}")
    ref = FL.backprop_fields(base.model, tok, pos)
    fields = ref["fields"]
    if arm == "BP-pred":                       # exact field at the loss positions only
        keep = torch.zeros(fields[0].shape[0], dtype=torch.bool)
        keep[pos] = True
        fields = [f * keep.unsqueeze(-1) for f in fields]
    return learner.reading_from_fields(tok, pos, fields, ref["keys"], rule=arm)


def learner_config(arm: str, install: str, cfg: C.RunConfig):
    FL = C.engine()
    cap = cfg.caps.get("single", {}).get(f"{arm}|{install}")
    kw = dict(rule=arm if arm in ENGINE_RULES else "FEC-A", projector=False, guard="off", install=install,
              normalize="per_block" if arm == "BP-1b" else "global", mu_cap=cap)
    if cfg.single_grid is not None:
        kw["step_grid"] = tuple(cfg.single_grid)
    return FL.ForwardLearningConfig(**kw)


def run_one_step(base, prep: dict, so, arm: str, install: str, cfg: C.RunConfig, run: str, world: int,
                 bpb_texts: Optional[List[str]]) -> dict:
    """One (edit, arm, install criterion): read, write, measure, undo. Returns the record."""
    FL = C.engine()
    item = prep["item"]
    t0 = time.time()
    learner = FL.ForwardLearner(base.model, learner_config(arm, install, cfg))
    try:
        result = learner.write(make_reading(base, learner, item, arm), tag=item["edit_id"])
        if result.installed:
            res = measure_installed(base, prep, so, bpb_texts)
        else:
            res = {"installed": False, "PF6": {}, "targets": [], "leak": {}}
    finally:
        learner.close()
    extra = {"bpb": res["bpb"]} if "bpb" in res else {}
    return C.make_record(
        "single", run, world, f"single|{item['edit_id']}|{arm}|{install}", edit_id=item["edit_id"], arm=arm,
        install=install, lived_case=item["lived_case"], installed=bool(result.installed and res["installed"]),
        n_targets=len(prep["targets"]), write_installed=bool(result.installed), mu=result.mu,
        flags=list(result.flags),
        n_forward=int(result.n_forward), nll_before=result.nll_before, nll_after=result.nll_after,
        block_share={str(b): v for b, v in result.block_share.items()}, PF6=res["PF6"], targets=res["targets"],
        OLD=res.get("OLD"), leak=res["leak"], seconds=round(time.time() - t0, 3), **extra)


def run_ft(base, inst, prep: dict, so, arm: str, run: str, world: int, bpb_texts: Optional[List[str]]) -> dict:
    """CEV-11's FT-L / FT-ALL at its frozen selection, measured with the same families (and SO)."""
    from cev11 import experiment as E
    spec = C.FT_L if arm == "FT-L" else C.FT_ALL
    item = prep["item"]
    t0 = time.time()
    w = E.compute_write(base, item, arm, spec["layer"], spec["hp"], world)
    try:
        if arm == "FT-L":
            inst.set_M(spec["layer"], w["M"])
        else:
            inst.load_state(w["state"])
        res = measure_installed(base, prep, so, bpb_texts)
    finally:
        inst.restore()
    diag = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            for k, v in w["diag"].items() if isinstance(v, (int, float, str, bool))}
    extra = {"bpb": res["bpb"]} if "bpb" in res else {}
    return C.make_record(
        "single", run, world, f"single|{item['edit_id']}|{arm}|ft", edit_id=item["edit_id"], arm=arm,
        install="ft", lived_case=item["lived_case"], installed=bool(res["installed"]),
        n_targets=len(prep["targets"]), write_installed=bool(res["installed"]), mu=None, flags=[], n_forward=0,
        config=spec["config"],
        diag=diag, PF6=res["PF6"], targets=res["targets"], OLD=res.get("OLD"), leak=res["leak"],
        seconds=round(time.time() - t0, 3), **extra)


def run_part_a(base, editable, world: int, run: str, cfg: C.RunConfig, writer: C.RecordWriter,
               log=print) -> dict:
    """All single-edit records of one world (resumable). Returns timing information."""
    from cev11 import experiment as E
    from cev11.model import model_sha256
    items = protocol_items(base, editable, world, cfg)
    answers = E.Answers(base)
    inst = E.Installer(base)
    held = list(base.prose_held[:cfg.bpb_texts])
    timing = {"prepare": 0.0, "arms": 0.0, "n_edits": len(items)}
    jobs = [(arm, inst_) for arm in cfg.single_arms if arm not in C.SINGLE_FT for inst_ in cfg.installs]
    jobs += [(arm, "ft") for arm in cfg.single_arms if arm in C.SINGLE_FT]
    for n, item in enumerate(items):
        todo = [(a, i) for a, i in jobs if not writer.has(f"single|{item['edit_id']}|{a}|{i}")]
        if not todo:
            continue
        t0 = time.time()
        prep = E.prepare_item(base, item, answers, seed=world + 3)
        so = so_family(base, item, answers, world, cfg.so_entities)
        if "fwl_bpb_base" not in base.layer_cache:
            base.layer_cache["fwl_bpb_base"] = E.bits_per_byte(base, held)
        timing["prepare"] += time.time() - t0
        t0 = time.time()
        answers.locked = True
        try:
            for arm, install in todo:
                primary = install in ("ft", cfg.installs[0])
                bpb = held if (n < cfg.bpb_edits and arm in cfg.bpb_arms and primary) else None
                if arm in C.SINGLE_FT:
                    rec = run_ft(base, inst, prep, so, arm, run, world, bpb)
                else:
                    rec = run_one_step(base, prep, so, arm, install, cfg, run, world, bpb)
                writer.write(rec)
        finally:
            answers.locked = False
        timing["arms"] += time.time() - t0
        log(f"[{run} w{world}] part A edit {n + 1}/{len(items)} {item['edit_id']} "
            f"({timing['prepare'] + timing['arms']:.0f}s)")
    release(inst)
    if model_sha256(base.model) != base.base_sha:
        raise RuntimeError("the base model changed during part A")
    return timing


def release(inst) -> None:
    """Restore the base state and detach the CEV-11 memory hooks an ``Installer`` created."""
    inst.restore()
    for mem in inst.mems.values():
        mem.remove()
    inst.mems.clear()
