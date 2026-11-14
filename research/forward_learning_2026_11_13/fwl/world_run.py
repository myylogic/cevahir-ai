# -*- coding: utf-8 -*-
"""One world: Part A (single edits) and Part B (streams, gate, capacity, retraction), resumable JSONL records."""
from __future__ import annotations

import time
from typing import Dict, List, Sequence

import torch

from . import common as C
from .single import protocol_items, run_part_a
from .stream import run_stream

# v1 pilot values on discovery world 0 (scratchpad pilots fec_pilot_cev11.py with 6 edits and
# fec_stream_pilot2.py / fec_stream_ftref.py with 30 facts; smallest installing step, the pilots' grids).
PILOT_W0 = {
    "single": {"FEC-A": {"PE6": 1.0, "IE5": 0.9, "mu_median": 0.7071}, "FEC-1": {"PE6": 1.0, "IE5": 0.9},
               "FEC-0": {"PE6": 0.7778, "IE5": 0.8333}, "FEC-TOP": {"PE6": 0.8333, "IE5": 0.8333},
               "BP-1": {"PE6": 0.8056, "IE5": 0.7278}, "BP-pred": {"PE6": 0.9167, "IE5": 0.9},
               "RAND-0": {"WE": 0.0}},
    "stream30": {"FEC-TOP+RLS": {"retention": 1.0, "PF6": 0.678, "IF": 0.561, "damage_AB": 0.1042,
                                 "dbpb": 0.0561},
                 "FEC-A+RLS": {"retention": 0.867, "PF6": 0.628, "IF": 0.529, "damage_AB": 0.0542,
                               "dbpb": 0.0084}},
    "stream10_note": "the FT-L pilot ran 30 facts (retention 0.333, damage 0.2375); the smoke stops it at 10",
}


def header(base, world: int, run: str, cfg: C.RunConfig, items: List[dict]) -> dict:
    return C.make_record("meta", run, world, "meta|run", config=cfg.to_json(),
                         config_sha256=C.sha256_json(cfg.to_json()), base_sha256=base.base_sha,
                         engine_sha256=C.sha256_file(C.ENGINE_FILE), items_sha256=C.sha256_json(items),
                         torch=torch.__version__, threads=torch.get_num_threads(), n_items=len(items))


def run_world(world: int, run: str, cfg: C.RunConfig, out_dir, *, base_dir=None, parts: Sequence[str] = ("A", "B"),
              base=None, editable=None, log=print) -> dict:
    """Run (or resume) one world. A results file started with another configuration, base or engine is refused."""
    t0 = time.time()
    torch.set_num_threads(1)
    if base is None:
        base, editable = C.load_base(world, base_dir)
    else:
        C.check_base_world(base, world)
    load_s = time.time() - t0
    writer = C.RecordWriter(C.results_path(out_dir, world))
    if writer.repaired:
        log(f"[{run} w{world}] {writer.path.name}: unfinished last line {writer.repaired}")
    items = protocol_items(base, editable, world, cfg)
    head = header(base, world, run, cfg, items)
    if writer.has("meta|run"):
        old = next(r for r in C.read_records(writer.path) if r["key"] == "meta|run")
        for k in ("config_sha256", "base_sha256", "engine_sha256", "items_sha256", "run"):
            if old[k] != head[k]:
                raise RuntimeError(f"{writer.path}: existing records were made with another {k}; "
                                   f"use a new output folder")
    else:
        writer.write(head)
    timing: Dict[str, object] = {"load": round(load_s, 1)}
    if "A" in parts:
        timing["A"] = {k: round(v, 1) for k, v in run_part_a(base, editable, world, run, cfg, writer, log).items()}
    if "B" in parts:
        for s in range(cfg.streams):
            timing[f"B{s}"] = {k: round(v, 1) for k, v in
                               run_stream(base, editable, world, s, run, cfg, writer, log).items()}
    timing["total"] = round(time.time() - t0, 1)
    return {"world": world, "run": run, "records": str(writer.path), "n_records": len(writer.done),
            "timing": timing}


# ---------------------------------------------------------------------------------------------------- exactness
def exactness_check(base, items: List[dict]) -> dict:
    """The engine's closed forms against autograd on a trained base (the unit tests use random tiny models):
    relative errors of the output error, of the FEC-A and FEC-0 fc2 gradients against their stop-gradient graphs
    and of FEC-TOP against the last fc2 gradient; cosine of the FEC-A and FEC-0 fc2 gradients with backprop."""
    import math
    from .single import fact_tokens
    FL = C.engine()
    model = base.model
    n_blocks = len(model.layers)
    worst = {"delta": 0.0, "FEC-A": 0.0, "FEC-0": 0.0, "FEC-TOP": 0.0}
    cos = {"FEC-A": [[] for _ in range(n_blocks)], "FEC-0": [[] for _ in range(n_blocks)]}

    def grads(fields, keys):
        return [fields[b].double().t() @ keys[b].double() for b in range(n_blocks)]

    def rel(a, b):
        num = math.sqrt(sum(float((x - y).pow(2).sum()) for x, y in zip(a, b)))
        return num / max(math.sqrt(sum(float(y.pow(2).sum()) for y in b)), 1e-300)

    for item in items:
        tok, pos = fact_tokens(base, item)
        ref = FL.backprop_fields(model, tok, pos)
        with torch.no_grad():
            with FL.ForwardTape(model, True) as tape:
                model(tok[:-1].unsqueeze(0), return_attention_weights=True)
            delta = FL.output_error(model, tape.x_final[0][pos], tok[1:][pos]) / len(pos)
            fld = torch.zeros_like(tape.x_final[0])
            fld[pos] = delta
            est = {r: FL.error_fields(model, tape, fld, r) for r in ("FEC-A", "FEC-0")}
        g = ref["x_final_grad"][pos]
        worst["delta"] = max(worst["delta"], float((delta - g).abs().max() / g.abs().max()))
        exact = grads(ref["fields"], ref["keys"])
        for rule, cut in (("FEC-A", True), ("FEC-0", False)):
            with FL.stop_gradient_graph(model, value=cut):
                sg = FL.backprop_fields(model, tok, pos)
            mine = grads(est[rule], ref["keys"])
            worst[rule] = max(worst[rule], rel(mine, grads(sg["fields"], ref["keys"])))
            for b in range(n_blocks):
                cos[rule][b].append(float((mine[b] * exact[b]).sum() / (mine[b].norm() * exact[b].norm())))
        worst["FEC-TOP"] = max(worst["FEC-TOP"], rel(grads(est["FEC-0"], ref["keys"])[-1:], exact[-1:]))
    return {"n": len(items), "max_relative_error": worst,
            "cos_fc2_gradient_with_backprop_per_block": {r: [round(sum(v) / len(v), 3) for v in rows]
                                                         for r, rows in cos.items()}}


# ------------------------------------------------------------------------------------------------------ summary
def summarise(records: List[dict]) -> dict:
    """Descriptive per-arm means of one world's records (no test)."""
    from .analysis import single_table, stream_table
    out = {"single": single_table(records), "stream": stream_table(records)}
    gates = [r for r in records if r["kind"] == "gate"]
    if gates:
        out["gate"] = [{k: r.get(k) for k in ("arm", "setting", "retention", "recall_other", "leak_hard",
                                               "leak_generic", "dmg_A", "dbpb")} for r in gates]
    caps = [r for r in records if r["kind"] == "capacity"]
    if caps:
        out["capacity"] = [{"arm": r["arm"], "keys_end": r["keys_end"], "nu_mean_end": r["nu_mean_end"],
                            "predicted_end": r["predicted_end"], "predicted_v2_end": r["predicted_v2_end"],
                            "knees": r["knees"], "knees_v2": r["knees_v2"], "d_ww": r["d_ww"],
                            "shares": r["shares"]} for r in caps]
    rets = [r for r in records if r["kind"] == "retract"]
    if rets:
        out["retract"] = [{k: r.get(k) for k in ("index", "p_before", "p_retract", "p_replay",
                                                  "kl_replay_retract", "others_retract", "others_replay")}
                          for r in rets]
    return out


def pilot_comparison(summary: dict) -> dict:
    """Smoke on discovery world 0 next to the v1 pilot values (same items, grids and install criterion)."""
    out: Dict[str, dict] = {"single": {}, "stream30": {}}
    for arm, ref in PILOT_W0["single"].items():
        row = summary["single"].get(f"{arm}|greedy")
        if row:
            out["single"][arm] = {k: {"pilot": v, "this_run": row.get(k)} for k, v in ref.items()}
    for arm, ref in PILOT_W0["stream30"].items():
        row = summary["stream"].get(f"{arm}|30")
        if row:
            out["stream30"][arm] = {k: {"pilot": v, "this_run": row.get(k)} for k, v in ref.items()}
    return out
