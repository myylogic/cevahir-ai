# -*- coding: utf-8 -*-
"""Part A: one fact written through five mid-stack addresses, each edit applied alone (PROTOCOL §3; KA-H1, S2).

For every TEST edit and step the frozen CEV-11 STD write at block ``GATE_LAYER`` gives A-TOK (``rome_update`` on
the prefix-averaged native key) and the shared residual ``delta``; the radius gates A-TOKG / A-RES / A-CTX / A-SUP
install the same ``delta`` inside cosine balls of their own code at matched c0 false-match rates. Reference arms on
the same edits: FT-L at the gate block, FT-L and FT-ALL at CEV-11's selected configurations.

Per (edit, address, configuration) one record with:
``E`` (the lived frame produces the new object), ``PE6``, FUSED / IDENT informal transfer, context transfer, and
the leak families RN, RD, C (relations B and C of the edited entity), PX — all with CEV-11's batched exact check
against the base model's own answers — plus, for the gates, the pre-edit in-ball indicators of every FUSED, RN and
RD probe.

In-ball indicator. The batched exact check scores a probe with one teacher-forced pass over
``[BOS] + prefix + object tokens`` (the final '.' is never an input), and an installed gate is live at every
position of that pass, the object positions included. The indicator is therefore computed, in the base state and
before any write, on exactly that sequence (:func:`scored_ids`): ``in_<family>`` holds one row
``[fires anywhere in the scored sequence, outcome, fires inside the prefix]`` per probe. The first flag is where
the write lands in the scored pass; the third separates fires at the name or another prefix position from fires
that happen only at object positions.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import KA_VERSION  # noqa: F401
from .codes import capture
from .gates import (ADDRESSES, CAP_MULT, GATE_ADDRESSES, GATE_CODE, GATE_LAYER, QS, STEPS, GateMemory, code_specs,
                    codes_from_tap, gate_fire)
from .streams import base_answers
from cev11.morph_tr import CASES, INFLECTED
from cev11.world import frame_prefix
from tokenizer_management.byte_bpe import BOS_ID

N_TEST = 16
FT_REFS = ("FT-L@gate", "FT-L@cev11", "FT-ALL@cev11")
LEAK_FAMILIES = ("RN", "RD", "C", "PX")
IN_BALL_FAMILIES = ("FUSED", "RN", "RD")     # families whose per-probe in-ball rows are recorded for the gates


def eligible_entities(base, editable: Sequence[str]) -> List[str]:
    """Editable probe entities with at least one FUSED informal case and one root neighbour in the probe pool."""
    from cev11.tokenizers import classify_informal, last_root_token
    tok, world = base.tok, base.world
    probe = sorted(n for n in world.members("probe"))
    lrt = {n: last_root_token(tok, n) for n in probe}
    count: Dict[int, int] = {}
    for n in probe:
        count[lrt[n]] = count.get(lrt[n], 0) + 1
    ok = set(editable)
    return [n for n in probe if n in ok and count[lrt[n]] > 1
            and any(classify_informal(tok, n, c) == "FUSED" for c in INFLECTED)]


def test_items(base, editable: Sequence[str], world_seed: int, n: int = N_TEST) -> List[dict]:
    """TEST edits (seed w+21): ``n`` eligible entities, relation A, lived cases rotating over the six inflected
    cases; the lived case always leaves at least one FUSED case to test."""
    from cev11.tokenizers import classify_informal
    rnd = random.Random(world_seed + 21)
    pool = eligible_entities(base, editable)
    if len(pool) < n:
        n = len(pool)
    chosen = rnd.sample(pool, n)
    items = []
    for i, name in enumerate(chosen):
        fused = [c for c in INFLECTED if classify_informal(base.tok, name, c) == "FUSED"]
        order = INFLECTED[i % len(INFLECTED):] + INFLECTED[:i % len(INFLECTED)]
        lived = next((c for c in order if any(f != c for f in fused)), order[0])
        old = base.world.facts[(name, "A")]
        new = rnd.choice([o for o in base.world.objects["A"] if o != old])
        items.append({"edit_id": f"w{world_seed}-a{i:02d}", "name": name, "rel": "A", "lived_case": lived,
                      "old": old, "new": new})
    return items


def build_probes(base, item: dict, seed: int) -> dict:
    """Probe families of one edit (BASE state only): positives expect the new object, leak probes the base
    model's own answer. Returns ``{"probes": [...], "dropped": n}``; each probe has ``family, prefix, gold``."""
    from cev11.probes import neighbour_sets
    from cev11.tokenizers import classify_informal, last_root_token
    tok, world = base.tok, base.world
    n, rel, lc = item["name"], item["rel"], item["lived_case"]
    rnd = random.Random(f"{seed}|{item['edit_id']}")
    neigh = neighbour_sets(world, tok, item, rnd, last_root_token)
    probes: List[dict] = []

    def add(family: str, name: str, r: str, case: str, orth: str = "formal", ctx: str = "none", **extra):
        probes.append({"family": family, "name": name, "prefix": frame_prefix(name, r, case, orth, ctx),
                       "gold": item["new"], **extra})

    add("WF", n, rel, lc)
    for c in CASES:
        if c != lc:
            add("PF6", n, rel, c)
    for c in INFLECTED:
        if c != lc:
            cls = classify_informal(tok, n, c)
            add("FUSED" if cls == "FUSED" else "IDENT", n, rel, c, "informal")
    for x in ("P1", "P2"):
        add("CTX", n, rel, lc, "formal", x)
    leak: List[dict] = []
    for fam, names, r_list in (("RN", neigh["RN"], [rel]), ("RD", neigh["RD"], [rel]), ("PX", neigh["PX"], [rel]),
                               ("C", [n], ["B", "C"])):
        for m in names:
            for r in r_list:
                for c in CASES:
                    leak.append({"family": fam, "name": m, "prefix": frame_prefix(m, r, c),
                                 "gold": world.facts[(m, r)]})
    ans = base_answers(base, [d["prefix"] for d in leak], [d["gold"] for d in leak])
    dropped = 0
    for d, a in zip(leak, ans):
        if a is None:
            dropped += 1
            continue
        d["gold"] = a
        probes.append(d)
    return {"probes": probes, "dropped": dropped, "neigh": {k: v for k, v in neigh.items() if k != "RN_cluster_size"}}


def score_probes(base, probes: Sequence[dict]) -> List[bool]:
    from cev11.decode import exact_check_batch
    return exact_check_batch(base.model, base.tok, [p["prefix"] for p in probes], [p["gold"] for p in probes])


def scored_ids(tok, probe: dict) -> Tuple[List[int], int]:
    """Input ids of the pass that scores ``probe`` and the length of their prefix part.

    CEV-11's batched exact check runs ``[BOS] + encode(prefix) + encode(" " + gold + ".")[:-1]`` (teacher forcing
    on the expected object; the final '.' is predicted, never fed). A gate installed during scoring reads its
    code at every one of these positions."""
    pre = [BOS_ID] + list(tok.encode(probe["prefix"]))
    return pre + list(tok.encode(" " + probe["gold"] + "."))[:-1], len(pre)


def summarise(probes: Sequence[dict], ok: Sequence[bool]) -> dict:
    """Counts per family: positives ``ok/n``, leak families ``changed/n``."""
    out: dict = {}
    for fam in ("WF", "PF6", "FUSED", "IDENT", "CTX"):
        v = [bool(k) for p, k in zip(probes, ok) if p["family"] == fam]
        out[fam] = {"ok": int(sum(v)), "n": len(v)}
    for fam in LEAK_FAMILIES:
        v = [not bool(k) for p, k in zip(probes, ok) if p["family"] == fam]
        out[fam] = {"changed": int(sum(v)), "n": len(v)}
    return out


def cell_metrics(rec: dict) -> dict:
    """Per-edit values entering the frontier: E, Y (FUSED transfer), X = Leak_RN - Leak_RD, R = Leak_RD, C, PE6.
    ITT: an edit without probes of a family contributes 0 for that family."""
    share = lambda d, k: (d[k] / d["n"]) if d["n"] else 0.0  # noqa: E731
    rn, rd = share(rec["RN"], "changed"), share(rec["RD"], "changed")
    return {"E": share(rec["WF"], "ok"), "Y": share(rec["FUSED"], "ok"), "X": rn - rd, "R": rd, "RN": rn,
            "C": share(rec["C"], "changed"), "PE6": share(rec["PF6"], "ok"), "IDENT": share(rec["IDENT"], "ok"),
            "PX": share(rec["PX"], "changed"), "CTX": share(rec["CTX"], "ok")}


def config_id(address: str, step: Optional[float], q: Optional[float]) -> str:
    if address in GATE_ADDRESSES:
        return f"{address}|step={step:g}|q={q:g}"
    if address == "A-TOK":
        return f"A-TOK|step={step:g}"
    return address


def expected_configs(addresses: Sequence[str] = ADDRESSES, steps: Sequence[float] = STEPS,
                     qs: Sequence[float] = QS, refs: Sequence[str] = FT_REFS) -> List[str]:
    out = []
    for a in addresses:
        for s in steps:
            if a == "A-TOK":
                out.append(config_id(a, s, None))
            else:
                out += [config_id(a, s, q) for q in qs]
    return out + list(refs)


class PartA:
    """Per-world context of Part A: gate memories, thresholds, the SUP projection and the CEV-11 installer."""

    def __init__(self, base, world_seed: int, thresholds: Dict[str, Dict[float, float]],
                 sup_proj: Optional[Tuple[torch.Tensor, torch.Tensor]], ft_cfg: Dict[str, dict],
                 layer: int = GATE_LAYER, steps: Sequence[float] = STEPS, qs: Sequence[float] = QS,
                 addresses: Sequence[str] = ADDRESSES, refs: Sequence[str] = FT_REFS, max_steps: int = 120,
                 greedy_n: int = 8):
        from cev11 import experiment as E
        self.E = E
        self.greedy_n = int(greedy_n)
        self.base, self.world_seed, self.layer = base, int(world_seed), int(layer)
        self.thresholds, self.sup_proj, self.ft_cfg = thresholds, sup_proj, ft_cfg
        self.steps, self.qs = tuple(steps), tuple(qs)
        self.addresses, self.refs, self.max_steps = tuple(addresses), tuple(refs), int(max_steps)
        self.inst = E.Installer(base)
        self.gates: Dict[str, GateMemory] = {}
        for a in self.addresses:
            if a in GATE_ADDRESSES:
                code = GATE_CODE[a]
                self.gates[a] = GateMemory(base.model, self.layer, code, sup_proj if code == "sup" else None)

    def close(self) -> None:
        for g in self.gates.values():
            g.remove()
        self.gates = {}
        self.inst.restore()
        for m in self.inst.mems.values():
            m.remove()
        self.inst.mems = {}

    def _in_ball(self, probes: Sequence[dict], key_ids: Sequence[int], key_pos: int) -> Tuple[dict, dict]:
        """Unit keys per gate address and, per (address, q), one ``(fires in the scored sequence, fires in the
        prefix)`` pair per probe. Call in the BASE state: the codes are read below the write, so the pairs say
        where an installed gate will fire in the pass that scores the probe."""
        tok = self.base.tok
        codes = sorted({GATE_CODE[a] for a in self.gates})
        specs = code_specs(codes, self.layer)
        proj = None if self.sup_proj is None else (self.sup_proj[0].float(), self.sup_proj[1].float())
        scored = [scored_ids(tok, p) for p in probes]
        seqs = [list(key_ids)] + [ids for ids, _ in scored]
        recs = capture(self.base.model, seqs, specs, dtype=torch.float32)
        keys, fire, self._lived_fire = {}, {}, {}
        for a in self.gates:
            code = GATE_CODE[a]
            u = [codes_from_tap(r, code, self.layer, proj) for r in recs]
            key = u[0][key_pos]
            keys[a] = key / key.norm().clamp_min(1e-12)
            for q in self.qs:
                th = self.thresholds[code][q]
                masks = [gate_fire(v, keys[a], th) for v in u[1:]]
                fire[(a, q)] = [(bool(m.any()), bool(m[:n_pre].any())) for m, (_, n_pre) in zip(masks, scored)]
                self._lived_fire[(a, q)] = int(gate_fire(u[0], keys[a], th).sum())
        return keys, fire

    def run_edit(self, item: dict) -> List[dict]:
        """All configurations of one edit; the model is left in the base state. Returns the records."""
        from cev11.keysites import site_prompts
        E, base = self.E, self.base
        prep = build_probes(base, item, self.world_seed + 22)
        probes = prep["probes"]
        lived = frame_prefix(item["name"], item["rel"], item["lived_case"], "formal", "none")
        key_ids, key_pos = site_prompts(base.tok, lived, item["name"], "none", site="STD")[0]
        keys, fire = self._in_ball(probes, key_ids, key_pos) if self.gates else ({}, {})
        fam = [p["family"] for p in probes]
        head = {"world": self.world_seed, "edit_id": item["edit_id"], "name": item["name"],
                "lived_case": item["lived_case"], "new": item["new"], "old": item["old"],
                "dropped_leak_probes": prep["dropped"], "layer": self.layer}
        out: List[dict] = []

        def record(address, step, q, extra):
            ok = score_probes(base, probes)
            rec = {**head, "address": address, "step": step, "q": q, "config": config_id(address, step, q),
                   **summarise(probes, ok), **extra}
            if self.greedy_n and address == "A-CTX" and step == self.steps[len(self.steps) // 2] \
                    and q == self.qs[len(self.qs) // 2]:
                # the batched exact check against true greedy decoding (KV cache on) under an installed gate
                from cev11.decode import greedy_reproduces
                pick = random.Random(f"{self.world_seed}|{item['edit_id']}|greedy").sample(
                    range(len(probes)), min(self.greedy_n, len(probes)))
                bad = sum(int(greedy_reproduces(base.model, base.tok, probes[i]["prefix"], probes[i]["gold"])
                              != bool(ok[i])) for i in pick)
                rec["greedy_check"] = {"n": len(pick), "disagree": int(bad)}
            if address in GATE_ADDRESSES:
                f = fire[(address, q)]
                for famname in IN_BALL_FAMILIES:
                    # [in the ball on the scored sequence, outcome, in the ball on the prefix]
                    rec[f"in_{famname}"] = [[int(a_), int(bool(k) if famname == "FUSED" else not bool(k)), int(p_)]
                                            for (a_, p_), k, ff in zip(f, ok, fam) if ff == famname]
                rec["theta"] = self.thresholds[GATE_CODE[address]][q]
                rec["lived_fire_positions"] = self._lived_fire[(address, q)]      # positions of the lived prompt
            out.append(rec)

        try:
            for step in self.steps:
                w = E.compute_write(base, item, "STD", self.layer,
                                    {"step": step, "cap_mult": CAP_MULT, "max_steps": self.max_steps},
                                    self.world_seed, None)
                diag = {"delta_norm": float(w["diag"]["delta_norm"]), "nll": float(w["diag"]["nll"]),
                        "search_steps": int(w["diag"]["steps"]), "cap": float(w["diag"]["cap"])}
                if "A-TOK" in self.addresses:
                    self.inst.set_M(self.layer, w["M"])
                    record("A-TOK", step, None, diag)
                    self.inst.zero_all()
                for a, gm in self.gates.items():
                    for q in self.qs:
                        gm.set([(keys[a], self.thresholds[GATE_CODE[a]][q], w["delta"])])
                        record(a, step, q, diag)
                        gm.clear()
            for ref in self.refs:
                arm, where = ref.split("@")
                cfg = self.ft_cfg[arm]
                layer = self.layer if where == "gate" else cfg["layer"]
                w = E.compute_write(base, item, arm, layer, cfg["hp"], self.world_seed)
                if arm == "FT-ALL":
                    self.inst.load_state(w["state"])
                else:
                    self.inst.set_M(layer, w["M"])
                record(ref, None, None, {"ft_steps": int(w["diag"].get("steps", 0)), "ft_layer": layer})
                self.inst.restore()
        finally:
            for gm in self.gates.values():
                gm.clear()
            self.inst.restore()
        return out


__all__ = ["N_TEST", "FT_REFS", "LEAK_FAMILIES", "IN_BALL_FAMILIES", "eligible_entities", "test_items",
           "build_probes", "score_probes", "scored_ids", "summarise", "cell_metrics", "config_id",
           "expected_configs", "PartA"]
