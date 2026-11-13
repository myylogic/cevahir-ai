# -*- coding: utf-8 -*-
"""Discovery selection and the preregistered confirmatory analysis of the KA study (PROTOCOL §6-§8).

* :func:`select` — operating points from the discovery worlds only: the prediction-position layer l*, the frozen
  frontier mixtures of every Part A address, the G-KA1 screens (which hypotheses enter the confirmatory families),
  and the gates that block the freeze when they fail (G-KA1c, the consistency checks).
* :func:`decide` — per-world effects, exact sign-flip tests, Holm within each family, labels and guards, and the
  consistency checks whose failure replaces every label by "audit required".

Everything is a pure function of the per-world record files (``ka.world_run`` layout) and of the selection. A
world is read only through its DONE.json: files it does not hash are refused, phases it does not list are not
loaded, and both entry points refuse a world that lacks a phase the protocol requires of it.
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import KA_VERSION
from . import stats as S
from .ft import FT_ARMS
from .gates import ADDRESSES, GATE_ADDRESSES, H1_BASELINES
from .part_a import FT_REFS, IN_BALL_FAMILIES, cell_metrics, expected_configs
from .world_run import RECORD_FILES, read_jsonl, required_phases, sha256_file, sha256_json

SESOI = {"KA-H1": 0.15, "KA-H2": 0.10, "S1a": 0.10, "S1b": 0.10}
ECE_MAX = 0.05
GROWTH_MAX = 0.01
FIXED_GROWTH_MIN = 0.03
AUC_MIN = 0.85
CORR_MIN = 0.80
MATCH_TRANSFER_MIN = 0.5
MIN_TEST_EDITS = 12
MIN_FUSED_FACTS = 15
SCREEN_H1_GAP = 0.10
SCREEN_ATT_LOGLOSS_RATIO = 2.0
SCREEN_FT_NEWEST = 0.9        # G-KA1c: lived-form efficacy on the newest fact, per FT arm, discovery world 0
CONSISTENCY_MAX = 0.01        # largest pooled disagreement share between two scorers of the same probes
N_CONFIRM_WORLDS = 8          # confirmatory units (worlds 1000-1007); used by the guard diagnostic at selection
LAYER_FALLBACK_GAP = 0.02
GUARD = {"h1_efficacy": 0.05, "h1_pe6": 0.05, "h1_x": 0.03, "h1_r": 0.005, "h2_structured": 0.02,
         "h2_leak_u": 0.01, "h2_dbpb": 0.01}
PRIMARY = ("KA-H1", "KA-H2", "KA-H3")
SECONDARY = ("S1a", "S1b", "S2")
AUDIT = "audit required (consistency check failed)"
LOCALITY_COST = "not supported (locality cost)"
LOCALITY_OPEN = "not supported (locality not established)"


# ---------------------------------------------------------------------------------------------------- loading
def load_world(path: pathlib.Path) -> dict:
    """Records of one finished world, read through its DONE.json only.

    Refused (ValueError): another record version, a listed file that is missing or changed, a record file that
    DONE.json does not hash, a listed phase without its file, a required phase that is not listed, and a
    DONE.json that disagrees with setup.json on the protocol configuration. Phases that DONE.json does not list
    are not loaded (their record lists are empty)."""
    path = pathlib.Path(path)
    done = json.loads((path / "DONE.json").read_text(encoding="utf-8"))
    if done.get("ka_version") != KA_VERSION:
        raise ValueError(f"{path}: records of version {done.get('ka_version')!r}; this code reads {KA_VERSION!r}")
    for key in ("phases", "required_phases", "files"):
        if key not in done:
            raise ValueError(f"{path / 'DONE.json'} has no {key!r}")
    if "setup.json" not in done["files"]:
        raise ValueError(f"{path / 'DONE.json'} does not hash setup.json")
    for name, digest in done["files"].items():
        if not (path / name).exists():
            raise ValueError(f"{path / name} is listed in DONE.json but missing")
        if sha256_file(path / name) != digest:
            raise ValueError(f"{path / name} does not match the hash recorded in DONE.json")
    for phase, name in RECORD_FILES.items():
        if name not in done["files"]:
            if (path / name).exists():
                raise ValueError(f"{path / name} is not covered by DONE.json")
            if phase in done["phases"]:
                raise ValueError(f"{path}: DONE.json lists phase {phase!r} without its record file")
    missing = [p for p in done["required_phases"] if p not in done["phases"]]
    if missing:
        raise ValueError(f"{path}: required phases {missing} are not complete")
    setup = json.loads((path / "setup.json").read_text(encoding="utf-8"))
    if bool(done.get("protocol_config")) != bool(setup["config"]["protocol"]):
        raise ValueError(f"{path}: DONE.json and setup.json disagree on the protocol configuration")
    has = set(done["phases"])
    out = {"dir": str(path), "done": done, "setup": setup}
    out["part_a"] = read_jsonl(path / RECORD_FILES["parta"]) if "parta" in has else []
    out["episodic"] = read_jsonl(path / RECORD_FILES["episodic"]) if "episodic" in has else []
    out["ft"] = read_jsonl(path / RECORD_FILES["ft"]) if "ft" in has else []
    out["checks"] = json.loads((path / RECORD_FILES["checks"]).read_text(encoding="utf-8")) if "checks" in has else {}
    return out


def load_worlds(results_dir: pathlib.Path, worlds: Optional[Sequence[int]] = None) -> Dict[int, dict]:
    """Finished worlds (DONE.json present) under ``results_dir``; :func:`world_status` accounts for the others."""
    results_dir = pathlib.Path(results_dir)
    out = {}
    for d in sorted(results_dir.glob("w*")):
        if not (d / "DONE.json").exists() or (d / "EXCLUDED.json").exists():
            continue
        w = int(d.name[1:])
        if worlds is None or w in worlds:
            out[w] = load_world(d)
    return out


def world_status(results_dir: pathlib.Path, worlds: Sequence[int]) -> dict:
    """Account for every expected world: ``complete`` (DONE.json), ``excluded`` (EXCLUDED.json: the base is not a
    valid unit; the reason is kept), ``incomplete`` (a directory with neither, or with both) or ``absent``."""
    results_dir = pathlib.Path(results_dir)
    out: dict = {"complete": [], "excluded": {}, "incomplete": [], "absent": []}
    for w in sorted(int(x) for x in worlds):
        d = results_dir / f"w{w}"
        done, excl = (d / "DONE.json").exists(), (d / "EXCLUDED.json").exists()
        if not d.is_dir():
            out["absent"].append(w)
        elif done and not excl:
            out["complete"].append(w)
        elif excl and not done:
            out["excluded"][str(w)] = json.loads((d / "EXCLUDED.json").read_text(encoding="utf-8")).get("reason")
        else:
            out["incomplete"].append(w)
    return out


def check_phases(world: Mapping, required: Sequence[str]) -> None:
    """ValueError unless the world's DONE.json lists every phase in ``required``."""
    missing = [p for p in required if p not in world["done"]["phases"]]
    if missing:
        raise ValueError(f"world {world['setup']['world']}: phases {missing} are required here but not complete "
                         f"(DONE.json lists {world['done']['phases']})")


# ---------------------------------------------------------------------------------------------------- Part A
def address_configs(cfg: Mapping) -> Dict[str, List[str]]:
    """Configuration ids per address, in the fixed order used by the frozen mixtures."""
    out = {a: expected_configs([a], cfg["steps"], cfg["qs"], []) for a in cfg["addresses"]}
    for r in cfg["ft_refs"]:
        out[r] = [r]
    return out


def part_a_cells(records: Sequence[dict]) -> Tuple[Dict[str, dict], int]:
    """Per-configuration means over the world's TEST edits (ITT) and the number of edits."""
    by_cfg: Dict[str, List[dict]] = {}
    edits = set()
    for r in records:
        by_cfg.setdefault(r["config"], []).append(cell_metrics(r))
        edits.add(r["edit_id"])
    n = len(edits)
    cells = {}
    for c, rows in by_cfg.items():
        cells[c] = {k: sum(row[k] for row in rows) / n for k in rows[0]} if n else {}
    return cells, n


def _pooled_records(worlds: Sequence[dict]) -> List[dict]:
    return [dict(r, edit_id=f"{w['setup']['world']}|{r['edit_id']}") for w in worlds for r in w["part_a"]]


def pooled_cells(worlds: Sequence[dict]) -> Dict[str, dict]:
    """Cell means over the pooled TEST edits of several (discovery) worlds."""
    return part_a_cells(_pooled_records(worlds))[0]


def edit_cells(records: Sequence[dict]) -> Dict[str, Dict[str, dict]]:
    """Per-edit cell metrics: edit id -> configuration -> metrics."""
    out: Dict[str, Dict[str, dict]] = {}
    for r in records:
        out.setdefault(r["edit_id"], {})[r["config"]] = cell_metrics(r)
    return out


def edit_mixture(by_edit: Mapping[str, Mapping[str, dict]], configs: Sequence[str], w: Sequence[float],
                 key: str) -> List[float]:
    """Value of ``key`` at the mixture ``w`` over ``configs`` for every edit (id order). Its mean over the edits
    is the mixture value of the cell means; ITT: a missing cell counts 0."""
    if len(configs) != len(w):
        raise ValueError("mixture and configurations differ in length")
    return [sum(float(wk) * float(by_edit[e].get(c, {}).get(key, 0.0)) for wk, c in zip(w, configs))
            for e in sorted(by_edit)]


def locality_guard_outlook(records: Sequence[dict], configs: Sequence[str], w: Sequence[float], n_test: int,
                           n_worlds: int = N_CONFIRM_WORLDS) -> dict:
    """Design diagnostic of the KA-H1 locality guard for one frozen mixture, from discovery edits only.

    The guard asks that the upper 95% bound of the across-world mean of X (and of R) stay below the frontier
    constraint plus a margin. For each of X and R this returns the pooled discovery mean at the mixture, the
    per-edit standard deviation, the across-world standard deviation that edit sampling alone would give
    (``sd_edit / sqrt(n_test)``; real worlds also differ from each other, so this is a lower bound) and the
    probability that the guard would pass on ``n_worlds`` such worlds if the true mean equalled the discovery
    mean. ``p_pass`` is the smaller of the two: an optimistic estimate. A low value says the guard would
    reject a mixture that keeps to its leak budget; it is information for the freeze decision, not a test.
    """
    by_edit = edit_cells(records)
    out: dict = {"n_edits": len(by_edit), "n_worlds": int(n_worlds), "n_test": int(n_test)}
    if len(by_edit) < 2 or not any(v > 0 for v in w):
        return {**out, "p_pass": None}
    ps = []
    for key, limit in (("X", S.X_MAX + GUARD["h1_x"]), ("R", S.R_MAX + GUARD["h1_r"])):
        vals = edit_mixture(by_edit, configs, w, key)
        mean = sum(vals) / len(vals)
        sd_edit = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
        sd_world = sd_edit / math.sqrt(n_test)
        p = S.upper_bound_pass_probability(mean, sd_world, n_worlds, limit)
        out[key] = {"mean": mean, "sd_edit": sd_edit, "sd_world_lower": sd_world, "limit": limit, "p_pass": p}
        ps.append(p)
    return {**out, "p_pass": min(ps)}


def _cells_of(cells: Mapping[str, dict], configs: Sequence[str]) -> List[dict]:
    return [cells[c] for c in configs]


def part_a_summary(world: dict, mixtures: Optional[Mapping[str, Sequence[float]]] = None) -> dict:
    """Per-world Part A statistics: in-sample F*, F*_CEV, frozen-mixture values, best efficacy, S2 pieces."""
    cfg = world["setup"]["config"]
    cells, n = part_a_cells(world["part_a"])
    confs = address_configs(cfg)
    out: dict = {"n_edits": n, "valid": n >= min(MIN_TEST_EDITS, cfg["n_test"]), "addresses": {}}
    if not n:
        out["valid"] = False
        return out
    by_edit = edit_cells(world["part_a"])
    out["frozen_Y_by_edit"] = {}
    for a, cl in confs.items():
        rows = _cells_of(cells, cl)
        f, w = S.frontier(rows)
        fc, _ = S.frontier(rows, c_max=S.C_MAX)
        entry = {"F_star": f, "F_star_w": w, "F_star_CEV": fc, "best_E": max(r["E"] for r in rows),
                 "best_Y": max(r["Y"] for r in rows)}
        if mixtures is not None and a in mixtures:
            mw = list(mixtures[a])
            entry["frozen"] = {k: S.mixture_value(rows, mw, k) for k in ("Y", "X", "R", "E", "C", "PE6")}
            out["frozen_Y_by_edit"][a] = edit_mixture(by_edit, cl, mw, "Y")
        out["addresses"][a] = entry
    # S2(ii), S2(iii): the pre-edit in-ball indicator vs observed transfer and leak over the radius-gate cells.
    # A row is [in the ball on the scored sequence, outcome, in the ball on the prefix]; S2 uses the first flag
    # (where the write lands in the pass that scores the probe). The same statistics with the prefix flag and the
    # number of probes that fire only at object positions are reported next to it.
    cols = {"scored": 0, "prefix": 2}
    pred_t: Dict[str, list] = {k: [] for k in cols}
    pred_l: Dict[str, list] = {k: [] for k in cols}
    hit = {k: 0 for k in cols}
    tot = {k: 0 for k in cols}
    obs_t, obs_l = [], []
    only = {fam: 0 for fam in IN_BALL_FAMILIES}
    n_rows = 0
    by_cfg: Dict[str, List[dict]] = {}
    for r in world["part_a"]:
        if r["address"] in GATE_ADDRESSES:
            by_cfg.setdefault(r["config"], []).append(r)
    for c, rows in sorted(by_cfg.items()):
        def share(key, col):
            return sum((sum(x[col] for x in r[key]) / len(r[key])) if r[key] else 0.0 for r in rows) / n

        for name, col in cols.items():
            pred_t[name].append(share("in_FUSED", col))
            pred_l[name].append(share("in_RN", col))
        obs_t.append(cells[c]["Y"])
        obs_l.append(cells[c]["RN"])
        for r in rows:
            for fam in IN_BALL_FAMILIES:
                only[fam] += sum(1 for x in r[f"in_{fam}"] if x[0] and not x[2])
                n_rows += len(r[f"in_{fam}"])
            if r["address"] == "A-CTX":
                for name, col in cols.items():
                    hit[name] += sum(x[1] for x in r["in_FUSED"] if x[col])
                    tot[name] += sum(1 for x in r["in_FUSED"] if x[col])

    def s2(name):
        return {"r_transfer": S.pearson(pred_t[name], obs_t), "r_leak": S.pearson(pred_l[name], obs_l),
                "transfer_given_match": (hit[name] / tot[name]) if tot[name] else None, "n_matched": tot[name]}

    out["S2"] = {**s2("scored"), "n_cells": len(by_cfg), "prefix_indicator": s2("prefix"),
                 "object_only": {**only, "rows": n_rows}}
    g = [r["greedy_check"] for r in world["part_a"] if "greedy_check" in r]
    out["greedy_check"] = {"n": sum(x["n"] for x in g), "disagree": sum(x["disagree"] for x in g)}
    return out


# ---------------------------------------------------------------------------------------------------- consistency
def consistency(worlds: Mapping[int, dict], pa: Mapping[int, dict], st: Mapping[int, dict]) -> dict:
    """The consistency checks of PROTOCOL §6, pooled over the given worlds.

    Three pairs of scorers of the same probes must agree: the cached evaluator and the hooked float32 model, the
    hooked model and true greedy decoding (both on the 5% sample of the stream battery), and the batched exact
    check and true greedy decoding under an installed gate (Part A, 8 probes per edit). Each pooled disagreement
    share must be <= ``CONSISTENCY_MAX``, a stream comparison without any sample fails, the Part A comparison is
    skipped when Part A was not run, and the base model hash must be unchanged after the checks on every world.
    """
    pairs = {"cached_vs_hooked": [0, 0], "hooked_vs_greedy": [0, 0], "batched_vs_greedy_part_a": [0, 0]}
    per_world: dict = {}
    unchanged = True
    for w in sorted(worlds):
        chk = st[w].get("checks") or {}
        g = pa[w].get("greedy_check") or {}
        n = int(chk.get("n_sample") or 0)
        row = {"n_sample": n, "cached_vs_hooked": chk.get("cached_vs_hooked_disagree"),
               "hooked_vs_greedy": chk.get("hooked_vs_greedy_disagree"), "part_a_n": int(g.get("n") or 0),
               "batched_vs_greedy_part_a": g.get("disagree"),
               "model_unchanged": bool(worlds[w].get("checks", {}).get("model_unchanged"))}
        per_world[str(w)] = row
        unchanged = unchanged and row["model_unchanged"]
        for k in ("cached_vs_hooked", "hooked_vs_greedy"):
            if n and row[k] is not None:
                pairs[k][0] += int(row[k])
                pairs[k][1] += n
        if row["part_a_n"]:
            pairs["batched_vs_greedy_part_a"][0] += int(row["batched_vs_greedy_part_a"] or 0)
            pairs["batched_vs_greedy_part_a"][1] += row["part_a_n"]
    pooled = {k: {"disagree": bad, "n": n, "share": (bad / n) if n else None} for k, (bad, n) in pairs.items()}
    failures = [k for k, v in pooled.items()
                if (v["share"] is None and k != "batched_vs_greedy_part_a")
                or (v["share"] is not None and v["share"] > CONSISTENCY_MAX)]
    if not unchanged:
        failures.append("model_unchanged")
    return {"max_share": CONSISTENCY_MAX, "pooled": pooled, "per_world": per_world, "model_unchanged": unchanged,
            "failures": failures, "pass": not failures}


# ---------------------------------------------------------------------------------------------------- episodic
def epi(world: dict, address: str, read: str, n: int) -> Optional[dict]:
    for r in world["episodic"]:
        if r["address"] == address and r["read"] == read and r["N"] == n:
            return r
    return None


def ft_rec(world: dict, arm: str, n: int) -> Optional[dict]:
    for r in world["ft"]:
        if r["arm"] == arm and r["N"] == n:
            return r
    return None


def stream_summary(world: dict, pred_layer: int) -> dict:
    """Per-world values of KA-H2, KA-H3, S1a, S1b and S2(i) at the frozen layer."""
    cfg = world["setup"]["config"]
    res, att = f"PRED-RES@{pred_layer}", f"PRED-ATT@{pred_layer}"
    n1 = cfg["n_primary"]
    lo, hi = cfg["checkpoints"][0], cfg["checkpoints"][-1]
    out: dict = {}
    mem = epi(world, res, "rule", n1)
    if mem is None:
        return {"valid": False}
    out["valid"] = (mem["n_fused"] >= min(MIN_FUSED_FACTS, n1)) and mem["fused"] is not None
    out["mem"] = {k: mem.get(k) for k in ("fused", "lived", "para", "leadin", "Leak_U", "Leak_C", "Leak_RN", "PX",
                                          "OBJ", "OBJ1", "structured", "ece", "auc_leak", "auc_fused",
                                          "fused_first25", "n_fused", "cap_hit_share")}
    chk = world["checks"].get("addresses", {}).get(res, {})
    out["mem"]["dbpb"] = chk.get("dbpb")
    out["checks"] = {k: chk.get(k) for k in ("n_sample", "cached_vs_hooked_disagree", "hooked_vs_greedy_disagree")}
    fts = {}
    out["fused_facts"] = {"mem": mem.get("fused_facts")}
    for arm in FT_ARMS:
        r = ft_rec(world, arm, n1)
        if r is not None:
            fts[arm] = {k: r.get(k) for k in ("fused", "lived", "Leak_U", "structured", "dbpb", "newest_lived",
                                              "newest_efficacy", "fused_first25", "Leak_C", "Leak_RN")}
            out["fused_facts"][arm] = r.get("fused_facts")
    out["ft"] = fts
    usable = {a: v for a, v in fts.items() if v["fused"] is not None}
    if usable and out["valid"]:
        best = sorted(usable, key=lambda a: (-usable[a]["fused"], a))[0]
        out["ft_best"] = best
        out["d_H2"] = mem["fused"] - usable[best]["fused"]
    else:
        out["valid_H2"] = False
    a_lo, a_hi = epi(world, res, "rule", lo), epi(world, res, "rule", hi)
    f_lo, f_hi = epi(world, res, "fixed", lo), epi(world, res, "fixed", hi)
    if a_lo and a_hi and a_lo["Leak_U"] is not None and a_hi["Leak_U"] is not None:
        out["growth"] = a_hi["Leak_U"] - a_lo["Leak_U"]
    if f_lo and f_hi and f_lo["Leak_U"] is not None and f_hi["Leak_U"] is not None:
        out["growth_fixed"] = f_hi["Leak_U"] - f_lo["Leak_U"]
    out["ece"] = mem["ece"]
    aucs = [v for v in (mem["auc_leak"], mem["auc_fused"]) if v is not None]
    out["auc_min"] = min(aucs) if aucs else None
    fr = {r["address"]: r for r in world["episodic"] if r["read"] == "frontier"}
    if res in fr and att in fr:
        g_r, g_a = fr[res]["G_F@2"], fr[att]["G_F@2"]
        p_r, p_a = fr[res]["PX@2"], fr[att]["PX@2"]
        out["d_S1a"] = (g_a - g_r) if g_r is not None and g_a is not None else None
        out["d_S1b"] = (p_r - p_a) if p_r is not None and p_a is not None else None
    out["law"] = [{"N": r["N"], "any_over": r.get("U_any_over_theta"), "law": r.get("law_false_fire"),
                   "Leak_U": r.get("Leak_U"), "markov_obs": r.get("U_markov_obs"), "fused": r.get("fused")}
                  for r in world["episodic"] if r["address"] == res and r["read"] == "rule"]
    return out


# ---------------------------------------------------------------------------------------------------- selection
def select(discovery: Mapping[int, dict]) -> dict:
    """Operating points from the discovery worlds (never from confirmatory data). Returns the selection body."""
    worlds = [discovery[w] for w in sorted(discovery)]
    if not worlds:
        raise ValueError("no completed discovery world")
    for w in worlds:
        if w["setup"]["run"] != "discovery" or not w["setup"]["config"]["protocol"]:
            raise ValueError("selection needs discovery worlds run with the protocol configuration")
        check_phases(w, required_phases("discovery"))          # every phase: the selection reads all of them
    cfg = worlds[0]["setup"]["config"]
    layers = list(cfg["pred_layers"])
    n1 = cfg["n_primary"]
    # l*: minimum mean cross-fitted c0 calibration log-loss; fallback to the best discovery G_F at the rule when
    # the write-free choice is more than LAYER_FALLBACK_GAP below it
    loss = {l: sum(w["setup"]["addresses"][f"PRED-RES@{l}"]["logloss"] for w in worlds) / len(worlds) for l in layers}
    gf = {}
    for l in layers:
        vals = []
        for w in worlds:
            v = epi(w, f"PRED-RES@{l}", "rule", n1)
            if v is None:          # a missing record is not a G_F of zero: the fallback rule needs every layer
                raise ValueError(f"world {w['setup']['world']}: no rule record of PRED-RES@{l} at N = {n1}")
            vals.append(v["fused"] or 0.0)
        gf[l] = sum(vals) / len(worlds)
    by_loss = min(layers, key=lambda l: (loss[l], l))
    best_gf = max(layers, key=lambda l: (gf[l], -l))
    pred_layer = by_loss if gf[by_loss] >= gf[best_gf] - LAYER_FALLBACK_GAP else best_gf
    att_ratio = None
    if cfg.get("pred_att", True):
        la = sum(w["setup"]["addresses"][f"PRED-ATT@{pred_layer}"]["logloss"] for w in worlds) / len(worlds)
        att_ratio = la / loss[pred_layer] if loss[pred_layer] > 0 else math.inf
    pred_att = att_ratio is not None and att_ratio <= SCREEN_ATT_LOGLOSS_RATIO
    # frozen mixtures on the pooled discovery TEST edits
    pooled = pooled_cells(worlds)
    pooled_records = _pooled_records(worlds)
    confs = address_configs(cfg)
    mixtures, pooled_stats = {}, {}
    for a in cfg["addresses"]:
        rows = _cells_of(pooled, confs[a])
        f, w_ = S.frontier(rows)
        mixtures[a] = w_
        pooled_stats[a] = {"F_star": f, "X": S.mixture_value(rows, w_, "X"), "R": S.mixture_value(rows, w_, "R"),
                           "E": S.mixture_value(rows, w_, "E"), "feasible": any(v > 0 for v in w_),
                           "locality_guard": locality_guard_outlook(pooled_records, confs[a], w_, cfg["n_test"])}
    # G-KA1(a): in-sample gap per discovery world and transfer given a gate match
    per_world = {}
    pa_disc, st_disc = {}, {}
    for w in worlds:
        wid = int(w["setup"]["world"])
        sm = pa_disc[wid] = part_a_summary(w)
        gap = None
        if all(a in sm["addresses"] for a in ("A-CTX",) + H1_BASELINES):
            gap = min(sm["addresses"]["A-CTX"]["F_star"] - sm["addresses"][b]["F_star"] for b in H1_BASELINES)
        st = st_disc[wid] = stream_summary(w, pred_layer)
        per_world[str(wid)] = {
            "h1_gap_in_sample": gap, "transfer_given_match": sm.get("S2", {}).get("transfer_given_match"),
            "S2": sm.get("S2"),
            "F_star": {a: v["F_star"] for a, v in sm["addresses"].items()},
            "F_star_CEV": {a: v["F_star_CEV"] for a, v in sm["addresses"].items()},
            "mem_fused": st.get("mem", {}).get("fused"), "ft": st.get("ft"), "d_H2": st.get("d_H2")}
    gaps = [v["h1_gap_in_sample"] for v in per_world.values() if v["h1_gap_in_sample"] is not None]
    tgm = [v["transfer_given_match"] for v in per_world.values() if v["transfer_given_match"] is not None]
    h1 = bool(gaps) and len(gaps) == len(worlds) and sum(gaps) / len(gaps) >= SCREEN_H1_GAP \
        and any(t >= MATCH_TRANSFER_MIN for t in tgm)
    # G-KA1c: both FT streams ran to N = n1 on the first discovery world and wrote their newest fact (lived-form
    # efficacy right after each write, averaged over the stream) at least SCREEN_FT_NEWEST of the time
    first = worlds[0]
    ft_first = {arm: ft_rec(first, arm, n1) or {} for arm in FT_ARMS}
    newest = {arm: ft_first[arm].get("newest_efficacy") for arm in FT_ARMS}
    g1c = all(v is not None and v >= SCREEN_FT_NEWEST for v in newest.values())
    cons = consistency({int(w["setup"]["world"]): w for w in worlds}, pa_disc, st_disc)
    blocking = [name for name, ok in (("G-KA1c", g1c), ("consistency", cons["pass"])) if not ok]
    d_h2 = [v["d_H2"] for v in per_world.values() if v["d_H2"] is not None]
    body = {
        "ka_version": KA_VERSION,
        "discovery_worlds": sorted(int(w["setup"]["world"]) for w in worlds),
        "inputs": {str(w["setup"]["world"]): w["done"]["files"] for w in worlds},
        "config": cfg,
        "pred_layer": int(pred_layer),
        "pred_layer_rule": {"c0_logloss": {str(l): loss[l] for l in layers}, "G_F_rule": {str(l): gf[l] for l in layers},
                            "by_logloss": int(by_loss), "best_G_F": int(best_gf)},
        "pred_att": bool(pred_att), "pred_att_logloss_ratio": att_ratio,
        "mixtures": mixtures, "mixture_configs": {a: confs[a] for a in cfg["addresses"]},
        "mixture_discovery": pooled_stats,
        "h1_confirmatory": bool(h1),
        "screens": {"G-KA1a": {"per_world": per_world, "mean_gap": (sum(gaps) / len(gaps)) if gaps else None,
                               "pass": bool(h1)},
                    "G-KA1b": {"pass": bool(pred_att)},
                    "G-KA1c": {"world": int(first["setup"]["world"]), "N": int(n1), "newest_efficacy": newest,
                               "newest_lived": {arm: ft_first[arm].get("newest_lived") for arm in FT_ARMS},
                               "threshold": SCREEN_FT_NEWEST, "pass": bool(g1c)},
                    "consistency": cons},
        # gates whose failure forbids the freeze (verify.py --freeze and --preconditions refuse)
        "blocking_gates": blocking, "go": not blocking,
        "h2_predicted_direction": None if not d_h2 else ("memory" if sum(d_h2) > 0 else "FT"),
        "ft_configs": first["setup"]["ft_configs"],
        "primary_family": [h for h in PRIMARY if h != "KA-H1" or h1],
        "secondary_family": [h for h in SECONDARY if pred_att or h == "S2"],
    }
    return {"sha256": sha256_json(body), "body": body}


def check_selection(sel: Mapping) -> dict:
    if sha256_json(sel["body"]) != sel["sha256"]:
        raise ValueError("selection.json: the body does not match its recorded sha256")
    body = sel["body"]
    if body.get("ka_version") != KA_VERSION:
        raise ValueError(f"selection.json: written by {body.get('ka_version')!r}; this code reads {KA_VERSION!r}")
    return body


def selection_blockers(body: Mapping) -> List[str]:
    """Reasons why a selection forbids the freeze and any confirmatory run (empty list = go)."""
    out = []
    for gate in body.get("blocking_gates", ["blocking_gates missing from the selection"]):
        if gate == "G-KA1c":
            g = body["screens"]["G-KA1c"]
            out.append(f"G-KA1c failed: lived-form efficacy on the newest fact {g['newest_efficacy']} on discovery "
                       f"world {g['world']} (needs >= {g['threshold']} for both FT arms)")
        elif gate == "consistency":
            out.append("consistency checks failed on the discovery worlds: "
                       + ", ".join(body["screens"]["consistency"]["failures"]) + " (audit before the freeze)")
        else:
            out.append(str(gate))
    return out


# ---------------------------------------------------------------------------------------------------- decisions
def _vals(per_world: Mapping[int, Optional[float]]) -> Tuple[List[int], List[float]]:
    ws = [w for w in sorted(per_world) if per_world[w] is not None]
    return ws, [float(per_world[w]) for w in ws]


def _desc(per_world: Mapping[int, Optional[float]]) -> dict:
    ws, d = _vals(per_world)
    lo, hi = S.interval(d) if len(d) >= 2 else (math.nan, math.nan)
    return {"n": len(d), "worlds": ws, "values": d, "mean": (sum(d) / len(d)) if d else None, "ci95": [lo, hi]}


def decide(confirm: Mapping[int, dict], selection: Mapping) -> dict:
    """The preregistered confirmatory analysis. ``selection`` is the checked selection body."""
    sel = selection
    L = sel["pred_layer"]
    mixtures = sel["mixtures"]
    worlds = sorted(confirm)
    for w in worlds:          # a world without a required phase is an unfinished job, not an invalid unit
        check_phases(confirm[w], required_phases("confirm", bool(sel["h1_confirmatory"])))
    pa = {w: part_a_summary(confirm[w], mixtures) if confirm[w]["part_a"] else {"valid": False} for w in worlds}
    st = {w: stream_summary(confirm[w], L) for w in worlds}
    out: dict = {"ka_version": KA_VERSION, "worlds": worlds, "selection_sha256": sha256_json(sel),
                 "primary_family": sel["primary_family"], "secondary_family": sel["secondary_family"]}

    def frz(w, a, k):
        return pa[w]["addresses"][a]["frozen"][k]

    # ---- per-world effects
    eff: Dict[str, Dict[int, Optional[float]]] = {h: {} for h in ("KA-H1", "KA-H2", "ECE", "growth", "growth_fixed",
                                                                 "S1a", "S1b", "S2i", "S2ii", "S2iii")}
    for w in worlds:
        ok_a = pa[w].get("valid") and all(a in pa[w]["addresses"] for a in ("A-CTX",) + H1_BASELINES)
        eff["KA-H1"][w] = min(frz(w, "A-CTX", "Y") - frz(w, b, "Y") for b in H1_BASELINES) if ok_a else None
        s = st[w]
        ok_s = bool(s.get("valid"))
        eff["KA-H2"][w] = s.get("d_H2") if ok_s else None
        eff["ECE"][w] = s.get("ece") if ok_s else None
        eff["growth"][w] = s.get("growth") if ok_s else None
        eff["growth_fixed"][w] = s.get("growth_fixed") if ok_s else None
        eff["S1a"][w] = s.get("d_S1a") if ok_s else None
        eff["S1b"][w] = s.get("d_S1b") if ok_s else None
        eff["S2i"][w] = s.get("auc_min") if ok_s else None
        if ok_a and pa[w].get("S2"):
            s2 = pa[w]["S2"]
            rs = [s2["r_transfer"], s2["r_leak"]]
            eff["S2ii"][w] = min(S.fisher_z(r) for r in rs) if all(r is not None for r in rs) else None
            eff["S2iii"][w] = s2["transfer_given_match"]
        else:
            eff["S2ii"][w] = eff["S2iii"][w] = None
    out["per_world"] = {h: {str(w): v for w, v in d.items()} for h, d in eff.items()}

    def enough(h):
        return len(_vals(eff[h])[1]) >= S.MIN_WORLDS

    # ---- primary family
    pv: Dict[str, float] = {}
    tests: Dict[str, dict] = {}
    if "KA-H1" in sel["primary_family"]:
        d = _vals(eff["KA-H1"])[1]
        pv["KA-H1"] = S.signflip_pvalue(d, SESOI["KA-H1"]) if enough("KA-H1") else 1.0
    d2 = _vals(eff["KA-H2"])[1]
    tests["KA-H2"] = S.two_sided(d2, SESOI["KA-H2"]) if enough("KA-H2") else {"p_plus": 1.0, "p_minus": 1.0, "p": 1.0}
    pv["KA-H2"] = tests["KA-H2"]["p"]
    h3a = S.threshold_claim(_vals(eff["ECE"])[1], ECE_MAX, "less") if enough("ECE") else None
    h3b = S.threshold_claim(_vals(eff["growth"])[1], GROWTH_MAX, "less") if enough("growth") else None
    pv["KA-H3"] = max(h3a["p"], h3b["p"]) if h3a and h3b else 1.0
    rej = S.holm(pv, S.ALPHA)
    primary: dict = {}
    if "KA-H1" in sel["primary_family"]:
        d = _vals(eff["KA-H1"])[1]
        label = S.label_contrast(d, SESOI["KA-H1"], rej["KA-H1"])
        guards = {}
        if enough("KA-H1"):
            ws = _vals(eff["KA-H1"])[0]
            be_c = [pa[w]["addresses"]["A-CTX"]["best_E"] for w in ws]
            be_t = [pa[w]["addresses"]["A-TOK"]["best_E"] for w in ws]
            guards["editable"] = {"ctx_best_E": sum(be_c) / len(ws), "tok_best_E": sum(be_t) / len(ws),
                                  "pass": sum(be_c) / len(ws) >= sum(be_t) / len(ws) - GUARD["h1_efficacy"]}
            dpe = [frz(w, "A-CTX", "PE6") - frz(w, "A-TOK", "PE6") for w in ws]
            guards["pe6"] = {"lower95": S.lower_bound(dpe), "pass": S.lower_bound(dpe) >= -GUARD["h1_pe6"]}
            xs, rs = [frz(w, "A-CTX", "X") for w in ws], [frz(w, "A-CTX", "R") for w in ws]
            # pass: the upper bounds stay below budget + margin. A failed guard is a "locality cost" only when the
            # data also show the budget exceeded (a lower bound above the frontier constraint); otherwise the
            # interval is too wide to tell and the label says "not established" (PROTOCOL §6, §10.2).
            guards["locality"] = {"X_upper95": S.upper_bound(xs), "R_upper95": S.upper_bound(rs),
                                  "X_lower95": S.lower_bound(xs), "R_lower95": S.lower_bound(rs),
                                  "X_limit": S.X_MAX + GUARD["h1_x"], "R_limit": S.R_MAX + GUARD["h1_r"],
                                  "X_budget": S.X_MAX, "R_budget": S.R_MAX,
                                  "pass": S.upper_bound(xs) <= S.X_MAX + GUARD["h1_x"]
                                  and S.upper_bound(rs) <= S.R_MAX + GUARD["h1_r"],
                                  "exceeded": S.lower_bound(xs) > S.X_MAX or S.lower_bound(rs) > S.R_MAX}
            guards["relative"] = {b: {"dX_mean": sum(frz(w, "A-CTX", "X") - frz(w, b, "X") for w in ws) / len(ws),
                                      "dR_mean": sum(frz(w, "A-CTX", "R") - frz(w, b, "R") for w in ws) / len(ws)}
                                  for b in H1_BASELINES}
            if label == S.SUPPORTED:
                if not guards["editable"]["pass"]:
                    label = "not supported (address not editable)"
                elif not guards["pe6"]["pass"]:
                    label = "not supported (formal transfer cost)"
                elif not guards["locality"]["pass"]:
                    label = LOCALITY_COST if guards["locality"]["exceeded"] else LOCALITY_OPEN
        primary["KA-H1"] = {"p": pv["KA-H1"], "holm_reject": rej["KA-H1"], "sesoi": SESOI["KA-H1"], "label": label,
                            "guards": guards, **_desc(eff["KA-H1"])}
    else:
        primary["KA-H1"] = {"label": "not in the confirmatory family (G-KA1 discovery screen)"}
    label2 = S.label_two_sided(d2, SESOI["KA-H2"], rej["KA-H2"], tests["KA-H2"], ("memory better", "FT better"))
    g2 = {}
    if enough("KA-H2"):
        ws = _vals(eff["KA-H2"])[0]
        ftb = {w: st[w]["ft"][st[w]["ft_best"]] for w in ws}
        sd = [st[w]["mem"]["structured"] - ftb[w]["structured"] for w in ws
              if st[w]["mem"]["structured"] is not None and ftb[w]["structured"] is not None]
        mem_u = [st[w]["mem"]["Leak_U"] for w in ws if st[w]["mem"]["Leak_U"] is not None]
        ft_u = [ftb[w]["Leak_U"] for w in ws if ftb[w]["Leak_U"] is not None]
        mem_b = [st[w]["mem"]["dbpb"] for w in ws if st[w]["mem"]["dbpb"] is not None]
        ft_b = [ftb[w]["dbpb"] for w in ws if ftb[w]["dbpb"] is not None]
        g2 = {"structured_mem_minus_ft": {"values": sd, "upper95": S.upper_bound(sd) if len(sd) >= 2 else None,
                                          "lower95": S.lower_bound(sd) if len(sd) >= 2 else None},
              "Leak_U": {"memory": (sum(mem_u) / len(mem_u)) if mem_u else None,
                         "FT": (sum(ft_u) / len(ft_u)) if ft_u else None},
              "dbpb_upper95": {"memory": S.upper_bound(mem_b) if len(mem_b) >= 2 else None,
                               "FT": S.upper_bound(ft_b) if len(ft_b) >= 2 else None}}
        better = {"memory better": "memory", "FT better": "FT"}.get(label2)
        if better:
            sign = 1.0 if better == "memory" else -1.0
            sdiff = [sign * v for v in sd]
            fails = []
            if len(sdiff) < 2 or S.upper_bound(sdiff) > GUARD["h2_structured"]:
                fails.append("structured leak")
            lu = g2["Leak_U"][better]
            if lu is None or lu > GUARD["h2_leak_u"]:
                fails.append("Leak_U")
            ub = g2["dbpb_upper95"][better]
            if ub is None or ub > GUARD["h2_dbpb"]:
                fails.append("general ability")
            g2["failed"] = fails
            if fails:
                label2 = f"not supported ({better} better on G_F, guard failed: {', '.join(fails)})"
    primary["KA-H2"] = {**tests["KA-H2"], "holm_reject": rej["KA-H2"], "sesoi": SESOI["KA-H2"], "label": label2,
                        "guards": g2, **_desc(eff["KA-H2"])}
    fixed = _vals(eff["growth_fixed"])[1]
    informative = bool(fixed) and sum(fixed) / len(fixed) >= FIXED_GROWTH_MIN
    if not (h3a and h3b):
        label3 = S.INSUFFICIENT
    elif rej["KA-H3"]:
        label3 = S.SUPPORTED if informative else "inconclusive (gallery-size check uninformative)"
    elif h3a["ci95"][0] > ECE_MAX or h3b["ci95"][0] > GROWTH_MAX:
        label3 = S.REFUTED
    else:
        label3 = S.INCONCLUSIVE
    primary["KA-H3"] = {"p": pv["KA-H3"], "holm_reject": rej["KA-H3"], "label": label3,
                        "a_ece": {**(h3a or {}), **_desc(eff["ECE"])}, "b_growth": {**(h3b or {}), **_desc(eff["growth"])},
                        "manipulation_fixed_growth": {**_desc(eff["growth_fixed"]), "informative": informative}}
    out["primary"] = primary
    out["primary_holm"] = {"pvalues": pv, "adjusted": S.holm_adjusted(pv), "alpha": S.ALPHA, "m": len(pv)}

    # ---- secondary family
    spv: Dict[str, float] = {}
    stest: Dict[str, dict] = {}
    fam2 = list(sel["secondary_family"])
    if "S1a" in fam2:
        d = _vals(eff["S1a"])[1]
        stest["S1a"] = S.two_sided(d, SESOI["S1a"]) if enough("S1a") else {"p_plus": 1.0, "p_minus": 1.0, "p": 1.0}
        spv["S1a"] = stest["S1a"]["p"]
    if "S1b" in fam2:
        d = _vals(eff["S1b"])[1]
        spv["S1b"] = S.signflip_pvalue(d, SESOI["S1b"]) if enough("S1b") else 1.0
    parts = {"i": ("S2i", AUC_MIN), "ii": ("S2ii", S.fisher_z(CORR_MIN)), "iii": ("S2iii", MATCH_TRANSFER_MIN)}
    if "KA-H1" not in sel["primary_family"]:
        parts = {"i": parts["i"]}
    s2 = {k: (S.threshold_claim(_vals(eff[h])[1], thr, "greater") if enough(h) else None)
          for k, (h, thr) in parts.items()}
    spv["S2"] = max(v["p"] for v in s2.values()) if all(v is not None for v in s2.values()) else 1.0
    srej = S.holm(spv, S.ALPHA)
    secondary: dict = {}
    if "S1a" in fam2:
        d = _vals(eff["S1a"])[1]
        secondary["S1a"] = {**stest["S1a"], "holm_reject": srej["S1a"], "sesoi": SESOI["S1a"],
                            "label": S.label_two_sided(d, SESOI["S1a"], srej["S1a"], stest["S1a"],
                                                       ("PRED-ATT better", "PRED-RES better")), **_desc(eff["S1a"])}
    else:
        secondary["S1a"] = {"label": S.NOT_TESTED}
    if "S1b" in fam2:
        d = _vals(eff["S1b"])[1]
        secondary["S1b"] = {"p": spv["S1b"], "holm_reject": srej["S1b"], "sesoi": SESOI["S1b"],
                            "label": S.label_contrast(d, SESOI["S1b"], srej["S1b"]), **_desc(eff["S1b"])}
    else:
        secondary["S1b"] = {"label": S.NOT_TESTED}
    if any(v is None for v in s2.values()):
        l2 = S.INSUFFICIENT
    elif srej["S2"]:
        l2 = S.SUPPORTED
    elif any(v["ci95"][1] < v["threshold"] for v in s2.values()):
        l2 = S.REFUTED
    else:
        l2 = S.INCONCLUSIVE
    secondary["S2"] = {"p": spv["S2"], "holm_reject": srej["S2"], "label": l2,
                       "parts": {k: {**(v or {}), **_desc(eff[parts[k][0]])} for k, v in s2.items()}}
    out["secondary"] = secondary
    out["secondary_holm"] = {"pvalues": spv, "adjusted": S.holm_adjusted(spv), "alpha": S.ALPHA, "m": len(spv)}

    # ---- laws (count rules, outside Holm) and descriptive tables
    law_ok = {}
    for w in worlds:
        rows = [r for r in st[w].get("law", []) if r["any_over"] is not None and r["law"] is not None]
        law_ok[w] = all(r["any_over"] <= r["law"] + 0.01 for r in rows) if rows else None
    n_ok = sum(1 for v in law_ok.values() if v)
    n_def = sum(1 for v in law_ok.values() if v is not None)
    out["laws"] = {
        "false_fire_gaussian": {"rule": "observed share of U probes with a score above theta_N <= N sf((theta_N+m)/"
                                        "sqrt(2m)) + 0.01 at every checkpoint, on at least 6 of 8 worlds",
                                "per_world": {str(w): v for w, v in law_ok.items()}, "n_hold": n_ok,
                                "n_defined": n_def, "holds": n_def >= S.MIN_WORLDS and n_ok >= 6},
        "by_world": {str(w): st[w].get("law") for w in worlds},
        "markov_bound": "N-invariant (P_t/(1-P_t) = 1 expected false firing per unrelated query at P_t = 0.5) but "
                        "vacuous at the Leak_U <= 1% scale; reported, not tested",
    }
    # ---- hierarchical bootstrap (worlds, then edits or facts): descriptive intervals of the two contrasts
    boot: dict = {"resamples": S.BOOTSTRAP_N, "seed": S.BOOTSTRAP_SEED}
    if "KA-H1" in sel["primary_family"]:
        tab = {}
        for w in _vals(eff["KA-H1"])[0]:
            fy = pa[w].get("frozen_Y_by_edit", {})
            if all(a in fy for a in ("A-CTX",) + H1_BASELINES):
                tab[w] = [[fy["A-CTX"][i]] + [fy[b][i] for b in H1_BASELINES] for i in range(len(fy["A-CTX"]))]
        if tab:
            boot["KA-H1"] = {"unit": "edit", "worlds": sorted(tab), "ci95": list(S.hier_bootstrap(
                tab, combine=lambda m: (m[:, :1] - m[:, 1:]).min(axis=1)))}
    tab = {}
    for w in _vals(eff["KA-H2"])[0]:
        ff = st[w].get("fused_facts") or {}
        if all(ff.get(k) for k in ("mem",) + FT_ARMS):
            cols = [{int(i): float(v) for i, v in ff[k]} for k in ("mem",) + FT_ARMS]
            common = sorted(set(cols[0]).intersection(*cols[1:]))
            if common:
                tab[w] = [[c[i] for c in cols] for i in common]
    if tab:
        boot["KA-H2"] = {"unit": "fact", "worlds": sorted(tab), "ci95": list(S.hier_bootstrap(
            tab, combine=lambda m: m[:, 0] - m[:, 1:].max(axis=1)))}
    out["descriptive"] = {
        "part_a": {str(w): {a: {k: v for k, v in e.items() if k != "F_star_w"}
                            for a, e in pa[w].get("addresses", {}).items()} for w in worlds},
        "part_a_S2": {str(w): pa[w].get("S2") for w in worlds},
        "part_a_greedy_check": {str(w): pa[w].get("greedy_check") for w in worlds},
        "streams": {str(w): {k: st[w].get(k) for k in ("mem", "ft", "ft_best", "checks", "valid")} for w in worlds},
        "geometry": {str(w): confirm[w]["setup"].get("geometry") for w in worlds},
        "bootstrap": boot,
    }
    # ---- consistency checks (PROTOCOL §6): a failure means the scorers disagree, so no label may be read
    cons = consistency(confirm, pa, st)
    out["consistency"] = cons
    out["audit_required"] = not cons["pass"]
    if not cons["pass"]:
        for fam in ("primary", "secondary"):
            for entry in out[fam].values():
                entry["label_before_audit"] = entry.get("label")
                entry["label"] = AUDIT
    return out


def render_tables(dec: Mapping) -> str:
    """Markdown summary of the decisions (per-world values are always shown)."""
    lines = ["# KA — confirmatory decisions", "", f"Worlds: {dec['worlds']}", ""]
    cons = dec.get("consistency", {})
    if dec.get("audit_required"):
        lines += ["**AUDIT REQUIRED.** A consistency check failed (" + ", ".join(cons.get("failures", []))
                  + "): the scorers disagree or the base model changed. No label below may be read as a result; "
                  "the statistical labels are kept in `label_before_audit` of decisions.json.", ""]
    lines += ["| Hypothesis | family | n | mean | 95% CI | p | Holm | label |", "|---|---|---|---|---|---|---|---|"]

    def row(name, fam, e):
        ci = e.get("ci95") or [None, None]
        fmt = lambda v: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"  # noqa: E731
        lines.append(f"| {name} | {fam} | {e.get('n', '—')} | {fmt(e.get('mean'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
                     f"{fmt(e.get('p'))} | {e.get('holm_reject', '—')} | {e.get('label')} |")

    for h, e in dec["primary"].items():
        row(h, "primary", e)
    for h, e in dec["secondary"].items():
        row(h, "secondary", e)
    lines += ["", "## Per-world effects", "", "| effect | " + " | ".join(str(w) for w in dec["worlds"]) + " |",
              "|---|" + "---|" * len(dec["worlds"])]
    for h, d in dec["per_world"].items():
        lines.append(f"| {h} | " + " | ".join("—" if d.get(str(w)) is None else f"{d[str(w)]:.4f}"
                                              for w in dec["worlds"]) + " |")
    lf = dec["laws"]["false_fire_gaussian"]
    lines += ["", "## Laws (count rules, outside Holm)", "",
              f"- Gaussian false-firing law holds on {lf['n_hold']} of {lf['n_defined']} worlds: "
              f"{'holds' if lf['holds'] else 'does not hold'}.", ""]
    if cons:
        lines += ["## Consistency checks (pooled over the worlds, each share must be <= "
                  f"{cons['max_share']:g})", ""]
        for k, v in cons["pooled"].items():
            share = "not run" if v["share"] is None else f"{v['share']:.4f}"
            lines.append(f"- {k}: {v['disagree']} of {v['n']} ({share})")
        lines += [f"- base model unchanged on every world: {cons['model_unchanged']}",
                  f"- result: {'pass' if cons['pass'] else 'FAIL'}", ""]
    boot = dec.get("descriptive", {}).get("bootstrap", {})
    rows = [(h, boot[h]) for h in ("KA-H1", "KA-H2") if h in boot]
    if rows:
        lines += [f"## Hierarchical bootstrap (descriptive; worlds, then units; {boot['resamples']} resamples)", ""]
        for h, b in rows:
            lines.append(f"- {h}: 95% interval [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}] over {len(b['worlds'])} "
                         f"worlds, unit = {b['unit']}")
        lines.append("")
    return "\n".join(lines)


__all__ = ["SESOI", "PRIMARY", "SECONDARY", "AUDIT", "LOCALITY_COST", "LOCALITY_OPEN", "load_world", "load_worlds",
           "world_status", "check_phases", "address_configs", "part_a_cells", "pooled_cells", "edit_cells",
           "edit_mixture", "locality_guard_outlook", "part_a_summary", "consistency", "stream_summary", "select",
           "check_selection", "selection_blockers", "decide", "render_tables"]
