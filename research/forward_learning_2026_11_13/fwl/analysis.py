# -*- coding: utf-8 -*-
"""Tables, discovery selection and confirmatory decisions (PROTOCOL §4, §6, §7). The unit is the world.

Statistics come from the frozen CEV-11 package (``cev11.stats``): the exact sign-flip permutation test, Holm's
step-down adjustment and the t-interval. Edits and streams are averaged within a world first.

Primary tests: H1 G(FEC-A) - G(FT-ALL), margin 0.04; H2a G(FEC-A) - G(FEC-TOP); H2b against the best exact
one-step variant; H3 net facts per write of FEC* against every sequential comparator (intersection-union);
H4a gated against ungated read. The contrasts of the first draft (FT-L, FEC-0, composite S against one selected
comparator) are secondary tests.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import common as C

SUPPORTED, REFUTED, INCONCLUSIVE = "supported", "refuted", "inconclusive"
GUARD_FAILED, NOT_TESTABLE, CONTROL_FAILED = "guard failed", "not testable", "control failed"
PRIMARY = ("H1", "H2a", "H2b", "H3", "H4a")
MARGINS = {"H1": 0.04, "H1_FT_L": 0.05, "H2a": 0.05, "H2b": 0.05, "H3": 0.05, "H3_composite": 0.05, "H4a": 0.10,
           "leak_guard": 0.02, "bpb_guard": 0.01, "gate_leak_point": 0.02, "gate_leak_bound": 0.05,
           "rand_installs": 0.10, "knee_inner": 0.25, "knee_outer": 0.50, "kl": 0.05, "others": 0.02,
           "recalled": 0.5}
LEAKS = {"Leak_RN": "RN", "Leak_RD": "RD", "Leak_C": "RC", "Leak_RB": "RB", "Leak_SO": "SO"}


# -------------------------------------------------------------------------------------------------------- tables
def _rate(entry: Optional[dict]) -> Optional[float]:
    if not entry or not entry.get("n"):
        return None
    return entry["changed"] / entry["n"]


def single_rows(records: Iterable[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in records:
        if r["kind"] == "single":
            out.setdefault(f"{r['arm']}|{r['install']}", []).append(r)
    return out


def single_metrics(rows: List[dict]) -> dict:
    """Per-arm means over the edits of one world. Generalisation counts a failed install as 0 (all edits in the
    denominator); IE5 and PE5 are taken over the edits that have target cases; leaks over installed edits."""
    n = len(rows)
    ins = [r for r in rows if r["installed"]]
    pe6 = C.mean((sum(r["PF6"].values()) / len(r["PF6"])) if (r["installed"] and r["PF6"]) else 0.0 for r in rows)
    with_t = [r for r in rows if r.get("n_targets", len(r["targets"])) > 0]
    ie5 = C.mean((sum(t["informal"] for t in r["targets"]) / len(r["targets"]))
                 if (r["installed"] and r["targets"]) else 0.0 for r in with_t)
    pe5 = C.mean((sum(t["formal"] for t in r["targets"]) / len(r["targets"]))
                 if (r["installed"] and r["targets"]) else 0.0 for r in with_t)
    out = {"n": n, "WE": len(ins) / n if n else None, "PE6": pe6, "PE5": pe5, "IE5": ie5,
           "G": None if (pe6 is None or ie5 is None) else 0.5 * (pe6 + ie5)}
    for name, fam in LEAKS.items():
        out[name] = C.mean(v for v in (_rate(r["leak"].get(fam)) for r in ins) if v is not None)
    out["dbpb"] = C.mean(r["bpb"]["edited"] - r["bpb"]["base"] for r in ins if r.get("bpb"))
    out["mu_median"] = C.median([r["mu"] for r in ins if r.get("mu")])
    out["n_forward"] = C.mean(r["n_forward"] for r in rows)
    out["seconds"] = C.mean(r["seconds"] for r in rows)
    return out


def single_table(records: Iterable[dict]) -> Dict[str, dict]:
    return {k: single_metrics(v) for k, v in sorted(single_rows(records).items())}


def stream_table(records: Iterable[dict]) -> Dict[str, dict]:
    """Per (arm, checkpoint): means over the streams of one world."""
    groups: Dict[str, List[dict]] = {}
    for r in records:
        if r["kind"] == "stream":
            groups.setdefault(f"{r['arm']}|{r['n']}", []).append(r)
    out = {}
    for key, rows in sorted(groups.items()):
        row = {"streams": len(rows), "truncated": any(r.get("truncated") for r in rows)}
        for f in ("installed", "retention", "PF6", "IF", "dmg_A", "dmg_B", "own_BC", "net_per_write", "composite",
                  "mu_median", "mu_max", "seconds"):
            row[f] = C.mean(r[f] for r in rows if r.get(f) is not None)
        row["dbpb"] = C.mean(r["dbpb"] for r in rows if "dbpb" in r)
        row["damage_AB"] = C.mean(
            (r["dmg_A"] * r["known_A"] + r["dmg_B"] * r["known_B"]) / max(r["known_A"] + r["known_B"], 1)
            for r in rows if "known_A" in r)
        row["rms_ratio_max"] = C.mean(max(r["rms_ratio"].values()) for r in rows if r.get("rms_ratio"))
        row["evictions"] = C.mean(r["evictions"] for r in rows if "evictions" in r)
        out[key] = row
    return out


def final_n(records: Iterable[dict]) -> Optional[int]:
    ns = [r["n"] for r in records if r["kind"] == "stream" and not r.get("truncated")]
    return max(ns) if ns else None


# -------------------------------------------------------------------------------------------- per-world statistics
def gate_table(records: Iterable[dict]) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}
    for r in records:
        if r["kind"] == "gate":
            out.setdefault(r["arm"], {})[r["setting"]] = r
    return out


def world_stats(records: List[dict]) -> dict:
    """Everything the decisions need from one world's records."""
    st = single_table(records)
    n = final_n(records)
    streams = stream_table(records)
    out = {"single": st, "final_n": n,
           "stream": {k.split("|")[0]: v for k, v in streams.items() if n is not None and k.endswith(f"|{n}")},
           "gate": gate_table(records)}
    caps = [r for r in records if r["kind"] == "capacity"]
    out["capacity"] = caps[0] if caps else None
    rets = [r for r in records if r["kind"] == "retract" and r["p_before"] >= MARGINS["recalled"]]
    out["retract"] = {"n": len(rets),
                      "kl": C.mean(C.mean(r["kl_replay_retract"]) for r in rets),
                      "others_diff": C.mean(r["others_retract"] - r["others_replay"] for r in rets),
                      "p_retract": C.mean(r["p_retract"] for r in rets),
                      "p_replay": C.mean(r["p_replay"] for r in rets)}
    return out


# ------------------------------------------------------------------------------------------------------ selection
def _pooled(stats: Dict[int, dict], getter) -> Optional[float]:
    return C.mean(v for v in (getter(s) for s in stats.values()) if v is not None)


def choose_operating_point(tables: List[Dict[str, dict]], kind: str) -> dict:
    """The setting of one knob ('theta' or 'gain') with the highest mean recall on other case forms among those
    whose mean hard-negative leak is <= 2% and whose mean prose bpb change is <= 0.01; if none is feasible the
    most conservative setting is taken and flagged."""
    by_setting: Dict[str, List[dict]] = {}
    for t in tables:
        for label, r in t.items():
            if label.startswith(kind + "=") or (kind == "gain" and label == "theta=0"):
                by_setting.setdefault(label, []).append(r)
    rows = []
    for label, rs in by_setting.items():
        value = rs[0]["theta"] if kind == "theta" else rs[0]["gain"]
        dbpb = C.mean(r["dbpb"] for r in rs if "dbpb" in r)
        rows.append({"setting": label, "value": value, "recall_other": C.mean(r["recall_other"] for r in rs),
                     "leak_hard": C.mean(r["leak_hard"] for r in rs), "dbpb": dbpb,
                     "retention": C.mean(r["retention"] for r in rs), "worlds": len(rs)})
    if not rows:
        return {"value": None, "feasible": False, "table": []}
    for row in rows:
        row["feasible"] = bool(row["leak_hard"] <= MARGINS["gate_leak_point"]
                               and (row["dbpb"] is None or row["dbpb"] <= MARGINS["bpb_guard"]))
    ok = [r for r in rows if r["feasible"]]
    conservative = (lambda r: r["value"]) if kind == "theta" else (lambda r: -r["value"])
    if ok:
        best = max(ok, key=lambda r: (r["recall_other"], conservative(r)))
    else:
        best = max(rows, key=conservative)
    return {"value": best["value"], "setting": best["setting"], "feasible": bool(ok),
            "table": sorted(rows, key=lambda r: r["value"])}


def select(discovery: Dict[int, List[dict]]) -> dict:
    """Operating points from the discovery worlds only (PROTOCOL §6). Deterministic; the result carries its hash."""
    stats = {w: world_stats(recs) for w, recs in sorted(discovery.items())}
    caps_single: Dict[str, float] = {}
    for arm in C.SINGLE_ONE_STEP:
        for install in C.INSTALLS:
            mus = [r["mu"] for recs in discovery.values() for r in recs
                   if r["kind"] == "single" and r["arm"] == arm and r["install"] == install
                   and r["installed"] and r.get("mu")]
            if mus:
                caps_single[f"{arm}|{install}"] = 4.0 * C.median(mus)
    for install in C.INSTALLS:                       # the control has no installing steps of its own
        if f"FEC-0|{install}" in caps_single:
            caps_single[f"RAND-0|{install}"] = caps_single[f"FEC-0|{install}"]
    # Stream caps. Every arm whose step is the Frobenius norm of its write (all learner arms) gets ONE common
    # cap: 4 x the median installing step of the two FEC stream arms over the discovery streams. A cap from an
    # arm's own uncapped run is useless for an arm that collapses without a cap (its steps grow with the damage:
    # on discovery world 0 the first exact-gradient step was 22.6 against 2.0 for FEC-TOP), and the source
    # proposal asks for "the same step cap for all". RLSEdit's step is a multiple of its own value search and
    # keeps its own cap; sequential FT-L has no step search.
    caps_stream: Dict[str, float] = {}
    fec_meds = [s["stream"][arm]["mu_median"] for s in stats.values() for arm in C.FEC_STREAM
                if arm in s["stream"] and s["stream"][arm].get("mu_median")]
    if fec_meds:
        for arm in C.NORMALISED_STREAM_ARMS:
            caps_stream[arm] = 4.0 * C.median(fec_meds)
    for arm in C.STREAM_ARMS:
        if arm in C.NORMALISED_STREAM_ARMS:
            continue
        meds = [s["stream"][arm]["mu_median"] for s in stats.values()
                if arm in s["stream"] and s["stream"][arm].get("mu_median")]
        if meds:
            caps_stream[arm] = 4.0 * C.median(meds)

    def best(candidates: Sequence[str], getter) -> Tuple[Optional[str], Dict[str, Optional[float]]]:
        scores = {c: _pooled(stats, lambda s, c=c: getter(s, c)) for c in candidates}
        live = [c for c in candidates if scores[c] is not None]
        return (max(live, key=lambda c: (scores[c], -list(candidates).index(c))) if live else None), scores

    bp_star, bp_scores = best(("BP-1", "BP-1b", "BP-pred"),
                              lambda s, c: (s["single"].get(f"{c}|greedy") or {}).get("G"))
    # FEC* by net facts per write (the primary quantity of H3). CMP* is descriptive only: H3 is tested against
    # every comparator, because the discovery streams run without caps and cannot rank the comparators fairly.
    fec_star, fec_scores = best(C.FEC_STREAM, lambda s, c: (s["stream"].get(c) or {}).get("net_per_write"))
    cmp_star, cmp_scores = best(C.COMPARATORS, lambda s, c: (s["stream"].get(c) or {}).get("net_per_write"))
    gate = {}
    for arm in C.GATE_ARMS:
        tables = [s["gate"][arm] for s in stats.values() if arm in s["gate"]]
        if tables:
            gate[arm] = {"theta": choose_operating_point(tables, "theta"),
                         "gain": choose_operating_point(tables, "gain")}
    heads = [r for recs in discovery.values() for r in recs if r["kind"] == "meta"]
    engines = sorted({h["engine_sha256"] for h in heads})
    configs = sorted({h["config_sha256"] for h in heads})
    complete = len(heads) == len(discovery)
    sel = {"schema": "fwl.selection.v2", "worlds": sorted(discovery), "final_n": {str(w): s["final_n"]
                                                                                  for w, s in stats.items()},
           # the engine and the run configuration the discovery records were made with (one value each, or None)
           "engine_sha256": engines[0] if (complete and len(engines) == 1) else None,
           "discovery_config_sha256": configs[0] if (complete and len(configs) == 1) else None,
           "caps": {"single": caps_single, "stream": caps_stream}, "bp_star": bp_star, "bp_scores": bp_scores,
           "fec_star": fec_star, "fec_scores": fec_scores, "cmp_star": cmp_star, "cmp_scores": cmp_scores,
           "gate_arm": fec_star if fec_star in gate else (sorted(gate)[0] if gate else None), "gate": gate,
           "records_sha256": {str(w): C.sha256_json(recs) for w, recs in sorted(discovery.items())}}
    sel["sha256"] = C.sha256_json(sel)
    return sel


def selection_hash_ok(sel: dict) -> bool:
    body = {k: v for k, v in sel.items() if k != "sha256"}
    return sel.get("sha256") == C.sha256_json(body)


# ------------------------------------------------------------------------------------------------------ decisions
def _interval(d: Sequence[float]) -> Tuple[float, float]:
    from cev11 import stats as S
    return S.t_interval(list(d), 0.95)


def _effect(d: Sequence[float]) -> dict:
    lo, hi = _interval(d) if len(d) >= 2 else (math.nan, math.nan)
    return {"n": len(d), "mean": sum(d) / len(d) if d else None, "ci95": [lo if math.isfinite(lo) else None,
                                                                             hi if math.isfinite(hi) else None],
            "worlds": list(d)}


def test_noninferiority(d: Sequence[float], margin: float) -> dict:
    """H0: mean(d) <= -margin. Refuted when the 95% upper bound is below -margin."""
    from cev11 import stats as S
    out = _effect(d)
    out.update({"kind": "non-inferiority", "margin": margin, "p": S.signflip_pvalue(list(d), -margin)})
    return out


def test_superiority(d: Sequence[float], sesoi: float) -> dict:
    """H0: mean(d) <= 0, with a smallest effect of interest. Refuted when the 95% upper bound is below it."""
    from cev11 import stats as S
    out = _effect(d)
    out.update({"kind": "superiority", "sesoi": sesoi, "p": S.signflip_pvalue(list(d), 0.0)})
    return out


def label(test: dict, p_adjusted: float, alpha: float = 0.05) -> str:
    hi = test.get("refute_upper", test["ci95"][1])      # intersection-union tests: the lowest upper bound
    if test["kind"] == "non-inferiority":
        if p_adjusted <= alpha:
            return SUPPORTED
        return REFUTED if (hi is not None and hi < -test["margin"]) else INCONCLUSIVE
    if p_adjusted <= alpha and test["mean"] >= test["sesoi"]:
        return SUPPORTED
    return REFUTED if (hi is not None and hi < test["sesoi"]) else INCONCLUSIVE


def guard(d: Sequence[float], limit: float) -> dict:
    """A guard fails when the upper bound of its 95% interval is above the limit (not on the point estimate)."""
    out = _effect(d)
    hi = out["ci95"][1]
    out.update({"limit": limit, "ok": bool(hi is not None and hi <= limit)})
    return out


def test_against_every(stats: Dict[int, dict], a, comparators: Sequence[Tuple[str, object]], margin: float,
                       min_worlds: int) -> Tuple[Optional[dict], Optional[str]]:
    """Intersection-union non-inferiority test: ``a`` must be non-inferior to every comparator. The p-value is
    the largest of the single p-values (valid without a correction inside the union); the test is refuted when
    the 95% upper bound against any one comparator lies below the margin. Returns (test, None), or (None, name)
    when a comparator has fewer than ``min_worlds`` paired worlds."""
    per: Dict[str, dict] = {}
    for name, getter in comparators:
        d, worlds = _paired(stats, a, getter)
        if len(d) < min_worlds:
            return None, name
        per[name] = dict(test_noninferiority(d, margin), world_ids=worlds)
    if not per:
        return None, "no comparator"
    binding = max(per, key=lambda c: (per[c]["p"], -per[c]["mean"], c))
    uppers = [t["ci95"][1] for t in per.values() if t["ci95"][1] is not None]
    out = dict(per[binding])
    out.update({"kind": "non-inferiority", "against": binding, "union": "every comparator",
                "refute_upper": min(uppers) if uppers else None,
                "comparators": {c: {k: t[k] for k in ("n", "mean", "ci95", "p", "worlds")} for c, t in per.items()}})
    return out, None


def _paired(stats: Dict[int, dict], a, b) -> Tuple[List[float], List[int]]:
    d, worlds = [], []
    for w, s in sorted(stats.items()):
        x, y = a(s), b(s)
        if x is not None and y is not None:
            d.append(x - y)
            worlds.append(w)
    return d, worlds


def _single(key: str, metric: str = "G"):
    return lambda s: (s["single"].get(key) or {}).get(metric)


def _stream(arm: str, metric: str = "composite"):
    return lambda s: (s["stream"].get(arm) or {}).get(metric)


def _gate(arm: str, setting: str, metric: str):
    return lambda s: ((s["gate"].get(arm) or {}).get(setting) or {}).get(metric)


def compute_decisions(per_world: Dict[int, List[dict]], selection: dict, *, min_worlds: int = 6) -> dict:
    """Confirmatory decisions from the records of the confirmatory worlds and the frozen discovery selection."""
    from cev11 import stats as S
    stats = {w: world_stats(recs) for w, recs in sorted(per_world.items())}
    tests: Dict[str, dict] = {}
    notes: Dict[str, str] = {}

    def add(name: str, d_w, kind: str, size: float, need: Optional[str] = None):
        d, worlds = d_w
        if need is not None and not need:
            notes[name] = NOT_TESTABLE
            return
        if len(d) < min_worlds:
            notes[name] = f"{NOT_TESTABLE} ({len(d)} worlds)"
            return
        tests[name] = (test_noninferiority if kind == "ni" else test_superiority)(d, size)
        tests[name]["world_ids"] = worlds

    bp, fec, cmp_ = selection.get("bp_star"), selection.get("fec_star"), selection.get("cmp_star")
    garm = selection.get("gate_arm")
    gsel = (selection.get("gate") or {}).get(garm) or {}
    theta_s, gain_s = (gsel.get("theta") or {}).get("setting"), (gsel.get("gain") or {}).get("setting")
    add("H1", _paired(stats, _single("FEC-A|nll"), _single("FT-ALL|ft")), "ni", MARGINS["H1"])
    add("H2a", _paired(stats, _single("FEC-A|greedy"), _single("FEC-TOP|greedy")), "sup", MARGINS["H2a"])
    add("H2b", _paired(stats, _single("FEC-A|greedy"), _single(f"{bp}|greedy")), "ni", MARGINS["H2b"], bp or "")
    if not fec:
        notes["H3"] = NOT_TESTABLE
    else:
        h3, lacking = test_against_every(stats, _stream(fec, "net_per_write"),
                                         [(c, _stream(c, "net_per_write")) for c in C.COMPARATORS],
                                         MARGINS["H3"], min_worlds)
        if h3 is None:
            notes["H3"] = f"{NOT_TESTABLE} (comparator {lacking}: too few worlds)"
        else:
            tests["H3"] = dict(h3, arm=fec, metric="net_per_write")
    add("H4a", _paired(stats, _gate(garm or "", theta_s or "", "recall_other"),
                       _gate(garm or "", gain_s or "", "recall_other")), "sup", MARGINS["H4a"],
        (garm and theta_s and gain_s) or "")
    pvals = {h: (tests[h]["p"] if h in tests else 1.0) for h in PRIMARY}      # an untested member keeps p = 1
    adjusted = S.holm_adjusted(pvals)
    decisions: Dict[str, dict] = {}
    for h in PRIMARY:
        if h not in tests:
            decisions[h] = {"label": notes.get(h, NOT_TESTABLE)}
            continue
        t = dict(tests[h], p_holm=adjusted[h])
        t["label"] = label(t, adjusted[h])
        decisions[h] = t

    # guards and controls
    if "H1" in tests:
        guards = {name: guard(_paired(stats, _single("FEC-A|nll", name), _single("FT-ALL|ft", name))[0],
                              MARGINS["leak_guard"]) for name in ("Leak_RD", "Leak_C", "Leak_RN")}
        bpb = [v for v in (_single("FEC-A|nll", "dbpb")(s) for s in stats.values()) if v is not None]
        guards["dbpb"] = guard(bpb, MARGINS["bpb_guard"]) if len(bpb) >= 2 else {"ok": False, "n": len(bpb)}
        decisions["H1"]["guards"] = guards
        if decisions["H1"]["label"] == SUPPORTED and not all(g["ok"] for g in guards.values()):
            decisions["H1"]["label"] = GUARD_FAILED
    rand = [v for v in (_single("RAND-0|greedy", "WE")(s) for s in stats.values()) if v is not None]
    control = {"install_rate": C.mean(rand), "limit": MARGINS["rand_installs"], "worlds": rand,
               "ok": bool(rand) and C.mean(rand) <= MARGINS["rand_installs"]}
    for h in ("H2a", "H2b"):
        if h in tests:
            decisions[h]["control_RAND-0"] = control
            if not control["ok"] and decisions[h]["label"] == SUPPORTED:
                decisions[h]["label"] = CONTROL_FAILED
    if "H4a" in tests:
        leak = [v for v in (_gate(garm, theta_s, "leak_hard")(s) for s in stats.values()) if v is not None]
        g = guard(leak, MARGINS["gate_leak_bound"]) if len(leak) >= 2 else {"ok": False, "n": len(leak)}
        g["point_limit"] = MARGINS["gate_leak_point"]
        decisions["H4a"]["guard_leak_hard"] = g
        decisions["H4a"]["operating_point"] = {"arm": garm, "theta": theta_s, "gain": gain_s,
                                               "feasible_on_discovery": [(gsel.get("theta") or {}).get("feasible"),
                                                                         (gsel.get("gain") or {}).get("feasible")]}
        if decisions["H4a"]["label"] == SUPPORTED and not g["ok"]:
            decisions["H4a"]["label"] = GUARD_FAILED

    # H4b: capacity knee (count rule), at the operating threshold when it is a registered one, else 0.5
    ratios, used = [], []
    theta_star = (gsel.get("theta") or {}).get("value")
    for w, s in sorted(stats.items()):
        cap = s["capacity"]
        if not cap:
            continue
        knees = {k["theta"]: k for k in cap["knees"]}
        k = knees.get(theta_star) or knees.get(0.5)
        if k and k["ratio"] is not None:
            ratios.append(k["ratio"])
            used.append({"world": w, "theta": k["theta"], "ratio": k["ratio"], "mode": "knee"})
        elif cap["predicted_end"] > 0:                # the median never crossed: compare the level at the end
            r = cap["nu_mean_end"] / cap["predicted_end"]
            ratios.append(r)
            used.append({"world": w, "ratio": r, "mode": "level at the largest load"})
    inner = sum(abs(r - 1.0) <= MARGINS["knee_inner"] for r in ratios)
    outer = sum(abs(r - 1.0) > MARGINS["knee_outer"] for r in ratios)
    n_r = len(ratios)
    if n_r < min_worlds:
        h4b = NOT_TESTABLE
    elif inner >= math.ceil(0.875 * n_r) and outer == 0:
        h4b = SUPPORTED
    elif n_r - inner >= math.ceil(0.625 * n_r):
        h4b = REFUTED
    else:
        h4b = INCONCLUSIVE
    # descriptive: the same ratio with the crossing of the measured MEAN (what the law predicts)
    mean_ratios = []
    for w, s in sorted(stats.items()):
        cap = s["capacity"]
        km = {k["theta"]: k for k in (cap or {}).get("knees_mean", [])}
        k = km.get(theta_star) or km.get(0.5)
        if k and k["ratio"] is not None:
            mean_ratios.append({"world": w, "theta": k["theta"], "ratio": k["ratio"]})
    decisions["H4b"] = {"label": h4b, "ratios": used, "inside_25": inner, "outside_50": outer, "n": n_r,
                        "ratios_mean_crossing": mean_ratios}

    # H5: retraction against a replay without the fact
    kl = [s["retract"]["kl"] for s in stats.values() if s["retract"]["kl"] is not None]
    oth = [s["retract"]["others_diff"] for s in stats.values() if s["retract"]["others_diff"] is not None]
    h5 = {"kl": _effect(kl), "others_diff": _effect(oth), "limit_kl": MARGINS["kl"],
          "limit_others": MARGINS["others"]}
    if len(kl) < min_worlds:
        h5["label"] = NOT_TESTABLE
    else:
        lo, hi = h5["kl"]["ci95"]
        olo, ohi = h5["others_diff"]["ci95"]
        within = (hi is not None and hi <= MARGINS["kl"] and olo is not None
                  and olo >= -MARGINS["others"] and ohi <= MARGINS["others"])
        h5["label"] = SUPPORTED if within else (REFUTED if (lo is not None and lo > MARGINS["kl"]) else INCONCLUSIVE)
    decisions["H5"] = h5

    # secondary, unadjusted
    secondary = {}
    for name, a, b, kind, size in (
            ("routing_vs_FEC-0_greedy", _single("FEC-A|greedy"), _single("FEC-0|greedy"), "sup", MARGINS["H2a"]),
            ("routing_vs_FEC-0_nll", _single("FEC-A|nll"), _single("FEC-0|nll"), "sup", MARGINS["H2a"]),
            ("H2a_nll_rule", _single("FEC-A|nll"), _single("FEC-TOP|nll"), "sup", MARGINS["H2a"]),
            ("H1_vs_FT-L", _single("FEC-A|nll"), _single("FT-L|ft"), "ni", MARGINS["H1_FT_L"]),
            ("H1_greedy_rule", _single("FEC-A|greedy"), _single("FT-ALL|ft"), "ni", MARGINS["H1"]),
            ("H3_composite_vs_CMP", _stream(fec or ""), _stream(cmp_ or ""), "ni", MARGINS["H3_composite"]),
            ("H3_superiority_vs_CMP", _stream(fec or "", "net_per_write"), _stream(cmp_ or "", "net_per_write"),
             "sup", 0.05),
            ("selectivity_SO_FEC-A_vs_BP", _single("FEC-A|nll", "Leak_SO"), _single(f"{bp}|nll", "Leak_SO"),
             "sup", 0.02)):
        d, worlds = _paired(stats, a, b)
        if len(d) >= min_worlds:
            t = (test_noninferiority if kind == "ni" else test_superiority)(d, size)
            t["label"] = label(t, t["p"])
            secondary[name] = t
        else:
            secondary[name] = {"label": NOT_TESTABLE, "n": len(d)}
    return {"schema": "fwl.decisions.v2", "worlds": sorted(per_world), "selection_sha256": selection.get("sha256"),
            "margins": MARGINS, "decisions": decisions, "secondary": secondary,
            "tables": {str(w): {"single": s["single"], "stream": s["stream"]} for w, s in stats.items()}}
