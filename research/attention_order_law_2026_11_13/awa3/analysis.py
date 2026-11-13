# -*- coding: utf-8 -*-
"""AWA-3 analysis: the preregistered decisions of PROTOCOL sections 3 and 10, from the per-run eval records.

Usage (from the repository root, single thread):
    python research/attention_order_law_2026_11_13/run_analyze.py [--results DIR] [--phase 1] [--allow-truncated]

Reads  <results>/phase<P>/runs/<arm>_s<seed>/{meta.json, eval.json}
Writes <results>/phase<P>/decisions.json and <results>/phase<P>/tables/summary.md

``compute_decisions`` is a pure function of the loaded records (no file access, no randomness), so run_verify.py
can recompute the stored decisions. Nothing here looks at training curves; the unit is the training run (seed).
Every rule below quotes its PROTOCOL section; where PROGRAM.md left a detail open, the choice made here is
listed in PROTOCOL section 12 ("Open decisions to close before the freeze").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import stats

STUDY = pathlib.Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]

DECISIONS_FORMAT = "awa3-decisions"
STRATA = ("sorted", "oneswap", "anti", "shuffled")
ARM_SEEDS = {"AWA-2L": 5, "ChaCAL-2L": 5, "AWA-own-2L": 5, "LOOPED-2L": 4, "ARLOOP-2L": 4, "STD-2L-pm": 3}
WALK_ARMS = ("AWA-2L", "AWA-own-2L")
LOOP_ARMS = ("LOOPED-2L", "ARLOOP-2L")
LOWER_TRIANGULAR_ARMS = ("ChaCAL-2L", "AWA-own-2L")
ENDPOINT_DEPTHS = (6, 8)
TEST_BUDGET = 12
PLATEAU_BUDGET = 24
# thresholds (PROTOCOL section 3)
Y_SUCCESS = 0.80
PM_SHIFT = 0.15
PM_SESOI = 0.30
PM_MIN_T4 = 4
PM_MIN_SUCCESSES = 4
PM_REFUTE_MAX_SUCCESSES = 1
PM_REFUTE_MEDIAN = 0.50
PT_DEPTHS = (3, 4, 6, 8)
PT_INFORMATIVE_MARGIN = 0.30
PT_FLOOR_MARGIN = 0.10
PT_MIXTURE_TOL = 0.05
PT_REFUTE_MARGIN = 0.15
H3B_DEPTHS = (4, 6, 8)
H6_SESOI = 0.15
ANALYSIS_FILES = ("awa3/analysis.py", "awa3/stats.py")


# ------------------------------------------------------------------------------------------ access
def run_id(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def _acc(cell: dict, index: int) -> float:
    return cell["correct"][index] / cell["n"]


def _index(ev: dict, budget: int) -> int:
    if ev["forward"] in ("std", "exact"):
        return 0
    return ev["budgets"].index(budget)


def acc(ev: dict, k: int, stratum: str, budget: int, split: str = "test") -> float:
    return _acc(ev[split][str(k)][stratum], _index(ev, budget))


def worst_order(ev: dict, budget_by_depth: Dict[int, int], depths: Sequence[int] = ENDPOINT_DEPTHS,
                corrected: bool = False) -> float:
    """min over the four strata of the mean over ``depths`` of the accuracy (optionally chance-corrected)."""
    values = []
    for stratum in STRATA:
        per_depth = []
        for k in depths:
            a = acc(ev, k, stratum, budget_by_depth[k])
            if corrected:
                h = ev["floors"][str(k)]
                a = (a - h) / (1.0 - h)
            per_depth.append(a)
        values.append(stats.mean(per_depth))
    return min(values)


def oracle_budget(ev: dict, k: int) -> int:
    """Loop arms: the T that maximises the worst-stratum VALIDATION accuracy at depth k (ties: smallest T)."""
    best, best_value = ev["budgets"][0], -1.0
    for b in ev["budgets"]:
        value = min(acc(ev, k, s, b, split="val") for s in STRATA)
        if value > best_value + 1e-12:
            best, best_value = b, value
    return best


def test_budgets(ev: dict) -> Dict[int, int]:
    return {k: TEST_BUDGET for k in ENDPOINT_DEPTHS}


def plateau_budget(ev: dict, k: int) -> int:
    """Walks: K = 24. Loops: the oracle T of the depth (validation-chosen); at the in-distribution depths, which
    have no oracle validation set, the best T on the shuffled test cell. Exact arms: their single evaluation."""
    if ev["forward"] == "walk":
        return PLATEAU_BUDGET
    if ev["forward"] in ("looped", "arloop"):
        if str(k) in ev["val"]:
            return oracle_budget(ev, k)
        return max(ev["budgets"], key=lambda b: (acc(ev, k, "shuffled", b), -b))
    return 0


def plateau_budgets(ev: dict, depths: Sequence[int] = ENDPOINT_DEPTHS) -> Dict[int, int]:
    return {k: plateau_budget(ev, k) for k in depths}


def mean_over_strata(ev: dict, k: int, budget: int) -> float:
    return stats.mean([acc(ev, k, s, budget) for s in STRATA])


def per_run_summary(ev: dict, meta: dict) -> dict:
    plateau = plateau_budgets(ev)
    out = {
        "arm": ev["arm"], "seed": meta["seed"], "steps": meta["steps"], "extended": meta["extended"],
        "competent": bool(meta["competence"]["competent"]), "competence_value": meta["competence"]["value"],
        "Y": worst_order(ev, test_budgets(ev)), "Y_chance_corrected": worst_order(ev, test_budgets(ev), corrected=True),
        "Y_plateau": worst_order(ev, plateau), "plateau_budgets": {str(k): b for k, b in plateau.items()},
        "truncated_to": ev.get("truncated_to"),
    }
    return out


# ---------------------------------------------------------------------------------------- decisions
def _arm_runs(runs: Dict[str, dict], arm: str) -> List[dict]:
    return sorted((r for r in runs.values() if r["eval"]["arm"] == arm), key=lambda r: r["meta"]["seed"])


def decide_g0(runs: Dict[str, dict]) -> dict:
    """PROTOCOL 3.1: learnability gate and the negative control for assumption B."""
    awa = _arm_runs(runs, "AWA-2L")
    competent = sum(bool(r["meta"]["competence"]["competent"]) for r in awa)
    out: Dict[str, Any] = {"awa_runs": len(awa), "awa_competent": competent,
                           "passed": len(awa) == ARM_SEEDS["AWA-2L"] and competent >= 4}
    std = _arm_runs(runs, "STD-2L-pm")
    good = [r for r in std if r["meta"]["competence"]["competent"]]
    required = min(2, len(std)) if std else 2
    control: Dict[str, Any] = {"std_runs": len(std), "std_competent": len(good), "required": required}
    if len(good) < required:
        control["status"] = "uninformative"                       # B is reported as untested
    else:
        excess = {}
        for k in (2, 3):
            floor = good[0]["eval"]["floors"][str(k)]
            excess[str(k)] = stats.mean([acc(r["eval"], k, "shuffled", 0) for r in good]) - floor
        control["excess_over_floor"] = excess
        if excess["2"] > 0.15:
            control["status"] = "B false"
        elif max(excess.values()) <= 0.10:
            control["status"] = "B holds"
        else:
            control["status"] = "B not supported"
    out["control"] = control
    return out


def decide_pm_contrast(awa: List[dict], comp: List[dict]) -> dict:
    """PROTOCOL 3.2: one P-M contrast (AWA against a comparator), three criteria.

    PROTOCOL 12.3: a comparator run that is not ID-competent makes the contrast INCONCLUSIVE ("inconclusive"):
    its numbers are reported, but it can neither pass nor supply the "HL < 0.15" refutation."""
    ya = [r["summary"]["Y"] for r in awa]
    yb = [r["summary"]["Y"] for r in comp]
    t4 = stats.exact_t4(ya, yb, PM_SHIFT)
    hl = stats.hodges_lehmann(ya, yb, stats.ALPHA)
    mw = stats.mann_whitney_exact(ya, yb, PM_SHIFT, "greater")
    successes = sum(y >= Y_SUCCESS for y in ya)
    criteria = {"seed_count": successes >= PM_MIN_SUCCESSES, "T4": t4["T4"] >= PM_MIN_T4,
                "HL": hl["estimate"] >= PM_SESOI}
    not_competent = [r["meta"]["seed"] for r in comp if not r["meta"]["competence"]["competent"]]
    return {
        "Y_awa": ya, "Y_comparator": yb, "awa_successes": successes, "T4": t4, "hodges_lehmann": hl,
        "shifted_mann_whitney": mw, "criteria": criteria, "passes": all(criteria.values()),
        "comparator_not_id_competent": not_competent, "inconclusive": bool(not_competent),
    }


def decide_pm(runs: Dict[str, dict], g0: dict) -> dict:
    awa = _arm_runs(runs, "AWA-2L")
    out: Dict[str, Any] = {"contrasts": {}}
    comparators = {"a": "ChaCAL-2L", "b": "AWA-own-2L"}
    complete = len(awa) == ARM_SEEDS["AWA-2L"]
    for key, arm in comparators.items():
        comp = _arm_runs(runs, arm)
        if not awa or not comp:
            out["contrasts"][key] = {"comparator": arm, "missing": True}
            complete = False
            continue
        contrast = decide_pm_contrast(awa, comp)
        contrast["comparator"] = arm
        out["contrasts"][key] = contrast
        complete = complete and len(comp) == ARM_SEEDS[arm]
    if not complete:
        out["decision"] = "INCOMPLETE"
        return out
    if not g0["passed"]:
        out["decision"] = "NOT TESTED"                            # reported as a learnability result
        return out
    contrasts = out["contrasts"]
    ya = contrasts["a"]["Y_awa"]
    successes = sum(y >= Y_SUCCESS for y in ya)
    # Refutation (PROTOCOL 3.2) has two sources. The AWA-only clause involves no comparator. "HL < 0.15" can
    # only come from a contrast whose comparator runs are all ID-competent (PROTOCOL 12.3).
    refuted_by = []
    if successes <= PM_REFUTE_MAX_SUCCESSES and stats.median(ya) < PM_REFUTE_MEDIAN:
        refuted_by.append("AWA endpoint: at most 1 seed with Y >= 0.80 and median Y < 0.50")
    refuted_by += [f"contrast ({key}): Hodges-Lehmann estimate < {PM_SHIFT}" for key, c in sorted(contrasts.items())
                   if not c["inconclusive"] and c["hodges_lehmann"]["estimate"] < PM_SHIFT]
    out["inconclusive_contrasts"] = sorted(key for key, c in contrasts.items() if c["inconclusive"])
    out["refuted_by"] = refuted_by
    if all(c["passes"] and not c["inconclusive"] for c in contrasts.values()):
        out["decision"] = "CONFIRMED"
    elif refuted_by:
        out["decision"] = "REFUTED"
    else:
        out["decision"] = "INCONCLUSIVE"
    return out


def per_hop_classifier(runs: Dict[str, dict]) -> dict:
    """PROTOCOL 3.3: per-hop error law, fitted on k in {2, 3}, predicting the endpoint depths."""
    out: Dict[str, Any] = {"per_run": {}}
    for rid, r in sorted(runs.items()):
        ev = r["eval"]
        if ev["forward"] in ("std",):
            continue
        floors = {int(k): v for k, v in ev["floors"].items()}
        fit_acc = {k: acc(ev, k, "shuffled", plateau_budget(ev, k)) for k in (2, 3)}   # K = 24 or best T
        fit = stats.fit_epsilon(fit_acc, floors)
        predicted = stats.mean([stats.predict_accuracy(k, floors[k], fit["eps"]) for k in ENDPOINT_DEPTHS])
        observed = r["summary"]["Y_plateau"]
        all_depths = [int(k) for k in ev["test"] if int(k) >= 2]
        plateau = plateau_budgets(ev, all_depths)
        fit_all = stats.fit_epsilon({k: acc(ev, k, "shuffled", plateau[k]) for k in all_depths}, floors)
        out["per_run"][rid] = {
            "eps_id": fit["eps"], "predicted_endpoint": predicted, "observed_plateau": observed,
            "label": stats.classify_per_hop(r["summary"]["Y"], observed, predicted), "eps_all_depths": fit_all["eps"],
        }
    labels = [v["label"] for rid, v in out["per_run"].items() if runs[rid]["eval"]["arm"] == "AWA-2L"]
    out["awa_labels"] = labels
    out["study_label"] = stats.majority(labels) if labels else None
    return out


def decide_pt_arm(arm_runs: List[dict]) -> dict:
    """PROTOCOL 3.4: the order law for one lower-triangular arm."""
    if len(arm_runs) < 2:
        return {"status": "not testable", "reason": "fewer than two seeds"}
    evs = [r["eval"] for r in arm_runs]
    floors = {k: evs[0]["floors"][str(k)] for k in PT_DEPTHS}
    follow = {k: evs[0]["followable"][str(k)] for k in PT_DEPTHS}
    sorted_mean = {k: stats.mean([acc(ev, k, "sorted", 0 if ev["forward"] == "exact" else TEST_BUDGET) for ev in evs])
                   for k in PT_DEPTHS}
    informative = [k for k in PT_DEPTHS if sorted_mean[k] >= floors[k] + PT_INFORMATIVE_MARGIN]
    out: Dict[str, Any] = {"sorted_mean": {str(k): v for k, v in sorted_mean.items()}, "informative_depths": informative}
    if 3 not in informative:
        out.update(status="not testable", reason="k = 3 is not informative (sorted accuracy < h_3 + 0.30)")
        return out

    def a(ev, k, stratum):
        return acc(ev, k, stratum, 0 if ev["forward"] == "exact" else TEST_BUDGET)

    excess = {s: [stats.mean([a(ev, k, s) - floors[k] for k in informative]) for ev in evs]
              for s in ("oneswap", "anti")}
    mix_depths = [k for k in (3, 4) if k in informative]
    mixture = [stats.mean([abs(a(ev, k, "shuffled") - (follow[k] * a(ev, k, "sorted")
                                                        + (1.0 - follow[k]) * a(ev, k, "anti")))
                           for k in mix_depths]) for ev in evs]
    upper = {s: stats.t_upper(v) for s, v in excess.items()}
    lower = {s: stats.t_lower(v) for s, v in excess.items()}
    p_refute = min(1.0, 2.0 * min(stats.t_test_greater(v, PT_REFUTE_MARGIN) for v in excess.values()))
    mixture_upper = stats.t_upper(mixture)
    out.update(
        excess_over_floor=excess, upper_bounds=upper, lower_bounds=lower, mixture_error=mixture,
        mixture_upper_bound=mixture_upper, mixture_depths=mix_depths, p_refute=p_refute,
        supported=all(u <= PT_FLOOR_MARGIN for u in upper.values()) and mixture_upper <= PT_MIXTURE_TOL,
        status="tested",
    )
    return out


def decide_pt(runs: Dict[str, dict], g0: dict) -> dict:
    arms_out = {arm: decide_pt_arm(_arm_runs(runs, arm)) for arm in LOWER_TRIANGULAR_ARMS}
    tested = {arm: v for arm, v in arms_out.items() if v["status"] == "tested"}
    out: Dict[str, Any] = {"arms": arms_out, "assumption_B": g0["control"]["status"]}
    if tested:
        adjusted = stats.holm({arm: v["p_refute"] for arm, v in tested.items()})
        for arm, p in adjusted.items():
            arms_out[arm]["p_refute_holm"] = p
            arms_out[arm]["refuted"] = p <= stats.ALPHA
    complete = all(len(_arm_runs(runs, arm)) == ARM_SEEDS[arm] for arm in LOWER_TRIANGULAR_ARMS)
    if not complete:                                              # like P-M: no decision before every seed is in
        out["decision"] = "INCOMPLETE"
    elif any(v.get("refuted") for v in tested.values()):
        out["decision"] = "REFUTED"
    elif len(tested) < len(LOWER_TRIANGULAR_ARMS):
        out["decision"] = "NOT TESTABLE"
    elif all(v["supported"] for v in tested.values()):
        out["decision"] = "SUPPORTED"
    else:
        out["decision"] = "INCONCLUSIVE"
    return out


def decide_h3b(runs: Dict[str, dict]) -> dict:
    """PROTOCOL 3.5: order invariance of AWA."""
    awa = _arm_runs(runs, "AWA-2L")
    if len(awa) < 2:
        return {"decision": "INCOMPLETE"}
    evs = [r["eval"] for r in awa]
    floors = {k: evs[0]["floors"][str(k)] for k in H3B_DEPTHS}
    best = {k: max(stats.mean([acc(ev, k, s, TEST_BUDGET) for ev in evs]) for s in STRATA) for k in H3B_DEPTHS}
    informative = [k for k in H3B_DEPTHS if best[k] >= floors[k] + 0.30]
    out: Dict[str, Any] = {"informative_depths": informative}
    if 4 not in informative:
        out["decision"] = "NOT TESTABLE"
        return out
    gaps = [stats.mean([max(acc(ev, k, s, TEST_BUDGET) for s in STRATA) - min(acc(ev, k, s, TEST_BUDGET) for s in STRATA)
                        for k in informative]) for ev in evs]
    out.update(gap_per_seed=gaps, mean_gap=stats.mean(gaps), upper_bound=stats.t_upper(gaps))
    if out["mean_gap"] >= 0.15:
        out["decision"] = "REFUTED"
    elif out["mean_gap"] <= 0.05 and out["upper_bound"] <= 0.08:
        out["decision"] = "SUPPORTED"
    else:
        out["decision"] = "INCONCLUSIVE"
    return out


def _k_star(curve: Dict[int, float]) -> Optional[int]:
    """Smallest K whose accuracy is within 0.05 of the plateau (accuracy at K = 24)."""
    plateau = curve[PLATEAU_BUDGET]
    for K in sorted(curve):
        if curve[K] >= plateau - 0.05:
            return K
    return None


def decide_h4(runs: Dict[str, dict]) -> dict:
    """PROTOCOL 3.6: mechanism checks for AWA (dose-response in K, gate ablation, hop trace)."""
    awa = _arm_runs(runs, "AWA-2L")
    if not awa:
        return {"a": {"decision": "INCOMPLETE"}, "b": {"decision": "INCOMPLETE"}, "c": {"decision": "INCOMPLETE"}}
    depths = [k for k in (2, 3, 4, 5, 6, 8) if str(k) in awa[0]["eval"]["test"]]
    # (a) K*(k) - k is the same integer for every depth with plateau >= 0.5; a sharp step at K*
    per_seed, varying, shortcut = {}, 0, False
    for r in awa:
        ev = r["eval"]
        offsets, sharp = {}, True
        for k in depths:
            curve = {K: mean_over_strata(ev, k, K) for K in ev["budgets"]}
            if curve[PLATEAU_BUDGET] < 0.5:
                continue
            K_star = _k_star(curve)
            offsets[str(k)] = K_star - k
            if K_star >= 1 and curve[K_star - 1] > curve[K_star] - 0.30:
                sharp = False
            if K_star < k - 1:
                shortcut = True
        spread = (max(offsets.values()) - min(offsets.values())) if offsets else None
        per_seed[str(r["meta"]["seed"])] = {"offsets": offsets, "spread": spread, "sharp_step": sharp,
                                            "supported": bool(offsets) and len(offsets) >= 2 and spread == 0 and sharp}
        varying += bool(offsets) and spread is not None and spread > 1
    a: Dict[str, Any] = {"per_seed": per_seed, "seeds_varying_by_more_than_1": varying, "shortcut": shortcut}
    if varying >= 2 or shortcut:
        a["decision"] = "REFUTED"
    elif all(v["supported"] for v in per_seed.values()):
        a["decision"] = "SUPPORTED"
    else:
        a["decision"] = "INCONCLUSIVE"
    # (b) lambda := 0 at test brings k >= 2 accuracy to <= STD-2L-pm + 0.05
    std = _arm_runs(runs, "STD-2L-pm")
    if not std:
        b: Dict[str, Any] = {"decision": "INCOMPLETE"}
    else:
        diffs = {}
        for k in [int(x) for x in awa[0]["eval"]["test"]]:
            ablated = stats.mean([stats.mean([r["eval"]["zero_operator"][str(k)][s]["correct"]
                                              / r["eval"]["zero_operator"][str(k)][s]["n"] for s in STRATA]) for r in awa])
            control = stats.mean([mean_over_strata(r["eval"], k, 0) for r in std])
            diffs[str(k)] = ablated - control
        worst = max(v for k, v in diffs.items() if int(k) >= 2)
        b = {"ablated_minus_std": diffs, "worst_k_ge_2": worst, "decision": "SUPPORTED" if worst <= 0.05 else "REFUTED"}
    # (c) hop trace: per seed, (head, delta) chosen on validation; test medians >= 0.8 for every hop m <= k - 1
    c_seeds = {}
    for r in awa:
        tr = r["eval"].get("traces")
        if not tr:
            continue
        best, best_value = None, -1.0
        n_heads = len(next(iter(tr["val"].values()))["pooled"]["mean"])
        for head in range(n_heads):
            for di in range(len(tr["deltas"])):
                cells = [v for k in tr["val"] for v in tr["val"][k]["pooled"]["mean"][head][di] if v is not None]
                value = stats.mean(cells) if cells else -1.0
                if value > best_value + 1e-12:
                    best, best_value = (head, di), value
        head, di = best
        medians = {k: [v for v in tr["test"][k]["pooled"]["median"][head][di] if v is not None] for k in tr["test"]}
        lowest = min(min(v) for v in medians.values() if v)
        c_seeds[str(r["meta"]["seed"])] = {"head": head, "delta": tr["deltas"][di], "validation_mean": best_value,
                                           "test_medians": medians, "lowest_median": lowest,
                                           "supported": lowest >= 0.8}
    if not c_seeds:
        c: Dict[str, Any] = {"decision": "INCOMPLETE"}
    else:
        c = {"per_seed": c_seeds,
             "decision": "SUPPORTED" if all(v["supported"] for v in c_seeds.values()) else "NOT SUPPORTED"}
    return {"a": a, "b": b, "c": c}


def decide_h5(runs: Dict[str, dict]) -> dict:
    """PROTOCOL 3.7: no overthinking. Seed-mean accuracy for K from K*(k) to min(3k, 24) stays near its maximum.

    Only depths whose seed-mean plateau accuracy (K = 24) is at least 0.5 are tested: a flat curve at the chance
    floor would satisfy the rule vacuously."""
    awa = _arm_runs(runs, "AWA-2L")
    if not awa:
        return {"decision": "INCOMPLETE"}
    drops = {}
    for k in (3, 4, 6, 8):
        if str(k) not in awa[0]["eval"]["test"]:
            continue
        curve = {K: stats.mean([mean_over_strata(r["eval"], k, K) for r in awa]) for K in awa[0]["eval"]["budgets"]}
        if curve[PLATEAU_BUDGET] < 0.5:
            continue
        K_star = _k_star(curve)
        window = [curve[K] for K in range(K_star, min(3 * k, PLATEAU_BUDGET) + 1)]
        drops[str(k)] = {"K_star": K_star, "max": max(window), "min": min(window), "drop": max(window) - min(window)}
    if not drops:
        return {"per_depth": drops, "decision": "NOT TESTABLE"}
    worst = max(v["drop"] for v in drops.values())
    decision = "SUPPORTED" if worst <= 0.02 else "REFUTED" if worst >= 0.05 else "INCONCLUSIVE"
    return {"per_depth": drops, "worst_drop": worst, "decision": decision}


def _matched_cost(ev: dict, target: float) -> Optional[dict]:
    """Smallest budget whose worst-order endpoint accuracy reaches ``target``; cost = MACs / accuracy there."""
    for b in ev["budgets"]:
        value = worst_order(ev, {k: b for k in ENDPOINT_DEPTHS})
        if value >= target:
            return {"budget": b, "accuracy": value, "macs": ev["macs"][str(b)], "cost": ev["macs"][str(b)] / value}
    return None


def h6_reading(arm: str, entry: dict) -> str:
    """PROTOCOL 3.8 "Reading" for one baseline: (i) decides the capability claim when it is significant (Holm)
    with |HL| >= the SESOI; otherwise (ii) decides between the two cost readings and "equivalent".

    Choices listed in PROTOCOL 12.9: a baseline with fewer than its planned seeds is "descriptive only"; when
    (i) does not decide and (ii) has no cost estimate, the reading is "not decided" with the reason of (ii)."""
    if entry.get("missing"):
        return "INCOMPLETE"
    if entry["descriptive_only"]:
        return "descriptive only (fewer baseline seeds than planned)"
    if entry["capability"] == "AWA beats the baseline":
        return f"AWA beats {arm}"
    if entry["capability"] == "the baseline beats AWA":
        return f"{arm} beats AWA"
    cost = entry["cost_at_matched_accuracy"]["reading"]
    if cost == "AWA cheaper":
        return f"AWA matches {arm} at >= 2x lower cost"
    if cost == f"{arm} cheaper":
        return f"{arm} matches AWA at >= 2x lower cost"
    if cost == "comparable":
        return "equivalent"
    return f"not decided ({cost})"


def decide_h6(runs: Dict[str, dict]) -> dict:
    """PROTOCOL 3.8: is recurrence enough? Capability (exact Mann-Whitney, Holm), cost at matched accuracy, and
    the reading of the two per baseline. "decision" is INCOMPLETE until every AWA seed and at least one run of
    each baseline are in; otherwise it lists the two readings (no direction is predicted)."""
    awa = _arm_runs(runs, "AWA-2L")
    out: Dict[str, Any] = {"baselines": {}}
    if not awa:
        out["decision"] = "INCOMPLETE"
        return out
    ya = [r["summary"]["Y"] for r in awa]
    ya_plateau = [r["summary"]["Y_plateau"] for r in awa]
    pvals = {}
    for arm in LOOP_ARMS:
        base = _arm_runs(runs, arm)
        if not base:
            out["baselines"][arm] = {"missing": True}
            continue
        y_oracle = [r["summary"]["Y_plateau"] for r in base]              # loops: oracle T per depth
        y_fixed = [r["summary"]["Y"] for r in base]                       # T = 12
        mw = stats.mann_whitney_exact(ya, y_oracle, 0.0, "two-sided")
        hl = stats.hodges_lehmann(ya, y_oracle, stats.ALPHA, two_sided=True)
        entry: Dict[str, Any] = {"Y_awa": ya, "Y_baseline_oracle_T": y_oracle, "Y_baseline_T12": y_fixed,
                                 "mann_whitney": mw, "hodges_lehmann": hl,
                                 "descriptive_only": len(base) < ARM_SEEDS[arm]}
        pvals[arm] = mw["p"]
        target = 0.9 * min(stats.median(ya_plateau), stats.median(y_oracle))
        chance = stats.mean([awa[0]["eval"]["floors"][str(k)] for k in ENDPOINT_DEPTHS])
        cost_a = [_matched_cost(r["eval"], target) if target > chance else None for r in awa]
        cost_b = [_matched_cost(r["eval"], target) if target > chance else None for r in base]
        cost: Dict[str, Any] = {"target": target, "awa": cost_a, "baseline": cost_b}
        miss_a, miss_b = sum(c is None for c in cost_a), sum(c is None for c in cost_b)
        if target <= chance:
            cost["reading"] = "not testable: the matched accuracy is not above the chance floor"
        elif miss_a >= 3 or miss_b >= 3:
            cost["reading"] = ("AWA does not reach matched accuracy" if miss_a >= 3 else
                               f"{arm} does not reach matched accuracy")
        elif miss_a == len(cost_a) or miss_b == len(cost_b):
            cost["reading"] = "no seed of one arm reaches the target"
        else:
            la = [math.log(c["cost"]) for c in cost_a if c is not None]
            lb = [math.log(c["cost"]) for c in cost_b if c is not None]
            hl_cost = stats.hodges_lehmann(lb, la, stats.ALPHA, two_sided=True)   # log(C_B / C_AWA)
            cost["log_cost_ratio"] = hl_cost
            excludes_zero = hl_cost["lower"] > 0.0 or hl_cost["upper"] < 0.0
            if hl_cost["estimate"] >= math.log(2.0) and excludes_zero:
                cost["reading"] = "AWA cheaper"
            elif hl_cost["estimate"] <= -math.log(2.0) and excludes_zero:
                cost["reading"] = f"{arm} cheaper"
            else:
                cost["reading"] = "comparable"
        entry["cost_at_matched_accuracy"] = cost
        out["baselines"][arm] = entry
    adjusted = stats.holm(pvals) if pvals else {}
    for arm, p in adjusted.items():
        entry = out["baselines"][arm]
        entry["p_holm"] = p
        estimate = entry["hodges_lehmann"]["estimate"]
        significant = p <= stats.ALPHA and not entry["descriptive_only"]
        if significant and estimate >= H6_SESOI:
            entry["capability"] = "AWA beats the baseline"
        elif significant and estimate <= -H6_SESOI:
            entry["capability"] = "the baseline beats AWA"
        else:
            entry["capability"] = "no decided capability difference"
    awa_complete = len(awa) == ARM_SEEDS["AWA-2L"]
    for arm, entry in out["baselines"].items():
        entry["reading"] = h6_reading(arm, entry) if awa_complete else "INCOMPLETE"
    complete = awa_complete and not any(e.get("missing") for e in out["baselines"].values())
    out["decision"] = "; ".join(f"{arm}: {out['baselines'][arm]['reading']}" for arm in LOOP_ARMS) if complete \
        else "INCOMPLETE"
    return out


def analysis_code_sha256(study: pathlib.Path = STUDY) -> Dict[str, str]:
    return {name: hashlib.sha256((study / name).read_bytes()).hexdigest() for name in ANALYSIS_FILES}


def compute_decisions(runs: Dict[str, dict], *, allow_truncated: bool = False) -> dict:
    """All decisions from ``runs`` = {run_id: {"meta": ..., "eval": ...}}. Pure: no files, no randomness."""
    runs = {rid: dict(r) for rid, r in runs.items()}
    truncated = sorted(rid for rid, r in runs.items() if r["eval"].get("truncated_to") is not None)
    if truncated and not allow_truncated:
        raise ValueError(f"truncated evaluation records (smoke runs) cannot enter the analysis: {truncated}")
    for r in runs.values():
        if r["eval"]["arm"] != r["meta"]["arm"]:
            raise ValueError("eval record and meta disagree about the arm")
        r["summary"] = per_run_summary(r["eval"], r["meta"])
    expected = [run_id(arm, s) for arm, n in ARM_SEEDS.items() for s in range(n)]
    g0 = decide_g0(runs)
    out = {
        "format": DECISIONS_FORMAT,
        "runs": {"present": sorted(runs), "missing": [rid for rid in expected if rid not in runs],
                 "truncated": truncated},
        "per_run": {rid: r["summary"] for rid, r in sorted(runs.items())},
        "G0": g0,
        "P-M": decide_pm(runs, g0),
        "per_hop_classifier": per_hop_classifier(runs),
        "P-T": decide_pt(runs, g0),
        "H3b": decide_h3b(runs),
        "H4": decide_h4(runs),
        "H5": decide_h5(runs),
        "H6": decide_h6(runs),
    }
    return json.loads(json.dumps(out, allow_nan=True))


# -------------------------------------------------------------------------------------------- files
def load_runs(phase_dir: pathlib.Path) -> Dict[str, dict]:
    runs = {}
    for meta_path in sorted((phase_dir / "runs").glob("*/meta.json")):
        eval_path = meta_path.with_name("eval.json")
        if not eval_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("finished"):
            continue
        runs[meta_path.parent.name] = {"meta": meta, "eval": json.loads(eval_path.read_text(encoding="utf-8"))}
    return runs


def render_summary(decisions: dict, runs: Dict[str, dict]) -> str:
    lines = ["# AWA-3 summary tables (generated by run_analyze.py)", ""]
    h6 = decisions["H6"]
    h6_rows = [f"| H6 vs {arm} | {entry['reading']} |" for arm, entry in h6["baselines"].items()] \
        if h6["decision"] != "INCOMPLETE" else ["| H6 | INCOMPLETE |"]
    lines += ["## Decisions", "", "| Item | Decision |", "|---|---|",
              f"| G0 learnability gate | {'passed' if decisions['G0']['passed'] else 'not passed'} "
              f"({decisions['G0']['awa_competent']}/{decisions['G0']['awa_runs']} AWA seeds competent); "
              f"control: {decisions['G0']['control']['status']} |",
              f"| P-M | {decisions['P-M']['decision']} |",
              f"| P-T | {decisions['P-T']['decision']} |",
              f"| H3b | {decisions['H3b']['decision']} |",
              f"| H4(a) / (b) / (c) | {decisions['H4']['a']['decision']} / {decisions['H4']['b']['decision']} / "
              f"{decisions['H4']['c']['decision']} |",
              f"| H5 | {decisions['H5']['decision']} |", *h6_rows,
              f"| per-hop label (AWA) | {decisions['per_hop_classifier']['study_label']} |", ""]
    lines += ["## Per run", "", "| Run | steps | competent | Y | Y chance-corrected | Y plateau |", "|---|---|---|---|---|---|"]
    for rid, s in decisions["per_run"].items():
        lines.append(f"| {rid} | {s['steps']} | {'yes' if s['competent'] else 'no'} | {s['Y']:.3f} | "
                     f"{s['Y_chance_corrected']:.3f} | {s['Y_plateau']:.3f} |")
    lines += ["", "## Seed-median accuracy at the preregistered budget (K = T = 12; exact arms: no budget)", ""]
    arms = sorted({r["eval"]["arm"] for r in runs.values()}, key=list(ARM_SEEDS).index)
    for arm in arms:
        evs = [r["eval"] for r in _arm_runs(runs, arm)]
        depths = sorted(int(k) for k in evs[0]["test"])
        lines += [f"### {arm} ({len(evs)} seeds)", "", "| k | h_k | " + " | ".join(STRATA) + " |",
                  "|---|---|" + "---|" * len(STRATA)]
        for k in depths:
            cells = [f"{stats.median([acc(ev, k, s, TEST_BUDGET) for ev in evs]):.3f}" for s in STRATA]
            lines.append(f"| {k} | {evs[0]['floors'][str(k)]:.3f} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def write_outputs(phase_dir: pathlib.Path, decisions: dict, runs: Dict[str, dict]) -> None:
    decisions = dict(decisions)
    decisions["analysis_code_sha256"] = analysis_code_sha256()
    for path, text in ((phase_dir / "decisions.json", json.dumps(decisions, indent=1, sort_keys=True) + "\n"),
                       (phase_dir / "tables" / "summary.md", render_summary(decisions, runs) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(STUDY / "results"))
    ap.add_argument("--phase", default="1")
    ap.add_argument("--allow-truncated", action="store_true",
                    help="accept smoke-run records (truncated test sets); never for confirmatory results")
    args = ap.parse_args(argv)
    phase_dir = pathlib.Path(args.results) / f"phase{args.phase}"
    runs = load_runs(phase_dir)
    if not runs:
        print(f"no evaluated runs under {phase_dir / 'runs'}", file=sys.stderr)
        return 1
    try:
        decisions = compute_decisions(runs, allow_truncated=args.allow_truncated)
    except ValueError as exc:
        print(f"analysis refused: {exc}", file=sys.stderr)
        return 1
    write_outputs(phase_dir, decisions, runs)
    print(f"{len(runs)} runs; P-M {decisions['P-M']['decision']}; P-T {decisions['P-T']['decision']}; "
          f"wrote {phase_dir / 'decisions.json'}")
    return 0

