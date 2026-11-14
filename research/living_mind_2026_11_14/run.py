# -*- coding: utf-8 -*-
"""Live one life in a generated world and record the developmental curves T1, T2, T3, T5, T9 (JSONL).

    python research/living_mind_2026_11_14/run.py --seeds 0 1 2 3 4 --domains nitelik iliski \
        --variants as_written changed --n 600 --out research/living_mind_2026_11_14/results

One file per (domain, seed, variant). Every line is one record: ``{"test": ..., "at": experiences lived, ...}``.
Probes never touch the life: they run on a clone, by presenting a prefix and undoing it (black box), or by
reading the evidence on the relevant ties (white box). Scoring uses the world's truth tables; the mind never
sees them. Wall-clock time per moment is measured around ``Mind.step`` and is not part of any state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from world import World  # noqa: E402
from zihin import IMITATED, SENSED, TOLD, Config, Event, Mind  # noqa: E402
from zihin import read  # noqa: E402
from zihin.evidence import expectation  # noqa: E402
from zihin.store import ABSTRACT, ATOMIC, CATEGORY, EXPECT, W_NEG, W_POS  # noqa: E402

PROV = {"told": TOLD, "sensed": SENSED, "imitated": IMITATED}
VARIANTS = {"as_written": lambda **kw: Config(**kw), "changed": lambda **kw: Config.changed(**kw),
            "symmetric": lambda **kw: Config.symmetric(**kw),
            "symmetric_pooled": lambda **kw: Config.symmetric(**dict(dict(support="capped"), **kw)),
            "reliable": lambda **kw: Config.reliable(**kw),
            "reflective": lambda **kw: Config.reflective(**kw),
            "honest_reflective": lambda **kw: Config.reflective(**dict(dict(miss_scope="gens"), **kw)),
            "honest_trusted": lambda **kw: Config.reflective(**dict(dict(miss_scope="gens", support="trusted"), **kw)),
            "members": lambda **kw: Config.changed(**dict(dict(support="members"), **kw)),
            "sleeping": lambda **kw: Config.reflective(**dict(dict(reflect=8), **kw)),
            "trusted": lambda **kw: Config.reflective(**dict(dict(support="trusted"), **kw)),
            "symmetric_choice": lambda **kw: Config.symmetric(**dict(dict(support="choice"), **kw))}
CHANNEL = "text"


def utter(mind: Mind, symbols: List[str], prov: int, pause: int = 1) -> List[int]:
    """Present an utterance, one symbol per moment, then silence. Returns the moment ids of the symbols."""
    moments = []
    for symbol in symbols:
        mind.step([Event(CHANNEL, symbol, prov)])
        moments.append(mind.t)
    for _ in range(pause):
        mind.step([])
    return moments


class Probe:
    """Black-box and white-box readings on a clone of the mind."""

    def __init__(self, mind: Mind) -> None:
        self.twin = mind.clone()
        self.twin.cfg.undo_depth = 16
        self.mode = mind.cfg.predict

    def ask(self, prefix: List[str]) -> Dict[str, float]:
        """Present a prefix, read the expectations for the next symbol, undo the presentation."""
        twin = self.twin
        for symbol in prefix:
            twin.step([Event(CHANNEL, symbol, TOLD)])
        out = dict(twin.expected(CHANNEL, self.mode))
        for _ in prefix:
            twin.undo_last()
        return out

    def tie(self, prefix: List[str], value: str) -> Tuple[int, int, int]:
        """(context unit, target unit, tie id) of the belief "prefix -> value"; -1 where it does not exist."""
        twin = self.twin
        units = tuple(twin.unit_of(CHANNEL, s) for s in prefix)
        target = twin.unit_of(CHANNEL, value)
        context = twin.S.tuple_index.get(units, -1) if min(units) >= 0 else -1
        tie = twin.S.find_tie(context, target, EXPECT, 1) if context >= 0 and target >= 0 else -1
        return context, target, tie

    def belief(self, prefix: List[str], value: str) -> Dict[str, float]:
        context, target, _ = self.tie(prefix, value)
        if context < 0 or target < 0:
            return {"own_pos": 0.0, "own_neg": 0.0, "own_c": 0.0, "own_e": 0.5, "eff_c": 0.0, "eff_e": 0.5,
                    "own_f": 0.5, "eff_f": 0.5, "eff_pos": 0.0, "eff_neg": 0.0}
        return read.belief(self.twin, context, target)

    def raw(self, prefix: List[str], value: str) -> Optional[List[float]]:
        """Raw evidence rows (w+ and w- per class) of the belief tie, for bit-exact comparisons."""
        _, _, tie = self.tie(prefix, value)
        if tie < 0:
            return None
        return self.twin.S.t_w[:2, tie].flatten().tolist()


def score(expected: Dict[str, float], values: List[str], right: str) -> Dict[str, float]:
    """Is the right value the top expectation among the values of the dimension, and above 1/2?"""
    e = {v: expected.get(v, 0.5) for v in values}
    best_other = max(e[v] for v in values if v != right)
    return {"e": e[right], "other": best_other, "ok": float(e[right] > 0.5 and e[right] > best_other)}


def categories_found(mind: Mind) -> Dict[int, set]:
    """Emergent categories: category unit -> symbols of its fillers (ABSTRACT expectation above 1/2)."""
    S = mind.S
    out: Dict[int, set] = {}
    for (dst, form), ties in S.inc.items():
        if form != ABSTRACT or not int(S.u_kind[dst]) & CATEGORY:
            continue
        fillers = set()
        for tie in ties:
            w_pos = float((S.t_w[W_POS, tie] * mind.trust).sum())
            w_neg = float((S.t_w[W_NEG, tie] * mind.trust).sum())
            src = int(S.t_src[tie])
            if expectation(w_pos, w_neg, mind.cfg.k) > 0.5 and int(S.u_kind[src]) & ATOMIC:
                fillers.add(mind.symbol_of(src)[1])
        out[dst] = fillers
    return out


def best_jaccard(found: Dict[int, set], truth: set) -> float:
    best = 0.0
    for fillers in found.values():
        union = len(fillers | truth)
        if union:
            best = max(best, len(fillers & truth) / union)
    return best


def checkpoint(world: World, mind: Mind, state: dict, at: int) -> List[dict]:
    """All probes at one point of the life."""
    probe = Probe(mind)
    base = {"at": at, "moments": mind.t}
    rows: List[dict] = []
    # ---- T1: conviction of told facts by number of hearings (white box)
    groups: Dict[str, List[float]] = {}
    for (ent, dim), heard in state["heard"].items():
        if heard["lie"] or heard["sensed"]:
            continue
        b = probe.belief(world.prefix(ent, dim), world.value[(ent, dim)])
        key = str(min(heard["told"], 4))
        groups.setdefault("c%s" % key, []).append(b["own_c"])
        groups.setdefault("e%s" % key, []).append(b["own_e"])
    row = dict(base, test="T1")
    for key, vals in sorted(groups.items()):
        row[key] = sum(vals) / len(vals)
        row["n" + key[1:]] = len(vals)
    row["trust_told"] = float(mind.trust[TOLD])
    rows.append(row)
    # ---- T2: emergent categories against the world's classes
    found = categories_found(mind)
    members = {c: {e for e in world.entities if world.cat_of[e] == c} for c in world.categories}
    cat_j = [best_jaccard(found, m) for m in members.values()]
    val_j = [best_jaccard(found, set(world.values[d])) for d in world.dims]
    S = mind.S
    kinds = S.u_kind[:S.n_units]
    rows.append(dict(base, test="T2", cat_jaccard=sum(cat_j) / len(cat_j), cat_jaccard_min=min(cat_j),
                     value_jaccard=sum(val_j) / len(val_j), entity_jaccard=best_jaccard(found, set(world.entities)),
                     dim_jaccard=best_jaccard(found, set(world.dims)), categories=len(found),
                     frames=int((kinds & 4).bool().sum()), composed=int((kinds & 2).bool().sum())))
    # ---- T3: support from category to held-out instance; exceptions; ordinary members
    t3: Dict[str, List[dict]] = {"held_told_rule": [], "held_induced": [], "exception": [], "member": []}
    for ent in world.held_out:
        if not state["isa"].get(ent):
            continue
        for dim in world.ruled:
            s = score(probe.ask(world.prefix(ent, dim)), world.values[dim], world.rule[(world.cat_of[ent], dim)])
            b = probe.belief(world.prefix(ent, dim), world.rule[(world.cat_of[ent], dim)])
            s["own_c"], s["eff_e"] = b["own_c"], b["eff_e"]
            t3["held_told_rule" if dim == world.told_rule_dim else "held_induced"].append(s)
    for ent, dim in world.exceptions:
        if not state["heard"].get((ent, dim)) or not state["isa"].get(ent):
            continue
        expected = probe.ask(world.prefix(ent, dim))
        s = score(expected, world.values[dim], world.value[(ent, dim)])
        s["rule_e"] = expected.get(world.rule[(world.cat_of[ent], dim)], 0.5)
        s["kept"] = float(s["e"] > s["rule_e"])
        t3["exception"].append(s)
    known = [(e, d) for (e, d), h in state["heard"].items() if d in world.ruled and (e, d) not in world.exceptions
             and not h["lie"]]
    for ent, dim in known[:40]:
        t3["member"].append(score(probe.ask(world.prefix(ent, dim)), world.values[dim], world.value[(ent, dim)]))
    row = dict(base, test="T3")
    for name, items in t3.items():
        row[name + "_n"] = len(items)
        if items:
            for key in items[0]:
                row["%s_%s" % (name, key)] = sum(i[key] for i in items) / len(items)
    rows.append(row)
    # ---- T5: retention of facts told once early; integrity; size and time
    kept, same, n_early = [], [], 0
    for (ent, dim), first in state["early"].items():
        n_early += 1
        prefix, value = world.prefix(ent, dim), world.value[(ent, dim)]
        kept.append(score(probe.ask(prefix), world.values[dim], value)["ok"])
        same.append(float(probe.raw(prefix, value) == first))
    size = mind.size()
    window = state["times"][state["time_mark"]:]
    state["time_mark"] = len(state["times"])
    finite = bool(torch.isfinite(S.t_w[:, :S.n_ties]).all())
    rows.append(dict(base, test="T5", early_n=n_early, early_recall=sum(kept) / len(kept) if kept else None,
                     early_evidence_identical=sum(same) / len(same) if same else None, finite=finite,
                     exact=S.exact(), max_evidence=S.max_evidence(), units=size["units"], ties=size["ties"],
                     core_bytes=size["core_tensors"], log_bytes=size["log"],
                     ms_per_moment=1000.0 * sum(window) / len(window) if window else None,
                     ms_per_moment_max=1000.0 * max(window) if window else None))
    return rows


def live(world: World, variant: str, n: int, checkpoints: int = 10, skip: Optional[int] = None,
         record: bool = True, night_every: int = 0, night_length: int = 0, mind: Optional[Mind] = None,
         **over) -> Tuple[Mind, List[dict], dict]:
    """Live the schedule. ``skip`` leaves one experience out (the counterfactual life of T9).

    Sleep (LAYERS.md L7): after every ``night_every`` experiences the world is silent for ``night_length``
    moments; the mind's idle processes (reflection, F10) run on what it has lived. Nothing is presented at night.
    """
    if mind is None:                                   # a new mind; otherwise the same mind lives on (T6)
        mind = Mind(VARIANTS[variant](**over))
        mind.add_channel(CHANNEL, "act")
    schedule = world.schedule(n)
    state = {"heard": {}, "isa": {}, "early": {}, "times": [], "time_mark": 0, "moments": {}, "lie_value": {}}
    rows: List[dict] = []
    every = max(1, len(schedule) // checkpoints)
    for index, x in enumerate(schedule):
        if night_every and index and index % night_every == 0:
            for _ in range(night_length):
                mind.step([])
        if index == skip:
            continue
        before = None
        if record and x.tag == "correction":
            before = Probe(mind).raw(world.prefix(x.ent, x.dim), state["lie_value"][(x.ent, x.dim)])
        start = time.perf_counter()
        moments = utter(mind, x.symbols, PROV[x.prov])
        state["times"] += [(time.perf_counter() - start) / (len(x.symbols) + 1)] * (len(x.symbols) + 1)
        state["moments"][index] = moments
        if x.kind == "isa":
            state["isa"][x.ent] = state["isa"].get(x.ent, 0) + 1
        elif x.kind == "fact":
            heard = state["heard"].setdefault((x.ent, x.dim), {"told": 0, "sensed": 0, "lie": 0})
            heard["lie" if x.tag == "lie" else x.prov] += 1
            if x.tag == "lie":
                state["lie_value"][(x.ent, x.dim)] = x.val
            if x.tag == "early" and record:
                state["early"][(x.ent, x.dim)] = Probe(mind).raw(world.prefix(x.ent, x.dim), x.val)
            if record and x.tag in ("lie", "correction"):          # T1 in the stream: refinement without deletion
                probe = Probe(mind)
                prefix, wrong = world.prefix(x.ent, x.dim), state["lie_value"][(x.ent, x.dim)]
                false_b, true_b = probe.belief(prefix, wrong), probe.belief(prefix, world.value[(x.ent, x.dim)])
                after = probe.raw(prefix, wrong)
                rows.append({"test": "T1_lie", "at": index + 1, "step": x.tag, "prov": x.prov, "fact": [x.ent, x.dim],
                             "false_e": false_b["own_e"], "false_c": false_b["own_c"], "false_pos": false_b["own_pos"],
                             "false_neg": false_b["own_neg"], "true_e": true_b["own_e"], "true_c": true_b["own_c"],
                             "positive_untouched": None if before is None or after is None
                             else before[:len(before) // 2] == after[:len(after) // 2],
                             "trust_told": float(mind.trust[TOLD])})
        if record and ((index + 1) % every == 0 or index + 1 == len(schedule)):
            rows += checkpoint(world, mind, state, index + 1)
    state["schedule"] = schedule
    return mind, rows, state


def retraction(world: World, variant: str, n: int, mind: Mind, state: dict, **over) -> dict:
    """T9: retract the single moment in which a once-told fact was learned; compare with never having lived it."""
    schedule = state["schedule"]
    index = next(i for i, x in enumerate(schedule) if x.tag == "early")
    x = schedule[index]
    moment = state["moments"][index][-1]                         # the moment in which the last symbol arrived
    probe = Probe(mind)
    prefix = world.prefix(x.ent, x.dim)
    context, target, tie = probe.tie(prefix, x.val)
    others = {k: probe.raw(world.prefix(*k), world.value[k]) for k in state["early"] if k != (x.ent, x.dim)}
    S = mind.S
    before = S.t_w[:, :S.n_ties].clone()
    touched = torch.zeros(S.n_ties, dtype=torch.bool)
    for _, _, ids, _ in S.log[moment].ev:
        touched[ids] = True
    own_before = probe.belief(prefix, x.val)
    report = mind.retract(moment)
    after = S.t_w[:, :S.n_ties]
    ledger = torch.zeros_like(after)
    for m, entry in S.log.items():
        if not entry.retracted:
            for table, slot, ids, amounts in entry.ev:
                ledger[table, :, slot].index_add_(0, ids, amounts)
    probe2 = Probe(mind)
    own_after = probe2.belief(prefix, x.val)
    out = {"test": "T9", "at": len(schedule), "moment": moment, "fact": [x.ent, x.dim, x.val],
           "increments": report["increments"], "ties_touched": int(touched.sum()),
           "ties_removed": report["ties_removed"], "units_removed": report["units_removed"],
           "own_c_before": own_before["own_c"], "own_e_before": own_before["own_e"],
           "own_evidence_after": float(after[:2, tie].abs().sum()) if tie >= 0 else 0.0,
           "own_e_after": own_after["own_e"],
           "untouched_ties_identical": bool(torch.equal(after[:, ~touched], before[:, ~touched])),
           "equals_ledger_of_other_moments": bool(torch.equal(after, ledger)),
           "other_early_facts_identical": all(probe2.raw(world.prefix(*k), world.value[k]) == v
                                              for k, v in others.items())}
    # counterfactual life: the same stream without that experience
    twin, _, _ = live(world, variant, n, skip=index, record=False, **over)
    probe3 = Probe(twin)
    gaps, own_gaps = [], []
    for (ent, dim) in list(state["heard"])[:200]:
        value = world.value[(ent, dim)]
        a, b = probe2.belief(world.prefix(ent, dim), value), probe3.belief(world.prefix(ent, dim), value)
        gaps.append(abs(a["eff_e"] - b["eff_e"]))
        own_gaps.append(abs(a["own_e"] - b["own_e"]))
    out.update(counterfactual_own_e_after=probe3.belief(prefix, x.val)["own_e"],
               counterfactual_gap_max=max(gaps), counterfactual_gap_mean=sum(gaps) / len(gaps),
               counterfactual_own_gap_max=max(own_gaps), units_vs_counterfactual=S.n_units - twin.S.n_units,
               ties_vs_counterfactual=S.n_ties - twin.S.n_ties)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--domains", nargs="+", default=["nitelik", "iliski"])
    parser.add_argument("--variants", nargs="+", default=["as_written", "changed"])
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--out", default=os.path.join(HERE, "results"))
    parser.add_argument("--no-t9", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.makedirs(args.out, exist_ok=True)
    for domain in args.domains:
        for seed in args.seeds:
            for variant in args.variants:
                world = World.make(seed, domain)
                start = time.time()
                mind, rows, state = live(world, variant, args.n)
                if not args.no_t9:
                    rows.append(retraction(world, variant, args.n, mind, state))
                head = {"seed": seed, "domain": domain, "variant": variant, "n": args.n}
                path = os.path.join(args.out, "life_%s_%d_%s.jsonl" % (domain, seed, variant))
                with open(path, "w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(dict(head, **row), ensure_ascii=False) + "\n")
                print("%s seed %d %s: %d experiences, %d moments, %d units, %d ties, %.0f s" % (
                    domain, seed, variant, args.n, mind.t, mind.S.n_units, mind.S.n_ties, time.time() - start),
                    flush=True)


if __name__ == "__main__":
    main()
