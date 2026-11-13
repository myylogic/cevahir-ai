# -*- coding: utf-8 -*-
"""Keşif seçim kuralı (PROTOCOL §8) — yalnız keşif dünyaları 0 ve 1'in ``run="select"`` kayıtlarından.

Kural (kol başına): katman × ızgara yapılandırmaları arasında ortalama yazım etkinliğini (WE) en yükseğe çıkar;
koşul: Leak_RD ≤ %1 ve Leak_C ≤ %5 (C ilişkisi, düzenlenen varlık). Eşitlikte sığ katman, sonra küçük adım (FT: küçük
lr), sonra — protokolün tanımlamadığı ızgara boyutları için, açıkça ve muhafazakâr yönde — küçük cap_mult (FT-L: küçük
eps), en son yapılandırma etiketi. Hiçbir yapılandırma koşulları sağlamazsa en yüksek WE seçilir ve kol
"not installable under constraints" olarak işaretlenir. Harita-yalnız anahtarı: STD'nin SEÇİLEN yapılandırmasının keşif
WE'si %50'nin altındaysa H1–H3 test edilemez sayılır. NORM, STD'nin seçimini kullanır (önişlem).
Bütünlük: bir kolun her (katman, yapılandırma) satırı aynı (dünya, madde) kümesini — iki keşif dünyası 0 ve 1 —
kapsamalıdır ve hiçbir madde iki kez sayılamaz; yarım kalmış/çift koşu seçim yapmadan reddedilir (yalnız betimsel
inceleme için --allow-partial). results/lives/select_*.jsonl yalnız keşif dünyalarının run="select" kayıtlarını
içerebilir. Çıktı: results/selection.json ve results/tables/discovery_map.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict

STUDY = pathlib.Path(__file__).resolve().parent

LEAK_RD_MAX, LEAK_C_MAX, MAP_ONLY_WE = 0.01, 0.05, 0.50
DISCOVERY_WORLDS = (0, 1)


def _num(cfg: str, key: str) -> float:
    for part in cfg.split(","):
        k, _, v = part.partition("=")
        if k == key:
            return float(v)
    return 0.0


def summarise(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["arm"], r["layer"], r["config"])].append(r)
    rows = []
    for (arm, layer, cfg), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1], kv[0][2])):
        n = len(rs)
        we = sum(bool(r["installed"]) for r in rs) / n
        rd = sum(r["leak"]["RD"]["changed"] / max(r["leak"]["RD"]["n"], 1) for r in rs) / n
        rc = sum(r["leak"]["RC"]["changed"] / max(r["leak"]["RC"]["n"], 1) for r in rs) / n
        rn_rs = [r for r in rs if r["leak"].get("RN", {}).get("n", 0) > 0]
        rn = (sum(r["leak"]["RN"]["changed"] / r["leak"]["RN"]["n"] for r in rn_rs) / len(rn_rs)) if rn_rs else None
        pe6 = sum(sum(r["PF6"].values()) / max(len(r["PF6"]), 1) for r in rs) / n
        tg = [r for r in rs if r["targets"]]
        ie5 = (sum(sum(t["informal"] for t in r["targets"]) / len(r["targets"]) for r in tg) / len(tg)) if tg else None
        pe5 = (sum(sum(t["formal"] for t in r["targets"]) / len(r["targets"]) for r in tg) / len(tg)) if tg else None
        worlds = sorted({r["world"] for r in rs})
        rows.append({"arm": arm, "layer": layer, "config": cfg, "n": n, "worlds": worlds, "WE": we, "Leak_RD": rd,
                     "Leak_C": rc, "Leak_RN": rn, "PE6": pe6, "PE5": pe5, "IE5": ie5,
                     "feasible": rd <= LEAK_RD_MAX and rc <= LEAK_C_MAX})
    return rows


def _tie_key(r):
    """Ties: shallower layer, then the smaller step (FT: lr), then the smaller cap_mult (FT-L: eps), then the label.
    Explicit numeric order: the old string order of the labels put cap_mult=16 before cap_mult=8."""
    cfg = r["config"]
    return (-1 if r["layer"] is None else r["layer"], _num(cfg, "step") or _num(cfg, "lr"),
            _num(cfg, "cap_mult") or _num(cfg, "eps"), cfg)


def choose(rows, arm):
    cand = [r for r in rows if r["arm"] == arm]
    if not cand:
        return None
    def key(r):
        return (-r["WE"],) + _tie_key(r)
    feas = [r for r in cand if r["feasible"]]
    pick = sorted(feas or cand, key=key)[0]
    return {**pick, "status": "selected" if feas else "not installable under constraints"}


def completeness_problems(records):
    """Every (layer, config) of an arm must cover the same (world, edit) set, both discovery worlds, no duplicates."""
    cover = defaultdict(list)
    for r in records:
        cover[(r["arm"], r["layer"], r["config"])].append((r["world"], r["edit_id"]))
    problems = []
    by_arm = defaultdict(dict)
    for (arm, layer, cfg), items in sorted(cover.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        if len(set(items)) != len(items):
            problems.append(f"{arm} L{layer} {cfg}: duplicated (world, edit) records")
        by_arm[arm][(layer, cfg)] = frozenset(items)
    for arm, sets in sorted(by_arm.items()):
        union = frozenset().union(*sets.values())
        worlds = sorted({w for w, _ in union})
        if worlds != list(DISCOVERY_WORLDS):
            problems.append(f"{arm}: records from worlds {worlds}, both discovery worlds {DISCOVERY_WORLDS} required")
        for (layer, cfg), s in sorted(sets.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            if s != union:
                problems.append(f"{arm} L{layer} {cfg}: {len(s)} of {len(union)} (world, edit) records — incomplete")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-partial", action="store_true",
                    help="summarise incomplete selection records (descriptive only; selection.json is not written)")
    args = ap.parse_args(argv)
    sel_dir = STUDY / "results" / "select"
    records = []
    for p in sorted(sel_dir.glob("select_w*_*.jsonl")):
        for line in p.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                r = json.loads(line)
                if r["world"] not in DISCOVERY_WORLDS or r["run"] != "select":
                    raise SystemExit(f"non-discovery record in {p}")
                records.append(r)
    selected_arms = ("STD", "ROOT", "LAST", "PARA-ORTH", "FT-L", "FT-ALL")
    problems = completeness_problems([r for r in records if r["arm"] in selected_arms])
    for msg in completeness_problems([r for r in records if r["arm"] not in selected_arms]):
        print(f"[warning] map-only site: {msg}")   # PRED rows only feed the discovery map table
    if problems and not args.allow_partial:
        raise SystemExit("selection records are incomplete or duplicated:\n  " + "\n  ".join(problems))
    rows = summarise(records)
    out = {"rule": {"Leak_RD_max": LEAK_RD_MAX, "Leak_C_max": LEAK_C_MAX, "map_only_if_STD_WE_below": MAP_ONLY_WE,
                    "ties": "shallower layer, smaller step (FT: lr), smaller cap_mult (FT-L: eps), config label",
                    "map_only_uses": "WE of STD's selected configuration"},
           "n_records": len(records), "arms": {}}
    for arm in selected_arms:
        c = choose(rows, arm)
        if c is not None:
            out["arms"][arm] = c
    if "STD" in out["arms"]:
        out["arms"]["NORM"] = {**out["arms"]["STD"], "arm": "NORM", "status": "uses STD's selection (preprocessing)"}
        # Pre-freeze decision: PARA-ORTH reuses STD's delta, so it also uses STD's layer and configuration; S1 then
        # compares the keys only (no layer/step confound).
        sl, sc = out["arms"]["STD"]["layer"], out["arms"]["STD"]["config"]
        po = [r for r in rows if r["arm"] == "PARA-ORTH" and r["layer"] == sl and r["config"] == sc]
        out["arms"]["PARA-ORTH"] = {**(po[0] if po else {"arm": "PARA-ORTH", "layer": sl, "config": sc}),
                                    "status": "uses STD's selection (same delta; S1 compares keys only)"}
        out["map_only"] = out["arms"]["STD"]["WE"] < MAP_ONLY_WE
    # Ledger lambda for the lives (PROTOCOL §13): discovery lives only; maximise E_end subject to Leak_RD <= 1%.
    life_rows = []
    for p in sorted((STUDY / "results" / "lives").glob("select_*.jsonl")):
        for line in p.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                r = json.loads(line)
                if r["world"] not in DISCOVERY_WORLDS or r["run"] != "select":
                    raise SystemExit(f"non-discovery life record in {p}")
                if r["variant"] == "J":
                    life_rows.append(r)
    partial = list(problems)
    if life_rows:
        by_lam = defaultdict(list)
        for r in life_rows:
            by_lam[r["lam"]].append(r)
        lam_worlds = {lam: sorted(r["world"] for r in rs) for lam, rs in by_lam.items()}
        bad = {lam: ws for lam, ws in lam_worlds.items() if ws != list(DISCOVERY_WORLDS)}
        if bad:
            msg = (f"discovery lives must hold exactly one J life per world {DISCOVERY_WORLDS} for every lambda; "
                   f"got {bad}")
            if not args.allow_partial:
                raise SystemExit(msg)
            partial.append(msg)
        lam_stats = []
        for lam, rs in sorted(by_lam.items()):
            e_end = sum(sum(x["correct"] for x in r["facts"]) / (7 * len(r["facts"])) for r in rs) / len(rs)
            rd = sum(r["leak"]["RD"]["changed"] / max(r["leak"]["RD"]["n"], 1) for r in rs) / len(rs)
            lam_stats.append({"lam": lam, "E_end": e_end, "Leak_RD": rd, "n_lives": len(rs),
                              "feasible": rd <= LEAK_RD_MAX})
        feas = [x for x in lam_stats if x["feasible"]] or lam_stats
        best = sorted(feas, key=lambda x: (-x["E_end"], x["lam"]))[0]
        out["life_lambda"] = best["lam"]
        out["life_lambda_table"] = lam_stats
    if partial:   # --allow-partial: descriptive look only; nothing is written
        for arm, c in out["arms"].items():
            print(f"[partial] {arm:9s} L{c['layer']} {c['config']:32s} WE={c['WE']:.3f} RD={c['Leak_RD']:.3f} "
                  f"C={c['Leak_C']:.3f} n={c['n']}")
        print("[partial] not written (incomplete records):\n  " + "\n  ".join(partial))
        return 1
    (STUDY / "results").mkdir(exist_ok=True)
    (STUDY / "results" / "selection.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    tdir = STUDY / "results" / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    with (tdir / "discovery_map.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["arm"])
        w.writeheader(); [w.writerow(r) for r in rows]
    for arm, c in out["arms"].items():
        print(f"{arm:9s} L{c['layer']} {c['config']:32s} WE={c['WE']:.3f} RD={c['Leak_RD']:.3f} C={c['Leak_C']:.3f} "
              f"IE5={c['IE5']} {c['status']}")
    print("map_only:", out.get("map_only"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
