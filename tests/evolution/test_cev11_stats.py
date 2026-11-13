# -*- coding: utf-8 -*-
"""CEV-11 statistics and decision layer: cev11/stats.py, analyze.py, verify.py (fast, tmp_path only).

The end-to-end scenario writes a tiny synthetic JSONL result set whose decisions are known by construction:
H1 supported, H2 refuted, H3 inconclusive, S1 not supported (locality cost), S2 supported, S3 supported, S4 refuted,
plus discovery-world / F_freq / selection rows with adversarial values that must not move any decision.
"""
from __future__ import annotations

import itertools
import json
import math
import pathlib
import random
import sys

import pytest

STUDY = pathlib.Path(__file__).resolve().parents[2] / "research" / "living_learning_paradigm_2026_11_13"
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

import analyze  # noqa: E402
import verify  # noqa: E402
from cev11 import stats as S  # noqa: E402

INFL = ("gen", "dat", "acc", "loc", "abl", "ins")
CASES7 = ("nom",) + INFL
BOOT, PERM = 200, 500
PROTOCOL_EDITS_PER_CASE = analyze.EDITS_PER_LIVED_CASE      # read before the fixture below lowers it
SCENARIO_EDITS_PER_CASE = 3                                  # build_results: 21 edits per world, 3 per lived case


@pytest.fixture(autouse=True, scope="module")
def _scenario_item_set():
    """The synthetic scenario is small (3 edits per lived case instead of the protocol's 10); the item-set check
    itself is exercised with the protocol value in test_item_set_check_with_protocol_size."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(analyze, "EDITS_PER_LIVED_CASE", SCENARIO_EDITS_PER_CASE)
        yield


# ------------------------------------------------------------------------------------------------ stats: sign flip
def test_signflip_all_positive_margin_shifted_is_one_over_2n():
    d = [0.25, 0.30, 0.20, 0.40, 0.35, 0.28, 0.22, 0.31]          # every d - 0.15 > 0
    assert S.signflip_pvalue(d, 0.15) == 1 / 256
    assert S.signflip_pvalue([0.3] * 8, 0.15) == 1 / 256


def test_signflip_hand_computed_and_ties():
    # x = (0.3, 0.1, -0.05): signed sums >= 0.35 are (+,+,+)=0.45 and (+,+,-)=0.35 -> 2/8
    assert S.signflip_pvalue([0.3, 0.1, -0.05], 0.0) == 2 / 8
    # a zero difference ties under its own flip: 2 of 256 patterns reach the observed sum
    assert S.signflip_pvalue([0.15] + [0.3] * 7, 0.15) == 2 / 256
    # all exactly at the margin: every pattern ties -> p = 1
    assert S.signflip_pvalue([0.15] * 8, 0.15) == 1.0
    # float round-off must not break mathematical ties: 0.1 + 0.2 - 0.3 vs 0.3 - 0.1 - 0.2
    assert S.signflip_pvalue([0.1, 0.2, -0.3], 0.0) == S.signflip_pvalue([1, 2, -3], 0.0)


def test_signflip_wrong_sign_gives_p_one():
    assert S.signflip_pvalue([-0.1, -0.2, -0.05, -0.3, -0.1, -0.2, -0.4, -0.15], 0.15) == 1.0


def test_signflip_matches_brute_force():
    rnd = random.Random(11)
    for _ in range(60):
        n = rnd.randint(1, 9)
        d = [round(rnd.uniform(-0.4, 0.6), rnd.choice((1, 2, 3))) for _ in range(n)]
        m = rnd.choice((0.0, 0.05, 0.15))
        x = [v - m for v in d]
        tol = 1e-10 * max(1.0, sum(abs(v) for v in x))
        hits = sum(1 for s in itertools.product((1, -1), repeat=n)
                   if sum(a * b for a, b in zip(s, x)) >= sum(x) - tol)
        assert S.signflip_pvalue(d, m) == hits / 2 ** n


def test_signflip_rejects_bad_input():
    with pytest.raises(ValueError):
        S.signflip_pvalue([], 0.1)
    with pytest.raises(ValueError):
        S.signflip_pvalue([0.1, float("nan")], 0.1)


# ------------------------------------------------------------------------------------------------------ stats: Holm
def test_holm_step_down():
    assert S.holm({"a": 0.01, "b": 0.04, "c": 0.02}) == {"a": True, "b": True, "c": True}
    assert S.holm({"a": 0.01, "b": 0.04, "c": 0.026}) == {"a": True, "b": False, "c": False}
    assert S.holm({"a": 0.02, "b": 0.03, "c": 0.04}) == {"a": False, "b": False, "c": False}   # 0.02 > 0.05/3
    # m = 4: thresholds 0.0125, 0.0167, 0.025, 0.05; stops at the first failure even if a later p is small enough
    assert S.holm({"S1": 0.004, "S2": 0.016, "S3": 0.03, "S4": 0.049}) == \
        {"S1": True, "S2": True, "S3": False, "S4": False}
    # untested members enter with p = 1 and keep m at its preregistered size
    assert S.holm({"H1": 1 / 256, "H2": 1.0, "H3": 1.0}) == {"H1": True, "H2": False, "H3": False}
    assert S.holm({"H1": 0.02, "H2": 1.0, "H3": 1.0}) == {"H1": False, "H2": False, "H3": False}
    # ties are rejected together
    assert S.holm({"a": 0.01, "b": 0.01, "c": 0.5}) == {"a": True, "b": True, "c": False}
    with pytest.raises(ValueError):
        S.holm({"a": 1.2})


def test_holm_adjusted():
    adj = S.holm_adjusted({"a": 0.01, "b": 0.04, "c": 0.02})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.04) and adj["b"] == pytest.approx(0.04)


# ------------------------------------------------------------------------------------------ stats: t and decisions
def test_t_quantile_and_interval():
    assert S.student_t_ppf(0.975, 1) == pytest.approx(math.tan(0.475 * math.pi), rel=1e-12)
    assert S.student_t_ppf(0.975, 2) == pytest.approx(0.95 * math.sqrt(2 / (1 - 0.95 ** 2)), rel=1e-12)
    assert S.student_t_ppf(0.975, 7) == pytest.approx(2.364624251592785, rel=1e-12)
    assert S.student_t_ppf(0.025, 7) == pytest.approx(-2.364624251592785, rel=1e-12)
    d = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    half = 2.364624251592785 * math.sqrt(sum((v - 0.45) ** 2 for v in d) / 7) / math.sqrt(8)
    lo, hi = S.t_interval(d)
    assert lo == pytest.approx(0.45 - half, rel=1e-12) and hi == pytest.approx(0.45 + half, rel=1e-12)
    assert S.t_interval([0.2] * 8) == (0.2, 0.2)
    assert all(math.isnan(v) for v in S.t_interval([0.2]))


def test_decide_three_way_rule():
    big = [0.30, 0.35, 0.28, 0.40, 0.33, 0.31, 0.29, 0.36]
    assert S.decide(big, 0.15, True) == S.SUPPORTED
    # tight effect near zero: upper t bound far below the margin -> refuted
    near_zero = [0.01, -0.02, 0.00, 0.02, 0.01, -0.01, 0.00, 0.01]
    assert S.decide(near_zero, 0.15, False) == S.REFUTED
    # an effect of the wrong sign is refuted too
    assert S.decide([-0.2, -0.25, -0.18, -0.3, -0.22, -0.21, -0.27, -0.24], 0.15, False) == S.REFUTED
    # mean above the margin but noisy: neither rejected nor refuted
    noisy = [0.6, -0.3, 0.5, -0.2, 0.4, 0.1, -0.4, 0.5]
    assert S.decide(noisy, 0.15, False) == S.INCONCLUSIVE
    # just below the margin with an upper bound above it: inconclusive, not refuted
    assert S.decide([0.10, 0.20, 0.05, 0.18, 0.12, 0.15, 0.08, 0.16], 0.15, False) == S.INCONCLUSIVE
    # manipulation check failed: a would-be refutation becomes 'cannot be refuted'
    assert S.decide(near_zero, 0.15, False, can_refute=False) == S.WEAK_MANIPULATION
    assert S.decide(big, 0.15, True, can_refute=False) == S.SUPPORTED
    assert S.decide(noisy, 0.15, False, can_refute=False) == S.INCONCLUSIVE


def test_decide_insufficient_data_and_consistency():
    six = [0.30, 0.35, 0.28, 0.40, 0.33, 0.31]
    assert S.decide(six, 0.15, False) == S.INSUFFICIENT                  # fewer than 7 qualifying worlds
    assert S.decide(six, 0.15, True) == S.INSUFFICIENT
    assert S.decide(six + [0.3], 0.15, False) == S.INCONCLUSIVE           # 7 worlds are enough to decide
    assert S.decide([0.01, 0.0, 0.02], 0.15, False, min_worlds=3) == S.REFUTED
    with pytest.raises(ValueError):
        S.decide([0.0] * 8, 0.15, True)                                   # 'rejected' contradicts the data


# --------------------------------------------------------------------------------- stats: bootstrap and Spearman
def test_hierarchical_bootstrap():
    data = {w: [0.2 + 0.01 * ((w + j) % 5) for j in range(20)] for w in range(8)}
    a = S.hierarchical_bootstrap(data, n=500, seed=3)
    assert a == S.hierarchical_bootstrap(data, n=500, seed=3)            # deterministic in the seed
    assert a != S.hierarchical_bootstrap(data, n=500, seed=4)
    grand = sum(sum(v) / len(v) for v in data.values()) / len(data)
    assert a[0] <= grand <= a[1]
    assert S.hierarchical_bootstrap({0: [0.4] * 5, 1: [0.4] * 9}, n=100) == pytest.approx((0.4, 0.4))
    med = S.hierarchical_bootstrap({0: [(1.0, 3.0), (2.0,)], 1: [(5.0,)]}, n=50, seed=1,
                                   stat=lambda items: sum(sum(x) for x in items))
    assert med[0] <= med[1]
    with pytest.raises(ValueError):
        S.hierarchical_bootstrap({0: []}, n=10)


def test_spearman_within_world():
    groups = {w: ([0.1 * j + w for j in range(12)], [j + 10 * w for j in range(12)]) for w in range(4)}
    res = S.spearman_within_world(groups, n_perm=999, seed=0)
    assert res["rho"] == pytest.approx(1.0) and res["p"] == pytest.approx(1 / 1000)
    anti = {w: (x, [-v for v in y]) for w, (x, y) in groups.items()}
    res = S.spearman_within_world(anti, n_perm=999, seed=0)
    assert res["rho"] == pytest.approx(-1.0) and res["p"] == 1.0
    # negative within worlds, positive only between worlds: the within-world statistic stays negative
    shifted = {w: ([j + 100 * w for j in range(6)], [5 - j + 100 * w for j in range(6)]) for w in range(3)}
    assert S.spearman_within_world(shifted, n_perm=99)["rho"] == pytest.approx(-1.0)
    # ties use average ranks; no variance -> rho undefined, p = 1
    assert S.spearman_within_world({0: ([1, 2, 3], [0, 0, 0])}, n_perm=9) == \
        {"n": 3, "n_worlds": 1, "n_perm": 9, "seed": 0, "rho": None, "p": 1.0}
    assert list(S.rankdata([3.0, 1.0, 3.0, 2.0])) == [3.5, 1.0, 3.5, 2.0]


# ------------------------------------------------------------------------------------- synthetic result scenario
def _edit(world, arm, k, lived, classes, formal, informal, *, run="confirm", layer=2, installed=True,
          cf14_p2=1, rn=(0, 2), rd=(0, 10), **extra):
    cases = [c for c in INFL if c != lived][:len(classes)]
    rec = {
        "schema": "cev11.edit.v1", "run": run, "world": world, "arm": arm, "layer": layer,
        "edit_id": f"w{world}-e{k:03d}", "name": f"Adkent{k}", "rel": "A", "lived_case": lived,
        "old": "Eskiköy", "new": "Yenipınar", "installed": installed,
        "targets": [{"case": c, "cls": cl, "formal": f, "informal": i}
                    for c, cl, f, i in zip(cases, classes, formal, informal)],
        "PF6": {c: 1 for c in CASES7 if c != lived},
        "CF14": {"P1": {c: 1 for c in CASES7}, "P2": {c: cf14_p2 for c in CASES7}},
        "leak": {"RN": {"changed": rn[0], "n": 7 * rn[1], "entities": rn[1]},
                 "RD": {"changed": rd[0], "n": 7 * rd[1], "entities": rd[1]},
                 "SS": {"changed": 0, "n": 20, "entities": 20},
                 "RB": {"changed": 0, "n": 7}, "RC": {"changed": 0, "n": 7}},
    }
    rec.update(extra)
    return rec


def _write_jsonl(path: pathlib.Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def build_results(root: pathlib.Path, *, mc_fail=False, h3_insufficient=False, noise=True) -> pathlib.Path:
    res = root / "results"
    for wi in range(8):
        w = 1000 + wi
        std, para, rootarm = [], [], []
        for k in range(21):
            lived = INFL[k % 6] if k < 18 else "nom"
            if mc_fail:            # 1 NONIDENT of 5 targets (20% < 25%); FUSED informal alternates -> mean g = 0
                classes, inf = ["IDENT"] * 4 + ["FUSED"], [1, 0, 1, 0, k % 2]
            elif (k + wi) % 9 == 0:   # a few edits with g = 1 vary H2 across worlds
                classes, inf = ["IDENT", "IDENT", "FUSED", "RESEG"], [1, 1, 0, 0]
            else:                  # same formal-informal gap for IDENT and NONIDENT: g = 0, PE5 - IE5 = 0.5
                classes, inf = ["IDENT", "IDENT", "FUSED", "RESEG"], [1, 0, 1, 0]
            if k == 16:            # fewer than 2 target cases: excluded from H1/H2/S1 and counted
                classes, inf = classes[:1], inf[:1]
            formal = [1] * len(classes)
            if h3_insufficient and wi >= 6:
                rn = (0, 0)        # no RN neighbours -> no H3 units in two worlds
            elif wi % 2 == 0:
                rn = (2 if k % 2 == 0 else 0, 2)
            else:
                rn = (1 if k % 3 == 0 else 0, 2)
            installed = k != 17
            std.append(_edit(w, "STD", k, lived, classes, formal, inf, installed=installed, rn=rn))
            para.append(_edit(w, "PARA-ORTH", k, lived, classes, formal, [1] * len(classes), installed=installed,
                              rn=rn, rd=(7, 10)))
            rootarm.append(_edit(w, "ROOT", k, lived, classes, formal, inf, installed=installed, rn=rn, layer=1,
                                 cf14_p2=0))
        for arm, recs in (("STD", std), ("PARA-ORTH", para), ("ROOT", rootarm)):
            _write_jsonl(res / "edits" / f"confirm_{w}_{arm}.jsonl", recs)
        # S3 lives: J keeps everything; I fails more on facts with a larger max tau
        facts_j = [{"fact_id": f"f{j:02d}", "entity": f"Ad{j}", "rel": "A" if j < 15 else "C", "believed": "Obj",
                    "correct": 7, "n": 7, "max_tau": None} for j in range(20)]
        facts_i = [dict(f, correct=7 - j // 4, max_tau=0.02 * j + 0.001 * wi) for j, f in enumerate(facts_j)]
        lives = [{"schema": "cev11.life.v1", "run": "confirm", "world": w, "variant": "J", "order": 0, "layer": 2,
                  "lam": 0.1, "facts": facts_j, "retraction_residue": 0.0},
                 {"schema": "cev11.life.v1", "run": "confirm", "world": w, "variant": "I", "order": 0, "layer": 2,
                  "lam": None, "facts": facts_i, "retraction_residue": 0.0}]
        lives += [{"schema": "cev11.life.v1", "run": "confirm", "world": w, "variant": "Q", "order": o, "layer": 2,
                   "lam": None, "facts": facts_i, "leak": {"RN": {"changed": 3, "n": 400}}} for o in range(4)]
        _write_jsonl(res / "lives" / f"confirm_{w}.jsonl", lives)
        # S4 tau: the same tau at layers 0 and 4 up to a small world-specific shift -> d_w ~ 0
        shift = (0.0, 0.02, -0.01, 0.01, 0.0, 0.03, -0.02, 0.01)[wi]
        taus = []
        for e in range(16):
            for layer, s in ((0, 0.0), (2, 0.0), (4, shift)):
                taus.append({"schema": "cev11.tau.v1", "run": "confirm", "world": w, "layer": layer,
                             "entity": f"Ad{e}", "site": "ROOT", "tau_RN": [0.5 - s, 0.3 - s]})
        _write_jsonl(res / "tau" / f"confirm_{w}.jsonl", taus)
    if noise:   # adversarial rows that must never enter a decision
        bad = []
        for w in (0, 1):     # discovery worlds, even when (wrongly) labelled confirm
            for k in range(21):
                lived = INFL[k % 6] if k < 18 else "nom"
                cls = ["IDENT", "IDENT", "FUSED", "RESEG"]
                bad.append(_edit(w, "STD", k, lived, cls, [0] * 4, [1] * 4, installed=False, rn=(14, 2)))
                bad.append(_edit(w, "PARA-ORTH", k, lived, cls, [0] * 4, [0] * 4, rd=(0, 10)))
                bad.append(_edit(w, "STD", k, lived, cls, [0] * 4, [1] * 4, run="select", layer=4))
        _write_jsonl(res / "edits" / "discovery.jsonl", bad)
        ffreq = [_edit(1000, "STD", k, INFL[k % 6], ["FUSED"] * 4, [0] * 4, [1] * 4, run="ffreq", layer=3,
                       tok_frequent=bool(k % 2)) for k in range(18)]
        _write_jsonl(res / "edits" / "ffreq_1000_STD.jsonl", ffreq)
        _write_jsonl(res / "lives" / "discovery.jsonl", [
            {"schema": "cev11.life.v1", "run": "confirm", "world": 1, "variant": v, "order": 0, "layer": 2,
             "lam": None, "facts": [{"fact_id": "x", "entity": "Ad", "rel": "A", "believed": "O",
                                     "correct": 0 if v == "J" else 7, "n": 7, "max_tau": 0.1}]} for v in "JI"])
        _write_jsonl(res / "tau" / "discovery.jsonl", [
            {"schema": "cev11.tau.v1", "run": "confirm", "world": 0, "layer": layer, "entity": f"Ad{e}",
             "site": "ROOT", "tau_RN": [9.0 if layer == 0 else -9.0]} for e in range(20) for layer in (0, 4)])
    return res


EXPECTED = {"H1": S.SUPPORTED, "H2": S.REFUTED, "H3": S.INCONCLUSIVE, "S1": S.LOCALITY_COST,
            "S2": S.SUPPORTED, "S3": S.SUPPORTED, "S4": S.REFUTED}


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    root = tmp_path_factory.mktemp("cev11_stats")
    results = build_results(root)
    dec = analyze.run_analysis(results, bootstrap_n=BOOT, perm_n=PERM, seed=0, figures=False, log=lambda s: None)
    return root, results, dec


def test_end_to_end_known_decisions(scenario):
    _, results, dec = scenario
    assert dec["summary"] == EXPECTED
    h = dec["hypotheses"]
    assert h["H1"]["qualifying_worlds"] == list(range(1000, 1008))
    assert h["H1"]["d_qualifying"] == [0.5] * 8 and h["H1"]["p_signflip"] == 1 / 256
    assert all(pw["n_units"] == 17 and pw["components"]["excluded_lt2_targets"] == 1
               for pw in h["H1"]["per_world"].values())
    assert h["H2"]["p_signflip"] == 1.0 and h["H2"]["t_interval_95"][1] < 0.30
    assert all(pw["components"]["excluded_not_installed"] == 1 for pw in h["H2"]["per_world"].values())
    assert h["H3"]["p_signflip"] > 0.05 and h["H3"]["t_interval_95"][1] > 0.05
    assert h["S1"]["holm_rejected"] and not h["S1"]["noninferiority"]["met"]
    assert h["S1"]["noninferiority"]["RD"]["mean"] == pytest.approx(0.1)
    s1w = h["S1"]["per_world"]["1000"]["components"]
    assert s1w["IE_IDENT_PARA_pooled"] == 1.0 and s1w["IE_NONIDENT_PARA_pooled"] == 1.0
    assert s1w["IE_IDENT_STD_pooled"] + s1w["IE_NONIDENT_STD_pooled"] == pytest.approx(1.0)
    assert h["S2"]["d_qualifying"] == [0.5] * 8
    assert h["S3"]["d_qualifying"] == [pytest.approx(2 / 7)] * 8
    assert h["S3"]["spearman"]["met"] and h["S3"]["spearman"]["rho"] > 0.9
    assert h["S4"]["d_qualifying"] == [pytest.approx(s) for s in (0.0, 0.02, -0.01, 0.01, 0.0, 0.03, -0.02, 0.01)]
    fam = dec["families"]
    assert fam["primary"]["holm_rejected"] == {"H1": True, "H2": False, "H3": False}
    assert fam["secondary"]["holm_rejected"] == {"S1": True, "S2": True, "S3": True, "S4": False}
    assert dec["manipulation_check"]["passed"] and dec["manipulation_check"]["mean_nonident_share"] == 0.5
    for hyp in h.values():
        lo, hi = hyp["bootstrap_95"]
        assert lo <= hi
    for name in ["decisions.json", "analysis_env.json", "tables/edit_metrics.csv", "tables/life_metrics.csv",
                 "tables/tau_by_layer.csv", "tables/decisions_summary.csv"] + \
            [f"tables/per_world_{x}.csv" for x in EXPECTED]:
        assert (results / name).is_file(), name
    assert json.loads((results / "decisions.json").read_text(encoding="utf-8")) == dec


def test_discovery_and_descriptive_rows_never_enter_decisions(scenario, tmp_path):
    _, _, dec = scenario
    for hyp in dec["hypotheses"].values():
        assert not {0, 1} & set(hyp["worlds_with_data"])
        assert "0" not in hyp["per_world"] and "1" not in hyp["per_world"]
    excluded = dec["excluded_records"]
    assert excluded["edits"]["discovery world | world 0"] == 63 and excluded["edits"]["run=ffreq | world 1000"] == 18
    assert excluded["lives"]["discovery world | world 1"] == 2 and excluded["tau"]["discovery world | world 0"] == 40
    clean = analyze.compute_decisions(analyze.load_results(build_results(tmp_path, noise=False)),
                                      bootstrap_n=BOOT, perm_n=PERM, seed=0)
    assert clean["hypotheses"] == dec["hypotheses"] and clean["families"] == dec["families"]
    with pytest.raises(ValueError):
        analyze.confirmatory_worlds([0, 1000, 1001])
    with pytest.raises(ValueError):
        analyze.confirmatory_worlds([1000, 1009])          # 1009 only after 1008
    assert analyze.confirmatory_worlds([1000, 1001, 1002, 1004, 1005, 1006, 1007, 1008])[-1] == 1008


def test_analysis_is_deterministic(scenario):
    _, results, dec = scenario
    data = analyze.load_results(results)
    again = analyze.compute_decisions(data, bootstrap_n=BOOT, perm_n=PERM, seed=0)
    assert again == dec
    rendered = analyze.render_outputs(again, data)
    assert rendered["decisions.json"] == (results / "decisions.json").read_text(encoding="utf-8")
    assert rendered["tables/per_world_H1.csv"].splitlines()[1].startswith("1000,17,true,0.500000")


def test_insufficient_data_weak_manipulation_and_overrides(tmp_path):
    d1 = analyze.compute_decisions(analyze.load_results(build_results(tmp_path / "a", h3_insufficient=True,
                                                                      noise=False)),
                                   bootstrap_n=0, perm_n=PERM)
    assert d1["summary"]["H3"] == S.INSUFFICIENT and d1["hypotheses"]["H3"]["n_qualifying_worlds"] == 6
    assert d1["hypotheses"]["H3"]["p_signflip"] is None and d1["families"]["primary"]["p_for_holm"]["H3"] == 1.0
    assert d1["summary"]["H1"] == S.SUPPORTED
    d2 = analyze.compute_decisions(analyze.load_results(build_results(tmp_path / "b", mc_fail=True, noise=False)),
                                   bootstrap_n=0, perm_n=PERM)
    assert not d2["manipulation_check"]["passed"] and d2["manipulation_check"]["mean_nonident_share"] == 0.2
    assert d2["summary"]["H2"] == S.WEAK_MANIPULATION and d2["summary"]["H1"] == S.SUPPORTED
    assert d2["summary"]["S4"] == S.REFUTED                  # the manipulation check concerns H1/H2 only
    data = analyze.load_results(build_results(tmp_path / "c", noise=False))
    d3 = analyze.compute_decisions(data, bootstrap_n=0, perm_n=PERM, map_only=True, precondition_failed=["S4"])
    assert [d3["summary"][h] for h in ("H1", "H2", "H3")] == [S.NOT_TESTABLE] * 3
    assert d3["summary"]["S4"] == S.PRECONDITION and d3["summary"]["S2"] == S.SUPPORTED
    d4 = analyze.compute_decisions(analyze.Data(edits=data.edits), bootstrap_n=0, perm_n=PERM)
    assert d4["summary"]["S3"] == S.NOT_RUN and d4["summary"]["S4"] == S.NOT_RUN
    assert d4["summary"]["S2"] == S.SUPPORTED                 # Holm m stays 4 with p = 1 for the missing members


def test_schema_and_integrity_errors(tmp_path):
    good = _edit(1000, "STD", 0, "gen", ["IDENT", "FUSED"], [1, 1], [1, 0])
    analyze.validate_edit_record(good)
    broken = [
        dict(good, arm="PARA_ORTH"),
        dict(good, PF6={c: 1 for c in CASES7}),                                   # lived frame not excluded
        dict(good, targets=[{"case": "gen", "cls": "IDENT", "formal": 1, "informal": 1}]),   # lived case target
        dict(good, targets=[{"case": "dat", "cls": "fused", "formal": 1, "informal": 1}]),
        dict(good, leak=dict(good["leak"], RN={"changed": 1, "n": 10, "entities": 2})),       # n != 7 * entities
        dict(good, leak=dict(good["leak"], RN={"changed": 1, "n": 63, "entities": 9})),       # |RN| > 8
        dict(good, installed=1),
        dict(good, new=good["old"]),
        dict(good, layer=None),
        {k: v for k, v in good.items() if k != "CF14"},
    ]
    for rec in broken:
        with pytest.raises(analyze.RecordError):
            analyze.validate_edit_record(rec)
    path = tmp_path / "results" / "edits" / "x.jsonl"
    _write_jsonl(path, [good, dict(good, CF14={"P1": {}})])
    with pytest.raises(analyze.RecordError, match=r"edits/x.jsonl:2"):
        analyze.load_results(tmp_path / "results")
    a = analyze.validate_edit_record(good)
    with pytest.raises(analyze.RecordError, match="duplicate"):
        analyze.index_confirmatory_edits([a, a])
    with pytest.raises(analyze.RecordError, match="layer/config"):
        analyze.index_confirmatory_edits([a, analyze.validate_edit_record(dict(good, world=1001, layer=3))])
    other = analyze.validate_edit_record(_edit(1000, "ROOT", 1, "gen", ["IDENT", "FUSED"], [1, 1], [1, 0], layer=1))
    with pytest.raises(analyze.RecordError, match="different edits"):
        analyze.index_confirmatory_edits([a, other])
    with pytest.raises(FileNotFoundError):
        analyze.load_results(tmp_path / "no_such_results")


def test_item_set_check_with_protocol_size(monkeypatch):
    # PROTOCOL §7: 70 edits on distinct entities, 10 per lived case, in every confirmatory (world, arm)
    assert PROTOCOL_EDITS_PER_CASE == 10
    monkeypatch.setattr(analyze, "EDITS_PER_LIVED_CASE", PROTOCOL_EDITS_PER_CASE)

    def arms(n_edits=70, rename=None):
        out = []
        for arm, layer in (("STD", 2), ("ROOT", 1), ("PARA-ORTH", 2)):
            for k in range(n_edits):
                rec = _edit(1000, arm, k, CASES7[k // 10], ["IDENT", "FUSED"], [1, 1], [1, 0], layer=layer)
                if rename and k in rename:
                    rec["name"] = rename[k]
                out.append(analyze.validate_edit_record(rec))
        return out

    idx = analyze.index_confirmatory_edits(arms())
    assert sorted(len(b) for b in idx.values()) == [70, 70, 70]
    # a crash that stopped every arm at the same edit is not caught by the cross-arm comparison alone
    with pytest.raises(analyze.RecordError, match="edits per lived case"):
        analyze.index_confirmatory_edits(arms(n_edits=65))
    with pytest.raises(analyze.RecordError, match="more than once"):
        analyze.index_confirmatory_edits(arms(rename={5: "Adkent4"}))


def test_incomplete_lives_and_tau_raise(tmp_path):
    res = build_results(tmp_path / "a", noise=False)
    f = res / "lives" / "confirm_1005.jsonl"
    f.write_text("".join(line + "\n" for line in f.read_text(encoding="utf-8").splitlines()
                         if json.loads(line)["variant"] != "I"), encoding="utf-8")
    with pytest.raises(analyze.RecordError, match="partner"):
        analyze.compute_decisions(analyze.load_results(res), bootstrap_n=0, perm_n=PERM)
    res = build_results(tmp_path / "b", noise=False)
    f = res / "tau" / "confirm_1003.jsonl"
    keep = [line for line in f.read_text(encoding="utf-8").splitlines()
            if not (json.loads(line)["layer"] == 4 and json.loads(line)["entity"] in ("Ad0", "Ad1"))]
    f.write_text("".join(line + "\n" for line in keep), encoding="utf-8")
    with pytest.raises(analyze.RecordError, match="different entities"):
        analyze.compute_decisions(analyze.load_results(res), bootstrap_n=0, perm_n=PERM)


def test_jsonl_with_unicode_line_separators_in_strings_loads(tmp_path):
    # json.dumps(..., ensure_ascii=False) leaves U+2028 / U+0085 unescaped; str.splitlines() would split there
    rec = _edit(1000, "STD", 0, "gen", ["IDENT", "FUSED"], [1, 1], [1, 0], diag={"note": "a b\x85c"})
    _write_jsonl(tmp_path / "results" / "edits" / "x.jsonl", [rec])
    data = analyze.load_results(tmp_path / "results")
    assert len(data.edits) == 1 and data.edits[0]["diag"]["note"] == "a b\x85c"


def test_make_edit_record_from_item():
    item = {"edit_id": "w1000-e003", "name": "Nurten", "rel": "A", "lived_case": "abl", "old": "Kemerköy",
            "new": "Sarıpınar"}
    good = _edit(1000, "STD", 3, "abl", ["IDENT", "FUSED"], [1, 1], [1, 0])
    rec = analyze.make_edit_record(item, run="confirm", world=1000, arm="STD", layer=2, installed=True,
                                   targets=good["targets"], pf6=good["PF6"], cf14=good["CF14"], leak=good["leak"],
                                   config="step=2.0")
    assert rec["name"] == "Nurten" and analyze.validate_edit_record(rec)["config"] == "step=2.0"
    with pytest.raises(analyze.RecordError, match="unknown optional"):
        analyze.make_edit_record(item, run="confirm", world=1000, arm="STD", layer=2, installed=True,
                                 targets=good["targets"], pf6=good["PF6"], cf14=good["CF14"], leak=good["leak"],
                                 lived="abl")
    with pytest.raises(analyze.RecordError):
        analyze.make_edit_record(dict(item, lived_case="gen"), run="confirm", world=1000, arm="STD", layer=2,
                                 installed=True, targets=good["targets"], pf6=good["PF6"], cf14=good["CF14"],
                                 leak=good["leak"])


# ------------------------------------------------------------------------------------------------------- verify.py
def _fake_tree(root: pathlib.Path) -> pathlib.Path:
    study = root / "repo" / "research" / "study"
    (study / "cev11").mkdir(parents=True)
    (study / "cev11" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    (study / "PROTOCOL.md").write_text("frozen\n", encoding="utf-8")
    (study / "cev11" / "__pycache__").mkdir()
    (study / "cev11" / "__pycache__" / "mod.cpython-313.pyc").write_bytes(b"\0")
    tests = root / "repo" / "tests" / "evolution"
    tests.mkdir(parents=True)
    (tests / "test_cev11_dummy.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    return study


def test_verify_manifest_and_rerun(tmp_path):
    results = build_results(tmp_path)
    analyze.run_analysis(results, bootstrap_n=BOOT, perm_n=PERM, figures=False, log=lambda s: None)
    study = _fake_tree(tmp_path)
    repo = study.parents[1]
    manifest, n = verify.write_manifest(results, study=study, repo=repo)
    labels = [line.split("  ", 1)[1] for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert "cev11/mod.py" in labels and "../../tests/evolution/test_cev11_dummy.py" in labels
    assert "results/decisions.json" in labels and "results/manifest.sha256" not in labels
    assert not any("__pycache__" in lab for lab in labels) and n == len(labels)
    rep = verify.rerun(results, study=study, repo=repo)
    assert rep["ok"], rep
    assert rep["manifest"]["ok"] and rep["decisions"]["ok"] and rep["tables"]["ok"] and rep["analysis_code"]["ok"]

    # decisions must have been computed with the analysis code listed in the freeze manifest
    code = analyze.analysis_code_hashes()
    frozen_ok = tmp_path / "freeze_ok.sha256"
    frozen_ok.write_text("".join(f"{code[rel]}  {rel}\n" for rel in sorted(code)), encoding="utf-8")
    assert verify.rerun(results, study=study, repo=repo, freeze_manifest=frozen_ok)["ok"]
    stale = tmp_path / "freeze_stale.sha256"
    stale.write_text("".join(f"{'0' * 64}  {rel}\n" for rel in sorted(code)), encoding="utf-8")
    rep = verify.rerun(results, study=study, repo=repo, freeze_manifest=stale)
    assert not rep["ok"] and not rep["freeze"]["ok"] and rep["decisions"]["ok"]

    # a changed input line breaks both the manifest and the recomputed decisions
    f = results / "edits" / "confirm_1000_STD.jsonl"
    lines = f.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["targets"][1]["informal"] = 1 - rec["targets"][1]["informal"]
    f.write_text("\n".join([json.dumps(rec, ensure_ascii=False)] + lines[1:]) + "\n", encoding="utf-8")
    rep = verify.rerun(results, study=study, repo=repo)
    assert not rep["ok"] and "results/edits/confirm_1000_STD.jsonl" in rep["manifest"]["mismatched"]
    assert rep["decisions"]["n_differences"] > 0 and not rep["tables"]["ok"]

    # code-only (freeze) manifest: no results lines, code changes are detected
    frozen, _ = verify.write_manifest(results, code_only=True, out=results / "freeze.sha256", study=study, repo=repo)
    assert all(not line.split("  ", 1)[1].startswith("results/")
               for line in frozen.read_text(encoding="utf-8").splitlines())
    assert verify.check_manifest(frozen, results, study=study, repo=repo)["ok"]
    (study / "cev11" / "mod.py").write_text("X = 2\n", encoding="utf-8")
    chk = verify.check_manifest(frozen, results, study=study, repo=repo)
    assert not chk["ok"] and chk["mismatched"] == ["cev11/mod.py"]
    with pytest.raises(ValueError):
        verify.write_manifest(results, out=tmp_path / "elsewhere" / "m.sha256", study=study, repo=repo)
    assert not verify.check_manifest(results / "missing.sha256", results, study=study, repo=repo)["ok"]


def test_verify_freeze_manifest_default_and_rewrite(tmp_path):
    results = build_results(tmp_path)
    analyze.run_analysis(results, bootstrap_n=BOOT, perm_n=PERM, figures=False, log=lambda s: None)
    study = _fake_tree(tmp_path)
    repo = study.parents[1]
    # the code-only (freeze) manifest has its own default path, so the later full manifest cannot overwrite it
    freeze, _ = verify.write_manifest(results, code_only=True, study=study, repo=repo)
    full, _ = verify.write_manifest(results, study=study, repo=repo)
    assert freeze == study / verify.FREEZE_NAME and full == results / verify.MANIFEST_NAME
    assert "freeze_manifest.sha256" in full.read_text(encoding="utf-8")      # the full manifest pins the freeze
    # rewriting a manifest that lies inside the code tree must not list its own (stale) hash
    freeze, _ = verify.write_manifest(results, code_only=True, study=study, repo=repo)
    assert "freeze_manifest.sha256" not in freeze.read_text(encoding="utf-8")
    chk = verify.check_manifest(freeze, results, study=study, repo=repo)
    assert chk["ok"] and chk["unlisted"] == [], chk


def test_verify_detects_tampered_decisions(tmp_path):
    results = build_results(tmp_path)
    analyze.run_analysis(results, bootstrap_n=BOOT, perm_n=PERM, figures=False, log=lambda s: None)
    study = _fake_tree(tmp_path)
    repo = study.parents[1]
    verify.write_manifest(results, study=study, repo=repo)
    assert verify.rerun(results, study=study, repo=repo)["ok"]
    path = results / "decisions.json"
    dec = json.loads(path.read_text(encoding="utf-8"))
    dec["summary"]["H2"] = S.SUPPORTED                                          # a forged decision
    dec["hypotheses"]["H2"]["decision"] = S.SUPPORTED
    path.write_text(json.dumps(dec, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    rep = verify.rerun(results, study=study, repo=repo)
    assert not rep["ok"] and "results/decisions.json" in rep["manifest"]["mismatched"]
    assert rep["decisions"]["n_differences"] == 2 and rep["decisions"]["summary_recomputed"]["H2"] == S.REFUTED
