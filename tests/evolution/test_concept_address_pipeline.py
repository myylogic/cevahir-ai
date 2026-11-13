# -*- coding: utf-8 -*-
"""KA study — statistics, selection, confirmatory decisions, verification tooling, FT comparators and the
per-world driver with resume (tiny random model and synthetic record sets; tmp_path only).

None of these tests reads the freeze state of the real study folder. Whatever depends on that state (the status
line of PROTOCOL.md, the freeze manifest, the selection, amendments) runs on a copy in ``tmp_path`` that
``_clone_study`` first puts into the NOT FROZEN state. The test files are part of the freeze manifest and cannot
be edited after the freeze, so they have to give the same outcome before and after it; the ``study`` fixture
checks this by cloning once from the folder as it is and once from a frozen copy.
"""
import importlib.util
import itertools
import json
import math
import pathlib
import random
import shutil
import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
KA_DIR = REPO / "research" / "concept_address_2026_11_13"
if str(KA_DIR) not in sys.path:
    sys.path.insert(0, str(KA_DIR))

from ka import KA_VERSION  # noqa: E402
from ka import analysis as A  # noqa: E402
from ka import codes as C  # noqa: E402
from ka import ft as FT  # noqa: E402
from ka import gates as G  # noqa: E402
from ka import part_a as PA  # noqa: E402
from ka import stats as S  # noqa: E402
from ka import streams as ST  # noqa: E402
from ka import world_run as WR  # noqa: E402
from ka.testing import tiny_base  # noqa: E402
from cev11.model import model_sha256  # noqa: E402
from cev11.world import frame_prefix  # noqa: E402
from tokenizer_management.byte_bpe import BOS_ID  # noqa: E402


def _load(name, path):
    """Load a script of a KA folder under a unique module name (CEV-11 has scripts with the same file names).

    The scripts put their own folder at the front of ``sys.path``; that is undone here, so that neither the real
    folder nor a clone in a temporary directory can shadow a same-named module of another study later on."""
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


V = _load("ka_verify", KA_DIR / "verify.py")


@pytest.fixture(autouse=True)
def _single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(autouse=True)
def _import_state():
    """Scripts loaded from a clone must leave no trace: the driver caches its verify module under a fixed name
    and puts its folder on ``sys.path`` when it loads it."""
    saved = list(sys.path)
    yield
    sys.path[:] = saved
    sys.modules.pop("ka_verify_tool", None)


# ------------------------------------------------------------------------------------------------ frontier LP
def _cell(Y, X, R, E, C=0.0):
    return {"Y": Y, "X": X, "R": R, "E": E, "C": C}


def test_frontier_hand_cases():
    assert S.frontier([_cell(0.7, 0.05, 0.0, 1.0)]) == (0.7, [1.0])
    assert S.frontier([_cell(0.7, 0.05, 0.0, 0.5)])[0] == 0.0                 # efficacy too low: infeasible
    assert S.frontier([_cell(0.7, 0.05, 0.02, 1.0)])[0] == 0.0                # random leak too high, E drops if mixed
    assert S.frontier([])[0] == 0.0
    # mixing a leaky strong configuration with a clean weak one up to the leak budget: w = (0.5, 0.5)
    f, w = S.frontier([_cell(1.0, 0.2, 0.0, 1.0), _cell(0.0, 0.0, 0.0, 1.0)])
    assert f == pytest.approx(0.5) and w[0] == pytest.approx(0.5) and 0.3 - 1e-9 <= w[1] <= 0.5 + 1e-9
    # the empty write cannot dilute: efficacy must stay >= 0.8, so at most 20% of the mass can be dropped
    f, w = S.frontier([_cell(1.0, 0.12, 0.0, 1.0)])
    assert f == pytest.approx(10 / 12) and w == pytest.approx([10 / 12])
    # CEV-11's extra constraint on Leak_C
    cells = [_cell(1.0, 0.0, 0.0, 1.0, C=0.25), _cell(0.2, 0.0, 0.0, 1.0, C=0.0)]
    assert S.frontier(cells)[0] == pytest.approx(1.0)
    assert S.frontier(cells, c_max=0.05)[0] == pytest.approx(0.2 * 1.0 + 0.2 * 0.8)


def test_frontier_can_need_three_configurations():
    """Three constraints plus the simplex bound: the optimum can need more than a pair of configurations, so a
    search over pairs is not exact (here no single configuration and no pair is even feasible)."""
    cells = [_cell(0.84, 0.357, 0.014, 0.481), _cell(0.725, 0.241, 0.003, 0.739), _cell(0.739, 0.007, 0.001, 0.604),
             _cell(0.982, -0.003, 0.019, 0.914), _cell(0.663, -0.009, 0.042, 0.587)]
    f, w = S.frontier(cells)
    assert f > 0.8 and sum(1 for v in w if v > 1e-9) == 3
    best_pair = max(S.frontier([cells[i] for i in sub])[0]
                    for k in (1, 2) for sub in itertools.combinations(range(5), k))
    assert best_pair == 0.0


def _lp_exhaustive(cells, c_max=None):
    """Independent exhaustive solution of the frontier programme in its original variables.

    The feasible set is {w : G w <= h} with the rows sum w <= 1, sum w X <= x_max, sum w R <= r_max,
    -sum w E <= -e_min (and sum w C <= c_max), and -w_k <= 0 for every k. A bounded feasible linear programme
    attains its optimum at a vertex, and a vertex has n linearly independent active rows: every choice of n rows
    out of n + m is solved and the best feasible point is kept. Empty feasible set -> 0 (the frontier's rule).
    """
    n = len(cells)
    rows = [[1.0] * n, [c["X"] for c in cells], [c["R"] for c in cells], [-c["E"] for c in cells]]
    h = [1.0, S.X_MAX, S.R_MAX, -S.E_MIN]
    if c_max is not None:
        rows.append([c["C"] for c in cells])
        h.append(c_max)
    Gm = np.vstack([np.array(rows, dtype=np.float64), -np.eye(n)])
    hv = np.array(h + [0.0] * n)
    y = np.array([c["Y"] for c in cells])
    best = None
    for active in itertools.combinations(range(len(hv)), n):
        M = Gm[list(active)]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        w = np.linalg.solve(M, hv[list(active)])
        if (Gm @ w <= hv + 1e-10).all():
            best = float(y @ w) if best is None else max(best, float(y @ w))
    return 0.0 if best is None else max(best, 0.0)


def test_frontier_equals_an_exhaustive_linear_programme_on_random_instances():
    """F* against an independent exhaustive solution of the same linear programme, to 1e-9, on random instances
    with and without the Leak_C constraint; the returned mixture is feasible and attains the value."""
    rnd = random.Random(0)
    feasible_n = infeasible_n = 0
    for trial in range(150):
        n = rnd.randint(1, 8)
        cells = [_cell(rnd.random(), rnd.uniform(-0.05, 0.4), rnd.uniform(0, 0.03), rnd.uniform(0.5, 1.0),
                       rnd.uniform(0, 0.3)) for _ in range(n)]
        for c_max in (None, 0.05):
            f, w = S.frontier(cells, c_max=c_max)
            assert f == pytest.approx(_lp_exhaustive(cells, c_max), abs=1e-9), (trial, c_max)
            if any(x > 0 for x in w):
                feasible_n += 1
                assert sum(w) <= 1 + 1e-9 and min(w) >= -1e-12
                assert S.mixture_value(cells, w, "X") <= S.X_MAX + 1e-9
                assert S.mixture_value(cells, w, "R") <= S.R_MAX + 1e-9
                assert S.mixture_value(cells, w, "E") >= S.E_MIN - 1e-9
                assert c_max is None or S.mixture_value(cells, w, "C") <= c_max + 1e-9
                assert f == pytest.approx(S.mixture_value(cells, w, "Y"), abs=1e-9)
            else:
                infeasible_n += 1
                assert f == 0.0
    assert feasible_n >= 60 and infeasible_n >= 30                           # both regimes are exercised


# ------------------------------------------------------------------------------------------------ decision rules
def test_contrast_labels():
    strong = [0.30, 0.28, 0.35, 0.31, 0.29, 0.33, 0.27, 0.32]
    assert S.signflip_pvalue(strong, 0.15) == 1 / 256
    assert S.label_contrast(strong, 0.15, True) == S.SUPPORTED
    assert S.label_contrast([0.05, 0.06, 0.04, 0.05, 0.07, 0.05, 0.06, 0.05], 0.15, False) == S.BELOW_SESOI
    assert S.label_contrast([-0.05, 0.02, -0.01, 0.0, 0.01, -0.02, 0.0, 0.01], 0.15, False) == S.REFUTED
    assert S.label_contrast([0.3, -0.1, 0.25, 0.0, 0.2, 0.1, 0.4, -0.2], 0.15, False) == S.INCONCLUSIVE
    assert S.label_contrast(strong[:5], 0.15, True) == S.INSUFFICIENT


def test_two_sided_contrast_and_labels():
    up = [0.30, 0.28, 0.35, 0.31, 0.29, 0.33, 0.27, 0.32]
    t = S.two_sided(up, 0.10)
    assert t["p_plus"] == 1 / 256 and t["p_minus"] == 1.0 and t["p"] == 2 / 256
    assert S.label_two_sided(up, 0.10, True, t, ("memory better", "FT better")) == "memory better"
    down = [-v for v in up]
    t2 = S.two_sided(down, 0.10)
    assert t2["p_minus"] == 1 / 256 and S.label_two_sided(down, 0.10, True, t2, ("memory better", "FT better")) == "FT better"
    flat = [0.01, -0.02, 0.0, 0.02, -0.01, 0.01, 0.0, -0.01]
    t3 = S.two_sided(flat, 0.10)
    assert t3["p"] == 1.0 and S.label_two_sided(flat, 0.10, False, t3) == S.EQUIVALENT
    wide = [0.3, -0.3, 0.2, -0.2, 0.25, -0.25, 0.1, -0.1]
    assert S.label_two_sided(wide, 0.10, False, S.two_sided(wide, 0.10)) == S.INCONCLUSIVE
    # with n = 6 the two-sided p cannot go below 2/64: it can never pass the first Holm level of a family of 3
    assert S.two_sided(up[:6], 0.10)["p"] == 2 / 64 > 0.05 / 3


def test_threshold_claims_and_degenerate_variance():
    ece = [0.02, 0.03, 0.025, 0.03, 0.02, 0.035, 0.03, 0.025]
    c = S.threshold_claim(ece, 0.05, "less", rejected=True)
    assert c["p"] < 0.001 and c["label"] == S.SUPPORTED and c["ci95"][1] < 0.05
    bad = [0.08, 0.09, 0.07, 0.10, 0.08, 0.09, 0.085, 0.09]
    c = S.threshold_claim(bad, 0.05, "less", rejected=False)
    assert c["p"] > 0.99 and c["label"] == S.REFUTED
    mid = [0.02, 0.09, 0.03, 0.08, 0.04, 0.07, 0.05, 0.06]
    assert S.threshold_claim(mid, 0.05, "less", rejected=False)["label"] == S.INCONCLUSIVE
    # a constant sample has no variance: p = 0 strictly on the claimed side, 1 otherwise, never a NaN
    assert S.t_pvalue([0.0] * 8, 0.01, "less") == 0.0
    assert S.t_pvalue([0.01] * 8, 0.01, "less") == 1.0
    assert S.t_pvalue([0.9] * 8, 0.85, "greater") == 0.0
    assert S.interval([0.0] * 8) == (0.0, 0.0)
    assert S.threshold_claim([0.9] * 8, 0.85, "greater", rejected=True)["label"] == S.SUPPORTED
    assert S.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0) and S.pearson([1, 1, 1], [1, 2, 3]) is None
    assert S.fisher_z(0.8) == pytest.approx(math.atanh(0.8)) and math.isfinite(S.fisher_z(1.0))


def test_upper_bound_pass_probability_is_the_power_of_an_upper_bound_guard():
    """P(upper 95% bound of the mean of n values <= limit): against simulation, and at the values that matter
    for the KA-H1 locality guard — a mixture sitting on its leak budget passes the guard only rarely."""
    rng = np.random.default_rng(0)
    t = S.student_t_ppf(0.975, 7)
    for mean, sd, limit in ((0.10, 0.071, 0.13), (0.10, 0.034, 0.13), (0.010, 0.0117, 0.015), (0.05, 0.071, 0.13)):
        x = rng.normal(mean, sd, size=(200_000, 8))
        sim = float(np.mean(x.mean(1) + t * x.std(1, ddof=1) / math.sqrt(8) <= limit))
        assert S.upper_bound_pass_probability(mean, sd, 8, limit) == pytest.approx(sim, abs=0.005)
    # a mean exactly on the constraint with the spread seen across CEV-11's worlds (scaled to 16 edits): the
    # guard "upper bound <= constraint + margin" passes about one time in five for X and for R
    assert 0.15 < S.upper_bound_pass_probability(0.10, 0.071, 8, 0.13) < 0.21
    assert 0.15 < S.upper_bound_pass_probability(0.010, 0.0117, 8, 0.015) < 0.21
    # monotone in the mean, the spread and the number of worlds; exact in the degenerate cases
    assert S.upper_bound_pass_probability(0.05, 0.071, 8, 0.13) > S.upper_bound_pass_probability(0.10, 0.071, 8, 0.13)
    assert S.upper_bound_pass_probability(0.10, 0.02, 8, 0.13) > S.upper_bound_pass_probability(0.10, 0.071, 8, 0.13)
    assert S.upper_bound_pass_probability(0.10, 0.071, 30, 0.13) > S.upper_bound_pass_probability(0.10, 0.071, 8, 0.13)
    assert S.upper_bound_pass_probability(0.10, 0.0, 8, 0.13) == 1.0
    assert S.upper_bound_pass_probability(0.14, 0.0, 8, 0.13) == 0.0
    assert S.upper_bound_pass_probability(0.0, 1e-4, 8, 0.13) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        S.upper_bound_pass_probability(0.1, 0.05, 1, 0.13)


def test_hierarchical_bootstrap_matches_the_frozen_engine_and_combines_columns():
    from cev11.stats import hierarchical_bootstrap
    rng = np.random.default_rng(1)
    one = {w: list(rng.random(10 + w)) for w in range(5)}
    assert S.hier_bootstrap(one, n=1500, seed=3) == hierarchical_bootstrap(one, n=1500, seed=3)
    # two columns per unit, statistic = difference of the column means: the interval is around the true difference
    two = {w: [[float(rng.random() < 0.8), float(rng.random() < 0.3)] for _ in range(60)] for w in range(8)}
    lo, hi = S.hier_bootstrap(two, combine=lambda m: m[:, 0] - m[:, 1], n=2000, seed=0)
    mean = sum(sum(a - b for a, b in rows) / len(rows) for rows in two.values()) / len(two)
    assert lo < mean < hi and 0.3 < lo and hi < 0.7
    assert S.hier_bootstrap(two, combine=lambda m: m[:, 0] - m[:, 1], n=2000, seed=0) == (lo, hi)      # deterministic
    with pytest.raises(ValueError):
        S.hier_bootstrap({})
    with pytest.raises(ValueError):
        S.hier_bootstrap({0: []})


# ------------------------------------------------------------------------------------------------ synthetic records
PHASE_FILES = ("setup.json",) + tuple(WR.RECORD_FILES[p] for p in WR.PHASES)


def _cfg():
    return json.loads(json.dumps(asdict(WR.RunConfig())))


def _part_a_records(world, cfg, y_ctx, y_base, leak_ctx=0.0, in_ball=True, prefix_ball=True):
    recs = []
    rnd = random.Random(world)
    for i in range(cfg["n_test"]):
        for conf in PA.expected_configs(cfg["addresses"], cfg["steps"], cfg["qs"], cfg["ft_refs"]):
            address = conf.split("|")[0]
            gate = address in ("A-TOKG", "A-RES", "A-CTX", "A-SUP")
            best = conf == "A-CTX|step=1|q=0.01"
            y = y_ctx if best else (y_base if address in ("A-TOK", "A-TOKG", "A-RES") and "step=1" in conf else 0.0)
            ok = round(y * 2 + rnd.choice([-0.01, 0.01]))
            ok = max(0, min(2, ok))
            rn = round((leak_ctx if best else 0.0) * 56)
            r = {"world": world, "edit_id": f"w{world}-a{i:02d}", "address": address, "config": conf,
                 "WF": {"ok": 1, "n": 1}, "PF6": {"ok": 6, "n": 6}, "FUSED": {"ok": ok, "n": 2},
                 "IDENT": {"ok": 3, "n": 3}, "CTX": {"ok": 2, "n": 2}, "RN": {"changed": rn, "n": 56},
                 "RD": {"changed": 0, "n": 70}, "C": {"changed": 3, "n": 14}, "PX": {"changed": 0, "n": 0}}
            if gate:
                # rows: [in the ball on the scored sequence, outcome, in the ball on the prefix]
                hit = 1 if (best and in_ball) else 0
                pre = hit if prefix_ball else 0
                r["in_FUSED"] = [[hit, 1 if k < ok else 0, pre] for k in range(2)]
                r["in_RN"] = [[1 if k < rn else 0, 1 if k < rn else 0, 1 if k < rn else 0] for k in range(56)]
                r["in_RD"] = [[0, 0, 0]] * 70
            if best:
                r["greedy_check"] = {"n": 8, "disagree": 0}
            recs.append(r)
    return recs


def _fused_facts(share, n=40):
    return [[i, 1 if i < round(share * n) else 0] for i in range(n)]


def _episodic_records(world, cfg, layers, fused_mem, ece, leak_lo, leak_hi, fixed_hi, att_delta=0.0):
    recs = []
    cps = cfg["checkpoints"]
    for l in layers:
        for kind, d in (("PRED-RES", 0.0), ("PRED-ATT", att_delta)):
            name = f"{kind}@{l}"
            for n in cps:
                frac = (math.log(n) - math.log(25)) / (math.log(400) - math.log(25))
                reads = [("rule", leak_hi), ("fixed", fixed_hi)]
                if n in (cps[0], cfg["n_primary"], cps[-1]):
                    reads.append(("nonull", 1.0))
                for read, hi in reads:
                    lu = leak_lo + frac * (hi - leak_lo)
                    fused = fused_mem - 0.02 * (4 - l) + d
                    rec = {"world": world, "address": name, "read": read, "N": n, "lived": 1.0, "para": 1.0,
                           "fused": fused, "leadin": 0.98, "Leak_U": lu, "Leak_C": 0.01,
                           "Leak_RN": 0.02, "PX": 0.05, "OBJ": 0.0, "OBJ1": 0.01, "structured": 0.018,
                           "ece": ece, "auc_leak": 0.97, "auc_fused": 0.99, "fused_first25": fused_mem,
                           "n_fused": 40, "cap_hit_share": 0.2, "U_any_over_theta": lu,
                           "law_false_fire": 0.02, "U_markov_obs": 0.4}
                    if read == "rule":
                        rec["fused_facts"] = _fused_facts(fused)
                    recs.append(rec)
            recs.append({"world": world, "address": name, "read": "frontier", "N": cfg["n_primary"], "points": [],
                         "G_F@2": fused_mem + d, "PX@2": 0.30 - (0.2 if kind == "PRED-ATT" else 0.0)})
    return recs


def _ft_records(world, cfg, fused_ft, dbpb=0.002, newest=1.0):
    recs = []
    for arm, off in (("FT-L", -0.1), ("FT-ALL", 0.0)):
        for n in cfg["ft_checkpoints"]:
            rec = {"world": world, "arm": arm, "N": n, "fused": fused_ft + off, "lived": 0.9, "Leak_U": 0.004,
                   "structured": 0.02, "dbpb": dbpb if n == cfg["n_primary"] else None, "newest_lived": True,
                   "newest_efficacy": newest, "fused_first25": fused_ft, "Leak_C": 0.01, "Leak_RN": 0.03,
                   "fused_facts": _fused_facts(fused_ft + off)}
            if n == cfg["ft_checkpoints"][-1]:
                rec["arm_done"] = True
            recs.append(rec)
    return recs


def _rehash(d):
    """Make DONE.json agree with the files again after a deliberate edit of one of them."""
    done = json.loads((d / "DONE.json").read_text())
    done["files"] = {f: WR.sha256_file(d / f) for f in done["files"]}
    (d / "DONE.json").write_text(json.dumps(done))


def _write_world(root, world, run, layers, noise=0.0, phases=WR.PHASES, required=None, done=True, **kw):
    """A synthetic world directory in the runner's layout. ``phases`` are the record files written and listed,
    ``required`` what DONE.json declares as required (default: what the protocol requires of such a run)."""
    cfg = _cfg()
    d = root / f"w{world}"
    d.mkdir(parents=True)
    addresses = {f"{k}@{l}": {"logloss": 0.02 - 0.003 * l + (0.004 if k == "PRED-ATT" else 0.0)}
                 for l in layers for k in ("PRED-RES", "PRED-ATT")}
    setup = {"ka_version": KA_VERSION, "world": world, "run": run, "config": cfg, "addresses": addresses,
             "partitions": {"n_test": cfg["n_test"], "n_stream": cfg["n_stream"]},
             "ft_configs": {"FT-L": {"layer": 0}, "FT-ALL": {"layer": None}}, "geometry": {}}
    (d / "setup.json").write_text(json.dumps(setup), encoding="utf-8")
    if "parta" in phases:
        WR.write_jsonl_atomic(d / "part_a.jsonl", _part_a_records(
            world, cfg, kw.get("y_ctx", 0.9) + noise, kw.get("y_base", 0.1), kw.get("leak_ctx", 0.0),
            kw.get("in_ball", True), kw.get("prefix_ball", True)))
    if "episodic" in phases:
        WR.write_jsonl_atomic(d / "episodic.jsonl", _episodic_records(
            world, cfg, layers, kw.get("fused_mem", 0.9) + noise, kw.get("ece", 0.03) + noise / 10,
            kw.get("leak_lo", 0.002), kw.get("leak_hi", 0.004) + noise / 100, kw.get("fixed_hi", 0.08),
            kw.get("att_delta", 0.0)))
    if "ft" in phases:
        WR.write_jsonl_atomic(d / "ft_stream.jsonl", _ft_records(world, cfg, kw.get("fused_ft", 0.5) - noise,
                                                                 newest=kw.get("newest", 1.0)))
    if "checks" in phases:
        chk = {"world": world, "model_unchanged": True,
               "addresses": {f"PRED-RES@{l}": {"dbpb": 0.0005 + noise / 100, "n_sample": 40,
                                               "cached_vs_hooked_disagree": kw.get("disagree", 0),
                                               "hooked_vs_greedy_disagree": 0} for l in layers}}
        (d / "checks.json").write_text(json.dumps(chk), encoding="utf-8")
    if done:
        files = ["setup.json"] + [WR.RECORD_FILES[p] for p in WR.PHASES if p in phases]
        rec = {"world": world, "run": run, "ka_version": KA_VERSION, "phases": [p for p in WR.PHASES if p in phases],
               "required_phases": list(WR.required_phases(run) if required is None else required),
               "protocol_config": True, "files": {f: WR.sha256_file(d / f) for f in files}, "seconds": {}}
        (d / "DONE.json").write_text(json.dumps(rec), encoding="utf-8")
    return d


def _discovery(tmp_path, **kw):
    root = tmp_path / "discovery"
    for w in (0, 1):
        _write_world(root, w, "discovery", [2, 3, 4, 5], **kw)
    return A.select(A.load_worlds(root, (0, 1)))


def _confirm(tmp_path, layers=(4,), n=8, leak_by_world=None, **kw):
    root = tmp_path / "confirm"
    for i in range(n):
        if leak_by_world is not None:
            kw["leak_ctx"] = leak_by_world[i]
        _write_world(root, 1000 + i, "confirm", list(layers), noise=0.01 * ((i % 3) - 1), **kw)
    return root


# ------------------------------------------------------------------------------------------------ selection
def test_selection_from_discovery_worlds(tmp_path):
    sel = _discovery(tmp_path)
    body = A.check_selection(sel)
    assert body["discovery_worlds"] == [0, 1]
    assert body["pred_layer"] == 5                         # lowest c0 log-loss, and within the fallback gap on G_F
    assert body["pred_att"] is True                        # log-loss ratio <= 2
    assert body["h1_confirmatory"] is True and body["primary_family"] == ["KA-H1", "KA-H2", "KA-H3"]
    assert body["secondary_family"] == ["S1a", "S1b", "S2"]
    best = body["mixture_configs"]["A-CTX"].index("A-CTX|step=1|q=0.01")
    assert len(body["mixture_configs"]["A-CTX"]) == 9 and len(body["mixture_configs"]["A-TOK"]) == 3
    assert body["mixture_discovery"]["A-CTX"]["feasible"] and body["mixtures"]["A-CTX"][best] == pytest.approx(1.0)
    assert body["mixture_discovery"]["A-CTX"]["F_star"] > body["mixture_discovery"]["A-TOK"]["F_star"] + 0.5
    assert body["screens"]["G-KA1a"]["pass"] and body["screens"]["G-KA1c"]["pass"]
    assert body["screens"]["G-KA1c"]["newest_efficacy"] == {"FT-L": 1.0, "FT-ALL": 1.0}
    assert body["screens"]["consistency"]["pass"] and body["go"] is True and body["blocking_gates"] == []
    assert A.selection_blockers(body) == []
    assert body["h2_predicted_direction"] == "memory"
    # the body is hashed: any change is detected
    sel["body"]["pred_layer"] = 2
    with pytest.raises(ValueError):
        A.check_selection(sel)


def test_selection_screens_remove_h1_and_pred_att(tmp_path):
    sel = _discovery(tmp_path, y_ctx=0.12, y_base=0.1)      # in-sample gap below 0.10
    body = A.check_selection(sel)
    assert body["h1_confirmatory"] is False
    assert body["primary_family"] == ["KA-H2", "KA-H3"]


def test_selection_gate_g_ka1c_and_the_consistency_checks_block_the_freeze(tmp_path):
    """G-KA1c is enforced: an FT comparator that writes its newest fact less than 90% of the time forbids the
    freeze. So does a scorer disagreement above 1% on the discovery worlds."""
    weak = A.check_selection(_discovery(tmp_path / "weak", newest=0.85))
    assert weak["screens"]["G-KA1c"]["pass"] is False and weak["screens"]["G-KA1c"]["threshold"] == 0.9
    assert weak["go"] is False and weak["blocking_gates"] == ["G-KA1c"]
    assert any("G-KA1c failed" in p for p in A.selection_blockers(weak))
    edge = A.check_selection(_discovery(tmp_path / "edge", newest=0.9))
    assert edge["go"] is True                                                       # >= 0.9 passes
    bad = A.check_selection(_discovery(tmp_path / "bad", disagree=2))               # 4 of 80 probes: 5%
    assert bad["screens"]["consistency"]["failures"] == ["cached_vs_hooked"]
    assert bad["blocking_gates"] == ["consistency"]
    assert any("consistency" in p for p in A.selection_blockers(bad))
    # a selection without the field (written by other code) never counts as a go
    assert A.selection_blockers({}) != []


def test_selection_reports_the_outlook_of_the_locality_guard(tmp_path):
    """For every frozen mixture the selection says how likely the KA-H1 locality guard is to pass if the
    confirmatory worlds behave like the discovery edits (design information for the freeze decision)."""
    clean = A.check_selection(_discovery(tmp_path / "clean"))["mixture_discovery"]["A-CTX"]["locality_guard"]
    assert clean["X"]["mean"] == 0.0 and clean["p_pass"] == 1.0 and clean["n_edits"] == 32
    # a mixture on the leak budget: every discovery edit leaks to 6 of 56 root neighbours (X = 0.107 > 0.10 is
    # infeasible alone, so the frontier dilutes it down to X = 0.10 exactly)
    body = A.check_selection(_discovery(tmp_path / "edge", leak_ctx=6 / 56))
    edge = body["mixture_discovery"]["A-CTX"]
    assert edge["feasible"] and edge["X"] == pytest.approx(S.X_MAX)
    g = edge["locality_guard"]
    assert g["X"]["mean"] == pytest.approx(S.X_MAX) and g["X"]["limit"] == pytest.approx(0.13)
    assert g["X"]["sd_world_lower"] == pytest.approx(g["X"]["sd_edit"] / 4)         # 16 TEST edits per world
    assert g["p_pass"] == min(g["X"]["p_pass"], g["R"]["p_pass"]) and 0.0 <= g["p_pass"] <= 1.0
    # an infeasible address is frozen at the empty write: nothing to guard
    assert body["mixture_discovery"]["A-TOK"]["locality_guard"]["p_pass"] in (None, 1.0)


def test_selection_refuses_non_protocol_and_confirmatory_inputs(tmp_path):
    root = tmp_path / "x"
    d = _write_world(root, 0, "discovery", [4])
    setup = json.loads((d / "setup.json").read_text())
    setup["config"]["protocol"] = False                      # a smoke run
    (d / "setup.json").write_text(json.dumps(setup))
    _rehash(d)
    with pytest.raises(ValueError, match="disagree on the protocol configuration"):
        A.load_worlds(root)                                  # DONE.json still says protocol: the field is read
    done = json.loads((d / "DONE.json").read_text())
    done["protocol_config"] = False
    (d / "DONE.json").write_text(json.dumps(done))
    with pytest.raises(ValueError, match="protocol configuration"):
        A.select(A.load_worlds(root))
    # a file edited after DONE.json is refused on load
    (d / "ft_stream.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="does not match the hash"):
        A.load_worlds(root)
    # records of another format version are refused
    old = _write_world(tmp_path / "old", 0, "discovery", [4])
    done = json.loads((old / "DONE.json").read_text())
    done["ka_version"] = "ka/1"
    (old / "DONE.json").write_text(json.dumps(done))
    with pytest.raises(ValueError, match="version"):
        A.load_worlds(tmp_path / "old")
    assert any("version" in p for p in V.validate_world(old))


# ------------------------------------------------------------------------------------------------ partial worlds
def test_partial_discovery_worlds_cannot_reach_the_selection(tmp_path):
    """A discovery world holding only Part A (as after ``--phases parta``) is not a finished world: it is refused
    on load, by the selection and by the record validation, whatever its DONE.json claims."""
    # (a) no DONE.json, which is what the runner leaves behind: not loaded at all, and reported
    root = tmp_path / "a"
    for w in (0, 1):
        _write_world(root, w, "discovery", [2, 3, 4, 5], phases=("parta",), done=False)
    assert A.load_worlds(root, (0, 1)) == {}
    problems, worlds = V.validate_results(root)
    assert worlds == [] and sum("DONE.json is missing" in p for p in problems) == 2
    # (b) a DONE.json that lists only Part A while the protocol requires four phases
    root = tmp_path / "b"
    for w in (0, 1):
        _write_world(root, w, "discovery", [2, 3, 4, 5], phases=("parta",))
    with pytest.raises(ValueError, match="required phases"):
        A.load_worlds(root, (0, 1))
    assert all("required but not complete" in p for p in V.validate_results(root)[0])
    # (c) a DONE.json that declares Part A as all it needs: the selection and the validation know better
    root = tmp_path / "c"
    for w in (0, 1):
        _write_world(root, w, "discovery", [2, 3, 4, 5], phases=("parta",), required=("parta",))
    with pytest.raises(ValueError, match="required here but not complete"):
        A.select(A.load_worlds(root, (0, 1)))
    problems, worlds = V.validate_results(root)
    assert worlds == [] and any("phase 'episodic' is required but not complete" in p for p in problems)
    assert V.validate_world(root / "w0") == []               # alone, a world is only held to its own declaration
    select = _load("ka_run_select_partial", KA_DIR / "run_select.py")
    with pytest.raises(SystemExit):
        select.main(["--results", str(root), "--out", str(tmp_path / "sel.json")])
    assert not (tmp_path / "sel.json").exists()
    # (d) a candidate layer without its record is an error, not a G_F of zero
    root = tmp_path / "d"
    for w in (0, 1):
        d = _write_world(root, w, "discovery", [2, 3, 4, 5])
        recs = [r for r in WR.read_jsonl(d / "episodic.jsonl")
                if not (r["address"] == "PRED-RES@3" and r["read"] == "rule" and r["N"] == 100)]
        WR.write_jsonl_atomic(d / "episodic.jsonl", recs)
        _rehash(d)
    with pytest.raises(ValueError, match="no rule record of PRED-RES@3"):
        A.select(A.load_worlds(root, (0, 1)))
    assert any("episodic record missing (PRED-RES@3, rule, N=100)" in p for p in V.validate_results(root)[0])


def test_partial_confirmatory_worlds_cannot_reach_the_analysis(clone, tmp_path):
    """Eight confirmatory worlds, two of them without the FT phase: nothing is decided on the other six."""
    study = clone
    sel = _discovery(tmp_path)
    (study / "results").mkdir()
    (study / "results" / "selection.json").write_text(json.dumps(sel), encoding="utf-8")
    body = A.check_selection(sel)
    analyze = _load("ka_analyze_partial", KA_DIR / "analyze.py")
    no_ft = ("parta", "episodic", "checks")
    root = tmp_path / "confirm"
    for i in range(8):
        part = i in (2, 5)
        _write_world(root, 1000 + i, "confirm", [body["pred_layer"]], noise=0.01 * ((i % 3) - 1),
                     phases=no_ft if part else WR.PHASES, required=no_ft if part else None)
    with pytest.raises(ValueError, match=r"world 1002: phases \['ft'\] are required here"):
        A.decide(A.load_worlds(root), body)
    with pytest.raises(SystemExit, match="required here but not complete"):
        analyze.compute(root, study / "results" / "selection.json")
    problems, worlds = V.validate_results(root, study)
    assert worlds == [1000, 1001, 1003, 1004, 1006, 1007]
    assert sorted(p.split(":")[0] for p in problems) == ["w1002", "w1005"]
    assert any("decisions.json is missing" in p for p in V.rerun(root, study))
    # the same two worlds as the runner leaves them (no DONE.json): the analysis does not run on the rest
    for w in (1002, 1005):
        (root / f"w{w}" / "DONE.json").unlink()
    assert A.world_status(root, range(1000, 1008)) == {
        "complete": [1000, 1001, 1003, 1004, 1006, 1007], "excluded": {}, "incomplete": [1002, 1005], "absent": []}
    with pytest.raises(SystemExit, match=r"incomplete \[1002, 1005\]"):
        analyze.compute(root, study / "results" / "selection.json")
    # a world that was never run is not silently dropped either
    shutil.rmtree(root / "w1002")
    shutil.rmtree(root / "w1005")
    with pytest.raises(SystemExit, match=r"absent \[1002, 1005\]"):
        analyze.compute(root, study / "results" / "selection.json")
    # a base that failed CEV-11's gates is excluded explicitly by the driver: accounted for, analysed without it
    WR.write_excluded(root, 1002, "confirm", "the base failed CEV-11's gates")
    _write_world(root, 1005, "confirm", [body["pred_layer"]])
    out = analyze.compute(root, study / "results" / "selection.json")
    assert out["world_status"]["excluded"] == {"1002": "the base failed CEV-11's gates"}
    assert out["decisions"]["worlds"] == [1000, 1001, 1003, 1004, 1005, 1006, 1007]
    problems, worlds = V.validate_results(root, study)
    assert problems == [] and 1002 not in worlds and len(worlds) == 7
    # without Part A in the family, a confirmatory world needs only the three stream phases
    sel2 = A.check_selection(_discovery(tmp_path / "noh1", y_ctx=0.12, y_base=0.1))
    root2 = tmp_path / "confirm2"
    three = WR.required_phases("confirm", False)
    assert three == ("episodic", "ft", "checks")
    for i in range(8):
        _write_world(root2, 1000 + i, "confirm", [sel2["pred_layer"]], noise=0.01 * ((i % 3) - 1), phases=three,
                     required=three)
    dec = A.decide(A.load_worlds(root2), sel2)
    assert dec["primary"]["KA-H2"]["label"] == "memory better" and not dec["audit_required"]
    with pytest.raises(ValueError, match=r"phases \['parta'\] are required here"):
        A.decide(A.load_worlds(root2), body)                 # the same records under a selection that keeps KA-H1


def test_done_json_must_cover_every_record_file(tmp_path):
    """DONE.json after a single-phase re-run of the old runner listed two files; edits of the others went
    unnoticed. A record file that DONE.json does not hash is now refused."""
    d = _write_world(tmp_path, 1000, "confirm", [4])
    done = json.loads((d / "DONE.json").read_text())
    done["phases"] = ["checks"]
    done["required_phases"] = ["checks"]
    done["files"] = {f: done["files"][f] for f in ("setup.json", "checks.json")}
    (d / "DONE.json").write_text(json.dumps(done))
    problems = V.validate_world(d)
    assert sum("is not covered by DONE.json" in p for p in problems) == 3
    with pytest.raises(ValueError, match="is not covered by DONE.json"):
        A.load_world(d)
    # a listed phase whose file is gone, and a DONE.json without the phase bookkeeping
    d2 = _write_world(tmp_path, 1001, "confirm", [4])
    (d2 / "ft_stream.jsonl").unlink()
    assert any("ft_stream.jsonl is missing" in p for p in V.validate_world(d2))
    with pytest.raises(ValueError, match="listed in DONE.json but missing"):
        A.load_world(d2)
    d3 = _write_world(tmp_path, 1003, "confirm", [4])
    done = json.loads((d3 / "DONE.json").read_text())
    del done["required_phases"]
    (d3 / "DONE.json").write_text(json.dumps(done))
    assert any("has no 'required_phases'" in p for p in V.validate_world(d3))
    with pytest.raises(ValueError, match="required_phases"):
        A.load_world(d3)
    # phases that DONE.json does not list are not loaded
    d4 = _write_world(tmp_path, 1004, "confirm", [4], phases=("episodic", "ft", "checks"),
                      required=("episodic", "ft", "checks"))
    w = A.load_world(d4)
    assert w["part_a"] == [] and w["episodic"] and w["ft"] and w["checks"]


# ------------------------------------------------------------------------------------------------ decisions
def test_confirmatory_decisions_supported_case(tmp_path):
    sel = A.check_selection(_discovery(tmp_path))
    root = _confirm(tmp_path, layers=(sel["pred_layer"],), leak_ctx=0.02)
    dec = A.decide(A.load_worlds(root), sel)
    assert dec["worlds"] == list(range(1000, 1008))
    p = dec["primary"]
    assert p["KA-H1"]["label"] == S.SUPPORTED and p["KA-H1"]["p"] == 1 / 256
    assert p["KA-H1"]["guards"]["locality"]["pass"] and p["KA-H1"]["guards"]["editable"]["pass"]
    assert not p["KA-H1"]["guards"]["locality"]["exceeded"]
    assert p["KA-H2"]["label"] == "memory better" and p["KA-H2"]["guards"]["failed"] == []
    assert p["KA-H3"]["label"] == S.SUPPORTED
    assert p["KA-H3"]["manipulation_fixed_growth"]["informative"]
    assert dec["primary_holm"]["m"] == 3
    s = dec["secondary"]
    assert s["S1a"]["label"] == S.EQUIVALENT                  # PRED-ATT and PRED-RES within the SESOI
    assert s["S1b"]["label"] == S.SUPPORTED                   # prefix-partner leak 0.30 vs 0.10 at the matched point
    assert s["S2"]["label"] == S.SUPPORTED and set(s["S2"]["parts"]) == {"i", "ii", "iii"}
    # the consistency checks pass, so the labels stand
    assert dec["audit_required"] is False and dec["consistency"]["pass"]
    assert dec["consistency"]["pooled"]["cached_vs_hooked"] == {"disagree": 0, "n": 320, "share": 0.0}
    assert dec["consistency"]["pooled"]["batched_vs_greedy_part_a"]["n"] == 8 * 16 * 8
    # descriptive hierarchical bootstrap (worlds, then edits or facts) around the two contrasts
    boot = dec["descriptive"]["bootstrap"]
    assert boot["resamples"] == 10_000 and boot["KA-H1"]["unit"] == "edit" and boot["KA-H2"]["unit"] == "fact"
    for h in ("KA-H1", "KA-H2"):
        lo, hi = boot[h]["ci95"]
        assert boot[h]["worlds"] == list(range(1000, 1008)) and lo <= p[h]["mean"] <= hi
    # with no root-neighbour leak at all the leak correlation is undefined: S2 cannot be tested, and says so
    root0 = tmp_path / "noleak"
    for i in range(8):
        _write_world(root0, 1000 + i, "confirm", [sel["pred_layer"]], noise=0.01 * ((i % 3) - 1))
    assert A.decide(A.load_worlds(root0), sel)["secondary"]["S2"]["label"] == S.INSUFFICIENT
    assert set(dec["per_world"]["KA-H2"]) == {str(w) for w in range(1000, 1008)}
    text = A.render_tables(dec)
    assert "KA-H1" in text and "1007" in text and "AUDIT" not in text and "Hierarchical bootstrap" in text


def test_confirmatory_decisions_ft_better_and_guards(tmp_path):
    sel = A.check_selection(_discovery(tmp_path))
    root = _confirm(tmp_path, layers=(sel["pred_layer"],), fused_mem=0.4, fused_ft=0.8, ece=0.09,
                    leak_hi=0.05, y_ctx=0.9, leak_ctx=0.5)
    dec = A.decide(A.load_worlds(root), sel)
    p = dec["primary"]
    assert p["KA-H2"]["label"] == "FT better"
    assert p["KA-H3"]["label"] == S.REFUTED                   # ECE above 0.05 and Leak_U grows by more than 1 pt
    # A-CTX reaches the fused forms but spends five times the root-neighbour budget: locality guard
    assert p["KA-H1"]["label"] == A.LOCALITY_COST == "not supported (locality cost)"
    assert not p["KA-H1"]["guards"]["locality"]["pass"] and p["KA-H1"]["guards"]["locality"]["exceeded"]
    # a drifted FT model cannot be declared better
    root2 = tmp_path / "drift"
    for i in range(8):
        d = _write_world(root2, 1000 + i, "confirm", [sel["pred_layer"]], noise=0.01 * ((i % 3) - 1),
                         fused_mem=0.4, fused_ft=0.8)
        recs = [dict(r, dbpb=0.05 if r["dbpb"] is not None else None) for r in WR.read_jsonl(d / "ft_stream.jsonl")]
        WR.write_jsonl_atomic(d / "ft_stream.jsonl", recs)
        _rehash(d)
    dec2 = A.decide(A.load_worlds(root2), sel)
    assert dec2["primary"]["KA-H2"]["label"].startswith("not supported (FT better on G_F, guard failed: general ability")


def test_locality_guard_tells_a_cost_from_an_interval_that_is_too_wide(tmp_path):
    """The locality guard needs the upper 95% bound below budget + margin. When it fails, the label says
    'locality cost' only if the data show the budget exceeded; a mixture near its budget with the spread that
    eight worlds of sixteen edits give is 'not established', which is what the interval can support."""
    sel = A.check_selection(_discovery(tmp_path))
    leaks = [0.0, 0.25, 0.05, 0.2, 0.0, 0.25, 0.05, 0.1]      # mean X = 0.11, wide across worlds
    root = _confirm(tmp_path, layers=(sel["pred_layer"],), leak_by_world=leaks)
    h1 = A.decide(A.load_worlds(root), sel)["primary"]["KA-H1"]
    g = h1["guards"]["locality"]
    assert g["X_upper95"] > g["X_limit"] and g["X_lower95"] < g["X_budget"] == S.X_MAX
    assert not g["pass"] and not g["exceeded"]
    assert h1["label"] == A.LOCALITY_OPEN == "not supported (locality not established)"
    assert h1["holm_reject"] is True                          # the transfer contrast itself is clear


def test_consistency_failure_puts_every_label_under_audit(clone, tmp_path):
    """PROTOCOL §6: scorers that disagree on more than 1% of the probes mean an audit before anything is read."""
    study = clone
    sel = _discovery(tmp_path)
    (study / "results").mkdir()
    (study / "results" / "selection.json").write_text(json.dumps(sel), encoding="utf-8")
    body = A.check_selection(sel)
    root = _confirm(tmp_path, layers=(body["pred_layer"],), leak_ctx=0.02, disagree=1)        # 8 of 320: 2.5%
    dec = A.decide(A.load_worlds(root), body)
    assert dec["audit_required"] and dec["consistency"]["failures"] == ["cached_vs_hooked"]
    assert dec["consistency"]["pooled"]["cached_vs_hooked"]["share"] == pytest.approx(0.025)
    for fam in ("primary", "secondary"):
        assert {e["label"] for e in dec[fam].values()} == {A.AUDIT}
    assert dec["primary"]["KA-H2"]["label_before_audit"] == "memory better"
    assert "AUDIT REQUIRED" in A.render_tables(dec)
    analyze = _load("ka_analyze_audit", KA_DIR / "analyze.py")
    assert analyze.main(["--results", str(root), "--selection", str(study / "results" / "selection.json")]) == 2
    assert "AUDIT REQUIRED" in (root / "tables.md").read_text(encoding="utf-8")
    # one disagreement in one world (1 of 320) is within the rule; a changed base model never is
    d = root / "w1000"
    for w in range(1001, 1008):
        chk = json.loads((root / f"w{w}" / "checks.json").read_text())
        for a in chk["addresses"].values():
            a["cached_vs_hooked_disagree"] = 0
        (root / f"w{w}" / "checks.json").write_text(json.dumps(chk))
        _rehash(root / f"w{w}")
    ok = A.decide(A.load_worlds(root), body)
    assert not ok["audit_required"] and ok["consistency"]["pooled"]["cached_vs_hooked"]["disagree"] == 1
    chk = json.loads((d / "checks.json").read_text())
    chk["model_unchanged"] = False
    (d / "checks.json").write_text(json.dumps(chk))
    _rehash(d)
    assert "model_unchanged" in A.decide(A.load_worlds(root), body)["consistency"]["failures"]
    assert any("base model changed" in p for p in V.validate_results(root, study)[0])
    # Part A's own check: the batched exact check against true greedy decoding under an installed gate
    root2 = tmp_path / "gate"
    for i in range(8):
        d2 = _write_world(root2, 1000 + i, "confirm", [body["pred_layer"]], noise=0.01 * ((i % 3) - 1), leak_ctx=0.02)
        recs = [dict(r, greedy_check={"n": 8, "disagree": 1}) if "greedy_check" in r else r
                for r in WR.read_jsonl(d2 / "part_a.jsonl")]
        WR.write_jsonl_atomic(d2 / "part_a.jsonl", recs)
        _rehash(d2)
    assert A.decide(A.load_worlds(root2), body)["consistency"]["failures"] == ["batched_vs_greedy_part_a"]


def test_s2_uses_the_scored_sequence_indicator_and_reports_object_only_fires(tmp_path):
    """S2 reads the first flag of a row (the gate fires somewhere in the scored sequence). Probes that fire only
    at object positions are counted, and the same statistics with the prefix flag are reported next to it."""
    w = A.load_world(_write_world(tmp_path, 1000, "confirm", [4], leak_ctx=0.02, prefix_ball=False))
    s2 = A.part_a_summary(w)["S2"]
    assert s2["transfer_given_match"] == pytest.approx(1.0) and s2["n_matched"] > 0
    assert s2["prefix_indicator"]["transfer_given_match"] is None and s2["prefix_indicator"]["n_matched"] == 0
    assert s2["object_only"]["FUSED"] == s2["n_matched"] and s2["object_only"]["RN"] == 0
    assert s2["object_only"]["rows"] == 16 * 36 * (2 + 56 + 70)
    w2 = A.load_world(_write_world(tmp_path, 1001, "confirm", [4], leak_ctx=0.02))
    s2b = A.part_a_summary(w2)["S2"]
    assert s2b["object_only"]["FUSED"] == 0
    assert s2b["prefix_indicator"]["transfer_given_match"] == s2b["transfer_given_match"]


def test_confirmatory_decisions_need_six_worlds_and_respect_the_families(tmp_path):
    sel = A.check_selection(_discovery(tmp_path, y_ctx=0.12, y_base=0.1))      # H1 removed by the screen
    root = _confirm(tmp_path, layers=(sel["pred_layer"],), n=5)
    dec = A.decide(A.load_worlds(root), sel)
    assert dec["primary"]["KA-H1"]["label"].startswith("not in the confirmatory family")
    assert dec["primary_holm"]["m"] == 2
    assert dec["primary"]["KA-H2"]["label"] == S.INSUFFICIENT
    assert dec["primary"]["KA-H3"]["label"] == S.INSUFFICIENT
    assert list(dec["secondary"]["S2"]["parts"]) == ["i"]      # without Part A, S2 is S2(i) alone


# ------------------------------------------------------------------------------------------------ verify tooling
def _clone_study(tmp_path, source=None):
    """A copy of a study folder inside a fake repository layout, always in the NOT FROZEN state.

    Whatever state ``source`` is in (the real folder as committed, or a frozen folder with a manifest, a
    selection, amendments and logs), the clone has the status line 'Status: NOT FROZEN' and no freeze manifest,
    no results and no amendments. The freeze-dependent tests run on such a clone and never look at the real
    folder, so freezing the study cannot change their outcome."""
    source = KA_DIR if source is None else source
    repo = tmp_path / "repo"
    ka = repo / "research" / "concept_address_2026_11_13"
    shutil.copytree(source, ka, ignore=shutil.ignore_patterns("__pycache__", "results", "*.pyc", "*.log", "*.part",
                                                              V.FREEZE_NAME, "AMENDMENTS.md"))
    lines = (ka / "PROTOCOL.md").read_text(encoding="utf-8").split("\n")
    status = [i for i, line in enumerate(lines) if line.startswith("Status:")]
    assert len(status) == 1, "PROTOCOL.md carries exactly one status line"
    lines[status[0]] = "Status: NOT FROZEN"
    (ka / "PROTOCOL.md").write_text("\n".join(lines), encoding="utf-8")
    (repo / "tests" / "evolution").mkdir(parents=True)
    for p in (REPO / "tests" / "evolution").glob("test_concept_address_*.py"):
        shutil.copy(p, repo / "tests" / "evolution" / p.name)
    cev = repo / "research" / "living_learning_paradigm_2026_11_13"
    (cev / "cev11").mkdir(parents=True)
    (cev / "cev11" / "memory.py").write_text("# frozen engine stand-in\n")
    (cev / "freeze_manifest.sha256").write_text("")
    return ka


def _set_frozen(ka, date="2026-11-14"):
    text = (ka / "PROTOCOL.md").read_text(encoding="utf-8")
    assert text.count("Status: NOT FROZEN") == 1
    (ka / "PROTOCOL.md").write_text(text.replace("Status: NOT FROZEN", f"Status: FROZEN {date}", 1), encoding="utf-8")


def _freeze(ka, work, **kw):
    """The documented freeze procedure (PROTOCOL §8) on a clone: selection, status line, manifest."""
    sel = _discovery(work, **kw)
    (ka / "results").mkdir(exist_ok=True)
    (ka / "results" / "selection.json").write_text(json.dumps(sel), encoding="utf-8")
    _set_frozen(ka)
    assert V.freeze(ka) == []
    return sel


@pytest.fixture(params=["as committed", "frozen"])
def study(request, tmp_path):
    """A NOT FROZEN clone of the study folder: cloned from the folder as it is, and from a frozen copy of it
    (status FROZEN, selection, freeze manifest, AMENDMENTS.md, a job log). Both must behave the same."""
    if request.param == "as committed":
        return _clone_study(tmp_path)
    src = _clone_study(tmp_path / "frozen-source")
    _freeze(src, tmp_path / "frozen-source")
    (src / "AMENDMENTS.md").write_text("# Amendments\n\nA deviation recorded after the freeze.\n", encoding="utf-8")
    (src / "ka_confirm_w1000.log").write_text("log\n", encoding="utf-8")
    assert V.protocol_frozen(src) and V.check(src) == []
    return _clone_study(tmp_path, src)


@pytest.fixture
def clone(tmp_path):
    """A NOT FROZEN clone for tests that need a study folder of their own but not both source states."""
    return _clone_study(tmp_path)


def test_clone_is_not_frozen_whatever_the_source_state(study):
    assert not V.protocol_frozen(study)
    assert "Status: NOT FROZEN" in (study / "PROTOCOL.md").read_text(encoding="utf-8")
    assert not (study / V.FREEZE_NAME).exists() and not (study / "results").exists()
    assert not (study / "AMENDMENTS.md").exists() and not list(study.glob("*.log"))


def test_freeze_is_refused_until_the_protocol_is_frozen_and_selected(study):
    problems = V.freeze(study)
    assert any("NOT FROZEN" in p for p in problems) and any("selection.json" in p for p in problems)
    assert not (study / V.FREEZE_NAME).exists()
    assert any("not frozen" in p for p in V.preconditions(study))
    # frozen status alone is not enough: the selection must exist
    _set_frozen(study)
    assert V.protocol_frozen(study)
    assert V.freeze(study) == ["results/selection.json is missing (run the discovery jobs and run_select.py first)"]
    assert not (study / V.FREEZE_NAME).exists()


def test_freeze_is_refused_when_a_blocking_gate_failed(study, tmp_path):
    sel = _discovery(tmp_path, newest=0.5)                       # G-KA1c fails on discovery world 0
    (study / "results").mkdir()
    (study / "results" / "selection.json").write_text(json.dumps(sel), encoding="utf-8")
    _set_frozen(study)
    problems = V.freeze(study)
    assert len(problems) == 1 and "G-KA1c failed" in problems[0]
    assert not (study / V.FREEZE_NAME).exists()
    assert any("G-KA1c failed" in p for p in V.preconditions(study))


def test_freeze_check_detects_changed_missing_and_new_files(study, tmp_path):
    sel = _freeze(study, tmp_path)
    listed = V.read_manifest(study / V.FREEZE_NAME)
    assert "PROTOCOL.md" in listed and "ka/episodic.py" in listed and "results/selection.json" in listed
    assert "jobs_confirm.json" in listed and "../../tests/evolution/test_concept_address_codes.py" in listed
    assert "verify.py" in listed and "run_world.py" in listed and "ka/__init__.py" in listed
    assert "../living_learning_paradigm_2026_11_13/cev11/memory.py" in listed
    assert not any(doc in listed for doc in V.LIVING_DOCS) and V.FREEZE_NAME not in listed
    assert V.check(study) == []
    (study / "ka" / "stats.py").write_text((study / "ka" / "stats.py").read_text(encoding="utf-8") + "\n# edit\n",
                                           encoding="utf-8")
    (study / "ka" / "new_module.py").write_text("x = 1\n")
    (study / "sitecustomize.py").write_text("x = 1\n")
    (study / "jobs_extra.json").write_text("[]\n")
    (study / "jobs_discovery.json").unlink()
    problems = V.check(study)
    assert "changed: ka/stats.py" in problems and "missing: jobs_discovery.json" in problems
    assert "not in the manifest: ka/new_module.py" in problems and "not in the manifest: sitecustomize.py" in problems
    assert "not in the manifest: jobs_extra.json" in problems and len(problems) == 5
    # a tampered selection is caught by its own hash
    sel["body"]["pred_layer"] = 2
    (study / "results" / "selection.json").write_text(json.dumps(sel), encoding="utf-8")
    assert any("sha256" in p for p in V.preconditions(study))


def test_amendments_readme_and_stray_files_do_not_break_a_frozen_study(study, tmp_path):
    """PROTOCOL §8 sends post-freeze deviations to AMENDMENTS.md and the README has to follow the state of the
    study. Neither is in the freeze manifest, and files that are not code never block a confirmatory run."""
    _freeze(study, tmp_path)
    assert V.check(study) == [] and V.preconditions(study) == []
    (study / "AMENDMENTS.md").write_text("# Amendments\n\n2026-11-15: world 1003 was rerun after a power cut.\n",
                                         encoding="utf-8")
    (study / "README.md").write_text((study / "README.md").read_text(encoding="utf-8") + "\nFrozen; running.\n",
                                     encoding="utf-8")
    (study / "Thumbs.db").write_bytes(b"\x00\x01")
    (study / "ka_confirm_w1000.log").write_text("log\n", encoding="utf-8")
    (study / "notes.txt").write_text("scratch\n", encoding="utf-8")
    (study / "scratch.json").write_text("{}\n", encoding="utf-8")
    (study / "ka" / "__pycache__").mkdir()
    (study / "ka" / "__pycache__" / "stats.cpython-313.pyc").write_bytes(b"\x00")
    assert V.check(study) == [] and V.preconditions(study) == []
    # the driver gets past the preconditions (it then stops for lack of the weights, which no test touches)
    drv = _load("ka_run_world_frozen", study / "run_world.py")
    with pytest.raises(SystemExit) as exc:
        drv.main(["--run", "confirm", "--world", "1000", "--base", str(tmp_path / "no-such-base")])
    assert "model.pt not found" in str(exc.value) and "refused" not in str(exc.value)
    # the protocol itself and the code stay frozen
    (study / "PROTOCOL.md").write_text((study / "PROTOCOL.md").read_text(encoding="utf-8") + "\nedit\n",
                                       encoding="utf-8")
    assert V.check(study) == ["changed: PROTOCOL.md"]
    with pytest.raises(SystemExit) as exc:
        drv.main(["--run", "confirm", "--world", "1000", "--base", str(tmp_path / "no-such-base")])
    assert "confirmatory run refused" in str(exc.value) and "changed: PROTOCOL.md" in str(exc.value)


def test_freeze_is_never_overwritten_silently_and_never_after_confirmatory_records(study, tmp_path):
    _freeze(study, tmp_path)
    first = (study / V.FREEZE_NAME).read_bytes()
    (study / "ka" / "stats.py").write_text((study / "ka" / "stats.py").read_text(encoding="utf-8") + "\n# fix\n",
                                           encoding="utf-8")
    problems = V.freeze(study)
    assert len(problems) == 1 and "already exists" in problems[0] and "--refreeze" in problems[0]
    assert (study / V.FREEZE_NAME).read_bytes() == first             # untouched
    assert V.freeze(study, refreeze=True) == [] and V.check(study) == []
    second = (study / V.FREEZE_NAME).read_bytes()
    assert second != first
    # once a confirmatory record exists the manifest is final
    assert V.confirm_records(study) == []
    (study / "results" / "confirm" / "w1000").mkdir(parents=True)
    assert V.confirm_records(study) == ["w1000"]
    for refreeze in (False, True):
        problems = V.freeze(study, refreeze=refreeze)
        assert len(problems) == 1 and "confirmatory records exist" in problems[0] and "AMENDMENTS.md" in problems[0]
    assert (study / V.FREEZE_NAME).read_bytes() == second
    # a deleted manifest cannot be re-created after confirmatory records either
    (study / V.FREEZE_NAME).unlink()
    assert any("confirmatory records exist" in p for p in V.freeze(study))
    assert not (study / V.FREEZE_NAME).exists()


def test_record_validation_and_rerun(clone, tmp_path):
    study = clone
    sel = _discovery(tmp_path)
    (study / "results").mkdir()
    (study / "results" / "selection.json").write_text(json.dumps(sel), encoding="utf-8")
    body = A.check_selection(sel)
    root = _confirm(tmp_path, layers=(body["pred_layer"],))
    problems, worlds = V.validate_results(root, study)
    assert problems == [] and worlds == list(range(1000, 1008))
    # which phases a confirmatory world needs is read from the selection: without one nothing can be validated
    problems, worlds = V.validate_results(root, tmp_path / "no-study")
    assert worlds == [] and len(problems) == 8 and all("cannot tell which phases" in p for p in problems)
    analyze = _load("ka_analyze", KA_DIR / "analyze.py")
    out = analyze.compute(root, study / "results" / "selection.json")
    assert out["world_status"] == {"complete": list(range(1000, 1008)), "excluded": {}, "incomplete": [], "absent": []}
    WR.write_json_atomic(root / "decisions.json", out)
    again = [p for p in V.rerun(root, study) if "freeze" not in p and "analysis code" not in p]
    assert again == []
    # a changed number in the stored decisions is found
    stored = json.loads((root / "decisions.json").read_text())
    stored["decisions"]["primary"]["KA-H2"]["mean"] += 0.01
    (root / "decisions.json").write_text(json.dumps(stored))
    assert any("decisions differ" in p and "KA-H2" in p for p in V.rerun(root, study))
    # an incomplete edit, a missing record and a file changed after DONE.json are reported
    d = root / "w1003"
    recs = WR.read_jsonl(d / "part_a.jsonl")[:-1]
    WR.write_jsonl_atomic(d / "part_a.jsonl", recs)
    assert any("differs from DONE.json" in p for p in V.validate_world(d))
    _rehash(d)
    assert any("configurations" in p for p in V.validate_world(d))
    assert any("DONE.json is missing" in p for p in V.validate_world(tmp_path))
    # malformed in-ball rows (the old two-column format) are reported
    d = root / "w1004"
    recs = [dict(r, in_FUSED=[x[:2] for x in r["in_FUSED"]]) if "in_FUSED" in r else r
            for r in WR.read_jsonl(d / "part_a.jsonl")]
    WR.write_jsonl_atomic(d / "part_a.jsonl", recs)
    _rehash(d)
    assert any("in-ball rows malformed" in p for p in V.validate_world(d))
    # an unfinished FT arm and a missing no-null record are reported
    d = root / "w1005"
    WR.write_jsonl_atomic(d / "ft_stream.jsonl", [{k: v for k, v in r.items() if k != "arm_done"} if r["arm"] == "FT-ALL"
                                                  else r for r in WR.read_jsonl(d / "ft_stream.jsonl")])
    WR.write_jsonl_atomic(d / "episodic.jsonl", [r for r in WR.read_jsonl(d / "episodic.jsonl")
                                                 if not (r["read"] == "nonull" and r["N"] == 400)])
    _rehash(d)
    problems = V.validate_world(d)
    assert any("the FT-ALL stream is not finished" in p for p in problems)
    assert any("episodic record missing" in p and "nonull, N=400" in p for p in problems)
    # analyze refuses discovery worlds and smoke records
    with pytest.raises(SystemExit):
        analyze.compute(tmp_path / "discovery", study / "results" / "selection.json")


def test_command_line_entry_points(study, tmp_path, capsys):
    """run_select -> analyze -> verify through their main() functions, on synthetic record sets. The verify
    script is the clone's own, so every default path points into the clone."""
    select = _load("ka_run_select", KA_DIR / "run_select.py")
    analyze = _load("ka_analyze_cli", KA_DIR / "analyze.py")
    Vc = _load("ka_verify_clone", study / "verify.py")
    assert Vc.KA_DIR == study.resolve()
    disc = tmp_path / "discovery"
    for w in (0, 1):
        _write_world(disc, w, "discovery", [2, 3, 4, 5])
    sel_path = study / "results" / "selection.json"
    assert select.main(["--results", str(disc), "--out", str(sel_path)]) == 0
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    assert (study / "results" / "selection.sha256").read_text(encoding="utf-8").startswith(sel["sha256"])
    with pytest.raises(SystemExit):
        select.main(["--results", str(tmp_path / "empty"), "--out", str(sel_path)])      # a world is missing
    # a failed blocking gate: the selection is written, the exit code says no-go
    weak = tmp_path / "weak"
    for w in (0, 1):
        _write_world(weak, w, "discovery", [2, 3, 4, 5], newest=0.2)
    assert select.main(["--results", str(weak), "--out", str(tmp_path / "weak.json")]) == 2
    root = _confirm(tmp_path, layers=(sel["body"]["pred_layer"],), leak_ctx=0.02)
    assert analyze.main(["--results", str(root), "--selection", str(sel_path)]) == 0
    stored = json.loads((root / "decisions.json").read_text(encoding="utf-8"))
    assert stored["selection_sha256"] == sel["sha256"] and set(stored["code"]) >= {"analyze.py", "ka/stats.py"}
    assert "KA-H1" in (root / "tables.md").read_text(encoding="utf-8")
    capsys.readouterr()
    assert Vc.main(["--records", "--results", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["valid_worlds"] == list(range(1000, 1008))
    assert Vc.main(["--records", "--results", str(tmp_path / "empty")]) == 1                # nothing to validate
    assert Vc.main(["--check"]) == 1 and Vc.main(["--preconditions"]) == 1                  # the clone is not frozen
    assert Vc.main(["--freeze"]) == 1 and not (study / V.FREEZE_NAME).exists()
    # the documented freeze: status line, --freeze, --check, --preconditions; then no silent second freeze
    _set_frozen(study)
    assert Vc.main(["--freeze"]) == 0 and (study / V.FREEZE_NAME).exists()
    assert Vc.main(["--check"]) == 0 and Vc.main(["--preconditions"]) == 0
    assert Vc.main(["--freeze"]) == 1 and Vc.main(["--freeze", "--refreeze"]) == 0
    assert Vc.main(["--rerun", "--results", str(root)]) == 0                                # frozen code = analysis code
    with pytest.raises(SystemExit):
        Vc.main(["--check", "--refreeze"])
    capsys.readouterr()


def test_job_files_are_well_formed():
    for name, n_worlds in (("jobs_discovery.json", 2), ("jobs_confirm.json", 8)):
        jobs = json.loads((KA_DIR / name).read_text(encoding="utf-8"))
        ids = [j["id"] for j in jobs]
        assert len(ids) == len(set(ids))
        for j in jobs:
            assert set(j) >= {"id", "cmd", "deps", "log"}
            assert all(d in ids for d in j["deps"])
            assert "research/concept_address_2026_11_13/" in j["cmd"]          # runnable from the repository root
            assert not j["cmd"].startswith("/") and "/workspace" not in j["cmd"] and "/tmp" not in j["cmd"]
            assert "--phases" not in j["cmd"]                                  # a job always runs a whole world
        worlds = [j for j in jobs if "run_world.py" in j["cmd"]]
        assert len(worlds) == n_worlds
        assert all("CEV_BASES" in j["cmd"] for j in worlds)
    confirm = json.loads((KA_DIR / "jobs_confirm.json").read_text(encoding="utf-8"))
    pre = [j for j in confirm if "--preconditions" in j["cmd"]]
    assert len(pre) == 1 and all(pre[0]["id"] in j["deps"] for j in confirm if "run_world.py" in j["cmd"])


def test_driver_refuses_wrong_worlds_and_unfrozen_confirmatory_runs(study, monkeypatch):
    """No weights are touched: every refusal happens before a base is loaded. The driver is the clone's own, so
    'not frozen' is the state of the clone, not of the real study folder."""
    drv = _load("ka_run_world", study / "run_world.py")
    assert drv.KA_DIR == study.resolve()
    monkeypatch.delenv("CEV_BASES", raising=False)
    for argv, text in ((["--run", "discovery", "--world", "1000"], "worlds 0 and 1 only"),
                       (["--run", "confirm", "--world", "0"], "discovery worlds cannot be used"),
                       (["--run", "confirm", "--world", "2000"], "confirmatory worlds are"),
                       (["--run", "confirm", "--world", "1000", "--smoke"], "discovery worlds only"),
                       (["--run", "confirm", "--world", "1000"], "confirmatory run refused"),
                       (["--run", "discovery", "--world", "0"], "CEV_BASES")):
        with pytest.raises(SystemExit) as exc:
            drv.main(argv)
        assert text in str(exc.value), argv
    monkeypatch.setenv("CEV_BASES", "/nonexistent-bases")
    assert drv.base_dir_for("discovery", 1, None) == pathlib.Path("/nonexistent-bases/discovery_w1_K20")
    assert drv.base_dir_for("confirm", 1003, None) == pathlib.Path("/nonexistent-bases/confirm_w1003_K20")
    with pytest.raises(SystemExit) as exc:
        drv.main(["--run", "discovery", "--world", "0"])
    assert "model.pt not found" in str(exc.value)


def test_driver_skips_part_a_on_confirmatory_worlds_when_h1_was_screened_out(study, tmp_path):
    _freeze(study, tmp_path, y_ctx=0.12, y_base=0.1)             # the discovery screen removes KA-H1
    drv = _load("ka_run_world_noh1", study / "run_world.py")
    with pytest.raises(SystemExit) as exc:
        drv.main(["--run", "confirm", "--world", "1000", "--phases", "parta", "--base", str(tmp_path / "none")])
    assert "none of the requested phases" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        drv.main(["--run", "confirm", "--world", "1000", "--base", str(tmp_path / "none")])
    assert "model.pt not found" in str(exc.value)


def test_cev11_blinding_condition_is_recorded():
    """CEV-11's confirmatory analysis hash is in its AMENDMENTS.md, so KA may open the confirmatory bases once
    its own protocol is frozen (PROTOCOL §8)."""
    assert V.cev11_analysis_recorded()


# ------------------------------------------------------------------------------------------------ FT comparators
@pytest.fixture(scope="module")
def base():
    return tiny_base(layers=2, d=32, seed=0)


@pytest.fixture(scope="module")
def base3():
    """Three blocks: a write at block 1 has a block above it. In a two-block model block 1 is the last one, a
    change at the subject position there cannot reach the prediction position, and the delta search returns
    exactly zero — every Part A write would be the empty write."""
    return tiny_base(layers=3, d=32, seed=0)


def test_cev11_ft_configs_are_the_frozen_selection():
    cfg = FT.cev11_ft_configs()
    assert cfg["FT-L"] == {"layer": 0, "hp": {"eps": 0.02, "lr": 0.001, "steps": 60}, "config": "eps=0.02,lr=0.001,steps=60"}
    assert cfg["FT-ALL"]["layer"] is None and cfg["FT-ALL"]["hp"] == {"lr": 0.0001, "steps": 40}


def test_ft_stream_step_is_the_frozen_optimiser_on_the_current_weights(base):
    from cev11.baselines import ft_layer
    model, tok = base.model, base.tok
    pool = sorted(base.world.members("probe"))[:20]
    facts = ST.make_stream(base.world, pool, seed=3, n_max=3)
    cfg = {"layer": 0, "hp": {"eps": 0.02, "lr": 0.01, "steps": 3}, "config": "t"}
    W = model.layers[0].ffn.fc2.weight
    original = {k: v.clone() for k, v in model.state_dict().items()}
    try:
        w0 = W.detach().clone()
        prompts, target, kl = FT.fact_prompt(tok, facts[0])
        expect1 = ft_layer(model, 0, prompts, target, 0.01, 3, 0.02, kl_prompt_ids=kl)
        assert torch.equal(W.detach(), w0)                               # the frozen optimiser restores W
        FT.ft_step(model, tok, facts[0], "FT-L", cfg)
        assert torch.equal(W.detach(), w0 + expect1)
        # the second fact starts from the weights the first one left (sequential, current weights)
        w1 = W.detach().clone()
        prompts, target, kl = FT.fact_prompt(tok, facts[1])
        expect2 = ft_layer(model, 0, prompts, target, 0.01, 3, 0.02, kl_prompt_ids=kl)
        FT.ft_step(model, tok, facts[1], "FT-L", cfg)
        assert torch.equal(W.detach(), w1 + expect2)
        assert float((W.detach() - w0).abs().max()) > 0
    finally:
        model.load_state_dict(original)
    assert model_sha256(model) == base.base_sha


def test_ft_stream_checkpoints_and_bitwise_restore(base, monkeypatch):
    model, tok = base.model, base.tok
    pool = sorted(base.world.members("probe"))[:20]
    facts = ST.make_stream(base.world, pool, seed=3, n_max=4)
    seen, written = [], []
    real = FT.newest_produced

    def spy(m, t, fact):
        written.append((fact.fact_id, real(m, t, fact)))
        return written[-1][1]

    monkeypatch.setattr(FT, "newest_produced", spy)

    def at(n):
        seen.append((n, model_sha256(model) != base.base_sha))
        return {"probe": n}

    for arm, cfg in (("FT-L", {"layer": 0, "hp": {"eps": 0.02, "lr": 0.01, "steps": 2}, "config": "a"}),
                     ("FT-ALL", {"layer": None, "hp": {"lr": 0.001, "steps": 2}, "config": "b"})):
        written.clear()
        recs = FT.ft_stream(model, tok, facts, arm, cfg, (2, 4, 9), at)
        assert [r["N"] for r in recs] == [2, 4] and all(r["arm"] == arm for r in recs)
        # the newest fact is checked right after its own write, once per fact, in stream order
        assert [fid for fid, _ in written] == [f.fact_id for f in facts]
        for r in recs:
            assert r["newest_efficacy"] == sum(ok for _, ok in written[:r["N"]]) / r["N"]
        assert model_sha256(model) == base.base_sha                      # restored bitwise after the stream
    assert seen == [(2, True), (4, True), (2, True), (4, True)]          # at a checkpoint the model holds the facts
    with pytest.raises(RuntimeError):
        FT.ft_stream(model, tok, facts, "FT-L", {"layer": 0, "hp": {"eps": 0.02, "lr": 0.01, "steps": 1}, "config": "a"},
                     (1,), lambda n: (_ for _ in ()).throw(RuntimeError("boom")))
    assert model_sha256(model) == base.base_sha                          # also restored when a checkpoint fails


def test_newest_produced_is_the_exact_check_of_the_lived_prompt(base):
    from cev11.decode import greedy_ids
    model, tok = base.model, base.tok
    fact = ST.make_stream(base.world, sorted(base.world.members("probe"))[:20], seed=3, n_max=1)[0]
    prompts, target, _ = FT.fact_prompt(tok, fact)
    assert FT.newest_produced(model, tok, fact) == (greedy_ids(model, prompts[0][0], max_new=len(target)) == target)
    assert FT.newest_produced(model, tok, fact) is False                 # a random model does not know the fact


# ------------------------------------------------------------------------------------------------ Part A mechanics
def _part_a(base, addresses, qs, thresholds, refs=(), layer=1):
    torch.manual_seed(0)
    sup = (torch.randn(32, dtype=torch.float64), torch.linalg.qr(torch.randn(32, 8, dtype=torch.float64))[0])
    return PA.PartA(base, 11, thresholds, sup, {}, layer=layer, steps=(1.0,), qs=qs, addresses=addresses, refs=refs,
                    max_steps=3, greedy_n=0)


def _fire_of_installed_gate(base, gm, key, theta, key_ids, scored):
    """(fires in the scored sequence, fires in the prefix) per probe, read off the installed gate itself. The
    sequences go through the model in the batches ``capture`` used (the key sequence first), so the codes the
    gate reads are bit for bit the ones the indicator was computed from."""
    seqs = [(list(key_ids), len(key_ids))] + list(scored)
    gm.set([(key, theta, torch.ones(32))])
    got = []
    try:
        for s0 in range(0, len(seqs), 64):
            chunk = seqs[s0:s0 + 64]
            with torch.no_grad():
                base.model(C.pad_batch([ids for ids, _ in chunk]))
            mask = gm.last_fire[0]
            got += [(bool(mask[j, :len(ids)].any()), bool(mask[j, :n_pre].any())) for j, (ids, n_pre) in enumerate(chunk)]
    finally:
        gm.clear()
    return got[1:]


def test_in_ball_indicator_is_where_the_gate_fires_in_the_scored_pass(base):
    """The pre-edit indicator is computed on the sequence the exact check runs (prefix plus teacher-forced object
    tokens). It equals, probe by probe, what the installed gate does in that pass, object positions included."""
    from cev11.keysites import site_prompts
    item = PA.test_items(base, sorted(base.world.members("probe")), 11, 2)[0]
    probes = PA.build_probes(base, item, 33)["probes"]
    scored = [PA.scored_ids(base.tok, p) for p in probes]
    for (ids, n_pre), p in zip(scored, probes):                           # exactly what exact_check_batch feeds
        assert ids[:n_pre] == [BOS_ID] + base.tok.encode(p["prefix"]) and len(ids) > n_pre
        assert ids[n_pre:] == base.tok.encode(" " + p["gold"] + ".")[:-1]
    lived = frame_prefix(item["name"], item["rel"], item["lived_case"], "formal", "none")
    key_ids, key_pos = site_prompts(base.tok, lived, item["name"], "none", site="STD")[0]
    q = 0.05
    thresholds = {"key": {q: 0.1}, "res": {q: 0.3}, "att": {q: 0.6}, "sup": {q: 0.2}}
    pa = _part_a(base, G.GATE_ADDRESSES, (q,), thresholds)
    try:
        keys, fire = pa._in_ball(probes, key_ids, key_pos)
        for a, gm in pa.gates.items():
            flags = fire[(a, q)]
            assert len(flags) == len(probes) and all(s or not pfx for s, pfx in flags)      # prefix fire => fire
            assert flags == _fire_of_installed_gate(base, gm, keys[a], thresholds[G.GATE_CODE[a]][q], key_ids, scored)
            assert pa._lived_fire[(a, q)] >= 1                             # the gate fires at its own key
        assert any(s for a in pa.gates for s, _ in fire[(a, q)])
        # a key taken at an object position, radius ~ 0: that probe is in the ball although no prefix position
        # is. The old prefix-only indicator recorded it as "not in the ball" while the write landed.
        ids0, n0 = scored[0]
        tight = {code: {q: 0.9999} for code in thresholds}
        pa.thresholds = tight
        keys, fire = pa._in_ball(probes, ids0, n0)
        for a, gm in pa.gates.items():
            flags = fire[(a, q)]
            assert flags[0] == (True, False)
            assert flags == _fire_of_installed_gate(base, gm, keys[a], 0.9999, ids0, scored)
    finally:
        pa.close()
    assert model_sha256(base.model) == base.base_sha
    assert sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in base.model.modules()) == 0


def test_a_tok_is_cev11_std_write_bitwise(base3, monkeypatch):
    """Programme test 3 / check M3 on the tiny model: what Part A installs and scores as A-TOK is CEV-11's frozen
    STD write for the same item and step — the same matrix and the same logits, bit for bit."""
    from cev11 import experiment as E
    base = base3
    model, layer, step = base.model, 1, 1.0
    item = PA.test_items(base, sorted(base.world.members("probe")), 11, 2)[0]
    x = C.pad_batch([[BOS_ID] + base.tok.encode(frame_prefix(item["name"], "A", c)) for c in ("nom", "gen", "abl")])
    with torch.no_grad():
        clean = model(x)[0].clone()
    pa = _part_a(base, ("A-TOK",), (), {})
    seen = {}
    real = PA.score_probes

    def spy(b, probes):
        seen["M"] = pa.inst.mems[layer].M.detach().clone()
        with torch.no_grad():
            seen["logits"] = model(x)[0].clone()
        return real(b, probes)

    monkeypatch.setattr(PA, "score_probes", spy)
    try:
        recs = pa.run_edit(item)
    finally:
        pa.close()
    assert [r["config"] for r in recs] == ["A-TOK|step=1"] and model_sha256(model) == base.base_sha
    # CEV-11's own path for its STD arm: compute_write(..., "STD", ...) installed through its Installer
    w = E.compute_write(base, item, "STD", layer, {"step": step, "cap_mult": G.CAP_MULT, "max_steps": 3}, 11, None)
    inst = E.Installer(base)
    try:
        inst.set_M(layer, w["M"])
        ref_M = inst.mems[layer].M.detach().clone()
        with torch.no_grad():
            ref_logits = model(x)[0].clone()
    finally:
        inst.restore()
        for m in inst.mems.values():
            m.remove()
    assert float(ref_M.abs().max()) > 0 and not torch.equal(ref_logits, clean)       # a real write
    assert torch.equal(seen["M"], ref_M) and torch.equal(seen["logits"], ref_logits)
    with torch.no_grad():
        assert torch.equal(model(x)[0], clean)                                       # and it is gone afterwards


# ------------------------------------------------------------------------------------------------ driver + resume
TINY = dict(protocol=False, n_test=2, steps=(1.0,), qs=(0.01, 0.05), n_stream=12, checkpoints=(4, 8),
            ft_checkpoints=(4,), n_primary=4, pred_layers=(1,), c0_entities=8, threshold_seqs=120,
            geometry_entities=12, bpb_paragraphs=2, search_max_steps=3, n_u=10, gate_layer=1)


def test_required_phases():
    assert WR.required_phases("discovery") == WR.PHASES == ("parta", "episodic", "ft", "checks")
    assert WR.required_phases("discovery", False) == WR.PHASES          # the screen applies to confirmatory worlds
    assert WR.required_phases("confirm") == WR.PHASES
    assert WR.required_phases("confirm", False) == ("episodic", "ft", "checks")
    with pytest.raises(ValueError):
        WR.required_phases("smoke")


def test_world_driver_runs_end_to_end_and_resumes(base3, tmp_path):
    base = base3                                                         # block 1 of 3: the writes are not empty
    cfg = WR.RunConfig(**TINY)
    log = []
    out = tmp_path / "w11"
    # a run restricted to one phase does not finish the world: no DONE.json, nothing for the selection to load
    part = WR.run_world(base, 11, "discovery", cfg, tmp_path, phases=["parta"], log=log.append)
    assert not part.done and not (out / "DONE.json").exists() and part.complete_phases() == ["parta"]
    assert A.load_worlds(tmp_path) == {} and any("DONE.json is missing" in p for p in V.validate_world(out))
    first_part_a = WR.sha256_file(out / "part_a.jsonl")
    log.clear()
    wr = WR.run_world(base, 11, "discovery", cfg, tmp_path, log=log.append)
    assert wr.done and (out / "DONE.json").exists() and model_sha256(base.model) == base.base_sha
    assert not any("part A" in m for m in log)                           # Part A was already there
    assert sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in base.model.modules()) == 0
    assert V.validate_world(out) == [] and V.validate_results(tmp_path) == ([], [11])
    setup = json.loads((out / "setup.json").read_text())
    assert setup["config"]["protocol"] is False and set(setup["thresholds"]) == {"att", "key", "res", "sup"}
    for th in setup["thresholds"].values():
        assert th["0.01"] > th["0.05"]                                   # tighter false-match rate, tighter radius
    recs = WR.read_jsonl(out / "part_a.jsonl")
    expected = PA.expected_configs(cfg.addresses, cfg.steps, cfg.qs, cfg.ft_refs)
    assert len(recs) == 2 * len(expected) == 2 * (1 + 4 * 2 + 3)
    gate = [r for r in recs if r["address"] == "A-CTX"]
    assert all(r["lived_fire_positions"] >= 1 for r in gate)             # the gate always fires at its own key
    assert all(r["delta_norm"] > 0 for r in recs if r["address"] in G.ADDRESSES)      # real writes, not empty ones
    for r in recs:
        if r["address"] in G.GATE_ADDRESSES:                             # [scored sequence, outcome, prefix]
            for fam in PA.IN_BALL_FAMILIES:
                assert len(r[f"in_{fam}"]) == r[fam]["n"] and all(len(x) == 3 and x[2] <= x[0] for x in r[f"in_{fam}"])
    epi = WR.read_jsonl(out / "episodic.jsonl")
    assert {r["read"] for r in epi} == {"rule", "fixed", "nonull", "frontier"}
    for r in epi:                                                        # G_F per fact, kept for the bootstrap
        assert ("fused_facts" in r) == (r["read"] == "rule")
        if r["read"] == "rule":
            ff = r["fused_facts"]
            assert len(ff) == r["n_fused"] and all(0 <= i < r["N"] and ok in (0, 1) for i, ok in ff)
            assert r["fused"] is None if not ff else r["fused"] == pytest.approx(sum(ok for _, ok in ff) / len(ff))
    ft = WR.read_jsonl(out / "ft_stream.jsonl")
    assert {(r["arm"], r["N"]) for r in ft} == {("FT-L", 4), ("FT-ALL", 4)}
    assert all(0.0 <= r["newest_efficacy"] <= 1.0 and "seconds" not in r and "fused_facts" in r for r in ft)
    chk = json.loads((out / "checks.json").read_text())
    assert chk["model_unchanged"] and chk["addresses"]["PRED-RES@1"]["cached_vs_hooked_disagree"] == 0
    done = json.loads((out / "DONE.json").read_text())
    assert done["phases"] == list(WR.PHASES) == done["required_phases"] and done["ka_version"] == KA_VERSION
    assert set(done["files"]) == set(PHASE_FILES) and done["files"]["part_a.jsonl"] == first_part_a
    assert {"setup", "parta", "episodic", "ft", "ft_FT-L", "ft_FT-ALL", "checks"} <= set(done["seconds"])
    before = dict(done["files"])
    # a completed world is not recomputed: identical files, no Part A work
    log.clear()
    WR.run_world(base, 11, "discovery", cfg, tmp_path, log=log.append)
    assert json.loads((out / "DONE.json").read_text())["files"] == before
    assert not any("part A" in m for m in log)
    # a single-phase re-run of a finished world leaves DONE.json describing the whole directory
    again = WR.run_world(base, 11, "discovery", cfg, tmp_path, phases=["checks"], log=log.append)
    done2 = json.loads((out / "DONE.json").read_text())
    assert again.done and done2["phases"] == list(WR.PHASES) and done2["files"] == before
    # ... so a record edited afterwards is still caught
    original = (out / "episodic.jsonl").read_bytes()
    (out / "episodic.jsonl").write_bytes(original.replace(b'"read": "rule"', b'"read": "fixed"', 1))
    assert any("episodic.jsonl differs from DONE.json" in p for p in V.validate_world(out))
    with pytest.raises(ValueError, match="episodic.jsonl does not match"):
        A.load_world(out)
    (out / "episodic.jsonl").write_bytes(original)
    assert V.validate_world(out) == []
    # an interrupted run: a torn line and a partial edit are dropped, the missing edit is recomputed identically;
    # the FT streams carry no wall-clock time, so they come back byte for byte as well
    lines = (out / "part_a.jsonl").read_text(encoding="utf-8").split("\n")
    keep = len(expected) + 3
    (out / "part_a.jsonl").write_text("\n".join(lines[:keep]) + "\n" + lines[keep][:40], encoding="utf-8")
    (out / "DONE.json").unlink()
    (out / "checks.json").unlink()
    (out / "ft_stream.jsonl").unlink()
    log.clear()
    wr2 = WR.run_world(base, 11, "discovery", cfg, tmp_path, log=log.append)
    assert sum("part A" in m for m in log) == 1                          # only the incomplete edit ran again
    after = json.loads((out / "DONE.json").read_text())["files"]
    assert after == before                                               # deterministic: byte-identical records
    assert wr2.setup["partitions"] == wr.setup["partitions"]
    # the phases a world needs are those the protocol requires of it, not those of the last invocation
    three = WR.required_phases("confirm", False)
    wr3 = WR.WorldRun(base, 11, "discovery", cfg, tmp_path, required=three)
    (out / "part_a.jsonl").rename(out / "part_a.keep")
    assert wr3.complete() and json.loads((out / "DONE.json").read_text())["phases"] == list(three)
    assert "part_a.jsonl" not in json.loads((out / "DONE.json").read_text())["files"]
    assert V.validate_world(out) == [] and any("phase 'parta' is required" in p for p in V.validate_results(tmp_path)[0])
    (out / "checks.json").rename(out / "checks.keep")
    assert not wr3.complete() and not (out / "DONE.json").exists()       # a stale DONE.json is removed
    (out / "checks.keep").rename(out / "checks.json")
    (out / "part_a.keep").rename(out / "part_a.jsonl")
    assert WR.WorldRun(base, 11, "discovery", cfg, tmp_path).complete()
    assert json.loads((out / "DONE.json").read_text())["files"] == before
    # a different configuration must not be mixed into the same directory
    with pytest.raises(RuntimeError):
        WR.run_world(base, 11, "discovery", WR.RunConfig(**{**TINY, "n_stream": 10}), tmp_path, log=log.append)
    with pytest.raises(ValueError):
        WR.run_world(base, 11, "discovery", cfg, tmp_path, phases=["partb"], log=log.append)
    # non-protocol (smoke) records are refused by the selection
    with pytest.raises(ValueError):
        A.select(A.load_worlds(tmp_path))
    # a world marked as excluded is never run
    WR.write_excluded(tmp_path, 12, "discovery", "test")
    with pytest.raises(RuntimeError, match="EXCLUDED"):
        WR.WorldRun(base, 12, "discovery", cfg, tmp_path)


def test_part_a_items_are_deterministic_and_leave_a_fused_case(base):
    from cev11.tokenizers import classify_informal
    probe = sorted(base.world.members("probe"))
    items = PA.test_items(base, probe, 11, 6)
    assert items == PA.test_items(base, probe, 11, 6)
    assert len({it["name"] for it in items}) == len(items)
    for it in items:
        fused = [c for c in ("gen", "dat", "acc", "loc", "abl", "ins") if classify_informal(base.tok, it["name"], c) == "FUSED"]
        assert any(c != it["lived_case"] for c in fused)                 # a FUSED form other than the lived one
        assert it["new"] != it["old"] and it["name"] in PA.eligible_entities(base, probe)
    rec = {"WF": {"ok": 1, "n": 1}, "PF6": {"ok": 3, "n": 6}, "FUSED": {"ok": 1, "n": 2}, "IDENT": {"ok": 0, "n": 0},
           "CTX": {"ok": 2, "n": 2}, "RN": {"changed": 14, "n": 56}, "RD": {"changed": 7, "n": 70},
           "C": {"changed": 7, "n": 14}, "PX": {"changed": 0, "n": 0}}
    m = PA.cell_metrics(rec)
    assert m == {"E": 1.0, "Y": 0.5, "X": pytest.approx(0.15), "R": 0.1, "RN": 0.25, "C": 0.5, "PE6": 0.5,
                 "IDENT": 0.0, "PX": 0.0, "CTX": 1.0}
