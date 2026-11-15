# -*- coding: utf-8 -*-
"""KA study — the output-side episodic memory: exact ledger, null-gated read, continuation, cached evaluator.

Tiny random model (readout boosted so that analytic values can reach the target margin). No trained weights.
"""
import copy
import pathlib
import sys

import pytest
import torch

KA_DIR = pathlib.Path(__file__).resolve().parents[2] / "research" / "concept_address_2026_11_13"
if str(KA_DIR) not in sys.path:
    sys.path.insert(0, str(KA_DIR))

from ka import codes as C  # noqa: E402
from ka import episodic as EP  # noqa: E402
from ka import streams as ST  # noqa: E402
from ka.testing import tiny_base  # noqa: E402
from cev11.decode import exact_check_ids, greedy_ids  # noqa: E402
from cev11.model import model_sha256  # noqa: E402
from cev11.world import RELATIONS, frame_prefix  # noqa: E402
from cev11.keysites import VARIANTS  # noqa: E402
from tokenizer_management.byte_bpe import BOS_ID  # noqa: E402

LAYER = 1
# a random readout needs a wide cap to reach a margin; the protocol values (m* = 6, c_cap = 1) are for trained bases
MV = dict(m_star=1.5, c_cap=50.0)


@pytest.fixture(autouse=True)
def _single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def base():
    return tiny_base(layers=3, d=32, seed=0, boost=10.0)


@pytest.fixture(scope="module")
def address(base):
    """PRED-RES address fitted on the c0 partition of the tiny world (write-free)."""
    c0 = sorted(base.world.members("c0"))[:10]
    items = [{"name": n, "rel": r, "case": c, "orth": o, "ctx": "none"} for n in c0 for r in RELATIONS for c, o in VARIANTS]
    seqs = [[BOS_ID] + base.tok.encode(frame_prefix(it["name"], it["rel"], it["case"], it["orth"])) for it in items]
    recs = C.capture(base.model, seqs, [("post", LAYER)], pick=[[len(s) - 1] for s in seqs])
    return C.fit_address(torch.stack([r[f"post{LAYER}"][0] for r in recs]), items, f"PRED-RES@{LAYER}", "post", LAYER, 5)


@pytest.fixture(scope="module")
def sharp(address):
    """The same address with a 20x steeper calibration: on this random model the self score of a key is about +3
    and other positions score below 0, so with theta = 0 the own anchor gets rho = 1 and nothing else fires."""
    import dataclasses
    return dataclasses.replace(address, a=address.a * 20.0, c=address.c * 20.0)


SELECT = EP.ReadRule("fixed", 0.0)


def _facts(base, n=4):
    pool = sorted(base.world.members("probe"))[:20]
    return ST.make_stream(base.world, pool, seed=3, n_max=n)


def _ids(base, fact):
    return [BOS_ID] + base.tok.encode(fact.prefix), base.tok.encode(" " + fact.new + ".")


def _logits(model, seqs):
    with torch.no_grad():
        return model(C.pad_batch(seqs))[0].clone()


# ------------------------------------------------------------------------------------------------ readout, values
def test_readout_replica_equals_model_logits(base):
    """Programme test 9: the float64 replica of the head equals the model's logits to 1e-5 (logits of size ~8
    here: float32 rounding). Against a float64 copy of the model it is the same function, to 1e-9."""
    seqs = [_ids(base, f)[0] for f in _facts(base)]
    for model, tol in ((base.model, 1e-5), (copy.deepcopy(base.model).double(), 1e-9)):
        with C.Tap(model, [("x", 0)]) as tap, torch.no_grad():
            logits = model(C.pad_batch(seqs))[0]
            x = tap.out["x"].clone()
        assert float(logits.abs().max()) > 1.0
        assert float((EP.Readout(model)(x) - logits.double()).abs().max()) <= tol


def test_analytic_value_reaches_the_margin_or_flags_the_cap(base):
    ro = EP.Readout(base.model)
    f = _facts(base)[0]
    ids = _ids(base, f)[0]
    x = C.capture(base.model, [ids], [("x", 0)])[0]["x"][-1]
    base_top = int(ro(x).argmax())
    other = (base_top + 7) % ro.E.shape[0]
    r, info = EP.analytic_value(ro, x, other, m_star=2.0, c_cap=1.0)
    assert info["margin0"] < 2.0
    if info["cap_hit"]:
        assert float(r.norm()) == pytest.approx(float(x.norm()), rel=1e-9)
    else:
        m = EP.margin(ro(x + r), other)
        assert 2.0 <= m < 2.0 + 1e-3                      # the smallest step that reaches m*
        assert float(r.norm()) <= float(x.norm())
    # a token that already holds the margin needs no value
    r0, info0 = EP.analytic_value(ro, x, base_top, m_star=min(2.0, info0_margin(ro, x, base_top)), c_cap=1.0)
    assert float(r0.norm()) == 0.0 and not info0["cap_hit"]
    # a cap too small to reach the margin is reported, never hidden
    r2, info2 = EP.analytic_value(ro, x, other, m_star=2.0, c_cap=1e-4)
    assert info2["cap_hit"] and float(r2.norm()) == pytest.approx(1e-4 * float(x.norm()), rel=1e-9)


def info0_margin(ro, x, y):
    return EP.margin(ro(x), y)


# ------------------------------------------------------------------------------------------------ exact reduction
def test_empty_store_is_bit_identical_and_outside_the_module_tree(base, address):
    model = base.model
    facts = _facts(base)
    seqs = [_ids(base, f)[0] for f in facts]
    ref = _logits(model, seqs)
    ref_ids = greedy_ids(model, seqs[0], max_new=6, use_cache=True)
    keys = list(model.state_dict().keys())
    cortex = EP.EpisodicCortex(model, address).attach()
    try:
        assert torch.equal(_logits(model, seqs), ref)
        assert greedy_ids(model, seqs[0], max_new=6, use_cache=True) == ref_ids
        assert greedy_ids(model, seqs[0], max_new=6, use_cache=False) == ref_ids
        pre, obj = _ids(base, facts[0])
        cortex.insert(facts[0].fact_id, pre, obj)
        assert not torch.equal(_logits(model, seqs), ref)
        cortex.retract(facts[0].fact_id)
        assert torch.equal(_logits(model, seqs), ref)          # retracting the last fact: empty again, bitwise
        assert list(model.state_dict().keys()) == keys
        assert model_sha256(model) == base.base_sha
    finally:
        cortex.detach()
    assert torch.equal(_logits(model, seqs), ref)
    assert sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()) == 0


def test_history_independence_write_a_write_b_retract_a_equals_write_b(base, address):
    model = base.model
    facts = _facts(base)
    seqs = [_ids(base, f)[0] + _ids(base, f)[1][:-1] for f in facts]
    A, B = facts[0], facts[1]
    c1 = EP.EpisodicCortex(model, address).attach()
    try:
        c1.insert(A.fact_id, *_ids(base, A))
        c1.insert(B.fact_id, *_ids(base, B))
        c1.retract(A.fact_id)
        out1, sha1 = _logits(model, seqs), c1.ledger.sha256()
    finally:
        c1.detach()
    c2 = EP.EpisodicCortex(model, address).attach()
    try:
        c2.insert(B.fact_id, *_ids(base, B))
        out2, sha2 = _logits(model, seqs), c2.ledger.sha256()
        assert torch.equal(out1, out2)
        assert sha1 == sha2
        # correct = retract + insert of the new object, bitwise; and the insertion order does not matter
        other = [o for o in base.world.objects[B.rel] if o not in (B.new, B.old)][0]
        new_obj = base.tok.encode(" " + other + ".")
        c2.correct(B.fact_id, new_obj)
        c2.insert(A.fact_id, *_ids(base, A))
        out3, sha3 = _logits(model, seqs), c2.ledger.sha256()
    finally:
        c2.detach()
    c3 = EP.EpisodicCortex(model, address).attach()
    try:
        c3.insert(A.fact_id, *_ids(base, A))
        c3.insert(B.fact_id, _ids(base, B)[0], new_obj)
        assert torch.equal(_logits(model, seqs), out3)
        assert c3.ledger.sha256() == sha3
        with pytest.raises(KeyError):
            c3.insert(A.fact_id, *_ids(base, A))               # a stored fact is corrected, never silently replaced
        with pytest.raises(KeyError):
            c3.retract("missing")
    finally:
        c3.detach()


def test_stored_fact_is_produced_and_cached_decoding_equals_full_recompute(base, sharp):
    """The lived sentence yields the stored multi-token object; KV-cache greedy = full recompute, token for token.

    The model is random, so its calibrated scores are weak: each fact is stored alone and read through the
    sharpened address (own anchor rho = 1); the calibrated rule itself is exercised on trained bases by the drivers."""
    model = base.model
    facts = _facts(base, 6)
    cortex = EP.EpisodicCortex(model, sharp, SELECT, **MV).attach()
    try:
        n_multi = 0
        for f in facts:
            p, o = _ids(base, f)
            ep = cortex.insert(f.fact_id, p, o)
            assert ep.cap_hits == 0
            assert exact_check_ids(model, [p], [o]) == [True]
            cached = greedy_ids(model, p, max_new=len(o), use_cache=True)
            full = greedy_ids(model, p, max_new=len(o), use_cache=False)
            assert cached == full == o
            n_multi += int(len(o) >= 2)
            # writes below the injection never invalidate the KV cache: a cached step equals the full pass
            full_logits = _logits(model, [p + o[:-1]])[0]
            model.clear_kv_cache()
            with torch.no_grad():
                model(torch.tensor([p]), use_cache=True, cache_position=torch.arange(len(p)))
                step = model(torch.tensor([[o[0]]]), use_cache=True, cache_position=torch.tensor([len(p)]))[0]
            model.clear_kv_cache()
            if len(o) >= 2:                                        # programme test 6: logits within 1e-5
                assert float((step[0, 0] - full_logits[len(p)]).abs().max()) <= 1e-5
            cortex.retract(f.fact_id)
        assert n_multi >= 1                                    # multi-token objects: the continuation rule is used
    finally:
        cortex.detach()


def test_null_gate_keeps_unrelated_queries_at_the_base(base, address):
    """With a strict threshold nothing fires on other entities; without the null slot the store always answers."""
    model = base.model
    facts = _facts(base, 4)
    others = sorted(base.world.members("probe"))[30:36]
    probes = [[BOS_ID] + base.tok.encode(frame_prefix(n, "A", "gen")) for n in others]
    ref = _logits(model, probes)
    strict = EP.EpisodicCortex(model, address, EP.ReadRule("fixed", 200.0)).attach()
    try:
        for f in facts:
            strict.insert(f.fact_id, *_ids(base, f))
        out = _logits(model, probes)
        assert torch.equal(out.argmax(-1), ref.argmax(-1))
        assert float(strict.last_rho.sum(-1).max()) < 1e-6
    finally:
        strict.detach()
    nonull = EP.EpisodicCortex(model, address, EP.ReadRule("nonull")).attach()
    try:
        for f in facts:
            nonull.insert(f.fact_id, *_ids(base, f))
        _logits(model, probes)
        assert torch.allclose(nonull.last_rho.sum(-1), torch.ones_like(nonull.last_rho.sum(-1)), atol=1e-9)
    finally:
        nonull.detach()


# ------------------------------------------------------------------------------------------------ compose
def _toy_pack(n=3, d=4, kmax=3):
    torch.manual_seed(0)
    R = torch.randn(n, kmax, d, dtype=torch.float64)
    Y = torch.tensor([[5, 6, 7], [5, 8, -1], [9, -1, -1]])
    R[1, 2] = 0
    R[2, 1:] = 0
    return EP.Pack([f"f{i}" for i in range(n)], torch.zeros(n, 2, dtype=torch.float64), R, Y, torch.tensor([3, 2, 1]))


class _ToyReadout:
    """Uniform base distribution over 10 tokens: p_base(next) = 0.1 everywhere."""

    def __call__(self, x):
        return torch.zeros(x.shape[:-1] + (10,), dtype=torch.float64)


def test_compose_bayesian_continuation_weights():
    pack = _toy_pack()
    rule = EP.ReadRule("fixed", 0.0)
    tokens = torch.tensor([[1, 2, 5, 6, 3]])               # anchor at t=1 (PRED), then y = 5, 6 emitted
    S = torch.full((1, 5, 3), -50.0, dtype=torch.float64)
    S[0, 1] = torch.tensor([1.0, 0.5, -1.0])                # only the PRED position addresses the store
    X = torch.zeros(1, 5, 4, dtype=torch.float64)
    O, rho, rho0 = EP.compose(tokens, S, X, pack, rule, _ToyReadout())
    e = torch.exp(torch.tensor([1.0, 0.5, -1.0], dtype=torch.float64))
    r = e / (1.0 + e.sum())
    r0 = 1.0 / (1.0 + e.sum())
    assert torch.allclose(rho[0, 1], r, atol=1e-12) and float(rho0[0, 1]) == pytest.approx(float(r0))
    assert torch.allclose(O[0, 1], r @ pack.R[:, 0], atol=1e-12)
    # t=2: token 5 follows the anchor; episodes 0 and 1 start with 5 and have a second value; episode 2 does not
    den = r0 * 0.1 + r[0] + r[1]
    expect2 = (r[0] / den) * pack.R[0, 1] + (r[1] / den) * pack.R[1, 1]
    assert torch.allclose(O[0, 2], expect2, atol=1e-9)
    # t=3: tokens 5, 6 match only episode 0, which has a third value
    den3 = r0 * 0.01 + r[0]
    assert torch.allclose(O[0, 3], (r[0] / den3) * pack.R[0, 2], atol=1e-9)
    # t=4: no episode has a fourth token; position 0 has no anchor
    assert float(O[0, 4].abs().max()) < 1e-12 and float(O[0, 0].abs().max()) < 1e-12
    # a query whose continuation does not start with a stored first token gets no continuation value
    tokens2 = torch.tensor([[1, 2, 4, 6, 3]])
    O2, _, _ = EP.compose(tokens2, S, X, pack, rule, _ToyReadout())
    assert float(O2[0, 2:].abs().max()) < 1e-12
    # precomputed base log-probabilities give the same injection
    lp = torch.full((1, 5), -2.302585092994046, dtype=torch.float64)
    O3, _, _ = EP.compose(tokens, S, X, pack, rule, None, lp)
    assert torch.allclose(O3, O, atol=1e-12)


def test_ledger_pack_is_id_sorted_and_subsettable(base, address):
    cortex = EP.EpisodicCortex(base.model, address)
    facts = _facts(base, 4)
    eps = {f.fact_id: cortex.make_episode(f.fact_id, *_ids(base, f)) for f in facts}
    a, b = EP.EpisodeLedger(), EP.EpisodeLedger()
    for fid in sorted(eps):
        a.insert(eps[fid])
    for fid in sorted(eps, reverse=True):
        b.insert(eps[fid])
    pa, pb = a.pack(address), b.pack(address)
    assert pa.ids == pb.ids == sorted(eps)
    assert torch.equal(pa.KU, pb.KU) and torch.equal(pa.R, pb.R) and torch.equal(pa.Y, pb.Y)
    assert a.sha256() == b.sha256()
    sub = a.pack(address, sorted(eps)[:2])
    assert sub.n == 2
    # the address embedding is one batched matmul: BLAS does not promise bit-identical rows across batch sizes
    # (seen on CI), so the subset's rows are compared with float64 rounding allowed and nothing more
    torch.testing.assert_close(sub.KU, pa.KU[:2], rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError):
        EP.EpisodeLedger().pack(address)


# ------------------------------------------------------------------------------------------------ cached evaluator
def test_cached_evaluator_agrees_with_the_hooked_model(base, address, sharp):
    """One base forward per probe + ``compose`` gives the decisions of the hooked float32 model."""
    model, tok = base.model, base.tok
    facts = _facts(base, 6)
    others = sorted(base.world.members("probe"))[30:40]
    probes = []
    for f in facts:
        probes.append({"kind": "lived", "ci": f.idx, "name": f.name, "rel": f.rel, "prefix": f.prefix, "gold": f.new})
        probes.append({"kind": "para", "ci": f.idx, "name": f.name, "rel": f.rel,
                       "prefix": frame_prefix(f.name, f.rel, "nom"), "gold": f.new})
    for n in others:
        probes.append({"kind": "U", "ci": -1, "name": n, "rel": "A", "prefix": frame_prefix(n, "A", "gen"),
                       "gold": base.world.facts[(n, "A")]})
    kmax = max(len(tok.encode(" " + f.new + ".")) for f in facts)
    cb = ST.CachedBattery(base, probes, [address], kmax)
    eps = cb.episodes(facts, address, **MV)
    cortex = EP.EpisodicCortex(model, address, **MV)
    # the cached episode equals the hooked write up to batch-composition rounding
    hooked_ep = cortex.make_episode(facts[0].fact_id, *_ids(base, facts[0]))
    assert hooked_ep.obj == eps[0].obj
    assert torch.allclose(hooked_ep.code, eps[0].code, atol=1e-4)
    assert torch.allclose(hooked_ep.values, eps[0].values, atol=1e-2 * float(eps[0].values.norm()) + 1e-6)
    for e in eps:
        cortex.ledger.insert(e)
    pre, tgt = zip(*[ST.probe_ids(tok, p) for p in probes])
    for rule in (EP.ReadRule("rule"), EP.ReadRule("fixed", -5.0), EP.ReadRule("nonull")):
        cortex.rule = rule
        res = cb.evaluate(cortex.ledger.pack(address), address, rule)
        cortex.attach()
        try:
            hooked = exact_check_ids(model, list(pre), list(tgt))
        finally:
            cortex.detach()
        assert res["ok"].tolist() == hooked, rule.mode
        assert res["rho"].shape == (len(probes), len(facts))
    # a fact stored alone and read with rho = 1 at its own anchor is produced on its lived probe
    cb.addresses[sharp.name + "-sharp"] = sharp
    for f in facts[:3]:
        one = cortex.ledger.pack(sharp, [f.fact_id])
        res = cb.evaluate(one, sharp, SELECT)
        assert eps[f.idx].cap_hits == 0 and bool(res["ok"][cb.lived_row(f.idx)])
        assert float(res["mass"][cb.lived_row(f.idx)]) == pytest.approx(1.0, abs=1e-6)
    assert model_sha256(model) == base.base_sha


# ------------------------------------------------------------------------------------------------ battery, metrics
def test_stream_and_partitions_are_deterministic_and_disjoint(base):
    U, pool = ST.split_probe(base.world, 25, n_u=20)
    U2, pool2 = ST.split_probe(base.world, 25, n_u=20)
    assert (U, pool) == (U2, pool2) and not set(U) & set(pool)
    assert set(U) | set(pool) == set(base.world.members("probe"))
    facts = ST.make_stream(base.world, pool, 23, 40)
    assert [f.fact_id for f in facts] == sorted(f.fact_id for f in facts)      # id order = stream order
    assert len({(f.name, f.rel) for f in facts}) == 40                         # one fact per concept
    assert all(f.new != f.old and f.name in pool for f in facts)
    assert ST.make_stream(base.world, pool, 23, 40) == facts
    assert ST.make_stream(base.world, pool, 23, 10) == facts[:10]              # checkpoints are prefixes
    assert {ST.kl_relation(r) for r in RELATIONS} == set(RELATIONS) and all(ST.kl_relation(r) != r for r in RELATIONS)


def test_battery_metrics_checkpoint_accounting():
    kind = ["lived", "fused", "lived", "fused", "U", "U", "REL", "REL", "LTOK", "PX", "OBJ"]
    ci = torch.tensor([0, 0, 1, 1, -1, -1, 0, 1, 1, 0, 1])
    first = torch.tensor([3, 3, 4, 4, 3, 9, 1, 1, 1, 1, 1])
    ok = torch.tensor([1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=torch.bool)
    m1 = ST.battery_metrics(kind, ci, first, ok, 1, [3])
    assert m1["lived"] == 1.0 and m1["fused"] == 1.0 and m1["n_fused"] == 1
    assert m1["Leak_U"] == 0.5 and m1["Leak_C"] == 1.0 and m1["n_Leak_C"] == 1
    assert m1["Leak_RN"] is None and m1["PX"] == 1.0 and m1["OBJ"] is None      # facts not live yet do not count
    assert m1["OBJ1"] == 0.0 and m1["n_OBJ1"] == 1                              # the U probe starting with token 3
    m2 = ST.battery_metrics(kind, ci, first, ok, 2, [3, 4])
    assert m2["fused"] == 0.5 and m2["Leak_C"] == 0.5 and m2["Leak_RN"] == 0.0 and m2["OBJ"] == 0.0
    assert m2["structured"] == pytest.approx((0.5 + 0.0 + 1.0 + 0.0 + 0.0) / 5)
    usable = torch.tensor([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=torch.bool)
    assert ST.battery_metrics(kind, ci, first, ok, 2, [3, 4], usable)["Leak_U"] == 0.0


def test_auc_ece_interpolation_and_laws():
    assert ST.auc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert ST.auc([0.1, 0.2, 0.9, 0.8], [True, True, False, False]) == 0.0
    assert ST.auc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5
    assert ST.auc([0.5, 0.6], [True, True]) is None
    p = torch.tensor([0.05, 0.05, 0.95, 0.95])
    assert ST.ece(p, torch.tensor([0, 0, 1, 1])) == pytest.approx(0.05)
    assert ST.ece(p, torch.tensor([1, 1, 0, 0])) == pytest.approx(0.95)
    pts = [{"x": 0.0, "y": 0.5}, {"x": 0.04, "y": 0.9}, {"x": 0.10, "y": 1.0}]
    assert ST.interp_at(pts, "x", "y", 0.02) == pytest.approx(0.7)
    assert ST.interp_at(pts, "x", "y", 0.5) == 1.0
    assert ST.interp_at([{"x": 0.3, "y": 1.0}], "x", "y", 0.02) is None
    # Markov: N-invariant and equal to 1 at P_t = 0.5 (vacuous at the 1% scale); the Gaussian law is small only
    # for a large separation m and falls with m
    assert ST.markov_bound(0.5) == 1.0
    th = C.theta_n(100)
    assert ST.false_fire_law(100, 2.0, th) > ST.false_fire_law(100, 12.0, th) > ST.false_fire_law(100, 40.0, th) > 0
    assert ST.false_fire_law(100, 12.0, th) == pytest.approx(100 * ST.normal_sf((th + 12.0) / (24.0 ** 0.5)))
    assert ST.false_fire_law(100, 40.0, th) < 1e-4 < 0.01 < ST.markov_bound(0.5)
    assert ST.normal_sf(0.0) == 0.5 and ST.normal_sf(1.6448536269514722) == pytest.approx(0.05)
    assert ST.false_fire_law(100, 0.0, th) is None
    assert ST.generalisation_law(10.0, C.theta_n(25)) > ST.generalisation_law(10.0, C.theta_n(400))
