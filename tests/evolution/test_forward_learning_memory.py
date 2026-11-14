# -*- coding: utf-8 -*-
"""Forward learning, fast-weight state: read gate, float64 ledger, retraction, guards, capacity law.

What is pinned here
-------------------
* Read gate: nu = 1 - k'C^-1k / k'C0^-1k lies in [0, 1), has the one-key closed form, is 0 for a dead key and
  equals its dual (ledger) form; where no gate opens the logits are bit-identical to the frozen model, also for
  the positions before the first open gate of a sequence.
* Ledger: F = sum r z' and C = C0 + xi sum k k' to float64 round-off; retract(write) equals the state rebuilt
  from the remaining entries; retracting everything gives F = 0, C = C0 and frozen logits exactly; retraction is
  exact at the ledger level and is not a replay without the write.
* RLS projector: Woodbury update and downdate against direct inverses; a stored key is protected by the factor
  1 / (1 + xi p).
* Capacity guards (generic RMS ratio, ledger budget) with the three policies.
* Capacity law: predicted mean familiarity and knee against measurements on synthetic one- and two-population
  keys.

CPU only; tiny random models or pure linear algebra; nothing is written anywhere.
"""
from __future__ import annotations

import logging
import math

import pytest
import torch

from model_management.config_schema import tiny_model_config
from model_management.model_manager import ModelManager
from src.neural_network_module.ortak_katman_module import forward_learning as fl

VOCAB = 97
WIDE_GRID = (0.0,) + tuple(0.25 * 2.0 ** (i / 2.0) for i in range(44))


@pytest.fixture(autouse=True)
def _single_thread_quiet():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(previous)
    torch.set_num_threads(threads)


def build(seed: int = 0, perturb: float = 0.05, **overrides):
    torch.manual_seed(seed)
    cfg = dict(vocab_size=VOCAB, embed_dim=32, seq_proj_dim=32, num_heads=4, num_kv_heads=4, num_layers=3,
               ffn_dim=None, max_seq_length=64, max_cache_len=64, dropout=0.0)
    cfg.update(overrides)
    mm = ModelManager(dict(tiny_model_config(**cfg)))
    mm.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    model = mm.model.eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(perturb * torch.randn(p.shape, generator=g))
    return model


def reference(n_seq: int = 150, length: int = 18, seed: int = 9):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(5, VOCAB, (length,), generator=g).tolist() for _ in range(n_seq)]


def facts(n: int, length: int = 12, seed: int = 30):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(5, VOCAB, (length,), generator=g) for _ in range(n)]


def learner_for(model, **cfg):
    base = dict(rule="FEC-TOP", guard="off", step_grid=WIDE_GRID, install="greedy", n0=5.0)
    base.update(cfg)
    learner = fl.ForwardLearner(model, fl.ForwardLearningConfig(**base))
    learner.calibrate(reference())
    return learner


def spd(f: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(f, 3 * f, generator=g, dtype=torch.float64)
    return a @ a.t() / (3 * f)


# --------------------------------------------------------------------------------------------------- read gate
def test_where_no_gate_opens_the_model_is_the_frozen_model_exactly():
    model = build(seed=1)
    top = len(model.layers) - 1
    unrelated = torch.randint(5, VOCAB, (4, 16), generator=torch.Generator().manual_seed(77))
    fact = facts(1)[0]
    pos = [8, 9, 10]
    with torch.no_grad():
        base_unrelated = model(unrelated)[0].clone()
        base_fact = model(fact[:-1].unsqueeze(0))[0].clone()
    learner = learner_for(model, theta=0.5)
    result = learner.write(learner.read(fact, pos))
    assert result.installed and result.write_id is not None
    with torch.no_grad():
        out = model(unrelated)[0]
        assert not bool(learner.gate_open[top].any())            # nothing familiar in the unrelated batch
        assert torch.equal(out, base_unrelated)                   # bit-identical, not merely close
        out_fact = model(fact[:-1].unsqueeze(0))[0]
        gate = learner.gate_open[top][0]
    assert gate[pos].all()                                        # the read positions are familiar
    first = int(gate.nonzero()[0])
    assert first > 0 and torch.equal(out_fact[0, :first], base_fact[0, :first])   # before the first open gate
    assert not torch.equal(out_fact[0, pos], base_fact[0, pos])
    # familiarity of the stored keys is high, of unrelated keys low
    reading = learner.read(fact, pos)
    nu = learner.familiarity(top, reading.keys[top])
    assert float(nu[pos].min()) >= 0.5 > float(nu[:first].max())
    assert float(nu.min()) >= 0.0 and float(nu.max()) < 1.0
    # theta = 0: the gate is not consulted and the fast weight acts as a plain fc2 delta everywhere
    learner.theta = 0.0
    with torch.no_grad():
        ungated = model(unrelated)[0]
    assert not torch.equal(ungated, base_unrelated)
    w = model.layers[top].ffn.fc2.weight
    saved = w.detach().clone()
    learner.enabled = False
    with torch.no_grad():
        w.add_(learner.fast_weight(top).to(w.dtype))
        merged = model(unrelated)[0]
        w.copy_(saved)
    assert torch.allclose(ungated, merged, rtol=1e-4, atol=1e-5)  # same function as editing fc2 itself
    learner.enabled = True
    # a threshold above every familiarity closes the memory completely
    learner.theta = 0.999999
    with torch.no_grad():
        assert torch.equal(model(fact[:-1].unsqueeze(0))[0], base_fact)
    # the read gain scales the fast output; gain 0 is the frozen model
    learner.theta, learner.read_gain = 0.0, 0.0
    with torch.no_grad():
        assert torch.equal(model(unrelated)[0], base_unrelated)
    learner.close()


def test_gate_without_key_statistics_is_refused_and_leaves_no_state():
    model = build(seed=2)
    learner = fl.ForwardLearner(model, fl.ForwardLearningConfig(rule="FEC-TOP", theta=0.5, projector=False,
                                                                guard="off", step_grid=WIDE_GRID,
                                                                install="greedy"))
    fact = facts(1)[0]
    with torch.no_grad():
        base = model(fact[:-1].unsqueeze(0))[0].clone()
    with pytest.raises(fl.ForwardLearningError, match="calibrate"):
        learner.write(learner.read(fact, [8, 9, 10]))
    assert len(learner) == 0 and float(learner.fast_weight(2).abs().max()) == 0.0
    with torch.no_grad():
        assert torch.equal(model(fact[:-1].unsqueeze(0))[0], base)
    with pytest.raises(fl.ForwardLearningError):
        learner.familiarity(2, torch.zeros(4, learner.fast_weight(2).shape[1]))
    learner.close()


def test_familiarity_closed_forms():
    f, xi = 24, 3.0
    C0 = spd(f, 1) + 0.1 * torch.eye(f, dtype=torch.float64)
    P0 = torch.linalg.inv(C0)
    g = torch.Generator().manual_seed(2)
    K = torch.randn(5, f, generator=g, dtype=torch.float64)
    q = torch.randn(40, f, generator=g, dtype=torch.float64)
    # one stored key: nu(q) = xi p0 / (1 + xi p0) * cos^2 of q and k in the C0^-1 metric
    k = K[:1]
    P1 = fl.woodbury_update(P0, k, xi)
    p0 = float(k @ P0 @ k.t())
    cos2 = (q @ P0 @ k.t()).squeeze(-1) ** 2 / (p0 * ((q @ P0) * q).sum(-1))
    nu = fl.familiarity_from(P1, P0, q)
    assert torch.allclose(nu, xi * p0 / (1 + xi * p0) * cos2, rtol=1e-9, atol=1e-12)
    assert abs(float(fl.familiarity_from(P1, P0, k)) - xi * p0 / (1 + xi * p0)) < 1e-12
    # several keys: in [0, 1), scale invariant, equal to the dual form over the stored keys
    P5 = fl.woodbury_update(P0, K, xi)
    nu = fl.familiarity_from(P5, P0, q)
    assert float(nu.min()) >= 0.0 and float(nu.max()) < 1.0
    assert torch.allclose(nu, fl.familiarity_from(P5, P0, 7.5 * q), rtol=1e-9)
    a = q @ P0 @ K.t()
    G = K @ P0 @ K.t()
    dual = ((a @ torch.linalg.inv(torch.eye(5, dtype=torch.float64) / xi + G)) * a).sum(-1) / ((q @ P0) * q).sum(-1)
    assert torch.allclose(nu, dual, rtol=1e-8, atol=1e-12)
    # k'C^-1k = (1 - nu) k'C0^-1k
    assert torch.allclose(((q @ P5) * q).sum(-1), (1 - nu) * ((q @ P0) * q).sum(-1), rtol=1e-9)
    # a dead key has familiarity 0, not NaN; float32 keys are accepted
    dead = fl.familiarity_from(P5, P0, torch.zeros(3, f))
    assert torch.equal(dead, torch.zeros(3, dtype=torch.float64))
    assert fl.familiarity_from(P5, P0, q.float()).dtype == torch.float64


# ---------------------------------------------------------------------------------------------- RLS projector
def test_woodbury_update_and_downdate_match_direct_inverses():
    f, xi = 20, 2.5
    C0 = spd(f, 3) + 0.05 * torch.eye(f, dtype=torch.float64)
    P = torch.linalg.inv(C0)
    g = torch.Generator().manual_seed(4)
    C = C0.clone()
    batches = [torch.randn(n, f, generator=g, dtype=torch.float64) for n in (1, 3, 2, 4)]
    for K in batches:
        k = K[0]
        before = float(k @ P @ k)
        P = fl.woodbury_update(P, K, xi)
        C = C + xi * K.t() @ K
        assert torch.allclose(P, torch.linalg.inv(C), rtol=1e-8, atol=1e-10)
        assert torch.equal(P, P.t()) and float(torch.linalg.eigvalsh(P).min()) > 0.0
        if K.shape[0] == 1:                                        # k'Pk shrinks by exactly 1 / (1 + xi k'Pk)
            assert abs(float(k @ P @ k) - before / (1 + xi * before)) < 1e-12
    for K in reversed(batches):
        P = fl.woodbury_downdate(P, K, xi)
        C = C - xi * K.t() @ K
        assert torch.allclose(P, torch.linalg.inv(C), rtol=1e-7, atol=1e-9)
    assert torch.allclose(P, torch.linalg.inv(C0), rtol=1e-7, atol=1e-9)


def test_the_projector_protects_a_stored_key_by_the_rls_factor():
    model = build(seed=3)
    learner = learner_for(model, xi=2.0)
    top = len(model.layers) - 1
    f = learner.fast_weight(top).shape[1]
    g = torch.Generator().manual_seed(5)
    k_old, k_new = torch.randn(f, generator=g, dtype=torch.float64), torch.randn(f, generator=g, dtype=torch.float64)
    P0 = torch.linalg.inv(learner.prior(top))
    r = torch.randn(1, 32, generator=g, dtype=torch.float64)
    gain = learner.rls_gain(top, k_old)
    p_old = float(k_old @ P0 @ k_old)
    assert torch.allclose(gain, (P0 @ k_old) / (0.5 + p_old), rtol=1e-10)
    learner.write_terms({top: (r, gain.unsqueeze(0), k_old.unsqueeze(0))}, tag="old")
    # soft RLS: the write lands the fraction p / (1/xi + p) of its value at its own key
    assert torch.allclose(learner.fast_weight(top) @ k_old, r[0] * p_old / (0.5 + p_old), rtol=1e-10)
    # the cross-talk of a later key with the stored one shrinks by 1 / (1 + xi p_old)
    after = float(k_old @ torch.linalg.inv(learner.prior(top) + 2.0 * torch.outer(k_old, k_old)) @ k_new)
    assert abs(after - float(k_old @ P0 @ k_new) / (1 + 2.0 * p_old)) < 1e-10
    z_new = learner.rls_gain(top, k_new)
    assert abs(float(z_new @ k_old)) < abs(float(((P0 @ k_new) / (0.5 + float(k_new @ P0 @ k_new))) @ k_old))
    learner.close()


# ------------------------------------------------------------------------------------------------------ ledger
def test_ledger_identity_and_exact_retraction():
    model = build(seed=4)
    x = torch.randint(5, VOCAB, (3, 14), generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        base = model(x)[0].clone()
    learner = learner_for(model, rule="FEC-A", xi=1.5)
    empty = learner.fingerprint()
    ids = []
    for i, fact in enumerate(facts(5)):
        result = learner.write(learner.read(fact, [8, 9, 10]), mu=4.0, tag=f"fact-{i}")
        ids.append(result.write_id)
    assert ids == [0, 1, 2, 3, 4] and [e.tag for e in learner.ledger] == [f"fact-{i}" for i in range(5)]
    n_blocks = len(model.layers)

    def from_ledger(entries):
        out = {}
        for b in range(n_blocks):
            Fb = torch.zeros_like(learner.fast_weight(b))
            Cb = learner.prior(b)
            for e in entries:
                R, Z, K = e.terms[b]
                Fb = Fb + R.t() @ Z
                Cb = Cb + 1.5 * K.t() @ K
            out[b] = (Fb, Cb)
        return out

    full = from_ledger(learner.ledger)
    for b in range(n_blocks):
        scale = float(full[b][0].abs().max())
        assert float((learner.fast_weight(b) - full[b][0]).abs().max()) <= 1e-12 * max(1.0, scale)
        assert learner.n_keys(b) == sum(int(e.terms[b][2].shape[0]) for e in learner.ledger)
    err = learner.rebuild()
    assert err["F"] <= 1e-12 and err["C"] <= 1e-9
    assert sum(e.n_keys for e in learner.ledger) == sum(learner.n_keys(b) for b in range(n_blocks))

    learner.retract(ids[1])                                       # a write in the middle of the stream
    rest = from_ledger(learner.ledger)
    for b in range(n_blocks):
        assert float((learner.fast_weight(b) - rest[b][0]).abs().max()) <= 1e-10
    err = learner.rebuild()                                       # running C and P against the rebuilt ones
    assert err["F"] <= 1e-10 and err["C"] <= 1e-9 and err["P"] <= 1e-6
    with pytest.raises(KeyError):
        learner.retract(ids[1])
    for i in (ids[0], ids[4], ids[2], ids[3]):
        learner.retract(i)
    for b in range(n_blocks):
        assert float(learner.fast_weight(b).abs().max()) == 0.0   # an empty ledger carries no float residue
    assert learner.fingerprint() == empty and len(learner) == 0
    with torch.no_grad():
        assert torch.equal(model(x)[0], base)
    learner.close()


def test_retraction_is_exact_in_the_ledger_but_is_not_a_replay_without_the_write():
    model = build(seed=5)
    top = len(model.layers) - 1
    g = torch.Generator().manual_seed(6)
    f = model.layers[top].ffn.fc2.weight.shape[1]
    k1 = torch.randn(f, generator=g, dtype=torch.float64)
    k2 = k1 + 0.5 * torch.randn(f, generator=g, dtype=torch.float64)      # correlated with the first key
    r1, r2 = torch.randn(1, 32, generator=g, dtype=torch.float64), torch.randn(1, 32, generator=g, dtype=torch.float64)

    both = learner_for(model)
    a = both.write_terms({top: (r1, both.rls_gain(top, k1).unsqueeze(0), k1.unsqueeze(0))})
    both.write_terms({top: (r2, both.rls_gain(top, k2).unsqueeze(0), k2.unsqueeze(0))})
    z2_with_first = both.ledger[1].terms[top][1].clone()
    both.retract(a)
    assert torch.allclose(both.fast_weight(top), r2.t() @ z2_with_first, rtol=1e-12, atol=1e-14)   # ledger level
    assert both.rebuild()["C"] <= 1e-9
    both.close()

    only = learner_for(build(seed=5))
    only.write_terms({top: (r2, only.rls_gain(top, k2).unsqueeze(0), k2.unsqueeze(0))})
    gap = float((both.fast_weight(top) - only.fast_weight(top)).norm() / only.fast_weight(top).norm())
    assert gap > 1e-3            # the second gain vector was computed with the first key in C: not a replay
    # with an orthogonal first key (in the C0^-1 metric) the two coincide
    only.reset()
    P0 = torch.linalg.inv(only.prior(top))
    k_orth = k1 - (float(k1 @ P0 @ k2) / float(k2 @ P0 @ k2)) * k2
    b = only.write_terms({top: (r1, only.rls_gain(top, k_orth).unsqueeze(0), k_orth.unsqueeze(0))})
    only.write_terms({top: (r2, only.rls_gain(top, k2).unsqueeze(0), k2.unsqueeze(0))})
    only.retract(b)
    replay = learner_for(build(seed=5))
    replay.write_terms({top: (r2, replay.rls_gain(top, k2).unsqueeze(0), k2.unsqueeze(0))})
    assert torch.allclose(only.fast_weight(top), replay.fast_weight(top), rtol=1e-9, atol=1e-12)
    only.close()
    replay.close()


def test_snapshot_restore_and_deterministic_fingerprint():
    def run():
        model = build(seed=6)
        learner = learner_for(model, rule="FEC-0")
        for fact in facts(3):
            learner.write(learner.read(fact, [8, 9, 10]), mu=2.0)
        return model, learner

    model, learner = run()
    _, twin = run()
    assert learner.fingerprint() == twin.fingerprint()            # equal inputs, equal float64 state
    snap = learner.snapshot()
    x = facts(1, seed=99)[0][:-1].unsqueeze(0)
    with torch.no_grad():
        at_snap = model(x)[0].clone()
    learner.write(learner.read(facts(1, seed=41)[0], [8, 9, 10]), mu=2.0)
    learner.retract(0)
    assert learner.fingerprint() != twin.fingerprint()
    learner.restore(snap)
    assert learner.fingerprint() == twin.fingerprint() and len(learner) == 3
    with torch.no_grad():
        assert torch.equal(model(x)[0], at_snap)
    nxt = learner.write(learner.read(facts(1, seed=41)[0], [8, 9, 10]), mu=2.0)
    assert nxt.write_id == 3                                       # ids continue from the snapshot
    learner.reset()
    assert len(learner) == 0 and learner.n_keys(0) == 0
    learner.close()
    twin.close()


# ------------------------------------------------------------------------------------------------------ guards
def test_capacity_guards_keep_the_generic_rms_ratio_and_the_budget():
    stream = facts(8)

    def run(**cfg):
        model = build(seed=7)
        learner = learner_for(model, **cfg)
        results = [learner.write(learner.read(fact, [8, 9, 10]), mu=8.0) for fact in stream]
        return learner, results

    top = 2
    free, res = run(guard="off")
    ratios = free.generic_rms_ratio(top)
    assert len(free) == 8 and ratios > 0.0 and all(r.write_id is not None for r in res)
    one = fl.ForwardLearner(build(seed=7), fl.ForwardLearningConfig(rule="FEC-TOP", guard="off", n0=5.0))
    one.calibrate(reference())
    one.write(one.read(stream[0], [8, 9, 10]), mu=8.0)
    single = one.generic_rms_ratio(top)
    assert ratios > 1.5 * single                                   # the ratio grows with the number of writes
    # the ratio is the RMS of F h over the reference keys relative to the native fc2 output RMS
    assert abs(single - _measured_ratio(one, top)) <= 1e-6 * max(1.0, single)
    cap = 2.5 * single

    evict, res = run(guard="evict", rms_cap=cap)
    assert evict.generic_rms_ratio(top) <= cap and evict.evictions > 0 and len(evict) < 8
    assert all(r.installed for r in res) and sum(len(r.evicted) for r in res) == evict.evictions
    assert [e.write_id for e in evict.ledger] == sorted(e.write_id for e in evict.ledger)
    assert evict.ledger[-1].write_id == 7 and 0 not in [e.write_id for e in evict.ledger]   # oldest go first

    refuse, res = run(guard="refuse", rms_cap=cap)
    refused = [r for r in res if not r.installed]
    assert refused and all(r.write_id is None and r.flags[-1].startswith("refused:rms@") for r in refused)
    assert refuse.generic_rms_ratio(top) <= cap and refuse.evictions == 0
    assert [e.write_id for e in refuse.ledger][0] == 0             # the earliest writes are the ones kept

    budget, res = run(guard="evict", rms_cap=1e9, ledger_budget=3)
    assert len(budget) == 3 and [e.write_id for e in budget.ledger] == [5, 6, 7]
    assert budget.rebuild()["F"] <= 1e-10

    alone, res = run(guard="evict", rms_cap=0.2 * single)          # a single write already breaks the invariant
    assert len(alone) == 0 and not any(r.installed for r in res)
    assert float(alone.fast_weight(top).abs().max()) == 0.0
    for learner in (free, one, evict, refuse, budget, alone):
        learner.close()


def _measured_ratio(learner, block: int) -> float:
    """Generic RMS ratio measured directly on the reference sequences (not from the stored key moment)."""
    model = learner.model
    Fb = learner.fast_weight(block)
    fast_sq = native_sq = 0.0
    count = 0
    learner.enabled = False

    def hook(mod, args, out):
        nonlocal fast_sq, native_sq, count
        h = args[0][0].double()
        fast_sq += float((h @ Fb.t()).pow(2).sum())
        native_sq += float(out[0].double().pow(2).sum())
        count += h.shape[0]

    handle = model.layers[block].ffn.fc2.register_forward_hook(hook)
    with torch.no_grad():
        for seq in reference():
            model(torch.tensor([seq]))
    handle.remove()
    learner.enabled = True
    return math.sqrt(fast_sq / count) / math.sqrt(native_sq / count)


# ------------------------------------------------------------------------------------------------ capacity law
def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def _measure(sample_w, sample_q, C0, xi, n, reps=6, n_q=600):
    P0 = torch.linalg.inv(C0)
    vals = []
    for _ in range(reps):
        P = fl.woodbury_update(P0, sample_w(n), xi)
        vals.append(fl.familiarity_from(P, P0, sample_q(n_q)))
    return torch.cat(vals)


def test_capacity_law_predicts_familiarity_and_knee_for_one_population():
    d, xi, lam0 = 96, 10.0, 0.01
    g = torch.Generator().manual_seed(0)
    W = torch.randn(d, d // 2, generator=g, dtype=torch.float64) / math.sqrt(d // 2)
    bias = 0.3 * torch.randn(d, generator=g, dtype=torch.float64)

    def keys(n):
        return _unit(torch.relu(torch.randn(n, d // 2, generator=g, dtype=torch.float64) @ W.t() + bias))

    ref = keys(30000)
    C0 = lam0 * torch.eye(d, dtype=torch.float64) + d * ref.t() @ ref / ref.shape[0]
    S_w = fl.whitened_second_moment(ref, C0)
    S_q = fl.whitened_second_moment(ref, C0, unit=True)
    assert abs(float(torch.trace(S_q)) - 1.0) < 1e-9
    assert fl.predicted_familiarity(S_w, S_q, xi, 0) == pytest.approx(0.0, abs=1e-12)
    means, grid = [], (4, 8, 16, 32, 48, 64, 96)
    for n in grid:
        m = float(_measure(keys, keys, C0, xi, n).mean())
        means.append(m)
        assert abs(m - fl.predicted_familiarity(S_w, S_q, xi, n)) <= 0.02, n
    # monotone, and the knee at theta = 0.5 within 10% of the interpolated measurement
    preds = [fl.predicted_familiarity(S_w, S_q, xi, n) for n in grid]
    assert preds == sorted(preds)
    knee = fl.capacity_knee(S_w, S_q, xi, 0.5)
    lo = max(i for i, m in enumerate(means) if m < 0.5)
    measured = grid[lo] + (0.5 - means[lo]) / (means[lo + 1] - means[lo]) * (grid[lo + 1] - grid[lo])
    assert abs(knee / measured - 1.0) <= 0.10
    assert abs(fl.predicted_familiarity(S_w, S_q, xi, knee) - 0.5) < 1e-6
    # the nominal law theta * d misses here; the registered law does not
    assert abs(0.5 * d / measured - 1.0) > 0.10
    with pytest.raises(ValueError):
        fl.capacity_knee(S_w, S_q, xi, 1.0)
    with pytest.raises(ValueError):
        fl.predicted_familiarity(S_w, S_q, 0.0, 3)


def test_capacity_law_two_populations_stored_keys_are_a_sub_population():
    d, sub, a, xi, lam0 = 96, 16, 0.6, 10.0, 0.01
    g = torch.Generator().manual_seed(1)
    W = torch.randn(d, d // 2, generator=g, dtype=torch.float64) / math.sqrt(d // 2)
    u = _unit(torch.relu(torch.randn(d, generator=g, dtype=torch.float64)))
    B = torch.linalg.qr(torch.randn(d, sub, generator=g, dtype=torch.float64))[0]

    def prose(n):
        return _unit(torch.relu(torch.randn(n, d // 2, generator=g, dtype=torch.float64) @ W.t()))

    def fact(n):
        e = _unit(torch.randn(n, sub, generator=g, dtype=torch.float64) @ B.t())
        return _unit(a * u + math.sqrt(1 - a * a) * e)

    mix = torch.cat([prose(27000), fact(3000)])                    # C0 from all positions: 10% fact positions
    C0 = lam0 * torch.eye(d, dtype=torch.float64) + d * mix.t() @ mix / mix.shape[0]
    S_w = fl.whitened_second_moment(fact(20000), C0)
    S_fact = fl.whitened_second_moment(fact(20000), C0, unit=True)
    S_prose = fl.whitened_second_moment(prose(20000), C0, unit=True)
    grid = (2, 4, 8, 12, 16, 24, 32)
    fact_means = []
    for n in grid:
        mf = float(_measure(fact, fact, C0, xi, n).mean())
        mp = float(_measure(fact, prose, C0, xi, n).mean())
        fact_means.append(mf)
        assert abs(mf - fl.predicted_familiarity(S_w, S_fact, xi, n)) <= 0.03, n
        assert abs(mp - fl.predicted_familiarity(S_w, S_prose, xi, n)) <= 0.02, n
        assert mp < mf                                             # unrelated prose stays unfamiliar
    knee = fl.capacity_knee(S_w, S_fact, xi, 0.5)
    lo = max(i for i, m in enumerate(fact_means) if m < 0.5)
    measured = grid[lo] + (0.5 - fact_means[lo]) / (fact_means[lo + 1] - fact_means[lo]) * (grid[lo + 1] - grid[lo])
    assert abs(knee / measured - 1.0) <= 0.15
    # a one-population law from the generic covariance is far off for the fact population
    S_gen = fl.whitened_second_moment(mix, C0)
    generic = fl.capacity_knee(S_gen, fl.whitened_second_moment(mix, C0, unit=True), xi, 0.5)
    assert generic / measured > 2.0
    # prose never reaches the threshold within any realistic load of fact keys
    assert fl.predicted_familiarity(S_w, S_prose, xi, 4 * grid[-1]) < 0.5
    assert fl.capacity_knee(S_w, S_prose, xi, 0.999, n_max=64.0) is None


def test_capacity_law_mixture_form_for_stored_keys_from_two_position_classes():
    """Stored keys are 20% diverse 'prediction-site' keys and 80% keys of a small, quickly saturated class
    (like the later positions of a fact); the queries are prediction-site keys. One self-energy for the pooled
    population over-predicts the fill; one self-energy per class does not."""
    d, xi, lam0, share_a = 96, 4.0, 0.01, 0.2
    g = torch.Generator().manual_seed(2)
    basis = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0]
    A, B = basis[:, :56], basis[:, 56:62]

    def site(n):
        return _unit(torch.randn(n, 56, generator=g, dtype=torch.float64) @ A.t())

    def later(n):
        return 2.0 * _unit(torch.randn(n, 6, generator=g, dtype=torch.float64) @ B.t())

    def generic(n):
        return _unit(torch.randn(n, d, generator=g, dtype=torch.float64))

    def stored(n):
        n_a = int(round(share_a * n))
        return torch.cat([site(n_a), later(n - n_a)])

    ref = torch.cat([generic(20000), site(2000), later(8000)])
    C0 = lam0 * torch.eye(d, dtype=torch.float64) + d * ref.t() @ ref / ref.shape[0]
    S_a, S_b = fl.whitened_second_moment(site(20000), C0), fl.whitened_second_moment(later(20000), C0)
    S_q = fl.whitened_second_moment(site(20000), C0, unit=True)
    pooled = share_a * S_a + (1 - share_a) * S_b
    worst_single = worst_mixture = 0.0
    state = None
    for n in (10, 20, 40, 80, 160):
        measured = float(_measure(stored, site, C0, xi, n).mean())
        single = fl.predicted_familiarity(pooled, S_q, xi, n)
        mixture, state = fl.predicted_familiarity([S_a, S_b], S_q, xi, n, shares=[share_a, 1 - share_a],
                                                  warm=state, return_state=True)
        cold = fl.predicted_familiarity([S_a, S_b], S_q, xi, n, shares=[share_a, 1 - share_a])
        assert abs(mixture - cold) < 1e-6                           # the warm start does not change the value
        assert abs(mixture - measured) <= 0.02, (n, mixture, measured)
        assert single > measured                                    # pooled self-energy: fills too fast
        worst_single = max(worst_single, abs(single - measured))
        worst_mixture = max(worst_mixture, abs(mixture - measured))
    assert worst_single > 3.0 * worst_mixture and worst_single > 0.05
    # one class is the two-population form; the knee of the mixture form is where its curve crosses theta
    one = fl.predicted_familiarity([pooled], S_q, xi, 40, shares=[1.0])
    assert abs(one - fl.predicted_familiarity(pooled, S_q, xi, 40)) < 1e-9
    knee = fl.capacity_knee([S_a, S_b], S_q, xi, 0.4, shares=[share_a, 1 - share_a])
    assert abs(fl.predicted_familiarity([S_a, S_b], S_q, xi, knee, shares=[share_a, 1 - share_a]) - 0.4) < 1e-3
    assert knee > fl.capacity_knee(pooled, S_q, xi, 0.4)
    with pytest.raises(ValueError):
        fl.predicted_familiarity([S_a, S_b], S_q, xi, 10)
    with pytest.raises(ValueError):
        fl.predicted_familiarity([S_a, S_b], S_q, xi, 10, shares=[0.5, 0.6])
    with pytest.raises(ValueError):
        fl.predicted_familiarity(pooled, S_q, xi, 10, shares=[1.0])
