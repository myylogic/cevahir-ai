# -*- coding: utf-8 -*-
"""KA study — address codes, PLDA scores, the calibrated null-slot read and the radius gates (tiny random model).

No dependence on trained weights. Checks exact identities, not learning.
"""
import copy
import math
import pathlib
import random
import sys

import pytest
import torch

KA_DIR = pathlib.Path(__file__).resolve().parents[2] / "research" / "concept_address_2026_11_13"
if str(KA_DIR) not in sys.path:
    sys.path.insert(0, str(KA_DIR))

from ka import codes as C  # noqa: E402
from ka import gates as G  # noqa: E402
from ka.testing import tiny_base  # noqa: E402
from cev11.decode import greedy_ids  # noqa: E402
from cev11.model import model_sha256  # noqa: E402
from cev11.world import frame_prefix  # noqa: E402
from tokenizer_management.byte_bpe import BOS_ID  # noqa: E402


@pytest.fixture(autouse=True)
def _single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def base():
    return tiny_base(layers=3, d=32, seed=0)


def _seqs(base, n=6):
    names = sorted(base.world.members("probe"))[:n]
    return [[BOS_ID] + base.tok.encode(frame_prefix(m, "A", c)) for m, c in zip(names, ("nom", "gen", "dat", "acc", "loc", "abl"))]


# ------------------------------------------------------------------------------------------------ decomposition
def _decomposition(model, seqs):
    """Tapped codes of one forward pass, with the embedding output and every block's FFN output read by
    independent hooks: ``(out, [(a_pre^(l), expected a_pre^(l), a_post^(l), expected a_post^(l)) per block])``."""
    L = len(model.layers)
    h0, f = {}, {}
    handles = [model.layers[0].register_forward_pre_hook(lambda m, i: h0.__setitem__("e", i[0].detach().clone()))]
    for k in range(L):
        handles.append(model.layers[k].ffn.register_forward_hook(
            lambda m, i, o, k=k: f.__setitem__(k, (o if isinstance(o, torch.Tensor) else o[0]).detach().clone())))
    specs = [(kind, l) for l in range(L) for kind in ("pre", "att", "post")] + [("x", 0)]
    try:
        with C.Tap(model, specs) as tap, torch.no_grad():
            model(C.pad_batch(seqs))
            out = {k: v.clone() for k, v in tap.out.items()}
    finally:
        for h in handles:
            h.remove()
    pairs = [(out[f"pre{l}"], h0["e"] + sum(f[k] for k in range(l)) + out[f"att{l}"],
              out[f"post{l}"], out[f"pre{l}"] + f[l]) for l in range(L)]
    return out, pairs


def test_residual_decomposition_identities_at_every_block(base):
    """a_pre^(l) = e + sum_{k<l} f_k + sum_{k<=l} g_k and a_post^(l) = a_pre^(l) + f_l; x = a_post^(L-1).

    Programme test 5 asks for 1e-6. A float64 copy of the model shows the identities themselves (1e-12). In the
    float32 model what is left is the rounding of sums of activations: within 1e-6 of the activation size (the
    measured error is 5e-7 at activations of size 3)."""
    L = len(base.model.layers)
    seqs = _seqs(base)
    out, pairs = _decomposition(base.model, seqs)
    assert len(pairs) == L
    scale = max(1.0, max(float(post.abs().max()) for _, _, post, _ in pairs))
    for l, (pre, pre_expect, post, post_expect) in enumerate(pairs):
        assert float((pre - pre_expect).abs().max()) <= 1e-6 * scale, l
        assert float((post - post_expect).abs().max()) <= 1e-6 * scale, l
    assert torch.equal(out["x"], out[f"post{L - 1}"])
    out64, pairs64 = _decomposition(copy.deepcopy(base.model).double(), seqs)
    for l, (pre, pre_expect, post, post_expect) in enumerate(pairs64):
        assert pre.dtype == torch.float64 and float(pre.abs().max()) > 0.1
        assert float((pre - pre_expect).abs().max()) <= 1e-12, l
        assert float((post - post_expect).abs().max()) <= 1e-12, l
    assert torch.equal(out64["x"], out64[f"post{L - 1}"])
    assert torch.allclose(out64["x"].float(), out["x"], atol=1e-4)           # the copy computes the same function
    assert model_sha256(base.model) == base.base_sha


def test_tiny_base_leaves_the_thread_count_alone():
    """The fixtures that build a tiny base are module-scoped and run before the per-test thread fixture can
    record the old value: the helper itself has to put the thread count back."""
    old = torch.get_num_threads()
    try:
        for n in (2, 3):
            torch.set_num_threads(n)
            tiny_base(n_names=120, layers=1, d=16, seed=1)
            assert torch.get_num_threads() == n
    finally:
        torch.set_num_threads(old)


def test_tap_reads_only_and_leaves_no_hooks(base):
    model = base.model
    x = C.pad_batch(_seqs(base))
    with torch.no_grad():
        ref = model(x)[0].clone()
    n_hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
    with C.Tap(model, [("key", 1), ("res", 1), ("att", 2), ("post", 2), ("pre", 0), ("x", 0)]) as tap, torch.no_grad():
        got = model(x)[0]
        assert set(tap.out) == {"key1", "res1", "att2", "post2", "pre0", "x"}
    assert torch.equal(ref, got)
    assert sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()) == n_hooks
    assert model_sha256(model) == base.base_sha


def test_capture_matches_single_sequence_and_pick(base):
    seqs = _seqs(base)
    full = C.capture(base.model, seqs, [("att", 1), ("post", 2)])
    alone = C.capture(base.model, [seqs[2]], [("att", 1), ("post", 2)])[0]
    assert full[2]["att1"].shape == (len(seqs[2]), 32)
    assert torch.allclose(full[2]["att1"], alone["att1"], atol=1e-5)
    picked = C.capture(base.model, seqs, [("post", 2)], pick=[[len(s) - 1, 1] for s in seqs])
    assert torch.equal(picked[3]["post2"][0], full[3]["post2"][-1])
    assert torch.equal(picked[3]["post2"][1], full[3]["post2"][1])


# ------------------------------------------------------------------------------------------------ the read
def test_theta_n_is_the_bayes_prior_threshold():
    for n in (1, 25, 100, 400):
        for pt in (0.5, 0.2, 0.9):
            th = C.theta_n(n, pt)
            assert th == pytest.approx(math.log(n) - math.log(pt / (1 - pt)))
            # Markov identity: N e^{-theta_N} = P_t / (1 - P_t) for every N
            assert n * math.exp(-th) == pytest.approx(pt / (1 - pt))
    with pytest.raises(ValueError):
        C.theta_n(0)


def test_null_slot_read_is_a_posterior_with_prior_pt_over_n():
    torch.manual_seed(0)
    S = torch.randn(5, 7, dtype=torch.float64) * 3
    th = C.theta_n(7, 0.5)
    rho, rho0 = C.null_slot_read(S, th)
    assert torch.allclose(rho.sum(-1) + rho0, torch.ones(5, dtype=torch.float64), atol=1e-12)
    # Bayes: prior P_t/N per episode with likelihood ratio e^{s_j}, prior 1 - P_t for the null
    n, pt = 7, 0.5
    num = (pt / n) * torch.exp(S)
    assert torch.allclose(rho, num / ((1 - pt) + num.sum(-1, keepdim=True)), atol=1e-12)
    # a -> infinity at fixed theta/a: nearest key inside the radius, else the null (GRACE's rule)
    hard, hard0 = C.null_slot_read(1e4 * (S - 1.0), 0.0)
    for i in range(5):
        j = int(S[i].argmax())
        if S[i, j] > 1.0:
            assert hard[i, j] == pytest.approx(1.0) and hard0[i] == pytest.approx(0.0, abs=1e-12)
        else:
            assert hard0[i] == pytest.approx(1.0)


def test_read_without_null_equals_the_repository_attention(base):
    """(R) with unnormalised scores q.k/sqrt(d) and no null slot is softmax attention exactly."""
    attn = base.model.layers[0].attn
    torch.manual_seed(1)
    d, n, dv = 8, 6, 5
    q = torch.randn(1, 1, 1, d)
    K = torch.randn(1, 1, n, d)
    V = torch.randn(1, 1, n, d)
    out, w = attn.scaled_dot_product_attention(q, K, V, apply_dropout=False, return_attention_weights=True)
    S = (q[0, 0] @ K[0, 0].T) / math.sqrt(d)
    cap = float(getattr(attn, "attn_logit_cap", 0.0) or 0.0)
    if cap > 0:
        S = cap * torch.tanh(S / cap)
    rho, rho0 = C.null_slot_read(S.double(), None)
    assert float(rho0.abs().max()) == 0.0
    assert torch.allclose(rho.float(), w[0, 0], atol=1e-6)
    assert torch.allclose((rho.float() @ V[0, 0]), out[0, 0], atol=1e-6)
    assert dv == 5


# ------------------------------------------------------------------------------------------------ PLDA
def _two_cov_data(seed=0, d=6, classes=40, per=12):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    Bm = torch.randn(d, d, generator=g, dtype=torch.float64) * 1.5
    X, lab = [], []
    for c in range(classes):
        y = Bm @ torch.randn(d, generator=g, dtype=torch.float64)
        for _ in range(per):
            X.append(y + 0.5 * (A @ torch.randn(d, generator=g, dtype=torch.float64)))
            lab.append(c)
    return torch.stack(X), lab


def test_plda_llr_equals_the_direct_gaussian_log_density_ratio():
    """In PLDA coordinates (within = I, between = diag psi) the closed form is the exact two-covariance LLR."""
    X, lab = _two_cov_data()
    m, T, psi = C.fit_plda(X, lab, 0.05)
    assert bool((psi > 0).all())
    U = (X[:9] - m) @ T
    got = C.plda_llr(U[:4], U[4:], psi)
    k = psi.numel()
    Psi = torch.diag(psi)
    same = torch.cat([torch.cat([Psi + torch.eye(k), Psi], 1), torch.cat([Psi, Psi + torch.eye(k)], 1)], 0).double()
    diff = torch.block_diag(Psi + torch.eye(k), Psi + torch.eye(k)).double()
    mv = torch.distributions.MultivariateNormal
    z = torch.zeros(2 * k, dtype=torch.float64)
    for i in range(4):
        for j in range(5):
            pair = torch.cat([U[i], U[4 + j]])
            direct = mv(z, covariance_matrix=same).log_prob(pair) - mv(z, covariance_matrix=diff).log_prob(pair)
            assert float(got[i, j]) == pytest.approx(float(direct), abs=1e-8)


def test_plda_projection_whitens_within_and_diagonalises_between():
    X, lab = _two_cov_data(seed=3)
    m, T, psi = C.fit_plda(X, lab, 0.0)
    U = (X - m) @ T
    groups = {}
    for i, l in enumerate(lab):
        groups.setdefault(l, []).append(i)
    Sw = sum((U[v] - U[v].mean(0)).T @ (U[v] - U[v].mean(0)) for v in groups.values()) / (len(lab) - len(groups))
    assert torch.allclose(Sw, torch.eye(psi.numel(), dtype=torch.float64), atol=1e-8)


def test_logistic_fit_recovers_a_known_calibration():
    g = torch.Generator().manual_seed(0)
    s = torch.randn(20000, generator=g, dtype=torch.float64) * 4
    a0, b0 = 0.7, -1.3
    y = (torch.rand(20000, generator=g, dtype=torch.float64) < torch.sigmoid(a0 * s + b0)).double()
    a, b, loss = C.logistic_fit(s, y)
    assert a == pytest.approx(a0, abs=0.05) and b == pytest.approx(b0, abs=0.08)
    assert loss < float(torch.nn.functional.binary_cross_entropy_with_logits(torch.zeros_like(s), y))


def test_fit_address_gives_calibrated_scores_on_synthetic_concepts():
    """Same-concept trials score above different-concept trials and E[e^s | H0] stays near or below 1."""
    g = torch.Generator().manual_seed(5)
    d = 10
    names = [f"E{i:02d}" for i in range(20)]
    items, X = [], []
    for n in names:
        ent = torch.randn(d, generator=g, dtype=torch.float64)
        for r in ("A", "B", "C"):
            con = ent + 1.5 * torch.randn(d, generator=g, dtype=torch.float64)
            for case, orth in (("nom", "formal"), ("gen", "formal"), ("dat", "formal"), ("gen", "informal"),
                               ("dat", "informal"), ("abl", "formal")):
                items.append({"name": n, "rel": r, "case": case, "orth": orth, "ctx": "none"})
                X.append(con + 0.3 * torch.randn(d, generator=g, dtype=torch.float64))
    addr = C.fit_address(torch.stack(X), items, "T", "post", 1, seed=3)
    assert addr.shrink in C.SHRINK_GRID
    assert addr.stats["h1_mean"] > 0 > addr.stats["h0_mean"]
    assert addr.stats["h0_mean_exp_s"] < 3.0
    assert addr.logloss < 0.1
    U = addr.embed(torch.stack(X))
    S = addr.score(U[:6], U[:12])
    assert float(S[0, 1]) > float(S[0, 7])            # same concept vs another relation of the entity


def test_lda_projection_separates_classes():
    X, lab = _two_cov_data(seed=7, d=8, classes=12, per=20)
    mu, P = C.lda_fit(X, lab, 64)
    assert P.shape == (8, 8)                          # min(64, d) directions, at most classes - 1 kept
    Z = (X - mu) @ P[:, :3]
    cents = torch.stack([Z[[i for i, l in enumerate(lab) if l == c]].mean(0) for c in range(12)])
    pred = torch.cdist(Z, cents).argmin(1)
    assert float((pred == torch.tensor(lab)).double().mean()) > 0.9


# ------------------------------------------------------------------------------------------------ radius gates
def test_gate_thresholds_reproduce_the_false_match_rate():
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(6000, 16, generator=g)
    keys = torch.randn(40, 16, generator=g)
    pos_ent = [f"e{i % 50}" for i in range(6000)]
    key_ent = [f"e{i}" for i in range(40)]
    th = G.gate_thresholds(pos, pos_ent, keys, key_ent, G.QS)
    assert th[0.002] > th[0.01] > th[0.05]
    for q in G.QS:
        rate = G.false_match_rate(pos, pos_ent, keys, key_ent, th[q])
        assert abs(rate - q) <= 0.001                 # within 0.1 percentage point


def test_temperature_is_a_cosine_radius():
    for T in (3.2, 16.0, 25.6):
        th = G.radius_from_temperature(T)
        assert th == pytest.approx(1 - T / 64)
        assert G.temperature_from_radius(th) == pytest.approx(T)
    # a pair with cosine kappa merges iff kappa > 1 - T/d_c  <=>  the radius gate with theta fires
    key = torch.tensor([1.0, 0.0])
    for kappa in (0.9, 0.7, 0.5):
        u = torch.tensor([kappa, math.sqrt(1 - kappa ** 2)])
        for T in (3.2, 16.0, 25.6, 40.0):
            assert bool(G.gate_fire(u, key, G.radius_from_temperature(T))) == (kappa >= 1 - T / 64 - G.GATE_EPS)


@pytest.mark.parametrize("address", G.GATE_ADDRESSES)
def test_gate_theta_limits_and_exact_delta(base, address):
    """theta = 1 gates only the key position, theta = -1 every position, and the fired positions move by delta."""
    model, layer = base.model, 1
    code = G.GATE_CODE[address]
    ids = _seqs(base)[1]
    pos = 3
    proj = None
    if code == "sup":
        torch.manual_seed(0)
        proj = (torch.randn(32), torch.linalg.qr(torch.randn(32, 8))[0])
    key = G.gate_key(model, layer, code, ids, pos, proj)
    delta = torch.linspace(-1, 1, 32)
    x = C.pad_batch([ids])
    cap = {}
    h = model.layers[layer].ffn.fc2.register_forward_hook(lambda m, i, o: cap.__setitem__("o", o.detach().clone()))
    with torch.no_grad():
        model(x)
    base_out = cap["o"]
    h.remove()
    gm = G.GateMemory(model, layer, code, proj)
    h = model.layers[layer].ffn.fc2.register_forward_hook(lambda m, i, o: cap.__setitem__("o", o.detach().clone()))
    try:
        gm.set([(key, 1.0, delta)])
        with torch.no_grad():
            model(x)
        fire = gm.last_fire[0][0]
        assert fire.tolist() == [t == pos for t in range(len(ids))]
        assert torch.allclose(cap["o"][0, pos] - base_out[0, pos], delta, atol=1e-5)
        others = [t for t in range(len(ids)) if t != pos]
        assert torch.equal(cap["o"][0, others], base_out[0, others])
        gm.set([(key, -1.0, delta)])
        with torch.no_grad():
            model(x)
        assert bool(gm.last_fire[0].all())
        assert torch.allclose(cap["o"][0] - base_out[0], delta.expand(len(ids), 32), atol=1e-5)
    finally:
        h.remove()
        gm.remove()


def test_empty_gate_memory_is_bit_identical_with_and_without_cache(base):
    model = base.model
    seqs = _seqs(base)
    x = C.pad_batch(seqs)
    with torch.no_grad():
        ref = model(x)[0].clone()
    ref_ids = greedy_ids(model, seqs[0], max_new=6, use_cache=True)
    keys_before = list(model.state_dict().keys())
    gms = [G.GateMemory(model, 1, c, (torch.zeros(32), torch.eye(32)[:, :4]) if c == "sup" else None)
           for c in ("key", "res", "att", "sup")]
    try:
        with torch.no_grad():
            assert torch.equal(model(x)[0], ref)
        assert greedy_ids(model, seqs[0], max_new=6, use_cache=True) == ref_ids
        assert greedy_ids(model, seqs[0], max_new=6, use_cache=False) == ref_ids
        # set, clear -> still bit-identical; nothing entered the module tree
        key = G.gate_key(model, 1, "att", seqs[0], 2)
        gms[2].set([(key, 0.5, torch.ones(32))])
        gms[2].clear()
        with torch.no_grad():
            assert torch.equal(model(x)[0], ref)
        assert list(model.state_dict().keys()) == keys_before
        assert model_sha256(model) == base.base_sha
    finally:
        for g in gms:
            g.remove()
    with torch.no_grad():
        assert torch.equal(model(x)[0], ref)


def test_gate_fires_where_the_pre_edit_codes_predict_and_cached_decoding_agrees(base):
    """The gate reads a code computed before f_l: the pre-edit in-ball mask is the mask of the installed write.
    Cached greedy decoding equals full recomputation with the gate installed."""
    model, layer = base.model, 1
    seqs = _seqs(base)
    key = G.gate_key(model, layer, "att", seqs[0], 2)
    recs = C.capture(model, seqs, G.code_specs(["att"], layer), dtype=torch.float32)
    th = 0.3
    predicted = [G.gate_fire(r[f"att{layer}"], key, th) for r in recs]
    assert any(bool(p.any()) for p in predicted)
    gm = G.GateMemory(model, layer, "att")
    try:
        gm.set([(key, th, 5.0 * torch.ones(32))])
        with torch.no_grad():
            model(C.pad_batch(seqs))
        for i, s in enumerate(seqs):
            assert gm.last_fire[0][i, :len(s)].tolist() == predicted[i].tolist()
        a = greedy_ids(model, seqs[0], max_new=6, use_cache=True)
        b = greedy_ids(model, seqs[0], max_new=6, use_cache=False)
        assert a == b
    finally:
        gm.remove()


def test_a_tok_obeys_the_kernel_law(base):
    """A rank-one ROME write moves every position's fc2 output by delta * tau(x')."""
    from cev11 import experiment as E
    from cev11.memory import FFNMemory, rome_update
    from cev11.vstar import v_star
    model, layer = base.model, 1
    st = E.layer_stats(base, layer, 11)
    ids = _seqs(base)[2]
    x = C.pad_batch([ids])
    cap = {}
    h = model.layers[layer].ffn.fc2.register_forward_hook(
        lambda m, i, o: (cap.__setitem__("h", i[0].detach().clone()), cap.__setitem__("o", o.detach().clone())) and None)
    with torch.no_grad():
        model(x)
    H, base_out = cap["h"][0].double(), cap["o"][0].double()
    h.remove()
    k_bar = H[3]
    delta = torch.linspace(-2, 2, 32, dtype=torch.float64)
    M = rome_update(st["W"], st["C0"], k_bar, v_star(st["W"], k_bar, delta))
    mem = FFNMemory(model, layer)
    # registered after the memory hook, so it sees the edited output
    h = model.layers[layer].ffn.fc2.register_forward_hook(lambda m, i, o: cap.__setitem__("o", o.detach().clone()))
    try:
        mem.set_M(M.float())
        with torch.no_grad():
            model(x)
        moved = cap["o"][0].double() - base_out
    finally:
        mem.remove()
        h.remove()
    tau = G.kernel_tau(H, k_bar, st["C0_inv"])
    assert float(tau[3]) == pytest.approx(1.0, abs=1e-9)
    assert torch.allclose(moved, tau[:, None] * delta[None, :], atol=2e-4)


def test_entity_windows_and_cam():
    e = lambda *v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
    F0 = {"a": e(1, 0, 0), "b": e(0.6, 0.8, 0), "c": e(0, 0, 1)}
    forms = {"a": [e(0.95, 0.1, 0)], "b": [e(0.0, 1.0, 0.0)]}
    ctx = {"a": [e(0.99, 0.0, 0.05)]}
    rn = {"a": ["b"], "b": ["a"]}
    out = G.entity_windows(F0, forms, ctx, rn)
    assert out["n_entities"] == 2
    assert out["window_share"] == 1.0                 # both entities: forms closer than the root neighbour (cos 0.6)
    assert out["CAM"] == pytest.approx(out["cos_fused_own"] - out["cos_rn"])
    forms["a"] = [e(0.5, 0.86, 0)]                    # a's fused form now sits nearer to b
    assert G.entity_windows(F0, forms, ctx, rn)["window_share"] == 0.5
