# -*- coding: utf-8 -*-
"""CEV-11 memory stack: verification checks V1, V3, V4, V5, V7 and module contracts.

Fast (CPU, tiny random CevahirNeuralNetwork D32/L2 built via tiny_model_config + ModelManager), tmp_path only.
The ledger checks (V4, V5) use synthetic float64 matrices with the tiny model's fc2 shape.
"""
import logging
import pathlib
import random
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
STUDY = REPO / "research" / "living_learning_paradigm_2026_11_13"
for _p in (str(REPO), str(STUDY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tokenizer_management.byte_bpe import BOS_ID, ByteLevelBPE, bytes_to_unicode  # noqa: E402

from cev11 import baselines, keysites, memory, vstar  # noqa: E402
from cev11.memory import EntityJointLedger, FFNMemory, Ledger, make_identity, rome_update  # noqa: E402
from cev11.world import CARRIERS, NP, sentence  # noqa: E402  (fixed text-level functions)

NAMES = ["Aytenur", "Belgin", "Cemre", "Doruk", "Elvan", "Ferhan", "Gülten", "Harun", "Ilgaz", "Kerem"]
TOWNS = ["Kuşköy", "Taşpınar", "Akdere", "Yenikent", "Gölova", "Sarıtepe"]
NOM_PREFIX = "Aytenur hakkında bilinen: memleketi"
ABL_PREFIX = "Aytenur'dan bahsedenler bilir: memleketi"


@pytest.fixture(autouse=True)
def _single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def tok():
    rnd = random.Random(0)
    texts = []
    for n in NAMES:
        for case in CARRIERS:
            for orth in ("formal", "informal"):
                texts.append(sentence(n, "A", case, rnd.choice(TOWNS), orth))
                texts.append(sentence(n, "B", case, "doktor", orth))
    # Like a 4096-entry tokenizer trained on Turkish prose, every two-byte Turkish letter is one symbol before any
    # other merge, so no token boundary falls inside a character (token_spans rightly refuses such boundaries).
    b2u = bytes_to_unicode()
    letters = [(b2u[c.encode()[0]], b2u[c.encode()[1]]) for c in "çğıöşüâÇĞİÖŞÜ"]
    trained = ByteLevelBPE.train(texts + list(keysites.PROSE_PREFIXES), vocab_size=330)
    tok = ByteLevelBPE(letters + [m for m in trained.merges if m not in set(letters)])
    assert tok.encode(NOM_PREFIX)[: len(tok.encode("Aytenur"))] == tok.encode("Aytenur")
    for pp in ("", *keysites.PROSE_PREFIXES):
        for text in (NOM_PREFIX, ABL_PREFIX):
            keysites._local_token_spans(tok, pp + text)       # raises if a boundary splits a character
    return tok


def _tiny_model(vocab_size, seed=0):
    logging.disable(logging.WARNING)
    from model_management.config_schema import tiny_model_config
    from model_management.model_manager import ModelManager
    torch.manual_seed(seed)
    cfg = tiny_model_config(vocab_size=vocab_size, embed_dim=32, seq_proj_dim=32, num_heads=4, num_kv_heads=4,
                            num_layers=2, ffn_dim=None, max_seq_length=64, max_cache_len=64, dropout=0.0)
    mm = ModelManager(cfg)
    mm.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    return mm.model.eval()


@pytest.fixture
def model(tok):
    return _tiny_model(tok.vocab_size)


def _ids(tok, text):
    return [BOS_ID] + tok.encode(text)


def _logits(model, x):
    with torch.no_grad():
        return model(x)[0]


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same_state(model, snap):
    return all(torch.equal(v, snap[k]) for k, v in model.state_dict().items())


def _no_hooks(model):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def _c0(model, tok, layer):
    seqs = [_ids(tok, sentence(n, "A", c, TOWNS[i % 6])) for i, n in enumerate(NAMES) for c in CARRIERS]
    return memory.add_ridge(memory.key_covariance(model, layer, [seqs[i:i + 16] for i in range(0, len(seqs), 16)]))


def _synthetic(d=32, f=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    W0 = torch.randn(d, f, generator=g, dtype=torch.float64) * 0.05
    X = torch.randn(4 * f, f, generator=g, dtype=torch.float64)
    C0 = X.T @ X / X.shape[0] + 0.01 * torch.eye(f, dtype=torch.float64)
    return W0, C0, g


def _identity(C0, lam, base="base-sha", layer=1):
    return make_identity(base, layer, lam, C0, "tok-identity")


def _direct(C0, lam, facts):
    """Reference batch solve R Kᵀ (λC0 + K Kᵀ)⁻¹, written independently of the Ledger code."""
    K = torch.stack([k for k, _ in facts.values()], 1)
    R = torch.stack([r for _, r in facts.values()], 1)
    return torch.linalg.solve(lam * C0 + K @ K.T, K @ R.T).T


def _rel(a, b):
    return float((a - b).norm() / b.norm())


# ------------------------------------------------------------------------------------------------ V1: FFNMemory
def test_v1_zero_memory_is_exact_and_remove_restores(model, tok):
    fc2 = model.layers[1].ffn.fc2
    x = torch.randint(4, tok.vocab_size, (3, 11), generator=torch.Generator().manual_seed(1))
    base = _logits(model, x)
    mem = FFNMemory(model, 1)
    mem.set_M(torch.zeros_like(fc2.weight))
    assert (_logits(model, x) - base).abs().max().item() == 0.0
    mem.set_M(torch.randn(fc2.weight.shape, generator=torch.Generator().manual_seed(2)) * 0.1)
    assert (_logits(model, x) - base).abs().max().item() > 0.0
    mem.zero()
    assert (_logits(model, x) - base).abs().max().item() == 0.0
    # a write clears the KV cache
    with torch.no_grad():
        model(x[:1, :5], use_cache=True)
    assert model.layers[0].attn.kv_cache.cache_len > 0
    mem.set_M(torch.zeros_like(fc2.weight))
    assert model.layers[0].attn.kv_cache.cache_len == 0
    mem.set_M(torch.ones_like(fc2.weight))
    mem.remove()
    assert (_logits(model, x) - base).abs().max().item() == 0.0
    assert _no_hooks(model)
    with pytest.raises(RuntimeError):
        mem.set_M(torch.zeros_like(fc2.weight))


def test_substitute_replaces_output_at_given_positions(model, tok):
    fc2 = model.layers[0].ffn.fc2
    x = torch.randint(4, tok.vocab_size, (2, 7), generator=torch.Generator().manual_seed(3))
    seen = []
    mem = FFNMemory(model, 0)
    tap = fc2.register_forward_hook(lambda m, i, o: seen.append(o.detach().clone()))   # runs after the memory
    _logits(model, x)
    value = torch.arange(fc2.weight.shape[0], dtype=torch.float32)
    mem.substitute([2, None], value)
    _logits(model, x)
    base, sub = seen
    assert torch.equal(sub[0, 2], value) and torch.equal(sub[1], base[1])
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[0, 2] = False
    assert torch.equal(sub[mask], base[mask])
    with pytest.raises(RuntimeError):            # substitution set for batch 2, forward with batch 1
        _logits(model, x[:1])
    mem.substitute(None, None)
    _logits(model, x)
    assert torch.equal(seen[-1], base)
    rowwise = torch.stack([value, -value])                       # [B, d] values with a skipped row
    mem.substitute([None, 4], rowwise)
    _logits(model, x)
    assert torch.equal(seen[-1][1, 4], -value) and torch.equal(seen[-1][0], base[0])
    tap.remove()
    mem.remove()
    assert _no_hooks(model)


# ------------------------------------------------------------------------------------------------ key sites
def test_local_token_spans_match_encode(tok):
    for text in (NOM_PREFIX, ABL_PREFIX, "Dün öğrendik ki Aytenurdan bahsedenler bilir: memleketi"):
        spans = keysites._local_token_spans(tok, text)
        assert [s[0] for s in spans] == tok.encode(text)
        assert "".join(text[a:b] for _, a, b in spans) == text
        assert all(spans[i][2] == spans[i + 1][1] for i in range(len(spans) - 1))
        assert keysites.token_spans(tok, text) == spans    # sibling cev11.tokenizers (if present) must agree


def test_sites_on_synthetic_spans_fused_and_glued_space():
    text = "Aytenden bahsedenler bilir: memleketi"           # FUSED informal: "tende" spans the root end
    spans = [(10, 0, 2), (11, 2, 7), (12, 7, 8), (13, 8, 20), (14, 20, 26), (15, 26, 27), (16, 27, 37)]
    st = keysites._sites_from_spans(spans, text, 0, 5, 8)
    assert st == {"ROOT": 1, "STD": 2, "LAST": 3, "PRED": 7}
    text2 = "Dün öğrendik ki Ayten'den"                       # name after a context: " Ay" carries the space
    spans2 = [(1, 0, 3), (2, 3, 12), (3, 12, 15), (4, 15, 18), (5, 18, 21), (6, 21, 25)]
    assert keysites._sites_from_spans(spans2, text2, 16, 21, 25) == {"ROOT": 5, "STD": 5, "LAST": 6, "PRED": 6}
    with pytest.raises(keysites.SiteUndefined):              # whole name fused into one token with the suffix
        keysites._sites_from_spans([(1, 0, 8), (2, 8, 20)], "Aytenden bahsedenler", 0, 5, 8)
    lax = keysites._sites_from_spans([(1, 0, 8), (2, 8, 20)], "Aytenden bahsedenler", 0, 5, 8, strict=False)
    assert lax["ROOT"] is None and lax["STD"] == 1


def test_v3_nom_sites_coincide_and_root_last_writes_identical(model, tok):
    layer = 0
    st = keysites.sites(tok, NOM_PREFIX, "Aytenur")
    assert st["ROOT"] == st["STD"] == st["LAST"] == len(tok.encode("Aytenur"))
    st_abl = keysites.sites(tok, ABL_PREFIX, "Aytenur")       # formal: boundary at the root end, suffix after
    assert st_abl["ROOT"] == st_abl["STD"] == st["ROOT"] < st_abl["LAST"]
    k_root = keysites.key_at(model, layer, tok, NOM_PREFIX, st["ROOT"])
    k_last = keysites.key_at(model, layer, tok, NOM_PREFIX, st["LAST"])
    k_std = keysites.key_at(model, layer, tok, NOM_PREFIX, st["STD"])
    assert torch.equal(k_root, k_last) and torch.equal(k_root, k_std)
    pr_root = keysites.site_prompts(tok, NOM_PREFIX, "Aytenur", site="ROOT")
    pr_last = keysites.site_prompts(tok, NOM_PREFIX, "Aytenur", site="LAST")
    assert pr_root == pr_last == [(_ids(tok, NOM_PREFIX), st["ROOT"])]
    C0 = _c0(model, tok, layer)
    W = model.layers[layer].ffn.fc2.weight
    target = tok.encode(" Yenikent.")
    kl = _ids(tok, "Aytenur hakkında bilinen: mesleği")
    writes = []
    for pr, k in ((pr_root, k_root), (pr_last, k_last)):
        d, _ = vstar.search_delta(model, layer, pr, target, kl, step_size=0.5, cap=50.0, max_steps=5)
        writes.append(rome_update(W, C0, k, vstar.v_star(W, k, d)))
    assert torch.equal(writes[0], writes[1])


def test_std_and_para_orth_keys(model, tok):
    layer = 1
    keys = keysites.variant_std_keys(model, layer, tok, "Aytenur", "A")
    formal = [keys[(c, "formal")] for c in CARRIERS]
    assert all(torch.equal(k, formal[0]) for k in formal)    # V2 on truncated prefixes: bitwise identical
    assert torch.equal(keysites.std_key(model, layer, tok, NOM_PREFIX, "Aytenur"), formal[0])
    # std_key = mean over the lived prefix and its 10 prose copies, positions derived by hand (reviewer addition)
    n_name = len(tok.encode("Aytenur"))
    manual = [keysites.key_at(model, layer, tok, NOM_PREFIX, n_name)]
    for pp in keysites.PROSE_PREFIXES:
        head = tok.encode(pp[:-1]) + tok.encode(" Aytenur")       # the space before the name joins the name chunk
        assert tok.encode(pp + NOM_PREFIX)[:len(head)] == head
        manual.append(keysites.key_at(model, layer, tok, pp + NOM_PREFIX, len(head)))
    assert len(manual) == 11
    assert torch.allclose(formal[0], torch.stack(manual).mean(0), rtol=0, atol=1e-12)
    loo = keysites.para_orth_key(model, layer, tok, "Aytenur", "A", ("dat", "informal"))
    rest = [keys[v] for v in keysites.VARIANTS if v != ("dat", "informal")]
    assert len(rest) == 12 and torch.allclose(loo, torch.stack(rest).mean(0), rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        keysites.para_orth_key(model, layer, tok, "Aytenur", "A", ("nom", "informal"))
    assert len(keysites.PROSE_PREFIXES) == 10 and all(p.endswith(". ") for p in keysites.PROSE_PREFIXES)
    assert keysites.prose_subset(4) == keysites.prose_subset(4) and len(keysites.prose_subset(4)) == 4


def test_para_orth_leaves_out_exactly_the_tested_variant(model, tok):
    """Reviewer addition: with IDENT-only names all 13 variant keys are bitwise equal, so a wrong exclusion is
    invisible; check the exclusion on distinct synthetic keys and on a name with a FUSED informal form."""
    g = torch.Generator().manual_seed(11)
    syn = {v: torch.randn(8, generator=g, dtype=torch.float64) for v in keysites.VARIANTS}
    full = torch.stack([syn[v] for v in keysites.VARIANTS]).mean(0)
    assert torch.allclose(keysites.para_orth_from_variants(syn, None), full, rtol=0, atol=1e-12)
    for v in keysites.VARIANTS:
        rest = torch.stack([syn[u] for u in keysites.VARIANTS if u != v]).mean(0)
        assert torch.allclose(keysites.para_orth_from_variants(syn, v), rest, rtol=0, atol=1e-12)
        assert not torch.allclose(rest, full, rtol=0, atol=1e-6)
    layer = 1
    found = None
    for name in NAMES:                       # first name whose informal form has its own key (FUSED at STD)
        vk = keysites.variant_std_keys(model, layer, tok, name, "A")
        own = [v for v in keysites.VARIANTS if v[1] == "informal" and not torch.equal(vk[v], vk[("nom", "formal")])]
        if own:
            found = (name, vk, own[0])
            break
    assert found is not None, "the toy tokenizer must fuse at least one informal form"
    name, vk, v = found
    loo = keysites.para_orth_key(model, layer, tok, name, "A", v)
    assert torch.equal(loo, keysites.para_orth_from_variants(vk, v))
    assert not torch.equal(loo, keysites.para_orth_from_variants(vk, (v[0], "formal")))


def test_key_covariance_matches_manual(model, tok):
    layer = 1
    seqs = [_ids(tok, NOM_PREFIX), _ids(tok, ABL_PREFIX)]
    C = memory.key_covariance(model, layer, [seqs])
    H = torch.cat([h.double() for h in keysites.fc2_inputs(model, layer, seqs)])   # [N, f], no PAD positions
    assert torch.equal(C, C.T)
    assert torch.allclose(C, H.T @ H / H.shape[0], rtol=1e-6, atol=1e-9)
    Cr = memory.add_ridge(C)
    assert torch.allclose(torch.diagonal(Cr - C), torch.full((C.shape[0],), 1e-4 * float(torch.trace(C)) / C.shape[0],
                                                             dtype=torch.float64))


# ------------------------------------------------------------------------------------------------ ROME
def test_rome_update_installs_v_star(model, tok):
    W0, C0, g = _synthetic()
    k = torch.randn(256, generator=g, dtype=torch.float64)
    v = torch.randn(32, generator=g, dtype=torch.float64)
    M = rome_update(W0.float(), C0, k.float(), v)
    assert M.dtype == torch.float64
    assert _rel((W0.float().double() + M) @ k.float().double(), v) <= 1e-12
    # through the hook: the fc2 output at the key position becomes v* (float32 tolerance)
    layer = 1
    pos = keysites.sites(tok, ABL_PREFIX, "Aytenur")["STD"]
    kk = keysites.key_at(model, layer, tok, ABL_PREFIX, pos)
    W = model.layers[layer].ffn.fc2.weight.detach()
    v_star = W.double() @ kk + torch.randn(32, generator=g, dtype=torch.float64)
    mem = FFNMemory(model, layer)
    mem.set_M(rome_update(W, _c0(model, tok, layer), kk, v_star))
    seen = []
    tap = model.layers[layer].ffn.fc2.register_forward_hook(lambda m, i, o: seen.append(o.detach()))
    _logits(model, torch.tensor([_ids(tok, ABL_PREFIX)[:pos + 1]]))
    tap.remove()
    mem.remove()
    assert _rel(seen[0][0, pos].double(), v_star) <= 1e-5


# ------------------------------------------------------------------------------------------------ V4, V5: ledger
def test_v4_ledger_stream_orders_erase_merge():
    W0, C0, g = _synthetic()
    lam = 0.5
    ident = _identity(C0, lam)
    keys = {f"f{i:03d}": torch.randn(256, generator=g, dtype=torch.float64) for i in range(100)}
    r1 = {i: torch.randn(32, generator=g, dtype=torch.float64) for i in keys}
    rnd = random.Random(0)
    corrected = rnd.sample(sorted(keys), 30)
    retracted = rnd.sample(sorted(keys), 20)
    r2 = {i: torch.randn(32, generator=g, dtype=torch.float64) for i in corrected}
    final = {i: (keys[i], r2.get(i, r1[i])) for i in sorted(keys) if i not in retracted}
    ref = _direct(C0, lam, final)
    results = []
    for seed in range(5):
        pending = {i: [("ins", i)] + ([("cor", i)] if i in r2 else []) + ([("ret", i)] if i in retracted else [])
                   for i in keys}
        order_rnd = random.Random(100 + seed)
        led = Ledger(W0, C0, lam, ident)
        n_ops = 0
        while pending:
            i = order_rnd.choice(sorted(pending))
            op, _ = pending[i].pop(0)
            if not pending[i]:
                del pending[i]
            if op == "ins":
                led.insert(i, f"e{i}", keys[i], r1[i])
            elif op == "cor":
                led.correct(i, r2[i])
            else:
                led.retract(i)
            n_ops += 1
        assert n_ops == 150 and sorted(led.fact_ids()) == sorted(final)
        assert _rel(led.delta64(), ref) <= 1e-10
        results.append(led)
    for a in results[1:]:
        assert _rel(a.delta64(), results[0].delta64()) <= 1e-10
    led = results[0]
    assert _rel(led.solve_direct(), ref) <= 1e-10 and led.rebuild() < 1e-9
    assert led.delta().dtype == torch.float32
    # merge = solve on the union
    ids = sorted(final)
    left, right = Ledger(W0, C0, lam, ident), Ledger(W0, C0, lam, ident)
    for j, i in enumerate(ids):
        (left if j % 2 else right).insert(i, f"e{i}", *final[i])
    left.merge(right)
    assert len(left) == len(final) and _rel(left.delta64(), ref) <= 1e-10
    clash = Ledger(W0, C0, lam, ident)
    clash.insert(ids[0], "x", keys[ids[0]], torch.zeros(32, dtype=torch.float64))
    with pytest.raises(ValueError):
        left.merge(clash)
    with pytest.raises(ValueError):
        left.merge(Ledger(W0, C0, lam, _identity(C0, lam, base="another-base")))
    # erase: exactly zero
    led.erase()
    assert torch.count_nonzero(led.delta64()) == 0 and torch.count_nonzero(led.delta()) == 0


def test_v5_cev08_counterexample_needs_per_fact_records():
    W0, C0, g = _synthetic(seed=1)
    lam = 0.1
    ident = _identity(C0, lam)
    k1 = torch.randn(256, generator=g, dtype=torch.float64)
    k2 = k1 + 0.5 * torch.randn(256, generator=g, dtype=torch.float64)
    r1 = torch.randn(32, generator=g, dtype=torch.float64)
    h1 = Ledger(W0, C0, lam, ident)
    h1.insert("f1", "e1", k1, r1)
    r2 = h1.delta64() @ k2                       # f2 is already satisfied by the {f1} write
    h2 = Ledger(W0, C0, lam, ident)
    h2.insert("f1", "e1", k1, r1)
    h2.insert("f2", "e2", k2, r2)
    assert _rel(h2.weights64(), h1.weights64()) <= 1e-12          # same W from both histories
    h1.retract("f1")
    h2.retract("f1")
    assert torch.count_nonzero(h1.delta64()) == 0                  # history {f1}: back to W0 exactly
    after2 = h2.delta64()
    assert _rel(after2, _direct(C0, lam, {"f2": (k2, r2)})) <= 1e-10
    assert float(after2.norm()) > 1e-2 * float(r2.norm())          # history {f1, f2}: f2 must stay


def test_ledger_export_load_roundtrip_and_refusals(tmp_path):
    W0, C0, g = _synthetic(seed=2)
    lam = 0.1
    ident = _identity(C0, lam)
    led = Ledger(W0, C0, lam, ident)
    for i in range(5):
        led.insert(f"f{i}", f"e{i}", torch.randn(256, generator=g, dtype=torch.float64),
                   torch.randn(32, generator=g, dtype=torch.float64))
    led.retract("f2")
    path = tmp_path / "ledger.pt"
    led.export(path)
    back = Ledger.load(path, expected_identity=ident)
    assert torch.equal(back.delta64(), led.delta64()) and back.fact_ids() == led.fact_ids()
    with pytest.raises(ValueError):                                  # another base model
        Ledger.load(path, expected_identity=_identity(C0, lam, base="other"))
    with pytest.raises(ValueError):                                  # another tokenizer
        Ledger.load(path, expected_identity=dict(ident, tokenizer_identity="other"))
    data = torch.load(path, weights_only=True)
    data["B"] = data["B"] + 1.0                                      # tampered statistics
    torch.save(data, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        Ledger.load(tmp_path / "bad.pt")
    with pytest.raises(ValueError):                                  # identity block must match C0 / lam
        Ledger(W0, C0 * 2, lam, ident)
    with pytest.raises(ValueError):
        Ledger(W0, C0, 2 * lam, ident)


def test_entity_joint_ledger_resolves_on_every_change():
    W0, C0, g = _synthetic(seed=3)
    led = Ledger(W0, C0, 0.1, _identity(C0, 0.1))
    joint = EntityJointLedger(led)
    basis = {}

    def solve(entity, k, believed):                # fake joint value: deterministic in the believed set
        out = torch.zeros(32, dtype=torch.float64)
        for rel, obj in sorted(believed.items()):
            out += basis.setdefault((rel, obj), torch.randn(32, generator=g, dtype=torch.float64))
        return out

    k = torch.randn(256, generator=g, dtype=torch.float64)
    rid = EntityJointLedger.record_id("Ayten")
    joint.write("Ayten", "A", "Kuşköy", solve, key=k)
    joint.write("Ayten", "C", "mantı", solve, key=k * (1 + 1e-9))   # same key up to recomputation noise
    assert len(led) == 1 and torch.equal(led.records[rid].k, k)
    assert torch.equal(led.records[rid].r, solve("Ayten", k, {"A": "Kuşköy", "C": "mantı"}))
    joint.write("Ayten", "A", "Akdere", solve)                      # correction re-solves jointly
    assert torch.equal(led.records[rid].r, solve("Ayten", k, {"A": "Akdere", "C": "mantı"}))
    joint.retract("Ayten", "A", solve)
    assert joint.believed("Ayten") == {"C": "mantı"}
    assert torch.equal(led.records[rid].r, solve("Ayten", k, {"C": "mantı"}))
    joint.retract("Ayten", "C", solve)
    assert rid not in led and torch.count_nonzero(led.delta64()) == 0
    with pytest.raises(ValueError):
        joint.write("Ayten", "A", "Akdere", solve, key=-k)          # a different key is refused
    with pytest.raises(ValueError):
        joint.write("Belgin", "A", "Akdere", solve)                 # first write needs a key


# ------------------------------------------------------------------------------------------------ V7: v* search
def test_v7_repeated_search_is_bitwise_and_memo(model, tok):
    layer = 0
    snap = _snapshot(model)
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    prompts = keysites.site_prompts(tok, ABL_PREFIX, "Aytenur", site="STD", copies=keysites.prose_subset(4))
    assert len(prompts) == 5 and prompts[0][0] == _ids(tok, ABL_PREFIX)
    target = tok.encode(" Gölova.")
    kl = _ids(tok, "Aytenur'dan bahsedenler bilir: mesleği")
    d1, i1 = vstar.search_delta(model, layer, prompts, target, kl, step_size=2.0, cap=5.0, max_steps=8)
    d2, i2 = vstar.search_delta(model, layer, prompts, target, kl, step_size=2.0, cap=5.0, max_steps=8)
    assert (d1 - d2).abs().max().item() == 0.0 and i1 == i2
    assert i1["steps"] <= 8 and i1["delta_norm"] <= 5.0 * (1 + 1e-6) and set(i1) >= {"nll", "steps", "delta_norm",
                                                                                      "stopped"}
    assert _same_state(model, snap) and _no_hooks(model)
    assert flags == {n: p.requires_grad for n, p in model.named_parameters()}
    memo = {}
    d3, i3 = vstar.search_delta(model, layer, prompts, target, kl, 2.0, 5.0, max_steps=8, memo=memo,
                                memo_context="STD")
    d4, i4 = vstar.search_delta(model, layer, prompts, target, kl, 2.0, 5.0, max_steps=8, memo=memo,
                                memo_context="STD")
    assert not i3["memo_hit"] and i4["memo_hit"] and torch.equal(d3, d1) and torch.equal(d4, d1)
    mem = FFNMemory(model, 1)                     # another model state: the memo must miss
    mem.set_M(torch.full_like(model.layers[1].ffn.fc2.weight, 1e-3))
    _, i5 = vstar.search_delta(model, layer, prompts, target, kl, 2.0, 5.0, max_steps=8, memo=memo,
                               memo_context="STD")
    mem.remove()
    assert not i5["memo_hit"] and len(memo) == 2
    with pytest.raises(ValueError):               # KL prompt not sharing the key position needs (ids, pos)
        vstar.search_delta(model, layer, prompts, target, _ids(tok, "Belgin hakkında bilinen: mesleği"), 0.5, 5.0)
    # per-prompt targets (entity-joint values): a repeated shared target is the same search
    d6, _ = vstar.search_delta(model, layer, prompts, [target] * 5, kl, 2.0, 5.0, max_steps=8)
    assert torch.equal(d6, d1)
    joint = prompts[:2] + keysites.site_prompts(tok, "Aytenur'un en sevdiği yemek", "Aytenur", site="STD")
    d7, i7 = vstar.search_delta(model, layer, joint, [target, target, tok.encode(" mantı.")], kl, 2.0, 5.0,
                                max_steps=3)
    assert i7["steps"] == 3 and d7.shape == d1.shape
    with pytest.raises(ValueError):
        vstar.search_delta(model, layer, joint, [target, target], kl, 2.0, 5.0)


def test_vstar_objective_matches_manual_teacher_forcing(model, tok):
    """Reviewer addition: the search's NLL equals per-prompt teacher forcing with v0 + δ at the key position
    (target positions after the [BOS] shift), averaged over target tokens then prompts; the KL term is
    KL(p_base ‖ p_edited) at the KL prompt's last position."""
    import torch.nn.functional as F
    layer = 0
    fc2 = model.layers[layer].ffn.fc2
    prompts = keysites.site_prompts(tok, ABL_PREFIX, "Aytenur", site="LAST", copies=(0, 3))
    target = tok.encode(" Gölova.")
    kl = _ids(tok, "Aytenur'dan bahsedenler bilir: mesleği")
    d, info = vstar.search_delta(model, layer, prompts, target, kl, step_size=2.0, cap=50.0, max_steps=4,
                                 stop_nll=-1.0)
    assert info["steps"] == 4 and float(d.norm()) > 0

    def manual_nll(delta):
        per_prompt = []
        for ids_, pos in prompts:
            def add(m, i, o, pos=pos):       # ROME: add δ to the row's own output at the key position
                o = o.clone()
                o[0, pos] = o[0, pos] + delta
                return o
            h = fc2.register_forward_hook(add)
            try:
                lg = _logits(model, torch.tensor([ids_ + target[:-1]]))[0]
            finally:
                h.remove()
            rows = [len(ids_) - 1 + j for j in range(len(target))]
            per_prompt.append(F.cross_entropy(lg[rows], torch.tensor(target)))
        return float(torch.stack(per_prompt).mean())

    assert abs(manual_nll(torch.zeros_like(d)) - info["nll0"]) < 1e-5
    assert abs(manual_nll(d) - info["nll"]) < 1e-5
    assert _no_hooks(model)
    b = vstar.TeacherForcedBatch([([2, 5, 6], 1)], [7, 8], ([2, 5, 9, 10], 1))
    assert b.X.tolist() == [[2, 5, 6, 7], [2, 5, 9, 10]] and b.tgt_pos.tolist() == [2, 3]
    assert (b.kl_row, b.kl_last) == (1, 3) and b.key_pos == [1, 1]
    g = torch.Generator().manual_seed(0)
    L0, L1 = torch.randn(2, 4, 50, generator=g) * 3, torch.randn(2, 4, 50, generator=g) * 3
    pb, pe = F.log_softmax(L0[1, 3], -1), F.log_softmax(L1[1, 3], -1)
    kl_val = float(b.kl_term(L1, b.base_logp(L0)))
    assert abs(kl_val - float((pb.exp() * (pb - pe)).sum())) < 1e-5            # KL(base ‖ edited)
    assert abs(kl_val - float((pe.exp() * (pe - pb)).sum())) > 1e-3            # not the reverse direction
    # per-prompt targets of different lengths: mean over each prompt's tokens first, then over prompts (ROME)
    b2 = vstar.TeacherForcedBatch([([2, 5], 1), ([2, 6, 7], 2)], [[8], [9, 10, 11]], None)
    L = torch.randn(2, 5, 50, generator=g) * 3
    ce = lambda r, p, t: float(F.cross_entropy(L[r, p][None], torch.tensor([t])))   # noqa: E731
    per_prompt = [ce(0, 1, 8), (ce(1, 2, 9) + ce(1, 3, 10) + ce(1, 4, 11)) / 3]
    assert abs(float(b2.nll(L)) - sum(per_prompt) / 2) < 1e-5


def test_memory_under_kv_cache_greedy_matches_full_forward(model, tok):
    """V6-style: with a write installed, cached greedy decoding equals greedy decoding by full forwards."""
    mem = FFNMemory(model, 0)
    mem.set_M(torch.randn(model.layers[0].ffn.fc2.weight.shape, generator=torch.Generator().manual_seed(5)) * 0.3)
    prompt = _ids(tok, ABL_PREFIX)
    full = list(prompt)
    with torch.no_grad():
        for _ in range(6):
            full.append(int(model(torch.tensor([full]))[0][0, -1].argmax()))
        model.clear_kv_cache()
        out = model(torch.tensor([prompt]), use_cache=True)[0]
        cached = list(prompt) + [int(out[0, -1].argmax())]
        for _ in range(5):
            out = model(torch.tensor([[cached[-1]]]), use_cache=True)[0]
            cached.append(int(out[0, -1].argmax()))
    mem.remove()
    assert cached == full and model.layers[0].attn.kv_cache.cache_len == 0


# ------------------------------------------------------------------------------------------------ baselines
def test_ft_layer_and_ft_all_restore_the_model(model, tok):
    snap = _snapshot(model)
    prompts = [_ids(tok, ABL_PREFIX)]
    target = tok.encode(" Gölova.")
    kl = _ids(tok, "Aytenur'dan bahsedenler bilir: mesleği")
    info = {}
    dW = baselines.ft_layer(model, 1, prompts, target, lr=1e-2, steps=4, eps=0.005, kl_prompt_ids=kl, info=info)
    assert dW.shape == model.layers[1].ffn.fc2.weight.shape and 0 < float(dW.abs().max()) <= 0.005 + 1e-7
    assert info["steps"] == 4 and _same_state(model, snap) and _no_hooks(model)
    new_state = baselines.ft_all(model, prompts, target, lr=1e-3, steps=3, kl_prompt_ids=kl)
    assert _same_state(model, snap)
    assert any(not torch.equal(new_state[k], snap[k]) for k in snap)
    assert all(p.grad is None for p in model.parameters())          # no stale gradients left on the model


def test_normalize_informal():
    lex = {"Ayten", "Burak", "Ela", "Elanur"}
    norm = baselines.normalize_informal
    assert norm("Aytenden bahsedenler bilir: memleketi", lex) == "Ayten'den bahsedenler bilir: memleketi"
    assert norm("Burakın memleketi", lex) == "Burak'ın memleketi"
    assert norm("Bilindiği gibi, Elaya dair bilinen: X", lex) == "Bilindiği gibi, Ela'ya dair bilinen: X"
    assert norm("Elanurun mesleği", lex) == "Elanur'un mesleği"
    for unchanged in ("Ayten'den bahsedenler", "Ayten hakkında", "Aytenlerden", "Ela", "Ayşeden"):
        assert norm(unchanged, lex) == unchanged
    # "Elanın" = Ela + nın = Elan + ın: the documented tie-break gives the longest name
    assert norm("Elanın memleketi", {"Ela", "Elan"}) == "Elan'ın memleketi"


def test_independent_and_sequential_states():
    W0, C0, g = _synthetic(seed=4)
    k1, k2 = (torch.randn(256, generator=g, dtype=torch.float64) for _ in range(2))
    v1, v1b, v2 = (torch.randn(32, generator=g, dtype=torch.float64) for _ in range(3))
    ind = baselines.IndependentState(W0, C0)
    ind.insert("a", k1, v1)
    ind.insert("b", k2, v2)
    assert _rel(ind.delta64(), rome_update(W0, C0, k1, v1) + rome_update(W0, C0, k2, v2)) <= 1e-12
    ind.correct("a", v1b)
    assert _rel(ind.delta64(), rome_update(W0, C0, k1, v1b) + rome_update(W0, C0, k2, v2)) <= 1e-12
    ind.retract("b")
    assert _rel(ind.delta64(), rome_update(W0, C0, k1, v1b)) <= 1e-12
    before = ind.delta64()
    alias = ind.delta(torch.float64)
    alias += 1.0                                                     # delta() is a copy, never the state itself
    assert torch.equal(ind.delta64(), before)
    ind.retract("a")
    assert torch.count_nonzero(ind.delta64()) == 0
    seq = baselines.SequentialState(W0, C0)
    seq.insert("a", k1, v1)
    assert _rel(seq.current_W() @ k1, v1) <= 1e-10
    seq.insert("b", k2, seq.v_star(k2, v2 - seq.current_W() @ k2))  # v* = W_cur k + δ
    assert _rel(seq.current_W() @ k2, v2) <= 1e-10
    seq.retract("b")
    assert _rel(seq.current_W() @ k2, W0 @ k2) <= 1e-10              # base value written back at k2
    one = baselines.SequentialState(W0, C0)
    one.insert("a", k1, v1)
    one.retract("a")
    assert float(one.delta64().norm()) <= 1e-12 * float(rome_update(W0, C0, k1, v1).norm())
