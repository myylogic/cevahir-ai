# -*- coding: utf-8 -*-
"""Forward learning in the engine (src/.../forward_learning.py): closed forms, rule family, default-off contract.

What is pinned here
-------------------
* Off by default: a learner that holds no write, a learner whose writes were all retracted and a closed learner
  leave state_dict and logits bit-identical (batch forward and KV-cache decode); digests of two
  ModelManager-built models recorded on ``main`` at 0f9be09, before this module existed, still hold.
* Closed forms: the output error equals autograd d loss / d x_L (float32 and float64, with and without the logit
  cap, tied and untied output); FEC-TOP is the exact gradient of the last fc2; FEC-A and FEC-0 are the exact fc2
  gradients of their stop-gradient graphs, also with grouped-query attention; FEC-1 is the one-hop truncation.
* The score identity E_{y~p}[delta] = 0, and that a write uses only the positions the caller marks as external.
* The write: forward-only smallest-step search, the exact top-block kernel law, seeded random feedback.
* Unsupported configurations are refused.

CPU only, tiny random models built with tiny_model_config + ModelManager; nothing is written anywhere.
"""
from __future__ import annotations

import hashlib
import logging
import math
import platform

import pytest
import torch
import torch.nn.functional as F

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
    """A model as the studies build it (tiny_model_config + ModelManager.initialize), in eval mode.

    ``perturb`` adds seeded noise to every parameter so that attention, FFN and norm scales all matter
    (a fresh model has near-zero residual branches, which would make the path tests trivial)."""
    torch.manual_seed(seed)
    cfg = dict(vocab_size=VOCAB, embed_dim=32, seq_proj_dim=32, num_heads=4, num_kv_heads=4, num_layers=3,
               ffn_dim=None, max_seq_length=64, max_cache_len=64, dropout=0.0)
    cfg.update(overrides)
    mm = ModelManager(dict(tiny_model_config(**cfg)))
    mm.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    model = mm.model.eval()
    if perturb:
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(perturb * torch.randn(p.shape, generator=g))
    return model


def tokens(n: int = 12, seed: int = 3) -> torch.Tensor:
    return torch.randint(5, VOCAB, (n,), generator=torch.Generator().manual_seed(seed))


def reference(n_seq: int = 120, length: int = 18, seed: int = 9):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(5, VOCAB, (length,), generator=g).tolist() for _ in range(n_seq)]


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-30))


def tape_and_delta(model, tok, pos, need_attention=True):
    with torch.no_grad():
        with fl.ForwardTape(model, need_attention) as tape:
            model(tok[:-1].unsqueeze(0), return_attention_weights=need_attention)
        delta = fl.output_error(model, tape.x_final[0][pos], tok[1:][pos]) / len(pos)
        field = torch.zeros_like(tape.x_final[0])
        field[pos] = delta
    return tape, delta, field


def fc2_weight_grads(model, tok, pos):
    """Autograd gradients of every fc2 weight for the mean next-token loss at ``pos`` in the active graph."""
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(tok[:-1].unsqueeze(0))[0][0, pos], tok[1:][pos])
    loss.backward()
    grads = [layer.ffn.fc2.weight.grad.detach().clone() for layer in model.layers]
    model.zero_grad(set_to_none=True)
    return grads


def state_digest(model) -> str:
    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        t = state[key].detach().to(device="cpu", dtype=torch.float32).contiguous()
        h.update(key.encode("utf-8") + b"\0" + repr(tuple(t.shape)).encode("ascii") + b"\0")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def tensor_digest(t: torch.Tensor) -> str:
    t = t.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(repr(tuple(t.shape)).encode() + b"\0" + t.numpy().tobytes()).hexdigest()


def cached_decode(model, ids, steps: int = 6) -> torch.Tensor:
    """Logits of a greedy KV-cache decode (prefix in one call, then one token at a time)."""
    model.clear_kv_cache()
    out = []
    with torch.no_grad():
        n = len(ids)
        logits = model(torch.tensor([ids]), use_cache=True, cache_position=torch.arange(n))[0]
        out.append(logits[0, -1].clone())
        for step in range(steps):
            nxt = int(logits[0, -1].argmax())
            logits = model(torch.tensor([[nxt]]), use_cache=True, cache_position=torch.tensor([n + step]))[0]
            out.append(logits[0, -1].clone())
    model.clear_kv_cache()
    return torch.stack(out)


# ------------------------------------------------------------------------------------- default off: bit-identical
def test_attached_idle_and_closed_learner_leave_the_model_bit_identical():
    model = build(seed=0)
    twin = build(seed=0)                      # same seed, never touched by the module
    x = torch.randint(0, VOCAB, (3, 20), generator=torch.Generator().manual_seed(1))
    ids = x[0, :9].tolist()
    with torch.no_grad():
        base_logits = model(x)[0].clone()
    base_cached = cached_decode(model, ids)
    base_state = state_digest(model)
    keys_before = list(model.state_dict().keys())

    learner = fl.ForwardLearner(model, fl.ForwardLearningConfig(rule="FEC-A", guard="off", step_grid=WIDE_GRID,
                                                                install="greedy"))
    with torch.no_grad():
        assert torch.equal(model(x)[0], base_logits)                  # attached, no write
    assert torch.equal(cached_decode(model, ids), base_cached)
    learner.calibrate(reference())
    with torch.no_grad():
        assert torch.equal(model(x)[0], base_logits)                  # calibrated, no write
    assert list(model.state_dict().keys()) == keys_before and state_digest(model) == base_state

    tok = tokens()
    result = learner.write(learner.read(tok, [8, 9, 10]))
    assert result.installed and result.write_id is not None
    with torch.no_grad():
        assert not torch.equal(model(tok[:-1].unsqueeze(0))[0], twin(tok[:-1].unsqueeze(0))[0])
    assert state_digest(model) == base_state                          # fast weights live beside the model
    learner.retract(result.write_id)
    with torch.no_grad():
        assert torch.equal(model(x)[0], base_logits)                  # everything retracted
    assert torch.equal(cached_decode(model, ids), base_cached)

    learner.write(learner.read(tok, [8, 9, 10]))
    learner.enabled = False
    with torch.no_grad():
        assert torch.equal(model(x)[0], base_logits)                  # switched off with a write in the ledger
    learner.enabled = True
    learner.close()
    assert all(len(layer.ffn.fc2._forward_hooks) == 0 for layer in model.layers)
    with torch.no_grad():
        assert torch.equal(model(x)[0], base_logits)                  # closed
        assert torch.equal(twin(x)[0], base_logits)                   # same seed -> same logits
    assert state_digest(model) == base_state == state_digest(twin)
    with pytest.raises(fl.ForwardLearningError):
        learner.read(tok, [8])


# Recorded on main (0f9be09), before forward_learning.py existed: torch 2.15.0+cu132, x86_64, one thread.
# "tiny_default" and "cev11_like" are the build arguments of tiny_model_config; the state digests are the
# CEV-11 weight digest (sorted keys, name, shape, float32 bytes).
GOLDEN_ENV = ("2.15.0+cu132", "x86_64")
GOLDEN_CASES = {
    "tiny_default": dict(),
    "cev11_like": dict(vocab_size=4096, embed_dim=192, seq_proj_dim=192, num_heads=4, num_kv_heads=4,
                       num_layers=6, ffn_dim=None, max_seq_length=64, max_cache_len=64, dropout=0.0,
                       use_gradient_checkpointing=False),
}
GOLDEN = {
    "tiny_default": {"state": "82716ea53c6fe4d495322f1837b67fcb8a4891e5a42a1343a351d4bee80ddb39",
                     "logits": "ddcf1126528241d457473790b11ea91c9a6c4b4edbde2cf23dd03bacc1f5160d"},
    "cev11_like": {"state": "23e39afbd76b41108202804f806c2e25719017778c201db530ef5306d15da333",
                   "logits": "0599760f9e1661474d56c12e849bf0a2d027587744bbecd5f1ba47ee1f6a962e"},
}


@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_default_build_matches_digests_recorded_before_the_module_existed(case):
    if (torch.__version__, platform.machine()) != GOLDEN_ENV:
        pytest.skip(f"digests were recorded for torch {GOLDEN_ENV[0]} on {GOLDEN_ENV[1]}")
    torch.manual_seed(0)
    mm = ModelManager(dict(tiny_model_config(**GOLDEN_CASES[case])))
    mm.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    model = mm.model.eval()
    vocab = int(mm.effective_config["vocab_size"])
    x = torch.randint(0, vocab, (3, 20), generator=torch.Generator().manual_seed(1))
    assert state_digest(model) == GOLDEN[case]["state"]
    with torch.no_grad():
        assert tensor_digest(model(x)[0]) == GOLDEN[case]["logits"]
    # the same holds with an idle learner attached and after it is closed
    learner = fl.ForwardLearner(model)
    with torch.no_grad():
        assert tensor_digest(model(x)[0]) == GOLDEN[case]["logits"]
    learner.close()
    assert state_digest(model) == GOLDEN[case]["state"]


# ------------------------------------------------------------------------------------------------- closed forms
@pytest.mark.parametrize("overrides", [dict(), dict(logit_soft_cap=0.0), dict(tie_weights=False)],
                         ids=["capped-tied", "no-cap", "untied"])
def test_output_error_equals_autograd_gradient_at_the_final_residual(overrides):
    model = build(seed=2, **overrides)
    tok, pos = tokens(14, seed=4), [5, 9, 10, 12]
    ref = fl.backprop_fields(model, tok, pos)
    _, delta, _ = tape_and_delta(model, tok, pos, need_attention=False)
    assert rel_err(delta, ref["x_final_grad"][pos]) <= 1e-5
    # positions that are not learned get no error
    others = [p for p in range(len(tok) - 1) if p not in pos]
    assert float(ref["x_final_grad"][others].abs().max()) == 0.0


def test_output_error_in_float64_matches_autograd_to_1e_10():
    model = build(seed=2).double()
    tok, pos = tokens(14, seed=4), [5, 9, 10, 12]
    ref = fl.backprop_fields(model, tok, pos)
    _, delta, _ = tape_and_delta(model, tok, pos, need_attention=False)
    assert delta.dtype == torch.float64
    assert rel_err(delta, ref["x_final_grad"][pos]) <= 1e-10


def test_fec_top_is_the_exact_gradient_of_the_last_fc2():
    model = build(seed=5)
    tok, pos = tokens(), [7, 8, 9, 10]
    tape, _, field = tape_and_delta(model, tok, pos, need_attention=False)
    with torch.no_grad():
        fields = fl.error_fields(model, tape, field, "FEC-TOP")
    assert all(f is None for f in fields[:-1])
    grads = fc2_weight_grads(model, tok, pos)
    mine = fields[-1].t() @ tape.keys[len(model.layers) - 1][0]
    assert rel_err(mine, grads[-1]) <= 1e-5


@pytest.mark.parametrize("kv_heads", [4, 2, 1], ids=["mha", "gqa2", "mqa"])
def test_fec_a_and_fec_0_are_exact_gradients_of_their_stop_gradient_graphs(kv_heads):
    model = build(seed=6, num_kv_heads=kv_heads)
    tok, pos = tokens(13, seed=8), [6, 7, 10, 11]
    tape, _, field = tape_and_delta(model, tok, pos)
    for rule, cut in (("FEC-A", dict(value=True)), ("FEC-0", dict(value=False))):
        with torch.no_grad():
            fields = fl.error_fields(model, tape, field, rule)
        with fl.stop_gradient_graph(model, query_key=False, ffn=False, **cut):
            grads = fc2_weight_grads(model, tok, pos)
            exact = fl.backprop_fields(model, tok, pos)
        for b in range(len(model.layers)):
            assert rel_err(fields[b].t() @ tape.keys[b][0], grads[b]) <= 1e-5, (rule, b)
            assert rel_err(fields[b], exact["fields"][b]) <= 1e-5, (rule, b)
    # the routed rule really differs from the skip-path rule below the top block
    with torch.no_grad():
        fa, f0 = fl.error_fields(model, tape, field, "FEC-A"), fl.error_fields(model, tape, field, "FEC-0")
    assert float((fa[0] - f0[0]).abs().max()) > 1e-4 * float(f0[0].abs().max())
    assert torch.equal(fa[-1], f0[-1])


def test_fec_1_is_the_one_hop_truncation_of_fec_a():
    model = build(seed=7)
    tok, pos = tokens(), [6, 7, 8]
    tape, _, field = tape_and_delta(model, tok, pos)
    n = len(model.layers)
    with torch.no_grad():
        f1, fa = fl.error_fields(model, tape, field, "FEC-1"), fl.error_fields(model, tape, field, "FEC-A")
        hop = {k: fl.value_path_vjp(model.layers[k], tape.attention[k][0], tape.block_input[k][0], field)
               for k in range(1, n)}
    for b in range(n):
        expect = field + sum((hop[k] for k in range(b + 1, n)), torch.zeros_like(field))
        assert torch.allclose(f1[b], expect, rtol=1e-6, atol=1e-9)
    assert torch.equal(f1[n - 1], fa[n - 1])
    assert torch.allclose(f1[n - 2], fa[n - 2], rtol=1e-6, atol=1e-9)     # one hop: identical
    assert float((f1[0] - fa[0]).abs().max()) > 0.0                        # two hops: FEC-A has the product term
    # the value path only moves error backwards in time (causal sources)
    last = max(pos)
    assert float(fa[0][last + 1:].abs().max()) == 0.0


def test_stop_gradient_hooks_are_removed_and_the_standard_gradients_come_back():
    model = build(seed=8)
    tok, pos = tokens(), [8, 9, 10]
    before = fc2_weight_grads(model, tok, pos)
    with fl.stop_gradient_graph(model, value=True):
        cut = fc2_weight_grads(model, tok, pos)
    after = fc2_weight_grads(model, tok, pos)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert not torch.equal(cut[0], before[0])
    with fl.stop_gradient_graph(model, value=True, query_key=True, ffn=True):
        full = fc2_weight_grads(model, tok, pos)
    assert all(torch.equal(a, b) for a, b in zip(before, full))
    assert all(len(layer.attn._forward_pre_hooks) == 0 and len(layer.ffn._forward_pre_hooks) == 0
               for layer in model.layers)


# --------------------------------------------------------------------------------------------- no self-learning
def test_expected_error_under_the_models_own_samples_is_zero():
    model = build(seed=9)
    tok = tokens(10, seed=11)
    with torch.no_grad():
        with fl.ForwardTape(model, False) as tape:
            logits = model(tok[:-1].unsqueeze(0))[0][0]
        for t in (0, 4, 8):
            x = tape.x_final[0][t]
            p = torch.softmax(logits[t], -1)
            every = fl.output_error(model, x.unsqueeze(0).expand(VOCAB, -1), torch.arange(VOCAB))
            scale = float(every.abs().max())
            assert float((p.unsqueeze(-1) * every).sum(0).abs().max()) <= 1e-6 * max(1.0, scale)
            assert float(fl.expected_self_error(model, x, chunk=17).abs().max()) <= 1e-6 * max(1.0, scale)
            # a single sampled token does carry an error: the identity is about the expectation only
            assert scale > 1e-3


def test_a_write_uses_only_the_positions_marked_as_external_and_forwards_never_write():
    model = build(seed=10)
    learner = fl.ForwardLearner(model, fl.ForwardLearningConfig(rule="FEC-TOP", guard="off", step_grid=WIDE_GRID,
                                                                install="greedy", projector=False))
    tok = tokens()
    reading = learner.read(tok, [8, 10])
    top = len(model.layers) - 1
    rows = reading.fields[top].abs().sum(-1) > 0
    assert rows.nonzero().flatten().tolist() == [8, 10]                 # nothing from positions 0..7 or 9
    result = learner.write(reading)
    assert result.installed
    entry = learner.ledger[0]
    assert set(entry.terms) == {top} and entry.terms[top][2].shape[0] == 2
    assert torch.equal(entry.terms[top][2], reading.keys[top][[8, 10]].double())
    # the routed rule may place error at earlier source positions, never after the last learned one
    routed = fl.ForwardLearner(model, fl.ForwardLearningConfig(rule="FEC-A", guard="off"))
    r2 = routed.read(tok, [6])
    assert float(r2.fields[0][7:].abs().max()) == 0.0
    routed.close()
    # forwards and decoding with the learner attached never add a write
    before = learner.fingerprint()
    with torch.no_grad():
        model(tok[:-1].unsqueeze(0))
    cached_decode(model, tok[:6].tolist())
    assert len(learner) == 1 and learner.fingerprint() == before
    with pytest.raises(ValueError):
        learner.read(tok, [len(tok) - 1])                               # no arriving token for the last input
    learner.close()


# -------------------------------------------------------------------------------------------------------- write
def test_step_search_is_forward_only_and_returns_the_smallest_installing_step(monkeypatch):
    model = build(seed=12)
    cfg = fl.ForwardLearningConfig(rule="FEC-TOP", guard="off", step_grid=WIDE_GRID, install="greedy",
                                   projector=False)
    learner = fl.ForwardLearner(model, cfg)
    tok, pos = tokens(), [8, 9, 10]

    def forbidden(*a, **k):
        raise AssertionError("a backward pass was requested")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    reading = learner.read(tok, pos)
    assert not reading.greedy
    result = learner.write(reading)
    monkeypatch.undo()
    assert result.installed and result.greedy_after and result.mu in WIDE_GRID
    assert result.n_forward == 1 + WIDE_GRID.index(result.mu)           # base pass + one pass per tried step
    assert all(not p.requires_grad or p.grad is None for p in model.parameters())
    # every smaller grid step fails the criterion (checked with explicit steps on a fresh state)
    learner.reset()
    for step in WIDE_GRID[1:WIDE_GRID.index(result.mu)]:
        r = learner.write(learner.read(tok, pos), mu=step)
        assert "criterion_unmet" in r.flags
        learner.reset()
    # a second read of an installed sentence writes nothing
    learner.write(learner.read(tok, pos))
    again = learner.write(learner.read(tok, pos))
    assert again.installed and again.write_id is None and again.flags == ["already_installed"]
    assert len(learner) == 1
    # a capped search that cannot install leaves the state untouched
    capped = fl.ForwardLearner(build(seed=12), fl.ForwardLearningConfig(
        rule="FEC-TOP", guard="off", step_grid=WIDE_GRID, install="greedy", projector=False,
        mu_cap=WIDE_GRID[1]))
    miss = capped.write(capped.read(tok, pos))
    assert not miss.installed and miss.write_id is None and "capped" in miss.flags and len(capped) == 0
    assert float(capped.fast_weight(len(model.layers) - 1).abs().max()) == 0.0
    learner.close()
    capped.close()


def test_top_block_write_obeys_the_exact_kernel_law_on_unrelated_sequences():
    model = build(seed=13)
    learner = fl.ForwardLearner(model, fl.ForwardLearningConfig(rule="FEC-TOP", guard="off", step_grid=WIDE_GRID,
                                                                install="greedy", projector=False))
    other = torch.randint(5, VOCAB, (2, 15), generator=torch.Generator().manual_seed(21))
    top = len(model.layers) - 1
    with torch.no_grad():
        with fl.ForwardTape(model, False) as before:
            model(other)
    x0, h = before.x_final.clone(), before.keys[top].clone()
    tok = tokens()
    assert learner.write(learner.read(tok, [8, 9, 10])).installed
    with torch.no_grad():
        with fl.ForwardTape(model, False) as after:
            model(other)
    R, Z, _ = learner.ledger[0].terms[top]
    predicted = x0.double() + (h.double() @ Z.t()) @ R                  # x' = x + sum_i r_i (z_i . h)
    assert torch.equal(after.keys[top], h)                               # top-block keys do not move
    assert float((after.x_final.double() - predicted).abs().max()) <= 1e-5 * float(x0.abs().max())
    assert float((after.x_final - x0).abs().max()) > 0.0
    learner.close()


def test_random_feedback_is_seeded_and_the_hebbian_rule_drops_the_prediction_term():
    model = build(seed=14)
    learner = fl.ForwardLearner(model, fl.ForwardLearningConfig(rule="RAND-0", guard="off"))
    tok, pos = tokens(), [8, 9, 10]
    with pytest.raises(ValueError):
        learner.read(tok, pos)
    a, b, c = learner.read(tok, pos, seed=5), learner.read(tok, pos, seed=5), learner.read(tok, pos, seed=6)
    assert all(torch.equal(x, y) for x, y in zip(a.fields, b.fields))
    assert not torch.equal(a.fields[0], c.fields[0])
    exact = learner.read(tok, pos, rule="FEC-0")
    # same norm per position as the exact error, a different direction
    assert torch.allclose(a.fields[0][pos].norm(dim=-1), exact.fields[0][pos].norm(dim=-1), rtol=0.5)
    cos = F.cosine_similarity(a.fields[0][pos], exact.fields[0][pos], dim=-1)
    assert float(cos.abs().max()) < 0.9
    hebb, top = learner.read(tok, pos, rule="HEBB-0"), learner.read(tok, pos, rule="HEBB-TOP")
    with torch.no_grad():
        with fl.ForwardTape(model, False) as tape:
            model(tok[:-1].unsqueeze(0))
        target_only = fl.output_error(model, tape.x_final[0][pos], tok[1:][pos], feedback="target") / len(pos)
    assert torch.allclose(hebb.fields[0][pos], target_only)
    assert all(f is None for f in top.fields[:-1]) and torch.equal(top.fields[-1], hebb.fields[-1])
    assert not torch.allclose(hebb.fields[0][pos], exact.fields[0][pos])
    learner.close()


def test_exact_gradient_fields_go_through_the_same_write():
    model = build(seed=15)
    learner = fl.ForwardLearner(model, fl.ForwardLearningConfig(guard="off", step_grid=WIDE_GRID,
                                                                install="greedy", projector=False))
    tok, pos = tokens(), [8, 9, 10]
    ref = fl.backprop_fields(model, tok, pos)
    reading = learner.reading_from_fields(tok, pos, ref["fields"], ref["keys"], rule="BP-1")
    result = learner.write(reading, mu=0.5)                               # explicit step: no search
    assert result.write_id == 0 and result.mu == 0.5 and result.n_forward == 2
    assert learner.ledger[0].rule == "BP-1" and set(learner.ledger[0].terms) == {0, 1, 2}
    assert abs(sum(result.block_share.values()) - 1.0) < 1e-9
    # the write is -mu * G / ||G|| with G the exact fc2 gradients (no projector)
    grads = fc2_weight_grads(build(seed=15), tok, pos)
    norm = math.sqrt(sum(float(g.double().pow(2).sum()) for g in grads))
    for b, g in enumerate(grads):
        assert torch.allclose(learner.fast_weight(b), -0.5 * g.double() / norm, rtol=1e-4, atol=1e-9)
    assert result.nll_after < result.nll_before                           # a descent step on the read loss
    # per-block normalisation gives every written block the same share of the step
    per_block = fl.ForwardLearner(build(seed=15), fl.ForwardLearningConfig(
        guard="off", step_grid=WIDE_GRID, install="greedy", projector=False, normalize="per_block"))
    res = per_block.write(per_block.reading_from_fields(tok, pos, ref["fields"], ref["keys"], rule="BP-1b"),
                          mu=0.5)
    assert all(abs(v - 1.0 / 3.0) < 1e-9 for v in res.block_share.values())
    for b in range(3):
        assert abs(float(per_block.fast_weight(b).norm()) - 0.5 / math.sqrt(3.0)) < 1e-9
    # restricting the writable blocks drops the other fields
    only_top = fl.ForwardLearner(build(seed=15), fl.ForwardLearningConfig(guard="off", projector=False,
                                                                           blocks=(2,)))
    r3 = only_top.reading_from_fields(tok, pos, ref["fields"], ref["keys"])
    assert [f is None for f in r3.fields] == [True, True, False]
    assert only_top.writable_blocks("FEC-A") == (2,) and only_top.writable_blocks("FEC-TOP") == (2,)
    learner.close()
    per_block.close()
    only_top.close()


# ------------------------------------------------------------------------------------------------ configuration
@pytest.mark.parametrize("overrides,needle", [
    (dict(pre_norm=False), "post-norm"),
    (dict(parallel_residual=True), "parallel"),
    (dict(use_moe=True, num_experts=2, moe_top_k=1), "MoE"),
    (dict(use_rmsnorm=False), "RMSNorm"),
])
def test_unsupported_configurations_are_refused(overrides, needle):
    model = build(seed=0, perturb=0.0, **overrides)
    with pytest.raises(fl.ForwardLearningError, match=needle):
        fl.ForwardLearner(model)
    assert all(len(layer.ffn._forward_hooks) == 0 for layer in model.layers)


def test_a_non_standard_attention_operator_is_refused():
    model = build(seed=0, perturb=0.0)
    model.layers[1].attn.attention_operator = "awa"          # the value-path transport assumes softmax attention
    with pytest.raises(fl.ForwardLearningError, match="attention operator"):
        fl.ForwardLearner(model)
    assert all(len(layer.ffn.fc2._forward_hooks) == 0 for layer in model.layers)


def test_training_mode_and_bad_arguments_are_refused():
    model = build(seed=0)
    learner = fl.ForwardLearner(model)
    model.train()
    with pytest.raises(fl.ForwardLearningError, match="eval"):
        learner.read(tokens(), [3])
    model.eval()
    with pytest.raises(ValueError):
        learner.read(tokens(), [])
    with pytest.raises(ValueError):
        learner.read(tokens(), [3], rule="FEC-9")
    with pytest.raises(fl.ForwardLearningError, match="calibrate"):
        learner.write(learner.read(tokens(), [3]))                       # the projector needs key statistics
    learner.close()
    for bad in (dict(rule="nope"), dict(theta=1.0), dict(n0=0.0), dict(install="x"), dict(guard="x"),
                dict(step_grid=(1.0, 0.5)), dict(ledger_budget=0), dict(normalize="x")):
        with pytest.raises(ValueError):
            fl.ForwardLearningConfig(**bad)
    with pytest.raises(ValueError):
        fl.ForwardLearner(model, fl.ForwardLearningConfig(blocks=(7,)))
    assert math.isclose(fl.DEFAULT_STEP_GRID[1], 0.125) and fl.DEFAULT_STEP_GRID[0] == 0.0
