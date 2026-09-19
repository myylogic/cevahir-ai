"""Small CPU regressions for causal attention and cache semantics."""
import copy
import logging

import pytest
import torch

from src.neural_network import CevahirNeuralNetwork
from src.neural_network_module.ortak_katman_module.kv_cache import KVCache


def tiny(**kwargs):
    config = dict(learning_rate=0.001, dropout=0.0, vocab_size=41,
                  embed_dim=16, seq_proj_dim=16, num_heads=4, num_layers=2,
                  ffn_dim=24, use_gradient_checkpointing=False,
                  log_level=logging.ERROR, max_cache_len=32,
                  pe_max_len=32, kv_num_sink_tokens=2)
    config.update(kwargs)
    return CevahirNeuralNetwork(**config)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('sdpa', [False, True])
@pytest.mark.parametrize('kv_heads', [1, 2, 4])
@pytest.mark.parametrize('window', [None, 3])
def test_cached_matches_full(sdpa, kv_heads, window):
    model = tiny(use_pytorch_sdpa=sdpa, num_kv_heads=kv_heads,
                 sliding_window=window).eval()
    ids = torch.randint(0, 41, (2, 9))
    with torch.no_grad():
        expected = model(ids)[0]
        actual = []
        for start, end in [(0, 3), (3, 5), (5, 6), (6, 9)]:
            actual.append(model(ids[:, start:end], use_cache=True,
                                cache_position=torch.arange(start, end))[0])
    torch.testing.assert_close(torch.cat(actual, 1), expected, atol=2e-6, rtol=2e-5)


def test_eviction_preserves_sinks_and_absolute_positions():
    cache = KVCache(1, 1, 2, max_cache_len=6, num_sink_tokens=2)
    for index in range(12):
        values = torch.full((1, 1, 1, 2), float(index))
        cache.update(values, values, torch.tensor([index]))
    assert cache.seen_tokens == 12
    assert len(cache) == 6
    assert cache.get()[0][0, 0, :, 0].tolist() == [0, 1, 8, 9, 10, 11]


def test_cache_append_avoids_sorting_and_preserves_replacement(monkeypatch):
    cache = KVCache(1, 1, 2, max_cache_len=6, num_sink_tokens=2)
    with torch.no_grad():
        with monkeypatch.context() as scoped:
            def forbidden(*args, **kwargs):
                raise AssertionError("append must not sort retained context")
            scoped.setattr(torch, "argsort", forbidden)
            scoped.setattr(torch, "isin", forbidden)
            for index in range(3):
                value = torch.full((1, 1, 1, 2), float(index))
                cache.update(value, value, torch.tensor([index * 2]))
        replacement = torch.full((1, 1, 1, 2), 99.)
        cache.update(replacement, replacement, torch.tensor([2]))
    assert cache.positions.tolist() == [0, 2, 4]
    assert cache.get()[0][0, 0, :, 0].tolist() == [0, 99, 2]


def test_requested_attention_weights_match_sdpa_output_without_changing_backend():
    attention = tiny(use_pytorch_sdpa=True).eval().layers[0].attn
    x = torch.randn(2, 5, 16)
    blocked = torch.zeros(2, 1, 1, 5, dtype=torch.bool)
    blocked[0, :, :, 2] = True
    expected = attention(x, mask=blocked, causal_mask=True)
    actual, weights = attention(x, mask=blocked, causal_mask=True, return_attention_weights=True)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert weights.shape == (2, 4, 5, 5)
    assert torch.count_nonzero(weights.triu(1)) == 0
    assert torch.count_nonzero(weights[0, :, :, 2]) == 0
    assert attention.use_pytorch_sdpa is True


@pytest.mark.parametrize('sdpa', [False, True])
def test_causal_prefix_and_masked_rows(sdpa):
    model = tiny(use_pytorch_sdpa=sdpa).eval()
    ids = torch.randint(0, 41, (1, 6))
    changed = ids.clone()
    changed[:, 3:] = (changed[:, 3:] + 7) % 41
    torch.testing.assert_close(model(ids)[0][:, :3], model(changed)[0][:, :3])
    attn = model.layers[0].attn
    x = torch.randn(1, 3, 16, requires_grad=True)
    output = attn(x, mask=torch.ones(3, 3, dtype=torch.bool), causal_mask=True)
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_softcap_effective_with_default_backend():
    model = tiny(attn_logit_cap=0.1).eval()
    manual = tiny(attn_logit_cap=0.1, use_pytorch_sdpa=False).eval()
    manual.load_state_dict(model.state_dict())
    ids = torch.randint(0, 41, (1, 6))
    torch.testing.assert_close(model(ids)[0], manual(ids)[0], atol=2e-6, rtol=2e-5)


def test_state_dict_roundtrip_and_gradients():
    model = tiny(use_qk_norm=True)
    clone = tiny(use_qk_norm=True)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    ids = torch.randint(0, 41, (2, 5))
    output = model(ids)[0]
    torch.testing.assert_close(output, clone(ids)[0])
    output.square().mean().backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


@pytest.mark.parametrize('mode', ['rope', 'sinusoidal', 'learned'])
def test_append_cache_position_encoding(mode):
    model = tiny(pe_mode=mode).eval()
    ids = torch.randint(0, 41, (1, 8))
    with torch.no_grad():
        expected = model(ids)[0]
        actual = torch.cat([model(ids[:, i:i+1], use_cache=True)[0] for i in range(8)], dim=1)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('sdpa', [False, True])
def test_padding_masks_chunked_and_eviction(sdpa):
    model = tiny(use_pytorch_sdpa=sdpa, max_cache_len=6, sliding_window=3).eval()
    ids = torch.randint(0, 41, (2, 11))
    blocked = torch.tensor([[True, False, True, False, False, False, False, False, False, False, False],
                            [False, False, False, True, False, False, False, False, False, False, False]])
    with torch.no_grad():
        expected = model(ids, mask=blocked[:, None, :].expand(2, 11, 11))[0]
        actual = []
        for i in range(11):
            actual.append(model(ids[:, i:i+1], use_cache=True,
                                mask=blocked[:, None, :i+1])[0])
    torch.testing.assert_close(torch.cat(actual, 1), expected, atol=3e-6, rtol=3e-5)


def test_cache_reset_batch_dtype_and_oversized_prefill():
    model = tiny(max_cache_len=6).eval()
    with torch.no_grad():
        ids = torch.randint(0, 41, (1, 12))
        torch.testing.assert_close(model(ids, use_cache=True)[0], model(ids)[0])
        assert len(model.layers[0].attn.kv_cache) == 6
        ids2 = torch.randint(0, 41, (2, 4))
        torch.testing.assert_close(model(ids2, use_cache=True)[0], model(ids2)[0])
        assert model.layers[0].attn.kv_cache.seen_tokens == 4
        model.double()
        assert model.layers[0].attn.kv_cache is None
        torch.testing.assert_close(model(ids2, use_cache=True)[0], model(ids2)[0])
        assert model.layers[0].attn.kv_cache.dtype == torch.float64
        model.clear_kv_cache()
        assert model.layers[0].attn.kv_cache.seen_tokens == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_external_flash_failure_is_causal(monkeypatch, dtype):
    model = tiny(use_pytorch_sdpa=False).to(dtype).eval()
    attention = model.layers[0].attn
    module = __import__(attention.__class__.__module__, fromlist=['flash_attn_func'])
    def fail(*args, **kwargs):
        raise RuntimeError('forced backend failure')
    monkeypatch.setattr(module, 'flash_attn_func', fail)
    attention.use_flash_attention = True
    x = torch.randn(1, 5, 16, dtype=dtype)
    changed = x.clone()
    changed[:, 3:] += 5
    actual = attention(x, causal_mask=True)
    torch.testing.assert_close(actual[:, :3], attention(changed, causal_mask=True)[:, :3])
    attention.use_flash_attention = False
    torch.testing.assert_close(actual, attention(x, causal_mask=True))


@pytest.mark.parametrize('advanced', [False, True])
def test_moe_checkpoint_auxiliary_loss_once_and_gradient_parity(advanced):
    ordinary = tiny(use_moe=True, num_experts=3, moe_top_k=2,
                    moe_jitter_noise=0.0)
    checkpointed = tiny(use_moe=True, num_experts=3, moe_top_k=2,
                        moe_jitter_noise=0.0, use_gradient_checkpointing=True,
                        use_advanced_checkpointing=advanced)
    checkpointed.load_state_dict(ordinary.state_dict())
    ids = torch.randint(0, 41, (2, 5))
    for model in (ordinary, checkpointed):
        logits = model(ids)[0]
        aux = model.get_and_reset_moe_loss()
        assert aux is not None and aux.requires_grad
        assert model.get_and_reset_moe_loss() is None
        (logits.square().mean() + aux).backward()
        assert model.get_and_reset_moe_loss() is None
        router_grad = model.layers[0].ffn.router.router.weight.grad
        assert torch.isfinite(router_grad).all() and router_grad.norm() > 0
    for (_, first), (_, second) in zip(ordinary.named_parameters(), checkpointed.named_parameters()):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, atol=3e-6, rtol=3e-5)


def test_dynamic_quantization_is_inplace_and_reloadable(tmp_path):
    model = tiny(quantization_type='int8_dynamic').eval()
    assert model.apply_quantization() is model
    assert model.get_quantization_info()['is_quantized']
    ids = torch.randint(0, 41, (1, 5))
    logits = model(ids)[0]
    assert torch.isfinite(logits).all()
    path = tmp_path / 'quantized.pth'
    torch.save(model.state_dict(), path)
    clone = tiny(quantization_type='int8_dynamic').eval()
    clone.apply_quantization()
    clone.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(logits, clone(ids)[0])


def test_batched_rope_positions_are_independent():
    pe = tiny().pos_encoding
    x = torch.randn(2, 4, 3, 4)
    positions = torch.tensor([[0, 1, 2], [5, 6, 7]])
    actual = pe.apply_rotary_pos_emb(x, positions)
    for batch in range(2):
        torch.testing.assert_close(actual[batch:batch+1],
                                   pe.apply_rotary_pos_emb(x[batch:batch+1], positions[batch]))
