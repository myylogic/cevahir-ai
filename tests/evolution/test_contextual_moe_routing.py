"""Small CPU contracts for opt-in routing; only bounded miniature models are used."""
import copy
import logging
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.neural_network import CevahirNeuralNetwork
from src.neural_network_module.ortak_katman_module.mixture_of_experts import MixtureOfExperts, Router
from src.neural_network_module.ortak_katman_module.transformer_encoder_layer import TransformerEncoderLayer
from model_management.model_manager import ModelManager
from model.cevahir import CevahirModelAPI, CevahirProcessingError
from cognitive_management.cognitive_types import DecodingConfig


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        yield
    torch.set_num_threads(previous)


def make_moe():
    return MixtureOfExperts(embed_dim=8, ffn_dim=16, num_experts=3, top_k=2,
        dropout=0., jitter_noise=0., log_level=50)


def test_missing_and_zero_prior_leave_router_values_and_state_keys_unchanged():
    router = Router(8, 3, top_k=2, jitter_noise=0., log_level=50).eval()
    x = torch.randn(2, 3, 8)
    keys = tuple(router.state_dict())
    baseline = router(x)
    for candidate in (router(x, None), router(x, torch.zeros(2, 3))):
        assert all(torch.equal(a, b) for a, b in zip(baseline, candidate))
    assert tuple(router.state_dict()) == keys


def test_each_batch_row_has_its_own_prior_and_prior_receives_gradients():
    router = Router(8, 3, top_k=2, jitter_noise=0., log_level=50).eval()
    nn.init.zeros_(router.router.weight)
    prior = torch.tensor([[.8, -.3, -.6], [-.7, .9, .2]], requires_grad=True)
    weights, indices, logits = router(torch.zeros(2, 3, 8), prior)
    assert indices[0, :, 0].tolist() == [0, 0, 0]
    assert indices[1, :, 0].tolist() == [1, 1, 1]
    weights[..., 0].sum().backward()
    assert prior.grad is not None and torch.isfinite(prior.grad).all()
    assert prior.grad.abs().sum() > 0
    torch.testing.assert_close(logits, prior.detach()[:, None, :].expand(2, 3, 3))


@pytest.mark.parametrize('bad', [torch.zeros(3), torch.zeros(2, 2),
    torch.zeros(2, 3, dtype=torch.long), torch.full((2, 3), float('nan')),
    torch.full((2, 3), 1.01)])
def test_invalid_prior_is_rejected(bad):
    with pytest.raises((ValueError, TypeError), match='routing_bias'):
        Router(8, 3, log_level=50)(torch.zeros(2, 3, 8), bad)


def test_auxiliary_loss_and_gradients_ignore_padding():
    moe = make_moe().eval()
    x = torch.randn(1, 3, 8, requires_grad=True)
    prior = torch.tensor([[.5, -.5, 0.]], requires_grad=True)
    _, reference = moe(x, routing_bias=prior)
    dx, dp = torch.autograd.grad(reference, (x, prior))
    padded = torch.cat((x.detach(), torch.randn(1, 2, 8)), dim=1).requires_grad_()
    prior2 = prior.detach().clone().requires_grad_()
    _, actual = moe(padded, routing_bias=prior2,
        valid_token_mask=torch.tensor([[True, True, True, False, False]]))
    px, pp = torch.autograd.grad(actual, (padded, prior2))
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(px[:, :3], dx)
    torch.testing.assert_close(pp, dp)
    assert torch.count_nonzero(px[:, 3:]) == 0


def test_fully_masked_auxiliary_loss_is_finite_differentiable_zero():
    moe = make_moe()
    x = torch.randn(2, 3, 8, requires_grad=True)
    prior = torch.zeros(2, 3, requires_grad=True)
    _, loss = moe(x, prior, torch.zeros(2, 3, dtype=torch.bool))
    assert loss.requires_grad and loss.item() == 0.0
    loss.backward()
    assert torch.equal(x.grad, torch.zeros_like(x))
    assert torch.equal(prior.grad, torch.zeros_like(prior))


@pytest.mark.parametrize('bad', [torch.ones(2, 3), torch.ones(2, 2, dtype=torch.bool)])
def test_token_mask_requires_explicit_boolean_batch_sequence_shape(bad):
    with pytest.raises((ValueError, TypeError), match='valid_token_mask'):
        make_moe()(torch.zeros(2, 3, 8), valid_token_mask=bad)


class ZeroAttention(nn.Module):
    def forward(self, q, k, v, **kwargs):
        return q * 0, None


def minimal_layer(*, checkpoint, parallel=False, pre_norm=True):
    # Exercise the real layer forward/checkpoint machinery without constructing attention.
    layer = TransformerEncoderLayer.__new__(TransformerEncoderLayer)
    nn.Module.__init__(layer)
    layer.attn = ZeroAttention()
    layer.ffn = make_moe()
    layer.norm1 = layer.norm2 = layer.dropout = nn.Identity()
    layer.use_moe = True
    layer.parallel_residual, layer.pre_norm = parallel, pre_norm
    layer.use_gradient_checkpointing = checkpoint
    layer.use_advanced_checkpointing = False
    layer.drop_path_rate = 0.
    layer._record_moe_aux = True
    return layer


@pytest.mark.parametrize('parallel,pre_norm', [(False, True), (False, False), (True, True)])
def test_checkpoint_recompute_preserves_prior_gradients_and_consumes_aux_once(parallel, pre_norm):
    baseline = minimal_layer(checkpoint=False, parallel=parallel, pre_norm=pre_norm)
    recomputed = copy.deepcopy(baseline)
    recomputed.use_gradient_checkpointing = True
    x = torch.randn(2, 3, 8)
    bias = torch.tensor([[.4, -.2, .1], [-.5, .3, .6]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    results = []
    for layer in (baseline, recomputed):
        inputs, prior = x.clone().requires_grad_(), bias.clone().requires_grad_()
        out, _ = layer(inputs, routing_bias=prior, valid_token_mask=mask)
        aux = layer.get_and_reset_moe_loss()
        assert aux is not None
        (out.square().mean() + aux).backward()
        assert layer.get_and_reset_moe_loss() is None
        results.append((out.detach(), inputs.grad, prior.grad))
    for left, right in zip(*results):
        torch.testing.assert_close(left, right)
    assert results[1][2].abs().sum() > 0


def cache_contract_host(moe=True):
    core = CevahirNeuralNetwork.__new__(CevahirNeuralNetwork)
    nn.Module.__init__(core)
    cache = SimpleNamespace(cache_len=0)
    cache.clear = lambda: setattr(cache, 'cache_len', 0)
    core.layers = [SimpleNamespace(use_moe=moe, ffn=SimpleNamespace(num_experts=3),
        attn=SimpleNamespace(kv_cache=cache))]
    core.training = False
    core.logger = logging.getLogger('routing-test')
    return core, cache


def test_cache_prior_is_content_owned_and_cannot_change_until_clear():
    core, cache = cache_contract_host()
    x, bias = torch.ones(1, 2, dtype=torch.long), torch.tensor([[.4, 0., -.4]])
    core._prepare_routing_bias(x, bias, use_cache=True)
    cache.cache_len = 2
    core._prepare_routing_bias(x, bias.clone(), use_cache=True)
    bias[0, 0] = .8
    with pytest.raises(ValueError, match='clear_kv_cache'):
        core._prepare_routing_bias(x, bias, use_cache=True)
    with pytest.raises(ValueError, match='clear_kv_cache'):
        core._prepare_routing_bias(x, None, use_cache=True)
    core.clear_kv_cache()
    core._prepare_routing_bias(x, bias, use_cache=True)


def test_prior_cannot_attach_to_an_existing_baseline_prefix_or_dense_model():
    core, cache = cache_contract_host()
    cache.cache_len = 2
    with pytest.raises(ValueError, match='clear_kv_cache'):
        core._prepare_routing_bias(torch.ones(1, 1), torch.zeros(1, 3), use_cache=True)
    dense, _ = cache_contract_host(moe=False)
    with pytest.raises(ValueError, match='use_moe'):
        dense._prepare_routing_bias(torch.ones(1, 1), torch.zeros(1, 3), use_cache=False)


class CapturingModel(nn.Module):
    def forward(self, inputs, *, mask=None, routing_bias=None, valid_token_mask=None):
        self.received = (mask, routing_bias, valid_token_mask)
        return torch.zeros(*inputs.shape, 8), None


def test_manager_preserves_prior_and_distinguishes_attention_from_validity_mask():
    manager = ModelManager.__new__(ModelManager)
    manager.model, manager._device, manager.config = CapturingModel(), torch.device('cpu'), {}
    bias = torch.zeros(2, 3, requires_grad=True)
    valid = torch.tensor([[True, False, True], [False, True, True]])
    manager.forward(torch.ones(2, 3, dtype=torch.long), routing_bias=bias, mask=valid)
    blocked, received, objective_mask = manager.model.received
    assert received is bias
    assert torch.equal(blocked[:, 0, :], ~valid)
    assert torch.equal(objective_mask, valid)


class TinyTokenizer:
    def encode(self, text, **kwargs):
        return ['pieces-are-not-ids'], [1, 2] if text != 'candidate' else [3]

    def get_vocab(self):
        return {'<EOS>': 7}

    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))


class RecordingManager:
    def __init__(self):
        self.model = SimpleNamespace(supports_routing_bias=True,
            layers=[SimpleNamespace(ffn=SimpleNamespace(num_experts=3))])
        self.calls, self.clears = [], 0
        self.config = {'use_kv_cache': True}

    def eval_mode(self):
        pass

    def clear_kv_cache(self):
        self.clears += 1

    def forward(self, inputs, **kwargs):
        self.calls.append((inputs.clone(), kwargs))
        logits = torch.zeros(*inputs.shape, 8)
        logits[..., 3] = 1.
        return logits, None


def facade_host():
    api = CevahirModelAPI.__new__(CevahirModelAPI)
    api.model_manager, api.tokenizer_core = RecordingManager(), TinyTokenizer()
    api._device, api._generation_lock = torch.device('cpu'), threading.RLock()
    return api


def test_generation_and_scoring_receive_same_request_prior_without_silent_fallback():
    api = facade_host()
    prior = [.5, -.2, .1]
    cfg = DecodingConfig(max_new_tokens=2, temperature=0., repetition_penalty=1.)
    api.generate('prompt', cfg, routing_bias=prior)
    api.score('prompt', 'candidate', routing_bias=prior)
    assert len(api.model_manager.calls) == 3 and api.model_manager.clears == 1
    for _, kwargs in api.model_manager.calls:
        torch.testing.assert_close(kwargs['routing_bias'], torch.tensor([prior]))
    assert api.model_manager.calls[-1][1].get('use_cache', False) is False
    api.model_manager.forward = lambda *a, **kw: (_ for _ in ()).throw(ValueError('bad forward'))
    with pytest.raises(CevahirProcessingError, match='routing_bias'):
        api.score('prompt', 'candidate', routing_bias=prior)


def test_beam_and_dense_prior_requests_fail_before_generation():
    api = facade_host()
    with pytest.raises(ValueError, match='beam'):
        api.generate('p', DecodingConfig(num_beams=2), routing_bias=[0., 0., 0.])
    api.model_manager.model.supports_routing_bias = False
    assert api.supports_routing_bias is False
    with pytest.raises(ValueError, match='MoE'):
        api.generate('p', DecodingConfig(), routing_bias=[0., 0., 0.])
    assert api.model_manager.calls == []


def test_entropy_uses_ids_and_reports_availability_without_claiming_calibration():
    api = facade_host()
    result = api.entropy_details('prompt')
    assert api.model_manager.calls[-1][0].tolist() == [[1, 2]]
    assert result['available'] is True and result['calibrated'] is False
    assert 0. < result['value'] < 1.
    api.model_manager.forward = lambda *a, **kw: (_ for _ in ()).throw(ValueError('unavailable'))
    missing = api.entropy_details('prompt')
    assert missing['available'] is False and missing['value'] == .5
    assert missing['error'] == 'ValueError'


def tiny_core():
    # Explicitly bounded integration specimen: two layers, width eight, vocab sixteen.
    return CevahirNeuralNetwork(learning_rate=.001, dropout=0., vocab_size=16,
        embed_dim=8, seq_proj_dim=8, num_heads=2, num_layers=2, ffn_dim=16,
        use_moe=True, num_experts=3, moe_top_k=2, moe_jitter_noise=0.,
        use_gradient_checkpointing=False, max_cache_len=8, pe_max_len=8,
        rope_original_max_len=8, log_level=50).eval()


def test_actual_core_prior_cache_matches_full_sequence_and_preserves_baseline():
    core = tiny_core()
    ids = torch.tensor([[1, 3, 5]])
    prior = torch.tensor([[.7, -.2, -.4]])
    keys = tuple(core.state_dict())
    with torch.no_grad():
        baseline = core(ids)[0]
        assert torch.equal(core(ids, routing_bias=torch.zeros(1, 3))[0], baseline)
        full = core(ids, routing_bias=prior)[0][:, -1:]
        core(ids[:, :2], use_cache=True, routing_bias=prior)
        cached = core(ids[:, 2:], use_cache=True, routing_bias=prior)[0]
        torch.testing.assert_close(cached, full, rtol=1e-4, atol=1e-6)
        with pytest.raises(ValueError, match='clear_kv_cache'):
            core(ids[:, :1], use_cache=True, routing_bias=-prior)
        core.clear_kv_cache()
        core(ids[:, :1], use_cache=True, routing_bias=-prior)
    assert tuple(core.state_dict()) == keys


def test_active_training_helper_uses_input_validity_and_keeps_duck_typed_models():
    from training_management.v2.core.training_loop import TrainingLoop
    inputs = torch.tensor([[1, 2, 0]])  # EOS=2 is valid even when its target is PAD.
    calls = []
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.pad_token_id = 0
    loop._supports_valid_token_mask = True
    loop.model = lambda x, **kw: calls.append(kw) or x
    assert loop._forward_inputs(inputs) is inputs
    assert calls[-1]['valid_token_mask'].tolist() == [[True, True, False]]
    loop._supports_valid_token_mask = False
    loop.model = lambda x: x
    assert loop._forward_inputs(inputs) is inputs
