"""Cross-boundary checks for the actual decoder/FFN/MoE implementations."""
import pytest
import torch
from model_management.config_schema import ModelArchConfig
from src.neural_network import CevahirNeuralNetwork
from src.neural_network_module.ortak_katman_module.transformer_encoder_layer import TransformerEncoderLayer


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("width,explicit", [(16, None), (32, 48), (128, None)])
def test_schema_ffn_width_matches_dense_and_expert_weights(gated, width, explicit):
    cfg = ModelArchConfig(embed_dim=width, num_heads=4, ffn_dim=explicit, use_swiglu=gated)
    for moe in (False, True):
        model = CevahirNeuralNetwork(learning_rate=.001, dropout=0., vocab_size=32,
            embed_dim=width, seq_proj_dim=width, num_heads=4, num_layers=1,
            ffn_dim=explicit, use_swiglu=gated, use_moe=moe, num_experts=2,
            use_gradient_checkpointing=False, log_level=50)
        ffn = model.layers[0].ffn.experts[0] if moe else model.layers[0].ffn
        assert ffn.ffn_dim == cfg.effective_ffn_dim
        projection = ffn.gate_up_proj if gated else ffn.fc1
        assert projection.out_features == cfg.effective_ffn_dim * (2 if gated else 1)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_layer_bias_reaches_dense_or_all_experts_and_receives_gradients(moe, bias):
    layer = TransformerEncoderLayer(embed_dim=16, num_heads=4, ffn_dim=24,
        dropout=0., use_moe=moe, num_experts=2, moe_top_k=2,
        ffn_use_bias=bias, ffn_activation="swiglu", use_gradient_checkpointing=False, log_level=50)
    experts = layer.ffn.experts if moe else [layer.ffn]
    for expert in experts:
        assert (expert.gate_up_proj.bias is not None) == bias
        assert (expert.fc2.bias is not None) == bias
    output = layer(torch.randn(2, 3, 16))[0]
    output.square().mean().backward()
    if bias:
        assert all(expert.fc2.bias.grad is not None for expert in experts)


def test_parameter_estimate_accounts_for_gqa_and_gated_projection():
    base = dict(embed_dim=32, num_heads=4, num_layers=2, ffn_dim=48)
    mha = ModelArchConfig(**base)
    gqa = ModelArchConfig(**base, num_kv_heads=1)
    assert mha.parameter_count_estimate - gqa.parameter_count_estimate == 2 * 2 * 32 * 24
    plain = ModelArchConfig(**base, use_swiglu=False)
    assert mha.parameter_count_estimate - plain.parameter_count_estimate == 2 * 32 * 48


@pytest.mark.parametrize("pre_norm", [False, True])
def test_parallel_flag_freezes_only_the_actually_unused_norm(pre_norm):
    layer = TransformerEncoderLayer(embed_dim=16, num_heads=4, ffn_dim=24,
        dropout=0., pre_norm=pre_norm, parallel_residual=True,
        use_gradient_checkpointing=False, log_level=50)
    layer(torch.randn(2, 3, 16))[0].square().sum().backward()
    for parameter in layer.norm2.parameters():
        assert parameter.requires_grad == (not pre_norm)
        assert (parameter.grad is not None) == (not pre_norm)


@pytest.mark.parametrize("pre_norm,parallel", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("checkpointed", [False, True, "advanced"])
def test_model_sdpa_and_explicit_diagnostics_match_outputs_and_gradients(monkeypatch, pre_norm, parallel, checkpointed):
    import torch.nn.functional as F
    calls = []
    original = F.scaled_dot_product_attention
    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(F, "scaled_dot_product_attention", observed)
    torch.manual_seed(17)
    model = CevahirNeuralNetwork(learning_rate=.001, dropout=0., vocab_size=32,
        embed_dim=16, seq_proj_dim=16, num_heads=4, num_kv_heads=2,
        num_layers=2, ffn_dim=24, pre_norm=pre_norm, parallel_residual=parallel,
        use_gradient_checkpointing=checkpointed is True,
        use_advanced_checkpointing=checkpointed == "advanced",
        use_pytorch_sdpa=True, log_level=50)
    model.train()
    tokens = torch.tensor([[1, 2, 3], [3, 2, 1]])
    fast, weights = model(tokens)
    assert weights is None
    assert len(calls) == 2
    fast.square().mean().backward()
    fast_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    calls.clear()
    manual, weights = model(tokens, return_attention_weights=True)
    assert weights.shape == (2, 4, 3, 3)
    assert not calls
    torch.testing.assert_close(fast, manual, atol=2e-6, rtol=2e-5)
    manual.square().mean().backward()
    assert not calls
    for name, p in model.named_parameters():
        if name in fast_grads:
            torch.testing.assert_close(p.grad, fast_grads[name], atol=2e-6, rtol=2e-4)


def test_fast_cached_decode_matches_explicit_attention():
    model = CevahirNeuralNetwork(learning_rate=.001, dropout=0., vocab_size=32,
        embed_dim=16, seq_proj_dim=16, num_heads=4, num_layers=1,
        ffn_dim=24, use_gradient_checkpointing=False, log_level=50).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(tokens, return_attention_weights=True)[0]
        outputs = [model(tokens[:, i:i+1], use_cache=True) for i in range(4)]
    assert all(out[1] is None and len(out) == 3 for out in outputs)
    torch.testing.assert_close(torch.cat([out[0] for out in outputs], dim=1), expected,
        atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("strategy,indices", [("selective", [0,1,3,5]), ("layer_wise", [0,2,4]), ("adaptive", [0,2,4,5])])
def test_checkpoint_factory_and_model_select_same_layers(strategy, indices):
    from src.neural_network_module.ortak_katman_module.advanced_checkpointing import create_checkpointing_strategy
    policy = create_checkpointing_strategy(strategy, num_layers=6)
    assert [i for i in range(6) if policy.should_checkpoint(i, 6)] == indices
    assert not any(policy.should_checkpoint(i, 6, training=False) for i in range(6))
    model = CevahirNeuralNetwork(learning_rate=.001, dropout=0., vocab_size=16,
        embed_dim=16, seq_proj_dim=16, num_heads=4, num_layers=6,
        ffn_dim=24, use_advanced_checkpointing=True, checkpointing_strategy=strategy, log_level=50)
    assert [i for i, layer in enumerate(model.layers) if layer.advanced_checkpointing.should_checkpoint(layer.layer_idx, layer.total_layers)] == indices


def test_checkpoint_factory_rejects_invalid_settings_and_honors_overrides():
    from src.neural_network_module.ortak_katman_module.advanced_checkpointing import create_checkpointing_strategy
    for kwargs in [dict(strategy="typo"), dict(num_layers=0), dict(checkpoint_every_n=0), dict(checkpoint_layers=[-1]), dict(num_layers=2, checkpoint_layers=[2])]:
        with pytest.raises(ValueError):
            create_checkpointing_strategy(**kwargs)
    policy = create_checkpointing_strategy("layer_wise", num_layers=6, checkpoint_every_n=3)
    assert [i for i in range(6) if policy.should_checkpoint(i, 6)] == [0, 3]
    assert create_checkpointing_strategy("selective", checkpoint_layers=[]).checkpoint_layers == []
    with pytest.raises(ValueError, match="checkpointing"):
        ModelArchConfig(checkpointing_strategy="typo").validate()


def test_legacy_projection_does_not_constrain_attention_geometry():
    model = CevahirNeuralNetwork(learning_rate=.001, dropout=0., vocab_size=16,
        embed_dim=16, seq_proj_dim=7, num_heads=4, num_layers=1,
        ffn_dim=24, use_gradient_checkpointing=False, log_level=50)
    assert not model.tie_weights  # Preserve legacy checkpoint weight independence.
    assert model.output_layer.weight.shape == (16, 16)
    assert model(torch.tensor([[1, 2]]))[0].shape == (1, 2, 16)


def test_standalone_layer_selective_checkpoint_is_active():
    layer = TransformerEncoderLayer(embed_dim=16, num_heads=4, ffn_dim=24,
        dropout=0., use_advanced_checkpointing=True, log_level=50)
    assert layer.advanced_checkpointing.should_checkpoint(0, 1)
    x = torch.randn(1, 3, 16, requires_grad=True)
    layer(x)[0].square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
