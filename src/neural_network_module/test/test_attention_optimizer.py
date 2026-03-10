import pytest
import torch
from neural_network_module.ortak_katman_module.attention_manager_module.attention_optimizer import AttentionOptimizer,ScalingMethod

# Pytest fixture
@pytest.fixture
def optimizer():
    """AttentionOptimizer örneği sağlar."""
    return AttentionOptimizer(epsilon=1e-9, verbose=True, default_scaling_method="softmax", default_clipping_value=10.0)

@pytest.fixture
def optimizer_verbose():
    """Verbose modu aktif olan AttentionOptimizer örneği sağlar."""
    return AttentionOptimizer(epsilon=1e-9, verbose=True)

def test_init(optimizer):
    """__init__ metodunu test eder."""
    assert optimizer.epsilon == 1e-9, "Epsilon değeri hatalı."
    assert optimizer.verbose is True, "Verbose parametresi hatalı."
    
    # ScalingMethod enum ile doğru şekilde karşılaştırma yapıyoruz.
    assert optimizer.default_scaling_method == ScalingMethod.SOFTMAX, (
        f"Scaling method beklenen değer: {ScalingMethod.SOFTMAX}, "
        f"bulunan değer: {optimizer.default_scaling_method}"
    )

    assert optimizer.default_clipping_value == 10.0, "Varsayılan clipping değeri hatalı."


@pytest.mark.parametrize("method", ["softmax", "sigmoid", "zscore"])
def test_normalize_attention_methods(optimizer, method):
    """normalize_attention metodunu farklı yöntemlerle test eder."""
    tensor = torch.randn(2, 4, 10, 10)
    normalized = optimizer.normalize_attention(tensor, method=method)
    assert normalized.shape == tensor.shape

    if method == "softmax":
        assert torch.allclose(normalized.sum(dim=-1), torch.ones_like(normalized.sum(dim=-1)), atol=1e-6)
    elif method == "sigmoid":
        assert (normalized >= 0).all() and (normalized <= 1).all()
    elif method == "zscore":
        assert torch.allclose(normalized.mean(dim=-1), torch.zeros_like(normalized.mean(dim=-1)), atol=1e-6)

def test_scale_attention(optimizer):
    """scale_attention metodunu test eder."""
    scores = torch.randn(2, 4, 10, 10)
    scaled_scores = optimizer.scale_attention(scores, scaling_factor=2.0)
    assert scaled_scores.shape == scores.shape
    assert torch.allclose(scaled_scores, scores / 2.0)

def test_clip_attention(optimizer):
    """clip_attention metodunu test eder."""
    scores = torch.randn(2, 4, 10, 10) * 100
    clipped_scores = optimizer.clip_attention(scores, clip_value=50.0)
    assert clipped_scores.shape == scores.shape
    assert (clipped_scores <= 50.0).all() and (clipped_scores >= -50.0).all()


def test_large_tensor_performance(optimizer):
    """Büyük tensörlerle performans testleri."""
    large_tensor = torch.randn(100, 100, 100)
    try:
        optimizer.log_tensor_info(large_tensor, "Large Tensor")
    except Exception as e:
        pytest.fail(f"Büyük tensör testinde hata: {e}")

def test_validate_attention_scores(optimizer):
    """validate_attention_scores metodunu test eder."""
    valid_scores = torch.randn(2, 4, 10, 10)
    assert optimizer.validate_attention_scores(valid_scores)

    invalid_scores = torch.tensor([[float('inf'), float('-inf')], [float('nan'), 0.0]])
    assert not optimizer.validate_attention_scores(invalid_scores)





def test_check_for_nan_and_inf(optimizer):
    """
    check_for_nan ve check_for_inf metodlarını test eder.
    """
    tensor_with_nan = torch.tensor([1.0, float("nan"), 3.0])
    tensor_with_inf = torch.tensor([1.0, float("inf"), -float("inf")])
    valid_tensor = torch.rand(3, 3)

    # NaN kontrolü
    nan_exists, cleaned_tensor = optimizer.check_for_nan(tensor_with_nan, replace_with_zero=True)
    assert nan_exists is True
    assert torch.equal(cleaned_tensor, torch.tensor([1.0, 0.0, 3.0]))  # NaN sıfırla değiştirilmiş olmalı

    nan_exists, _ = optimizer.check_for_nan(valid_tensor, replace_with_zero=False)
    assert nan_exists is False

    # Sonsuzluk kontrolü
    inf_exists, cleaned_tensor = optimizer.check_for_inf(tensor_with_inf, replace_with_max=True, clip_value=1e9)
    assert inf_exists is True
    assert torch.equal(cleaned_tensor, torch.tensor([1.0, 1e9, -1e9]))  # Sonsuz değerler ±1e9 ile değiştirilmiş olmalı

    inf_exists, _ = optimizer.check_for_inf(valid_tensor, replace_with_max=False)
    assert inf_exists is False




def test_optimize_attention(optimizer):
    """optimize_attention metodunu test eder."""
    scores = torch.randn(2, 4, 10, 10)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[:, :, :, 5:] = 0

    optimized_scores = optimizer.optimize_attention(
        scores,
        attention_mask=mask,
        scaling_factor=2.0,
        clip_value=10.0,
        normalize_method="softmax",
        mask_type="default"
    )
    assert optimized_scores.shape == scores.shape
    assert torch.isinf(optimized_scores[:, :, :, 5:]).all()
    assert torch.allclose(
        optimized_scores.sum(dim=-1),
        torch.ones_like(optimized_scores.sum(dim=-1)),
        atol=1e-6
    )


@pytest.mark.parametrize("mask_type", ["default", "causal"])
def test_mask_attention_types(optimizer, mask_type):
    """mask_attention metodunu farklı maske türleriyle test eder."""
    scores = torch.randn(2, 4, 10, 10)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[:, :, :, 5:] = 0

    masked_scores = optimizer.mask_attention(scores, mask, mask_type=mask_type)
    assert masked_scores.shape == scores.shape

    if mask_type == "default":
        assert torch.isinf(masked_scores[:, :, :, 5:]).all()
    elif mask_type == "causal":
        seq_len = scores.size(-1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(scores.device)
        expected_scores = scores.masked_fill(causal_mask == 1, float('-inf'))
        assert torch.equal(masked_scores, expected_scores)

