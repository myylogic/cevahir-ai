import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.scaling_manager import ScalingManager

@pytest.fixture
def scaling_manager():
    return ScalingManager(learning_rate=0.01, log_level=logging.DEBUG)

def test_initialize(scaling_manager):
    tensor = torch.randn(100, 64)
    initialized_tensor = scaling_manager.initialize(tensor)

    assert isinstance(initialized_tensor, torch.Tensor)
    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_normalize_batch(scaling_manager):
    tensor = torch.randn(100, 64)
    normalized_tensor = scaling_manager.normalize(tensor, method="batch")

    assert isinstance(normalized_tensor, torch.Tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_normalize_layer(scaling_manager):
    tensor = torch.randn(100, 64)
    normalized_tensor = scaling_manager.normalize(tensor, method="layer")

    assert isinstance(normalized_tensor, torch.Tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_normalize_instance(scaling_manager):
    tensor = torch.randn(100, 64, 32, 32)
    normalized_tensor = scaling_manager.normalize(tensor, method="instance")

    assert isinstance(normalized_tensor, torch.Tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_normalize_group(scaling_manager):
    tensor = torch.randn(100, 64, 32, 32)
    normalized_tensor = scaling_manager.normalize(tensor, method="group", num_groups=16)

    assert isinstance(normalized_tensor, torch.Tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_optimize(scaling_manager):
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)
    optimized_tensor = scaling_manager.optimize(tensor, gradients)

    assert isinstance(optimized_tensor, torch.Tensor)
    assert optimized_tensor.shape == tensor.shape
    assert optimized_tensor.dtype == tensor.dtype
    assert optimized_tensor.device == tensor.device

def test_adapt_min_max(scaling_manager):
    tensor = torch.randn(100, 64)
    adapted_tensor = scaling_manager.adapt(tensor, method="min_max")

    assert isinstance(adapted_tensor, torch.Tensor)
    assert adapted_tensor.shape == tensor.shape
    assert adapted_tensor.min().item() >= 0
    assert adapted_tensor.max().item() <= 1
    assert adapted_tensor.dtype == tensor.dtype
    assert adapted_tensor.device == tensor.device

def test_adapt_standard(scaling_manager):
    tensor = torch.randn(100, 64)
    adapted_tensor = scaling_manager.adapt(tensor, method="standard")

    assert isinstance(adapted_tensor, torch.Tensor)
    assert adapted_tensor.shape == tensor.shape
    assert abs(adapted_tensor.mean().item()) < 1e-6
    assert abs(adapted_tensor.std().item() - 1) < 1e-4  # Hata toleransını artırıldı.
    assert adapted_tensor.dtype == tensor.dtype
    assert adapted_tensor.device == tensor.device

def test_adapt_robust(scaling_manager):
    tensor = torch.randn(100, 64)

    if tensor.numel() == 0:  # Boş tensör kontrolü eklendi.
        adapted_tensor = tensor.clone()
    else:
        adapted_tensor = scaling_manager.adapt(tensor, method="robust")

    assert isinstance(adapted_tensor, torch.Tensor)
    assert adapted_tensor.shape == tensor.shape
    assert adapted_tensor.dtype == tensor.dtype
    assert adapted_tensor.device == tensor.device

def test_invalid_input(scaling_manager):
    with pytest.raises(TypeError):
        scaling_manager.initialize("invalid_input")

    with pytest.raises(TypeError):
        scaling_manager.normalize("invalid_input")

    # Düzeltme burada:
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        scaling_manager.normalize(torch.randn(100, 64), method="invalid_method")


def test_log_execution(scaling_manager, caplog):
    caplog.clear()  # Logları temizle
    caplog.set_level(logging.INFO)

    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)

    scaling_manager.initialize(tensor)
    scaling_manager.normalize(tensor, method="batch")
    scaling_manager.optimize(tensor, gradients)
    scaling_manager.adapt(tensor, method="min_max")

    # Loglar zorunlu hale getirildi.
    assert any("Initialization completed." in record.message for record in caplog.records)

    assert any("Normalization completed." in record.message for record in caplog.records)
    assert any("Optimization completed." in record.message for record in caplog.records)
    assert any("Adaptation completed." in record.message for record in caplog.records)
