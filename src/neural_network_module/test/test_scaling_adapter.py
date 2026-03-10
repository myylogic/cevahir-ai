import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.scaling_manager_module.scaling_adapter import ScalingAdapter

@pytest.fixture
def scaling_adapter():
    return ScalingAdapter(log_level=logging.DEBUG)

def test_scale_min_max(scaling_adapter):
    tensor = torch.randn(100, 64)
    scaled_tensor = scaling_adapter.scale(tensor, method="min_max")

    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.min() >= 0
    assert scaled_tensor.max() <= 1
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_scale_standard(scaling_adapter):
    tensor = torch.randn(100, 64)
    scaled_tensor = scaling_adapter.scale(tensor, method="standard")

    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.mean().abs() < 1e-6
    assert scaled_tensor.std().abs() - 1 < 1e-6
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_scale_robust(scaling_adapter):
    tensor = torch.randn(100, 64)
    scaled_tensor = scaling_adapter.scale(tensor, method="robust")

    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_invalid_input(scaling_adapter):
    with pytest.raises(TypeError):
        scaling_adapter.scale("invalid_input")

    with pytest.raises(ValueError):
        scaling_adapter.scale(torch.randn(100, 64), method="invalid_method")

def test_log_scaling(scaling_adapter, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    scaled_tensor = scaling_adapter.scale(tensor, method="min_max")
    assert "Min-Max Scaling completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Scaled tensor shape: {scaled_tensor.shape}" in caplog.text