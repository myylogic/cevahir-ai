import pytest
import logging
import torch
from neural_network_module.ortak_katman_module.memory_manager_module.memory_utils_module.memory_scaler import MemoryScaler

@pytest.fixture
def scaler():
    return MemoryScaler(scaling_type="min_max", log_level=logging.DEBUG)

def test_scale_memory_min_max(scaler):
    tensor = torch.randn(10, 20, 64)
    scaled_tensor = scaler.scale_memory(tensor)
    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device
    assert scaled_tensor.min() >= 0
    assert scaled_tensor.max() <= 1

def test_scale_memory_standard():
    scaler = MemoryScaler(scaling_type="standard", log_level=logging.DEBUG)
    tensor = torch.randn(10, 20, 64)
    scaled_tensor = scaler.scale_memory(tensor)
    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device
    assert torch.isclose(scaled_tensor.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(scaled_tensor.std(), torch.tensor(1.0), atol=1e-5)

def test_scale_memory_robust():
    scaler = MemoryScaler(scaling_type="robust", log_level=logging.DEBUG)
    tensor = torch.randn(10, 20, 64)
    scaled_tensor = scaler.scale_memory(tensor)
    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_scale_memory_invalid_input(scaler):
    with pytest.raises(TypeError):
        scaler.scale_memory("invalid_tensor")

def test_scale_memory_invalid_type():
    with pytest.raises(ValueError):
        scaler = MemoryScaler(scaling_type="invalid", log_level=logging.DEBUG)
        tensor = torch.randn(10, 20, 64)
        scaler.scale_memory(tensor)

def test_log_scaling(scaler, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(10, 20, 64)
    scaled_tensor = scaler.scale_memory(tensor)
    assert "Memory scaling completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Scaled tensor shape: {scaled_tensor.shape}" in caplog.text
    assert f"Original tensor dtype: {tensor.dtype}" in caplog.text
    assert f"Scaled tensor dtype: {scaled_tensor.dtype}" in caplog.text
    assert f"Original tensor device: {tensor.device}" in caplog.text
    assert f"Scaled tensor device: {scaled_tensor.device}" in caplog.text