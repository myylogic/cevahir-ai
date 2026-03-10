import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.quantum_adapter_module.quantum_scaler import QuantumScaler

@pytest.fixture
def quantum_scaler():
    return QuantumScaler(scale_range=(0, 1), log_level=logging.DEBUG)

def test_scale_min_max(quantum_scaler):
    tensor = torch.randn(100, 64)
    scaled_tensor = quantum_scaler.scale_min_max(tensor)

    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.min() >= 0
    assert scaled_tensor.max() <= 1
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_scale_standard(quantum_scaler):
    tensor = torch.randn(100, 64)
    scaled_tensor = quantum_scaler.scale_standard(tensor)

    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.mean().abs() < 1e-6
    assert scaled_tensor.std().abs() - 1 < 1e-6
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_scale_robust(quantum_scaler):
    tensor = torch.randn(100, 64)
    scaled_tensor = quantum_scaler.scale_robust(tensor)

    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_invalid_input(quantum_scaler):
    with pytest.raises(TypeError):
        quantum_scaler.scale_min_max("invalid_input")

    with pytest.raises(TypeError):
        quantum_scaler.scale_standard("invalid_input")

    with pytest.raises(TypeError):
        quantum_scaler.scale_robust("invalid_input")

def test_log_scaling(quantum_scaler, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    scaled_tensor = quantum_scaler.scale_min_max(tensor)
    assert "Min-Max Scaling completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Scaled tensor shape: {scaled_tensor.shape}" in caplog.text