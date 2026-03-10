import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.quantum_adapter_module.quantum_utils_module.quantum_normalizer import QuantumNormalizer

@pytest.fixture
def quantum_normalizer():
    return QuantumNormalizer(log_level=logging.DEBUG)

def test_normalize_min_max(quantum_normalizer):
    tensor = torch.randn(100, 64)
    normalized_tensor = quantum_normalizer.normalize_min_max(tensor)

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.min() >= 0
    assert normalized_tensor.max() <= 1
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_normalize_standard(quantum_normalizer):
    tensor = torch.randn(100, 64)
    normalized_tensor = quantum_normalizer.normalize_standard(tensor)

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.mean().abs() < 1e-6
    assert normalized_tensor.std().abs() - 1 < 1e-6
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_normalize_robust(quantum_normalizer):
    tensor = torch.randn(100, 64)
    normalized_tensor = quantum_normalizer.normalize_robust(tensor)

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_invalid_input(quantum_normalizer):
    with pytest.raises(TypeError):
        quantum_normalizer.normalize_min_max("invalid_input")

    with pytest.raises(TypeError):
        quantum_normalizer.normalize_standard("invalid_input")

    with pytest.raises(TypeError):
        quantum_normalizer.normalize_robust("invalid_input")

def test_log_normalization(quantum_normalizer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    normalized_tensor = quantum_normalizer.normalize_min_max(tensor)
    assert "Min-Max Normalization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Normalized tensor shape: {normalized_tensor.shape}" in caplog.text