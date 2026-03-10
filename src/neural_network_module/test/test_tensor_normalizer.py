import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.tensor_adapter_module.tensor_utils_module.tensor_normalizer import TensorNormalizer

@pytest.fixture
def tensor_normalizer():
    return TensorNormalizer(log_level=logging.DEBUG)

def test_normalize_batch(tensor_normalizer):
    tensor = torch.randn(100, 64)
    normalized_tensor = tensor_normalizer.normalize_batch(tensor)

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_normalize_layer(tensor_normalizer):
    tensor = torch.randn(100, 64)
    normalized_tensor = tensor_normalizer.normalize_layer(tensor)

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_normalize_instance(tensor_normalizer):
    tensor = torch.randn(100, 64, 32, 32)
    normalized_tensor = tensor_normalizer.normalize_instance(tensor)

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_normalize_group(tensor_normalizer):
    tensor = torch.randn(100, 64, 32, 32)
    normalized_tensor = tensor_normalizer.normalize_group(tensor, num_groups=16)

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_invalid_input(tensor_normalizer):
    with pytest.raises(TypeError):
        tensor_normalizer.normalize_batch("invalid_input")

    with pytest.raises(TypeError):
        tensor_normalizer.normalize_layer("invalid_input")

    with pytest.raises(TypeError):
        tensor_normalizer.normalize_instance("invalid_input")

    with pytest.raises(TypeError):
        tensor_normalizer.normalize_group("invalid_input", num_groups=16)

def test_log_normalization(tensor_normalizer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    normalized_tensor = tensor_normalizer.normalize_batch(tensor)
    assert "Batch Normalization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Normalized tensor shape: {normalized_tensor.shape}" in caplog.text