import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.memory_manager_module.memory_utils_module.memory_normalizer import MemoryNormalizer

@pytest.fixture
def normalizer():
    return MemoryNormalizer(normalization_type="layer_norm", log_level=logging.DEBUG)

def test_normalize_memory_layer_norm(normalizer):
    tensor = torch.randn(10, 20, 64)
    normalized_tensor = normalizer.normalize_memory(tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device



def test_normalize_memory_instance_norm():
    normalizer = MemoryNormalizer(normalization_type="instance_norm", log_level=logging.DEBUG)
    tensor = torch.randn(10, 20, 64)
    normalized_tensor = normalizer.normalize_memory(tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device



def test_normalize_memory_invalid_input(normalizer):
    with pytest.raises(TypeError):
        normalizer.normalize_memory("invalid_tensor")

def test_normalize_memory_invalid_type():
    with pytest.raises(ValueError):
        normalizer = MemoryNormalizer(normalization_type="invalid", log_level=logging.DEBUG)
        tensor = torch.randn(10, 20, 64)
        normalizer.normalize_memory(tensor)

def test_log_normalization(normalizer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(10, 20, 64)
    normalized_tensor = normalizer.normalize_memory(tensor)
    assert "Memory normalization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Normalized tensor shape: {normalized_tensor.shape}" in caplog.text
    assert f"Original tensor dtype: {tensor.dtype}" in caplog.text
    assert f"Normalized tensor dtype: {normalized_tensor.dtype}" in caplog.text
    assert f"Original tensor device: {tensor.device}" in caplog.text
    assert f"Normalized tensor device: {normalized_tensor.device}" in caplog.text

def test_normalize_memory_batch_norm():
    normalizer = MemoryNormalizer(normalization_type="batch_norm", log_level=logging.DEBUG)
    tensor = torch.randn(10, 20, 64)
    normalized_tensor = normalizer.normalize_memory(tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_normalize_memory_group_norm():
    normalizer = MemoryNormalizer(normalization_type="group_norm", num_groups=4, log_level=logging.DEBUG)
    tensor = torch.randn(10, 20, 64)
    normalized_tensor = normalizer.normalize_memory(tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device