import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.scaling_manager_module.scaling_normalizer import ScalingNormalizer

@pytest.fixture
def scaling_normalizer():
    return ScalingNormalizer(log_level=logging.DEBUG)

def test_batch_normalize(scaling_normalizer):
    tensor = torch.randn(100, 64)
    normalized_tensor = scaling_normalizer.normalize(tensor, method="batch")

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_layer_normalize(scaling_normalizer):
    tensor = torch.randn(100, 64)
    normalized_tensor = scaling_normalizer.normalize(tensor, method="layer")

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_instance_normalize(scaling_normalizer):
    tensor = torch.randn(100, 64, 32, 32)
    normalized_tensor = scaling_normalizer.normalize(tensor, method="instance")

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_group_normalize(scaling_normalizer):
    tensor = torch.randn(100, 64, 32, 32)
    normalized_tensor = scaling_normalizer.normalize(tensor, method="group", num_groups=16)

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_invalid_input(scaling_normalizer):
    with pytest.raises(TypeError):
        scaling_normalizer.normalize("invalid_input")

    with pytest.raises(ValueError):
        scaling_normalizer.normalize(torch.randn(100, 64), method="invalid_method")

def test_log_normalization(scaling_normalizer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    normalized_tensor = scaling_normalizer.normalize(tensor, method="batch")
    assert "Batch Normalization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Normalized tensor shape: {normalized_tensor.shape}" in caplog.text