import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.normalization_manager_module.instance_normalization import InstanceNormalization


@pytest.fixture
def instance_normalization():
    return InstanceNormalization(num_features=64, log_level=logging.DEBUG)

def test_apply(instance_normalization):
    tensor = torch.randn(10, 64, 32, 32)
    normalized_tensor = instance_normalization.apply(tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_apply_invalid_input(instance_normalization):
    with pytest.raises(TypeError):
        instance_normalization.apply("invalid_input")

def test_log_normalization(instance_normalization, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(10, 64, 32, 32)
    normalized_tensor = instance_normalization.apply(tensor)
    assert "Instance normalization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Normalized tensor shape: {normalized_tensor.shape}" in caplog.text
    assert f"Original tensor dtype: {tensor.dtype}" in caplog.text
    assert f"Normalized tensor dtype: {normalized_tensor.dtype}" in caplog.text
    assert f"Original tensor device: {tensor.device}" in caplog.text
    assert f"Normalized tensor device: {normalized_tensor.device}" in caplog.text

def test_instance_normalization_parameters():
    instance_normalization = InstanceNormalization(num_features=128, eps=1e-4, affine=False, log_level=logging.DEBUG)
    assert instance_normalization.num_features == 128
    assert instance_normalization.eps == 1e-4
    assert not instance_normalization.affine

def test_instance_normalization_with_different_shapes():
    instance_normalization = InstanceNormalization(num_features=32, log_level=logging.DEBUG)
    tensor = torch.randn(10, 32, 32, 32)
    normalized_tensor = instance_normalization.apply(tensor)
    assert normalized_tensor.shape == tensor.shape