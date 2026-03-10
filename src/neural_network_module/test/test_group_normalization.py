import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.normalization_manager_module.group_normalization import GroupNormalization

@pytest.fixture
def group_normalization():
    return GroupNormalization(num_groups=4, num_channels=64, log_level=logging.DEBUG)

def test_apply(group_normalization):
    tensor = torch.randn(10, 64, 32, 32)
    normalized_tensor = group_normalization.apply(tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_apply_invalid_input(group_normalization):
    with pytest.raises(TypeError):
        group_normalization.apply("invalid_input")

def test_log_normalization(group_normalization, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(10, 64, 32, 32)
    normalized_tensor = group_normalization.apply(tensor)
    assert "Group normalization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Normalized tensor shape: {normalized_tensor.shape}" in caplog.text
    assert f"Original tensor dtype: {tensor.dtype}" in caplog.text
    assert f"Normalized tensor dtype: {normalized_tensor.dtype}" in caplog.text
    assert f"Original tensor device: {tensor.device}" in caplog.text
    assert f"Normalized tensor device: {normalized_tensor.device}" in caplog.text

def test_group_normalization_parameters():
    group_normalization = GroupNormalization(num_groups=8, num_channels=128, eps=1e-4, affine=False, log_level=logging.DEBUG)
    assert group_normalization.num_groups == 8
    assert group_normalization.num_channels == 128
    assert group_normalization.eps == 1e-4
    assert not group_normalization.affine

def test_group_normalization_with_different_shapes():
    group_normalization = GroupNormalization(num_groups=4, num_channels=32, log_level=logging.DEBUG)
    tensor = torch.randn(10, 32, 32, 32)
    normalized_tensor = group_normalization.apply(tensor)
    assert normalized_tensor.shape == tensor.shape