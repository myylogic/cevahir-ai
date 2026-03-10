import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.normalization_manager_module.batch_normalization import BatchNormalization

@pytest.fixture
def batch_normalization():
    return BatchNormalization(num_features=64, log_level=logging.DEBUG)

def test_apply(batch_normalization):
    tensor = torch.randn(10, 64)
    normalized_tensor = batch_normalization.apply(tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_apply_invalid_input(batch_normalization):
    with pytest.raises(TypeError):
        batch_normalization.apply("invalid_input")

def test_log_normalization(batch_normalization, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(10, 64)
    normalized_tensor = batch_normalization.apply(tensor)
    assert "Batch normalization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Normalized tensor shape: {normalized_tensor.shape}" in caplog.text
    assert f"Original tensor dtype: {tensor.dtype}" in caplog.text
    assert f"Normalized tensor dtype: {normalized_tensor.dtype}" in caplog.text
    assert f"Original tensor device: {tensor.device}" in caplog.text
    assert f"Normalized tensor device: {normalized_tensor.device}" in caplog.text

def test_batch_normalization_parameters():
    batch_normalization = BatchNormalization(num_features=128, eps=1e-4, momentum=0.2, affine=False, track_running_stats=False, log_level=logging.DEBUG)
    assert batch_normalization.num_features == 128
    assert batch_normalization.eps == 1e-4
    assert batch_normalization.momentum == 0.2
    assert not batch_normalization.affine
    assert not batch_normalization.track_running_stats

def test_batch_normalization_with_different_shapes():
    batch_normalization = BatchNormalization(num_features=32, log_level=logging.DEBUG)
    tensor = torch.randn(10, 32)
    normalized_tensor = batch_normalization.apply(tensor)
    assert normalized_tensor.shape == tensor.shape