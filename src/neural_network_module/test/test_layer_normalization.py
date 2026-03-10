import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.normalization_manager_module.layer_normalization import LayerNormalization

@pytest.fixture
def layer_normalization():
    return LayerNormalization(normalized_shape=64, log_level=logging.DEBUG)

def test_apply(layer_normalization):
    tensor = torch.randn(10, 64)
    normalized_tensor = layer_normalization.apply(tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_apply_invalid_input(layer_normalization):
    with pytest.raises(TypeError):
        layer_normalization.apply("invalid_input")

def test_log_normalization(layer_normalization, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(10, 64)
    normalized_tensor = layer_normalization.apply(tensor)
    assert "Layer normalization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Normalized tensor shape: {normalized_tensor.shape}" in caplog.text
    assert f"Original tensor dtype: {tensor.dtype}" in caplog.text
    assert f"Normalized tensor dtype: {normalized_tensor.dtype}" in caplog.text
    assert f"Original tensor device: {tensor.device}" in caplog.text
    assert f"Normalized tensor device: {normalized_tensor.device}" in caplog.text

def test_layer_normalization_parameters():
    layer_normalization = LayerNormalization(normalized_shape=128, eps=1e-4, elementwise_affine=False, log_level=logging.DEBUG)
    assert layer_normalization.normalized_shape == 128
    assert layer_normalization.eps == 1e-4
    assert not layer_normalization.elementwise_affine

def test_layer_normalization_with_different_shapes():
    layer_normalization = LayerNormalization(normalized_shape=(32, 32), log_level=logging.DEBUG)
    tensor = torch.randn(10, 32, 32)
    normalized_tensor = layer_normalization.apply(tensor)
    assert normalized_tensor.shape == tensor.shape