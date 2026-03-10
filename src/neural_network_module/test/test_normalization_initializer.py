import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.normalization_manager_module.normalization_utils_module.normalization_initializer import NormalizationInitializer

@pytest.fixture
def normalization_initializer():
    return NormalizationInitializer(init_type="xavier", log_level=logging.DEBUG)

def test_initialize_weights_xavier(normalization_initializer):
    tensor = torch.empty(3, 5)
    initialized_tensor = normalization_initializer.initialize_weights(tensor)
    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_initialize_weights_he():
    initializer = NormalizationInitializer(init_type="he", log_level=logging.DEBUG)
    tensor = torch.empty(3, 5)
    initialized_tensor = initializer.initialize_weights(tensor)
    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_initialize_weights_normal():
    initializer = NormalizationInitializer(init_type="normal", log_level=logging.DEBUG)
    tensor = torch.empty(3, 5)
    initialized_tensor = initializer.initialize_weights(tensor)
    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_initialize_bias(normalization_initializer):
    tensor = torch.empty(5)
    initialized_tensor = normalization_initializer.initialize_bias(tensor)
    assert torch.all(initialized_tensor == 0)
    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_apply_invalid_input(normalization_initializer):
    with pytest.raises(TypeError):
        normalization_initializer.initialize_weights("invalid_input")

def test_log_initialization(normalization_initializer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.empty(3, 5)
    initialized_tensor = normalization_initializer.initialize_weights(tensor)
    assert "Weights Initialization completed." in caplog.text
    assert f"Initialized tensor shape: {tensor.shape}" in caplog.text
    assert f"Initialized tensor dtype: {tensor.dtype}" in caplog.text
    assert f"Initialized tensor device: {tensor.device}" in caplog.text