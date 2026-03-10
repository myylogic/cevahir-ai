import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.scaling_manager_module.scaling_initializer import ScalingInitializer

@pytest.fixture
def scaling_initializer():
    return ScalingInitializer(log_level=logging.DEBUG)

def test_initialize(scaling_initializer):
    tensor = torch.randn(100, 64)
    initialized_tensor = scaling_initializer.initialize(tensor)

    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_initialize_invalid_input(scaling_initializer):
    with pytest.raises(TypeError):
        scaling_initializer.initialize("invalid_input")

def test_log_initialization(scaling_initializer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    initialized_tensor = scaling_initializer.initialize(tensor)
    assert "Initialization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Initialized tensor shape: {initialized_tensor.shape}" in caplog.text