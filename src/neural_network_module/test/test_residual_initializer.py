import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.residual_manager_module.residual_initializer import ResidualInitializer

@pytest.fixture
def residual_initializer():
    return ResidualInitializer(log_level=logging.DEBUG)

def test_initialize(residual_initializer):
    tensor = torch.randn(100, 64)
    initialized_tensor = residual_initializer.initialize(tensor)

    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_initialize_invalid_input(residual_initializer):
    with pytest.raises(TypeError):
        residual_initializer.initialize("invalid_input")

def test_log_initialization(residual_initializer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    initialized_tensor = residual_initializer.initialize(tensor)
    assert "Residual initialization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Initialized tensor shape: {initialized_tensor.shape}" in caplog.text