import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.quantum_adapter_module.quantum_initializer import QuantumInitializer

@pytest.fixture
def quantum_initializer():
    return QuantumInitializer(log_level=logging.DEBUG)

def test_initialize(quantum_initializer):
    tensor = torch.randn(100, 64)
    initialized_tensor = quantum_initializer.initialize(tensor)

    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_initialize_invalid_input(quantum_initializer):
    with pytest.raises(TypeError):
        quantum_initializer.initialize("invalid_input")

def test_log_initialization(quantum_initializer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    initialized_tensor = quantum_initializer.initialize(tensor)
    assert "Quantum initialization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Initialized tensor shape: {initialized_tensor.shape}" in caplog.text