import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.quantum_adapter_module.quantum_utils_module.quantum_projection import QuantumProjection

@pytest.fixture
def quantum_projection():
    return QuantumProjection(projection_dim=128, log_level=logging.DEBUG)

def test_project(quantum_projection):
    tensor = torch.randn(100, 64)
    projected_tensor = quantum_projection.project(tensor)

    assert projected_tensor.shape[0] == tensor.shape[0]
    assert projected_tensor.shape[1] == quantum_projection.projection_dim
    assert projected_tensor.dtype == tensor.dtype
    assert projected_tensor.device == tensor.device

def test_project_invalid_input(quantum_projection):
    with pytest.raises(TypeError):
        quantum_projection.project("invalid_input")

def test_log_projection(quantum_projection, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    projected_tensor = quantum_projection.project(tensor)
    assert "Projection completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Projected tensor shape: {projected_tensor.shape}" in caplog.text