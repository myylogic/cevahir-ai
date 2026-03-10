import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.residual_manager_module.residual_connection import ResidualConnection

@pytest.fixture
def residual_connection():
    return ResidualConnection(log_level=logging.DEBUG)

def test_apply(residual_connection):
    tensor1 = torch.randn(100, 64)
    tensor2 = torch.randn(100, 64)
    result_tensor = residual_connection.apply(tensor1, tensor2)

    assert result_tensor.shape == tensor1.shape
    assert result_tensor.dtype == tensor1.dtype
    assert result_tensor.device == tensor1.device

def test_apply_invalid_input(residual_connection):
    tensor1 = torch.randn(100, 64)
    tensor2 = "invalid_input"
    with pytest.raises(TypeError):
        residual_connection.apply(tensor1, tensor2)

    with pytest.raises(TypeError):
        residual_connection.apply("invalid_input", tensor1)

def test_apply_shape_mismatch(residual_connection):
    tensor1 = torch.randn(100, 64)
    tensor2 = torch.randn(64, 100)
    with pytest.raises(ValueError):
        residual_connection.apply(tensor1, tensor2)

def test_log_connection(residual_connection, caplog):
    caplog.set_level(logging.DEBUG)
    tensor1 = torch.randn(100, 64)
    tensor2 = torch.randn(100, 64)
    result_tensor = residual_connection.apply(tensor1, tensor2)
    assert "Residual connection applied." in caplog.text
    assert f"Tensor1 shape: {tensor1.shape}" in caplog.text
    assert f"Tensor2 shape: {tensor2.shape}" in caplog.text
    assert f"Result tensor shape: {result_tensor.shape}" in caplog.text