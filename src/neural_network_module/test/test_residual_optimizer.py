import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.residual_manager_module.residual_optimizer import ResidualOptimizer

@pytest.fixture
def residual_optimizer():
    return ResidualOptimizer(learning_rate=0.01, log_level=logging.DEBUG)

def test_optimize(residual_optimizer):
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)
    optimized_tensor = residual_optimizer.optimize(tensor, gradients)

    assert optimized_tensor.shape == tensor.shape
    assert optimized_tensor.dtype == tensor.dtype
    assert optimized_tensor.device == tensor.device

def test_optimize_invalid_input(residual_optimizer):
    tensor = torch.randn(100, 64)
    gradients = "invalid_input"
    with pytest.raises(TypeError):
        residual_optimizer.optimize(tensor, gradients)

    with pytest.raises(TypeError):
        residual_optimizer.optimize("invalid_input", tensor)

def test_log_optimization(residual_optimizer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)
    optimized_tensor = residual_optimizer.optimize(tensor, gradients)
    assert "Optimization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Gradients shape: {gradients.shape}" in caplog.text
    assert f"Optimized tensor shape: {optimized_tensor.shape}" in caplog.text