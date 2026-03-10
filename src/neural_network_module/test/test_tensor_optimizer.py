import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.tensor_adapter_module.tensor_utils_module.tensor_optimizer import TensorOptimizer

@pytest.fixture
def tensor_optimizer():
    return TensorOptimizer(learning_rate=0.01, log_level=logging.DEBUG)

def test_optimize_sgd(tensor_optimizer):
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)
    optimized_tensor = tensor_optimizer.optimize_sgd(tensor, gradients)

    assert optimized_tensor.shape == tensor.shape
    assert optimized_tensor.dtype == tensor.dtype
    assert optimized_tensor.device == tensor.device

def test_optimize_adam(tensor_optimizer):
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)
    optimized_tensor = tensor_optimizer.optimize_adam(tensor, gradients, t=1)

    assert optimized_tensor.shape == tensor.shape
    assert optimized_tensor.dtype == tensor.dtype
    assert optimized_tensor.device == tensor.device

def test_invalid_input(tensor_optimizer):
    tensor = torch.randn(100, 64)
    gradients = "invalid_input"
    with pytest.raises(TypeError):
        tensor_optimizer.optimize_sgd(tensor, gradients)

    with pytest.raises(TypeError):
        tensor_optimizer.optimize_adam(tensor, gradients, t=1)

def test_log_optimization(tensor_optimizer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)

    tensor_optimizer.optimize_sgd(tensor, gradients)
    assert "SGD Optimization completed." in caplog.text

    tensor_optimizer.optimize_adam(tensor, gradients, t=1)
    assert "Adam Optimization completed." in caplog.text