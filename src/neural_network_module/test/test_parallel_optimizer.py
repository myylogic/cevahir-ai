import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.parallel_execution_module.parallel_utils_module.parallel_optimizer import ParallelOptimizer

@pytest.fixture
def parallel_optimizer():
    return ParallelOptimizer(num_tasks=4, task_dim=0, learning_rate=0.01, log_level=logging.DEBUG)

def test_optimize_tasks(parallel_optimizer):
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)
    optimized_tensor = parallel_optimizer.optimize_tasks(tensor, gradients)
    task_size = tensor.size(0) // 4

    assert optimized_tensor.shape == tensor.shape
    assert optimized_tensor.dtype == tensor.dtype
    assert optimized_tensor.device == tensor.device

def test_optimize_tasks_invalid_input(parallel_optimizer):
    tensor = torch.randn(100, 64)
    gradients = "invalid_input"
    with pytest.raises(TypeError):
        parallel_optimizer.optimize_tasks(tensor, gradients)

def test_log_optimization(parallel_optimizer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)
    optimized_tensor = parallel_optimizer.optimize_tasks(tensor, gradients)
    assert "Optimization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Gradients shape: {gradients.shape}" in caplog.text
    assert f"Optimized tensor shape: {optimized_tensor.shape}" in caplog.text