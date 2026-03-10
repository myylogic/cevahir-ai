import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.memory_manager_module.memory_optimizer import MemoryOptimizer

@pytest.fixture
def optimizer():
    return MemoryOptimizer(log_level=logging.DEBUG)

def test_optimize_memory(optimizer):
    tensor = torch.randn(10, 20, 64)
    optimized_tensor = optimizer.optimize_memory(tensor)
    assert optimized_tensor.shape == tensor.shape
    assert optimized_tensor.dtype == tensor.dtype
    assert optimized_tensor.device == tensor.device

def test_optimize_memory_invalid_input(optimizer):
    with pytest.raises(TypeError):
        optimizer.optimize_memory("invalid_tensor")

def test_compact_memory(optimizer):
    optimizer.compact_memory()

def test_clear_cache(optimizer):
    optimizer.clear_cache()

def test_log_memory_usage(optimizer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(10, 20, 64)
    optimizer.log_memory_usage(tensor)
    assert "Memory usage logged." in caplog.text
    assert f"Tensor shape: {tensor.shape}" in caplog.text
    assert f"Tensor dtype: {tensor.dtype}" in caplog.text
    assert f"Tensor device: {tensor.device}" in caplog.text