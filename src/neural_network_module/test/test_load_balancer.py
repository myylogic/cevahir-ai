import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.parallel_execution_module.parallel_utils_module.load_balancer import LoadBalancer

@pytest.fixture
def load_balancer():
    return LoadBalancer(num_tasks=4, task_dim=0, log_level=logging.DEBUG)

def test_balance_load(load_balancer):
    tensor = torch.randn(100, 64)
    tasks = load_balancer.balance_load(tensor)
    task_size = tensor.size(0) // 4

    assert len(tasks) == 4
    for i, task in enumerate(tasks):
        if i != 3:
            assert task.size(0) == task_size
        else:
            assert task.size(0) == tensor.size(0) - (3 * task_size)
        assert task.size(1) == tensor.size(1)
        assert task.dtype == tensor.dtype
        assert task.device == tensor.device

def test_balance_load_invalid_input(load_balancer):
    with pytest.raises(TypeError):
        load_balancer.balance_load("invalid_input")

def test_log_balancing(load_balancer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    tasks = load_balancer.balance_load(tensor)
    assert "Load balancing completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    for i, task in enumerate(tasks):
        assert f"Task {i} tensor shape: {task.shape}" in caplog.text
        assert f"Task {i} tensor dtype: {task.dtype}" in caplog.text
        assert f"Task {i} tensor device: {task.device}" in caplog.text