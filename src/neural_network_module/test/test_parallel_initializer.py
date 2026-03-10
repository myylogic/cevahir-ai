import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.parallel_execution_module.parallel_initializer import ParallelInitializer

@pytest.fixture
def parallel_initializer():
    return ParallelInitializer(num_tasks=4, task_dim=0, log_level=logging.DEBUG)

def test_initialize_tasks(parallel_initializer):
    tensor = torch.randn(100, 64)
    tasks = parallel_initializer.initialize_tasks(tensor)
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

def test_initialize_tasks_invalid_input(parallel_initializer):
    with pytest.raises(TypeError):
        parallel_initializer.initialize_tasks("invalid_input")

def test_log_initialization(parallel_initializer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    tasks = parallel_initializer.initialize_tasks(tensor)
    assert "Parallel initialization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    for i, task in enumerate(tasks):
        assert f"Task {i} tensor shape: {task.shape}" in caplog.text
        assert f"Task {i} tensor dtype: {task.dtype}" in caplog.text
        assert f"Task {i} tensor device: {task.device}" in caplog.text