import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.parallel_execution_module.parallel_scaler import ParallelScaler

@pytest.fixture
def parallel_scaler():
    return ParallelScaler(num_tasks=4, task_dim=0, log_level=logging.DEBUG)

def test_scale_min_max(parallel_scaler):
    tensor = torch.randn(100, 64)
    scaled_tasks = parallel_scaler.scale_min_max(tensor)
    task_size = tensor.size(0) // 4

    assert len(scaled_tasks) == 4
    for i, task in enumerate(scaled_tasks):
        if i != 3:
            assert task.size(0) == task_size
        else:
            assert task.size(0) == tensor.size(0) - (3 * task_size)
        assert task.size(1) == tensor.size(1)
        assert task.min() >= 0
        assert task.max() <= 1
        assert task.dtype == tensor.dtype
        assert task.device == tensor.device

def test_scale_standard(parallel_scaler):
    tensor = torch.randn(100, 64)
    scaled_tasks = parallel_scaler.scale_standard(tensor)
    task_size = tensor.size(0) // 4

    assert len(scaled_tasks) == 4
    for i, task in enumerate(scaled_tasks):
        if i != 3:
            assert task.size(0) == task_size
        else:
            assert task.size(0) == tensor.size(0) - (3 * task_size)
        assert task.size(1) == tensor.size(1)
        assert task.mean().abs() < 1e-6
        assert task.std().abs() - 1 < 1e-6
        assert task.dtype == tensor.dtype
        assert task.device == tensor.device

def test_scale_robust(parallel_scaler):
    tensor = torch.randn(100, 64)
    scaled_tasks = parallel_scaler.scale_robust(tensor)
    task_size = tensor.size(0) // 4

    assert len(scaled_tasks) == 4
    for i, task in enumerate(scaled_tasks):
        if i != 3:
            assert task.size(0) == task_size
        else:
            assert task.size(0) == tensor.size(0) - (3 * task_size)
        assert task.size(1) == tensor.size(1)
        assert task.dtype == tensor.dtype
        assert task.device == tensor.device

def test_initialize_tasks_invalid_input(parallel_scaler):
    with pytest.raises(TypeError):
        parallel_scaler.scale_min_max("invalid_input")

def test_log_scaling(parallel_scaler, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    scaled_tasks = parallel_scaler.scale_min_max(tensor)
    assert "Min-Max Scaling completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    for i, task in enumerate(scaled_tasks):
        assert f"Task {i} tensor shape: {task.shape}" in caplog.text
        assert f"Task {i} tensor dtype: {task.dtype}" in caplog.text
        assert f"Task {i} tensor device: {task.device}" in caplog.text