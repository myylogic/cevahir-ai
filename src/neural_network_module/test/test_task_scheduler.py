import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.parallel_execution_module.parallel_utils_module.task_scheduler import TaskScheduler

@pytest.fixture
def task_scheduler():
    return TaskScheduler(num_tasks=4, task_dim=0, log_level=logging.DEBUG)

def test_schedule_tasks(task_scheduler):
    tensor = torch.randn(100, 64)
    tasks = task_scheduler.schedule_tasks(tensor)
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

def test_schedule_tasks_invalid_input(task_scheduler):
    with pytest.raises(TypeError):
        task_scheduler.schedule_tasks("invalid_input")

def test_log_scheduling(task_scheduler, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    tasks = task_scheduler.schedule_tasks(tensor)
    assert "Task scheduling completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    for i, task in enumerate(tasks):
        assert f"Task {i} tensor shape: {task.shape}" in caplog.text
        assert f"Task {i} tensor dtype: {task.dtype}" in caplog.text
        assert f"Task {i} tensor device: {task.device}" in caplog.text