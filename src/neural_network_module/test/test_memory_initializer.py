import pytest
import torch
from neural_network_module.ortak_katman_module.memory_manager_module.memory_utils_module.memory_initializer import MemoryInitializer
import logging
@pytest.fixture
def initializer():
    return MemoryInitializer(init_type="xavier", log_level=logging.DEBUG)

def test_initialize_memory_xavier(initializer):
    tensor = torch.empty(10, 20, 64)
    initialized_tensor = initializer.initialize_memory(tensor)
    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_initialize_memory_he():
    initializer = MemoryInitializer(init_type="he", log_level=logging.DEBUG)
    tensor = torch.empty(10, 20, 64)
    initialized_tensor = initializer.initialize_memory(tensor)
    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_initialize_memory_normal():
    initializer = MemoryInitializer(init_type="normal", log_level=logging.DEBUG)
    tensor = torch.empty(10, 20, 64)
    initialized_tensor = initializer.initialize_memory(tensor)
    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_initialize_memory_invalid_input(initializer):
    with pytest.raises(TypeError):
        initializer.initialize_memory("invalid_tensor")

def test_initialize_memory_invalid_type():
    with pytest.raises(ValueError):
        initializer = MemoryInitializer(init_type="invalid", log_level=logging.DEBUG)
        tensor = torch.empty(10, 20, 64)
        initializer.initialize_memory(tensor)

def test_validate_initialization_invalid_input(initializer):
    with pytest.raises(TypeError):
        initializer.validate_initialization("invalid_tensor")

def test_validate_initialization_invalid_dimensions(initializer):
    tensor = torch.empty(1)
    with pytest.raises(ValueError):
        initializer.validate_initialization(tensor)

def test_log_initialization(initializer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.empty(10, 20, 64)
    initializer.log_initialization(tensor)
    assert "Memory initialized." in caplog.text
    assert f"Tensor shape: {tensor.shape}" in caplog.text
    assert f"Tensor dtype: {tensor.dtype}" in caplog.text
    assert f"Tensor device: {tensor.device}" in caplog.text