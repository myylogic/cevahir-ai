import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.tensor_adapter_module.tensor_utils_module.tensor_initializer import TensorInitializer

@pytest.fixture
def tensor_initializer():
    return TensorInitializer(log_level=logging.DEBUG)

def test_initialize_zeros(tensor_initializer):
    shape = (100, 64)
    tensor = tensor_initializer.initialize_zeros(shape)

    assert tensor.shape == shape
    assert torch.all(tensor == 0)
    assert tensor.dtype == torch.float32
    assert tensor.device == torch.device('cpu')

def test_initialize_ones(tensor_initializer):
    shape = (100, 64)
    tensor = tensor_initializer.initialize_ones(shape)

    assert tensor.shape == shape
    assert torch.all(tensor == 1)
    assert tensor.dtype == torch.float32
    assert tensor.device == torch.device('cpu')

def test_initialize_random(tensor_initializer):
    shape = (100, 64)
    tensor = tensor_initializer.initialize_random(shape)

    assert tensor.shape == shape
    assert tensor.dtype == torch.float32
    assert tensor.device == torch.device('cpu')

def test_initialize_normal(tensor_initializer):
    shape = (100, 64)
    mean, std = 0.0, 1.0
    tensor = tensor_initializer.initialize_normal(shape, mean, std)

    assert tensor.shape == shape
    assert tensor.dtype == torch.float32
    assert tensor.device == torch.device('cpu')
    assert torch.abs(tensor.mean() - mean) < 1e-1
    assert torch.abs(tensor.std() - std) < 1e-1

def test_log_initialization(tensor_initializer, caplog):
    caplog.set_level(logging.DEBUG)
    shape = (100, 64)
    tensor = tensor_initializer.initialize_zeros(shape)
    assert "Zeros Initialization completed." in caplog.text
    assert f"Tensor shape: {tensor.shape}" in caplog.text