import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.residual_manager import ResidualManager

@pytest.fixture
def residual_manager():
    return ResidualManager(learning_rate=0.01, log_level=logging.DEBUG)

def test_initialize(residual_manager):
    tensor = torch.randn(100, 64)
    initialized_tensor = residual_manager.initialize(tensor)

    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_normalize(residual_manager):
    tensor = torch.randn(100, 64)
    normalized_tensor = residual_manager.normalize(tensor)

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_optimize(residual_manager):
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)
    optimized_tensor = residual_manager.optimize(tensor, gradients)

    assert optimized_tensor.shape == tensor.shape
    assert optimized_tensor.dtype == tensor.dtype
    assert optimized_tensor.device == tensor.device

def test_scale_min_max(residual_manager):
    tensor = torch.randn(100, 64)
    scaled_tensor = residual_manager.scale(tensor, method="min_max")

    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.min() >= 0
    assert scaled_tensor.max() <= 1
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_scale_standard(residual_manager):
    tensor = torch.randn(100, 64)
    scaled_tensor = residual_manager.scale(tensor, method="standard")

    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.mean().abs() < 1e-6
    assert scaled_tensor.std().abs() - 1 < 1e-6
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_scale_robust(residual_manager):
    tensor = torch.randn(100, 64)
    scaled_tensor = residual_manager.scale(tensor, method="robust")

    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_apply_connection(residual_manager):
    tensor1 = torch.randn(100, 64)
    tensor2 = torch.randn(100, 64)
    residual_tensor = residual_manager.apply_residual_connection(tensor1, tensor2)

    assert residual_tensor.shape == tensor1.shape
    assert residual_tensor.dtype == tensor1.dtype
    assert residual_tensor.device == tensor1.device

def test_invalid_scale_method(residual_manager):
    tensor = torch.randn(100, 64)
    with pytest.raises(ValueError):
        residual_manager.scale(tensor, method="invalid_method")

def test_invalid_tensor_input(residual_manager):
    with pytest.raises(TypeError):
        residual_manager.initialize("invalid_input")

    with pytest.raises(TypeError):
        residual_manager.normalize("invalid_input")

    with pytest.raises(TypeError):
        residual_manager.optimize("invalid_input", torch.randn(100, 64))

    with pytest.raises(TypeError):
        residual_manager.optimize(torch.randn(100, 64), "invalid_input")

    with pytest.raises(TypeError):
        residual_manager.apply_residual_connection("invalid_input", torch.randn(100, 64))

    with pytest.raises(TypeError):
        residual_manager.apply_residual_connection(torch.randn(100, 64), "invalid_input")

def test_log_execution(residual_manager, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)

    residual_manager.initialize(tensor)
    residual_manager.normalize(tensor)
    residual_manager.optimize(tensor, gradients)
    residual_manager.scale(tensor, method="min_max")
    residual_manager.apply_residual_connection(tensor, tensor)

    assert "Initialization completed." in caplog.text
    assert "Normalization completed." in caplog.text
    assert "Optimization completed." in caplog.text
    assert "Scaling (min_max) completed." in caplog.text
    assert "Residual Connection completed." in caplog.text