import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.quantum_adapter import QuantumAdapter

@pytest.fixture
def quantum_adapter():
    """ QuantumAdapter örneğini oluşturur. """
    return QuantumAdapter(projection_dim=128, num_tasks=4, learning_rate=0.01, scale_range=(0, 1), log_level=logging.DEBUG)

def test_initialize(quantum_adapter):
    """ Tensör başlatma işlemini test eder. """
    tensor = torch.randn(100, 64)
    initialized_tensor = quantum_adapter.initialize(tensor)

    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_scale_methods(quantum_adapter):
    """ Tüm ölçeklendirme metodlarını test eder. """
    tensor = torch.randn(100, 64)

    # Min-Max Scaling
    scaled_tensor_min_max = quantum_adapter.scale(tensor, method="min_max")
    assert scaled_tensor_min_max.shape == tensor.shape
    assert scaled_tensor_min_max.min() >= 0
    assert scaled_tensor_min_max.max() <= 1

    # Standard Scaling
    scaled_tensor_standard = quantum_adapter.scale(tensor, method="standard")
    assert scaled_tensor_standard.shape == tensor.shape
    assert torch.abs(scaled_tensor_standard.mean()) < 1e-6
    assert torch.abs(scaled_tensor_standard.std() - 1) < 1e-4


    # Robust Scaling
    scaled_tensor_robust = quantum_adapter.scale(tensor, method="robust")
    assert scaled_tensor_robust.shape == tensor.shape

def test_normalization_methods(quantum_adapter):
    """ Tüm normalizasyon metodlarını test eder. """
    tensor = torch.randn(100, 64)

    # Min-Max Normalization
    normalized_tensor_min_max = quantum_adapter.normalize(tensor, method="min_max")
    assert normalized_tensor_min_max.shape == tensor.shape
    assert normalized_tensor_min_max.min() >= 0
    assert normalized_tensor_min_max.max() <= 1

    # Standard Normalization
    normalized_tensor_standard = quantum_adapter.normalize(tensor, method="standard")
    assert normalized_tensor_standard.shape == tensor.shape
    assert torch.abs(normalized_tensor_standard.mean()) < 1e-6
    assert torch.abs(normalized_tensor_standard.std() - 1) < 1e-6

    # Robust Normalization
    normalized_tensor_robust = quantum_adapter.normalize(tensor, method="robust")
    assert normalized_tensor_robust.shape == tensor.shape

def test_optimize(quantum_adapter):
    """ Optimizasyon işlemini test eder. """
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)
    optimized_tensor = quantum_adapter.optimize(tensor, gradients)

    assert optimized_tensor.shape == tensor.shape
    assert optimized_tensor.dtype == tensor.dtype
    assert optimized_tensor.device == tensor.device

def test_project(quantum_adapter):
    """ Projeksiyon işlemini test eder. """
    tensor = torch.randn(100, 64)
    projected_tensor = quantum_adapter.project(tensor)

    assert projected_tensor.shape[0] == tensor.shape[0]
    assert projected_tensor.shape[1] == quantum_adapter.projector.projection_dim
    assert projected_tensor.dtype == tensor.dtype
    assert projected_tensor.device == tensor.device

def test_invalid_methods(quantum_adapter):
    """ Geçersiz ölçeklendirme ve normalizasyon metodlarını test eder. """
    tensor = torch.randn(100, 64)

    with pytest.raises(ValueError):
        quantum_adapter.scale(tensor, method="invalid_method")

    with pytest.raises(ValueError):
        quantum_adapter.normalize(tensor, method="invalid_method")

def test_invalid_tensor_types(quantum_adapter):
    """ Geçersiz tensör girişlerini test eder. """
    with pytest.raises(TypeError):
        quantum_adapter.initialize("invalid_input")

    with pytest.raises(TypeError):
        quantum_adapter.scale("invalid_input", method="min_max")

    with pytest.raises(TypeError):
        quantum_adapter.normalize("invalid_input", method="min_max")

    with pytest.raises(TypeError):
        quantum_adapter.optimize("invalid_input", torch.randn(100, 64))

    with pytest.raises(TypeError):
        quantum_adapter.optimize(torch.randn(100, 64), "invalid_input")

    with pytest.raises(TypeError):
        quantum_adapter.project("invalid_input")

def test_log_execution(quantum_adapter, caplog):
    """ Loglama işlemlerini test eder. """
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    gradients = torch.randn(100, 64)

    quantum_adapter.initialize(tensor)
    quantum_adapter.scale(tensor, method="min_max")
    quantum_adapter.normalize(tensor, method="standard")
    quantum_adapter.optimize(tensor, gradients)
    quantum_adapter.project(tensor)

    assert "Initialization completed successfully." in caplog.text
    assert "Scaling completed successfully." in caplog.text
    assert "Normalization completed successfully." in caplog.text
    assert "Optimization completed successfully." in caplog.text
    assert "Projection completed successfully." in caplog.text
