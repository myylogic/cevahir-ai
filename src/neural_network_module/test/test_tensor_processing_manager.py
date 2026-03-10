import pytest
import torch
import logging
import time

from neural_network_module.ortak_katman_module.tensor_processing_manager import TensorProcessingManager


# ---- Fixtures ----
@pytest.fixture
def tensor_processing_manager():
    # input_dim=128, output_dim=64, num_tasks=1, lr=1e-4
    return TensorProcessingManager(
        input_dim=128, output_dim=64, num_tasks=1, learning_rate=0.0001, log_level=logging.DEBUG
    )


# ---- Basic validations ----
def test_validate_dimension():
    tpm = TensorProcessingManager(input_dim=128, output_dim=64, num_tasks=1, learning_rate=0.0001, log_level=logging.DEBUG)
    assert tpm._validate_dimension(128, "input_dim") == 128
    with pytest.raises(ValueError):
        tpm._validate_dimension(-5, "input_dim")
    with pytest.raises(ValueError):
        tpm._validate_dimension("128", "input_dim")


# ---- Initialize ----
def test_initialize_zeros(tensor_processing_manager):
    shape = (32, 128)
    tensor = tensor_processing_manager.initialize(shape, method="zeros")
    assert tensor.shape == torch.Size(shape)
    assert torch.all(tensor == 0)


def test_invalid_initialize_method(tensor_processing_manager):
    with pytest.raises(Exception):
        tensor_processing_manager.initialize((32, 128), method="invalid_method")


# ---- Projection ----
def test_project(tensor_processing_manager):
    # 2D giriş -> [N, input_dim] => [N, output_dim]
    x = torch.randn(32, tensor_processing_manager.input_dim)
    y = tensor_processing_manager.project(x)
    assert y.shape == (32, tensor_processing_manager.output_dim)
    assert y.dtype == x.dtype


def test_project_3d_input(tensor_processing_manager):
    # 3D giriş -> [B, T, input_dim] => [B, T, output_dim]
    x = torch.randn(8, 10, tensor_processing_manager.input_dim)
    y = tensor_processing_manager.project(x)
    assert y.shape == (8, 10, tensor_processing_manager.output_dim)


def test_project_invalid_input_type(tensor_processing_manager):
    with pytest.raises(TypeError):
        tensor_processing_manager.project("invalid_input")


def test_project_last_dim_mismatch(tensor_processing_manager):
    # Son boyut input_dim'e eşit değilse ValueError beklenir
    x = torch.randn(32, tensor_processing_manager.input_dim + 1)
    with pytest.raises(ValueError):
        tensor_processing_manager.project(x)


def test_projection_with_num_tasks_effective_dim():
    # num_tasks > 1 olduğunda effective_output_dim = output_dim // num_tasks
    tpm = TensorProcessingManager(input_dim=128, output_dim=64, num_tasks=4, learning_rate=1e-4, log_level=logging.DEBUG)
    x = torch.randn(16, 128)
    y = tpm.project(x)
    assert y.shape == (16, 64 // 4)  # 16


# ---- Optimize ----
def test_performance_of_optimize(tensor_processing_manager):
    # Küçük bir tensörde optimize makul sürede bitmeli (ortam bağımlı; geniş tolerans)
    tensor = torch.randn(64, 128)
    gradients = torch.randn(64, 128)
    start = time.time()
    _ = tensor_processing_manager.optimize(tensor, gradients, method="sgd")
    duration = time.time() - start
    assert duration < 0.5, f"Optimize işlemi çok yavaş: {duration:.6f} saniye"


def test_optimize_sgd(tensor_processing_manager):
    T = torch.ones(10, 128)
    G = torch.full((10, 128), 0.5)
    lr = tensor_processing_manager.learning_rate
    out = tensor_processing_manager.optimize(T, G, method="sgd")
    expected = T - lr * G
    assert torch.allclose(out, expected, atol=1e-6)


def test_optimize_adam(tensor_processing_manager):
    T = torch.ones(10, 128)
    G = torch.full((10, 128), 0.5)
    out = tensor_processing_manager.optimize(
        T, G, method="adam", beta1=0.9, beta2=0.999, epsilon=1e-8, t=1
    )
    assert out.shape == T.shape
    assert out.dtype == T.dtype


def test_optimize_gradient_shape_mismatch(tensor_processing_manager):
    T = torch.randn(8, 128)
    G = torch.randn(8, 64)  # yanlış şekil
    with pytest.raises(ValueError):
        tensor_processing_manager.optimize(T, G, method="sgd")


def test_optimize_gradient_cast_dtype_device(tensor_processing_manager):
    # Gradients dtype/device farklı olsa bile otomatik eşitlenmeli
    T = torch.randn(4, 128)  # float32 CPU
    G = torch.randn(4, 128, dtype=torch.float64)  # float64 CPU
    out = tensor_processing_manager.optimize(T, G, method="sgd")
    assert out.dtype == T.dtype
    assert out.device == T.device


# ---- Logging ----
def test_log_tensor_stats(caplog, tensor_processing_manager):
    caplog.set_level(logging.DEBUG)
    dummy = torch.randn(16, 128)
    tensor_processing_manager._log_tensor_stats(dummy, "Dummy Tensor")
    log_text = caplog.text
    assert "Dummy Tensor stats:" in log_text
    assert "shape=" in log_text
    assert "min=" in log_text
    assert "max=" in log_text
    assert "mean=" in log_text
    assert "std=" in log_text


def test_log_execution(caplog, tensor_processing_manager):
    caplog.set_level(logging.DEBUG)
    dummy = torch.randn(8, 64)
    tensor_processing_manager._log_execution("TestOperation", dummy)
    log_text = caplog.text
    assert "TestOperation completed successfully." in log_text
    assert "Tensor shape:" in log_text
