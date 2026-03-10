import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.normalization_manager_module.normalization_scaler import NormalizationScaler

@pytest.fixture
def normalization_scaler():
    return NormalizationScaler(scale_range=(0, 1), log_level=logging.DEBUG)

def test_min_max_scale(normalization_scaler):
    tensor = torch.randn(10, 64)
    scaled_tensor = normalization_scaler.min_max_scale(tensor)
    assert scaled_tensor.min() >= 0
    assert scaled_tensor.max() <= 1
    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_standard_scale(normalization_scaler):
    tensor = torch.randn(10, 64)
    scaled_tensor = normalization_scaler.standard_scale(tensor)
    assert scaled_tensor.mean().abs() < 1e-6  # Mean should be close to 0
    assert scaled_tensor.std().abs() - 1 < 1e-6  # Std should be close to 1
    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_robust_scale(normalization_scaler):
    tensor = torch.randn(10, 64)
    scaled_tensor = normalization_scaler.robust_scale(tensor)
    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_apply_invalid_input(normalization_scaler):
    with pytest.raises(TypeError):
        normalization_scaler.min_max_scale("invalid_input")

def test_log_scaling(normalization_scaler, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(10, 64)
    scaled_tensor = normalization_scaler.min_max_scale(tensor)
    assert "Min-Max Scaling completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Scaled tensor shape: {scaled_tensor.shape}" in caplog.text
    assert f"Original tensor dtype: {tensor.dtype}" in caplog.text
    assert f"Scaled tensor dtype: {scaled_tensor.dtype}" in caplog.text
    assert f"Original tensor device: {tensor.device}" in caplog.text
    assert f"Scaled tensor device: {scaled_tensor.device}" in caplog.text