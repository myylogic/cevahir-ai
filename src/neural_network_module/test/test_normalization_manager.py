import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.normalization_manager import NormalizationManager

@pytest.fixture
def normalization_manager():
    return NormalizationManager(log_level=logging.DEBUG)

def test_set_and_apply_batch_normalizer(normalization_manager):
    normalization_manager.set_batch_normalizer(num_features=64)
    tensor = torch.randn(10, 64)
    normalized_tensor = normalization_manager.batch_normalize(tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_set_and_apply_group_normalizer(normalization_manager):
    normalization_manager.set_group_normalizer(num_groups=4, num_channels=64)
    tensor = torch.randn(10, 64)
    normalized_tensor = normalization_manager.group_normalize(tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_set_and_apply_instance_normalizer(normalization_manager):
    normalization_manager.set_instance_normalizer(num_features=64)
    tensor = torch.randn(10, 64, 32)  # Giriş boyutunu InstanceNorm1d ile uyumlu hale getir
    normalized_tensor = normalization_manager.instance_normalize(tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_set_and_apply_layer_normalizer(normalization_manager):
    normalization_manager.set_layer_normalizer(normalized_shape=64)
    tensor = torch.randn(10, 64)
    normalized_tensor = normalization_manager.layer_normalize(tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_set_and_apply_scaler_min_max(normalization_manager):
    normalization_manager.set_scaler(scale_range=(0, 1))
    tensor = torch.randn(10, 64)
    scaled_tensor = normalization_manager.scale(tensor, method="min_max")
    assert scaled_tensor.min() >= 0
    assert scaled_tensor.max() <= 1
    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_set_and_apply_scaler_standard(normalization_manager):
    normalization_manager.set_scaler(scale_range=(0, 1))
    tensor = torch.randn(10, 64)
    scaled_tensor = normalization_manager.scale(tensor, method="standard")
    assert scaled_tensor.mean().abs() < 1e-6  # Mean should be close to 0
    assert scaled_tensor.std().abs() - 1 < 1e-6  # Std should be close to 1
    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_set_and_apply_scaler_robust(normalization_manager):
    normalization_manager.set_scaler(scale_range=(0, 1))
    tensor = torch.randn(10, 64)
    scaled_tensor = normalization_manager.scale(tensor, method="robust")
    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_set_and_initialize_weights(normalization_manager):
    normalization_manager.set_initializer(init_type="xavier")
    tensor = torch.empty(3, 5)
    initialized_tensor = normalization_manager.initialize_weights(tensor)
    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_set_and_initialize_bias(normalization_manager):
    normalization_manager.set_initializer(init_type="xavier")
    tensor = torch.empty(5)
    initialized_tensor = normalization_manager.initialize_bias(tensor)
    assert torch.all(initialized_tensor == 0)
    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_batch_normalizer_not_set(normalization_manager):
    tensor = torch.randn(10, 64)
    with pytest.raises(ValueError):
        normalization_manager.batch_normalize(tensor)

def test_group_normalizer_not_set(normalization_manager):
    tensor = torch.randn(10, 64)
    with pytest.raises(ValueError):
        normalization_manager.group_normalize(tensor)

def test_instance_normalizer_not_set(normalization_manager):
    tensor = torch.randn(10, 64)
    with pytest.raises(ValueError):
        normalization_manager.instance_normalize(tensor)

def test_layer_normalizer_not_set(normalization_manager):
    tensor = torch.randn(10, 64)
    with pytest.raises(ValueError):
        normalization_manager.layer_normalize(tensor)

def test_scaler_not_set(normalization_manager):
    tensor = torch.randn(10, 64)
    with pytest.raises(ValueError):
        normalization_manager.scale(tensor, method="min_max")

def test_initializer_not_set_for_weights(normalization_manager):
    tensor = torch.empty(3, 5)
    with pytest.raises(ValueError):
        normalization_manager.initialize_weights(tensor)

def test_initializer_not_set_for_bias(normalization_manager):
    tensor = torch.empty(5)
    with pytest.raises(ValueError):
        normalization_manager.initialize_bias(tensor)