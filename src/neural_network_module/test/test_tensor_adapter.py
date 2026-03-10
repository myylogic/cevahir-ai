import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.tensor_adapter import TensorAdapter

@pytest.fixture
def tensor_adapter():
    return TensorAdapter(log_level=logging.DEBUG)

def test_tensor_adapter(tensor_adapter):
    learning_rate = 0.01
    input_dim = 128
    output_dim = 128
    shape = (100, input_dim)
    num_groups = 64  

    tensor_adapter.set_optimizer(learning_rate)
    tensor_adapter.set_projection(input_dim, output_dim)

    # Başlatma
    tensor = tensor_adapter.initialize(shape, method="random")
    assert tensor.shape == shape
    assert tensor.dtype == torch.float32

    # Batch Normalization
    tensor = tensor_adapter.normalize(tensor, method="batch")
    assert tensor.shape == shape

    # Group Normalization
    tensor = tensor_adapter.normalize(tensor, method="group", num_groups=num_groups)
    assert tensor.shape == shape

    # Optimizasyon
    gradients = torch.randn(100, input_dim)
    tensor = tensor_adapter.optimize(tensor, gradients, method="adam", t=1)
    assert tensor.shape == shape

    # Projeksiyon
    tensor = tensor_adapter.project(tensor)
    assert tensor.shape == (100, output_dim)

    # Ölçekleme
    tensor = tensor_adapter.scale(tensor, method="min_max")
    assert tensor.shape == (100, output_dim)

def test_invalid_input(tensor_adapter):
    with pytest.raises(ValueError):
        tensor_adapter.set_projection(128, 64)  
        tensor = torch.randn(100, 64)
        tensor_adapter.project(tensor)

# **YENİ TEST METOTLARI**

def test_initialize_zeros(tensor_adapter):
    shape = (50, 128)
    tensor = tensor_adapter.initialize(shape, method="zeros")
    assert tensor.shape == shape
    assert torch.all(tensor == 0), "Tensor should be initialized with zeros"

def test_initialize_ones(tensor_adapter):
    shape = (30, 256)
    tensor = tensor_adapter.initialize(shape, method="ones")
    assert tensor.shape == shape
    assert torch.all(tensor == 1), "Tensor should be initialized with ones"

def test_initialize_normal_distribution(tensor_adapter):
    shape = (100, 64)
    tensor = tensor_adapter.initialize(shape, method="normal")
    assert tensor.shape == shape
    assert torch.is_tensor(tensor), "Returned object should be a tensor"

def test_invalid_initialization_method(tensor_adapter):
    shape = (10, 10)
    with pytest.raises(ValueError):
        tensor_adapter.initialize(shape, method="invalid_method")

def test_batch_normalization_invalid_tensor(tensor_adapter):
    with pytest.raises(TypeError):
        tensor_adapter.normalize("invalid_tensor", method="batch")

def test_projection_without_setting(tensor_adapter):
    tensor = torch.randn(100, 128)
    with pytest.raises(ValueError):
        tensor_adapter.project(tensor)

def test_optimization_without_setting(tensor_adapter):
    tensor = torch.randn(100, 128)
    gradients = torch.randn(100, 128)
    with pytest.raises(ValueError):
        tensor_adapter.optimize(tensor, gradients, method="adam")

def test_scaling_with_standard_method(tensor_adapter):
    tensor = torch.randn(100, 128)
    scaled_tensor = tensor_adapter.scale(tensor, method="standard")
    assert scaled_tensor.shape == tensor.shape
    assert torch.is_tensor(scaled_tensor), "Scaling should return a tensor"

def test_scaling_with_invalid_method(tensor_adapter):
    tensor = torch.randn(50, 50)
    with pytest.raises(ValueError):
        tensor_adapter.scale(tensor, method="invalid")

def test_group_normalization_invalid_groups(tensor_adapter):
    tensor = torch.randn(100, 128)

    # Geçerli grup sayıları için test
    for valid_groups in [1, 2, 4, 8, 16, 32, 64, 128]:  # 128'i tam bölebilen değerler
        normalized_tensor = tensor_adapter.normalize(tensor, method="group", num_groups=valid_groups)
        assert normalized_tensor.shape == tensor.shape

    # Geçersiz grup sayısı için test
    with pytest.raises(ValueError):
        tensor_adapter.normalize(tensor, method="group", num_groups=500)  # 128'i tam bölmeyen bir sayı

