import pytest
import torch
import logging
from neural_network_module.dil_katmani_module.language_embedding_module.embedding_scaler import EmbeddingScaler

@pytest.fixture
def tensor():
    return torch.randn(32, 50, 128)

def test_embedding_scaler_min_max(tensor):
    scaler = EmbeddingScaler(method="min_max", log_level=logging.DEBUG)
    scaled_tensor = scaler.scale(tensor)
    assert scaled_tensor.min() == 0
    assert scaled_tensor.max() == 1
    assert scaled_tensor.shape == tensor.shape

def test_embedding_scaler_standard(tensor):
    scaler = EmbeddingScaler(method="standard", log_level=logging.DEBUG)
    scaled_tensor = scaler.scale(tensor)
    assert torch.isclose(scaled_tensor.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(scaled_tensor.std(), torch.tensor(1.0), atol=1e-6)
    assert scaled_tensor.shape == tensor.shape

def test_embedding_scaler_robust(tensor):
    scaler = EmbeddingScaler(method="robust", log_level=logging.DEBUG)
    scaled_tensor = scaler.scale(tensor)
    median = tensor.median()
    q75, q25 = tensor.quantile(0.75), tensor.quantile(0.25)
    expected_scaled_tensor = (tensor - median) / (q75 - q25)
    assert torch.allclose(scaled_tensor, expected_scaled_tensor, atol=1e-6)
    assert scaled_tensor.shape == tensor.shape

def test_invalid_scaling_method(tensor):
    with pytest.raises(ValueError):
        EmbeddingScaler(method="invalid", log_level=logging.DEBUG).scale(tensor)