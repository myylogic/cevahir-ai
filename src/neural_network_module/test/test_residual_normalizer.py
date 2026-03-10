import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.residual_manager_module.residual_normalizer import ResidualNormalizer

@pytest.fixture
def residual_normalizer():
    return ResidualNormalizer(log_level=logging.DEBUG)

def test_normalize(residual_normalizer):
    tensor = torch.randn(100, 64)
    normalized_tensor = residual_normalizer.normalize(tensor)

    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_normalize_invalid_input(residual_normalizer):
    with pytest.raises(TypeError):
        residual_normalizer.normalize("invalid_input")

def test_log_normalization(residual_normalizer, caplog):
    caplog.set_level(logging.DEBUG)
    tensor = torch.randn(100, 64)
    normalized_tensor = residual_normalizer.normalize(tensor)
    assert "Normalization completed." in caplog.text
    assert f"Original tensor shape: {tensor.shape}" in caplog.text
    assert f"Normalized tensor shape: {normalized_tensor.shape}" in caplog.text