import pytest
import torch
import logging
from neural_network_module.dil_katmani_module.language_embedding_module.embedding_projection import EmbeddingProjection

@pytest.fixture
def input_tensor():
    return torch.randn(32, 50, 128)

def test_embedding_projection_forward(input_tensor):
    model = EmbeddingProjection(input_dim=128, proj_dim=64, init_method="xavier", log_level=logging.DEBUG)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (32, 50, 64)

def test_embedding_projection_initialization():
    model = EmbeddingProjection(input_dim=128, proj_dim=64, init_method="kaiming", log_level=logging.DEBUG)
    for param in model.parameters():
        assert param is not None

def test_invalid_initialization_method():
    with pytest.raises(ValueError):
        EmbeddingProjection(input_dim=128, proj_dim=64, init_method="invalid", log_level=logging.DEBUG)