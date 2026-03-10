import pytest
import torch
import logging
from neural_network_module.dil_katmani_module.language_embedding import LanguageEmbedding

@pytest.fixture
def input_tensor():
    return torch.randint(0, 5000, (32, 50))

def test_language_embedding_forward(input_tensor):
    model = LanguageEmbedding(vocab_size=5000, embed_dim=128, init_method="xavier", proj_dim=64, scale_method="min_max", log_level=logging.DEBUG)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (32, 50, 64)

def test_language_embedding_initialization():
    model = LanguageEmbedding(vocab_size=5000, embed_dim=128, init_method="kaiming", proj_dim=64, log_level=logging.DEBUG)
    for param in model.parameters():
        assert param is not None

def test_language_embedding_no_proj_no_scale(input_tensor):
    model = LanguageEmbedding(vocab_size=5000, embed_dim=128, init_method="xavier", log_level=logging.DEBUG)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (32, 50, 128)

def test_invalid_initialization_method():
    with pytest.raises(ValueError):
        LanguageEmbedding(vocab_size=5000, embed_dim=128, init_method="invalid", log_level=logging.DEBUG)

def test_invalid_scaling_method(input_tensor):
    with pytest.raises(ValueError):
        model = LanguageEmbedding(vocab_size=5000, embed_dim=128, init_method="xavier", scale_method="invalid", log_level=logging.DEBUG)
        model(input_tensor)