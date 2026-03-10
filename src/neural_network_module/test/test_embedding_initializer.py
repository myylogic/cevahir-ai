import pytest
import torch
import logging
from neural_network_module.dil_katmani_module.language_embedding_module.embedding_initializer import EmbeddingInitializer

@pytest.fixture
def tensor():
    return torch.empty(10, 10)

def test_embedding_initializer_xavier(tensor):
    initializer = EmbeddingInitializer(method="xavier", log_level=logging.DEBUG)
    initialized_tensor = initializer.initialize(tensor)
    assert initialized_tensor.shape == tensor.shape

def test_embedding_initializer_kaiming(tensor):
    initializer = EmbeddingInitializer(method="kaiming", log_level=logging.DEBUG)
    initialized_tensor = initializer.initialize(tensor)
    assert initialized_tensor.shape == tensor.shape

def test_embedding_initializer_normal(tensor):
    initializer = EmbeddingInitializer(method="normal", log_level=logging.DEBUG)
    initialized_tensor = initializer.initialize(tensor)
    assert initialized_tensor.shape == tensor.shape

def test_embedding_initializer_uniform(tensor):
    initializer = EmbeddingInitializer(method="uniform", log_level=logging.DEBUG)
    initialized_tensor = initializer.initialize(tensor)
    assert initialized_tensor.shape == tensor.shape

def test_invalid_initialization_method(tensor):
    with pytest.raises(ValueError):
        EmbeddingInitializer(method="invalid", log_level=logging.DEBUG).initialize(tensor)