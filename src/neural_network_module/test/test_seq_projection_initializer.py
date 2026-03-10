import pytest
import torch
import logging
from neural_network_module.dil_katmani_module.seq_projection_module.seq_projection_initializer import SeqProjectionInitializer

@pytest.fixture
def tensor():
    return torch.empty(10, 10)

def test_seq_projection_initializer_xavier(tensor):
    initializer = SeqProjectionInitializer(method="xavier", log_level=logging.DEBUG)
    initialized_tensor = initializer.initialize(tensor)
    assert initialized_tensor.shape == tensor.shape

def test_seq_projection_initializer_kaiming(tensor):
    initializer = SeqProjectionInitializer(method="kaiming", log_level=logging.DEBUG)
    initialized_tensor = initializer.initialize(tensor)
    assert initialized_tensor.shape == tensor.shape

def test_seq_projection_initializer_normal(tensor):
    initializer = SeqProjectionInitializer(method="normal", log_level=logging.DEBUG)
    initialized_tensor = initializer.initialize(tensor)
    assert initialized_tensor.shape == tensor.shape

def test_seq_projection_initializer_uniform(tensor):
    initializer = SeqProjectionInitializer(method="uniform", log_level=logging.DEBUG)
    initialized_tensor = initializer.initialize(tensor)
    assert initialized_tensor.shape == tensor.shape

def test_invalid_initialization_method(tensor):
    with pytest.raises(ValueError):
        SeqProjectionInitializer(method="invalid", log_level=logging.DEBUG).initialize(tensor)