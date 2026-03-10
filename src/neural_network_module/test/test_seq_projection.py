import pytest
import torch
import logging
from neural_network_module.dil_katmani_module.seq_projection import SeqProjection

@pytest.fixture
def input_tensor():
    return torch.randn(32, 50, 128)

def test_seq_projection_forward(input_tensor):
    model = SeqProjection(input_dim=128, proj_dim=64, init_method="xavier", optimizer_type="adam", learning_rate=0.001, log_level=logging.DEBUG)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (32, 50, 64)

def test_seq_projection_initialization():
    model = SeqProjection(input_dim=128, proj_dim=64, init_method="kaiming", optimizer_type="sgd", learning_rate=0.01, log_level=logging.DEBUG)
    for param in model.parameters():
        assert param is not None

def test_invalid_initialization_method():
    with pytest.raises(ValueError):
        SeqProjection(input_dim=128, proj_dim=64, init_method="invalid", optimizer_type="adam", learning_rate=0.001, log_level=logging.DEBUG)

def test_invalid_optimizer_type():
    with pytest.raises(ValueError):
        SeqProjection(input_dim=128, proj_dim=64, init_method="xavier", optimizer_type="invalid", learning_rate=0.001, log_level=logging.DEBUG)