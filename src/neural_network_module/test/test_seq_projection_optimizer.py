import pytest
import torch
import logging
from neural_network_module.dil_katmani_module.seq_projection_module.seq_projection_optimizer import SeqProjectionOptimizer

@pytest.fixture
def model_parameters():
    return [torch.randn(10, 10, requires_grad=True)]

def test_seq_projection_optimizer_adam(model_parameters):
    optimizer = SeqProjectionOptimizer(model_parameters, learning_rate=0.01, optimizer_type="adam", log_level=logging.DEBUG)
    optimizer.zero_grad()
    loss = model_parameters[0].sum()
    loss.backward()
    optimizer.step()
    assert model_parameters[0].grad is not None

def test_seq_projection_optimizer_sgd(model_parameters):
    optimizer = SeqProjectionOptimizer(model_parameters, learning_rate=0.01, optimizer_type="sgd", log_level=logging.DEBUG)
    optimizer.zero_grad()
    loss = model_parameters[0].sum()
    loss.backward()
    optimizer.step()
    assert model_parameters[0].grad is not None

def test_seq_projection_optimizer_rmsprop(model_parameters):
    optimizer = SeqProjectionOptimizer(model_parameters, learning_rate=0.01, optimizer_type="rmsprop", log_level=logging.DEBUG)
    optimizer.zero_grad()
    loss = model_parameters[0].sum()
    loss.backward()
    optimizer.step()
    assert model_parameters[0].grad is not None

def test_invalid_optimizer_type(model_parameters):
    with pytest.raises(ValueError):
        SeqProjectionOptimizer(model_parameters, learning_rate=0.01, optimizer_type="invalid", log_level=logging.DEBUG)