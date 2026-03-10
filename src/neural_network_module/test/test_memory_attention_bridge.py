import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.memory_manager_module.memory_attention_bridge import MemoryAttentionBridge

@pytest.fixture
def bridge():
    return MemoryAttentionBridge(log_level=logging.DEBUG)

def test_bridge_attention(bridge):
    memory_tensor = torch.randn(10, 20, 64)
    attention_tensor = torch.randn(10, 20, 64)
    bridged_tensor = bridge.bridge_attention(memory_tensor, attention_tensor)
    assert bridged_tensor.shape == (10, 20, 128)

def test_bridge_attention_invalid_memory_tensor(bridge):
    memory_tensor = "invalid_tensor"
    attention_tensor = torch.randn(10, 20, 64)
    with pytest.raises(ValueError):
        bridge.bridge_attention(memory_tensor, attention_tensor)

def test_bridge_attention_invalid_attention_tensor(bridge):
    memory_tensor = torch.randn(10, 20, 64)
    attention_tensor = "invalid_tensor"
    with pytest.raises(ValueError):
        bridge.bridge_attention(memory_tensor, attention_tensor)

def test_validate_tensors_invalid_input(bridge):
    with pytest.raises(ValueError):
        bridge.validate_tensors("invalid_tensor")

def test_log_attention_bridge(bridge, caplog):
    caplog.set_level(logging.DEBUG)
    memory_tensor = torch.randn(10, 20, 64)
    attention_tensor = torch.randn(10, 20, 64)
    bridged_tensor = bridge.bridge_attention(memory_tensor, attention_tensor)
    assert "Attention bridge created." in caplog.text
    assert f"Memory tensor shape: {memory_tensor.shape}" in caplog.text
    assert f"Attention tensor shape: {attention_tensor.shape}" in caplog.text
    assert f"Bridged tensor shape: {bridged_tensor.shape}" in caplog.text

def test_bridge_attention_with_masks(bridge):
    memory_tensor = torch.randn(10, 20, 64)
    attention_tensor = torch.randn(10, 20, 64)
    memory_mask = torch.randint(0, 2, (10, 20), dtype=torch.bool)
    attention_mask = torch.randint(0, 2, (10, 20), dtype=torch.bool)
    bridged_tensor = bridge.bridge_attention(memory_tensor, attention_tensor, memory_mask, attention_mask)
    assert bridged_tensor.shape == (10, 20, 128)