import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.memory_manager_module.memory_allocator import MemoryAllocator

@pytest.fixture
def allocator():
    return MemoryAllocator(log_level=logging.DEBUG)

def test_allocate_memory(allocator):
    tensor = allocator.allocate_memory((2, 3), dtype=torch.float32, device='cpu')
    assert tensor.shape == (2, 3)
    assert tensor.dtype == torch.float32
    assert tensor.device == torch.device('cpu')

def test_allocate_memory_invalid_size(allocator):
    with pytest.raises(ValueError):
        allocator.allocate_memory((-1, 3), dtype=torch.float32, device='cpu')

def test_release_memory(allocator):
    tensor = allocator.allocate_memory((2, 3), dtype=torch.float32, device='cpu')
    allocator.release_memory(tensor)

def test_release_memory_invalid_input(allocator):
    with pytest.raises(TypeError):
        allocator.release_memory("invalid_tensor")

def test_allocate_on_gpu(allocator):
    if torch.cuda.is_available():
        tensor = allocator.allocate_memory((2, 3), dtype=torch.float32, device='cuda')
        assert tensor.device == torch.device('cuda')