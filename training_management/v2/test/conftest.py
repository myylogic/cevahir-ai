"""
Pytest Configuration and Fixtures
==================================

Shared fixtures for V2 tests.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any

# Test fixtures


@pytest.fixture
def device():
    """Test device (CPU for consistency)"""
    return torch.device("cpu")


@pytest.fixture
def vocab_size():
    """Test vocabulary size"""
    return 128


@pytest.fixture
def seq_len():
    """Test sequence length"""
    return 16


@pytest.fixture
def batch_size():
    """Test batch size"""
    return 4


@pytest.fixture
def embed_dim():
    """Test embedding dimension"""
    return 32


class SimpleDataset(Dataset):
    """Simple test dataset"""
    def __init__(self, n_samples=32, seq_len=16, vocab_size=128, seed=42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.inputs = torch.randint(0, vocab_size, (n_samples, seq_len), generator=g, dtype=torch.long)
        self.targets = self.inputs.clone()
        self.vocab_size = vocab_size

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class SimpleModel(nn.Module):
    """Simple test model"""
    def __init__(self, vocab_size=128, embed_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        logits = self.proj(x)
        return logits


@pytest.fixture
def simple_model(vocab_size, embed_dim):
    """Simple model fixture"""
    return SimpleModel(vocab_size=vocab_size, embed_dim=embed_dim)


@pytest.fixture
def simple_dataset(vocab_size, seq_len):
    """Simple dataset fixture"""
    return SimpleDataset(n_samples=32, seq_len=seq_len, vocab_size=vocab_size)


@pytest.fixture
def simple_train_loader(simple_dataset, batch_size):
    """Training dataloader fixture"""
    return DataLoader(simple_dataset, batch_size=batch_size, shuffle=False)


@pytest.fixture
def simple_val_loader(simple_dataset, batch_size):
    """Validation dataloader fixture"""
    return DataLoader(simple_dataset, batch_size=batch_size, shuffle=False)


@pytest.fixture
def simple_config(vocab_size, device):
    """Simple config fixture"""
    return {
        "vocab_size": vocab_size,
        "device": device,
        "epochs": 2,
        "pad_token_id": 0,
        "use_amp": False,
        "grad_accum_steps": 1,
        "max_grad_norm": 1.0,
        "enable_file_logging": False,
        "calculate_advanced_metrics": False,
        "enable_visualizations": False,
        "track_memory": False,
        "track_performance": False,
        "use_progress_bar": False,
    }


@pytest.fixture
def simple_optimizer(simple_model):
    """Simple optimizer fixture"""
    return torch.optim.Adam(simple_model.parameters(), lr=0.001)


@pytest.fixture
def simple_criterion(vocab_size, device):
    """Simple criterion fixture (without EOS weight)"""
    return nn.CrossEntropyLoss(ignore_index=0, reduction="mean")


@pytest.fixture
def eos_weighted_criterion(vocab_size, device):
    """EOS weight'li criterion fixture"""
    eos_id = 3
    loss_weights = torch.ones(vocab_size, device=device)
    loss_weights[eos_id] = 0.1  # EOS weight
    return nn.CrossEntropyLoss(
        weight=loss_weights,
        ignore_index=0,
        reduction="mean",
        label_smoothing=0.1
    )


@pytest.fixture
def mock_logger():
    """Mock logger fixture"""
    class MockLogger:
        def log_info(self, msg):
            pass
        def log_error(self, msg):
            pass
        def log_warning(self, msg):
            pass
        def log_debug(self, msg):
            pass
    return MockLogger()

