# -*- coding: utf-8 -*-
"""Fixtures for architecture standards tests. Project root and minimal model config."""
import os
import sys
import pytest
import torch

# Proje kökü
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def seed():
    return 42


@pytest.fixture(scope="session")
def minimal_model_config(seed):
    """Minimal CevahirNeuralNetwork config for fast tests (train.py ile uyumlu parametre isimleri)."""
    torch.manual_seed(seed)
    return {
        "learning_rate": 1e-4,
        "dropout": 0.1,
        "vocab_size": 2000,
        "embed_dim": 64,
        "seq_proj_dim": 64,
        "num_heads": 2,
        "attention_type": "multi_head",
        "normalization_type": "layer_norm",
        "device": "cpu",
        "log_level": 40,  # WARNING
        "num_layers": 2,
        "ffn_dim": 256,
        "pre_norm": True,
        "causal_mask": True,
        "use_flash_attention": False,
        "pe_mode": "rope",
        "use_gradient_checkpointing": False,
        "tie_weights": True,
        "use_rmsnorm": True,
        "use_swiglu": True,
        "use_kv_cache": False,
        "max_cache_len": 512,
        "use_advanced_checkpointing": False,
        "checkpointing_strategy": "selective",
        "quantization_type": "none",
        "use_moe": False,
        "num_experts": 2,
        "moe_top_k": 1,
    }


@pytest.fixture(scope="session")
def minimal_model(minimal_model_config):
    """Minimal CevahirNeuralNetwork instance (eval mode)."""
    from src.neural_network import CevahirNeuralNetwork
    model = CevahirNeuralNetwork(**minimal_model_config)
    model.eval()
    return model


@pytest.fixture
def batch_minimal(seed):
    """Small batch for forward: [2, 8]."""
    torch.manual_seed(seed)
    return torch.randint(0, 2000, (2, 8))


@pytest.fixture
def rope_pe(seed):
    """RoPE PositionalEncoding for module-level tests (head_dim=32, num_heads=2)."""
    from src.neural_network_module.dil_katmani_module.positional_encoding import PositionalEncoding
    torch.manual_seed(seed)
    return PositionalEncoding(embed_dim=64, max_len=256, dropout=0.0, mode="rope", num_heads=2)


@pytest.fixture
def ffn_swiglu(seed):
    """FeedForwardNetwork SwiGLU for module-level tests."""
    from src.neural_network_module.ortak_katman_module.feed_forward_network import FeedForwardNetwork
    torch.manual_seed(seed)
    return FeedForwardNetwork(embed_dim=64, ffn_dim=256, dropout=0.0, activation="swiglu")
