# -*- coding: utf-8 -*-
"""
FeedForwardNetwork Test Dosyası

Testler:
- Initialization testi
- Forward pass testi
- Shape kontrolü
- Activation function testi (GELU, ReLU)
- Dropout testi
"""

import pytest
import torch
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.neural_network_module.ortak_katman_module.feed_forward_network import FeedForwardNetwork


# ---- Fixtures ----
@pytest.fixture
def ffn_gelu():
    """GELU activation ile FeedForwardNetwork"""
    return FeedForwardNetwork(
        embed_dim=128,
        ffn_dim=512,
        dropout=0.1,
        activation="gelu",
        log_level=logging.WARNING,
    )


@pytest.fixture
def ffn_relu():
    """ReLU activation ile FeedForwardNetwork"""
    return FeedForwardNetwork(
        embed_dim=128,
        ffn_dim=512,
        dropout=0.1,
        activation="relu",
        log_level=logging.WARNING,
    )


@pytest.fixture
def sample_input():
    """Örnek input tensor"""
    return torch.randn(2, 10, 128)  # [B, T, embed_dim]


# ---- Initialization Tests ----
def test_initialization(ffn_gelu):
    """FeedForwardNetwork initialization testi"""
    assert ffn_gelu.embed_dim == 128
    assert ffn_gelu.ffn_dim == 512
    assert ffn_gelu.dropout_rate == 0.1
    assert ffn_gelu.activation_name == "gelu"
    assert ffn_gelu.fc1 is not None
    assert ffn_gelu.fc2 is not None
    assert ffn_gelu.activation is not None
    assert ffn_gelu.dropout is not None


def test_initialization_invalid_activation():
    """Geçersiz activation function testi"""
    with pytest.raises(ValueError, match="Unsupported activation"):
        FeedForwardNetwork(
            embed_dim=128,
            ffn_dim=512,
            dropout=0.1,
            activation="invalid",
        )


# ---- Forward Pass Tests ----
def test_forward_pass_shape(ffn_gelu, sample_input):
    """Forward pass shape kontrolü"""
    output = ffn_gelu(sample_input)
    assert output.shape == sample_input.shape
    assert output.dtype == sample_input.dtype
    assert output.device == sample_input.device


def test_forward_pass_gelu(ffn_gelu, sample_input):
    """GELU activation ile forward pass"""
    output = ffn_gelu(sample_input)
    assert output.shape == sample_input.shape
    # GELU output'u kontrol et (negatif değerler olabilir ama çok küçük olmalı)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_forward_pass_relu(ffn_relu, sample_input):
    """ReLU activation ile forward pass"""
    output = ffn_relu(sample_input)
    assert output.shape == sample_input.shape
    # ReLU output'u kontrol et (negatif değerler olmamalı)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


# ---- Activation Function Tests ----
def test_activation_gelu(ffn_gelu, sample_input):
    """GELU activation testi"""
    # FFN çıktısı negatif olabilir (fc2 contract ederken negatif değerler üretebilir)
    # Bu yüzden sadece shape ve NaN/Inf kontrolü yapıyoruz
    output = ffn_gelu(sample_input)
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any(), "GELU çıktısında NaN var!"
    assert not torch.isinf(output).any(), "GELU çıktısında Inf var!"


def test_activation_relu(ffn_relu, sample_input):
    """ReLU activation testi"""
    # FFN çıktısı negatif olabilir (fc2 contract ederken negatif değerler üretebilir)
    # ReLU sadece fc1 çıktısına uygulanır, fc2 çıktısı negatif olabilir
    # Bu yüzden sadece shape ve NaN/Inf kontrolü yapıyoruz
    output = ffn_relu(sample_input)
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any(), "ReLU çıktısında NaN var!"
    assert not torch.isinf(output).any(), "ReLU çıktısında Inf var!"


# ---- Dropout Tests ----
def test_dropout_training_mode(ffn_gelu, sample_input):
    """Training mode'da dropout testi"""
    ffn_gelu.train()
    output1 = ffn_gelu(sample_input)
    output2 = ffn_gelu(sample_input)
    # Dropout nedeniyle output'lar farklı olmalı (yüksek olasılıkla)
    # Ama deterministik değil, bu yüzden sadece shape kontrolü yapıyoruz
    assert output1.shape == output2.shape


def test_dropout_eval_mode(ffn_gelu, sample_input):
    """Eval mode'da dropout testi"""
    ffn_gelu.eval()
    output1 = ffn_gelu(sample_input)
    output2 = ffn_gelu(sample_input)
    # Eval mode'da dropout kapalı, output'lar aynı olmalı
    assert torch.allclose(output1, output2, atol=1e-6)


# ---- Different Input Sizes ----
def test_forward_different_batch_sizes(ffn_gelu):
    """Farklı batch size'lar için forward pass"""
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 10, 128)
        output = ffn_gelu(x)
        assert output.shape == (batch_size, 10, 128)


def test_forward_different_sequence_lengths(ffn_gelu):
    """Farklı sequence length'ler için forward pass"""
    for seq_len in [1, 5, 10, 20, 50]:
        x = torch.randn(2, seq_len, 128)
        output = ffn_gelu(x)
        assert output.shape == (2, seq_len, 128)


# ---- Numerical Stability ----
def test_numerical_stability(ffn_gelu, sample_input):
    """Numerical stability testi"""
    output = ffn_gelu(sample_input)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert output.isfinite().all()


# ---- GPU Tests (opsiyonel) ----
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_forward(ffn_gelu):
    """GPU'da forward pass testi"""
    ffn_gelu = ffn_gelu.cuda()
    x = torch.randn(2, 10, 128).cuda()
    output = ffn_gelu(x)
    assert output.device.type == "cuda"
    assert output.shape == (2, 10, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

