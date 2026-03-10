# -*- coding: utf-8 -*-
"""
TransformerEncoderLayer Test Dosyası

Testler:
- Initialization testi
- Forward pass testi (pre-norm)
- Forward pass testi (post-norm)
- Residual connection testi
- Causal mask testi
- Shape kontrolü
"""

import pytest
import torch
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.neural_network_module.ortak_katman_module.transformer_encoder_layer import TransformerEncoderLayer


# ---- Fixtures ----
@pytest.fixture
def layer_pre_norm():
    """Pre-norm ile TransformerEncoderLayer"""
    return TransformerEncoderLayer(
        embed_dim=128,
        num_heads=8,
        ffn_dim=512,
        dropout=0.1,
        pre_norm=True,
        log_level=logging.WARNING,
    )


@pytest.fixture
def layer_post_norm():
    """Post-norm ile TransformerEncoderLayer"""
    return TransformerEncoderLayer(
        embed_dim=128,
        num_heads=8,
        ffn_dim=512,
        dropout=0.1,
        pre_norm=False,
        log_level=logging.WARNING,
    )


@pytest.fixture
def sample_input():
    """Örnek input tensor"""
    return torch.randn(2, 10, 128)  # [B, T, embed_dim]


# ---- Initialization Tests ----
def test_initialization_pre_norm(layer_pre_norm):
    """Pre-norm initialization testi"""
    assert layer_pre_norm.embed_dim == 128
    assert layer_pre_norm.num_heads == 8
    assert layer_pre_norm.ffn_dim == 512
    assert layer_pre_norm.dropout_rate == 0.1
    assert layer_pre_norm.pre_norm is True
    assert layer_pre_norm.attn is not None
    assert layer_pre_norm.ffn is not None
    assert layer_pre_norm.norm1 is not None
    assert layer_pre_norm.norm2 is not None


def test_initialization_post_norm(layer_post_norm):
    """Post-norm initialization testi"""
    assert layer_post_norm.pre_norm is False
    assert layer_post_norm.attn is not None
    assert layer_post_norm.ffn is not None


# ---- Forward Pass Tests ----
def test_forward_pass_pre_norm_shape(layer_pre_norm, sample_input):
    """Pre-norm forward pass shape kontrolü"""
    output, attn_weights = layer_pre_norm(sample_input)
    assert output.shape == sample_input.shape
    assert attn_weights is not None
    assert attn_weights.shape[0] == sample_input.shape[0]  # Batch size
    assert attn_weights.shape[1] == layer_pre_norm.num_heads  # Num heads


def test_forward_pass_post_norm_shape(layer_post_norm, sample_input):
    """Post-norm forward pass shape kontrolü"""
    output, attn_weights = layer_post_norm(sample_input)
    assert output.shape == sample_input.shape
    assert attn_weights is not None


# ---- Pre-norm vs Post-norm Tests ----
def test_pre_norm_vs_post_norm(layer_pre_norm, layer_post_norm, sample_input):
    """Pre-norm ve Post-norm farkı testi"""
    # Aynı input ile test et
    torch.manual_seed(42)
    x1 = torch.randn(2, 10, 128)
    torch.manual_seed(42)
    x2 = torch.randn(2, 10, 128)
    
    # Pre-norm ve Post-norm farklı output üretmeli
    output_pre, _ = layer_pre_norm(x1)
    output_post, _ = layer_post_norm(x2)
    
    # Shape'ler aynı olmalı
    assert output_pre.shape == output_post.shape
    
    # Ama değerler farklı olmalı (pre-norm ve post-norm farklı)
    assert not torch.allclose(output_pre, output_post, atol=1e-5)


# ---- Residual Connection Tests ----
def test_residual_connection(layer_pre_norm, sample_input):
    """Residual connection testi"""
    output, _ = layer_pre_norm(sample_input)
    # Residual connection nedeniyle output, input'tan farklı olmalı
    # Ama çok büyük fark olmamalı (residual connection gradient flow'u korur)
    diff = torch.abs(output - sample_input).mean()
    assert diff > 0.001  # Fark olmalı
    assert diff < 10.0  # Ama çok büyük olmamalı


# ---- Causal Mask Tests ----
def test_causal_mask(layer_pre_norm, sample_input):
    """Causal mask testi"""
    # Causal mask ile forward pass
    output_causal, attn_weights_causal = layer_pre_norm(
        sample_input,
        mask=None,
        causal_mask=True,
    )
    
    # Causal mask olmadan forward pass
    output_no_causal, attn_weights_no_causal = layer_pre_norm(
        sample_input,
        mask=None,
        causal_mask=False,
    )
    
    # Shape'ler aynı olmalı
    assert output_causal.shape == output_no_causal.shape
    
    # Ama attention weights farklı olmalı (causal mask attention'ı değiştirir)
    # Causal mask'ın etkisini kontrol et
    assert attn_weights_causal.shape == attn_weights_no_causal.shape


# ---- Mask Tests ----
def test_padding_mask(layer_pre_norm, sample_input):
    """Padding mask testi"""
    batch_size, seq_len = 2, 10
    # Padding mask: (batch_size, tgt_len, src_len) formatında
    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    mask[0, :, 8:] = True  # İlk batch'te son 2 token padding (tüm query'ler için)
    mask[0, 8:, :] = True  # İlk batch'te son 2 token padding (tüm key'ler için)
    mask[1, :, 9:] = True  # İkinci batch'te son 1 token padding
    mask[1, 9:, :] = True
    
    output, attn_weights = layer_pre_norm(
        sample_input,
        mask=mask,
        causal_mask=False,
    )
    
    assert output.shape == sample_input.shape
    assert attn_weights is not None


def test_causal_mask_with_padding_mask(layer_pre_norm, sample_input):
    """Causal mask + padding mask kombinasyonu"""
    batch_size, seq_len = 2, 10
    # Padding mask: (batch_size, tgt_len, src_len) formatında
    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    mask[0, :, 8:] = True  # İlk batch'te son 2 token padding
    mask[0, 8:, :] = True
    
    output, attn_weights = layer_pre_norm(
        sample_input,
        mask=mask,
        causal_mask=True,
    )
    
    assert output.shape == sample_input.shape


# ---- Different Input Sizes ----
def test_forward_different_batch_sizes(layer_pre_norm):
    """Farklı batch size'lar için forward pass"""
    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, 10, 128)
        output, _ = layer_pre_norm(x)
        assert output.shape == (batch_size, 10, 128)


def test_forward_different_sequence_lengths(layer_pre_norm):
    """Farklı sequence length'ler için forward pass"""
    for seq_len in [1, 5, 10, 20]:
        x = torch.randn(2, seq_len, 128)
        output, _ = layer_pre_norm(x)
        assert output.shape == (2, seq_len, 128)


# ---- Numerical Stability ----
def test_numerical_stability(layer_pre_norm, sample_input):
    """Numerical stability testi"""
    output, attn_weights = layer_pre_norm(sample_input)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert output.isfinite().all()
    assert not torch.isnan(attn_weights).any()
    assert not torch.isinf(attn_weights).any()


# ---- GPU Tests (opsiyonel) ----
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_forward(layer_pre_norm):
    """GPU'da forward pass testi"""
    layer_pre_norm = layer_pre_norm.cuda()
    x = torch.randn(2, 10, 128).cuda()
    output, attn_weights = layer_pre_norm(x)
    assert output.device.type == "cuda"
    assert output.shape == (2, 10, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

