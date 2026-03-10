# -*- coding: utf-8 -*-
"""
CevahirNeuralNetwork V-2 Test Dosyası

Testler:
- Layer stacking testi
- Forward pass testi (yeni imza)
- Causal mask testi
- Geriye dönük uyumluluk testi (eski imza ile)
- Shape kontrolü
"""

import pytest
import torch
import logging

from src.neural_network import CevahirNeuralNetwork


# ---- Fixtures ----
@pytest.fixture
def neural_network_v2():
    """V-2 CevahirNeuralNetwork (yeni parametreler ile)"""
    return CevahirNeuralNetwork(
        learning_rate=1e-4,
        dropout=0.1,
        vocab_size=1000,
        embed_dim=128,
        seq_proj_dim=128,
        num_heads=8,
        num_layers=3,  # ✅ YENİ: Layer stacking
        ffn_dim=512,  # ✅ YENİ: FFN dimension
        pre_norm=True,  # ✅ YENİ: Pre-norm
        causal_mask=True,  # ✅ YENİ: Causal mask
        log_level=logging.WARNING,
    )


@pytest.fixture
def neural_network_v2_post_norm():
    """V-2 CevahirNeuralNetwork (post-norm ile)"""
    return CevahirNeuralNetwork(
        learning_rate=1e-4,
        dropout=0.1,
        vocab_size=1000,
        embed_dim=128,
        seq_proj_dim=128,
        num_heads=8,
        num_layers=3,
        ffn_dim=512,
        pre_norm=False,  # Post-norm
        causal_mask=True,
        log_level=logging.WARNING,
    )


@pytest.fixture
def sample_input():
    """Örnek input tensor (token IDs)"""
    return torch.randint(0, 1000, (2, 10))  # [B, T]


# ---- Initialization Tests ----
def test_initialization_v2(neural_network_v2):
    """V-2 initialization testi"""
    assert neural_network_v2.dil_katmani is not None
    assert neural_network_v2.layers is not None
    assert len(neural_network_v2.layers) == 3  # num_layers
    assert neural_network_v2.num_layers == 3
    assert neural_network_v2.causal_mask is True
    assert neural_network_v2.output_layer is not None
    assert neural_network_v2.memory_manager is not None


def test_initialization_default_params():
    """Default parametreler ile initialization testi"""
    model = CevahirNeuralNetwork(
        learning_rate=1e-4,
        dropout=0.1,
        vocab_size=1000,
        embed_dim=128,
        seq_proj_dim=128,
        num_heads=8,
        log_level=logging.WARNING,
    )
    # Default değerler kontrol edilmeli
    assert model.num_layers == 12  # Default
    assert model.causal_mask is True  # Default


# ---- Layer Stacking Tests ----
def test_layer_stacking(neural_network_v2):
    """Layer stacking testi"""
    assert len(neural_network_v2.layers) == 3
    # Her layer'ın doğru yapılandırıldığını kontrol et
    for i, layer in enumerate(neural_network_v2.layers):
        assert layer.embed_dim == 128
        assert layer.num_heads == 8
        assert layer.ffn_dim == 512
        assert layer.pre_norm is True


def test_different_num_layers():
    """Farklı num_layers değerleri için test"""
    for num_layers in [1, 2, 6, 12]:
        model = CevahirNeuralNetwork(
            learning_rate=1e-4,
            dropout=0.1,
            vocab_size=1000,
            embed_dim=128,
            seq_proj_dim=128,
            num_heads=8,
            num_layers=num_layers,
            log_level=logging.WARNING,
        )
        assert len(model.layers) == num_layers


# ---- Forward Pass Tests ----
def test_forward_pass_v2_shape(neural_network_v2, sample_input):
    """V-2 forward pass shape kontrolü"""
    output, attn_weights = neural_network_v2(sample_input)
    assert output.shape == (2, 10, 1000)  # [B, T, vocab_size]
    assert attn_weights is not None


def test_forward_pass_v2_new_signature(neural_network_v2, sample_input):
    """Yeni imza ile forward pass (mask, causal_mask parametreleri)"""
    # Yeni imza: forward(x, mask=None, causal_mask=None)
    output1, _ = neural_network_v2(sample_input, mask=None, causal_mask=True)
    output2, _ = neural_network_v2(sample_input, mask=None, causal_mask=False)
    
    assert output1.shape == output2.shape
    # Causal mask farklı output üretmeli
    assert not torch.allclose(output1, output2, atol=1e-5)


def test_forward_pass_backward_compatibility(neural_network_v2, sample_input):
    """Geriye dönük uyumluluk testi (eski imza: sadece x)"""
    # Eski imza: forward(x) - hala çalışmalı
    output, attn_weights = neural_network_v2(sample_input)
    assert output.shape == (2, 10, 1000)
    assert attn_weights is not None


# ---- Causal Mask Tests ----
def test_causal_mask_enabled(neural_network_v2, sample_input):
    """Causal mask aktif testi"""
    output_causal, _ = neural_network_v2(sample_input, causal_mask=True)
    output_no_causal, _ = neural_network_v2(sample_input, causal_mask=False)
    
    # Shape'ler aynı olmalı
    assert output_causal.shape == output_no_causal.shape
    
    # Ama değerler farklı olmalı (causal mask etkisi)
    assert not torch.allclose(output_causal, output_no_causal, atol=1e-5)


def test_causal_mask_default(neural_network_v2, sample_input):
    """Default causal mask testi (None ise self.causal_mask kullanılır)"""
    # Model'i eval moduna al (dropout'u devre dışı bırak)
    neural_network_v2.eval()
    
    # causal_mask=None -> self.causal_mask=True kullanılmalı
    with torch.no_grad():
        output1, _ = neural_network_v2(sample_input, causal_mask=None)
        output2, _ = neural_network_v2(sample_input, causal_mask=True)
    
    # Aynı olmalı (çünkü self.causal_mask=True ve eval modunda)
    assert torch.allclose(output1, output2, atol=1e-5)


# ---- Pre-norm vs Post-norm Tests ----
def test_pre_norm_vs_post_norm(neural_network_v2, neural_network_v2_post_norm, sample_input):
    """Pre-norm vs Post-norm testi"""
    output_pre, _ = neural_network_v2(sample_input)
    output_post, _ = neural_network_v2_post_norm(sample_input)
    
    # Shape'ler aynı olmalı
    assert output_pre.shape == output_post.shape
    
    # Ama değerler farklı olmalı (pre-norm vs post-norm farkı)
    assert not torch.allclose(output_pre, output_post, atol=1e-5)


# ---- Mask Tests ----
def test_padding_mask(neural_network_v2, sample_input):
    """Padding mask testi"""
    # Padding mask oluştur: (batch_size, tgt_len, src_len) formatında
    # True = engelle (padding), False = izin ver
    batch_size, seq_len = 2, 10
    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    mask[0, :, 8:] = True  # İlk batch'te son 2 token padding (tüm query'ler için)
    mask[0, 8:, :] = True  # İlk batch'te son 2 token padding (tüm key'ler için)
    
    output, attn_weights = neural_network_v2(sample_input, mask=mask)
    assert output.shape == (2, 10, 1000)


def test_causal_mask_with_padding_mask(neural_network_v2, sample_input):
    """Causal mask + padding mask kombinasyonu"""
    batch_size, seq_len = 2, 10
    # Padding mask: (batch_size, tgt_len, src_len) formatında
    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    mask[0, :, 8:] = True  # İlk batch'te son 2 token padding
    mask[0, 8:, :] = True
    
    output, attn_weights = neural_network_v2(
        sample_input,
        mask=mask,
        causal_mask=True,
    )
    assert output.shape == (2, 10, 1000)


# ---- Different Input Sizes ----
def test_forward_different_batch_sizes(neural_network_v2):
    """Farklı batch size'lar için forward pass"""
    for batch_size in [1, 2, 4]:
        x = torch.randint(0, 1000, (batch_size, 10))
        output, _ = neural_network_v2(x)
        assert output.shape == (batch_size, 10, 1000)


def test_forward_different_sequence_lengths(neural_network_v2):
    """Farklı sequence length'ler için forward pass"""
    for seq_len in [1, 5, 10, 20]:
        x = torch.randint(0, 1000, (2, seq_len))
        output, _ = neural_network_v2(x)
        assert output.shape == (2, seq_len, 1000)


# ---- Numerical Stability ----
def test_numerical_stability(neural_network_v2, sample_input):
    """Numerical stability testi"""
    output, attn_weights = neural_network_v2(sample_input)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert output.isfinite().all()


# ---- GPU Tests (opsiyonel) ----
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_forward(neural_network_v2):
    """GPU'da forward pass testi"""
    neural_network_v2 = neural_network_v2.cuda()
    x = torch.randint(0, 1000, (2, 10)).cuda()
    output, attn_weights = neural_network_v2(x)
    assert output.device.type == "cuda"
    assert output.shape == (2, 10, 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

