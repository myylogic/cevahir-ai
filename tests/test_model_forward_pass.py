"""
Test: Model Forward Pass Analizi
Dosya: src/neural_network.py
Sınıf/Metod: CevahirNeuralNetwork.__init__(), CevahirNeuralNetwork.forward()

Bu test, CevahirNeuralNetwork sınıfının endüstri standartlarına uygunluğunu kontrol eder.

Endüstri Standartları:
- Forward pass flow doğru olmalı (Input → Embedding → PE → Transformer → Output)
- Output logits shape doğru olmalı [B, T, vocab_size]
- Padding handling doğru olmalı
- Autoregressive format doğru olmalı

Test Senaryoları:
1. Forward pass kontrolü
2. Autoregressive format kontrolü
3. Padding handling kontrolü
4. Gradient flow kontrolü
5. Output shape kontrolü
"""

import sys
import os

import pytest
import torch

# Proje kök dizinini path'e ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.neural_network import CevahirNeuralNetwork


def test_forward_pass():
    """Test 1: Forward pass kontrolü"""
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    num_heads = 8
    num_layers = 2  # Test için az layer
    ffn_dim = 512

    model = CevahirNeuralNetwork(
        learning_rate=0.0001,
        dropout=0.1,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        pre_norm=True,
        causal_mask=True,
        pe_mode="sinusoidal",  # Test için sinusoidal kullan
        use_gradient_checkpointing=False,  # Test için kapat
        tie_weights=False,  # Test için kapat
        use_rmsnorm=False,
        use_swiglu=False,
        use_kv_cache=False,
    )

    # Test input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass (model tuple döndürür: (logits, attn_weights) veya (logits, attn_weights, kv_cache))
    output = model(input_ids)
    logits = output[0] if isinstance(output, tuple) else output  # İlk eleman logits

    # Kontrol: Output shape doğru mu?
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, (
        f"Shape uyuşmazlığı! Beklenen: {expected_shape}, Alınan: {tuple(logits.shape)}"
    )

    # Kontrol: Output NaN veya Inf içermiyor mu?
    assert not torch.isnan(logits).any(), "Output (logits) NaN içeriyor!"
    assert not torch.isinf(logits).any(), "Output (logits) Inf içeriyor!"


def test_autoregressive_format():
    """Test 2: Autoregressive format kontrolü"""
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    num_heads = 8
    num_layers = 2
    ffn_dim = 512
    bos_id = 2

    model = CevahirNeuralNetwork(
        learning_rate=0.0001,
        dropout=0.0,  # Test için dropout kapat
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        pre_norm=True,
        causal_mask=True,
        pe_mode="sinusoidal",
        use_gradient_checkpointing=False,
        tie_weights=False,
        use_rmsnorm=False,
        use_swiglu=False,
        use_kv_cache=False,
    )

    # Autoregressive format: Input = [BOS, t1, t2, ..., tN]
    input_with_bos = torch.tensor([[bos_id, 100, 200, 300, 400]])

    output = model(input_with_bos)
    logits = output[0] if isinstance(output, tuple) else output  # İlk eleman logits

    # Kontrol: Output shape doğru mu?
    expected_shape = (1, 5, vocab_size)
    assert logits.shape == expected_shape, (
        f"Shape uyuşmazlığı! Input: {input_with_bos.tolist()}, "
        f"Beklenen: {expected_shape}, Alınan: {tuple(logits.shape)}"
    )

    # Kontrol: Her pozisyon için logits var mı?
    assert logits.shape[1] == input_with_bos.shape[1], (
        f"Her pozisyon için logits yok! Input uzunluğu: {input_with_bos.shape[1]}, "
        f"logits pozisyon sayısı: {logits.shape[1]}"
    )


def test_padding_handling():
    """Test 3: Padding handling kontrolü"""
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    num_heads = 8
    num_layers = 2
    ffn_dim = 512
    pad_id = 0
    bos_id = 2

    model = CevahirNeuralNetwork(
        learning_rate=0.0001,
        dropout=0.0,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        pre_norm=True,
        causal_mask=True,
        pe_mode="sinusoidal",
        use_gradient_checkpointing=False,
        tie_weights=False,
        use_rmsnorm=False,
        use_swiglu=False,
        use_kv_cache=False,
    )

    # Input with padding: [BOS, token1, token2, PAD, PAD]
    input_with_pad = torch.tensor([[bos_id, 100, 200, pad_id, pad_id]])

    output = model(input_with_pad)
    logits = output[0] if isinstance(output, tuple) else output  # İlk eleman logits

    # Kontrol: Output shape doğru mu? (Padding olsa bile output shape input ile aynı olmalı)
    expected_shape = (1, 5, vocab_size)
    assert logits.shape == expected_shape, (
        f"Padding'e rağmen output shape yanlış! Input: {input_with_pad.tolist()}, "
        f"Beklenen: {expected_shape}, Alınan: {tuple(logits.shape)}"
    )


def test_gradient_flow():
    """Test 4: Gradient flow kontrolü"""
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    num_heads = 8
    num_layers = 2
    ffn_dim = 512

    model = CevahirNeuralNetwork(
        learning_rate=0.0001,
        dropout=0.0,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        pre_norm=True,
        causal_mask=True,
        pe_mode="sinusoidal",
        use_gradient_checkpointing=False,  # Gradient flow test için kapat
        tie_weights=False,
        use_rmsnorm=False,
        use_swiglu=False,
        use_kv_cache=False,
    )

    # Test input
    input_ids = torch.randint(0, vocab_size, (2, 10))

    # Forward pass
    output = model(input_ids)
    logits = output[0] if isinstance(output, tuple) else output  # İlk eleman logits

    # Loss (dummy)
    loss = logits.mean()

    # Backward pass
    loss.backward()

    # Kontrol: Gradient'ler var mı?
    params_with_grad = [name for name, param in model.named_parameters() if param.grad is not None]
    assert params_with_grad, "Hiçbir parametre için gradient oluşmadı!"


def test_output_shape_various_inputs():
    """Test 5: Çeşitli input shape'ler için output shape kontrolü"""
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    num_heads = 8
    num_layers = 2
    ffn_dim = 512

    model = CevahirNeuralNetwork(
        learning_rate=0.0001,
        dropout=0.0,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        pre_norm=True,
        causal_mask=True,
        pe_mode="sinusoidal",
        use_gradient_checkpointing=False,
        tie_weights=False,
        use_rmsnorm=False,
        use_swiglu=False,
        use_kv_cache=False,
    )

    # Farklı batch size ve sequence length'ler ile test
    test_cases = [
        (1, 5),
        (2, 10),
        (4, 20),
    ]

    for batch_size, seq_len in test_cases:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output  # İlk eleman logits

        expected_shape = (batch_size, seq_len, vocab_size)
        assert logits.shape == expected_shape, (
            f"Batch={batch_size}, Seq={seq_len}: shape yanlış! "
            f"Beklenen: {expected_shape}, Alınan: {tuple(logits.shape)}"
        )


def test_model_structure():
    """Test 6: Model structure kontrolü (modüller var mı?)"""
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    num_heads = 8
    num_layers = 2
    ffn_dim = 512

    model = CevahirNeuralNetwork(
        learning_rate=0.0001,
        dropout=0.1,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        pre_norm=True,
        causal_mask=True,
        pe_mode="sinusoidal",
        use_gradient_checkpointing=False,
        tie_weights=False,
        use_rmsnorm=False,
        use_swiglu=False,
        use_kv_cache=False,
    )

    # Kontrol: Gerekli modüller var mı?
    modules = {
        "Dil katmanı (embedding)": hasattr(model, 'dil_katmani') or hasattr(model, 'embedding'),
        "Transformer layers": hasattr(model, 'layers') or hasattr(model, 'encoder_layers'),
        "Output projection": (
            hasattr(model, 'output_layer') or hasattr(model, 'output_proj') or hasattr(model, 'lm_head')
        ),
    }
    missing = [name for name, exists in modules.items() if not exists]
    assert not missing, f"Bazı modüller eksik: {missing}"

    # Layer sayısını kontrol et
    layers_attr = getattr(model, 'layers', None) or getattr(model, 'encoder_layers', None)
    layer_count = len(layers_attr) if layers_attr is not None else 0
    assert layer_count == num_layers, (
        f"Layer sayısı yanlış! Beklenen: {num_layers}, Alınan: {layer_count}"
    )


if __name__ == "__main__":
    # Testleri pytest ile koştur (cache dizini repoya yazılmaz)
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider", *sys.argv[1:]]))
