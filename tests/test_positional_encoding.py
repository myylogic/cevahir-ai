"""
Test: Positional Encoding Analizi
Dosya: src/neural_network_module/dil_katmani_module/positional_encoding.py
Sınıf/Metod: PositionalEncoding.__init__(), PositionalEncoding.forward(),
             PositionalEncoding._build_rope_freqs(), PositionalEncoding._build_sinusoidal_pe()

Bu test, PositionalEncoding sınıfının endüstri standartlarına uygunluğunu kontrol eder.

Endüstri Standartları:
- RoPE frequency computation doğru olmalı
- Position indexing doğru olmalı (0-based)
- Rotation matrix doğru oluşturulmalı
- Causal mask ile uyumlu olmalı

Test Senaryoları:
1. Position indexing kontrolü (0-based vs 1-based)
2. RoPE rotation kontrolü
3. Position invariance kontrolü (aynı token farklı pozisyonlarda farklı encoding)
4. Sequence length handling kontrolü
5. Sinusoidal PE kontrolü (eğer kullanılıyorsa)
"""

import sys
import os

import pytest
import torch

# Proje kök dizinini path'e ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.neural_network_module.dil_katmani_module.positional_encoding import PositionalEncoding


def test_sinusoidal_pe_basic():
    """Test 1: Sinusoidal PE temel kontrolü"""
    embed_dim = 512
    max_len = 2048

    pos_enc = PositionalEncoding(
        embed_dim=embed_dim,
        max_len=max_len,
        mode="sinusoidal",
        dropout=0.0
    )

    # Test input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    encoded = pos_enc(x)

    # Kontrol: Output shape input ile aynı olmalı
    assert encoded.shape == x.shape, (
        f"Shape uyuşmazlığı! Input: {tuple(x.shape)}, Output: {tuple(encoded.shape)}"
    )

    # Kontrol: PE buffer var mı?
    assert getattr(pos_enc, "pe", None) is not None, "Sinusoidal modda PE buffer bulunamadı!"


def test_rope_initialization():
    """Test 2: RoPE initialization kontrolü"""
    embed_dim = 512
    num_heads = 8
    max_len = 2048

    pos_enc = PositionalEncoding(
        embed_dim=embed_dim,
        max_len=max_len,
        mode="rope",
        num_heads=num_heads,
        dropout=0.0
    )

    # Kontrol: rope_freqs buffer var mı?
    assert getattr(pos_enc, "rope_freqs", None) is not None, "RoPE freqs buffer bulunamadı!"

    # Kontrol: rope_freqs shape [max_len, head_dim // 2] olmalı
    head_dim = embed_dim // num_heads
    expected_shape = (max_len, head_dim // 2)
    actual_shape = tuple(pos_enc.rope_freqs.shape)
    assert actual_shape == expected_shape, (
        f"RoPE freqs shape uyuşmazlığı! Beklenen: {expected_shape}, Alınan: {actual_shape} "
        f"(head_dim={head_dim})"
    )


def test_rope_forward():
    """Test 3: RoPE forward pass kontrolü"""
    embed_dim = 512
    num_heads = 8
    max_len = 2048

    pos_enc = PositionalEncoding(
        embed_dim=embed_dim,
        max_len=max_len,
        mode="rope",
        num_heads=num_heads,
        dropout=0.0
    )

    # Test input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    encoded = pos_enc(x)

    # RoPE modunda, forward pass input'u olduğu gibi döndürür
    # (RoPE aslında attention içinde kullanılır)
    # Ama burada forward pass'in çalıştığını kontrol ediyoruz

    # Kontrol: Output shape input ile aynı olmalı
    assert encoded.shape == x.shape, (
        f"Shape uyuşmazlığı! Input: {tuple(x.shape)}, Output: {tuple(encoded.shape)}"
    )


def test_position_indexing():
    """Test 4: Position indexing kontrolü (0-based)"""
    embed_dim = 512
    max_len = 2048

    # Sinusoidal PE ile test
    pos_enc = PositionalEncoding(
        embed_dim=embed_dim,
        max_len=max_len,
        mode="sinusoidal",
        dropout=0.0
    )

    # PE buffer'daki position indexing kontrolü
    assert getattr(pos_enc, "pe", None) is not None, "PE buffer bulunamadı!"
    pe = pos_enc.pe  # [1, max_len, embed_dim]

    # İlk pozisyon (position 0) ve ikinci pozisyon (position 1) farklı olmalı
    pos_0 = pe[0, 0, :]  # Position 0
    pos_1 = pe[0, 1, :]  # Position 1
    assert not torch.allclose(pos_0, pos_1, atol=1e-6), "Position 0 ve 1 aynı encoding'e sahip!"

    # "0 ve 1 farklı" kontrolü 1-based indexing'i ayırt edemez. 0-based ise ilk satır
    # pozisyon 0'dır: sin(0)=0 (çift kanallar) ve cos(0)=1 (tek kanallar).
    assert torch.allclose(pos_0[0::2], torch.zeros_like(pos_0[0::2]), atol=1e-6), (
        "PE[0] çift kanalları sin(0)=0 değil; PE 0-based değil! "
        f"(ilk 6 değer: {pos_0[:6].tolist()})"
    )
    assert torch.allclose(pos_0[1::2], torch.ones_like(pos_0[1::2]), atol=1e-6), (
        "PE[0] tek kanalları cos(0)=1 değil; PE 0-based değil! "
        f"(ilk 6 değer: {pos_0[:6].tolist()})"
    )

    # forward() da ilk token'a pozisyon 0'ı vermeli: sıfır girdi → PE[0:T]
    seq_len = 4
    encoded_zeros = pos_enc(torch.zeros(1, seq_len, embed_dim))
    assert torch.allclose(encoded_zeros[0], pe[0, :seq_len], atol=1e-6), (
        "forward() pozisyonları 0'dan başlatmıyor (sıfır girdi PE[0:T] ile eşleşmedi)"
    )


def test_sequence_length_handling():
    """Test 5: Sequence length handling kontrolü"""
    embed_dim = 512
    max_len = 2048

    pos_enc = PositionalEncoding(
        embed_dim=embed_dim,
        max_len=max_len,
        mode="sinusoidal",
        dropout=0.0
    )

    # Farklı sequence length'ler ile test (aynı modül ile ardışık)
    test_lengths = [5, 10, 50, 100, 500]

    for seq_len in test_lengths:
        x = torch.randn(1, seq_len, embed_dim)
        encoded = pos_enc(x)
        assert encoded.shape == x.shape, (
            f"Seq len {seq_len}: shape yanlış! Beklenen: {tuple(x.shape)}, "
            f"Alınan: {tuple(encoded.shape)}"
        )


def test_rope_frequency_computation():
    """Test 6: RoPE frequency computation kontrolü"""
    embed_dim = 512
    num_heads = 8
    max_len = 2048
    head_dim = embed_dim // num_heads  # 64

    pos_enc = PositionalEncoding(
        embed_dim=embed_dim,
        max_len=max_len,
        mode="rope",
        num_heads=num_heads,
        dropout=0.0
    )

    assert getattr(pos_enc, "rope_freqs", None) is not None, "RoPE freqs buffer bulunamadı!"
    rope_freqs = pos_enc.rope_freqs  # [max_len, head_dim // 2]

    # RoPE frequency formülü: theta_i = 10000^(-2i/d) where i in [0, d/2-1]
    # Buffer'daki değerler açıdır: rope_freqs[p, i] = p * theta_i
    #
    # Not: Eski test azalan sırayı pozisyon 0 satırında kontrol ediyordu; orada tüm
    # açılar 0 * theta_i = 0 olduğu için kontrol her koşulda (boş yere) geçiyordu.
    # Frekansların kendisi pozisyon 1 satırında görünür (rope_freqs[1] = theta).
    assert torch.count_nonzero(rope_freqs[0]) == 0, (
        "Pozisyon 0'da RoPE açıları sıfır olmalı (rotasyon yok)"
    )

    # Kontrol: Freqs azalan sırada olmalı (theta_0 > theta_1 > ...)
    freqs_pos_1 = rope_freqs[1, :]  # [head_dim // 2]
    assert torch.all(freqs_pos_1[:-1] >= freqs_pos_1[1:]), (
        f"Frequency'ler azalan sırada değil! (pos 1, ilk 10): {freqs_pos_1[:10].tolist()}"
    )

    # Kontrol: Açılar formülle birebir uyumlu olmalı
    expected_theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    expected_freqs = torch.arange(max_len, dtype=torch.float32).unsqueeze(1) * expected_theta.unsqueeze(0)
    torch.testing.assert_close(
        rope_freqs, expected_freqs,
        msg=lambda m: f"RoPE açıları theta_i = 10000^(-2i/d) formülüne uymuyor: {m}",
    )

    # Frequency'lerin aralığı makul olmalı
    # RoPE frequency = position * inv_freq olduğu için
    # max freq ≈ (max_len-1) * inv_freq[0] olabilir, bu normal!
    # Frequency'ler pozitif olmalı ve max, max_len'den çok büyük olmamalı
    # (Çok geniş aralık, sadece mantıksız değerleri filtrelemek için)
    freq_min = rope_freqs.min().item()
    freq_max = rope_freqs.max().item()
    assert 0 <= freq_min < freq_max <= max_len * 10, (
        f"Frequency aralığı makul değil (0 <= min < max <= max_len*10): "
        f"min={freq_min:.6f}, max={freq_max:.6f}"
    )


def test_position_difference():
    """Test 7: Position farklılığı kontrolü (aynı token farklı pozisyonlarda farklı olmalı)"""
    embed_dim = 512
    max_len = 2048

    # Sinusoidal PE ile test
    pos_enc = PositionalEncoding(
        embed_dim=embed_dim,
        max_len=max_len,
        mode="sinusoidal",
        dropout=0.0
    )

    assert getattr(pos_enc, "pe", None) is not None, "PE buffer bulunamadı!"
    pe = pos_enc.pe[0]  # [max_len, embed_dim]

    # Aynı embedding, farklı pozisyonlarda farklı olmalı
    # Basit test: İlk birkaç pozisyon birbirinden farklı olmalı
    positions_to_check = [0, 1, 2, 5, 10]

    identical_pairs = []
    for i, pos_i in enumerate(positions_to_check):
        for pos_j in positions_to_check[i + 1:]:
            if torch.allclose(pe[pos_i, :], pe[pos_j, :], atol=1e-6):
                identical_pairs.append((pos_i, pos_j))

    assert not identical_pairs, (
        f"Farklı pozisyonlar aynı encoding'e sahip: {identical_pairs}"
    )


if __name__ == "__main__":
    # Testleri pytest ile koştur (cache dizini repoya yazılmaz)
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider", *sys.argv[1:]]))
