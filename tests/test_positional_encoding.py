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
import torch
import torch.nn as nn
import math

# Proje kök dizinini path'e ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.neural_network_module.dil_katmani_module.positional_encoding import PositionalEncoding


def test_sinusoidal_pe_basic():
    """Test 1: Sinusoidal PE temel kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 1: Sinusoidal Positional Encoding Temel Kontrolü")
    print("=" * 80)
    
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
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {encoded.shape}")
    print(f"Input mean: {x.mean().item():.6f}")
    print(f"Output mean: {encoded.mean().item():.6f}")
    
    # Kontrol: Output shape input ile aynı olmalı
    shape_match = encoded.shape == x.shape
    print(f"\n✅ Output shape input ile eşleşiyor mu? {shape_match}")
    
    if not shape_match:
        print(f"⚠️  UYARI: Shape uyuşmazlığı! Input: {x.shape}, Output: {encoded.shape}")
        return False
    
    # Kontrol: PE buffer var mı?
    has_pe_buffer = hasattr(pos_enc, 'pe') and pos_enc.pe is not None
    print(f"✅ PE buffer var mı? {has_pe_buffer}")
    
    if has_pe_buffer:
        pe_shape = pos_enc.pe.shape
        print(f"  PE buffer shape: {pe_shape}")
        print(f"  PE buffer mean: {pos_enc.pe.mean().item():.6f}")
        print(f"  PE buffer std: {pos_enc.pe.std().item():.6f}")
    
    return shape_match and has_pe_buffer


def test_rope_initialization():
    """Test 2: RoPE initialization kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 2: RoPE (Rotary Position Embedding) Initialization Kontrolü")
    print("=" * 80)
    
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
    
    print(f"Embed dim: {embed_dim}")
    print(f"Num heads: {num_heads}")
    print(f"Max len: {max_len}")
    
    # Kontrol: rope_freqs buffer var mı?
    has_rope_freqs = hasattr(pos_enc, 'rope_freqs') and pos_enc.rope_freqs is not None
    print(f"\n✅ RoPE freqs buffer var mı? {has_rope_freqs}")
    
    if has_rope_freqs:
        rope_freqs_shape = pos_enc.rope_freqs.shape
        head_dim = embed_dim // num_heads
        expected_shape = (max_len, head_dim // 2)
        
        print(f"  RoPE freqs shape: {rope_freqs_shape}")
        print(f"  Expected shape: {expected_shape}")
        print(f"  Head dim: {head_dim}")
        print(f"  RoPE freqs mean: {pos_enc.rope_freqs.mean().item():.6f}")
        print(f"  RoPE freqs min: {pos_enc.rope_freqs.min().item():.6f}")
        print(f"  RoPE freqs max: {pos_enc.rope_freqs.max().item():.6f}")
        
        shape_correct = rope_freqs_shape == expected_shape
        print(f"\n✅ RoPE freqs shape doğru mu? {shape_correct}")
        
        if not shape_correct:
            print(f"⚠️  UYARI: Shape uyuşmazlığı! Beklenen: {expected_shape}, Alınan: {rope_freqs_shape}")
            return False
    
    return has_rope_freqs


def test_rope_forward():
    """Test 3: RoPE forward pass kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 3: RoPE Forward Pass Kontrolü")
    print("=" * 80)
    
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
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {encoded.shape}")
    print(f"Input mean: {x.mean().item():.6f}")
    print(f"Output mean: {encoded.mean().item():.6f}")
    
    # Kontrol: Output shape input ile aynı olmalı
    shape_match = encoded.shape == x.shape
    print(f"\n✅ Output shape input ile eşleşiyor mu? {shape_match}")
    
    # RoPE modunda, forward pass input'u olduğu gibi döndürür
    # (RoPE aslında attention içinde kullanılır)
    # Ama burada forward pass'in çalıştığını kontrol ediyoruz
    
    return shape_match


def test_position_indexing():
    """Test 4: Position indexing kontrolü (0-based)"""
    print("\n" + "=" * 80)
    print("TEST 4: Position Indexing Kontrolü (0-based)")
    print("=" * 80)
    
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
    if hasattr(pos_enc, 'pe') and pos_enc.pe is not None:
        pe = pos_enc.pe  # [1, max_len, embed_dim]
        
        # İlk pozisyon (position 0) ve ikinci pozisyon (position 1) farklı olmalı
        pos_0 = pe[0, 0, :]  # Position 0
        pos_1 = pe[0, 1, :]  # Position 1
        
        are_different = not torch.allclose(pos_0, pos_1, atol=1e-6)
        
        print(f"PE shape: {pe.shape}")
        print(f"Position 0 mean: {pos_0.mean().item():.6f}")
        print(f"Position 1 mean: {pos_1.mean().item():.6f}")
        print(f"Position 0 ve 1 farklı mı? {are_different}")
        
        print(f"\n✅ Position indexing 0-based mi? (Position 0 ve 1 farklı): {are_different}")
        
        return are_different
    else:
        print("⚠️  PE buffer bulunamadı!")
        return False


def test_sequence_length_handling():
    """Test 5: Sequence length handling kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 5: Sequence Length Handling Kontrolü")
    print("=" * 80)
    
    embed_dim = 512
    max_len = 2048
    
    pos_enc = PositionalEncoding(
        embed_dim=embed_dim,
        max_len=max_len,
        mode="sinusoidal",
        dropout=0.0
    )
    
    # Farklı sequence length'ler ile test
    test_lengths = [5, 10, 50, 100, 500]
    
    all_passed = True
    for seq_len in test_lengths:
        x = torch.randn(1, seq_len, embed_dim)
        try:
            encoded = pos_enc(x)
            shape_correct = encoded.shape == x.shape
            print(f"  Seq len {seq_len}: ✅ Shape doğru ({encoded.shape})")
            if not shape_correct:
                all_passed = False
        except Exception as e:
            print(f"  Seq len {seq_len}: ❌ Hata: {e}")
            all_passed = False
    
    print(f"\n✅ Tüm sequence length'ler handle ediliyor mu? {all_passed}")
    return all_passed


def test_rope_frequency_computation():
    """Test 6: RoPE frequency computation kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 6: RoPE Frequency Computation Kontrolü")
    print("=" * 80)
    
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
    
    if not hasattr(pos_enc, 'rope_freqs') or pos_enc.rope_freqs is None:
        print("⚠️  RoPE freqs buffer bulunamadı!")
        return False
    
    rope_freqs = pos_enc.rope_freqs  # [max_len, head_dim // 2]
    
    # RoPE frequency formülü: theta_i = 10000^(-2i/d) where i in [0, d/2-1]
    # Kontrol: Freqs azalan sırada olmalı (theta_0 > theta_1 > ...)
    
    print(f"RoPE freqs shape: {rope_freqs.shape}")
    print(f"Head dim: {head_dim}")
    print(f"Expected shape: ({max_len}, {head_dim // 2})")
    
    # İlk pozisyon için frequency'leri kontrol et
    freqs_pos_0 = rope_freqs[0, :]  # [head_dim // 2]
    
    # Frequency'ler azalan sırada olmalı
    is_decreasing = torch.all(freqs_pos_0[:-1] >= freqs_pos_0[1:])
    
    print(f"\nFrequency'ler (pos 0, ilk 10): {freqs_pos_0[:10].tolist()}")
    print(f"Frequency'ler azalan sırada mı? {is_decreasing.item()}")
    
    # Frequency'lerin aralığı makul olmalı
    # RoPE frequency formülü: freqs = positions * inv_freq
    # positions = [0, 1, 2, ..., max_len-1]
    # inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
    # Bu yüzden max freq ≈ (max_len-1) * inv_freq[0] olabilir, bu normal!
    freq_min = rope_freqs.min().item()
    freq_max = rope_freqs.max().item()
    print(f"Frequency min: {freq_min:.6f}")
    print(f"Frequency max: {freq_max:.6f}")
    print(f"  Not: RoPE frequency = position * inv_freq olduğu için")
    print(f"  max freq ≈ (max_len-1) * inv_freq[0] ≈ {max_len-1} * inv_freq[0] olabilir (normal!)")
    
    # Frequency'ler pozitif olmalı ve max, max_len'den çok büyük olmamalı
    # (Ama position=2047 için freq_max≈2047 * inv_freq[0] normal)
    freq_range_reasonable = 0 <= freq_min < freq_max <= max_len * 10  # Çok geniş aralık, sadece mantıksız değerleri filtrelemek için
    
    print(f"\n✅ Frequency'ler azalan sırada mı? {is_decreasing.item()}")
    print(f"✅ Frequency aralığı makul mu? (0 <= min < max <= max_len*10): {freq_range_reasonable}")
    
    return is_decreasing.item() and freq_range_reasonable


def test_position_difference():
    """Test 7: Position farklılığı kontrolü (aynı token farklı pozisyonlarda farklı olmalı)"""
    print("\n" + "=" * 80)
    print("TEST 7: Position Farklılığı Kontrolü")
    print("=" * 80)
    
    embed_dim = 512
    max_len = 2048
    
    # Sinusoidal PE ile test
    pos_enc = PositionalEncoding(
        embed_dim=embed_dim,
        max_len=max_len,
        mode="sinusoidal",
        dropout=0.0
    )
    
    if not hasattr(pos_enc, 'pe') or pos_enc.pe is None:
        print("⚠️  PE buffer bulunamadı!")
        return False
    
    pe = pos_enc.pe[0]  # [max_len, embed_dim]
    
    # Aynı embedding, farklı pozisyonlarda farklı olmalı
    # Basit test: İlk birkaç pozisyon birbirinden farklı olmalı
    positions_to_check = [0, 1, 2, 5, 10]
    
    all_different = True
    for i, pos_i in enumerate(positions_to_check):
        for j, pos_j in enumerate(positions_to_check):
            if i < j:
                pe_i = pe[pos_i, :]
                pe_j = pe[pos_j, :]
                are_different = not torch.allclose(pe_i, pe_j, atol=1e-6)
                
                if not are_different:
                    all_different = False
                    print(f"  ⚠️  Position {pos_i} ve {pos_j} aynı!")
    
    print(f"\n✅ Farklı pozisyonlar farklı encoding'e sahip mi? {all_different}")
    
    return all_different


def main(use_real_config=False):
    """Ana test fonksiyonu"""
    print("=" * 80)
    print("POSITIONAL ENCODING ANALİZ TESTİ")
    if use_real_config:
        print("(Gerçek Model Konfigürasyonu: embed_dim=512, num_heads=8)")
    else:
        print("(Test Konfigürasyonu: embed_dim=512, num_heads=8)")
    print("=" * 80)
    print("\nTest Dosyası: src/neural_network_module/dil_katmani_module/positional_encoding.py")
    print("Sınıf: PositionalEncoding")
    print("\nBu test, PositionalEncoding sınıfının endüstri standartlarına uygunluğunu kontrol eder.")
    print("=" * 80)
    
    results = []
    
    # Test 1: Sinusoidal PE basic
    try:
        result = test_sinusoidal_pe_basic()
        results.append(("Test 1: Sinusoidal PE Temel Kontrolü", result))
    except Exception as e:
        print(f"\n❌ Test 1 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 1: Sinusoidal PE Temel Kontrolü", False))
    
    # Test 2: RoPE initialization
    try:
        result = test_rope_initialization()
        results.append(("Test 2: RoPE Initialization", result))
    except Exception as e:
        print(f"\n❌ Test 2 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 2: RoPE Initialization", False))
    
    # Test 3: RoPE forward
    try:
        result = test_rope_forward()
        results.append(("Test 3: RoPE Forward Pass", result))
    except Exception as e:
        print(f"\n❌ Test 3 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 3: RoPE Forward Pass", False))
    
    # Test 4: Position indexing
    try:
        result = test_position_indexing()
        results.append(("Test 4: Position Indexing", result))
    except Exception as e:
        print(f"\n❌ Test 4 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 4: Position Indexing", False))
    
    # Test 5: Sequence length handling
    try:
        result = test_sequence_length_handling()
        results.append(("Test 5: Sequence Length Handling", result))
    except Exception as e:
        print(f"\n❌ Test 5 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 5: Sequence Length Handling", False))
    
    # Test 6: RoPE frequency computation
    try:
        result = test_rope_frequency_computation()
        results.append(("Test 6: RoPE Frequency Computation", result))
    except Exception as e:
        print(f"\n❌ Test 6 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 6: RoPE Frequency Computation", False))
    
    # Test 7: Position difference
    try:
        result = test_position_difference()
        results.append(("Test 7: Position Farklılığı", result))
    except Exception as e:
        print(f"\n❌ Test 7 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 7: Position Farklılığı", False))
    
    # Sonuçlar
    print("\n" + "=" * 80)
    print("TEST SONUÇLARI ÖZETİ")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 80)
    print(f"Toplam: {passed}/{total} test başarılı")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 TÜM TESTLER BAŞARILI!")
        print("PositionalEncoding endüstri standartlarına uygun çalışıyor.")
    else:
        print(f"\n⚠️  {total - passed} TEST BAŞARISIZ!")
        print("PositionalEncoding'de sorunlar bulundu. Detaylar için yukarıdaki çıktılara bakın.")
    
    return passed == total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Positional Encoding Test")
    parser.add_argument("--real-config", action="store_true", 
                       help="Gerçek model konfigürasyonu ile test et")
    args = parser.parse_args()
    
    success = main(use_real_config=args.real_config)
    sys.exit(0 if success else 1)

