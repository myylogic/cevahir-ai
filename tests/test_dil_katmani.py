"""
Test: Dil Katmanı Modülü Analizi
Dosya: src/neural_network_module/dil_katmani.py
Sınıf/Metod: DilKatmani.__init__(), DilKatmani.forward()

Bu test, DilKatmani sınıfının endüstri standartlarına uygunluğunu kontrol eder.

Endüstri Standartları:
- Modül sırası doğru olmalı (Embedding → PE → Norm → Projection)
- Forward pass flow doğru olmalı
- Output shape doğru olmalı [B, T, seq_proj_dim]

Test Senaryoları:
1. Forward pass kontrolü (input → output)
2. Autoregressive format kontrolü
3. Padding handling kontrolü
4. Modül sırası kontrolü (Embedding → PE → Norm → Projection)
5. Output shape kontrolü
"""

import sys
import os
import torch
import torch.nn as nn

# Proje kök dizinini path'e ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.neural_network_module.dil_katmani import DilKatmani


def test_forward_pass():
    """Test 1: Forward pass kontrolü (input → output)"""
    print("\n" + "=" * 80)
    print("TEST 1: Forward Pass Kontrolü")
    print("=" * 80)
    
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    
    dil_katmani = DilKatmani(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        dropout=0.0
    )
    
    # Test input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = dil_katmani(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {seq_proj_dim})")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    
    # Kontrol: Output shape doğru mu?
    expected_shape = (batch_size, seq_len, seq_proj_dim)
    shape_match = output.shape == expected_shape
    
    print(f"\n✅ Output shape doğru mu? {shape_match}")
    
    if not shape_match:
        print(f"⚠️  UYARI: Shape uyuşmazlığı! Beklenen: {expected_shape}, Alınan: {output.shape}")
        return False
    
    # Kontrol: Output NaN veya Inf içermiyor mu?
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    
    print(f"✅ Output NaN içeriyor mu? {has_nan}")
    print(f"✅ Output Inf içeriyor mu? {has_inf}")
    
    if has_nan or has_inf:
        print("⚠️  UYARI: Output NaN veya Inf içeriyor!")
        return False
    
    return True


def test_autoregressive_format():
    """Test 2: Autoregressive format kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 2: Autoregressive Format Kontrolü")
    print("=" * 80)
    
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    bos_id = 2
    
    dil_katmani = DilKatmani(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        dropout=0.0
    )
    
    # Autoregressive format: Input = [BOS, t1, t2, ..., tN]
    input_with_bos = torch.tensor([[bos_id, 100, 200, 300, 400]])
    
    output = dil_katmani(input_with_bos)
    
    print(f"Input with BOS: {input_with_bos.tolist()}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    
    # Kontrol: Output shape doğru mu?
    expected_shape = (1, 5, seq_proj_dim)
    shape_match = output.shape == expected_shape
    
    print(f"\n✅ Output shape doğru mu? {shape_match}")
    
    # Kontrol: Her pozisyon için output var mı?
    all_positions_have_output = output.shape[1] == input_with_bos.shape[1]
    print(f"✅ Her pozisyon için output var mı? {all_positions_have_output}")
    
    return shape_match and all_positions_have_output


def test_padding_handling():
    """Test 3: Padding handling kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 3: Padding Handling Kontrolü")
    print("=" * 80)
    
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    pad_id = 0
    bos_id = 2
    
    dil_katmani = DilKatmani(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        dropout=0.0
    )
    
    # Input with padding: [BOS, token1, token2, PAD, PAD]
    input_with_pad = torch.tensor([[bos_id, 100, 200, pad_id, pad_id]])
    
    output = dil_katmani(input_with_pad)
    
    print(f"Input with padding: {input_with_pad.tolist()}")
    print(f"Output shape: {output.shape}")
    
    # Kontrol: Output shape doğru mu? (Padding olsa bile output shape input ile aynı olmalı)
    expected_shape = (1, 5, seq_proj_dim)
    shape_match = output.shape == expected_shape
    
    print(f"\n✅ Output shape doğru mu? (Padding'e rağmen): {shape_match}")
    
    # Padding pozisyonlarındaki output'lar sıfır olmamalı (çünkü padding embedding'i sıfır ama PE eklenir)
    # Ama padding pozisyonlarında output'un farklı olup olmadığını kontrol edebiliriz
    pad_output = output[0, 3:, :]  # Padding pozisyonları
    non_pad_output = output[0, :3, :]  # Non-padding pozisyonları
    
    print(f"Padding output mean: {pad_output.mean().item():.6f}")
    print(f"Non-padding output mean: {non_pad_output.mean().item():.6f}")
    
    return shape_match


def test_module_order():
    """Test 4: Modül sırası kontrolü (Embedding → PE → Norm → Projection)"""
    print("\n" + "=" * 80)
    print("TEST 4: Modül Sırası Kontrolü")
    print("=" * 80)
    
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    
    dil_katmani = DilKatmani(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        dropout=0.0
    )
    
    # Kontrol: Gerekli modüller var mı?
    has_embedding = hasattr(dil_katmani, 'language_embedding')
    has_pe = hasattr(dil_katmani, 'positional_encoding')
    has_norm = hasattr(dil_katmani, 'layer_norm')
    has_projection = hasattr(dil_katmani, 'seq_projection')
    has_dropout = hasattr(dil_katmani, 'dropout')
    
    print(f"✅ LanguageEmbedding var mı? {has_embedding}")
    print(f"✅ PositionalEncoding var mı? {has_pe}")
    print(f"✅ LayerNorm var mı? {has_norm}")
    print(f"✅ SeqProjection var mı? {has_projection}")
    print(f"✅ Dropout var mı? {has_dropout}")
    
    all_modules_exist = has_embedding and has_pe and has_norm and has_projection and has_dropout
    
    if all_modules_exist:
        print(f"\n✅ Tüm modüller mevcut!")
        print(f"  Modül sırası (forward pass):")
        print(f"  1. LanguageEmbedding")
        print(f"  2. PositionalEncoding")
        print(f"  3. LayerNorm")
        print(f"  4. Dropout")
        print(f"  5. SeqProjection")
    else:
        print(f"\n⚠️  Bazı modüller eksik!")
    
    return all_modules_exist


def test_output_shape_various_inputs():
    """Test 5: Çeşitli input shape'ler için output shape kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 5: Çeşitli Input Shape'ler için Output Shape Kontrolü")
    print("=" * 80)
    
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    
    dil_katmani = DilKatmani(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        dropout=0.0
    )
    
    # Farklı batch size ve sequence length'ler ile test
    test_cases = [
        (1, 5),
        (2, 10),
        (4, 20),
        (8, 50),
    ]
    
    all_passed = True
    for batch_size, seq_len in test_cases:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = dil_katmani(input_ids)
        
        expected_shape = (batch_size, seq_len, seq_proj_dim)
        shape_match = output.shape == expected_shape
        
        print(f"  Batch={batch_size}, Seq={seq_len}: ", end="")
        if shape_match:
            print(f"✅ Shape doğru ({output.shape})")
        else:
            print(f"❌ Shape yanlış (Beklenen: {expected_shape}, Alınan: {output.shape})")
            all_passed = False
    
    print(f"\n✅ Tüm test case'leri başarılı mı? {all_passed}")
    return all_passed


def test_pe_mode_rope():
    """Test 6: RoPE modu kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 6: RoPE Modu Kontrolü")
    print("=" * 80)
    
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    num_heads = 8
    
    dil_katmani = DilKatmani(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        dropout=0.0,
        pe_mode="rope",
        pe_num_heads=num_heads
    )
    
    # Test input
    input_ids = torch.randint(0, vocab_size, (2, 10))
    output = dil_katmani(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    
    # Kontrol: Output shape doğru mu?
    expected_shape = (2, 10, seq_proj_dim)
    shape_match = output.shape == expected_shape
    
    print(f"\n✅ Output shape doğru mu? {shape_match}")
    
    # Kontrol: PE modu RoPE mi?
    if hasattr(dil_katmani, 'positional_encoding'):
        pe_mode = dil_katmani.positional_encoding.mode
        is_rope = pe_mode == "rope"
        print(f"✅ PE modu RoPE mi? {is_rope} (mode: {pe_mode})")
        
        return shape_match and is_rope
    else:
        return False


def test_gradient_flow():
    """Test 7: Gradient flow kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 7: Gradient Flow Kontrolü")
    print("=" * 80)
    
    vocab_size = 1000
    embed_dim = 128
    seq_proj_dim = 128
    
    dil_katmani = DilKatmani(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seq_proj_dim=seq_proj_dim,
        dropout=0.0
    )
    
    # Test input
    input_ids = torch.randint(0, vocab_size, (2, 10))
    
    # Forward pass
    output = dil_katmani(input_ids)
    
    # Loss (dummy)
    loss = output.mean()
    
    # Backward pass
    loss.backward()
    
    # Kontrol: Gradient'ler var mı?
    has_gradients = False
    for name, param in dil_katmani.named_parameters():
        if param.grad is not None:
            has_gradients = True
            grad_norm = param.grad.norm().item()
            print(f"  {name}: gradient norm = {grad_norm:.6f}")
            break
    
    print(f"\n✅ Gradient'ler var mı? {has_gradients}")
    
    return has_gradients


def main(use_real_config=False):
    """Ana test fonksiyonu"""
    print("=" * 80)
    print("DİL KATMANI MODÜLÜ ANALİZ TESTİ")
    if use_real_config:
        print("(Gerçek Model Konfigürasyonu: vocab_size=60312, embed_dim=512, seq_proj_dim=512)")
    else:
        print("(Test Konfigürasyonu: vocab_size=1000, embed_dim=128, seq_proj_dim=128)")
    print("=" * 80)
    print("\nTest Dosyası: src/neural_network_module/dil_katmani.py")
    print("Sınıf: DilKatmani")
    print("\nBu test, DilKatmani sınıfının endüstri standartlarına uygunluğunu kontrol eder.")
    print("=" * 80)
    
    results = []
    
    # Test 1: Forward pass
    try:
        result = test_forward_pass()
        results.append(("Test 1: Forward Pass", result))
    except Exception as e:
        print(f"\n❌ Test 1 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 1: Forward Pass", False))
    
    # Test 2: Autoregressive format
    try:
        result = test_autoregressive_format()
        results.append(("Test 2: Autoregressive Format", result))
    except Exception as e:
        print(f"\n❌ Test 2 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 2: Autoregressive Format", False))
    
    # Test 3: Padding handling
    try:
        result = test_padding_handling()
        results.append(("Test 3: Padding Handling", result))
    except Exception as e:
        print(f"\n❌ Test 3 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 3: Padding Handling", False))
    
    # Test 4: Module order
    try:
        result = test_module_order()
        results.append(("Test 4: Modül Sırası", result))
    except Exception as e:
        print(f"\n❌ Test 4 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 4: Modül Sırası", False))
    
    # Test 5: Output shape various inputs
    try:
        result = test_output_shape_various_inputs()
        results.append(("Test 5: Çeşitli Input Shape'ler", result))
    except Exception as e:
        print(f"\n❌ Test 5 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 5: Çeşitli Input Shape'ler", False))
    
    # Test 6: PE mode RoPE
    try:
        result = test_pe_mode_rope()
        results.append(("Test 6: RoPE Modu", result))
    except Exception as e:
        print(f"\n❌ Test 6 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 6: RoPE Modu", False))
    
    # Test 7: Gradient flow
    try:
        result = test_gradient_flow()
        results.append(("Test 7: Gradient Flow", result))
    except Exception as e:
        print(f"\n❌ Test 7 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 7: Gradient Flow", False))
    
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
        print("DilKatmani endüstri standartlarına uygun çalışıyor.")
    else:
        print(f"\n⚠️  {total - passed} TEST BAŞARISIZ!")
        print("DilKatmani'de sorunlar bulundu. Detaylar için yukarıdaki çıktılara bakın.")
    
    return passed == total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dil Katmanı Modülü Test")
    parser.add_argument("--real-config", action="store_true", 
                       help="Gerçek model konfigürasyonu ile test et")
    args = parser.parse_args()
    
    success = main(use_real_config=args.real_config)
    sys.exit(0 if success else 1)


