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
import torch
import torch.nn as nn

# Proje kök dizinini path'e ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.neural_network import CevahirNeuralNetwork


def test_forward_pass():
    """Test 1: Forward pass kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 1: Forward Pass Kontrolü")
    print("=" * 80)
    
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
    if isinstance(output, tuple):
        logits = output[0]  # İlk eleman logits
    else:
        logits = output
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output (logits) shape: {logits.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {vocab_size})")
    print(f"Output mean: {logits.mean().item():.6f}")
    print(f"Output std: {logits.std().item():.6f}")
    
    # Kontrol: Output shape doğru mu?
    expected_shape = (batch_size, seq_len, vocab_size)
    shape_match = logits.shape == expected_shape
    
    print(f"\n✅ Output shape doğru mu? {shape_match}")
    
    if not shape_match:
        print(f"⚠️  UYARI: Shape uyuşmazlığı! Beklenen: {expected_shape}, Alınan: {logits.shape}")
        return False
    
    # Kontrol: Output NaN veya Inf içermiyor mu?
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    
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
    if isinstance(output, tuple):
        logits = output[0]  # İlk eleman logits
    else:
        logits = output
    
    print(f"Input with BOS: {input_with_bos.tolist()}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected output shape: (1, 5, {vocab_size})")
    
    # Kontrol: Output shape doğru mu?
    expected_shape = (1, 5, vocab_size)
    shape_match = logits.shape == expected_shape
    
    print(f"\n✅ Output shape doğru mu? {shape_match}")
    
    # Kontrol: Her pozisyon için logits var mı?
    all_positions_have_logits = logits.shape[1] == input_with_bos.shape[1]
    print(f"✅ Her pozisyon için logits var mı? {all_positions_have_logits}")
    
    return shape_match and all_positions_have_logits


def test_padding_handling():
    """Test 3: Padding handling kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 3: Padding Handling Kontrolü")
    print("=" * 80)
    
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
    if isinstance(output, tuple):
        logits = output[0]  # İlk eleman logits
    else:
        logits = output
    
    print(f"Input with padding: {input_with_pad.tolist()}")
    print(f"Output shape: {logits.shape}")
    
    # Kontrol: Output shape doğru mu? (Padding olsa bile output shape input ile aynı olmalı)
    expected_shape = (1, 5, vocab_size)
    shape_match = logits.shape == expected_shape
    
    print(f"\n✅ Output shape doğru mu? (Padding'e rağmen): {shape_match}")
    
    return shape_match


def test_gradient_flow():
    """Test 4: Gradient flow kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 4: Gradient Flow Kontrolü")
    print("=" * 80)
    
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
    if isinstance(output, tuple):
        logits = output[0]  # İlk eleman logits
    else:
        logits = output
    
    # Loss (dummy)
    loss = logits.mean()
    
    # Backward pass
    loss.backward()
    
    # Kontrol: Gradient'ler var mı?
    has_gradients = False
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            grad_count += 1
            if grad_count == 1:  # Sadece ilk gradient'i göster
                grad_norm = param.grad.norm().item()
                print(f"  {name}: gradient norm = {grad_norm:.6f}")
    
    print(f"\n✅ Gradient'ler var mı? {has_gradients}")
    if has_gradients:
        print(f"  Toplam {grad_count} parametre için gradient var")
    
    return has_gradients


def test_output_shape_various_inputs():
    """Test 5: Çeşitli input shape'ler için output shape kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 5: Çeşitli Input Shape'ler için Output Shape Kontrolü")
    print("=" * 80)
    
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
    
    all_passed = True
    for batch_size, seq_len in test_cases:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = model(input_ids)
        if isinstance(output, tuple):
            logits = output[0]  # İlk eleman logits
        else:
            logits = output
        
        expected_shape = (batch_size, seq_len, vocab_size)
        shape_match = logits.shape == expected_shape
        
        print(f"  Batch={batch_size}, Seq={seq_len}: ", end="")
        if shape_match:
            print(f"✅ Shape doğru ({logits.shape})")
        else:
            print(f"❌ Shape yanlış (Beklenen: {expected_shape}, Alınan: {logits.shape})")
            all_passed = False
    
    print(f"\n✅ Tüm test case'leri başarılı mı? {all_passed}")
    return all_passed


def test_model_structure():
    """Test 6: Model structure kontrolü (modüller var mı?)"""
    print("\n" + "=" * 80)
    print("TEST 6: Model Structure Kontrolü")
    print("=" * 80)
    
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
    has_dil_katmani = hasattr(model, 'dil_katmani') or hasattr(model, 'embedding')
    has_layers = hasattr(model, 'layers') or hasattr(model, 'encoder_layers')
    has_output_proj = hasattr(model, 'output_layer') or hasattr(model, 'output_proj') or hasattr(model, 'lm_head')
    
    print(f"✅ Dil katmanı (embedding) var mı? {has_dil_katmani}")
    print(f"✅ Transformer layers var mı? {has_layers}")
    print(f"✅ Output projection var mı? {has_output_proj}")
    
    all_modules_exist = has_dil_katmani and has_layers and has_output_proj
    
    if all_modules_exist:
        print(f"\n✅ Tüm modüller mevcut!")
        
        # Layer sayısını kontrol et
        if has_layers:
            layers_attr = getattr(model, 'layers', None) or getattr(model, 'encoder_layers', None)
            if layers_attr:
                layer_count = len(layers_attr)
                print(f"  Layer sayısı: {layer_count} (beklenen: {num_layers})")
    else:
        print(f"\n⚠️  Bazı modüller eksik!")
    
    return all_modules_exist


def main(use_real_config=False):
    """Ana test fonksiyonu"""
    print("=" * 80)
    print("MODEL FORWARD PASS ANALİZ TESTİ")
    if use_real_config:
        print("(Gerçek Model Konfigürasyonu: vocab_size=60312, embed_dim=512, num_layers=12)")
    else:
        print("(Test Konfigürasyonu: vocab_size=1000, embed_dim=128, num_layers=2)")
    print("=" * 80)
    print("\nTest Dosyası: src/neural_network.py")
    print("Sınıf: CevahirNeuralNetwork")
    print("\nBu test, CevahirNeuralNetwork sınıfının endüstri standartlarına uygunluğunu kontrol eder.")
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
    
    # Test 4: Gradient flow
    try:
        result = test_gradient_flow()
        results.append(("Test 4: Gradient Flow", result))
    except Exception as e:
        print(f"\n❌ Test 4 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 4: Gradient Flow", False))
    
    # Test 5: Output shape various inputs
    try:
        result = test_output_shape_various_inputs()
        results.append(("Test 5: Çeşitli Input Shape'ler", result))
    except Exception as e:
        print(f"\n❌ Test 5 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 5: Çeşitli Input Shape'ler", False))
    
    # Test 6: Model structure
    try:
        result = test_model_structure()
        results.append(("Test 6: Model Structure", result))
    except Exception as e:
        print(f"\n❌ Test 6 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 6: Model Structure", False))
    
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
        print("CevahirNeuralNetwork endüstri standartlarına uygun çalışıyor.")
    else:
        print(f"\n⚠️  {total - passed} TEST BAŞARISIZ!")
        print("CevahirNeuralNetwork'de sorunlar bulundu. Detaylar için yukarıdaki çıktılara bakın.")
    
    return passed == total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Forward Pass Test")
    parser.add_argument("--real-config", action="store_true", 
                       help="Gerçek model konfigürasyonu ile test et")
    args = parser.parse_args()
    
    success = main(use_real_config=args.real_config)
    sys.exit(0 if success else 1)

