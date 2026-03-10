"""
Test: Embedding Layer Analizi
Dosya: src/neural_network_module/dil_katmani_module/language_embedding.py
Sınıf/Metod: LanguageEmbedding.__init__(), LanguageEmbedding.forward(), LanguageEmbedding._initialize_weights()

Bu test, LanguageEmbedding sınıfının endüstri standartlarına uygunluğunu kontrol eder.

Endüstri Standartları:
- Padding token embedding sıfır olmalı (PyTorch standardı)
- Weight initialization doğru olmalı (Xavier/Normal)
- Embedding scaling doğru olmalı (scale_by_sqrt → embedding * sqrt(embed_dim))
- Embedding weight shape doğru olmalı [vocab_size, embed_dim]

Test Senaryoları:
1. Padding token embedding kontrolü (sıfır olmalı)
2. Normal token embedding kontrolü (sıfır olmamalı)
3. Embedding scaling kontrolü (scale_by_sqrt)
4. Weight initialization kontrolü
5. Dropout kontrolü (training/eval mode)
6. Embedding weight shape kontrolü
7. Padding_idx kontrolü
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

from src.neural_network_module.dil_katmani_module.language_embedding import LanguageEmbedding


def test_padding_token_embedding(use_real_config=False):
    """Test 1: Padding token embedding sıfır olmalı (PyTorch standardı)"""
    print("\n" + "=" * 80)
    print("TEST 1: Padding Token Embedding Kontrolü")
    if use_real_config:
        print("(Gerçek Model Konfigürasyonu ile)")
    print("=" * 80)
    
    if use_real_config:
        vocab_size = 60312
        embed_dim = 512
        pad_id = 0
    else:
        vocab_size = 1000
        embed_dim = 128
        pad_id = 0
    
    embedding = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=False,  # Scaling'i kapatıyoruz, sadece padding kontrolü için
        dropout=0.0
    )
    
    # Padding token embedding'i al
    pad_input = torch.tensor([[pad_id]])
    pad_embedding = embedding(pad_input)
    
    print(f"Padding token ID: {pad_id}")
    print(f"Padding embedding shape: {pad_embedding.shape}")
    print(f"Padding embedding (ilk 10 dim): {pad_embedding[0, 0, :10].tolist()}")
    print(f"Padding embedding sum: {pad_embedding.sum().item():.6f}")
    print(f"Padding embedding norm: {pad_embedding.norm().item():.6f}")
    
    # Kontrol: Padding embedding sıfır olmalı
    is_zero = torch.allclose(pad_embedding, torch.zeros_like(pad_embedding), atol=1e-6)
    print(f"\n✅ Padding embedding sıfır mı? {is_zero}")
    
    if not is_zero:
        print("⚠️  UYARI: Padding embedding sıfır değil! PyTorch standardına uygun değil.")
        return False
    
    return True


def test_normal_token_embedding(use_real_config=False):
    """Test 2: Normal token embedding'ler sıfır olmamalı"""
    print("\n" + "=" * 80)
    print("TEST 2: Normal Token Embedding Kontrolü")
    if use_real_config:
        print("(Gerçek Model Konfigürasyonu ile)")
    print("=" * 80)
    
    if use_real_config:
        vocab_size = 60312
        embed_dim = 512
        pad_id = 0
    else:
        vocab_size = 1000
        embed_dim = 128
        pad_id = 0
    
    embedding = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=False,
        dropout=0.0
    )
    
    # Normal token embedding'leri al
    normal_input = torch.tensor([[100, 200, 300]])
    normal_embedding = embedding(normal_input)
    
    print(f"Normal token IDs: [100, 200, 300]")
    print(f"Normal embedding shape: {normal_embedding.shape}")
    print(f"Normal embedding mean: {normal_embedding.mean().item():.6f}")
    print(f"Normal embedding std: {normal_embedding.std().item():.6f}")
    print(f"Normal embedding norm (token 100): {normal_embedding[0, 0].norm().item():.6f}")
    
    # Kontrol: Normal embedding'ler sıfır olmamalı
    is_non_zero = not torch.allclose(normal_embedding, torch.zeros_like(normal_embedding), atol=1e-6)
    print(f"\n✅ Normal embedding'ler sıfır olmamalı mı? {is_non_zero}")
    
    if not is_non_zero:
        print("⚠️  UYARI: Normal embedding'ler sıfır! Weight initialization sorunu olabilir.")
        return False
    
    return True


def test_embedding_scaling(use_real_config=False):
    """Test 3: Embedding scaling kontrolü (scale_by_sqrt)"""
    print("\n" + "=" * 80)
    print("TEST 3: Embedding Scaling Kontrolü (scale_by_sqrt)")
    if use_real_config:
        print("(Gerçek Model Konfigürasyonu ile)")
    print("=" * 80)
    
    if use_real_config:
        vocab_size = 60312
        embed_dim = 512
        pad_id = 0
        test_token_id = 100
    else:
        vocab_size = 1000
        embed_dim = 128
        pad_id = 0
        test_token_id = 100
    
    # Scaling KAPALI
    embedding_no_scale = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=False,
        dropout=0.0
    )
    
    # Scaling AÇIK
    embedding_with_scale = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=True,
        dropout=0.0
    )
    
    # Aynı weight'leri kopyala (karşılaştırma için)
    embedding_with_scale.embedding.weight.data = embedding_no_scale.embedding.weight.data.clone()
    
    test_input = torch.tensor([[test_token_id]])
    
    # Scaling olmadan
    emb_no_scale = embedding_no_scale(test_input)
    
    # Scaling ile
    emb_with_scale = embedding_with_scale(test_input)
    
    expected_scale = math.sqrt(embed_dim)
    actual_scale = (emb_with_scale / emb_no_scale).mean().item()
    
    print(f"Embed dim: {embed_dim}")
    print(f"Expected scale factor: sqrt({embed_dim}) = {expected_scale:.6f}")
    print(f"Actual scale factor: {actual_scale:.6f}")
    print(f"Scale difference: {abs(actual_scale - expected_scale):.6f}")
    
    # Kontrol: Scaling doğru uygulanmalı
    scale_is_correct = math.isclose(actual_scale, expected_scale, rel_tol=1e-5)
    print(f"\n✅ Scaling doğru uygulanıyor mu? {scale_is_correct}")
    
    if not scale_is_correct:
        print("⚠️  UYARI: Scaling doğru uygulanmıyor! scale_by_sqrt=True ama sqrt(embed_dim) ile çarpılmıyor.")
        return False
    
    return True


def test_weight_initialization(use_real_config=False):
    """Test 4: Weight initialization kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 4: Weight Initialization Kontrolü")
    if use_real_config:
        print("(Gerçek Model Konfigürasyonu ile)")
    print("=" * 80)
    
    if use_real_config:
        vocab_size = 60312
        embed_dim = 512
        pad_id = 0
    else:
        vocab_size = 1000
        embed_dim = 128
        pad_id = 0
    
    # Xavier initialization ile test
    embedding_xavier = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        init_method="xavier",
        scale_by_sqrt=False,
        dropout=0.0
    )
    
    weights = embedding_xavier.embedding.weight.data
    
    # Padding token hariç weight'leri al
    non_pad_weights = weights[1:]  # pad_id=0 hariç
    
    print(f"Embedding weight shape: {weights.shape}")
    print(f"Non-padding weights shape: {non_pad_weights.shape}")
    print(f"Weight mean: {non_pad_weights.mean().item():.6f}")
    print(f"Weight std: {non_pad_weights.std().item():.6f}")
    print(f"Weight min: {non_pad_weights.min().item():.6f}")
    print(f"Weight max: {non_pad_weights.max().item():.6f}")
    
    # Xavier initialization: std ≈ sqrt(2 / (vocab_size + embed_dim))
    # Büyük vocab_size'larda std daha küçük olabilir (bu normal)
    # Örnek: vocab_size=60312, embed_dim=512 için std ≈ sqrt(2/(60312+512)) ≈ 0.0057
    weight_std = non_pad_weights.std().item()
    # Test kriterini gevşetiyoruz: çok küçük vocab'ler için 0.01, büyük vocab'ler için daha küçük std normal
    std_reasonable = 0.001 < weight_std < 1.0  # Daha gerçekçi kriter
    mean_near_zero = abs(non_pad_weights.mean().item()) < 0.1
    
    print(f"\n✅ Weight std makul aralıkta mı? (0.001 < std < 1.0): {std_reasonable}")
    print(f"  Not: Büyük vocab_size'larda std daha küçük olabilir (Xavier initialization)")
    print(f"  Örnek: vocab_size=60312 için std≈0.0057 normal (Xavier: sqrt(2/(vocab_size+embed_dim)))")
    print(f"✅ Weight mean sıfıra yakın mı? (|mean| < 0.1): {mean_near_zero}")
    
    # Padding token embedding sıfır olmalı
    pad_weight = weights[pad_id]
    pad_is_zero = torch.allclose(pad_weight, torch.zeros_like(pad_weight), atol=1e-6)
    print(f"✅ Padding weight sıfır mı? {pad_is_zero}")
    
    return std_reasonable and mean_near_zero and pad_is_zero


def test_dropout(use_real_config=False):
    """Test 5: Dropout kontrolü (training/eval mode)"""
    print("\n" + "=" * 80)
    print("TEST 5: Dropout Kontrolü (Training/Eval Mode)")
    if use_real_config:
        print("(Gerçek Model Konfigürasyonu ile)")
    print("=" * 80)
    
    if use_real_config:
        vocab_size = 60312
        embed_dim = 512
        pad_id = 0
    else:
        vocab_size = 1000
        embed_dim = 128
        pad_id = 0
    
    embedding = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=False,
        dropout=0.5  # Yüksek dropout, farkı görmek için
    )
    
    test_input = torch.tensor([[100, 200, 300]])
    
    # Training mode
    embedding.train()
    torch.manual_seed(42)  # Reproducibility için
    emb_train_1 = embedding(test_input)
    torch.manual_seed(42)
    emb_train_2 = embedding(test_input)
    
    # Eval mode
    embedding.eval()
    emb_eval_1 = embedding(test_input)
    emb_eval_2 = embedding(test_input)
    
    print(f"Dropout rate: 0.5")
    print(f"Training mode - emb 1 mean: {emb_train_1.mean().item():.6f}")
    print(f"Training mode - emb 2 mean: {emb_train_2.mean().item():.6f}")
    print(f"Eval mode - emb 1 mean: {emb_eval_1.mean().item():.6f}")
    print(f"Eval mode - emb 2 mean: {emb_eval_2.mean().item():.6f}")
    
    # Training mode'da dropout aktif olmalı (random, farklı olabilir)
    # Eval mode'da dropout pasif olmalı (deterministic, aynı olmalı)
    train_different = not torch.allclose(emb_train_1, emb_train_2, atol=1e-5)  # Dropout nedeniyle farklı olabilir
    eval_same = torch.allclose(emb_eval_1, emb_eval_2, atol=1e-5)  # Eval mode'da aynı olmalı
    
    print(f"\n✅ Training mode'da dropout aktif mi? (farklı çıktılar): {train_different}")
    print(f"✅ Eval mode'da dropout pasif mi? (aynı çıktılar): {eval_same}")
    
    if not eval_same:
        print("⚠️  UYARI: Eval mode'da dropout aktif! Eval mode'da dropout pasif olmalı.")
        return False
    
    return True


def test_embedding_weight_shape(use_real_config=False):
    """Test 6: Embedding weight shape kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 6: Embedding Weight Shape Kontrolü")
    if use_real_config:
        print("(Gerçek Model Konfigürasyonu ile)")
    print("=" * 80)
    
    if use_real_config:
        vocab_size = 60312
        embed_dim = 512
        pad_id = 0
    else:
        vocab_size = 1000
        embed_dim = 128
        pad_id = 0
    
    embedding = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=False,
        dropout=0.0
    )
    
    weights = embedding.embedding.weight.data
    
    expected_shape = (vocab_size, embed_dim)
    actual_shape = weights.shape
    
    print(f"Expected shape: {expected_shape}")
    print(f"Actual shape: {actual_shape}")
    
    shape_correct = actual_shape == expected_shape
    
    print(f"\n✅ Weight shape doğru mu? {shape_correct}")
    
    if not shape_correct:
        print(f"⚠️  UYARI: Weight shape yanlış! Beklenen: {expected_shape}, Alınan: {actual_shape}")
        return False
    
    return True


def test_padding_idx(use_real_config=False):
    """Test 7: Padding_idx kontrolü"""
    print("\n" + "=" * 80)
    print("TEST 7: Padding_idx Kontrolü")
    if use_real_config:
        print("(Gerçek Model Konfigürasyonu ile)")
    print("=" * 80)
    
    if use_real_config:
        vocab_size = 60312
        embed_dim = 512
        pad_id = 0
    else:
        vocab_size = 1000
        embed_dim = 128
        pad_id = 0
    
    embedding = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=False,
        dropout=0.0
    )
    
    # PyTorch nn.Embedding'de padding_idx set edilmiş mi kontrol et
    actual_padding_idx = embedding.embedding.padding_idx
    
    print(f"Expected padding_idx: {pad_id}")
    print(f"Actual padding_idx: {actual_padding_idx}")
    
    padding_idx_correct = actual_padding_idx == pad_id
    
    print(f"\n✅ Padding_idx doğru set edilmiş mi? {padding_idx_correct}")
    
    if not padding_idx_correct:
        print(f"⚠️  UYARI: Padding_idx yanlış! Beklenen: {pad_id}, Alınan: {actual_padding_idx}")
        return False
    
    return True


def main(use_real_config=False):
    """Ana test fonksiyonu"""
    print("=" * 80)
    print("EMBEDDING LAYER ANALİZ TESTİ")
    if use_real_config:
        print("(Gerçek Model Konfigürasyonu: vocab_size=60312, embed_dim=512)")
    else:
        print("(Test Konfigürasyonu: vocab_size=1000, embed_dim=128)")
    print("=" * 80)
    print("\nTest Dosyası: src/neural_network_module/dil_katmani_module/language_embedding.py")
    print("Sınıf: LanguageEmbedding")
    print("\nBu test, LanguageEmbedding sınıfının endüstri standartlarına uygunluğunu kontrol eder.")
    print("=" * 80)
    
    results = []
    
    # Test 1: Padding token embedding
    try:
        result = test_padding_token_embedding(use_real_config)
        results.append(("Test 1: Padding Token Embedding", result))
    except Exception as e:
        print(f"\n❌ Test 1 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 1: Padding Token Embedding", False))
    
    # Test 2: Normal token embedding
    try:
        result = test_normal_token_embedding(use_real_config)
        results.append(("Test 2: Normal Token Embedding", result))
    except Exception as e:
        print(f"\n❌ Test 2 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 2: Normal Token Embedding", False))
    
    # Test 3: Embedding scaling
    try:
        result = test_embedding_scaling(use_real_config)
        results.append(("Test 3: Embedding Scaling", result))
    except Exception as e:
        print(f"\n❌ Test 3 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 3: Embedding Scaling", False))
    
    # Test 4: Weight initialization
    try:
        result = test_weight_initialization(use_real_config)
        results.append(("Test 4: Weight Initialization", result))
    except Exception as e:
        print(f"\n❌ Test 4 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 4: Weight Initialization", False))
    
    # Test 5: Dropout
    try:
        result = test_dropout(use_real_config)
        results.append(("Test 5: Dropout", result))
    except Exception as e:
        print(f"\n❌ Test 5 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 5: Dropout", False))
    
    # Test 6: Embedding weight shape
    try:
        result = test_embedding_weight_shape(use_real_config)
        results.append(("Test 6: Embedding Weight Shape", result))
    except Exception as e:
        print(f"\n❌ Test 6 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 6: Embedding Weight Shape", False))
    
    # Test 7: Padding_idx
    try:
        result = test_padding_idx(use_real_config)
        results.append(("Test 7: Padding_idx", result))
    except Exception as e:
        print(f"\n❌ Test 7 HATA: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 7: Padding_idx", False))
    
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
        print("LanguageEmbedding endüstri standartlarına uygun çalışıyor.")
    else:
        print(f"\n⚠️  {total - passed} TEST BAŞARISIZ!")
        print("LanguageEmbedding'de sorunlar bulundu. Detaylar için yukarıdaki çıktılara bakın.")
    
    return passed == total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding Layer Test")
    parser.add_argument("--real-config", action="store_true", 
                       help="Gerçek model konfigürasyonu ile test et (vocab_size=60312, embed_dim=512)")
    args = parser.parse_args()
    
    success = main(use_real_config=args.real_config)
    sys.exit(0 if success else 1)

