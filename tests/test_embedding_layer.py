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

Konfigürasyon:
- Varsayılan: test konfigürasyonu (vocab_size=1000, embed_dim=128).
- Gerçek model konfigürasyonu (vocab_size=60312, embed_dim=512): daha yavaş
  olduğu için isteğe bağlıdır; CEVAHIR_REAL_CONFIG=1 ortam değişkeni ile ya da
  `python tests/test_embedding_layer.py --real-config` ile çalıştırılır.
"""

import sys
import os
import math

import pytest
import torch

# Proje kök dizinini path'e ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.neural_network_module.dil_katmani_module.language_embedding import LanguageEmbedding

# Gerçek model konfigürasyonunu etkinleştiren ortam değişkeni
REAL_CONFIG_ENV = "CEVAHIR_REAL_CONFIG"


@pytest.fixture
def use_real_config():
    """Gerçek model konfigürasyonu istendi mi? (varsayılan: hayır → test konfigürasyonu)"""
    return os.environ.get(REAL_CONFIG_ENV) == "1"


def _config(use_real_config):
    """(vocab_size, embed_dim, pad_id) döndürür."""
    if use_real_config:
        return 60312, 512, 0
    return 1000, 128, 0


def test_padding_token_embedding(use_real_config):
    """Test 1: Padding token embedding sıfır olmalı (PyTorch standardı)"""
    vocab_size, embed_dim, pad_id = _config(use_real_config)

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

    # Kontrol: Padding embedding sıfır olmalı
    assert torch.allclose(pad_embedding, torch.zeros_like(pad_embedding), atol=1e-6), (
        "Padding embedding sıfır değil! PyTorch standardına uygun değil. "
        f"sum={pad_embedding.sum().item():.6f}, norm={pad_embedding.norm().item():.6f}"
    )


def test_normal_token_embedding(use_real_config):
    """Test 2: Normal token embedding'ler sıfır olmamalı"""
    vocab_size, embed_dim, pad_id = _config(use_real_config)

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

    # Kontrol: Normal embedding'ler sıfır olmamalı
    assert not torch.allclose(normal_embedding, torch.zeros_like(normal_embedding), atol=1e-6), (
        "Normal embedding'ler sıfır! Weight initialization sorunu olabilir. "
        f"mean={normal_embedding.mean().item():.6f}, std={normal_embedding.std().item():.6f}"
    )


def test_embedding_scaling(use_real_config):
    """Test 3: Embedding scaling kontrolü (scale_by_sqrt)

    Beklenti düzeltmesi: forward() sırası lookup → (scale_by_sqrt ise) * sqrt(embed_dim)
    → norm → dropout şeklindedir ve norm varsayılan olarak LayerNorm'dur. LayerNorm
    ölçeği (eps dışında) yok ettiği için normalize edilmiş çıktıdaki oran sqrt(embed_dim)
    değil ~1'dir (embed_dim=128 için ~1.03); eski test bu yüzden sessizce False
    döndürüyordu. Ölçekleme bu nedenle gözlemlenebildiği yerde doğrulanır:
    (a) normalizasyon kapalıyken (norm_type="none") çıktı tam olarak sqrt(embed_dim) katıdır;
    (b) varsayılan LayerNorm ile çıktı norm(lookup * sqrt(embed_dim))'e eşittir,
        yani ölçek normdan ÖNCE uygulanır.
    """
    vocab_size, embed_dim, pad_id = _config(use_real_config)
    test_token_id = 100
    test_input = torch.tensor([[test_token_id]])
    expected_scale = math.sqrt(embed_dim)

    # (a) Normalizasyon KAPALI: scale_by_sqrt etkisi çıktıda doğrudan görünür
    # Scaling KAPALI
    embedding_no_scale = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=False,
        dropout=0.0,
        norm_type="none",
    )

    # Scaling AÇIK
    embedding_with_scale = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=True,
        dropout=0.0,
        norm_type="none",
    )

    # Aynı weight'leri kopyala (karşılaştırma için)
    embedding_with_scale.embedding.weight.data = embedding_no_scale.embedding.weight.data.clone()

    emb_no_scale = embedding_no_scale(test_input)
    emb_with_scale = embedding_with_scale(test_input)
    actual_scale = (emb_with_scale / emb_no_scale).mean().item()

    # Kontrol: Scaling doğru uygulanmalı
    assert math.isclose(actual_scale, expected_scale, rel_tol=1e-5), (
        "scale_by_sqrt=True ama çıktı sqrt(embed_dim) ile çarpılmıyor! "
        f"Beklenen: sqrt({embed_dim}) = {expected_scale:.6f}, Alınan: {actual_scale:.6f}"
    )

    # (b) Varsayılan LayerNorm (norm_type=None): ölçek normdan önce uygulanmalı
    embedding_ln = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=True,
        dropout=0.0,
    )
    with torch.no_grad():
        expected_ln = embedding_ln.norm(embedding_ln.embedding(test_input) * expected_scale)
        actual_ln = embedding_ln(test_input)
    max_diff = (actual_ln - expected_ln).abs().max().item()
    assert torch.allclose(actual_ln, expected_ln, atol=1e-6), (
        "Varsayılan LayerNorm ile çıktı norm(lookup * sqrt(embed_dim)) değil; "
        f"scaling/norm sırası beklenenden farklı (max |fark|={max_diff:.3e})"
    )


def test_weight_initialization(use_real_config):
    """Test 4: Weight initialization kontrolü"""
    vocab_size, embed_dim, pad_id = _config(use_real_config)

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

    # Not: _initialize_weights, seçilen init'ten (xavier vb.) sonra her token satırını
    # L2 normu 0.15 olacak şekilde yeniden ölçekler. Bu yüzden eleman std'si
    # vocab_size'dan bağımsız olarak ≈ 0.15 / sqrt(embed_dim)'dir
    # (embed_dim=128 için ≈0.0133, embed_dim=512 için ≈0.0066).
    # Alt sınır 0.001 bilinçli olarak gevşek tutulmuştur (büyük embed_dim'lerde std küçülür).
    weight_std = non_pad_weights.std().item()
    weight_mean = non_pad_weights.mean().item()

    assert 0.001 < weight_std < 1.0, (
        f"Weight std makul aralıkta değil (0.001 < std < 1.0): std={weight_std:.6f}"
    )
    assert abs(weight_mean) < 0.1, (
        f"Weight mean sıfıra yakın değil (|mean| < 0.1): mean={weight_mean:.6f}"
    )

    # Padding token embedding sıfır olmalı
    pad_weight = weights[pad_id]
    assert torch.allclose(pad_weight, torch.zeros_like(pad_weight), atol=1e-6), (
        f"Padding weight sıfır değil! norm={pad_weight.norm().item():.6f}"
    )


def test_dropout(use_real_config):
    """Test 5: Dropout kontrolü (training/eval mode)"""
    vocab_size, embed_dim, pad_id = _config(use_real_config)

    embedding = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=False,
        dropout=0.5  # Yüksek dropout, farkı görmek için
    )

    test_input = torch.tensor([[100, 200, 300]])

    # Training mode: dropout aktif olmalı → ardışık iki forward farklı maskeler kullanır.
    # Not: Eski test her forward'dan önce aynı seed'i (42) kuruyordu; iki maske birebir
    # aynı olduğundan "farklı çıktı" kontrolü hiç sağlanamıyordu. Seed bir kez kurulur.
    embedding.train()
    torch.manual_seed(42)  # Reproducibility için
    emb_train_1 = embedding(test_input)
    emb_train_2 = embedding(test_input)

    # Eval mode: dropout pasif olmalı (deterministic, aynı çıktı)
    embedding.eval()
    emb_eval_1 = embedding(test_input)
    emb_eval_2 = embedding(test_input)

    assert not torch.allclose(emb_train_1, emb_train_2, atol=1e-5), (
        "Training mode'da dropout aktif değil! (iki forward birebir aynı çıktıyı verdi)"
    )
    assert torch.allclose(emb_eval_1, emb_eval_2, atol=1e-5), (
        "Eval mode'da dropout aktif! Eval mode'da dropout pasif olmalı. "
        f"max |fark|={(emb_eval_1 - emb_eval_2).abs().max().item():.3e}"
    )


def test_embedding_weight_shape(use_real_config):
    """Test 6: Embedding weight shape kontrolü"""
    vocab_size, embed_dim, pad_id = _config(use_real_config)

    embedding = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=False,
        dropout=0.0
    )

    expected_shape = (vocab_size, embed_dim)
    actual_shape = tuple(embedding.embedding.weight.shape)

    assert actual_shape == expected_shape, (
        f"Weight shape yanlış! Beklenen: {expected_shape}, Alınan: {actual_shape}"
    )


def test_padding_idx(use_real_config):
    """Test 7: Padding_idx kontrolü"""
    vocab_size, embed_dim, pad_id = _config(use_real_config)

    embedding = LanguageEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        padding_idx=pad_id,
        scale_by_sqrt=False,
        dropout=0.0
    )

    # PyTorch nn.Embedding'de padding_idx set edilmiş mi kontrol et
    actual_padding_idx = embedding.embedding.padding_idx

    assert actual_padding_idx == pad_id, (
        f"Padding_idx yanlış! Beklenen: {pad_id}, Alınan: {actual_padding_idx}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Layer Test")
    parser.add_argument("--real-config", action="store_true",
                        help="Gerçek model konfigürasyonu ile test et (vocab_size=60312, embed_dim=512)")
    args, pytest_args = parser.parse_known_args()
    if args.real_config:
        os.environ[REAL_CONFIG_ENV] = "1"

    # Testleri pytest ile koştur (cache dizini repoya yazılmaz)
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider", *pytest_args]))
