
import pytest
import torch
import torch.nn.functional as F
import time
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.neural_network_module.ortak_katman_module.attention_manager_module.multi_head_attention import MultiHeadAttention

# MultiHeadAttention modülünü MultiHeadAttention içerisinden izole edebilmek için bir fixture tanımlıyoruz.
@pytest.fixture
def multi_head_attention():
    nlp = MultiHeadAttention(
        embed_dim=512,
        num_heads=32,
        dropout=0.2,
        debug=True,              # Debug modunu açarak daha fazla log alınabilir.
        normalization_type="layer_norm"
    )
    # Testlerde deterministik sonuç almak için model eval moduna alınabilir
    nlp.train()
    return nlp


def test_forward_output_shape(multi_head_attention):
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key   = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    output, attn_weights = multi_head_attention(query, key, value, mask=None, return_attention_weights=True)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights.shape == (batch_size, multi_head_attention.num_heads, seq_len, seq_len)

def test_no_nan_inf(multi_head_attention):
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key   = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    output, _ = multi_head_attention(query, key, value, mask=None, return_attention_weights=True)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_dropout_effect_training(multi_head_attention):
    multi_head_attention.train()
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key   = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    out1, _ = multi_head_attention(query, key, value, mask=None, return_attention_weights=True)
    out2, _ = multi_head_attention(query, key, value, mask=None, return_attention_weights=True)
    # Eğitim modunda dropout uygulandığı için, iki forward pass’in çıktıları aynı olmamalı
    with pytest.raises(AssertionError):
        assert torch.allclose(out1, out2, atol=1e-5)

def test_deterministic_eval(multi_head_attention):
    multi_head_attention.eval()
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key   = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    out1, _ = multi_head_attention(query, key, value, mask=None, return_attention_weights=True)
    out2, _ = multi_head_attention(query, key, value, mask=None, return_attention_weights=True)
    assert torch.allclose(out1, out2, atol=1e-5)

def test_attention_weights_range(multi_head_attention):
    multi_head_attention.eval()
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key   = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    _, attn_weights = multi_head_attention(query, key, value, mask=None, return_attention_weights=True)
    # Dikkat ağırlıklarının 0 ile 1 arasında olduğunu ve son boyutta toplamlarının 1'e yakın olduğunu kontrol et
    assert (attn_weights >= 0).all()
    assert (attn_weights <= 1).all()
    sums = attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

def test_mask_application(multi_head_attention):
    multi_head_attention.eval()
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key   = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    # Örnek maske: Son 5 tokeni maskeliyoruz.
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 5:] = 0
    output, attn_weights = multi_head_attention(query, key, value, mask=mask, return_attention_weights=True)
    # Mask uygulanan bölgelerdeki dikkat ağırlıkları neredeyse sıfıra eşit olmalı (clamp sonrası)
    assert (attn_weights[:, :, :, 5:] <= 1e-4).all()


def test_output_numerical_range(multi_head_attention):
    multi_head_attention.eval()
    batch_size, seq_len, embed_dim = 4, 15, 512
    # Girişleri [-1, 1] aralığına ölçekleyelim
    query = torch.rand(batch_size, seq_len, embed_dim) * 2 - 1
    key   = torch.rand(batch_size, seq_len, embed_dim) * 2 - 1
    value = torch.rand(batch_size, seq_len, embed_dim) * 2 - 1
    output, _ = multi_head_attention(query, key, value, mask=None, return_attention_weights=True)
    # Çıktının numerik aralığı -10 ile 10 arasında olsun
    assert output.min() > -10 and output.max() < 10

def test_wrong_input_dimension(multi_head_attention):
    multi_head_attention.eval()
    wrong_input = torch.rand(4, 10)  # 2D tensor, beklenen 3D tensor
    with pytest.raises(ValueError):
        multi_head_attention(wrong_input, wrong_input, wrong_input, mask=None, return_attention_weights=True)

def test_forward_performance(multi_head_attention):
    multi_head_attention.eval()
    batch_size, seq_len, embed_dim = 16, 50, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key   = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    start = time.time()
    _ , _ = multi_head_attention(query, key, value, mask=None, return_attention_weights=True)
    duration = time.time() - start
    # Küçük girdi boyutlarında forward pass süresi 0.1 saniyenin altında olmalı
    assert duration < 0.1

def test_mask_effect_extreme(multi_head_attention):
    """
    Maske uygulandığında, masklenen bölgelerdeki attention ağırlıkları
    çok düşük (neredeyse sıfır) olmalıdır.
    """
    multi_head_attention.eval()
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key   = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 5:] = 0  # Son yarı maskelensin
    _, attn_weights = multi_head_attention(query, key, value, mask=mask, return_attention_weights=True)
    assert (attn_weights[:, :, :, 5:] <= 1e-4).all()

def test_stability_with_large_input(multi_head_attention):
    """
    Çok büyük ölçekli rastgele tensorlerle (örneğin 1e4 ile çarpılmış) scaled_dot_product_attention
    metodunun çıktılarının sayısal olarak stabil olduğunu doğrular.
    """
    batch_size, seq_len, embed_dim = 4, 10, 512
    head_dim = embed_dim // multi_head_attention.num_heads
    query = torch.randn(batch_size, multi_head_attention.num_heads, seq_len, head_dim) * 1e4
    key   = torch.randn(batch_size, multi_head_attention.num_heads, seq_len, head_dim) * 1e4
    value = torch.randn(batch_size, multi_head_attention.num_heads, seq_len, head_dim) * 1e4
    output, attn_weights = multi_head_attention.scaled_dot_product_attention(query, key, value)
    assert torch.isfinite(output).all()
    assert torch.isfinite(attn_weights).all()

def test_logsumexp_stabilization(multi_head_attention):
    """
    Çok yüksek değerlerden oluşan skor tensorü oluşturup,
    Log-Sum-Exp stabilizasyonunun beklendiği gibi çalıştığını kontrol eder.
    """
    batch_size, seq_len, _ = 4, 10, 512
    scores = torch.full((batch_size, multi_head_attention.num_heads, seq_len, seq_len), 1e8)
    stabilized = scores - scores.max(dim=-1, keepdim=True)[0]
    softmaxed = F.softmax(stabilized, dim=-1)
    expected = torch.full(softmaxed.shape, 1.0 / seq_len)
    torch.testing.assert_close(softmaxed, expected, rtol=0.0, atol=1e-5)

def test_no_nan_inf_in_scaled_attention(multi_head_attention):
    """
    Çok yüksek ölçekli rastgele değerlerle scaled_dot_product_attention metodunun
    NaN veya Inf üretmediğini kontrol eder.
    """
    batch_size, seq_len, embed_dim = 4, 10, 512
    head_dim = embed_dim // multi_head_attention.num_heads
    query = torch.randn(batch_size, multi_head_attention.num_heads, seq_len, head_dim) * 1e4
    key   = torch.randn(batch_size, multi_head_attention.num_heads, seq_len, head_dim) * 1e4
    value = torch.randn(batch_size, multi_head_attention.num_heads, seq_len, head_dim) * 1e4
    output, attn_weights = multi_head_attention.scaled_dot_product_attention(query, key, value)
    assert torch.isfinite(output).all()
    assert torch.isfinite(attn_weights).all()



########################
###############################
#############################################

def test_extreme_large_scores(multi_head_attention):
    """
    Çok büyük (yüksek mutlak değer) giriş değerleri ile scaled_dot_product_attention
    metodunun sayısal stabilitesini test eder.
    Beklenen: Dropout devre dışı (apply_dropout=False) olduğunda softmax çıktıları uniform (eşit) olmalı.
    """
    batch_size, seq_len, embed_dim = 4, 10, 512
    # Tüm girişler çok büyük, aynı değer olsun.
    query = torch.full((batch_size, seq_len, embed_dim), 1e6)
    key   = torch.full((batch_size, seq_len, embed_dim), 1e6)
    value = torch.rand(batch_size, seq_len, embed_dim)  # Rastgele değerler
    # Deterministik softmax sonucu için dropout uygulanmayacak.
    output, attn_weights = multi_head_attention(query, key, value, mask=None, return_attention_weights=True, apply_dropout=False)
    assert torch.isfinite(output).all(), "Çok büyük giriş sonrası çıktı NaN/Inf içeriyor."
    assert torch.isfinite(attn_weights).all(), "Çok büyük giriş sonrası attention weights NaN/Inf içeriyor."
    expected = torch.full(attn_weights.shape, 1.0 / seq_len)
    torch.testing.assert_close(attn_weights, expected, rtol=0.0, atol=1e-4)

def test_extreme_small_scores(multi_head_attention):
    """
    Çok düşük (yüksek negatif) giriş değerleri ile scaled_dot_product_attention
    metodunun stabilitesini test eder.
    Beklenen: Dropout devre dışı (apply_dropout=False) olduğunda softmax çıktıları uniform (eşit) olmalı.
    """
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.full((batch_size, seq_len, embed_dim), -1e6)
    key   = torch.full((batch_size, seq_len, embed_dim), -1e6)
    value = torch.rand(batch_size, seq_len, embed_dim)
    output, attn_weights = multi_head_attention(query, key, value, mask=None, return_attention_weights=True, apply_dropout=False)
    assert torch.isfinite(output).all(), "Çok düşük giriş sonrası çıktı NaN/Inf içeriyor."
    assert torch.isfinite(attn_weights).all(), "Çok düşük giriş sonrası attention weights NaN/Inf içeriyor."
    expected = torch.full(attn_weights.shape, 1.0 / seq_len)
    torch.testing.assert_close(attn_weights, expected, rtol=0.0, atol=1e-4)

def test_constant_input_uniform_attention(multi_head_attention):
    """
    Giriş tensörleri sabit (aynı değer) olduğunda, attention ağırlıklarının
    uniform (eşit) olması beklenir.
    """
    batch_size, seq_len, embed_dim = 4, 10, 512
    constant = 3.14
    query = torch.full((batch_size, seq_len, embed_dim), constant)
    key   = torch.full((batch_size, seq_len, embed_dim), constant)
    value = torch.full((batch_size, seq_len, embed_dim), constant)
    output, attn_weights = multi_head_attention(query, key, value, mask=None, return_attention_weights=True, apply_dropout=False)
    expected = torch.full(attn_weights.shape, 1.0 / seq_len)
    torch.testing.assert_close(attn_weights, expected, rtol=0.0, atol=1e-5)

def test_temperature_extreme_high(multi_head_attention):
    """
    Çok yüksek sıcaklık kullanıldığında (temperature), softmax dağılımı
    neredeyse uniform (eşit) olmalıdır.
    """
    batch_size, seq_len, embed_dim = 4, 10, 512
    head_dim = embed_dim // multi_head_attention.num_heads
    torch.manual_seed(0)
    query = torch.rand(batch_size, multi_head_attention.num_heads, seq_len, head_dim)
    key   = torch.rand(batch_size, multi_head_attention.num_heads, seq_len, head_dim)
    value = torch.rand(batch_size, multi_head_attention.num_heads, seq_len, head_dim)
    _, attn_weights = multi_head_attention.scaled_dot_product_attention(query, key, value, mask=None, temperature=1000.0, apply_dropout=False)
    uniform_val = 1.0 / seq_len
    torch.testing.assert_close(attn_weights, torch.full(attn_weights.shape, uniform_val), rtol=0.0, atol=1e-3)

def test_logsumexp_stabilization(multi_head_attention):
    """
    Çok yüksek değerlerden oluşan skor tensorü oluşturup,
    Log-Sum-Exp stabilizasyonunun beklendiği gibi çalıştığını kontrol eder.
    """
    batch_size, seq_len, _ = 4, 10, 512
    scores = torch.full((batch_size, multi_head_attention.num_heads, seq_len, seq_len), 1e8)
    stabilized = scores - scores.max(dim=-1, keepdim=True)[0]
    softmaxed = F.softmax(stabilized, dim=-1)
    expected = torch.full(softmaxed.shape, 1.0 / seq_len)
    torch.testing.assert_close(softmaxed, expected, rtol=0.0, atol=1e-5)

def test_no_nan_inf_in_scaled_attention(multi_head_attention):
    """
    Çok yüksek ölçekli rastgele değerlerle scaled_dot_product_attention metodunun
    NaN veya Inf üretmediğini kontrol eder.
    """
    batch_size, seq_len, embed_dim = 4, 10, 512
    head_dim = embed_dim // multi_head_attention.num_heads
    query = torch.randn(batch_size, multi_head_attention.num_heads, seq_len, head_dim) * 1e4
    key   = torch.randn(batch_size, multi_head_attention.num_heads, seq_len, head_dim) * 1e4
    value = torch.randn(batch_size, multi_head_attention.num_heads, seq_len, head_dim) * 1e4
    output, attn_weights = multi_head_attention.scaled_dot_product_attention(query, key, value, apply_dropout=False)
    assert torch.isfinite(output).all()
    assert torch.isfinite(attn_weights).all()

def test_manual_softmax_computation(multi_head_attention):
    """
    Çok basit, tek bir örnek için softmax hesaplamasını elle hesaplayıp,
    scaled_dot_product_attention metodunun çıktıları ile karşılaştırır.
    """
    batch_size, seq_len, embed_dim = 1, 3, 4
    num_heads = 1
    head_dim = embed_dim
    query = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                            [4.0, 3.0, 2.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0]]])
    key = query.clone()
    value = query.clone()
    query = query.view(batch_size, num_heads, seq_len, head_dim)
    key = key.view(batch_size, num_heads, seq_len, head_dim)
    value = value.view(batch_size, num_heads, seq_len, head_dim)
    scaling_factor = math.sqrt(head_dim)
    scores = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor
    scores = scores - scores.max(dim=-1, keepdim=True)[0]
    softmax_manual = F.softmax(scores, dim=-1)
    _, attn_weights = multi_head_attention.scaled_dot_product_attention(query, key, value, apply_dropout=False)
    torch.testing.assert_close(attn_weights, softmax_manual, rtol=0.0, atol=1e-5)

def test_temperature_extreme_low(multi_head_attention):
    """
    Çok düşük sıcaklık (temperature) kullanıldığında softmax dağılımı
    çok keskinleşmeli, yani maksimum attention ağırlığı neredeyse 1'e yaklaşmalıdır.
    """
    batch_size, seq_len, embed_dim = 4, 10, 512
    head_dim = embed_dim // multi_head_attention.num_heads
    torch.manual_seed(0)
    # Burada dropout etkisini kaldırmak için apply_dropout=False
    query = torch.rand(batch_size, multi_head_attention.num_heads, seq_len, head_dim)
    key   = torch.rand(batch_size, multi_head_attention.num_heads, seq_len, head_dim)
    value = torch.rand(batch_size, multi_head_attention.num_heads, seq_len, head_dim)
    _, attn_weights = multi_head_attention.scaled_dot_product_attention(query, key, value, mask=None, temperature=0.001, apply_dropout=False)
    max_vals = attn_weights.max(dim=-1)[0]
    assert (max_vals > 0.99).all(), "Çok düşük sıcaklıkta softmax dağılımı beklenenden keskin değil."

# ✅ YENİ (V-2): Causal mask testleri
def test_causal_mask(multi_head_attention):
    """Causal mask testi"""
    multi_head_attention.eval()
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    
    # Causal mask ile forward pass
    output_causal, attn_weights_causal = multi_head_attention(
        query, key, value,
        mask=None,
        causal_mask=True,
        return_attention_weights=True,
    )
    
    # Causal mask olmadan forward pass
    output_no_causal, attn_weights_no_causal = multi_head_attention(
        query, key, value,
        mask=None,
        causal_mask=False,
        return_attention_weights=True,
    )
    
    # Shape'ler aynı olmalı
    assert output_causal.shape == output_no_causal.shape
    assert attn_weights_causal.shape == attn_weights_no_causal.shape
    
    # ✅ ENDÜSTRİ STANDARDI: Causal mask'in düzgün çalıştığını kontrol et
    # Causal mask ile, future token'lara attention verilmemeli
    # Bu yüzden upper triangle attention weights'ler çok düşük olmalı (≈0)
    
    # 1) Çıktılar farklı olmalı (causal mask etkisi var)
    output_diff = torch.abs(output_causal - output_no_causal).mean()
    assert output_diff > 1e-5, f"Causal mask etkisi yok: output_diff={output_diff:.6f}"
    
    # 2) Upper triangle attention weights kontrolü (endüstri standardı)
    # Causal mask ile, j > i için attention weights ≈ 0 olmalı
    for i in range(seq_len):
        for j in range(i + 1, seq_len):  # Upper triangle (j > i)
            # Tüm batch ve head'ler için ortalama attention weight
            avg_attention = attn_weights_causal[:, :, i, j].mean().item()
            # Endüstri standardı: Causal mask ile future tokens'a attention ≈ 0
            assert avg_attention < 1e-3, \
                f"Causal mask başarısız: position ({i},{j}) için attention={avg_attention:.6f} (beklenen: <0.001)"


def test_causal_mask_with_padding_mask(multi_head_attention):
    """Causal mask + padding mask kombinasyonu"""
    multi_head_attention.eval()
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    
    # Padding mask oluştur
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[0, 8:] = True  # İlk batch'te son 2 token padding
    
    output, attn_weights = multi_head_attention(
        query, key, value,
        mask=mask,
        causal_mask=True,
        return_attention_weights=True,
    )
    
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights.shape == (batch_size, multi_head_attention.num_heads, seq_len, seq_len)


if __name__ == "__main__":
    pytest.main([__file__])