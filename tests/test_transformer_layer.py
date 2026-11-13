"""
Test: Transformer Encoder Layer Analizi
Dosya: src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py
Sınıf/Metod: TransformerEncoderLayer.__init__(), TransformerEncoderLayer.forward()

Bu test, TransformerEncoderLayer sınıfının endüstri standartlarına uygunluğunu kontrol eder.

Endüstri Standartları:
- Pre-norm/Post-norm doğru olmalı
- Causal mask doğru uygulanmalı
- Residual connection doğru olmalı
- Attention mechanism doğru çalışmalı

Test Senaryoları:
1. Forward pass kontrolü
2. Causal mask kontrolü (autoregressive için kritik!)
3. Residual connection kontrolü
4. Pre-norm/Post-norm kontrolü
5. Output shape kontrolü
6. Gradient flow kontrolü

Not (API değişikliği): TransformerEncoderLayer.__init__ artık ``causal_mask`` parametresi
almıyor; causal mask her çağrıda ``layer(x, causal_mask=True)`` ile istenir
(CevahirNeuralNetwork da katmanlara bu şekilde iletir; forward varsayılanı False).
Eski testler ``causal_mask=True``'yu constructor'a veriyordu; argüman ``**kwargs``
tarafından sessizce yutulduğu için katmanlar maskesiz (çift yönlü) attention ile
çalışıyor ve causal mask testi hiçbir şeyi doğrulamıyordu.
"""

import sys
import os

import pytest
import torch

# Proje kök dizinini path'e ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.neural_network_module.ortak_katman_module.transformer_encoder_layer import TransformerEncoderLayer


def test_forward_pass():
    """Test 1: Forward pass kontrolü"""
    embed_dim = 512
    num_heads = 8
    ffn_dim = 2048
    dropout = 0.1

    layer = TransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        pre_norm=True,
    )

    # Test input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Forward pass (returns tuple: (output, attn_weights) veya (output, attn_weights, cache))
    result = layer(x, causal_mask=True)
    output = result[0] if isinstance(result, tuple) else result  # İlk eleman output tensor

    # Kontrol: Output shape doğru mu?
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, (
        f"Shape uyuşmazlığı! Beklenen: {expected_shape}, Alınan: {tuple(output.shape)}"
    )

    # Kontrol: Output NaN veya Inf içermiyor mu?
    assert not torch.isnan(output).any(), "Output NaN içeriyor!"
    assert not torch.isinf(output).any(), "Output Inf içeriyor!"


def test_pre_norm_post_norm():
    """Test 2: Pre-norm/Post-norm kontrolü"""
    embed_dim = 512
    num_heads = 8
    ffn_dim = 2048
    dropout = 0.0  # Dropout'u kapatıyoruz, norm testi için

    # Pre-norm layer
    layer_pre_norm = TransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        pre_norm=True,
    )

    # Post-norm layer
    layer_post_norm = TransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        pre_norm=False,
    )
    # Aynı ağırlıklar: çıktı farkı rastgele başlatmadan değil, yalnızca norm
    # yerleşiminden (pre vs post) kaynaklanmalı
    layer_post_norm.load_state_dict(layer_pre_norm.state_dict())

    # Test input
    x = torch.randn(2, 10, embed_dim)

    # Forward pass (returns tuple: (output, attn_weights))
    result_pre = layer_pre_norm(x, causal_mask=True)
    result_post = layer_post_norm(x, causal_mask=True)

    output_pre = result_pre[0] if isinstance(result_pre, tuple) else result_pre
    output_post = result_post[0] if isinstance(result_post, tuple) else result_post

    # Kontrol: Her iki mod da çalışıyor mu?
    assert output_pre.shape == x.shape, (
        f"Pre-norm çalışmıyor: Beklenen shape {tuple(x.shape)}, Alınan {tuple(output_pre.shape)}"
    )
    assert output_post.shape == x.shape, (
        f"Post-norm çalışmıyor: Beklenen shape {tuple(x.shape)}, Alınan {tuple(output_post.shape)}"
    )

    # Pre-norm ve Post-norm (aynı ağırlıklarla) farklı output üretmeli (normal)
    assert not torch.allclose(output_pre, output_post, atol=1e-5), (
        "Pre-norm ve Post-norm aynı ağırlıklarla aynı output üretiyor; pre_norm bayrağı etkisiz!"
    )


def test_residual_connection():
    """Test 3: Residual connection kontrolü"""
    embed_dim = 512
    num_heads = 8
    ffn_dim = 2048
    dropout = 0.0  # Dropout'u kapatıyoruz

    layer = TransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        pre_norm=True,
    )

    # Test input
    x = torch.randn(2, 10, embed_dim)

    # Forward pass (returns tuple: (output, attn_weights))
    result = layer(x, causal_mask=True)
    output = result[0] if isinstance(result, tuple) else result

    # Residual connection kontrolü: Output, input'u içermeli (gradient flow için)
    # Ancak layer içinde birçok transformasyon olduğu için exact eşitlik beklenmez
    # Sadece shape ve gradient flow kontrolü yapabiliriz

    # Kontrol: Output shape input ile aynı mı?
    assert output.shape == x.shape, (
        f"Output shape input ile aynı değil! Input: {tuple(x.shape)}, Output: {tuple(output.shape)}"
    )


def test_causal_mask():
    """Test 4: Causal mask kontrolü (autoregressive için kritik!)"""
    embed_dim = 512
    num_heads = 8
    ffn_dim = 2048
    dropout = 0.0

    # Causal mask forward'da istenir (constructor'da değil; bkz. modül notu)
    layer_with_mask = TransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        pre_norm=True,
    )
    layer_with_mask.eval()  # Deterministik karşılaştırma için

    # Test input
    batch_size = 1
    seq_len = 5
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(batch_size, seq_len, embed_dim, generator=generator)
    # Sadece son (gelecekteki) token'ı değiştir
    x_future_changed = x.clone()
    x_future_changed[:, -1, :] += torch.randn(embed_dim, generator=generator)

    with torch.no_grad():
        output = layer_with_mask(x, causal_mask=True)[0]
        output_changed = layer_with_mask(x_future_changed, causal_mask=True)[0]
        # Negatif kontrol: mask kapalıyken aynı değişiklik önceki pozisyonlara sızmalı
        output_no_mask = layer_with_mask(x, causal_mask=False)[0]
        output_no_mask_changed = layer_with_mask(x_future_changed, causal_mask=False)[0]

    # Kontrol: Output shape doğru mu?
    assert output.shape == x.shape, (
        f"Shape uyuşmazlığı! Beklenen: {tuple(x.shape)}, Alınan: {tuple(output.shape)}"
    )

    # Causal mask'ı attention scores'a erişmeden (black-box) doğrularız:
    # gelecekteki token değişince önceki pozisyonların çıktısı değişmemeli.
    leak = (output[:, :-1] - output_changed[:, :-1]).abs().max().item()
    assert leak <= 1e-6, (
        "Causal mask sızdırıyor: gelecekteki token önceki pozisyonların çıktısını değiştirdi "
        f"(max |fark|={leak:.3e})"
    )
    # Değişiklik kendi pozisyonunda etkili olmalı (aksi halde test bir şey ölçmez)
    assert not torch.allclose(output[:, -1], output_changed[:, -1]), (
        "Son token değişikliği kendi pozisyonundaki çıktıyı etkilemedi; test geçersiz"
    )
    # Mask kapalıyken sızıntı görülmeli; görülmüyorsa test causal mask'ı ayırt edemez
    no_mask_leak = (output_no_mask[:, :-1] - output_no_mask_changed[:, :-1]).abs().max().item()
    assert no_mask_leak > 1e-4, (
        "Mask kapalıyken bile önceki pozisyonlar değişmedi "
        f"(max |fark|={no_mask_leak:.3e}); test causal mask'ı ayırt edemiyor"
    )


def test_output_shape_various_inputs():
    """Test 5: Çeşitli input shape'ler için output shape kontrolü"""
    embed_dim = 512
    num_heads = 8
    ffn_dim = 2048
    dropout = 0.0

    layer = TransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        pre_norm=True,
    )

    # Farklı batch size ve sequence length'ler ile test
    test_cases = [
        (1, 5),
        (2, 10),
        (4, 20),
        (8, 50),
    ]

    for batch_size, seq_len in test_cases:
        x = torch.randn(batch_size, seq_len, embed_dim)
        result = layer(x, causal_mask=True)
        output = result[0] if isinstance(result, tuple) else result

        expected_shape = (batch_size, seq_len, embed_dim)
        assert output.shape == expected_shape, (
            f"Batch={batch_size}, Seq={seq_len}: shape yanlış! "
            f"Beklenen: {expected_shape}, Alınan: {tuple(output.shape)}"
        )


def test_gradient_flow():
    """Test 6: Gradient flow kontrolü"""
    embed_dim = 512
    num_heads = 8
    ffn_dim = 2048
    dropout = 0.0

    layer = TransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        pre_norm=True,
    )

    # Test input
    x = torch.randn(2, 10, embed_dim)

    # Forward pass (returns tuple: (output, attn_weights))
    result = layer(x, causal_mask=True)
    output = result[0] if isinstance(result, tuple) else result

    # Loss (dummy)
    loss = output.mean()

    # Backward pass
    loss.backward()

    # Kontrol: Gradient'ler var mı?
    params_with_grad = [name for name, param in layer.named_parameters() if param.grad is not None]
    assert params_with_grad, "Hiçbir parametre için gradient oluşmadı!"


def test_layer_structure():
    """Test 7: Layer structure kontrolü (modüller var mı?)"""
    embed_dim = 512
    num_heads = 8
    ffn_dim = 2048
    dropout = 0.1

    layer = TransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        pre_norm=True,
    )

    # Kontrol: Gerekli modüller var mı?
    modules = {
        "Attention": hasattr(layer, 'attn') or hasattr(layer, 'self_attn') or hasattr(layer, 'attention'),
        "Feed-forward": hasattr(layer, 'ffn') or hasattr(layer, 'feed_forward'),
        "Norm1 (attention öncesi/sonrası)": hasattr(layer, 'norm1') or hasattr(layer, 'ln1'),
        "Norm2 (FFN öncesi/sonrası)": hasattr(layer, 'norm2') or hasattr(layer, 'ln2'),
        "Dropout": hasattr(layer, 'dropout'),
    }
    missing = [name for name, exists in modules.items() if not exists]

    assert not missing, f"Bazı modüller eksik: {missing}"


if __name__ == "__main__":
    # Testleri pytest ile koştur (cache dizini repoya yazılmaz)
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider", *sys.argv[1:]]))
