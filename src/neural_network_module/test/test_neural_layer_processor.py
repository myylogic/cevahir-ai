# tests/test_neural_layer_processor.py
import pytest
import time
import torch
from neural_network_module.ortak_katman_module.neural_layer_processor import NeuralLayerProcessor


# ---------- Fixtures ----------
@pytest.fixture
def nlp_multi_head():
    return NeuralLayerProcessor(
        embed_dim=512,
        num_heads=32,
        attention_type="multi_head",
        dropout=0.2,
        debug=False,
        normalization_type="layer_norm",
        scaling_strategy="sqrt",
        scaling_method="softmax",
        verbose=True
    )

@pytest.fixture
def nlp_self():
    return NeuralLayerProcessor(
        embed_dim=512,
        num_heads=32,
        attention_type="self",
        dropout=0.2,
        debug=False,
        normalization_type="layer_norm",
        scaling_strategy="sqrt",
        scaling_method="softmax",
        verbose=True
    )

@pytest.fixture
def nlp_cross():
    return NeuralLayerProcessor(
        embed_dim=512,
        num_heads=32,
        attention_type="cross",
        dropout=0.2,
        debug=False,
        normalization_type="layer_norm",
        scaling_strategy="sqrt",
        scaling_method="softmax",
        verbose=True
    )


# ---------- Tests ----------
# Test 1: Geçerli multi_head initialization
def test_initialization_multi_head(nlp_multi_head):
    model = nlp_multi_head
    assert model.embed_dim == 512
    assert model.num_heads == 32
    assert hasattr(model, "multi_head_attention")


# Test 3: Multi-head attention için forward pass çıktısı ve dikkat ağırlıkları
def test_forward_multi_head(nlp_multi_head):
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    output, attn_weights = nlp_multi_head(query, key, value)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights is not None


# Test 5: Cross-attention için forward pass çıktısı
def test_forward_cross(nlp_cross):
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    output, attn_weights = nlp_cross(query, key, value)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights is not None


# Test 6: Forward pass sırasında NaN veya Inf yok
def test_forward_no_nan_inf(nlp_multi_head):
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    output, _ = nlp_multi_head(query, key, value)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


# Test 7: Eğitim modunda dropout etkisi (aynı girdiye farklı çıktı)
def test_dropout_training(nlp_multi_head):
    nlp_multi_head.train()
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    output1, _ = nlp_multi_head(query, key, value)
    output2, _ = nlp_multi_head(query, key, value)
    assert not torch.allclose(output1, output2, atol=1e-5)


# Test 8: Eval modunda deterministik çıktı
def test_eval_mode_determinism(nlp_multi_head):
    nlp_multi_head.eval()
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    with torch.no_grad():
        output1, _ = nlp_multi_head(query, key, value)
        output2, _ = nlp_multi_head(query, key, value)
    assert torch.allclose(output1, output2, atol=1e-5)


# Test 9: initialize_attention girişle aynı şekli döndürür
def test_initialize_attention(nlp_multi_head):
    batch_size, seq_len, embed_dim = 4, 10, 512
    # Not: 512512 gibi devasa değerler gereksiz bellek tüketir; pratik boyutta tutuyoruz.
    inputs = torch.rand(batch_size, seq_len, embed_dim)
    initialized = nlp_multi_head.initialize_attention(inputs)
    assert initialized.shape == inputs.shape


# Test 10: normalize_attention aynı şekli döndürür
def test_normalize_attention(nlp_multi_head):
    batch_size, seq_len, embed_dim = 4, 10, 512
    inputs = torch.rand(batch_size, seq_len, embed_dim)
    normalized = nlp_multi_head.normalize_attention(inputs)
    assert normalized.shape == inputs.shape


# Test 12: _validate_tensor yanlış tipte veri aldığında hata
def test_validate_tensor_error(nlp_multi_head):
    with pytest.raises(TypeError):
        nlp_multi_head._validate_tensor("not a tensor", name="test")


# Test 13: Yanlış boyutlu input verildiğinde forward hata verir
def test_forward_wrong_dimension(nlp_multi_head):
    wrong_input = torch.rand(4, 10)  # 2D tensor
    with pytest.raises(ValueError):
        nlp_multi_head(wrong_input)


# Test 14: AttentionOptimizer forward (3D tensör) aynı şekli korur
def test_optimize_attention(nlp_multi_head):
    batch_size, seq_len, embed_dim = 4, 10, 512
    outputs = torch.rand(batch_size, seq_len, embed_dim)
    optimized = nlp_multi_head.attention_optimizer.forward(outputs)
    assert optimized.shape == outputs.shape


# Test 15: Forward performansı makul sürede tamamlanır
def test_forward_performance(nlp_multi_head):
    nlp_multi_head.eval()
    batch_size, seq_len, embed_dim = 16, 50, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    start = time.time()
    with torch.no_grad():
        output, _ = nlp_multi_head(query, key, value)
    duration = time.time() - start
    # Ortam farklarını tolere etmek için eşiği makul tuttuk
    assert duration < 0.25


# Test 16: Cross-attention'da 4D mask doğru sıkıştırılır
def test_cross_attention_mask_squeeze(nlp_cross):
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    mask = torch.randint(0, 2, (batch_size, 1, 1, seq_len)).float()
    output, attn_weights = nlp_cross(query, key, value, mask=mask)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights is not None


# Test 17: Eval modunda birden fazla forward'ta çıktıların stabil olması
def test_forward_stability(nlp_multi_head):
    nlp_multi_head.eval()
    batch_size, seq_len, embed_dim = 8, 20, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    with torch.no_grad():
        outputs = [nlp_multi_head(query, key, value)[0] for _ in range(5)]
    stacked = torch.stack(outputs)
    variance = torch.var(stacked, dim=0)
    assert torch.max(variance) < 1e-5


# Test 18: Çıktı aralığı mantıklı (ör. -10, 10)
def test_forward_output_range(nlp_multi_head):
    nlp_multi_head.eval()
    batch_size, seq_len, embed_dim = 4, 15, 512
    query = torch.rand(batch_size, seq_len, embed_dim) * 2 - 1
    key = torch.rand(batch_size, seq_len, embed_dim) * 2 - 1
    value = torch.rand(batch_size, seq_len, embed_dim) * 2 - 1
    with torch.no_grad():
        output, _ = nlp_multi_head(query, key, value)
    assert output.min() > -10 and output.max() < 10


# Test 19: Farklı sequence uzunluklarında şekiller doğru
def test_varying_sequence_lengths(nlp_multi_head):
    nlp_multi_head.eval()
    batch_size = 1
    embed_dim = 512
    seq_lens = [10, 15, 20, 12]
    outputs = []
    with torch.no_grad():
        for seq_len in seq_lens:
            query = torch.rand(batch_size, seq_len, embed_dim)
            key = torch.rand(batch_size, seq_len, embed_dim)
            value = torch.rand(batch_size, seq_len, embed_dim)
            output, _ = nlp_multi_head(query, key, value)
            outputs.append(output)
            assert output.shape == (batch_size, seq_len, embed_dim)
    concatenated = torch.cat(outputs, dim=1)
    assert concatenated.shape[1] == sum(seq_lens)


# Ek: Self-Attention özel testi (ağırlık döndürmez, None beklenir)
def test_forward_self_attention(nlp_self):
    batch_size, seq_len, embed_dim = 3, 9, 512
    x = torch.rand(batch_size, seq_len, embed_dim)
    output, weights = nlp_self(x)  # NeuralLayerProcessor, (output, None) döndürür
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert weights is None
