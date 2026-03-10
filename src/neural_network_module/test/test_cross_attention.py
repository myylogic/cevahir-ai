import pytest
import torch
from neural_network_module.ortak_katman_module.attention_manager_module.cross_attention import CrossAttention


@pytest.fixture
def default_model():
    """Varsayılan bir CrossAttention modeli döner."""
    return CrossAttention(embed_dim=64, num_heads=8, dropout=0.1, debug=True)

@pytest.fixture
def dummy_tensors():
    """Testler için örnek giriş tensörleri döner."""
    query = torch.rand(4, 10, 64)  # (batch_size, seq_len, embed_dim)
    key = torch.rand(4, 10, 64)
    value = torch.rand(4, 10, 64)
    return query, key, value

def test_initialization():
    """CrossAttention başlatma testi."""
    model = CrossAttention(embed_dim=64, num_heads=8, dropout=0.1, normalization_type="batch_norm")
    assert model.embed_dim == 64
    assert model.num_heads == 8
    assert model.debug is False
    assert model.normalization_type == "batch_norm"

    # Yanlış embed_dim ve num_heads kombinasyonu
    with pytest.raises(ValueError):
        CrossAttention(embed_dim=63, num_heads=8)

    # Desteklenmeyen normalizasyon türü
    with pytest.raises(ValueError):
        CrossAttention(embed_dim=64, num_heads=8, normalization_type="unsupported_norm")

def test_forward_pass(default_model, dummy_tensors):
    """Forward metodu testi."""
    model = default_model
    query, key, value = dummy_tensors

    # Doğru çalıştığından emin olun
    output, attn_weights = model(query, key, value)
    assert output.shape == query.shape
    assert attn_weights.shape == (4, 10, 10)  # (batch_size, seq_len, seq_len)


def test_check_tensor_values(default_model):
    """NaN ve sonsuz değer kontrolünü test et."""
    model = default_model
    query = torch.tensor([[float('nan')]])
    key = torch.rand(1, 1, 64)
    value = torch.rand(1, 1, 64)

    with pytest.raises(ValueError):
        model(query, key, value)

    query = torch.tensor([[float('inf')]])
    with pytest.raises(ValueError):
        model(query, key, value)



def test_debug_logging(capsys):
    """Hata ayıklama modunun loglama çıktısını test et."""
    model = CrossAttention(embed_dim=64, num_heads=8, debug=True)
    query = torch.rand(4, 10, 64)
    key = torch.rand(4, 10, 64)
    value = torch.rand(4, 10, 64)
    model(query, key, value)

    captured = capsys.readouterr()
    assert "[DEBUG]" in captured.out

def test_scaling_strategy():
    """Ölçeklendirme stratejilerini test et."""
    model = CrossAttention(embed_dim=64, num_heads=8, scaling_strategy="linear")
    query = torch.rand(4, 10, 64)
    key = torch.rand(4, 10, 64)
    value = torch.rand(4, 10, 64)
    output, _ = model(query, key, value)
    assert output.shape == query.shape

    with pytest.raises(ValueError):
        CrossAttention(embed_dim=64, num_heads=8, scaling_strategy="unsupported_strategy")

def test_forward_invalid_shapes(default_model):
    """Forward metodu ile geçersiz giriş şekillerini test et."""
    model = default_model

    # Yanlış boyutta query
    query = torch.rand(4, 10, 32)
    key = torch.rand(4, 10, 64)
    value = torch.rand(4, 10, 64)
    with pytest.raises(ValueError):
        model(query, key, value)

    # Farklı batch boyutları
    query = torch.rand(4, 10, 64)
    key = torch.rand(5, 10, 64)
    value = torch.rand(5, 10, 64)
    with pytest.raises(ValueError):
        model(query, key, value)

    # Cihaz kontrolü (CUDA desteklenmiyorsa atlanacak)
    if torch.cuda.is_available():
        query = torch.rand(4, 10, 64).cuda()
        key = torch.rand(4, 10, 64)
        value = torch.rand(4, 10, 64)
        with pytest.raises(ValueError):
            model(query, key, value)


def test_extra_repr():
    """extra_repr metodunun doğru çıktısını test et."""
    model = CrossAttention(embed_dim=64, num_heads=8, dropout=0.2, normalization_type="layer_norm", debug=True)
    expected_repr = (
        "embed_dim=64, num_heads=8, dropout=0.2, attention_scaling=True, "
        "normalization_type=layer_norm, scaling_strategy=sqrt, debug=True"
    )
    assert model.extra_repr() == expected_repr

def test_forward_2d_key_padding_mask(default_model):
    """2D key_padding_mask ile forward metodu testi."""
    model = default_model
    query, key, value = torch.rand(4, 10, 64), torch.rand(4, 10, 64), torch.rand(4, 10, 64)
    key_padding_mask = torch.randint(0, 2, (4, 10)).bool()  # 2D key_padding_mask

    # Doğru çalıştığından emin olun
    output, attn_weights = model(query, key, value, key_padding_mask)
    assert output.shape == query.shape
    assert attn_weights.shape == (4, 10, 10)  # (batch_size, seq_len, seq_len)

def test_mask_processing(default_model):
    """Maske işleme fonksiyonunu test et."""
    model = default_model
    key_padding_mask = torch.randint(0, 2, (4, 10)).bool()
    attention_mask = torch.ones(10, 10)

    processed_key_padding_mask, processed_attention_mask = model.process_attention_masks(
        key_padding_mask, attention_mask, seq_len=10
    )

    # `MultiheadAttention` için `key_padding_mask`'in boyutunu kontrol et
    assert processed_key_padding_mask.shape == (4, 10)  # 2D boyutu kontrol et
    assert processed_attention_mask.shape == (10, 10)
