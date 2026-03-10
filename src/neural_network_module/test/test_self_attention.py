import pytest
import torch
from neural_network_module.ortak_katman_module.attention_manager_module.self_attention import SelfAttention 


@pytest.fixture
def attention_model():
    """Varsayılan parametrelerle SelfAttention modeli oluşturur."""
    return SelfAttention(embed_dim=64, num_heads=4, normalization_type="layer_norm", debug=True)


def test_model_initialization():
    """Modelin doğru şekilde başlatıldığını test eder."""
    model = SelfAttention(embed_dim=64, num_heads=4, normalization_type="layer_norm")
    assert model.embed_dim == 64
    assert model.num_heads == 4
    assert model.head_dim == 16
    assert model.normalization_type == "layer_norm"


def test_forward_pass(attention_model):
    """Forward geçişin doğru çalıştığını test eder."""
    model = attention_model
    x = torch.randn(8, 10, 64)  # [batch_size, seq_len, embed_dim]
    mask = torch.randint(0, 2, (8, 1, 1, 10), dtype=torch.bool)  # [batch_size, 1, 1, seq_len]

    output = model(x, mask)
    assert output.shape == (8, 10, 64)
    assert torch.isnan(output).sum() == 0  # NaN kontrolü
    assert torch.isinf(output).sum() == 0  # Sonsuzluk kontrolü



def test_mask_application(attention_model):
    """Maske uygulamasının doğru çalıştığını test eder."""
    model = attention_model
    x = torch.randn(8, 10, 64)
    mask = torch.ones(8, 1, 1, 10, dtype=torch.bool)  # Tam maske

    output = model(x, mask)
    assert output.shape == (8, 10, 64)

    mask[:, :, :, 5:] = 0  # Maskeyi kısmi hale getir
    output = model(x, mask)
    assert output.shape == (8, 10, 64)

def test_forward_pass_shape():
    """Girdi tensörü için doğru çıkış boyutunu doğrular."""
    model = SelfAttention(embed_dim=64, num_heads=4)
    x = torch.randn(8, 10, 64)
    mask = torch.ones(8, 1, 1, 10, dtype=torch.bool)

    output = model(x, mask)
    assert output.shape == (8, 10, 64)


def test_forward_pass_no_mask():
    """Maske kullanılmadan doğru çalışmayı test eder."""
    model = SelfAttention(embed_dim=64, num_heads=4)
    x = torch.randn(8, 10, 64)

    output = model(x)
    assert output.shape == (8, 10, 64)

def test_invalid_initialization():
    """Geçersiz parametrelerin hata verdiğini test eder."""

    # Gömme boyutu ve çok başlık sayısının tam bölünememesi
    with pytest.raises(ValueError, match=r"Gömme boyutu \(65\) çok başlık sayısına \(4\) tam bölünemiyor\. Gömme boyutunun çok başlık sayısına bölünebilir bir değer olması gerekir\."):
        SelfAttention(embed_dim=65, num_heads=4)

    # Geçersiz normalizasyon türü
    with pytest.raises(ValueError, match=r"Geçersiz normalizasyon tipi: invalid_norm\. Desteklenen tipler: \['layer_norm', 'batch_norm', 'group_norm', 'instance_norm'\]"):
        SelfAttention(embed_dim=64, num_heads=4, normalization_type="invalid_norm")

    # Geçersiz num_groups değeri
    with pytest.raises(ValueError, match=r"GroupNorm için num_groups pozitif bir tam sayı olmalıdır\. Verilen: -1"):
        SelfAttention(embed_dim=64, num_heads=4, normalization_type="group_norm", num_groups=-1)

    # embed_dim ve num_groups'un tam bölünememesi
    with pytest.raises(ValueError, match=r"GroupNorm için embed_dim \(64\) num_groups \(5\) ile tam bölünemiyor\."):
        SelfAttention(embed_dim=64, num_heads=4, normalization_type="group_norm", num_groups=5)

    # Geçersiz eps değeri
    with pytest.raises(ValueError, match=r"LayerNorm için eps pozitif bir sayı olmalıdır\. Verilen: -0\.001"):
        SelfAttention(embed_dim=64, num_heads=4, normalization_type="layer_norm", eps=-0.001)



def test_nan_handling():
    """NaN ve sonsuz değerlerin doğru şekilde işlendiğini test eder."""
    model = SelfAttention(embed_dim=64, num_heads=4, debug=True)
    x = torch.randn(8, 10, 64)
    x[0, 0, 0] = float("nan")  # NaN ekle
    x[0, 1, 0] = float("inf")  # Sonsuz ekle

    try:
        # Model çıktısını al
        output = model(x)

        # Çıkışta NaN/Sonsuz olup olmadığını kontrol et
        assert not torch.isnan(output).any(), "Model çıktısında NaN değerler var."
        assert not torch.isinf(output).any(), "Model çıktısında sonsuz değerler var."

    except AssertionError as e:
        print(f"[Error] Test başarısız: {e}")
        raise

    except RuntimeError as e:
        # Eğer model NaN veya inf üretiyorsa hata yükseltmesini bekleyebiliriz
        assert "NaN" in str(e) or "sonsuz" in str(e), f"Beklenmeyen hata mesajı: {e}"



@pytest.mark.parametrize("normalization_type", ["layer_norm", "batch_norm", "group_norm", "instance_norm"])
def test_different_normalization_types(normalization_type):
    """Farklı normalizasyon türlerinin desteklendiğini test eder."""
    model = SelfAttention(embed_dim=64, num_heads=4, normalization_type=normalization_type, num_groups=4)
    x = torch.randn(8, 10, 64)
    output = model(x)
    assert output.shape == (8, 10, 64)

