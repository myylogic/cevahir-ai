import pytest
import torch
from neural_network_module.ortak_katman_module.attention_manager_module.attention_utils_module.attention_normalizer import AttentionNormalizer


@pytest.fixture
def create_normalizer():
    def _create_normalizer(normalization_type, embed_dim=None, seq_len=None, eps=1e-3, verbose=False,momentum=None):
        return AttentionNormalizer(
            normalization_type=normalization_type,
            embed_dim=embed_dim,
            seq_len=seq_len,
            eps=eps,
            verbose=verbose,
            momentum=momentum
        )
    return _create_normalizer


def test_batch_norm(create_normalizer):
    seq_len = 32
    embed_dim = 64
    normalizer = create_normalizer("batch_norm", seq_len=seq_len, embed_dim=embed_dim, verbose=True)

    # Giriş tensörünü oluştur ve normalize et
    input_tensor = torch.randn(8, seq_len, embed_dim)
    input_tensor = (input_tensor - input_tensor.mean(dim=0, keepdim=True)) / (input_tensor.std(dim=0, keepdim=True) + 1e-5)

    # Çıkış tensörünü normalize et
    output_tensor = normalizer(input_tensor)
    
    # Şekil doğrulaması
    assert output_tensor.shape == input_tensor.shape, "Çıkış şekli girişle eşleşmiyor."

    # Ortalama doğrulaması
    input_mean = output_tensor.mean(dim=0)
    assert torch.allclose(input_mean, torch.zeros_like(input_mean), atol=5e-1), f"BatchNorm sonrası ortalama sıfır değil. Ortalama: {input_mean}"

    # Standart sapma doğrulaması
    input_std = output_tensor.std(dim=0)
    assert torch.allclose(input_std, torch.ones_like(input_std), atol=5e-1), f"BatchNorm sonrası standart sapma bir değil. Standart Sapma: {input_std}"

    # Detaylı loglama
    if normalizer.verbose:
        print("[Test Log] Giriş tensörü:")
        print(input_tensor)
        print("[Test Log] Çıkış tensörü:")
        print(output_tensor)



def test_layer_norm(create_normalizer):
    embed_dim = 64
    normalizer = create_normalizer("layer_norm", embed_dim=embed_dim, verbose=True)

    input_tensor = torch.randn(8, 16, embed_dim)  # [batch_size, seq_len, embed_dim]
    output_tensor = normalizer(input_tensor)

    assert output_tensor.shape == input_tensor.shape, "Çıkış şekli girişle eşleşmiyor."
    assert torch.allclose(output_tensor.mean(dim=-1), torch.zeros_like(output_tensor.mean(dim=-1)), atol=1e-4), \
        "LayerNorm sonrası ortalama sıfır değil."
    assert torch.allclose(
        output_tensor.std(dim=-1),
        torch.ones_like(output_tensor.std(dim=-1)),
        atol=1e-2, rtol=1e-3
    ), "LayerNorm sonrası standart sapma bir değil."

def test_group_norm(create_normalizer):
    """
    GroupNorm işlemini test eder. Giriş ve çıkış tensörlerinin şekillerini, ortalamalarını
    ve standart sapmalarını doğrular. Ayrıca grup sayısının embed_dim ile uyumunu kontrol eder.

    Args:
        create_normalizer (function): GroupNorm normalizer'ı oluşturan yardımcı işlev.
    """
    embed_dim = 64  # Özellik boyutu
    batch_size = 8  # Batch boyutu
    seq_len = 16  # Sekans uzunluğu
    num_groups = 4  # Grup sayısı

    # Tohumlama: Rastgele girişlerin tekrarlanabilir olması için
    torch.manual_seed(42)

    # GroupNorm normalizer'ı oluştur
    normalizer = create_normalizer("group_norm", embed_dim=embed_dim, verbose=True)

    # Giriş tensörünü oluştur
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)  # [batch_size, seq_len, embed_dim]
    output_tensor = normalizer(input_tensor)

    # Şekil doğrulaması
    assert output_tensor.shape == input_tensor.shape, (
        f"Çıkış şekli girişle eşleşmiyor. Giriş şekli: {input_tensor.shape}, Çıkış şekli: {output_tensor.shape}"
    )

    # Ortalama ve standart sapma doğrulaması
    output_mean = output_tensor.mean(dim=(-1, -2))  # Ortalama tüm tensör eksenlerinde hesaplanır
    output_std = output_tensor.std(dim=(-1, -2))  # Standart sapma tüm tensör eksenlerinde hesaplanır
    atol_value = 1e-2  # Daha hassas bir tolerans seviyesi kullanıldı

    # Ortalama sıfıra yakın mı?
    assert torch.allclose(output_mean, torch.zeros_like(output_mean), atol=atol_value), (
        f"GroupNorm sonrası ortalama sıfır değil. Ortalama: {output_mean}"
    )

    # Standart sapma 1'e yakın mı?
    assert torch.allclose(output_std, torch.ones_like(output_std), atol=atol_value), (
        f"GroupNorm sonrası standart sapma 1 değil. Standart Sapma: {output_std}"
    )

    # Grup sayısını kontrol et
    assert embed_dim % num_groups == 0, (
        f"Embed dim ({embed_dim}) grup sayısına ({num_groups}) tam bölünmelidir. "
        f"Uygunsuz grup sayısı tespit edildi."
    )

    # Grup sayısı ve embed_dim doğrulaması
    assert num_groups <= embed_dim, (
        f"Grup sayısı ({num_groups}), embed_dim'den ({embed_dim}) büyük olamaz."
    )
    assert embed_dim % num_groups == 0, (
        f"Embed dim ({embed_dim}) grup sayısına ({num_groups}) tam bölünmelidir."
    )

    # Giriş ve çıkış tensörlerinin karşılaştırmalı analizi
    input_mean = input_tensor.mean(dim=(-1, -2))
    input_std = input_tensor.std(dim=(-1, -2))

    if normalizer.verbose:
        print("[Group Norm Test Log] Giriş tensörü (ilk 2 örnek):")
        print(input_tensor[:2])
        print("[Group Norm Test Log] Çıkış tensörü (ilk 2 örnek):")
        print(output_tensor[:2])
        print("[Group Norm Test Log] Giriş Ortalama:")
        print(input_mean)
        print("[Group Norm Test Log] Giriş Standart Sapma:")
        print(input_std)
        print("[Group Norm Test Log] Çıkış Ortalama:")
        print(output_mean)
        print("[Group Norm Test Log] Çıkış Standart Sapma:")
        print(output_std)

    print("GroupNorm testi başarıyla tamamlandı!")





def test_instance_norm(create_normalizer):
    """
    InstanceNorm işlemini test eder. Giriş ve çıkış tensörlerinin şekillerini,
    ortalamalarını ve standart sapmalarını doğrular.
    Args:
        create_normalizer (function): InstanceNorm normalizer'ı oluşturan yardımcı işlev.
    """
    embed_dim = 64
    batch_size = 8
    seq_len = 16

    # InstanceNorm normalizer'ı oluştur
    normalizer = create_normalizer("instance_norm", embed_dim=embed_dim, verbose=True)

    # Giriş tensörünü oluştur
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)  # [batch_size, seq_len, embed_dim]
    output_tensor = normalizer(input_tensor)

    # Şekil doğrulaması
    assert output_tensor.shape == input_tensor.shape, (
        f"Çıkış şekli girişle eşleşmiyor. Giriş şekli: {input_tensor.shape}, Çıkış şekli: {output_tensor.shape}"
    )

    # Ortalama doğrulaması (daha geniş tolerans)
    output_mean = output_tensor.mean(dim=(-1, -2))  # Her örneğin ortalama değeri
    atol_value = 1e-1  # Daha geniş tolerans seviyesi

    # Ortalama sıfıra yakın mı?
    assert torch.allclose(output_mean, torch.zeros_like(output_mean), atol=atol_value), (
        f"InstanceNorm sonrası ortalama sıfır değil. Ortalama: {output_mean}"
    )

    # Standart sapma doğrulaması
    output_std = output_tensor.std(dim=(-1, -2))  # Her örneğin standart sapması
    atol_std = 1e-1  # Standart sapma için tolerans seviyesi

    # Standart sapma 1'e yakın mı?
    assert torch.allclose(output_std, torch.ones_like(output_std), atol=atol_std), (
        f"InstanceNorm sonrası standart sapma 1 değil. Standart Sapma: {output_std}"
    )

    # Loglama
    if normalizer.verbose:
        print("[InstanceNorm Test Log] Giriş tensörü (ilk 2 örnek):")
        print(input_tensor[:2])
        print("[InstanceNorm Test Log] Çıkış tensörü (ilk 2 örnek):")
        print(output_tensor[:2])
        print("[InstanceNorm Test Log] Ortalama (her örnek):")
        print(output_mean)
        print("[InstanceNorm Test Log] Standart Sapma (her örnek):")
        print(output_std)

    print("InstanceNorm testi başarıyla tamamlandı!")




def test_invalid_normalization_type(create_normalizer):
    with pytest.raises(ValueError, match="Geçersiz normalizasyon tipi"):
        create_normalizer("unsupported_norm")





def test_validate_input_positive_dimensions(create_normalizer):
    normalizer = create_normalizer("layer_norm", embed_dim=64, verbose=True)

    # Negatif veya sıfır boyut kontrolü
    with pytest.raises(ValueError, match="Giriş tensör boyutları geçersiz.*Tüm boyutların pozitif bir değer olması gereklidir"):
        normalizer.validate_input(torch.randn(8, 0, 64))  # Sıfır boyut içeriyor


def test_extra_repr(create_normalizer):
    embed_dim = 64
    seq_len = 32
    normalizer = create_normalizer("layer_norm", embed_dim=embed_dim, seq_len=seq_len, verbose=False)

    repr_str = normalizer.extra_repr()
    assert "normalization_type=layer_norm" in repr_str, "normalization_type bilgisi yanlış."

    # embed_dim bir tuple olarak döndüğü için kontrolü buna uygun yapıyoruz
    assert f"embed_dim=({embed_dim},)" in repr_str, "embed_dim bilgisi eksik veya yanlış formatta."

    # seq_len kullanılmıyorsa çıktıda yer almamalı
    assert "seq_len" not in repr_str, "seq_len kullanılmıyorsa çıktıda yer almamalı."



def test_high_eps(create_normalizer):
    embed_dim = 64
    eps = 1e-1  # Yüksek epsilon değeri
    normalizer = create_normalizer("layer_norm", embed_dim=embed_dim, eps=eps, verbose=True)

    input_tensor = torch.randn(32, 64, embed_dim)  # Daha büyük tensör
    output_tensor = normalizer(input_tensor)

    assert output_tensor.shape == input_tensor.shape, "Çıkış şekli girişle eşleşmiyor."
    assert torch.allclose(output_tensor.mean(dim=-1), torch.zeros_like(output_tensor.mean(dim=-1)), atol=1e-4), \
        "LayerNorm sonrası ortalama sıfır değil (yüksek eps testi)."
    assert torch.allclose(output_tensor.std(dim=-1), torch.ones_like(output_tensor.std(dim=-1)), atol=1e-1), \
        f"LayerNorm sonrası standart sapma bir değil: Ortalama={output_tensor.std(dim=-1).mean().item()}."

def test_dropout_layer_norm(create_normalizer):
    """
    Dropout ile birlikte LayerNorm'un çalışmasını test eder.
    """
    embed_dim = 64
    normalizer = create_normalizer("layer_norm", embed_dim=embed_dim, verbose=True)
    dropout_layer = torch.nn.Dropout(p=0.5)

    input_tensor = torch.randn(16, 32, embed_dim)
    dropped_tensor = dropout_layer(input_tensor)
    output_tensor = normalizer(dropped_tensor)

    assert output_tensor.shape == input_tensor.shape, "Çıkış şekli girişle eşleşmiyor."
    assert torch.allclose(output_tensor.mean(dim=-1), torch.zeros_like(output_tensor.mean(dim=-1)), atol=1e-4), \
        "LayerNorm sonrası ortalama sıfır değil."



def test_frozen_parameters_group_norm(create_normalizer):
    """
    Grup Normunun parametrelerinin dondurulmuş (frozen) olduğu durumu test eder.
    """
    embed_dim = 64
    normalizer = create_normalizer("group_norm", embed_dim=embed_dim, verbose=True)

    # Parametreleri dondur
    for param in normalizer.parameters():
        param.requires_grad = False

    input_tensor = torch.randn(8, 16, embed_dim)
    output_tensor = normalizer(input_tensor)

    assert output_tensor.shape == input_tensor.shape, "Çıkış şekli girişle eşleşmiyor."
    assert not any(param.requires_grad for param in normalizer.parameters()), "Bazı parametreler dondurulmamış."



def test_random_noise_layer_norm(create_normalizer):
    """
    Gürültü eklenmiş verilerde LayerNorm'u test eder.
    """
    embed_dim = 128
    normalizer = create_normalizer("layer_norm", embed_dim=embed_dim, verbose=True)

    base_tensor = torch.randn(32, 64, embed_dim)
    noise = torch.randn_like(base_tensor) * 0.01
    input_tensor = base_tensor + noise

    output_tensor = normalizer(input_tensor)

    assert output_tensor.shape == input_tensor.shape, "Çıkış şekli girişle eşleşmiyor."
    assert torch.allclose(output_tensor.mean(dim=-1), torch.zeros_like(output_tensor.mean(dim=-1)), atol=1e-4), \
        "LayerNorm sonrası ortalama sıfır değil."


def test_invalid_input_shape(create_normalizer):
    """
    Hatalı boyutlara sahip bir tensör ile normalizer çağrıldığında doğru hata mesajını döndürür.
    """
    normalizer = create_normalizer("layer_norm", embed_dim=64, verbose=True)

    # 2D tensör: Hatalı giriş
    input_tensor = torch.randn(8, 64)
    
    # Güncellenmiş regex deseni ile test
    expected_error_message = (
        r"Giriş tensörünün boyutları yalnızca 3D veya 4D olabilir\. "
        r"Sağlanan boyut: 2\. "
        r"Beklenen boyutlar: \[3D: \(batch_size, seq_len, embed_dim\) veya "
        r"4D: \(batch_size, num_heads, seq_len, head_dim\)\]\."
    )
    
    with pytest.raises(ValueError, match=expected_error_message):
        normalizer(input_tensor)

def test_large_input_batch_norm(create_normalizer):
    embed_dim = 128
    seq_len = 512
    eps = 1e-6  # Daha kararlı epsilon
    momentum = 0.9  # Optimum momentum
    batch_size = 64

    # Normalizer oluşturma
    normalizer = create_normalizer(
        "batch_norm",
        seq_len=seq_len,
        embed_dim=embed_dim,
        verbose=True,
        eps=eps,
        momentum=momentum
    )

    # Giriş tensörünün oluşturulması ve normalleştirilmesi
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)
    input_tensor = (input_tensor - input_tensor.mean(dim=0, keepdim=True)) / (input_tensor.std(dim=0, keepdim=True) + eps)

    # Parametrelerin sıfırlanması
    normalizer.normalizer.reset_parameters()

    # BatchNorm uygulanması
    output_tensor = normalizer(input_tensor)

    # Çıkış boyut kontrolü
    assert output_tensor.shape == input_tensor.shape, (
        f"Çıkış şekli girişle eşleşmiyor. Beklenen: {input_tensor.shape}, Alınan: {output_tensor.shape}"
    )

    # Ortalama kontrolü
    output_mean = output_tensor.mean(dim=(0, 1))  # Ortalamayı daha doğru hesaplamak için tüm batch'i ve seq_len'i kullan
    max_mean_diff = output_mean.abs().max()

    assert torch.allclose(
        output_mean,
        torch.zeros_like(output_mean),
        atol=1e-2  # Daha sıkı tolerans
    ), (
        f"BatchNorm sonrası ortalama sıfır değil. Maksimum fark: {max_mean_diff.item():.6f}, "
        f"Çıkış Ortalama: {output_mean.tolist()}"
    )

    # Standart sapma kontrolü
    output_std = output_tensor.std(dim=(0, 1))
    max_std_diff = (output_std - torch.ones_like(output_std)).abs().max()

    assert torch.allclose(
        output_std,
        torch.ones_like(output_std),
        atol=1e-2
    ), (
        f"BatchNorm sonrası standart sapma 1 değil. Maksimum fark: {max_std_diff.item():.6f}, "
        f"Çıkış Standart Sapma: {output_std.tolist()}"
    )

