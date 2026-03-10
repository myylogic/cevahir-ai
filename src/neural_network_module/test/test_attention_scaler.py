import pytest
import torch
from neural_network_module.ortak_katman_module.attention_manager_module.attention_utils_module.attention_scaler import AttentionScaler


@pytest.fixture
def create_attention_scaler():
    """
    AttentionScaler oluşturucu fixture.
    """
    def _create_attention_scaler(scale_factor=1.0, clip_range=None, verbose=False, num_heads=None):
        return AttentionScaler(scale_factor=scale_factor, clip_range=clip_range, verbose=verbose, num_heads=num_heads)
    return _create_attention_scaler


def test_attention_scaler_basic(create_attention_scaler):
    """
    Temel ölçekleme testini yapar.
    """
    scaler = create_attention_scaler(scale_factor=2.0, verbose=True, num_heads=8)
    input_tensor = torch.randn(16, 64, 128)  # 3D tensör
    output_tensor = scaler(input_tensor)

    assert output_tensor.shape == input_tensor.shape, (
        f"Çıkış boyutu beklenenle eşleşmiyor. "
        f"Beklenen: {input_tensor.shape}, Alınan: {output_tensor.shape}"
    )
    assert torch.allclose(output_tensor, input_tensor * 2.0, atol=1e-6), (
        "Çıkış tensörü beklenen ölçeklenmiş tensörle eşleşmiyor."
    )


def test_attention_scaler_with_clipping(create_attention_scaler):
    """
    Clip range ile ölçekleme testini yapar.
    """
    scaler = create_attention_scaler(scale_factor=1.5, clip_range=(-1.0, 1.0), verbose=True, num_heads=8)
    input_tensor = torch.randn(8, 32, 64)  # 3D tensör
    output_tensor = scaler(input_tensor)

    assert output_tensor.shape == input_tensor.shape, (
        f"Çıkış boyutu beklenenle eşleşmiyor. "
        f"Beklenen: {input_tensor.shape}, Alınan: {output_tensor.shape}"
    )
    assert output_tensor.min().item() >= -1.0, "Çıkış tensöründeki minimum değer clip range alt limitinden küçük."
    assert output_tensor.max().item() <= 1.0, "Çıkış tensöründeki maksimum değer clip range üst limitinden büyük."


def test_attention_scaler_3d_to_4d_conversion(create_attention_scaler):
    """
    3D tensörlerin 4D'ye dönüştürülmesini test eder.
    """
    scaler = create_attention_scaler(scale_factor=1.0, verbose=True, num_heads=8)
    input_tensor = torch.randn(16, 64, 128)  # [batch_size, seq_len, embed_dim]
    output_tensor = scaler(input_tensor)

    assert output_tensor.shape == input_tensor.shape, (
        f"Çıkış boyutu beklenenle eşleşmiyor. "
        f"Beklenen: {input_tensor.shape}, Alınan: {output_tensor.shape}"
    )


def test_attention_scaler_invalid_tensor(create_attention_scaler):
    """
    Geçersiz tensör girişlerini test eder.
    """
    scaler = create_attention_scaler(scale_factor=1.0, num_heads=8)
    with pytest.raises(TypeError):
        scaler("not_a_tensor")


def test_attention_scaler_invalid_scale_factor(create_attention_scaler):
    """
    Geçersiz scale_factor değerlerini test eder.
    """
    with pytest.raises(ValueError):
        create_attention_scaler(scale_factor=-1.0)


def test_attention_scaler_with_invalid_num_heads(create_attention_scaler):
    """
    Geçersiz num_heads değerlerini test eder.
    """
    with pytest.raises(ValueError):
        create_attention_scaler(num_heads=0)


def test_attention_scaler_with_high_scale(create_attention_scaler):
    """
    Yüksek scale_factor ile dikkat tensörlerinin ölçeklenmesini test eder.
    """
    scaler = create_attention_scaler(scale_factor=10.0, verbose=True, num_heads=8)
    input_tensor = torch.randn(4, 32, 64)
    output_tensor = scaler(input_tensor)

    assert torch.allclose(output_tensor, input_tensor * 10.0, atol=1e-6), (
        "Çıkış tensörü beklenen ölçeklenmiş tensörle eşleşmiyor."
    )


def test_attention_scaler_with_edge_clip(create_attention_scaler):
    """
    Kırpma sınırlarının çalışmasını test eder.
    """
    scaler = create_attention_scaler(scale_factor=2.0, clip_range=(0.0, 1.0), verbose=True, num_heads=8)
    input_tensor = torch.full((4, 8, 16), 0.5)  # Sabit bir tensör
    output_tensor = scaler(input_tensor)

    assert torch.allclose(output_tensor, torch.clamp(input_tensor * 2.0, 0.0, 1.0), atol=1e-6), (
        "Çıkış tensörü beklenen kırpılmış tensörle eşleşmiyor."
    )


def test_attention_scaler_4d_to_3d_conversion(create_attention_scaler):
    """
    4D tensörlerin 3D'ye dönüştürülmesini test eder.
    """
    scaler = create_attention_scaler(scale_factor=1.0, verbose=True, num_heads=8)
    input_tensor = torch.randn(16, 8, 64, 16)  # [batch_size, num_heads, seq_len, head_dim]
    output_tensor = scaler(input_tensor)

    assert output_tensor.shape == input_tensor.shape, (
        f"Çıkış boyutu beklenenle eşleşmiyor. "
        f"Beklenen: {input_tensor.shape}, Alınan: {output_tensor.shape}"
    )


def test_attention_scaler_extreme_values(create_attention_scaler):
    """
    Çok büyük veya küçük tensör değerlerinde ölçeklemeyi test eder.
    """
    scaler = create_attention_scaler(scale_factor=0.1, verbose=True, num_heads=8)
    input_tensor = torch.randn(4, 32, 64) * 1e6
    output_tensor = scaler(input_tensor)

    assert torch.allclose(output_tensor, input_tensor * 0.1, atol=1e-3), (
        "Çıkış tensörü beklenen ölçeklenmiş tensörle eşleşmiyor."
    )


def test_attention_scaler_extra_repr(create_attention_scaler):
    """
    Sınıfın extra_repr özelliğini test eder.
    """
    scaler = create_attention_scaler(scale_factor=1.5, clip_range=(-2.0, 2.0), verbose=True, num_heads=8)
    extra_repr = scaler.extra_repr()

    assert "scale_factor=1.5" in extra_repr
    assert "clip_range=(-2.0, 2.0)" in extra_repr
    assert "verbose=True" in extra_repr
    assert "num_heads=8" in extra_repr
