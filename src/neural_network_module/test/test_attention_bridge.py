import torch
import pytest
from neural_network_module.ortak_katman_module.attention_bridge import AttentionBridge

@pytest.fixture
def attention_bridge():
    """ AttentionBridge modülünün örneğini döndürür. """
    return AttentionBridge(projection_dim=2048)

def test_attention_bridge_shape(attention_bridge):
    """ AttentionBridge giriş ve çıkış tensörlerinin boyutlarını doğrular. """
    input_tensor = torch.randn(8, 2048, 2048)  # Örnek giriş tensörü
    output_tensor = attention_bridge(input_tensor)
    
    assert output_tensor.shape == input_tensor.shape, \
        f"Beklenen şekil {input_tensor.shape}, ancak {output_tensor.shape} elde edildi."

def test_attention_bridge_no_nan(attention_bridge):
    """ AttentionBridge çıktısının NaN içermediğini test eder. """
    input_tensor = torch.randn(8, 2048, 2048)
    output_tensor = attention_bridge(input_tensor)
    
    assert not torch.isnan(output_tensor).any(), "AttentionBridge çıktısı NaN içeriyor!"

def test_attention_bridge_consistency(attention_bridge):
    """ Aynı giriş için aynı çıktıyı üretiyor mu test eder. """
    input_tensor = torch.randn(8, 2048, 2048)
    output_1 = attention_bridge(input_tensor)
    output_2 = attention_bridge(input_tensor)
    
    assert torch.allclose(output_1, output_2, atol=1e-6), "AttentionBridge tutarsız çıktı üretiyor."

def test_attention_bridge_extreme_values(attention_bridge):
    """ Büyük ve küçük değerler içeren girişler için stabil mi test eder. """
    large_values = torch.full((8, 2048, 2048), 1e6)
    small_values = torch.full((8, 2048, 2048), 1e-6)
    
    output_large = attention_bridge(large_values)
    output_small = attention_bridge(small_values)

    assert not torch.isnan(output_large).any(), "Büyük değerler AttentionBridge tarafından yanlış işlendi."
    assert not torch.isnan(output_small).any(), "Küçük değerler AttentionBridge tarafından yanlış işlendi."

