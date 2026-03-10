import pytest
import torch
import torch.nn as nn
import logging
import pickle
from typing import List, Tuple

from src.neural_network import CevahirNeuralNetwork

# **Örnek Yapılandırma**
CONFIG = {
    "vocab_size": 150_000,
    "embed_dim": 64,
    "seq_proj_dim": 2048,
    "num_heads": 16,
    "dropout": 0.1,
    "learning_rate": 1e-5,
    "attention_type": "multi_head",
    "normalization_type": "layer_norm",
    # ✅ YENİ (V-2): Yeni parametreler
    "num_layers": 12,
    "ffn_dim": None,  # None ise 4x seq_proj_dim
    "pre_norm": True,
    "causal_mask": True,
}

# **Logger Ayarı**
logger = logging.getLogger("test_neural_network")
logger.setLevel(logging.DEBUG)


# ---- Fake TensorBoard Writer (Protocol uyumlu) ----
class FakeWriter:
    def __init__(self):
        self.scalars: List[Tuple[str, float, int]] = []
        self.hists: List[Tuple[str, torch.Size, int]] = []
        self.images: List[Tuple[str, torch.Size, int]] = []
        self.closed = False

    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0):
        # torch.tensor vs float gelebilir
        try:
            val = float(scalar_value)
        except Exception:
            val = float(torch.as_tensor(scalar_value).item())
        self.scalars.append((tag, val, global_step))

    def add_histogram(self, tag: str, values, global_step: int = 0):
        shape = torch.as_tensor(values).shape
        self.hists.append((tag, shape, global_step))

    def add_image(self, tag: str, img_tensor, global_step: int = 0):
        shape = torch.as_tensor(img_tensor).shape
        self.images.append((tag, shape, global_step))

    def close(self):
        self.closed = True


# ---- Fixtures ----
@pytest.fixture
def neural_network():
    """
    Varsayılan bir Cevahir Neural Network örneği oluşturur.
    ✅ GÜNCELLENDİ (V-2): Yeni parametreler eklendi.
    """
    return CevahirNeuralNetwork(
        vocab_size=CONFIG["vocab_size"],
        embed_dim=CONFIG["embed_dim"],
        seq_proj_dim=CONFIG["seq_proj_dim"],
        num_heads=CONFIG["num_heads"],
        attention_type=CONFIG["attention_type"],
        normalization_type=CONFIG["normalization_type"],
        dropout=CONFIG["dropout"],
        learning_rate=CONFIG["learning_rate"],
        # ✅ YENİ (V-2): Yeni parametreler
        num_layers=CONFIG["num_layers"],
        ffn_dim=CONFIG["ffn_dim"],
        pre_norm=CONFIG["pre_norm"],
        causal_mask=CONFIG["causal_mask"],
    )

@pytest.fixture
def neural_network_tb():
    """
    TensorBoard writer enjekte edilmiş örnek (her adımda loglar).
    ✅ GÜNCELLENDİ (V-2): Yeni parametreler eklendi.
    """
    fw = FakeWriter()
    model = CevahirNeuralNetwork(
        vocab_size=CONFIG["vocab_size"],
        embed_dim=CONFIG["embed_dim"],
        seq_proj_dim=CONFIG["seq_proj_dim"],
        num_heads=CONFIG["num_heads"],
        attention_type=CONFIG["attention_type"],
        normalization_type=CONFIG["normalization_type"],
        dropout=CONFIG["dropout"],
        learning_rate=CONFIG["learning_rate"],
        # ✅ YENİ (V-2): Yeni parametreler
        num_layers=CONFIG["num_layers"],
        ffn_dim=CONFIG["ffn_dim"],
        pre_norm=CONFIG["pre_norm"],
        causal_mask=CONFIG["causal_mask"],
        use_tensorboard=False,           # dahili writer olmasın
        tb_writer=fw,                    # harici fake writer
        tb_log_every_n=1,                # her adım log
        tb_log_histograms=True,
        tb_log_attention_image=True,
    )
    return model, fw


# ---- Temel Başlatma Testi ----
def test_initialization(neural_network):
    logger.info("[TEST] Model bileşen başlatma testi başladı.")

    assert neural_network.dil_katmani is not None, "DilKatmani başlatılamadı!"
    # ✅ V-2: layer_processor ve tensor_processing_manager deprecated, yerine layers kullanılıyor
    assert hasattr(neural_network, 'layers'), "TransformerEncoderLayer'lar (layers) başlatılamadı!"
    assert len(neural_network.layers) > 0, "En az bir TransformerEncoderLayer olmalı!"
    assert neural_network.memory_manager is not None, "MemoryManager başlatılamadı!"
    assert neural_network.output_layer is not None, "OutputLayer başlatılamadı!"

    logger.info("[TEST] Model bileşen başlatma testi başarılı.")


# ---- Forward Pass ----
def test_forward_pass(neural_network):
    logger.info("[TEST] İleri yayılım testi başladı.")
    torch.manual_seed(0)

    batch_size, seq_len = 2, 10
    input_tensor = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    output, attn_weights = neural_network(input_tensor)

    # ** Çıktının Tensor Olduğunu Doğrula**
    assert isinstance(output, torch.Tensor), "Çıktı bir tensör olmalı!"
    assert isinstance(attn_weights, torch.Tensor) or attn_weights is None, "Attention weights yanlış türde!"

    # ** Çıktı Boyutu**
    expected_output_shape = (batch_size, seq_len, CONFIG["vocab_size"])
    assert output.shape == torch.Size(expected_output_shape), \
        f"Çıktı boyutu hatalı! Beklenen: {expected_output_shape}, Gerçek: {output.shape}"

    # ** Veri Türü**
    assert output.dtype == torch.float32, f"Çıktı veri türü hatalı! Beklenen: float32, Gerçek: {output.dtype}"

    logger.info("[TEST] İleri yayılım testi başarıyla tamamlandı (V-2).")


# ✅ YENİ (V-2): Yeni parametreler için testler
def test_v2_new_parameters():
    """V-2 yeni parametreler testi"""
    logger.info("[TEST] V-2 yeni parametreler testi başladı.")
    
    model = CevahirNeuralNetwork(
        learning_rate=1e-4,
        dropout=0.1,
        vocab_size=1000,
        embed_dim=128,
        seq_proj_dim=128,
        num_heads=8,
        num_layers=6,  # ✅ YENİ
        ffn_dim=512,  # ✅ YENİ
        pre_norm=True,  # ✅ YENİ
        causal_mask=True,  # ✅ YENİ
        log_level=logging.WARNING,
    )
    
    assert model.num_layers == 6
    assert model.causal_mask is True
    assert len(model.layers) == 6
    
    # Forward pass testi
    x = torch.randint(0, 1000, (2, 10))
    output, attn_weights = model(x)
    assert output.shape == (2, 10, 1000)
    
    logger.info("[TEST] V-2 yeni parametreler testi başarılı.")


def test_v2_causal_mask():
    """V-2 causal mask testi"""
    logger.info("[TEST] V-2 causal mask testi başladı.")
    
    model = CevahirNeuralNetwork(
        learning_rate=1e-4,
        dropout=0.1,
        vocab_size=1000,
        embed_dim=128,
        seq_proj_dim=128,
        num_heads=8,
        num_layers=3,
        causal_mask=True,
        log_level=logging.WARNING,
    )
    
    x = torch.randint(0, 1000, (2, 10))
    
    # Causal mask ile
    output1, _ = model(x, causal_mask=True)
    # Causal mask olmadan
    output2, _ = model(x, causal_mask=False)
    
    assert output1.shape == output2.shape
    # Farklı olmalı
    assert not torch.allclose(output1, output2, atol=1e-5)
    
    logger.info("[TEST] V-2 causal mask testi başarılı.")


def test_v2_layer_stacking():
    """V-2 layer stacking testi"""
    logger.info("[TEST] V-2 layer stacking testi başladı.")
    
    for num_layers in [1, 3, 6, 12]:
        model = CevahirNeuralNetwork(
            learning_rate=1e-4,
            dropout=0.1,
            vocab_size=1000,
            embed_dim=128,
            seq_proj_dim=128,
            num_heads=8,
            num_layers=num_layers,
            log_level=logging.WARNING,
        )
        
        assert len(model.layers) == num_layers
        
        x = torch.randint(0, 1000, (2, 10))
        output, _ = model(x)
        assert output.shape == (2, 10, 1000)
    
    logger.info("[TEST] V-2 layer stacking testi başarılı.")


# ---- Attention Mekanizması ----
def test_attention_mechanism(neural_network):
    logger.info("[TEST] Dikkat mekanizması testi başladı.")
    torch.manual_seed(0)

    batch_size, seq_len = 2, 10
    input_tensor = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    _, attn_weights = neural_network(input_tensor)

    if attn_weights is not None:
        expected_attn_shape = (batch_size, CONFIG["num_heads"], seq_len, seq_len)
        assert attn_weights.shape == torch.Size(expected_attn_shape), \
            f"Dikkat ağırlıklarının boyutu hatalı! Beklenen: {expected_attn_shape}, Gerçek: {attn_weights.shape}"

        # ** 0-1 Aralığı**
        assert torch.all((attn_weights >= 0) & (attn_weights <= 1)), "Dikkat ağırlıkları 0-1 aralığında değil!"

    logger.info("[TEST] Dikkat mekanizması testi başarıyla tamamlandı.")


# ---- Memory Manager ----
def test_memory_manager(neural_network):
    logger.info("[TEST] Bellek yönetimi testi başladı.")
    torch.manual_seed(0)

    batch_size, seq_len = 2, 10
    input_tensor = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    output, _ = neural_network(input_tensor)

    stored_output = neural_network.memory_manager.retrieve("final_output")

    assert stored_output is not None, "Bellek yöneticisi çıktıyı kaydetmedi!"
    assert torch.equal(output, stored_output), "MemoryManager çıktısı ile model çıktısı eşleşmiyor!"

    logger.info("[TEST] Bellek yönetimi testi başarıyla tamamlandı.")


# ---- Snapshot (Panel için) ----
def test_snapshot_contents(neural_network):
    torch.manual_seed(0)
    batch_size, seq_len = 2, 12
    input_tensor = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    _ = neural_network(input_tensor)
    snap = neural_network.get_last_snapshot()

    assert "embedded" in snap and "attention_output" in snap and "final_output" in snap, "Snapshot ana anahtarlar eksik!"
    assert "timings" in snap, "Snapshot zaman bilgileri eksik!"
    assert tuple(snap["final_output"]["shape"]) == (batch_size, seq_len, CONFIG["vocab_size"]), "Snapshot shape yanlış!"
    # basit istatistikler
    fo_stats = snap["final_output"]["stats"]
    assert all(k in fo_stats for k in ("min", "max", "mean", "std")), "Snapshot istatistikleri eksik!"


# ---- TensorBoard Logging ----
def test_tensorboard_logging(neural_network_tb):
    model, fw = neural_network_tb
    torch.manual_seed(0)

    batch_size, seq_len = 2, 10
    x = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    # İlk çağrıda log atılmalı (tb_log_every_n=1)
    _ = model(x)

    # En azından bazı scalar/histogram logları beklenir
    assert any("01_embedded" in tag for tag, _, _ in fw.scalars), "Embedded scalar logları yok!"
    assert any("04_final_output" in tag for tag, _, _ in fw.scalars), "Final output scalar logları yok!"
    assert any("hist" in tag for tag, _, _ in fw.hists), "Histogram loglanmadı!"

    # Attention görüntüsü (multi-head ise)
    if any(tag.startswith("attn/heatmap") for tag, _, _ in fw.images):
        h, w = fw.images[0][1][-2], fw.images[0][1][-1]
        assert h > 0 and w > 0, "Attention image boyutu anlamsız!"


# ---- Gradient Logging (train) ----
def test_gradient_logging(neural_network_tb):
    model, fw = neural_network_tb
    torch.manual_seed(0)
    model.train()

    batch_size, seq_len = 2, 8
    x = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)
    y = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    out, _ = model(x)  # (B, T, V)
    crit = nn.CrossEntropyLoss()
    loss = crit(out.reshape(-1, CONFIG["vocab_size"]), y.reshape(-1))
    loss.backward()

    prev_scalar_count = len(fw.scalars)
    model.log_gradients()  # fake writer'a gider
    new_scalar_count = len(fw.scalars)

    assert new_scalar_count > prev_scalar_count, "Gradient logları eklenmedi!"
    assert any(tag.startswith("grads/") for tag, _, _ in fw.scalars[prev_scalar_count:]), "Grads scalar tag'ları yok!"


# ---- Determinism: eval modunda aynı giriş -> aynı çıkış ----
def test_eval_mode_determinism(neural_network):
    neural_network.eval()
    torch.manual_seed(42)

    batch_size, seq_len = 2, 10
    x = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    out1, _ = neural_network(x)
    out2, _ = neural_network(x)

    assert torch.allclose(out1, out2, atol=1e-6), "Eval modunda çıktılar deterministik olmalı!"


# ---- Dropout etkisi: train modunda aynı giriş -> farklı çıkış bekleriz ----
def test_train_mode_dropout_effect(neural_network):
    neural_network.train()
    torch.manual_seed(123)

    batch_size, seq_len = 2, 10
    x = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    out1, _ = neural_network(x)
    out2, _ = neural_network(x)

    # Tamamen eşit olmamalı (çok düşük bir olasılık hariç)
    assert not torch.allclose(out1, out2, atol=1e-6), "Train modunda dropout farkı görülmedi!"


# ---- Pickle Round-Trip ----
def test_pickle_roundtrip(neural_network):
    torch.manual_seed(0)

    # pickle
    blob = pickle.dumps(neural_network)
    restored = pickle.loads(blob)

    # ileri yayılım çalışmalı
    batch_size, seq_len = 2, 9
    x = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    out, _ = restored(x)
    assert out.shape == (batch_size, seq_len, CONFIG["vocab_size"]), "Pickle sonrası forward bozuldu!"
