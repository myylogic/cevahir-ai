"""
FAZE 5.1.1: ModelManager Checkpoint Loading - Endüstri Standartları Testi

Bu test, ModelManager'ın checkpoint yükleme mekanizmasının endüstri standartlarına
uygunluğunu test eder.

Endüstri Standartları:
1. Model state dict yüklenmeli (KRİTİK - exception fırlatmamalı)
2. Optimizer state dict yüklenmeli (opsiyonel - exception handle edilmeli)
3. Scheduler state dict yüklenmeli (opsiyonel - exception handle edilmeli)
4. Config/metadata yüklenmeli
5. Epoch/step bilgisi yüklenmeli
6. Yükleme sonrası doğrulama olmalı (model weights gerçekten yüklendi mi?)
7. Her bileşen için ayrı exception handling olmalı
8. Yükleme başarısız olsa bile bilgilendirici log mesajları olmalı

Test Senaryoları:
1. ✅ Normal checkpoint yükleme (tüm bileşenler mevcut)
2. ✅ Model state dict yükleme (optimizer/scheduler yok)
3. ✅ Optimizer state dict uyumsuzluğu (tam yükleme reddedilir, weights_only=True ile model yüklenir)
4. ✅ Farklı mimariden (oyuncak SimpleModel) checkpoint reddedilmeli
5. ✅ Gerçek checkpoint format analizi (tests/test_checkpoints/*.pth; yoksa skip)
6. ✅ Yükleme sonrası doğrulama (gerçek checkpoint ile model weights kontrolü; yoksa skip)

Güncel checkpoint sözleşmesi (model_management/checkpoint_contract.py, ModelManager.load):
- Checkpoint mimarisi canlı modelle eşleşmeli; eşleşmeyen checkpoint hiçbir canlı parametre
  değişmeden RuntimeError ile reddedilir (asıl neden __cause__ içindedir).
- Tam geri yükleme (varsayılan) transactional'dır: optimizer/scheduler state'i önce kopya
  üzerinde doğrulanır, uyumsuzsa yükleme tümden reddedilir. Yalnızca model ağırlıkları için
  weights_only=True kullanılır.
"""

import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn as nn

# Root dizini path'e ekle
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

# Gerçek (eğitilmiş) test checkpoint'leri repo'da tutulmaz; yol CWD'den bağımsızdır
TEST_CHECKPOINTS_DIR = REPO_ROOT / "tests" / "test_checkpoints"

# ModelManager bu config ile oyuncak boyutlu bir CevahirNeuralNetwork kurar
TOY_CONFIG = {
    "vocab_size": 1000,
    "embed_dim": 128,
    "learning_rate": 0.001,
    "device": "cpu",
}


class SimpleModel(nn.Module):
    """Test için basit bir model"""
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x.mean(dim=1))
        return x


def create_test_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    config: Optional[Dict[str, Any]] = None,
    format_type: str = "standard"
) -> Dict[str, Any]:
    """
    Test checkpoint'i oluştur

    format_type:
    - "standard": {"state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch", "config"}
    - "alternative": {"model_state_dict", "optimizer_state", "scheduler_state", "epoch", "config"}
    - "minimal": {"state_dict", "epoch"}
    """
    checkpoint = {}

    if format_type == "standard":
        checkpoint["state_dict"] = model.state_dict()
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        checkpoint["epoch"] = epoch
        if config is not None:
            checkpoint["config"] = config
    elif format_type == "alternative":
        checkpoint["model_state_dict"] = model.state_dict()
        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state"] = scheduler.state_dict()
        checkpoint["epoch"] = epoch
        if config is not None:
            checkpoint["config"] = config
    elif format_type == "minimal":
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch

    return checkpoint


def analyze_checkpoint_format(checkpoint_path: str) -> Dict[str, Any]:
    """Checkpoint format'ını analiz et"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    analysis = {
        "keys": list(checkpoint.keys()),
        "has_state_dict": "state_dict" in checkpoint,
        "has_model_state_dict": "model_state_dict" in checkpoint,
        "has_optimizer_state_dict": "optimizer_state_dict" in checkpoint,
        "has_optimizer_state": "optimizer_state" in checkpoint,
        "has_scheduler_state_dict": "scheduler_state_dict" in checkpoint,
        "has_scheduler_state": "scheduler_state" in checkpoint,
        "has_epoch": "epoch" in checkpoint,
        "has_config": "config" in checkpoint,
        "has_additional_info": "additional_info" in checkpoint,
    }

    return analysis


def check_model_weights_loaded(model: nn.Module, checkpoint_path: str) -> Dict[str, Any]:
    """Model weights'lerin gerçekten yüklenip yüklenmediğini kontrol et"""
    # Checkpoint'ten model weights'leri yükle
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        return {"error": "Model state dict bulunamadı"}

    # Model'in ilk parametresini al
    first_param = next(iter(model.parameters()))
    param_before = first_param.data.clone()

    # State dict'i yükle
    model.load_state_dict(state_dict, strict=True)

    # Model'in ilk parametresini tekrar al
    param_after = next(iter(model.parameters()))

    # Karşılaştır
    weights_changed = not torch.equal(param_before, param_after)
    param_mean = param_after.mean().item()
    param_std = param_after.std().item()

    return {
        "weights_changed": weights_changed,
        "param_mean": param_mean,
        "param_std": param_std,
        "is_random_weights": abs(param_mean) < 0.001 and param_std < 0.15,  # Random weights göstergesi
    }


def build_manager(config: Optional[Dict[str, Any]] = None):
    """Model, optimizer, criterion ve scheduler'ı kurulmuş ModelManager"""
    from model_management.model_manager import ModelManager

    model_manager = ModelManager(dict(config or TOY_CONFIG))
    model_manager.initialize()
    return model_manager


def snapshot_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """State'in bağımsız kopyası (load_state_dict tensörleri yerinde değiştirir)"""
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def differing_tensors(model: nn.Module, reference: Dict[str, torch.Tensor]) -> List[str]:
    """Canlı state'te referanstan farklı olan tensörlerin adları (key kümeleri aynı olmalı)"""
    live = model.state_dict()
    assert live.keys() == reference.keys(), (
        f"State key'leri farklı: eksik={sorted(reference.keys() - live.keys())[:5]}, "
        f"fazla={sorted(live.keys() - reference.keys())[:5]}"
    )
    return [name for name in reference if not torch.equal(live[name], reference[name])]


def assert_weights_loaded(
    model: nn.Module,
    expected: Dict[str, torch.Tensor],
    before: Dict[str, torch.Tensor],
) -> None:
    """Yükleme sonrası doğrulama: canlı ağırlıklar checkpoint'tekilerle birebir aynı olmalı"""
    # Kontrol boşa geçmesin: hedef model yüklemeden önce checkpoint'ten farklı olmalı
    assert any(not torch.equal(before[name], expected[name]) for name in expected), (
        "Hedef model yüklemeden önce de checkpoint ile aynıydı; yükleme doğrulanamaz"
    )
    mismatched = differing_tensors(model, expected)
    assert not mismatched, (
        f"Model weights yüklenmedi: {len(mismatched)}/{len(expected)} tensör checkpoint'ten farklı "
        f"(ilk 5: {mismatched[:5]})"
    )


def test_standard_checkpoint_loading(tmp_path):
    """Test 1: Normal checkpoint yükleme (tüm bileşenler mevcut)"""
    from model_management.checkpoint_contract import model_config

    # Eski test checkpoint'i oyuncak SimpleModel'den üretiyordu; ModelManager ise
    # CevahirNeuralNetwork kurar ve güncel sözleşme farklı mimariyi (doğru olarak) reddeder
    # (bkz. test_foreign_architecture_checkpoint_rejected). Kaynak bu yüzden aynı mimaride
    # ikinci bir ModelManager.
    source = build_manager()
    # Optimizer/scheduler durumunu taze nesnelerden ayırt edilebilir yap; aksi hâlde
    # hiç yüklenmeseler de eşit görünürlerdi.
    for group in source.optimizer.param_groups:
        group["lr"] = 0.000123
    source.scheduler.step(1.0)  # Varsayılan ReduceLROnPlateau: best=1.0

    checkpoint = create_test_checkpoint(
        model=source.model,
        optimizer=source.optimizer,
        scheduler=source.scheduler,
        epoch=5,
        # Sözleşme kaydedilen config'i mimarinin kaynağı sayar ve eksik alanları varsayılanla
        # doldurur: eski {"learning_rate", "vocab_size"} config'i "embed_dim: checkpoint=512,
        # model=128" ile reddedilir. ModelManager.save gibi tam kuruluş config'i kaydedilir.
        config=model_config(source.model),
        format_type="standard",
    )
    checkpoint_path = tmp_path / "standard_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)

    # ModelManager oluştur ve yükle
    model_manager = build_manager()
    before = snapshot_state(model_manager.model)
    model_manager.load(str(checkpoint_path), strict=True)

    # Doğrulama
    assert model_manager.config.get("current_epoch") == 5, "Epoch bilgisi yüklenemedi"
    assert_weights_loaded(model_manager.model, checkpoint["state_dict"], before)
    lrs = [group["lr"] for group in model_manager.optimizer.param_groups]
    assert lrs == [0.000123] * len(lrs), f"Optimizer state dict yüklenmedi: lr={lrs}"
    scheduler_best = model_manager.scheduler.state_dict()["best"]
    assert scheduler_best == 1.0, f"Scheduler state dict yüklenmedi: best={scheduler_best}"


def test_model_only_checkpoint_loading(tmp_path):
    """Test 2: Model state dict yükleme (optimizer/scheduler yok)"""
    # Kaynak, hedefle aynı mimaride (bkz. Test 1 notu)
    source = build_manager()

    # Minimal checkpoint oluştur (sadece model weights)
    checkpoint = create_test_checkpoint(
        model=source.model,
        epoch=3,
        format_type="minimal"
    )
    checkpoint_path = tmp_path / "minimal_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)

    # ModelManager oluştur ve yükle
    model_manager = build_manager()
    before = snapshot_state(model_manager.model)
    model_manager.load(str(checkpoint_path), strict=True)

    # Doğrulama
    assert model_manager.config.get("current_epoch") == 3, "Epoch bilgisi yüklenemedi"
    assert_weights_loaded(model_manager.model, checkpoint["state_dict"], before)


def test_optimizer_mismatch_handling(tmp_path):
    """Test 3: Optimizer state dict uyumsuzluğu"""
    # Kaynak model hedefle aynı mimaride (bkz. Test 1 notu)
    source = build_manager()
    # Farklı parameter groups ile optimizer oluştur (ModelManager'ın optimizer'ı 3 grup kurar)
    params = list(source.model.parameters())
    optimizer2 = torch.optim.AdamW([
        {"params": params[:2], "lr": 0.001},
        {"params": params[2:], "lr": 0.002}
    ])

    # Checkpoint oluştur (optimizer2 ile)
    checkpoint = create_test_checkpoint(
        model=source.model,
        optimizer=optimizer2,
        epoch=5,
        format_type="standard"
    )
    checkpoint_path = tmp_path / "optimizer_mismatch.pth"
    torch.save(checkpoint, checkpoint_path)

    model_manager = build_manager()
    before = snapshot_state(model_manager.model)

    # Eski beklenti: tam yükleme uyumsuz optimizer state'ini atlayıp modeli yine de yükler.
    # Güncel ModelManager.load optimizer/scheduler state'ini önce kopya üzerinde doğrular ve
    # uyumsuzlukta yüklemeyi tümden reddeder: eğitim taze optimizer ile sessizce "devam
    # ediyormuş" gibi yarım geri yüklenmez ve hiçbir canlı parametre değişmez.
    with pytest.raises(RuntimeError, match="yükleme") as excinfo:
        model_manager.load(str(checkpoint_path), strict=True)
    assert "parameter groups" in str(excinfo.value.__cause__), (
        f"Beklenmeyen ret nedeni: {excinfo.value.__cause__!r}"
    )
    changed = differing_tensors(model_manager.model, before)
    assert not changed, f"Reddedilen yükleme canlı parametreleri değiştirdi: {changed[:5]}"
    assert model_manager.config.get("current_epoch") is None, "Reddedilen yükleme epoch'u değiştirdi"

    # Optimizer yüklenemese bile model yüklenebilmeli: desteklenen yol weights_only=True
    model_manager.load(str(checkpoint_path), strict=True, weights_only=True)
    assert model_manager.config.get("current_epoch") == 5, "Epoch bilgisi yüklenemedi"
    assert_weights_loaded(model_manager.model, checkpoint["state_dict"], before)


@pytest.mark.parametrize(
    "format_type, expected_cause",
    [
        # Kısmi config, eksik alanları varsayılanla doldurularak karşılaştırılır (embed_dim 512 != 128)
        ("standard", "architecture mismatch"),
        # Config yok; state key'leri CevahirNeuralNetwork'ünkilerle eşleşmez
        ("minimal", "state keys mismatch"),
    ],
)
def test_foreign_architecture_checkpoint_rejected(tmp_path, format_type, expected_cause):
    """Test 4: Farklı mimariden checkpoint (eski Test 1/2'nin SimpleModel'i) reddedilmeli"""
    model = SimpleModel(vocab_size=1000, embed_dim=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    checkpoint = create_test_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        config={"learning_rate": 0.001, "vocab_size": 1000},
        format_type=format_type,
    )
    checkpoint_path = tmp_path / f"simple_model_{format_type}.pth"
    torch.save(checkpoint, checkpoint_path)

    # Dosya sağlam: kendi mimarisine strict yüklenir, yani ret mimari farkından kaynaklanmalı
    own_load = check_model_weights_loaded(SimpleModel(vocab_size=1000, embed_dim=128), str(checkpoint_path))
    assert own_load.get("weights_changed"), f"SimpleModel checkpoint'i kendi mimarisine yüklenemedi: {own_load}"

    model_manager = build_manager()
    before = snapshot_state(model_manager.model)
    with pytest.raises(RuntimeError, match="yükleme") as excinfo:
        model_manager.load(str(checkpoint_path), strict=True)
    assert expected_cause in str(excinfo.value.__cause__), (
        f"Beklenmeyen ret nedeni: {excinfo.value.__cause__!r}"
    )
    changed = differing_tensors(model_manager.model, before)
    assert not changed, f"Reddedilen checkpoint canlı parametreleri değiştirdi: {changed[:5]}"
    assert model_manager.config.get("current_epoch") is None, "Reddedilen checkpoint epoch'u değiştirdi"


@pytest.fixture(scope="module")
def real_checkpoint_path() -> Path:
    """tests/test_checkpoints altındaki ilk gerçek checkpoint; yoksa testler atlanır"""
    checkpoint_files = sorted(TEST_CHECKPOINTS_DIR.glob("*.pth"))
    if not checkpoint_files:
        pytest.skip(
            "Gerçek test checkpoint'i yok: tests/test_checkpoints/*.pth bulunamadı "
            "(eğitilmiş checkpoint'ler repo'da tutulmaz)"
        )
    logger.info(f"Checkpoint: {checkpoint_files[0]}")
    return checkpoint_files[0]


def test_checkpoint_format_analysis_real(real_checkpoint_path):
    """Test 5: Gerçek checkpoint format analizi"""
    analysis = analyze_checkpoint_format(str(real_checkpoint_path))

    logger.info("Checkpoint Format Analizi:")
    for key, value in analysis.items():
        logger.info(f"  {key}: {value}")

    # Analiz en azından model state dict'i (standart 1, KRİTİK) bulmalı
    assert analysis["has_state_dict"] or analysis["has_model_state_dict"], (
        f"Checkpoint'te model state dict yok: {real_checkpoint_path.name}, keys={analysis['keys']}"
    )


def test_model_weights_verification(real_checkpoint_path):
    """Test 6: Model weights doğrulama (gerçek checkpoint ile)"""
    from model_management.checkpoint_contract import unpack_checkpoint
    from model_management.model_manager import ModelManager
    from tokenizer_management.core.tokenizer_core import TokenizerCore
    # Config yükle (train.py'deki gibi)
    from training_system.train import TRAIN_CONFIG, load_tokenizer_config

    config = copy.deepcopy(TRAIN_CONFIG)
    config.update(load_tokenizer_config())
    # TensorBoard bu testin konusu değil; tensorboard kuruluysa model kurulumu repo'ya runs/ yazar
    config["use_tensorboard"] = False

    # TRAIN_CONFIG vocab_size içermez; train.py gibi TokenizerCore'dan alınır. Sözleşme, tokenizer
    # kimliği kayıtlı checkpoint'lerde tokenizer'ı zorunlu kılar ("requires a tokenizer").
    # Vocab için veri yükleyici gerekmez; göreli data_dir ('education') CWD'ye bağlıdır.
    tokenizer_core = TokenizerCore({k: v for k, v in config.items() if k != "data_dir"})
    config["vocab_size"] = tokenizer_core.get_vocab_size()

    # ModelManager oluştur
    model_manager = ModelManager(config, tokenizer=tokenizer_core)
    model_manager.initialize()

    # Model weights'i kontrol et (yüklemeden önce). Eski test parametre nesnesinin kendisini
    # saklıyordu; load_state_dict yerinde kopyaladığı için "önce" ve "sonra" hep aynı tensördü
    # ve weights_changed asla True olamazdı.
    first_param_before = next(iter(model_manager.model.parameters())).detach().clone()
    logger.info(f"Yükleme öncesi: mean={first_param_before.mean().item():.6f}, std={first_param_before.std().item():.6f}")

    # Checkpoint yükle
    model_manager.load(str(real_checkpoint_path), strict=False)

    # Model weights'i kontrol et (yüklemeden sonra)
    first_param_after = next(iter(model_manager.model.parameters()))
    param_mean_after = first_param_after.mean().item()
    param_std_after = first_param_after.std().item()
    logger.info(f"Yükleme sonrası: mean={param_mean_after:.6f}, std={param_std_after:.6f}")

    # Doğrulama
    weights_changed = not torch.equal(first_param_before, first_param_after)
    assert weights_changed, (
        "Model weights değişmedi (yükleme başarısız olabilir): "
        f"mean={param_mean_after:.6f}, std={param_std_after:.6f}"
    )

    # Eski "random weights" sezgisi (|mean| < 0.001 ve std < 0.15) yüklemenin başarısız olup
    # olmadığını tahmin ediyordu; taze init'in std'si ~0.007'dir ve eğitilmiş embedding'ler de bu
    # eşiklerin altında kalabilir. Bunun yerine yüklenen tensörler checkpoint'tekilerle birebir
    # karşılaştırılır (kesin kontrol).
    state, _, _, _ = unpack_checkpoint(torch.load(real_checkpoint_path, map_location="cpu"))
    live = model_manager.model.state_dict()
    common = sorted(
        name for name in live.keys() & state.keys()
        if isinstance(live[name], torch.Tensor) and isinstance(state[name], torch.Tensor)
    )
    assert common, "Checkpoint ile model arasında ortak state key'i yok"
    mismatched = [
        name for name in common
        if not torch.equal(live[name].cpu(), state[name].to(dtype=live[name].dtype))
    ]
    assert not mismatched, (
        f"Model weights checkpoint'ten yüklenmedi: {len(mismatched)}/{len(common)} tensör farklı "
        f"(ilk 5: {mismatched[:5]})"
    )



@pytest.mark.xfail(strict=True, reason=(
    "Ürün hatası: ModelManager.load scheduler durumunu derin kopyaya yükleyip istisna bekler; "
    "torch scheduler.load_state_dict yalnızca __dict__.update yaptığı için yanlış türde bir "
    "scheduler durumu (StepLR → ReduceLROnPlateau) sessizce kabul edilir. Düzeltilince XPASS olur."))
def test_wrong_scheduler_type_is_rejected(tmp_path):
    """Yanlış türde scheduler durumu içeren checkpoint reddedilmeli (şu an kabul ediliyor)."""
    from model_management.checkpoint_contract import model_config

    source = build_manager()
    foreign = torch.optim.lr_scheduler.StepLR(source.optimizer, step_size=3, gamma=0.5)
    checkpoint = create_test_checkpoint(
        model=source.model,
        optimizer=source.optimizer,
        scheduler=foreign,
        epoch=1,
        config=model_config(source.model),
        format_type="standard",
    )
    checkpoint_path = tmp_path / "foreign_scheduler.pth"
    torch.save(checkpoint, checkpoint_path)
    manager = build_manager()
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError)):
        manager.load(str(checkpoint_path), strict=True)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
