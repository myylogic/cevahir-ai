"""
FAZE 5.1.2: ModelManager Checkpoint Loading - Gerçek Model ile Endüstri Standartları Testi

Bu test, GERÇEK EĞİTİLMİŞ MODEL ile ModelManager'ın checkpoint yükleme mekanizmasını
endüstri standartlarına göre test eder.

Endüstri Standartları:
1. ✅ Model state dict yüklenmeli (KRİTİK - exception fırlatmamalı, warning vermeli)
2. ⚠️ Optimizer state dict yüklenmeli (opsiyonel - exception handle edilmeli, warning vermeli)
3. ⚠️ Scheduler state dict yüklenmeli (opsiyonel - exception handle edilmeli, warning vermeli)
4. ✅ Config/metadata yüklenmeli
5. ✅ Epoch/step bilgisi yüklenmeli
6. ✅ Yükleme sonrası doğrulama olmalı (model weights gerçekten yüklendi mi?)
7. ✅ Her bileşen için ayrı exception handling olmalı
8. ✅ Yükleme başarısız olsa bile bilgilendirici log mesajları olmalı

Test Senaryoları:
1. ✅ Gerçek checkpoint format kontrolü (hangi key'ler var?)
2. ✅ Model state dict yükleme (strict=False ile)
3. ✅ Optimizer state dict uyumsuzluğu (exception handle edilmeli)
4. ✅ Yükleme sonrası doğrulama (model weights kontrolü)
5. ✅ Endüstri standartlarına uygunluk kontrolü

Not: Eğitilmiş checkpoint'ler (tests/test_checkpoints/*.pth, saved_models/**/*.pth) repo'da
tutulmaz; bulunamazsa testler atlanır (skip).
"""

import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch

# Root dizini path'e ekle
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

# Eğitilmiş modellerin arandığı dizinler (CWD'den bağımsız, repo köküne göre)
TEST_CHECKPOINTS_DIR = REPO_ROOT / "tests" / "test_checkpoints"
SAVED_MODELS_DIR = REPO_ROOT / "saved_models"


def load_config() -> Dict[str, Any]:
    """Config dosyasını yükle - train.py'deki yöntemi kullan"""
    from training_system.train import TRAIN_CONFIG

    # Modül düzeyindeki dict değişmesin
    config = copy.deepcopy(TRAIN_CONFIG)
    tokenizer_config_from_train = load_tokenizer_config_from_train_py()
    config.update(tokenizer_config_from_train)
    # TensorBoard bu testlerin konusu değil; TRAIN_CONFIG'te açık ve tensorboard kuruluysa
    # model kurulumu repo içine runs/ dizini yazar.
    config["use_tensorboard"] = False
    return config


def load_tokenizer_config_from_train_py() -> Dict[str, Any]:
    """training_system/train.py'deki load_tokenizer_config fonksiyonunu simüle et"""
    try:
        from tokenizer_management.config import (
            get_bpe_detailed_config,
            TOKENIZER_CONFIG,
            BPE_CONFIG
        )
        bpe_config = get_bpe_detailed_config()
        tokenizer_config = TOKENIZER_CONFIG.copy()
        bpe_legacy = BPE_CONFIG.copy()

        config = {
            "max_seq_length": tokenizer_config.get("max_seq_length", 512),
            "vocab_path": (
                bpe_config.get("vocab_path") or
                bpe_config.get("vocab_file") or
                bpe_legacy.get("vocab_file") or
                str(REPO_ROOT / "data" / "vocab_lib" / "vocab.json")
            ),
            "merges_path": (
                bpe_config.get("merges_path") or
                bpe_config.get("merges_file") or
                bpe_legacy.get("merges_file") or
                str(REPO_ROOT / "data" / "merges_lib" / "merges.txt")
            ),
            "bpe_rebuild": False,
            "bpe_max_merges": bpe_config.get("target_merges", bpe_config.get("merge_operations", bpe_legacy.get("merge_operations", 50000))),
            "bpe_min_frequency": bpe_config.get("min_frequency", bpe_legacy.get("min_frequency", 2)),
            "bpe_max_iter": bpe_config.get("max_iter", bpe_legacy.get("max_iter", 60000)),
            "bpe_include_syllables": bpe_config.get("include_syllables", False),
            "bpe_include_whole_words": bpe_config.get("include_whole_words", True),
            "bpe_include_sep": bpe_config.get("include_sep", False),
            "train_include_whole_words": bpe_config.get("train_include_whole_words", True),
            "train_include_syllables": bpe_config.get("train_include_syllables", False),
            "train_include_sep": bpe_config.get("train_include_sep", False),
            "use_gpu": torch.cuda.is_available(),
            "tokenizer_batch_size": tokenizer_config.get("batch_size", 32),
            "vocab_size": tokenizer_config.get("vocab_size", None),
        }
        return config
    except ImportError:
        logger.error("tokenizer_management.config import edilemedi. Lütfen path'i kontrol edin.")
        return {}


def find_saved_models() -> List[Dict[str, str]]:
    """Eğitilmiş modelleri bul"""
    models = []

    # Test checkpoint'leri
    if TEST_CHECKPOINTS_DIR.exists():
        for checkpoint_file in sorted(TEST_CHECKPOINTS_DIR.glob("*.pth")):
            models.append({
                "name": checkpoint_file.stem,
                "path": str(checkpoint_file),
                "type": "test_checkpoints"
            })

    # Saved models
    if SAVED_MODELS_DIR.exists():
        for checkpoint_file in sorted(SAVED_MODELS_DIR.rglob("*.pth")):
            models.append({
                "name": checkpoint_file.stem,
                "path": str(checkpoint_file),
                "type": "saved_models"
            })

    return models


def select_checkpoint(models: List[Dict[str, str]]) -> Dict[str, str]:
    """En iyi checkpoint'i seç (saved_models/checkpoints öncelikli)"""
    def in_saved_checkpoints(model: Dict[str, str]) -> bool:
        # Yol saved_models'e göre değerlendirilir; repo'nun üst dizin adları eşleşmeyi bozmasın
        return model["type"] == "saved_models" and "checkpoints" in os.path.relpath(model["path"], SAVED_MODELS_DIR)

    # Önce saved_models/checkpoints içinde best.pth ara
    for model in models:
        if in_saved_checkpoints(model) and model["name"] == "best":
            return model

    # Bulunamazsa saved_models içindeki en yeni checkpoint (dosya tarihine göre)
    saved_models_checkpoints = [m for m in models if in_saved_checkpoints(m)]
    if saved_models_checkpoints:
        return max(saved_models_checkpoints, key=lambda m: os.path.getmtime(m["path"]))

    # Hala bulunamazsa ilk checkpoint'i kullan
    return models[0]


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
        "has_training_history": "training_history" in checkpoint,
        "has_metric": "metric" in checkpoint,
    }

    return analysis


def build_model_manager(config: Dict[str, Any], tokenizer_core):
    """ModelManager'ı TrainingService gibi kur: vocab_size TokenizerCore'dan gelir"""
    from model_management.model_manager import ModelManager

    model_config = dict(config)
    model_config["vocab_size"] = tokenizer_core.get_vocab_size()
    # Checkpoint sözleşmesi tokenizer kimliğini doğrular: tokenizer verilmezse kimlikli bir
    # checkpoint "Checkpoint requires a tokenizer to verify token identity" ile reddedilir.
    model_manager = ModelManager(model_config, tokenizer=tokenizer_core)
    model_manager.initialize()
    return model_manager


def compare_with_checkpoint(model, checkpoint_path: str) -> Tuple[List[str], List[str]]:
    """Canlı ağırlıkları checkpoint'teki tensörlerle birebir karşılaştır: (ortak, farklı) key'ler"""
    from model_management.checkpoint_contract import unpack_checkpoint

    state, _, _, _ = unpack_checkpoint(torch.load(checkpoint_path, map_location="cpu"))
    live = model.state_dict()
    common = sorted(
        name for name in live.keys() & state.keys()
        if isinstance(live[name], torch.Tensor) and isinstance(state[name], torch.Tensor)
    )
    mismatched = [
        name for name in common
        if not torch.equal(live[name].cpu(), state[name].to(dtype=live[name].dtype))
    ]
    return common, mismatched


@pytest.fixture(scope="module")
def checkpoint_path() -> str:
    """Kullanılacak eğitilmiş checkpoint; yoksa modüldeki testler atlanır"""
    models = find_saved_models()
    if not models:
        pytest.skip(
            "Eğitilmiş checkpoint yok: tests/test_checkpoints/*.pth ve saved_models/**/*.pth "
            "bulunamadı (checkpoint'ler repo'da tutulmaz; önce training_system/train.py ile model eğitin)"
        )

    logger.info(f"{len(models)} checkpoint bulundu")
    for i, model in enumerate(models, 1):
        logger.info(f"   {i}. {model['name']} ({model['type']})")

    selected_model = select_checkpoint(models)
    logger.info(f"Kullanılacak checkpoint: {selected_model['name']} ({selected_model['path']})")
    logger.info(f"Checkpoint tipi: {selected_model['type']}")
    return selected_model["path"]


@pytest.fixture(scope="module")
def config(checkpoint_path) -> Dict[str, Any]:
    """train.py config'i (checkpoint yoksa hiç yüklenmez)"""
    return load_config()


@pytest.fixture(scope="module")
def tokenizer_core(config):
    """TokenizerCore (vocab_size ve tokenizer kimliği için); modül başına bir kez kurulur"""
    from tokenizer_management.core.tokenizer_core import TokenizerCore

    tokenizer_config = dict(config)
    if isinstance(config.get("tokenizer"), dict):
        tokenizer_config.update(config["tokenizer"])
    # Yalnızca vocab/merges gerekir; TRAIN_CONFIG'in göreli data_dir'i ('education') CWD'ye bağlıdır
    tokenizer_config.pop("data_dir", None)

    if "vocab_path" not in tokenizer_config:
        tokenizer_config["vocab_path"] = str(REPO_ROOT / "data" / "vocab_lib" / "vocab.json")
    if "merges_path" not in tokenizer_config:
        tokenizer_config["merges_path"] = str(REPO_ROOT / "data" / "merges_lib" / "merges.txt")

    return TokenizerCore(tokenizer_config)


def test_checkpoint_format_analysis(checkpoint_path):
    """TEST 1: Gerçek checkpoint format kontrolü (hangi key'ler var?)"""
    format_analysis = analyze_checkpoint_format(checkpoint_path)
    logger.info("Checkpoint Format:")
    for key, value in format_analysis.items():
        logger.info(f"  {key}: {value}")

    # Eski özetteki ölçüt: format_ok = has_model_state_dict
    assert format_analysis["has_model_state_dict"], (
        f"Checkpoint'te 'model_state_dict' yok: {checkpoint_path}, keys={format_analysis['keys']}"
    )


def test_model_weights_verification_real(checkpoint_path, config, tokenizer_core):
    """TEST 2: Gerçek model ile model weights doğrulama"""
    model_manager = build_model_manager(config, tokenizer_core)

    # Model weights'i kontrol et (yüklemeden önce). Eski kod parametre nesnesinin kendisini
    # saklıyordu; load_state_dict yerinde kopyaladığı için "önce" ve "sonra" hep aynı tensördü
    # ve weights_changed asla True olamazdı.
    first_param_before = next(iter(model_manager.model.parameters())).detach().clone()
    logger.info(f"Yükleme öncesi: mean={first_param_before.mean().item():.6f}, std={first_param_before.std().item():.6f}")

    # Checkpoint yükle. Eski kod yükleme hatasını yakalayıp load_success=False döndürüyordu;
    # artık hata (asıl neden __cause__ zincirinde) testi doğrudan düşürür.
    model_manager.load(checkpoint_path, strict=False)

    # Model weights'i kontrol et (yüklemeden sonra)
    first_param_after = next(iter(model_manager.model.parameters()))
    param_mean_after = first_param_after.mean().item()
    param_std_after = first_param_after.std().item()
    logger.info(f"Yükleme sonrası: mean={param_mean_after:.6f}, std={param_std_after:.6f}")
    logger.info(f"epoch_loaded: {model_manager.config.get('current_epoch')}")

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
    common, mismatched = compare_with_checkpoint(model_manager.model, checkpoint_path)
    assert common, "Checkpoint ile model arasında ortak state key'i yok"
    assert not mismatched, (
        f"Model weights checkpoint'ten yüklenmedi: {len(mismatched)}/{len(common)} tensör farklı "
        f"(ilk 5: {mismatched[:5]})"
    )


def test_checkpoint_loading_industry_standards(checkpoint_path, config, tokenizer_core, caplog):
    """TEST 3: Endüstri standartlarına göre checkpoint yükleme testi"""
    standards = {
        "model_state_dict_loaded": False,
        "optimizer_state_dict_loaded": False,
        "scheduler_state_dict_loaded": False,
        "epoch_info_loaded": False,
        "config_info_loaded": False,
        "model_weights_verified": False,
        "exception_handling_proper": False,
        "warning_messages_proper": False,
    }

    model_manager = build_model_manager(config, tokenizer_core)

    # Checkpoint format'ını kontrol et
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # 1. Model state dict yüklenebilmeli
    standards["model_state_dict_loaded"] = "model_state_dict" in checkpoint or "state_dict" in checkpoint
    # 2. Optimizer state dict kontrolü
    standards["optimizer_state_dict_loaded"] = "optimizer_state_dict" in checkpoint
    # 3. Scheduler state dict kontrolü
    standards["scheduler_state_dict_loaded"] = "scheduler_state_dict" in checkpoint
    # 4. Epoch bilgisi kontrolü
    standards["epoch_info_loaded"] = "epoch" in checkpoint
    # 5. Config bilgisi kontrolü
    standards["config_info_loaded"] = "config" in checkpoint
    del checkpoint

    # Checkpoint yükle (exception handling kontrolü). Hata standart 7 olarak ölçülür ve
    # aşağıdaki assert mesajında nedeniyle birlikte raporlanır (yutulmaz).
    load_error = None
    with caplog.at_level(logging.INFO, logger="ModelManager"):
        try:
            model_manager.load(checkpoint_path, strict=False)
        except Exception as exc:
            load_error = exc

    if load_error is None:
        standards["exception_handling_proper"] = True  # Exception fırlatılmadı

        # Model weights doğrulama (eski "random weights" sezgisi yerine checkpoint ile birebir
        # karşılaştırma; bkz. TEST 2)
        common, mismatched = compare_with_checkpoint(model_manager.model, checkpoint_path)
        standards["model_weights_verified"] = bool(common) and not mismatched

        # Epoch bilgisi yüklendi mi?
        if model_manager.config.get("current_epoch") is not None:
            standards["epoch_info_loaded"] = True

        # 8. Bilgilendirici log mesajı. Eski kod bu standardı hiç ölçmüyordu (hep False kaldığı
        # için bu test asla geçemezdi); ModelManager başarılı yüklemede yolu INFO ile loglar.
        standards["warning_messages_proper"] = any(
            record.name == "ModelManager" and checkpoint_path in record.getMessage()
            for record in caplog.records
        )

    logger.info("Endüstri Standartları Kontrolü:")
    for key, value in standards.items():
        logger.info(f"  {'✅' if value else '❌'} {key}: {value}")

    failed = [name for name, passed in standards.items() if not passed]
    assert not failed, (
        f"Karşılanmayan endüstri standartları ({len(standards) - len(failed)}/{len(standards)} geçti): "
        f"{failed}; yükleme hatası: {load_error!r}, neden: {getattr(load_error, '__cause__', None)!r}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
