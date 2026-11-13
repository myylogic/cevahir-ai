"""
FAZE 3: ModelSaver Testleri - Endüstri Standartları

Bu test, ModelSaver'ın checkpoint kaydetme işleminin endüstri standartlarına
uygunluğunu test eder.

Endüstri Standartları:
1. ✅ Checkpoint format standart (model_state_dict, optimizer_state_dict, epoch, config, vb.)
2. ✅ Model state dict DOĞRU kaydedilmeli (tüm parametreler, config'teki tüm katmanlar)
3. ✅ Atomik kayıt (atomic save)
4. ✅ Doğrulama (validation) - Kaydedilen checkpoint doğrulanabilmeli
5. ✅ Logging - Detaylı log mesajları
6. ✅ Error handling - Hatalı durumlarda uygun exception/warning

Test Senaryoları:
1. ✅ Checkpoint format kontrolü (hangi key'ler var?)
2. ✅ Model state dict kaydetme (key sayısı, key names kontrolü)
3. ✅ Checkpoint doğrulama (kaydedilen checkpoint'i yükleme ve kontrol)
4. ✅ Keys eşleşmesi kontrolü (kaydedilen vs model state dict)

Model, training_system/train.py'deki TRAIN_CONFIG ile kurulur (vocab_size TokenizerCore'dan);
checkpoint pytest'in geçici dizinine yazılır.
"""

import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

# Root dizini path'e ekle
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Config dosyasını yükle"""
    from training_system.train import TRAIN_CONFIG

    # Modül düzeyindeki dict değişmesin
    config = copy.deepcopy(TRAIN_CONFIG)

    # Tokenizer config
    try:
        from tokenizer_management.config import (
            get_bpe_detailed_config,
            TOKENIZER_CONFIG,
            BPE_CONFIG
        )
        bpe_config = get_bpe_detailed_config()
        tokenizer_config = TOKENIZER_CONFIG.copy()
        bpe_legacy = BPE_CONFIG.copy()

        tokenizer_config_for_merge = {
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
        config.update(tokenizer_config_for_merge)
    except ImportError:
        logger.warning("tokenizer_management.config import edilemedi, varsayılanlar kullanılıyor")

    return config


def build_model_manager(config: Dict[str, Any]):
    """ModelManager oluştur ve initialize et"""
    from model_management.model_manager import ModelManager

    model_manager = ModelManager(dict(config))
    model_manager.initialize()
    return model_manager


@pytest.fixture(scope="module")
def config() -> Dict[str, Any]:
    """TRAIN_CONFIG + tokenizer ayarları + gerçek vocab_size (modül başına bir kez kurulur)"""
    from tokenizer_management.core.tokenizer_core import TokenizerCore

    config = load_config()
    # TensorBoard bu testlerin konusu değil; TRAIN_CONFIG'te açık ve tensorboard kuruluysa
    # model kurulumu repo içine runs/ dizini yazar.
    config["use_tensorboard"] = False

    # TokenizerCore oluştur (vocab_size almak için)
    tokenizer_config = dict(config)
    if isinstance(config.get("tokenizer"), dict):
        tokenizer_config.update(config["tokenizer"])
    # Yalnızca vocab/merges gerekir; TRAIN_CONFIG'in göreli data_dir'i ('education') CWD'ye bağlıdır
    tokenizer_config.pop("data_dir", None)

    if "vocab_path" not in tokenizer_config:
        tokenizer_config["vocab_path"] = str(REPO_ROOT / "data" / "vocab_lib" / "vocab.json")
    if "merges_path" not in tokenizer_config:
        tokenizer_config["merges_path"] = str(REPO_ROOT / "data" / "merges_lib" / "merges.txt")

    tokenizer_core = TokenizerCore(tokenizer_config)
    config["vocab_size"] = tokenizer_core.get_vocab_size()
    return config


@pytest.fixture(scope="module")
def saved_checkpoint(config, tmp_path_factory) -> Dict[str, Any]:
    """Üç testin ortak adımı (eskiden her testte tekrarlanıyordu): ModelManager'ı kur ve
    epoch=1 checkpoint'ini pytest'in geçici dizinine kaydet."""
    model_manager = build_model_manager(config)
    checkpoint_path = tmp_path_factory.mktemp("model_saver") / "test_checkpoint.pth"

    # Checkpoint kaydet
    model_manager.save(
        save_path=str(checkpoint_path),
        epoch=1,
        additional_info={
            "training_history": {"train_loss": [2.5], "val_loss": [2.4]},
            "metric": 2.4,
        }
    )
    return {"model_manager": model_manager, "path": checkpoint_path}


def test_checkpoint_format(saved_checkpoint):
    """Test 1: Checkpoint Format Kontrolü"""
    # Checkpoint'i yükle ve format kontrolü
    checkpoint = torch.load(saved_checkpoint["path"], map_location="cpu")

    # Format kontrolü
    expected_keys = [
        "model_state_dict",
        "state_dict",  # Geriye dönük uyumluluk
        "optimizer_state_dict",
        "optimizer_state",  # Geriye dönük uyumluluk
        "scheduler_state_dict",
        "scheduler_state",  # Geriye dönük uyumluluk
        "epoch",
        "config",
        "additional_info",
        "metadata",  # Geriye dönük uyumluluk
    ]

    logger.info("Checkpoint Keys:")
    checkpoint_keys = set(checkpoint.keys())
    for key in expected_keys:
        status = "✅" if key in checkpoint_keys else "❌"
        logger.info(f"  {status} {key}: {key in checkpoint_keys}")

    # Ana key'lerin varlığını kontrol et
    assert "model_state_dict" in checkpoint or "state_dict" in checkpoint, (
        f"Checkpoint'te model state dict yok: keys={sorted(checkpoint_keys)}"
    )
    assert "epoch" in checkpoint, f"Checkpoint'te epoch yok: keys={sorted(checkpoint_keys)}"
    assert checkpoint["epoch"] == 1, f"Kaydedilen epoch yanlış: {checkpoint['epoch']!r}"


def test_model_state_dict_saving(saved_checkpoint, config):
    """Test 2: Model State Dict Kaydetme Kontrolü"""
    # Model state dict'i al (kaydedilen model)
    model_state_dict = saved_checkpoint["model_manager"].model.state_dict()
    model_keys = set(model_state_dict.keys())
    logger.info(f"Model State Dict: {len(model_keys)} key, ilk 10: {sorted(model_keys)[:10]}")

    # Checkpoint'i yükle ve model state dict'i al (checkpoint'ten)
    checkpoint = torch.load(saved_checkpoint["path"], map_location="cpu")
    checkpoint_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
    checkpoint_keys = set(checkpoint_state_dict.keys())
    logger.info(f"Checkpoint Model State Dict: {len(checkpoint_keys)} key")

    # Keys karşılaştırması
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    assert not missing_keys and not unexpected_keys, (
        f"Kaydedilen key'ler modelinkilerle eşleşmiyor: model={len(model_keys)}, "
        f"checkpoint={len(checkpoint_keys)}, missing (ilk 20)={sorted(missing_keys)[:20]}, "
        f"unexpected (ilk 20)={sorted(unexpected_keys)[:20]}"
    )

    # Eski "200+ key" eşiği, eski mimarinin (dil_katmani, >200 key) tam CevahirNeuralNetwork'ünü
    # yanlışlıkla kaydedilen 3 key'lik oyuncak SimpleModel'den ayırmak içindi (bkz.
    # tests/analyze_checkpoint_keys_count.py). Güncel mimari katman başına daha az tensör tutar
    # (TRAIN_CONFIG'in 8 katmanı → 85 key); sabit sayı yerine config'teki her transformer
    # katmanının kaydedildiği doğrulanır.
    num_layers = int(config["num_layers"])
    saved_layers = {int(key.split(".")[1]) for key in checkpoint_keys if key.startswith("layers.")}
    assert saved_layers == set(range(num_layers)), (
        f"Checkpoint config'teki {num_layers} transformer katmanının hepsini içermiyor: "
        f"kaydedilen katmanlar={sorted(saved_layers)} ({len(checkpoint_keys)} key)"
    )

    # Tüm parametreler DOĞRU kaydedilmeli: değerler kaydedilen modelle birebir aynı
    mismatched = sorted(key for key in model_keys if not torch.equal(model_state_dict[key], checkpoint_state_dict[key]))
    assert not mismatched, (
        f"Kaydedilen tensörler modelinkilerden farklı: {len(mismatched)}/{len(model_keys)} "
        f"(ilk 10: {mismatched[:10]})"
    )


def test_checkpoint_verification(saved_checkpoint, config):
    """Test 3: Checkpoint Doğrulama (Kaydedilen checkpoint'i yükleme ve kontrol)"""
    model_manager_1 = saved_checkpoint["model_manager"]

    # İlk parametreyi al (kaydedilen model)
    first_param_before_copy = next(iter(model_manager_1.model.parameters())).clone().detach()
    param_mean_before = first_param_before_copy.mean().item()
    param_std_before = first_param_before_copy.std().item()

    # Yeni bir ModelManager oluştur (checkpoint'i yüklemek için). TRAIN_CONFIG seed=42 sabitlediği
    # için aynı config ile kurulan ikinci model birebir aynı ağırlıklarla başlar; eski test bu
    # yüzden yükleme hiç çalışmasa da geçerdi. Hedef model farklı bir seed ile kurulur.
    model_manager_2 = build_model_manager({**config, "seed": int(config.get("seed", 0)) + 1})
    fresh_param = next(iter(model_manager_2.model.parameters())).clone().detach()
    assert not torch.equal(fresh_param, first_param_before_copy), (
        "Hedef model yüklemeden önce de kaydedilen modelle aynı; doğrulama anlamsız olur"
    )

    # Checkpoint'i yükle
    model_manager_2.load(str(saved_checkpoint["path"]), strict=False)

    # Yükleme sonrası parametreleri al
    first_param_after_copy = next(iter(model_manager_2.model.parameters())).clone().detach()
    param_mean_after = first_param_after_copy.mean().item()
    param_std_after = first_param_after_copy.std().item()

    # Parametreler eşleşiyor mu?
    params_match = torch.allclose(first_param_before_copy, first_param_after_copy, atol=1e-6)
    mean_diff = abs(param_mean_before - param_mean_after)
    std_diff = abs(param_std_before - param_std_after)
    assert params_match, (
        f"Yüklenen ilk parametre kaydedilenle eşleşmiyor: mean fark={mean_diff:.10f}, std fark={std_diff:.10f}"
    )

    # Yalnızca ilk parametre değil, tüm state birebir geri yüklenmeli
    saved_state = model_manager_1.model.state_dict()
    loaded_state = model_manager_2.model.state_dict()
    assert saved_state.keys() == loaded_state.keys(), "Yüklenen modelin state key'leri farklı"
    mismatched = sorted(key for key in saved_state if not torch.equal(saved_state[key], loaded_state[key]))
    assert not mismatched, (
        f"Yüklenen tensörler kaydedilenlerden farklı: {len(mismatched)}/{len(saved_state)} "
        f"(ilk 10: {mismatched[:10]})"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
