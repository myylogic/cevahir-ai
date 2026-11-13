"""
FAZE 2: Training Management Checkpoint Kaydetme Testi

Bu test, training_management modülündeki eğitim sürecini simüle eder ve
checkpoint kaydetme sürecinde hangi model instance'ının kullanıldığını test eder.

Hedef:
- TrainingManager'ın hangi model instance'ını kullandığını doğrula
- CheckpointManager'ın hangi model state_dict'ini kaydettiğini doğrula
- Basit 3 keyli modelin nereden geldiğini bul

Notlar:
- Testler eğitim config'inin (training_system/train.py TRAIN_CONFIG) tam boyutlu
  modelini kurar; model modül başına bir kez kurulur ve testler arasında paylaşılır.
- Checkpoint'ler ve veri önbelleği pytest'in geçici dizinlerine yazılır; repo
  ağacına hiçbir şey yazılmaz. Eski sürümdeki sonuç JSON'u
  (tests/test_results_training_management.json) artık üretilmez; teşhis bilgisi
  assert mesajlarında.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

# Root dizini path'e ekle (dosya doğrudan çalıştırıldığında da import'lar çalışsın)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Modül seviyesinde logging.basicConfig yok: kök logger'ı yapılandırmak test
# oturumunun geri kalanını etkiliyordu. Standart logger yeterli (pytest yakalar).
logger = logging.getLogger(__name__)

# CheckpointManager.save() ve TrainingService ile aynı "SimpleModel" ölçütü
SIMPLE_MODEL_KEYS = ("embed.weight", "proj.weight", "proj.bias")


def is_simple_model(keys) -> bool:
    """3 keyli test modeli (SimpleModel) state_dict'i mi?"""
    keys = list(keys)
    return len(keys) == 3 and all(k in keys for k in SIMPLE_MODEL_KEYS)


class _StdTrainingLogger(logging.LoggerAdapter):
    """TrainingManager bileşenlerinin beklediği log_* arayüzü, standart logging'e yazar.

    Varsayılan TrainingLogger süreç genelinde tekildir (singleton) ve tensorboard
    kuruluysa ./logs/tb altına yazar; testler repo ağacına yazmamalı.
    """

    def __init__(self, name: str):
        super().__init__(logging.getLogger(name), {})

    def log_info(self, msg, *args, **kwargs):
        self.info(msg)

    def log_warning(self, msg, *args, **kwargs):
        self.warning(msg)

    def log_error(self, msg, *args, exc_info=False, **kwargs):
        self.error(msg, exc_info=exc_info)

    def log_debug(self, msg, *args, **kwargs):
        self.debug(msg)


def load_config():
    """Config dosyasını yükle (eğitim config'i + tokenizer config'i)"""
    from training_system.train import TRAIN_CONFIG
    config = TRAIN_CONFIG.copy()

    # Tokenizer config'i yükle
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
                "data/vocab_lib/vocab.json"
            ),
            "merges_path": (
                bpe_config.get("merges_path") or
                bpe_config.get("merges_file") or
                bpe_legacy.get("merges_file") or
                "data/merges_lib/merges.txt"
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


def load_checkpoint_state_dict(checkpoint_path):
    """Kaydedilen checkpoint'i yükle ve model state_dict'ini döndür."""
    # Ürünün resume yolu da weights_only=True ile yükler (training_manager.resume_from_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", None))
    assert state_dict is not None, (
        f"Checkpoint'te model_state_dict yok! Anahtarlar: {sorted(checkpoint)}"
    )
    return state_dict


def assert_checkpoint_matches_model(checkpoint_state_dict, model_state_dict, saver):
    """Checkpoint'teki state_dict, kaydedilen modelin state_dict'i mi?"""
    checkpoint_keys = list(checkpoint_state_dict.keys())
    assert not is_simple_model(checkpoint_keys), (
        f"KRİTİK: {saver}, SimpleModel state_dict kaydetmiş! "
        f"Bu, model instance'ının yanlış olduğunu gösteriyor. Keys: {checkpoint_keys}"
    )

    missing = set(model_state_dict) - set(checkpoint_keys)
    unexpected = set(checkpoint_keys) - set(model_state_dict)
    assert not missing and not unexpected, (
        f"{saver}: checkpoint key'leri modelinkiyle eşleşmiyor. "
        f"Missing ({len(missing)}, ilk 10): {sorted(missing)[:10]}; "
        f"Unexpected ({len(unexpected)}, ilk 10): {sorted(unexpected)[:10]}"
    )

    # Aynı mimarideki başka bir instance da aynı key'lere sahip olurdu; değerler de aynı olmalı.
    different = [
        key for key, value in model_state_dict.items()
        if not torch.equal(checkpoint_state_dict[key], value.detach().cpu())
    ]
    assert not different, (
        f"{saver}: checkpoint'teki ağırlıklar modelin ağırlıklarından farklı "
        f"({len(different)} tensör, ilk 10): {different[:10]}"
    )


# ---------------------------------------------------------------------------
# Fixture'lar
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def model_manager(config):
    """Eğitim config'iyle ModelManager (tam boyutlu model) kur; modül içinde paylaşılır."""
    from model_management.model_manager import ModelManager

    manager = ModelManager(dict(config))
    manager.initialize(
        build_optimizer=True,
        build_criterion=True,
        build_scheduler=True
    )
    return manager


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Geçici checkpoint dizini; tam boyutlu checkpoint'ler (~3 x 200 MB) test sonunda silinir."""
    path = tmp_path / "checkpoints"
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Testler
# ---------------------------------------------------------------------------

def test_training_manager_model_instance(model_manager):
    """TrainingManager'ın hangi model instance'ını kullandığını test et
    (ModelManager eğitim config'inden CevahirNeuralNetwork kurmalı, 3 keyli SimpleModel değil)."""
    from src.neural_network import CevahirNeuralNetwork

    model = model_manager.model
    assert model is not None, "ModelManager.model None!"

    model_keys = list(model.state_dict().keys())
    logger.info("ModelManager.model type: %s, state dict keys: %d, ilk 10: %s",
                type(model).__name__, len(model_keys), model_keys[:10])

    assert not is_simple_model(model_keys), (
        f"KRİTİK: ModelManager, SimpleModel instance'ı oluşturmuş! Keys: {model_keys}"
    )
    assert isinstance(model, CevahirNeuralNetwork), (
        f"ModelManager.model beklenen CevahirNeuralNetwork değil: {type(model).__name__} "
        f"(ilk 10 key: {model_keys[:10]})"
    )


def test_training_service_to_training_manager(config, tmp_path):
    """TrainingService'ten TrainingManager'a geçilen model instance'ını test et"""
    from training_system.v2.core.training_service import TrainingService
    from src.neural_network import CevahirNeuralNetwork

    service_config = dict(config)
    # Varsayılan cache_dir (".cache/preprocessed_data") çalışma dizinine, yani repo ağacına
    # yazıyordu; önbellek geçici dizinde tutulur. data_dir çalışma dizininden bağımsız olsun.
    service_config["cache_dir"] = str(tmp_path / "preprocessed_data")
    service_config["data_dir"] = str(ROOT / config["data_dir"])

    # TrainingService oluştur
    training_service = TrainingService(service_config)

    # ModelManager'dan model'i kontrol et
    model_manager = training_service.model_manager
    assert model_manager is not None and model_manager.model is not None, (
        "TrainingService.model_manager.model None!"
    )

    model = model_manager.model
    model_keys = list(model.state_dict().keys())
    logger.info("TrainingService.model_manager.model type: %s, state dict keys: %d",
                type(model).__name__, len(model_keys))

    assert not is_simple_model(model_keys), (
        f"KRİTİK: TrainingService, SimpleModel instance'ı oluşturmuş! Keys: {model_keys}"
    )
    assert isinstance(model, CevahirNeuralNetwork), (
        f"TrainingService modeli beklenen CevahirNeuralNetwork değil: {type(model).__name__}"
    )


def test_checkpoint_manager_save(model_manager, checkpoint_dir):
    """CheckpointManager'ın hangi model state_dict'ini kaydettiğini test et"""
    from training_management.v2.utils.checkpoint_manager import CheckpointManager

    model = model_manager.model
    assert model is not None, "Model None!"

    # Model state dict'ini al (kaydetmeden önce)
    model_state_dict_before = model.state_dict()

    # CheckpointManager oluştur
    checkpoint_manager = CheckpointManager(
        checkpoint_model_dir=str(checkpoint_dir),
        max_checkpoints=5
    )

    # Checkpoint kaydet
    checkpoint_path = checkpoint_manager.save(
        model=model,
        optimizer=model_manager.optimizer,
        epoch=1,
        metric=0.5,
        training_history={"train_loss": [1.0], "val_loss": [0.5]},
        with_optimizer=True
    )

    # Eski test sabit "checkpoint_epoch_1.pth" adını arıyordu; dosya adı kalıbı artık
    # checkpoint_epoch_{epoch:04d}.pth. save() kaydettiği dosyanın yolunu döndürür.
    assert os.path.isfile(checkpoint_path), (
        f"Checkpoint dosyası bulunamadı: {checkpoint_path} (dizin: {sorted(os.listdir(checkpoint_dir))})"
    )

    # Kaydedilen checkpoint'i yükle ve model state dict'iyle karşılaştır
    checkpoint_state_dict = load_checkpoint_state_dict(checkpoint_path)
    assert_checkpoint_matches_model(checkpoint_state_dict, model_state_dict_before, "CheckpointManager")


def test_training_manager_checkpoint_save(model_manager, config, checkpoint_dir):
    """TrainingManager'ın checkpoint kaydetme sürecini test et"""
    from training_management.v2.core.training_manager import TrainingManager
    from training_management.v2.utils.checkpoint_manager import CheckpointManager
    from torch.utils.data import DataLoader, TensorDataset

    model = model_manager.model
    assert model is not None, "Model None!"

    # Model state dict'ini al (kaydetmeden önce)
    model_state_dict_before = model.state_dict()

    # Dummy DataLoader'lar oluştur
    dummy_data = torch.randint(0, 100, (10, 20))
    dummy_targets = torch.randint(0, 100, (10, 20))
    dataset = TensorDataset(dummy_data, dummy_targets)
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)

    # CheckpointManager oluştur
    checkpoint_manager = CheckpointManager(
        checkpoint_model_dir=str(checkpoint_dir),
        max_checkpoints=5
    )

    # TensorBoard kapalı ve logger enjekte: varsayılanlar ./runs ve (tensorboard kuruluysa)
    # ./logs/tb altına, yani repo ağacına yazar. Kontrol edilen model/checkpoint davranışını etkilemez.
    tm_config = dict(config)
    tm_config.update(enable_tensorboard=False, checkpoint_dir=str(checkpoint_dir))

    # TrainingManager oluştur (TrainingService'in yaptığı gibi)
    training_manager = TrainingManager(
        model=model,  # ModelManager.model'i geçir
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=model_manager.optimizer,
        criterion=model_manager.criterion,
        config=tm_config,
        logger=_StdTrainingLogger(__name__),
        checkpoint_manager=checkpoint_manager
    )

    assert training_manager.model is model, (
        f"KRİTİK: TrainingManager farklı bir model instance'ı kullanıyor "
        f"({type(training_manager.model).__name__})"
    )

    # TrainingManager'ın model state dict'ini kontrol et
    training_manager_keys = set(training_manager.model.state_dict().keys())
    assert training_manager_keys == set(model_state_dict_before), (
        "KRİTİK: TrainingManager, farklı bir model instance'ı kullanıyor! "
        f"Yalnız ModelManager'da: {sorted(set(model_state_dict_before) - training_manager_keys)[:10]}, "
        f"yalnız TrainingManager'da: {sorted(training_manager_keys - set(model_state_dict_before))[:10]}"
    )

    # Checkpoint kaydet (TrainingManager'ın yaptığı gibi)
    checkpoint_manager.save(
        model=training_manager.model,
        optimizer=training_manager.optimizer,
        epoch=1,
        metric=0.5,
        training_history={"train_loss": [1.0], "val_loss": [0.5]},
        with_optimizer=True
    )

    # Kaydedilen checkpoint'i yükle
    checkpoint_path = checkpoint_dir / "best.pth"
    if not checkpoint_path.exists():
        checkpoint_path = checkpoint_dir / "last.pth"
    assert checkpoint_path.exists(), (
        f"Checkpoint dosyası bulunamadı: {checkpoint_dir} (içerik: {sorted(os.listdir(checkpoint_dir))})"
    )

    checkpoint_state_dict = load_checkpoint_state_dict(checkpoint_path)
    assert_checkpoint_matches_model(checkpoint_state_dict, model_state_dict_before, "TrainingManager")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
