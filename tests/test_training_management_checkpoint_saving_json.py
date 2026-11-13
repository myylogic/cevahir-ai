"""
FAZE 2: Training Management Checkpoint Kaydetme Testi (JSON Output varyantı)

Bu test, training_management modülündeki eğitim sürecini simüle eder ve
checkpoint kaydetme sürecinde hangi model instance'ının kullanıldığını test eder.
Checkpoint'ler is_best=True ile kaydedilir ve best.pth alias'ı üzerinden doğrulanır.

Hedef:
- TrainingManager'ın hangi model instance'ını kullandığını doğrula
- CheckpointManager'ın hangi model state_dict'ini kaydettiğini doğrula
- Basit 3 keyli modelin nereden geldiğini bul

Notlar:
- Eski sürüm sonuçları tests/test_results_training_management.json dosyasına, yani
  repo ağacına yazıyordu ve testler bool döndürüyordu. Testler artık assert eder;
  JSON'daki teşhis alanları (key sayıları, eksik/fazla key'ler, dosya yolu) assert
  mesajlarında yer alır.
- Testler eğitim config'inin tam boyutlu modelini kurar; model modül başına bir kez
  kurulur. Checkpoint'ler ve veri önbelleği pytest'in geçici dizinlerine yazılır.
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

# Modül seviyesinde logging.basicConfig yok; standart logger yeterli (pytest yakalar).
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


def load_best_checkpoint_state_dict(checkpoint_dir: Path):
    """is_best=True ile kaydedilen checkpoint'i (best.pth, yoksa last.pth) yükle."""
    checkpoint_path = checkpoint_dir / "best.pth"
    if not checkpoint_path.exists():
        checkpoint_path = checkpoint_dir / "last.pth"
    assert checkpoint_path.exists(), (
        f"Checkpoint dosyası bulunamadı: {checkpoint_dir} (içerik: {sorted(os.listdir(checkpoint_dir))})"
    )

    # Ürünün resume yolu da weights_only=True ile yükler (training_manager.resume_from_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", None))
    assert state_dict is not None, (
        f"Checkpoint'te model_state_dict yok ({checkpoint_path.name})! Anahtarlar: {sorted(checkpoint)}"
    )
    return state_dict, checkpoint_path


def assert_checkpoint_matches_model(checkpoint_state_dict, model_state_dict, saver, checkpoint_path):
    """Checkpoint'teki state_dict, kaydedilen modelin state_dict'i mi?"""
    checkpoint_keys = list(checkpoint_state_dict.keys())
    assert not is_simple_model(checkpoint_keys), (
        f"{saver}, SimpleModel state_dict kaydetmiş ({checkpoint_path}): {checkpoint_keys}"
    )

    missing = set(model_state_dict) - set(checkpoint_keys)
    unexpected = set(checkpoint_keys) - set(model_state_dict)
    assert not missing and not unexpected, (
        f"{saver}: keys_match=False ({checkpoint_path}); model_keys_before_save={len(model_state_dict)}, "
        f"checkpoint_keys_count={len(checkpoint_keys)}, missing_keys_count={len(missing)} "
        f"(ilk 10: {sorted(missing)[:10]}), unexpected_keys_count={len(unexpected)} "
        f"(ilk 10: {sorted(unexpected)[:10]})"
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
    assert model is not None, "Model None"

    model_keys = list(model.state_dict().keys())
    assert not is_simple_model(model_keys), (
        f"SimpleModel instance'ı oluşturulmuş: model_type={type(model).__name__}, keys={model_keys}"
    )
    assert isinstance(model, CevahirNeuralNetwork), (
        f"is_cevahir_neural_network=False: model_type={type(model).__name__}, "
        f"state_dict_keys_count={len(model_keys)}, first_10_keys={model_keys[:10]}"
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

    training_service = TrainingService(service_config)

    model_manager = training_service.model_manager
    assert model_manager is not None and model_manager.model is not None, (
        "TrainingService.model_manager.model None"
    )

    model = model_manager.model
    model_keys = list(model.state_dict().keys())
    assert not is_simple_model(model_keys), (
        f"SimpleModel instance'ı oluşturulmuş: model_type={type(model).__name__}, keys={model_keys}"
    )
    assert isinstance(model, CevahirNeuralNetwork), (
        f"is_cevahir_neural_network=False: model_type={type(model).__name__}, "
        f"state_dict_keys_count={len(model_keys)}, first_10_keys={model_keys[:10]}"
    )


def test_checkpoint_manager_save(model_manager, checkpoint_dir):
    """CheckpointManager'ın hangi model state_dict'ini kaydettiğini test et"""
    from training_management.v2.utils.checkpoint_manager import CheckpointManager

    model = model_manager.model
    assert model is not None, "Model None"

    model_state_dict_before = model.state_dict()

    checkpoint_manager = CheckpointManager(
        checkpoint_model_dir=str(checkpoint_dir),
        max_checkpoints=5
    )

    checkpoint_manager.save(
        model=model,
        optimizer=model_manager.optimizer,
        epoch=1,
        metric=0.5,
        training_history={"train_loss": [1.0], "val_loss": [0.5]},
        with_optimizer=True,
        is_best=True
    )

    checkpoint_state_dict, checkpoint_path = load_best_checkpoint_state_dict(checkpoint_dir)
    assert_checkpoint_matches_model(
        checkpoint_state_dict, model_state_dict_before, "CheckpointManager", checkpoint_path
    )


def test_training_manager_checkpoint_save(model_manager, config, checkpoint_dir):
    """TrainingManager'ın checkpoint kaydetme sürecini test et"""
    from training_management.v2.core.training_manager import TrainingManager
    from training_management.v2.utils.checkpoint_manager import CheckpointManager
    from torch.utils.data import DataLoader, TensorDataset

    model = model_manager.model
    assert model is not None, "Model None"

    model_state_dict_before = model.state_dict()

    dummy_data = torch.randint(0, 100, (10, 20))
    dummy_targets = torch.randint(0, 100, (10, 20))
    dataset = TensorDataset(dummy_data, dummy_targets)
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)

    checkpoint_manager = CheckpointManager(
        checkpoint_model_dir=str(checkpoint_dir),
        max_checkpoints=5
    )

    # TensorBoard kapalı ve logger enjekte: varsayılanlar ./runs ve (tensorboard kuruluysa)
    # ./logs/tb altına, yani repo ağacına yazar. Kontrol edilen model/checkpoint davranışını etkilemez.
    tm_config = dict(config)
    tm_config.update(enable_tensorboard=False, checkpoint_dir=str(checkpoint_dir))

    training_manager = TrainingManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=model_manager.optimizer,
        criterion=model_manager.criterion,
        config=tm_config,
        logger=_StdTrainingLogger(__name__),
        checkpoint_manager=checkpoint_manager
    )

    assert training_manager.model is model, (
        f"TrainingManager, farklı bir model instance'ı kullanıyor ({type(training_manager.model).__name__})"
    )

    training_manager_keys = set(training_manager.model.state_dict().keys())
    assert training_manager_keys == set(model_state_dict_before), (
        "keys_match_model_manager=False: TrainingManager, farklı bir model instance'ı kullanıyor "
        f"(model_keys_before_save={len(model_state_dict_before)}, "
        f"training_manager_keys_count={len(training_manager_keys)})"
    )

    checkpoint_manager.save(
        model=training_manager.model,
        optimizer=training_manager.optimizer,
        epoch=1,
        metric=0.5,
        training_history={"train_loss": [1.0], "val_loss": [0.5]},
        with_optimizer=True,
        is_best=True
    )

    checkpoint_state_dict, checkpoint_path = load_best_checkpoint_state_dict(checkpoint_dir)
    assert_checkpoint_matches_model(
        checkpoint_state_dict, model_state_dict_before, "TrainingManager", checkpoint_path
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
