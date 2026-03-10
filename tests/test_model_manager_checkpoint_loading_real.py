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
"""

import torch
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Root dizini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Config dosyasını yükle - train.py'deki yöntemi kullan"""
    try:
        from training_system.train import TRAIN_CONFIG
        config = TRAIN_CONFIG.copy()
        tokenizer_config_from_train = load_tokenizer_config_from_train_py()
        config.update(tokenizer_config_from_train)
        return config
    except Exception as e:
        logger.warning(f"TRAIN_CONFIG yüklenemedi, alternatif yöntem deneniyor: {e}")
        return {}


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
        return config
    except ImportError:
        logger.error("tokenizer_management.config import edilemedi. Lütfen path'i kontrol edin.")
        return {}


def find_saved_models():
    """Eğitilmiş modelleri bul"""
    models = []
    
    # Test checkpoint'leri
    test_checkpoints_dir = Path("tests/test_checkpoints")
    if test_checkpoints_dir.exists():
        for checkpoint_file in test_checkpoints_dir.glob("*.pth"):
            models.append({
                "name": checkpoint_file.stem,
                "path": str(checkpoint_file),
                "type": "test_checkpoints"
            })
    
    # Saved models
    saved_models_dir = Path("saved_models")
    if saved_models_dir.exists():
        for checkpoint_file in saved_models_dir.rglob("*.pth"):
            models.append({
                "name": checkpoint_file.stem,
                "path": str(checkpoint_file),
                "type": "saved_models"
            })
    
    return models


def test_checkpoint_format_analysis(checkpoint_path: str) -> Dict[str, Any]:
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


def test_model_weights_verification_real(checkpoint_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Gerçek model ile model weights doğrulama"""
    try:
        from model_management.model_manager import ModelManager
        from tokenizer_management.core.tokenizer_core import TokenizerCore
        
        # TokenizerCore oluştur (vocab_size almak için)
        tokenizer_config = config.copy()
        if "tokenizer" in config and isinstance(config["tokenizer"], dict):
            tokenizer_config.update(config["tokenizer"])
        
        if "vocab_path" not in tokenizer_config:
            tokenizer_config["vocab_path"] = config.get("vocab_path", "data/vocab_lib/vocab.json")
        if "merges_path" not in tokenizer_config:
            tokenizer_config["merges_path"] = config.get("merges_path", "data/merges_lib/merges.txt")
        
        tokenizer_core = TokenizerCore(tokenizer_config)
        vocab_size = tokenizer_core.get_vocab_size()
        config["vocab_size"] = vocab_size
        
        # ModelManager oluştur
        model_manager = ModelManager(config)
        model_manager.config["vocab_size"] = vocab_size
        model_manager.initialize()
        
        # Model weights'i kontrol et (yüklemeden önce)
        first_param_before = next(iter(model_manager.model.parameters()))
        param_mean_before = first_param_before.mean().item()
        param_std_before = first_param_before.std().item()
        
        logger.info(f"Yükleme öncesi: mean={param_mean_before:.6f}, std={param_std_before:.6f}")
        
        # Checkpoint yükle
        try:
            model_manager.load(checkpoint_path, strict=False)
            load_success = True
            load_error = None
        except Exception as e:
            load_success = False
            load_error = str(e)
            logger.error(f"Checkpoint yükleme hatası: {e}")
        
        # Model weights'i kontrol et (yüklemeden sonra)
        first_param_after = next(iter(model_manager.model.parameters()))
        param_mean_after = first_param_after.mean().item()
        param_std_after = first_param_after.std().item()
        
        logger.info(f"Yükleme sonrası: mean={param_mean_after:.6f}, std={param_std_after:.6f}")
        
        # Doğrulama
        weights_changed = not torch.equal(first_param_before, first_param_after)
        is_random_weights = abs(param_mean_after) < 0.001 and param_std_after < 0.15
        
        result = {
            "load_success": load_success,
            "load_error": load_error,
            "weights_changed": weights_changed,
            "param_mean_before": param_mean_before,
            "param_std_before": param_std_before,
            "param_mean_after": param_mean_after,
            "param_std_after": param_std_after,
            "is_random_weights": is_random_weights,
            "epoch_loaded": model_manager.config.get("current_epoch", None),
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Test hatası: {e}", exc_info=True)
        return {
            "load_success": False,
            "load_error": str(e),
            "weights_changed": False,
            "is_random_weights": True,
        }


def test_checkpoint_loading_industry_standards(checkpoint_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Endüstri standartlarına göre checkpoint yükleme testi"""
    logger.info("=" * 80)
    logger.info("ENDÜSTRİ STANDARTLARI KONTROLÜ")
    logger.info("=" * 80)
    
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
    
    try:
        from model_management.model_manager import ModelManager
        from tokenizer_management.core.tokenizer_core import TokenizerCore
        
        # TokenizerCore oluştur
        tokenizer_config = config.copy()
        if "tokenizer" in config and isinstance(config["tokenizer"], dict):
            tokenizer_config.update(config["tokenizer"])
        
        if "vocab_path" not in tokenizer_config:
            tokenizer_config["vocab_path"] = config.get("vocab_path", "data/vocab_lib/vocab.json")
        if "merges_path" not in tokenizer_config:
            tokenizer_config["merges_path"] = config.get("merges_path", "data/merges_lib/merges.txt")
        
        tokenizer_core = TokenizerCore(tokenizer_config)
        vocab_size = tokenizer_core.get_vocab_size()
        config["vocab_size"] = vocab_size
        
        # ModelManager oluştur
        model_manager = ModelManager(config)
        model_manager.config["vocab_size"] = vocab_size
        model_manager.initialize()
        
        # Checkpoint format'ını kontrol et
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 1. Model state dict yüklenebilmeli
        if "model_state_dict" in checkpoint or "state_dict" in checkpoint:
            standards["model_state_dict_loaded"] = True
        
        # 2. Optimizer state dict kontrolü
        if "optimizer_state_dict" in checkpoint:
            standards["optimizer_state_dict_loaded"] = True
        
        # 3. Scheduler state dict kontrolü
        if "scheduler_state_dict" in checkpoint:
            standards["scheduler_state_dict_loaded"] = True
        
        # 4. Epoch bilgisi kontrolü
        if "epoch" in checkpoint:
            standards["epoch_info_loaded"] = True
        
        # 5. Config bilgisi kontrolü
        if "config" in checkpoint:
            standards["config_info_loaded"] = True
        
        # Checkpoint yükle (exception handling kontrolü)
        try:
            model_manager.load(checkpoint_path, strict=False)
            
            # Model weights doğrulama
            first_param = next(iter(model_manager.model.parameters()))
            param_mean = first_param.mean().item()
            param_std = first_param.std().item()
            is_random_weights = abs(param_mean) < 0.001 and param_std < 0.15
            
            if not is_random_weights:
                standards["model_weights_verified"] = True
            
            # Epoch bilgisi yüklendi mi?
            if model_manager.config.get("current_epoch") is not None:
                standards["epoch_info_loaded"] = True
            
            standards["exception_handling_proper"] = True  # Exception fırlatılmadı
            
        except Exception as e:
            logger.error(f"Checkpoint yükleme hatası: {e}")
            standards["exception_handling_proper"] = False  # Exception fırlatıldı
        
        return standards
        
    except Exception as e:
        logger.error(f"Test hatası: {e}", exc_info=True)
        return standards


def main():
    """Ana test fonksiyonu"""
    logger.info("=" * 80)
    logger.info("FAZE 5.1.2: MODELMANAGER CHECKPOINT LOADING - GERÇEK MODEL İLE TEST")
    logger.info("=" * 80)
    logger.info("\nBu test, GERÇEK EĞİTİLMİŞ MODEL ile ModelManager'ın checkpoint")
    logger.info("yükleme mekanizmasını endüstri standartlarına göre test eder.")
    logger.info("=" * 80)
    
    # 1. Config yükle
    logger.info("\n1. Config yükleniyor...")
    config = load_config()
    if not config:
        logger.error("❌ Config yüklenemedi!")
        return False
    logger.info("✅ Config yüklendi")
    
    # 2. Checkpoint'leri bul
    logger.info("\n2. Checkpoint'ler aranıyor...")
    models = find_saved_models()
    if not models:
        logger.error("❌ Checkpoint bulunamadı!")
        return False
    
    logger.info(f"✅ {len(models)} checkpoint bulundu")
    for i, model in enumerate(models, 1):
        logger.info(f"   {i}. {model['name']} ({model['type']})")
    
    # En iyi checkpoint'i seç (saved_models/checkpoints öncelikli)
    selected_model = None
    # Önce saved_models/checkpoints içinde best.pth ara
    for model in models:
        if model['type'] == 'saved_models' and 'checkpoints' in model['path'] and model['name'] == 'best':
            selected_model = model
            break
    
    # Bulunamazsa en yeni checkpoint'i kullan
    if selected_model is None:
        # saved_models içindeki en yeni checkpoint
        saved_models_checkpoints = [m for m in models if m['type'] == 'saved_models' and 'checkpoints' in m['path']]
        if saved_models_checkpoints:
            # Dosya tarihine göre sırala (en yeni önce)
            saved_models_checkpoints.sort(key=lambda x: os.path.getmtime(x['path']), reverse=True)
            selected_model = saved_models_checkpoints[0]
    
    # Hala bulunamazsa ilk checkpoint'i kullan
    if selected_model is None:
        selected_model = models[0]
    
    checkpoint_path = selected_model['path']
    logger.info(f"\nKullanılacak checkpoint: {selected_model['name']} ({checkpoint_path})")
    logger.info(f"Checkpoint tipi: {selected_model['type']}")
    
    # 3. Checkpoint format analizi
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Checkpoint Format Analizi")
    logger.info("=" * 80)
    
    format_analysis = test_checkpoint_format_analysis(checkpoint_path)
    logger.info("Checkpoint Format:")
    for key, value in format_analysis.items():
        logger.info(f"  {key}: {value}")
    
    # 4. Model weights doğrulama
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Model Weights Doğrulama")
    logger.info("=" * 80)
    
    weights_result = test_model_weights_verification_real(checkpoint_path, config)
    logger.info("Model Weights Doğrulama Sonuçları:")
    for key, value in weights_result.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # 5. Endüstri standartları kontrolü
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Endüstri Standartları Kontrolü")
    logger.info("=" * 80)
    
    standards_result = test_checkpoint_loading_industry_standards(checkpoint_path, config)
    logger.info("Endüstri Standartları Kontrolü:")
    for key, value in standards_result.items():
        status = "✅" if value else "❌"
        logger.info(f"  {status} {key}: {value}")
    
    # Özet
    logger.info("\n" + "=" * 80)
    logger.info("TEST SONUÇLARI ÖZETİ")
    logger.info("=" * 80)
    
    # Test 1: Format analizi
    format_ok = format_analysis.get("has_model_state_dict", False)
    logger.info(f"{'✅' if format_ok else '❌'} TEST 1: Checkpoint Format Analizi")
    
    # Test 2: Weights doğrulama
    weights_ok = weights_result.get("load_success", False) and weights_result.get("weights_changed", False) and not weights_result.get("is_random_weights", True)
    logger.info(f"{'✅' if weights_ok else '❌'} TEST 2: Model Weights Doğrulama")
    if weights_result.get("load_error"):
        logger.info(f"   Hata: {weights_result['load_error'][:200]}")
    
    # Test 3: Endüstri standartları
    standards_ok = all(standards_result.values())
    passed_standards = sum(1 for v in standards_result.values() if v)
    total_standards = len(standards_result)
    logger.info(f"{'✅' if standards_ok else '⚠️'} TEST 3: Endüstri Standartları ({passed_standards}/{total_standards})")
    
    logger.info("=" * 80)
    
    # Genel değerlendirme
    all_tests_passed = format_ok and weights_ok and standards_ok
    
    if all_tests_passed:
        logger.info("\n🎉 TÜM TESTLER BAŞARILI!")
        logger.info("ModelManager checkpoint yükleme endüstri standartlarına uygun.")
    else:
        logger.info("\n⚠️ BAZI TESTLER BAŞARISIZ!")
        logger.info("ModelManager checkpoint yükleme mekanizmasında sorunlar tespit edildi.")
        logger.info("Test sonuçlarına göre ModelManager güncellenmeli.")
    
    return all_tests_passed


if __name__ == "__main__":
    success = main()

