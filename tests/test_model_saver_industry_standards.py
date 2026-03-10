"""
FAZE 3: ModelSaver Testleri - Endüstri Standartları

Bu test, ModelSaver'ın checkpoint kaydetme işleminin endüstri standartlarına
uygunluğunu test eder.

Endüstri Standartları:
1. ✅ Checkpoint format standart (model_state_dict, optimizer_state_dict, epoch, config, vb.)
2. ✅ Model state dict DOĞRU kaydedilmeli (tüm parametreler, 200+ key)
3. ✅ Atomik kayıt (atomic save)
4. ✅ Doğrulama (validation) - Kaydedilen checkpoint doğrulanabilmeli
5. ✅ Logging - Detaylı log mesajları
6. ✅ Error handling - Hatalı durumlarda uygun exception/warning

Test Senaryoları:
1. ✅ Checkpoint format kontrolü (hangi key'ler var?)
2. ✅ Model state dict kaydetme (key sayısı, key names kontrolü)
3. ✅ Checkpoint doğrulama (kaydedilen checkpoint'i yükleme ve kontrol)
4. ✅ Keys eşleşmesi kontrolü (kaydedilen vs model state dict)
"""

import torch
import sys
import os
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil
from datetime import datetime

# Root dizini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Config dosyasını yükle"""
    try:
        from training_system.train import TRAIN_CONFIG
        config = TRAIN_CONFIG.copy()
        
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
    except Exception as e:
        logger.warning(f"TRAIN_CONFIG yüklenemedi: {e}")
        return {}


def test_checkpoint_format():
    """Test 1: Checkpoint Format Kontrolü"""
    logger.info("=" * 80)
    logger.info("TEST 1: CHECKPOINT FORMAT KONTROLÜ")
    logger.info("=" * 80)
    
    try:
        from model_management.model_manager import ModelManager
        from tokenizer_management.core.tokenizer_core import TokenizerCore
        
        # Config yükle
        config = load_config()
        if not config:
            logger.error("❌ Config yüklenemedi")
            return False
        
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
        
        # ModelManager oluştur ve initialize et
        model_manager = ModelManager(config)
        model_manager.config["vocab_size"] = vocab_size
        model_manager.initialize()
        
        # Geçici dizin oluştur
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pth")
        
        try:
            # Checkpoint kaydet
            model_manager.save(
                save_path=checkpoint_path,
                epoch=1,
                additional_info={
                    "training_history": {"train_loss": [2.5], "val_loss": [2.4]},
                    "metric": 2.4,
                }
            )
            
            # Checkpoint'i yükle ve format kontrolü
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
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
            has_model_state_dict = "model_state_dict" in checkpoint or "state_dict" in checkpoint
            has_epoch = "epoch" in checkpoint
            has_config = "config" in checkpoint
            
            result = {
                "success": has_model_state_dict and has_epoch,
                "has_model_state_dict": has_model_state_dict,
                "has_state_dict": "state_dict" in checkpoint,
                "has_optimizer_state_dict": "optimizer_state_dict" in checkpoint or "optimizer_state" in checkpoint,
                "has_scheduler_state_dict": "scheduler_state_dict" in checkpoint or "scheduler_state" in checkpoint,
                "has_epoch": has_epoch,
                "has_config": has_config,
                "all_keys": list(checkpoint.keys()),
            }
            
            logger.info("Format Kontrolü Sonuçları:")
            logger.info(f"  ✅ Format uygun: {result['success']}")
            logger.info(f"  ✅ Model state dict: {result['has_model_state_dict']}")
            logger.info(f"  ✅ Epoch: {result['has_epoch']}")
            logger.info(f"  ✅ Config: {result['has_config']}")
            
            return result["success"]
            
        finally:
            # Geçici dizini temizle
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.error(f"❌ Test hatası: {e}", exc_info=True)
        return False


def test_model_state_dict_saving():
    """Test 2: Model State Dict Kaydetme Kontrolü"""
    logger.info("=" * 80)
    logger.info("TEST 2: MODEL STATE DICT KAYDETME KONTROLÜ")
    logger.info("=" * 80)
    
    try:
        from model_management.model_manager import ModelManager
        from tokenizer_management.core.tokenizer_core import TokenizerCore
        
        # Config yükle
        config = load_config()
        if not config:
            logger.error("❌ Config yüklenemedi")
            return False
        
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
        
        # ModelManager oluştur ve initialize et
        model_manager = ModelManager(config)
        model_manager.config["vocab_size"] = vocab_size
        model_manager.initialize()
        
        # Model state dict'i al (kaydetmeden önce)
        model = model_manager.model
        model_state_dict_before = model.state_dict()
        model_keys_before = set(model_state_dict_before.keys())
        num_keys_before = len(model_keys_before)
        
        logger.info(f"Model State Dict (Kaydetmeden Önce):")
        logger.info(f"  Key sayısı: {num_keys_before}")
        logger.info(f"  İlk 10 key: {list(model_keys_before)[:10]}")
        
        # Key sayısı kontrolü (endüstri standardı: 200+ key olmalı)
        if num_keys_before < 100:
            logger.warning(f"⚠️ Model state dict çok küçük: {num_keys_before} keys (beklenen: 200+)")
        elif num_keys_before < 200:
            logger.warning(f"⚠️ Model state dict beklenenden küçük: {num_keys_before} keys (beklenen: 200+)")
        else:
            logger.info(f"✅ Model state dict key sayısı normal: {num_keys_before} keys")
        
        # Geçici dizin oluştur
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pth")
        
        try:
            # Checkpoint kaydet
            model_manager.save(
                save_path=checkpoint_path,
                epoch=1,
            )
            
            # Checkpoint'i yükle
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Model state dict'i al (checkpoint'ten)
            checkpoint_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
            checkpoint_keys = set(checkpoint_state_dict.keys())
            num_keys_checkpoint = len(checkpoint_keys)
            
            logger.info(f"Checkpoint Model State Dict:")
            logger.info(f"  Key sayısı: {num_keys_checkpoint}")
            logger.info(f"  İlk 10 key: {list(checkpoint_keys)[:10]}")
            
            # Keys karşılaştırması
            common_keys = model_keys_before & checkpoint_keys
            missing_keys = model_keys_before - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys_before
            
            logger.info("Keys Karşılaştırması:")
            logger.info(f"  Model keys: {num_keys_before}")
            logger.info(f"  Checkpoint keys: {num_keys_checkpoint}")
            logger.info(f"  Ortak keys: {len(common_keys)}")
            logger.info(f"  Missing keys: {len(missing_keys)}")
            logger.info(f"  Unexpected keys: {len(unexpected_keys)}")
            
            # Sonuç
            keys_match = num_keys_before == num_keys_checkpoint and len(missing_keys) == 0 and len(unexpected_keys) == 0
            key_count_ok = num_keys_checkpoint >= 200  # Endüstri standardı
            
            result = {
                "success": keys_match and key_count_ok,
                "keys_match": keys_match,
                "key_count_ok": key_count_ok,
                "num_keys_before": num_keys_before,
                "num_keys_checkpoint": num_keys_checkpoint,
                "missing_keys_count": len(missing_keys),
                "unexpected_keys_count": len(unexpected_keys),
            }
            
            if missing_keys:
                logger.warning(f"⚠️ Missing keys (ilk 20): {list(missing_keys)[:20]}")
            if unexpected_keys:
                logger.warning(f"⚠️ Unexpected keys (ilk 20): {list(unexpected_keys)[:20]}")
            
            logger.info("Kaydetme Kontrolü Sonuçları:")
            logger.info(f"  ✅ Keys eşleşiyor: {result['keys_match']}")
            logger.info(f"  ✅ Key sayısı uygun (200+): {result['key_count_ok']}")
            logger.info(f"  ✅ Genel başarı: {result['success']}")
            
            return result["success"]
            
        finally:
            # Geçici dizini temizle
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.error(f"❌ Test hatası: {e}", exc_info=True)
        return False


def test_checkpoint_verification():
    """Test 3: Checkpoint Doğrulama (Kaydedilen checkpoint'i yükleme ve kontrol)"""
    logger.info("=" * 80)
    logger.info("TEST 3: CHECKPOINT DOĞRULAMA")
    logger.info("=" * 80)
    
    try:
        from model_management.model_manager import ModelManager
        from tokenizer_management.core.tokenizer_core import TokenizerCore
        
        # Config yükle
        config = load_config()
        if not config:
            logger.error("❌ Config yüklenemedi")
            return False
        
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
        
        # ModelManager oluştur ve initialize et
        model_manager_1 = ModelManager(config)
        model_manager_1.config["vocab_size"] = vocab_size
        model_manager_1.initialize()
        
        # İlk parametreyi al (kaydetmeden önce)
        first_param_before = next(iter(model_manager_1.model.parameters()))
        first_param_before_copy = first_param_before.clone().detach()
        param_mean_before = first_param_before_copy.mean().item()
        param_std_before = first_param_before_copy.std().item()
        
        logger.info(f"Model Parametreleri (Kaydetmeden Önce):")
        logger.info(f"  İlk parametre mean: {param_mean_before:.6f}")
        logger.info(f"  İlk parametre std: {param_std_before:.6f}")
        
        # Geçici dizin oluştur
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pth")
        
        try:
            # Checkpoint kaydet
            model_manager_1.save(
                save_path=checkpoint_path,
                epoch=1,
            )
            
            # Yeni bir ModelManager oluştur (checkpoint'i yüklemek için)
            model_manager_2 = ModelManager(config)
            model_manager_2.config["vocab_size"] = vocab_size
            model_manager_2.initialize()
            
            # Checkpoint'i yükle
            model_manager_2.load(checkpoint_path, strict=False)
            
            # Yükleme sonrası parametreleri al
            first_param_after = next(iter(model_manager_2.model.parameters()))
            first_param_after_copy = first_param_after.clone().detach()
            param_mean_after = first_param_after_copy.mean().item()
            param_std_after = first_param_after_copy.std().item()
            
            logger.info(f"Model Parametreleri (Yükleme Sonrası):")
            logger.info(f"  İlk parametre mean: {param_mean_after:.6f}")
            logger.info(f"  İlk parametre std: {param_std_after:.6f}")
            
            # Parametreler eşleşiyor mu?
            params_match = torch.allclose(first_param_before_copy, first_param_after_copy, atol=1e-6)
            mean_diff = abs(param_mean_before - param_mean_after)
            std_diff = abs(param_std_before - param_std_after)
            
            logger.info("Doğrulama Sonuçları:")
            logger.info(f"  ✅ Parametreler eşleşiyor: {params_match}")
            logger.info(f"  Mean fark: {mean_diff:.10f}")
            logger.info(f"  Std fark: {std_diff:.10f}")
            
            result = {
                "success": params_match,
                "params_match": params_match,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
            }
            
            return result["success"]
            
        finally:
            # Geçici dizini temizle
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.error(f"❌ Test hatası: {e}", exc_info=True)
        return False


def main():
    """Tüm testleri çalıştır"""
    logger.info("=" * 80)
    logger.info("FAZE 3: MODELSAVER TESTLERİ - ENDÜSTRİ STANDARTLARI")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Bu test, ModelSaver'ın checkpoint kaydetme işleminin")
    logger.info("endüstri standartlarına uygunluğunu test eder.")
    logger.info("=" * 80)
    logger.info("")
    
    results = {}
    
    # Test 1: Checkpoint Format
    results["test_1_format"] = test_checkpoint_format()
    logger.info("")
    
    # Test 2: Model State Dict Kaydetme
    results["test_2_state_dict"] = test_model_state_dict_saving()
    logger.info("")
    
    # Test 3: Checkpoint Doğrulama
    results["test_3_verification"] = test_checkpoint_verification()
    logger.info("")
    
    # Sonuçları özetle
    logger.info("=" * 80)
    logger.info("TEST SONUÇLARI ÖZETİ")
    logger.info("=" * 80)
    
    all_passed = all(results.values())
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        logger.info(f"{status} {test_name}: {'BAŞARILI' if result else 'BAŞARISIZ'}")
    
    logger.info("=" * 80)
    if all_passed:
        logger.info("🎉 TÜM TESTLER BAŞARILI!")
        logger.info("ModelSaver endüstri standartlarına uygun.")
    else:
        logger.info("⚠️ BAZI TESTLER BAŞARISIZ!")
        logger.info("ModelSaver güncellenmeli.")
    logger.info("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

