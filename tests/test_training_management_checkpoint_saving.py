"""
FAZE 2: Training Management Checkpoint Kaydetme Testi

Bu test, training_management modülündeki eğitim sürecini simüle eder ve
checkpoint kaydetme sürecinde hangi model instance'ının kullanıldığını test eder.

Hedef:
- TrainingManager'ın hangi model instance'ını kullandığını doğrula
- CheckpointManager'ın hangi model state_dict'ini kaydettiğini doğrula
- Basit 3 keyli modelin nereden geldiğini bul
"""

import torch
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import tempfile
import shutil
from datetime import datetime

# Root dizini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test sonuçları için global dict
test_results = {
    "test_timestamp": datetime.now().isoformat(),
    "tests": {}
}


def load_config():
    """Config dosyasını yükle"""
    try:
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
    except Exception as e:
        logger.error(f"Config yüklenemedi: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_training_manager_model_instance():
    """TrainingManager'ın hangi model instance'ını kullandığını test et"""
    test_name = "test_1_model_manager"
    logger.info("=" * 80)
    logger.info("TEST 1: TrainingManager Model Instance Kontrolü")
    logger.info("=" * 80)
    
    result = {
        "test_name": test_name,
        "status": "failed",
        "model_type": None,
        "is_cevahir_neural_network": False,
        "state_dict_keys_count": 0,
        "is_simple_model": False,
        "first_10_keys": [],
        "error": None
    }
    
    config = load_config()
    if config is None:
        result["error"] = "Config yüklenemedi"
        test_results["tests"][test_name] = result
        return False
    
    try:
        from model_management.model_manager import ModelManager
        from src.neural_network import CevahirNeuralNetwork
        
        # ModelManager oluştur
        model_manager = ModelManager(config)
        model_manager.initialize(
            build_optimizer=True,
            build_criterion=True,
            build_scheduler=True
        )
        
        # Model instance'ını kontrol et
        model = model_manager.model
        if model is None:
            logger.error("❌ ModelManager.model None!")
            return
        
        logger.info(f"✅ ModelManager.model type: {type(model).__name__}")
        logger.info(f"✅ ModelManager.model class: {model.__class__}")
        logger.info(f"✅ Is CevahirNeuralNetwork? {isinstance(model, CevahirNeuralNetwork)}")
        
        # Model state dict'ini kontrol et
        model_state_dict = model.state_dict()
        model_keys = list(model_state_dict.keys())
        logger.info(f"✅ Model state dict keys: {len(model_keys)}")
        logger.info(f"   İlk 10 key: {model_keys[:10]}")
        
        # SimpleModel kontrolü
        is_simple_model = (
            len(model_keys) == 3 and 
            all(k in model_keys for k in ["embed.weight", "proj.weight", "proj.bias"])
        )
        logger.info(f"❌ SimpleModel mi? {is_simple_model}")
        
        # Sonuçları kaydet
        result["status"] = "passed" if not is_simple_model else "failed"
        result["model_type"] = type(model).__name__
        result["is_cevahir_neural_network"] = isinstance(model, CevahirNeuralNetwork)
        result["state_dict_keys_count"] = len(model_keys)
        result["is_simple_model"] = is_simple_model
        result["first_10_keys"] = model_keys[:10]
        
        if is_simple_model:
            logger.error("🚨 KRİTİK: ModelManager, SimpleModel instance'ı oluşturmuş!")
            result["error"] = "SimpleModel instance'ı oluşturulmuş"
        else:
            logger.info("✅ ModelManager, CevahirNeuralNetwork instance'ı oluşturmuş")
            
        test_results["tests"][test_name] = result
        return not is_simple_model
            
    except Exception as e:
        logger.error(f"❌ Test hatası: {e}", exc_info=True)
        result["error"] = str(e)
        test_results["tests"][test_name] = result
        return False


def test_training_service_to_training_manager():
    """TrainingService'ten TrainingManager'a geçilen model instance'ını test et"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: TrainingService → TrainingManager Model Geçişi")
    logger.info("=" * 80)
    
    config = load_config()
    if config is None:
        return
    
    try:
        from training_system.v2.core.training_service import TrainingService
        from src.neural_network import CevahirNeuralNetwork
        
        # TrainingService oluştur
        training_service = TrainingService(config)
        
        # ModelManager'dan model'i kontrol et
        model_manager = training_service.model_manager
        if model_manager is None or model_manager.model is None:
            logger.error("❌ TrainingService.model_manager.model None!")
            return False
        
        model = model_manager.model
        logger.info(f"✅ TrainingService.model_manager.model type: {type(model).__name__}")
        logger.info(f"✅ Is CevahirNeuralNetwork? {isinstance(model, CevahirNeuralNetwork)}")
        
        # Model state dict'ini kontrol et
        model_state_dict = model.state_dict()
        model_keys = list(model_state_dict.keys())
        logger.info(f"✅ Model state dict keys: {len(model_keys)}")
        
        # SimpleModel kontrolü
        is_simple_model = (
            len(model_keys) == 3 and 
            all(k in model_keys for k in ["embed.weight", "proj.weight", "proj.bias"])
        )
        logger.info(f"❌ SimpleModel mi? {is_simple_model}")
        
        if is_simple_model:
            logger.error("🚨 KRİTİK: TrainingService, SimpleModel instance'ı oluşturmuş!")
            return False
        else:
            logger.info("✅ TrainingService, CevahirNeuralNetwork instance'ı oluşturmuş")
            return True
            
    except Exception as e:
        logger.error(f"❌ Test hatası: {e}", exc_info=True)
        return False


def test_checkpoint_manager_save():
    """CheckpointManager'ın hangi model state_dict'ini kaydettiğini test et"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: CheckpointManager Save Süreci")
    logger.info("=" * 80)
    
    config = load_config()
    if config is None:
        return
    
    try:
        from model_management.model_manager import ModelManager
        from training_management.v2.utils.checkpoint_manager import CheckpointManager
        from src.neural_network import CevahirNeuralNetwork
        import tempfile
        import shutil
        
        # Geçici checkpoint dizini oluştur
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            # ModelManager oluştur
            model_manager = ModelManager(config)
            model_manager.initialize(
                build_optimizer=True,
                build_criterion=True,
                build_scheduler=True
            )
            
            model = model_manager.model
            if model is None:
                logger.error("❌ Model None!")
                return False
            
            logger.info(f"✅ Model type: {type(model).__name__}")
            logger.info(f"✅ Is CevahirNeuralNetwork? {isinstance(model, CevahirNeuralNetwork)}")
            
            # Model state dict'ini al (kaydetmeden önce)
            model_state_dict_before = model.state_dict()
            model_keys_before = list(model_state_dict_before.keys())
            logger.info(f"✅ Model state dict keys (before save): {len(model_keys_before)}")
            logger.info(f"   İlk 10 key: {model_keys_before[:10]}")
            
            # CheckpointManager oluştur
            checkpoint_manager = CheckpointManager(
                checkpoint_model_dir=checkpoint_dir,
                max_checkpoints=5
            )
            
            # Checkpoint kaydet
            checkpoint_manager.save(
                model=model,
                optimizer=model_manager.optimizer,
                epoch=1,
                metric=0.5,
                training_history={"train_loss": [1.0], "val_loss": [0.5]},
                with_optimizer=True
            )
            
            # Kaydedilen checkpoint'i yükle
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch_1.pth")
            if not os.path.exists(checkpoint_path):
                # Best veya last checkpoint'i kontrol et
                best_path = os.path.join(checkpoint_dir, "best.pth")
                last_path = os.path.join(checkpoint_dir, "last.pth")
                if os.path.exists(best_path):
                    checkpoint_path = best_path
                elif os.path.exists(last_path):
                    checkpoint_path = last_path
                else:
                    logger.error(f"❌ Checkpoint dosyası bulunamadı: {checkpoint_dir}")
                    return False
            
            logger.info(f"✅ Checkpoint kaydedildi: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Checkpoint'teki model state dict'i kontrol et
            checkpoint_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", None))
            if checkpoint_state_dict is None:
                logger.error("❌ Checkpoint'te model_state_dict yok!")
                return False
            
            checkpoint_keys = list(checkpoint_state_dict.keys())
            logger.info(f"✅ Checkpoint state dict keys: {len(checkpoint_keys)}")
            logger.info(f"   Keys: {checkpoint_keys}")
            
            # SimpleModel kontrolü
            is_simple_model = (
                len(checkpoint_keys) == 3 and 
                all(k in checkpoint_keys for k in ["embed.weight", "proj.weight", "proj.bias"])
            )
            
            if is_simple_model:
                logger.error("🚨 KRİTİK: CheckpointManager, SimpleModel state_dict kaydetmiş!")
                logger.error("   Bu, model instance'ının yanlış olduğunu gösteriyor!")
                return False
            else:
                logger.info("✅ CheckpointManager, doğru model state_dict kaydetmiş")
                
                # Keys eşleşmesi kontrolü
                keys_match = set(model_keys_before) == set(checkpoint_keys)
                logger.info(f"✅ Keys eşleşmesi: {keys_match}")
                if not keys_match:
                    missing = set(model_keys_before) - set(checkpoint_keys)
                    unexpected = set(checkpoint_keys) - set(model_keys_before)
                    logger.warning(f"   Missing keys: {len(missing)}")
                    logger.warning(f"   Unexpected keys: {len(unexpected)}")
                    if missing:
                        logger.warning(f"   Missing (ilk 10): {list(missing)[:10]}")
                    if unexpected:
                        logger.warning(f"   Unexpected (ilk 10): {list(unexpected)[:10]}")
                
                return True
                
        finally:
            # Geçici dizini temizle
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.error(f"❌ Test hatası: {e}", exc_info=True)
        return False


def test_training_manager_checkpoint_save():
    """TrainingManager'ın checkpoint kaydetme sürecini test et"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: TrainingManager Checkpoint Save Süreci")
    logger.info("=" * 80)
    
    config = load_config()
    if config is None:
        return
    
    try:
        from model_management.model_manager import ModelManager
        from training_management.v2.core.training_manager import TrainingManager
        from training_management.v2.utils.checkpoint_manager import CheckpointManager
        from src.neural_network import CevahirNeuralNetwork
        from torch.utils.data import DataLoader, TensorDataset
        import tempfile
        import shutil
        
        # Geçici checkpoint dizini oluştur
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            # ModelManager oluştur
            model_manager = ModelManager(config)
            model_manager.initialize(
                build_optimizer=True,
                build_criterion=True,
                build_scheduler=True
            )
            
            model = model_manager.model
            if model is None:
                logger.error("❌ Model None!")
                return False
            
            logger.info(f"✅ Model type: {type(model).__name__}")
            logger.info(f"✅ Is CevahirNeuralNetwork? {isinstance(model, CevahirNeuralNetwork)}")
            
            # Model state dict'ini al (kaydetmeden önce)
            model_state_dict_before = model.state_dict()
            model_keys_before = list(model_state_dict_before.keys())
            logger.info(f"✅ Model state dict keys (before save): {len(model_keys_before)}")
            
            # Dummy DataLoader'lar oluştur
            dummy_data = torch.randint(0, 100, (10, 20))
            dummy_targets = torch.randint(0, 100, (10, 20))
            dataset = TensorDataset(dummy_data, dummy_targets)
            train_loader = DataLoader(dataset, batch_size=2)
            val_loader = DataLoader(dataset, batch_size=2)
            
            # CheckpointManager oluştur
            checkpoint_manager = CheckpointManager(
                checkpoint_model_dir=checkpoint_dir,
                max_checkpoints=5
            )
            
            # TrainingManager oluştur (TrainingService'in yaptığı gibi)
            training_manager = TrainingManager(
                model=model,  # ModelManager.model'i geçir
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=model_manager.optimizer,
                criterion=model_manager.criterion,
                config=config,
                checkpoint_manager=checkpoint_manager
            )
            
            logger.info(f"✅ TrainingManager.model type: {type(training_manager.model).__name__}")
            logger.info(f"✅ TrainingManager.model is same instance? {training_manager.model is model}")
            
            # TrainingManager'ın model state dict'ini kontrol et
            training_manager_state_dict = training_manager.model.state_dict()
            training_manager_keys = list(training_manager_state_dict.keys())
            logger.info(f"✅ TrainingManager.model state dict keys: {len(training_manager_keys)}")
            
            # Keys eşleşmesi kontrolü
            keys_match = set(model_keys_before) == set(training_manager_keys)
            logger.info(f"✅ ModelManager ve TrainingManager keys eşleşmesi: {keys_match}")
            
            if not keys_match:
                logger.error("🚨 KRİTİK: TrainingManager, farklı bir model instance'ı kullanıyor!")
                return False
            
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
            checkpoint_path = os.path.join(checkpoint_dir, "best.pth")
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_dir, "last.pth")
            
            if not os.path.exists(checkpoint_path):
                logger.error(f"❌ Checkpoint dosyası bulunamadı: {checkpoint_dir}")
                return False
            
            logger.info(f"✅ Checkpoint kaydedildi: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Checkpoint'teki model state dict'i kontrol et
            checkpoint_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", None))
            if checkpoint_state_dict is None:
                logger.error("❌ Checkpoint'te model_state_dict yok!")
                return False
            
            checkpoint_keys = list(checkpoint_state_dict.keys())
            logger.info(f"✅ Checkpoint state dict keys: {len(checkpoint_keys)}")
            
            # SimpleModel kontrolü
            is_simple_model = (
                len(checkpoint_keys) == 3 and 
                all(k in checkpoint_keys for k in ["embed.weight", "proj.weight", "proj.bias"])
            )
            
            if is_simple_model:
                logger.error("🚨 KRİTİK: TrainingManager, SimpleModel state_dict kaydetmiş!")
                logger.error("   Bu, model instance'ının yanlış olduğunu gösteriyor!")
                return False
            else:
                logger.info("✅ TrainingManager, doğru model state_dict kaydetmiş")
                return True
                
        finally:
            # Geçici dizini temizle
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.error(f"❌ Test hatası: {e}", exc_info=True)
        return False


def save_results_to_json(output_path: str = "tests/test_results_training_management.json"):
    """Test sonuçlarını JSON dosyasına kaydet"""
    # Özet istatistikler
    total_tests = len(test_results["tests"])
    passed_tests = sum(1 for t in test_results["tests"].values() if t.get("status") == "passed")
    failed_tests = total_tests - passed_tests
    
    test_results["summary"] = {
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "all_passed": failed_tests == 0
    }
    
    # JSON dosyasına kaydet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ Test sonuçları kaydedildi: {output_path}")
    return output_path


def main():
    """Ana test fonksiyonu"""
    logger.info("=" * 80)
    logger.info("FAZE 2: TRAINING MANAGEMENT CHECKPOINT KAYDETME TESTİ")
    logger.info("=" * 80)
    logger.info("")
    
    results = {}
    
    # Test 1: ModelManager model instance
    results["test_1_model_manager"] = test_training_manager_model_instance()
    
    # Test 2: TrainingService → TrainingManager
    results["test_2_training_service"] = test_training_service_to_training_manager()
    
    # Test 3: CheckpointManager save
    results["test_3_checkpoint_manager"] = test_checkpoint_manager_save()
    
    # Test 4: TrainingManager checkpoint save
    results["test_4_training_manager"] = test_training_manager_checkpoint_save()
    
    # Sonuçları JSON'a kaydet
    json_path = save_results_to_json()
    
    # Özet
    logger.info("\n" + "=" * 80)
    logger.info("TEST ÖZETİ")
    logger.info("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info(f"\n{'✅ TÜM TESTLER BAŞARILI' if all_passed else '❌ BAZI TESTLER BAŞARISIZ'}")
    logger.info(f"📄 Detaylı sonuçlar: {json_path}")
    
    return all_passed


if __name__ == "__main__":
    main()

