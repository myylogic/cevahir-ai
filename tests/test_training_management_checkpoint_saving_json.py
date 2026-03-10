"""
FAZE 2: Training Management Checkpoint Kaydetme Testi (JSON Output)

Bu test, training_management modülündeki eğitim sürecini simüle eder ve
checkpoint kaydetme sürecinde hangi model instance'ının kullanıldığını test eder.
Sonuçlar JSON dosyasına kaydedilir.

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

# Minimal logging (sadece hatalar)
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
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
            pass
        
        return config
    except Exception as e:
        return None


def test_training_manager_model_instance():
    """TrainingManager'ın hangi model instance'ını kullandığını test et"""
    test_name = "test_1_model_manager"
    
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
        
        model_manager = ModelManager(config)
        model_manager.initialize(
            build_optimizer=True,
            build_criterion=True,
            build_scheduler=True
        )
        
        model = model_manager.model
        if model is None:
            result["error"] = "Model None"
            test_results["tests"][test_name] = result
            return False
        
        model_state_dict = model.state_dict()
        model_keys = list(model_state_dict.keys())
        
        is_simple_model = (
            len(model_keys) == 3 and 
            all(k in model_keys for k in ["embed.weight", "proj.weight", "proj.bias"])
        )
        
        result["status"] = "passed" if not is_simple_model else "failed"
        result["model_type"] = type(model).__name__
        result["is_cevahir_neural_network"] = isinstance(model, CevahirNeuralNetwork)
        result["state_dict_keys_count"] = len(model_keys)
        result["is_simple_model"] = is_simple_model
        result["first_10_keys"] = model_keys[:10]
        
        if is_simple_model:
            result["error"] = "SimpleModel instance'ı oluşturulmuş"
            
        test_results["tests"][test_name] = result
        return not is_simple_model
            
    except Exception as e:
        result["error"] = str(e)
        test_results["tests"][test_name] = result
        return False


def test_training_service_to_training_manager():
    """TrainingService'ten TrainingManager'a geçilen model instance'ını test et"""
    test_name = "test_2_training_service"
    
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
        from training_system.v2.core.training_service import TrainingService
        from src.neural_network import CevahirNeuralNetwork
        
        training_service = TrainingService(config)
        
        model_manager = training_service.model_manager
        if model_manager is None or model_manager.model is None:
            result["error"] = "TrainingService.model_manager.model None"
            test_results["tests"][test_name] = result
            return False
        
        model = model_manager.model
        model_state_dict = model.state_dict()
        model_keys = list(model_state_dict.keys())
        
        is_simple_model = (
            len(model_keys) == 3 and 
            all(k in model_keys for k in ["embed.weight", "proj.weight", "proj.bias"])
        )
        
        result["status"] = "passed" if not is_simple_model else "failed"
        result["model_type"] = type(model).__name__
        result["is_cevahir_neural_network"] = isinstance(model, CevahirNeuralNetwork)
        result["state_dict_keys_count"] = len(model_keys)
        result["is_simple_model"] = is_simple_model
        result["first_10_keys"] = model_keys[:10]
        
        if is_simple_model:
            result["error"] = "SimpleModel instance'ı oluşturulmuş"
            
        test_results["tests"][test_name] = result
        return not is_simple_model
            
    except Exception as e:
        result["error"] = str(e)
        test_results["tests"][test_name] = result
        return False


def test_checkpoint_manager_save():
    """CheckpointManager'ın hangi model state_dict'ini kaydettiğini test et"""
    test_name = "test_3_checkpoint_manager"
    
    result = {
        "test_name": test_name,
        "status": "failed",
        "model_keys_before_save": 0,
        "checkpoint_keys_count": 0,
        "is_simple_model": False,
        "keys_match": False,
        "missing_keys_count": 0,
        "unexpected_keys_count": 0,
        "checkpoint_path": None,
        "error": None
    }
    
    config = load_config()
    if config is None:
        result["error"] = "Config yüklenemedi"
        test_results["tests"][test_name] = result
        return False
    
    try:
        from model_management.model_manager import ModelManager
        from training_management.v2.utils.checkpoint_manager import CheckpointManager
        from src.neural_network import CevahirNeuralNetwork
        
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            model_manager = ModelManager(config)
            model_manager.initialize(
                build_optimizer=True,
                build_criterion=True,
                build_scheduler=True
            )
            
            model = model_manager.model
            if model is None:
                result["error"] = "Model None"
                test_results["tests"][test_name] = result
                return False
            
            model_state_dict_before = model.state_dict()
            model_keys_before = list(model_state_dict_before.keys())
            result["model_keys_before_save"] = len(model_keys_before)
            
            checkpoint_manager = CheckpointManager(
                checkpoint_model_dir=checkpoint_dir,
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
            
            checkpoint_path = os.path.join(checkpoint_dir, "best.pth")
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_dir, "last.pth")
            
            if not os.path.exists(checkpoint_path):
                result["error"] = f"Checkpoint dosyası bulunamadı: {checkpoint_dir}"
                test_results["tests"][test_name] = result
                return False
            
            result["checkpoint_path"] = checkpoint_path
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            checkpoint_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", None))
            if checkpoint_state_dict is None:
                result["error"] = "Checkpoint'te model_state_dict yok"
                test_results["tests"][test_name] = result
                return False
            
            checkpoint_keys = list(checkpoint_state_dict.keys())
            result["checkpoint_keys_count"] = len(checkpoint_keys)
            
            is_simple_model = (
                len(checkpoint_keys) == 3 and 
                all(k in checkpoint_keys for k in ["embed.weight", "proj.weight", "proj.bias"])
            )
            result["is_simple_model"] = is_simple_model
            
            keys_match = set(model_keys_before) == set(checkpoint_keys)
            result["keys_match"] = keys_match
            
            if not keys_match:
                missing = set(model_keys_before) - set(checkpoint_keys)
                unexpected = set(checkpoint_keys) - set(model_keys_before)
                result["missing_keys_count"] = len(missing)
                result["unexpected_keys_count"] = len(unexpected)
            
            result["status"] = "passed" if not is_simple_model and keys_match else "failed"
            
            if is_simple_model:
                result["error"] = "CheckpointManager, SimpleModel state_dict kaydetmiş"
                
            test_results["tests"][test_name] = result
            return not is_simple_model and keys_match
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        result["error"] = str(e)
        test_results["tests"][test_name] = result
        return False


def test_training_manager_checkpoint_save():
    """TrainingManager'ın checkpoint kaydetme sürecini test et"""
    test_name = "test_4_training_manager"
    
    result = {
        "test_name": test_name,
        "status": "failed",
        "model_keys_before_save": 0,
        "training_manager_keys_count": 0,
        "checkpoint_keys_count": 0,
        "is_simple_model": False,
        "keys_match_model_manager": False,
        "keys_match_checkpoint": False,
        "checkpoint_path": None,
        "error": None
    }
    
    config = load_config()
    if config is None:
        result["error"] = "Config yüklenemedi"
        test_results["tests"][test_name] = result
        return False
    
    try:
        from model_management.model_manager import ModelManager
        from training_management.v2.core.training_manager import TrainingManager
        from training_management.v2.utils.checkpoint_manager import CheckpointManager
        from src.neural_network import CevahirNeuralNetwork
        from torch.utils.data import DataLoader, TensorDataset
        
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            model_manager = ModelManager(config)
            model_manager.initialize(
                build_optimizer=True,
                build_criterion=True,
                build_scheduler=True
            )
            
            model = model_manager.model
            if model is None:
                result["error"] = "Model None"
                test_results["tests"][test_name] = result
                return False
            
            model_state_dict_before = model.state_dict()
            model_keys_before = list(model_state_dict_before.keys())
            result["model_keys_before_save"] = len(model_keys_before)
            
            dummy_data = torch.randint(0, 100, (10, 20))
            dummy_targets = torch.randint(0, 100, (10, 20))
            dataset = TensorDataset(dummy_data, dummy_targets)
            train_loader = DataLoader(dataset, batch_size=2)
            val_loader = DataLoader(dataset, batch_size=2)
            
            checkpoint_manager = CheckpointManager(
                checkpoint_model_dir=checkpoint_dir,
                max_checkpoints=5
            )
            
            training_manager = TrainingManager(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=model_manager.optimizer,
                criterion=model_manager.criterion,
                config=config,
                checkpoint_manager=checkpoint_manager
            )
            
            training_manager_state_dict = training_manager.model.state_dict()
            training_manager_keys = list(training_manager_state_dict.keys())
            result["training_manager_keys_count"] = len(training_manager_keys)
            
            keys_match = set(model_keys_before) == set(training_manager_keys)
            result["keys_match_model_manager"] = keys_match
            
            if not keys_match:
                result["error"] = "TrainingManager, farklı bir model instance'ı kullanıyor"
                test_results["tests"][test_name] = result
                return False
            
            checkpoint_manager.save(
                model=training_manager.model,
                optimizer=training_manager.optimizer,
                epoch=1,
                metric=0.5,
                training_history={"train_loss": [1.0], "val_loss": [0.5]},
                with_optimizer=True,
                is_best=True
            )
            
            checkpoint_path = os.path.join(checkpoint_dir, "best.pth")
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_dir, "last.pth")
            
            if not os.path.exists(checkpoint_path):
                result["error"] = f"Checkpoint dosyası bulunamadı: {checkpoint_dir}"
                test_results["tests"][test_name] = result
                return False
            
            result["checkpoint_path"] = checkpoint_path
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            checkpoint_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", None))
            if checkpoint_state_dict is None:
                result["error"] = "Checkpoint'te model_state_dict yok"
                test_results["tests"][test_name] = result
                return False
            
            checkpoint_keys = list(checkpoint_state_dict.keys())
            result["checkpoint_keys_count"] = len(checkpoint_keys)
            
            is_simple_model = (
                len(checkpoint_keys) == 3 and 
                all(k in checkpoint_keys for k in ["embed.weight", "proj.weight", "proj.bias"])
            )
            result["is_simple_model"] = is_simple_model
            
            checkpoint_keys_match = set(model_keys_before) == set(checkpoint_keys)
            result["keys_match_checkpoint"] = checkpoint_keys_match
            
            result["status"] = "passed" if not is_simple_model and checkpoint_keys_match else "failed"
            
            if is_simple_model:
                result["error"] = "TrainingManager, SimpleModel state_dict kaydetmiş"
                
            test_results["tests"][test_name] = result
            return not is_simple_model and checkpoint_keys_match
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        result["error"] = str(e)
        test_results["tests"][test_name] = result
        return False


def save_results_to_json(output_path: str = "tests/test_results_training_management.json"):
    """Test sonuçlarını JSON dosyasına kaydet"""
    total_tests = len(test_results["tests"])
    passed_tests = sum(1 for t in test_results["tests"].values() if t.get("status") == "passed")
    failed_tests = total_tests - passed_tests
    
    test_results["summary"] = {
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "all_passed": failed_tests == 0
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Test sonuçları kaydedildi: {output_path}")
    return output_path


def main():
    """Ana test fonksiyonu"""
    print("=" * 80)
    print("FAZE 2: TRAINING MANAGEMENT CHECKPOINT KAYDETME TESTİ")
    print("=" * 80)
    print("")
    
    results = {}
    
    print("Test 1/4: ModelManager model instance...")
    results["test_1_model_manager"] = test_training_manager_model_instance()
    
    print("Test 2/4: TrainingService → TrainingManager...")
    results["test_2_training_service"] = test_training_service_to_training_manager()
    
    print("Test 3/4: CheckpointManager save...")
    results["test_3_checkpoint_manager"] = test_checkpoint_manager_save()
    
    print("Test 4/4: TrainingManager checkpoint save...")
    results["test_4_training_manager"] = test_training_manager_checkpoint_save()
    
    json_path = save_results_to_json()
    
    print("\n" + "=" * 80)
    print("TEST ÖZETİ")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'✅ TÜM TESTLER BAŞARILI' if all_passed else '❌ BAZI TESTLER BAŞARISIZ'}")
    print(f"📄 Detaylı sonuçlar: {json_path}")
    
    return all_passed


if __name__ == "__main__":
    main()

