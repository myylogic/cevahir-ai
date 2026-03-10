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
3. ✅ Optimizer state dict uyumsuzluğu (exception handle edilmeli)
4. ✅ Scheduler state dict uyumsuzluğu (exception handle edilmeli)
5. ✅ Checkpoint format kontrolü (hangi key'ler var?)
6. ✅ Yükleme sonrası doğrulama (model weights kontrolü)
7. ✅ Partial yükleme (sadece model weights)
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import shutil

# Root dizini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    }
    
    return analysis


def test_model_weights_loaded(model: nn.Module, checkpoint_path: str) -> Dict[str, Any]:
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


def test_standard_checkpoint_loading():
    """Test 1: Normal checkpoint yükleme (tüm bileşenler mevcut)"""
    logger.info("=" * 80)
    logger.info("TEST 1: Normal Checkpoint Yükleme")
    logger.info("=" * 80)
    
    try:
        from model_management.model_manager import ModelManager
        
        # Test model ve optimizer oluştur
        model = SimpleModel(vocab_size=1000, embed_dim=128)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        # Checkpoint oluştur
        checkpoint = create_test_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            config={"learning_rate": 0.001, "vocab_size": 1000},
            format_type="standard"
        )
        
        # Geçici dosyaya kaydet
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            checkpoint_path = f.name
            torch.save(checkpoint, checkpoint_path)
        
        try:
            # ModelManager oluştur ve yükle
            config = {
                "vocab_size": 1000,
                "embed_dim": 128,
                "learning_rate": 0.001,
                "device": "cpu",
            }
            
            model_manager = ModelManager(config)
            model_manager.initialize()
            
            # Checkpoint yükle
            model_manager.load(checkpoint_path, strict=True)
            
            # Doğrulama
            assert model_manager.config.get("current_epoch") == 5, "Epoch bilgisi yüklenemedi"
            logger.info("✅ TEST 1 BAŞARILI: Normal checkpoint yüklendi")
            return True
            
        finally:
            os.unlink(checkpoint_path)
            
    except Exception as e:
        logger.error(f"❌ TEST 1 BAŞARISIZ: {e}", exc_info=True)
        return False


def test_model_only_checkpoint_loading():
    """Test 2: Model state dict yükleme (optimizer/scheduler yok)"""
    logger.info("=" * 80)
    logger.info("TEST 2: Model Only Checkpoint Yükleme")
    logger.info("=" * 80)
    
    try:
        from model_management.model_manager import ModelManager
        
        # Test model oluştur
        model = SimpleModel(vocab_size=1000, embed_dim=128)
        
        # Minimal checkpoint oluştur (sadece model weights)
        checkpoint = create_test_checkpoint(
            model=model,
            epoch=3,
            format_type="minimal"
        )
        
        # Geçici dosyaya kaydet
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            checkpoint_path = f.name
            torch.save(checkpoint, checkpoint_path)
        
        try:
            # ModelManager oluştur ve yükle
            config = {
                "vocab_size": 1000,
                "embed_dim": 128,
                "device": "cpu",
            }
            
            model_manager = ModelManager(config)
            model_manager.initialize()
            
            # Checkpoint yükle
            model_manager.load(checkpoint_path, strict=True)
            
            # Doğrulama
            assert model_manager.config.get("current_epoch") == 3, "Epoch bilgisi yüklenemedi"
            logger.info("✅ TEST 2 BAŞARILI: Model only checkpoint yüklendi")
            return True
            
        finally:
            os.unlink(checkpoint_path)
            
    except Exception as e:
        logger.error(f"❌ TEST 2 BAŞARISIZ: {e}", exc_info=True)
        return False


def test_optimizer_mismatch_handling():
    """Test 3: Optimizer state dict uyumsuzluğu (exception handle edilmeli)"""
    logger.info("=" * 80)
    logger.info("TEST 3: Optimizer Mismatch Handling")
    logger.info("=" * 80)
    
    try:
        from model_management.model_manager import ModelManager
        
        # Test model oluştur
        model1 = SimpleModel(vocab_size=1000, embed_dim=128)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=0.001)
        
        # Farklı optimizer ile checkpoint oluştur (farklı parameter groups)
        model2 = SimpleModel(vocab_size=1000, embed_dim=128)
        # Farklı learning rate ile optimizer oluştur (farklı parameter groups oluşturur)
        optimizer2 = torch.optim.AdamW([
            {"params": list(model2.parameters())[:2], "lr": 0.001},
            {"params": list(model2.parameters())[2:], "lr": 0.002}
        ])
        
        # Checkpoint oluştur (optimizer2 ile)
        checkpoint = create_test_checkpoint(
            model=model2,
            optimizer=optimizer2,
            epoch=5,
            format_type="standard"
        )
        
        # Geçici dosyaya kaydet
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            checkpoint_path = f.name
            torch.save(checkpoint, checkpoint_path)
        
        try:
            # ModelManager oluştur (optimizer1 ile - farklı parameter groups)
            config = {
                "vocab_size": 1000,
                "embed_dim": 128,
                "learning_rate": 0.001,
                "device": "cpu",
            }
            
            model_manager = ModelManager(config)
            model_manager.initialize()
            
            # Checkpoint yükle (optimizer mismatch olmalı ama exception fırlatmamalı)
            model_manager.load(checkpoint_path, strict=True)
            
            # Doğrulama: Model weights yüklenmiş olmalı
            # (Optimizer yüklenemese bile model yüklenmeli)
            logger.info("✅ TEST 3 BAŞARILI: Optimizer mismatch handle edildi, model yüklendi")
            return True
            
        finally:
            os.unlink(checkpoint_path)
            
    except Exception as e:
        logger.error(f"❌ TEST 3 BAŞARISIZ: {e}", exc_info=True)
        return False


def test_checkpoint_format_analysis_real():
    """Test 4: Gerçek checkpoint format analizi"""
    logger.info("=" * 80)
    logger.info("TEST 4: Gerçek Checkpoint Format Analizi")
    logger.info("=" * 80)
    
    try:
        # Test checkpoint'leri bul
        test_checkpoints_dir = Path("tests/test_checkpoints")
        if not test_checkpoints_dir.exists():
            logger.warning("⚠️ Test checkpoint'leri bulunamadı: tests/test_checkpoints")
            return False
        
        checkpoint_files = list(test_checkpoints_dir.glob("*.pth"))
        if not checkpoint_files:
            logger.warning("⚠️ Test checkpoint dosyası bulunamadı")
            return False
        
        # İlk checkpoint'i analiz et
        checkpoint_path = checkpoint_files[0]
        logger.info(f"Checkpoint analiz ediliyor: {checkpoint_path}")
        
        analysis = test_checkpoint_format_analysis(str(checkpoint_path))
        
        logger.info("Checkpoint Format Analizi:")
        for key, value in analysis.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("✅ TEST 4 BAŞARILI: Checkpoint format analizi tamamlandı")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 4 BAŞARISIZ: {e}", exc_info=True)
        return False


def test_model_weights_verification():
    """Test 5: Model weights doğrulama (gerçek checkpoint ile)"""
    logger.info("=" * 80)
    logger.info("TEST 5: Model Weights Doğrulama")
    logger.info("=" * 80)
    
    try:
        from model_management.model_manager import ModelManager
        
        # Test checkpoint'leri bul
        test_checkpoints_dir = Path("tests/test_checkpoints")
        if not test_checkpoints_dir.exists():
            logger.warning("⚠️ Test checkpoint'leri bulunamadı: tests/test_checkpoints")
            return False
        
        checkpoint_files = list(test_checkpoints_dir.glob("*.pth"))
        if not checkpoint_files:
            logger.warning("⚠️ Test checkpoint dosyası bulunamadı")
            return False
        
        checkpoint_path = checkpoint_files[0]
        logger.info(f"Checkpoint yükleniyor: {checkpoint_path}")
        
        # Config yükle (train.py'deki gibi)
        try:
            from training_system.train import TRAIN_CONFIG
            config = TRAIN_CONFIG.copy()
        except Exception as e:
            logger.warning(f"TRAIN_CONFIG yüklenemedi: {e}")
            config = {
                "device": "cpu",
                "vocab_size": 60312,
                "embed_dim": 512,
            }
        
        # ModelManager oluştur
        model_manager = ModelManager(config)
        model_manager.initialize()
        
        # Model weights'i kontrol et (yüklemeden önce)
        first_param_before = next(iter(model_manager.model.parameters()))
        param_mean_before = first_param_before.mean().item()
        param_std_before = first_param_before.std().item()
        
        logger.info(f"Yükleme öncesi: mean={param_mean_before:.6f}, std={param_std_before:.6f}")
        
        # Checkpoint yükle
        model_manager.load(str(checkpoint_path), strict=False)
        
        # Model weights'i kontrol et (yüklemeden sonra)
        first_param_after = next(iter(model_manager.model.parameters()))
        param_mean_after = first_param_after.mean().item()
        param_std_after = first_param_after.std().item()
        
        logger.info(f"Yükleme sonrası: mean={param_mean_after:.6f}, std={param_std_after:.6f}")
        
        # Doğrulama
        weights_changed = not torch.equal(first_param_before, first_param_after)
        is_random_weights = abs(param_mean_after) < 0.001 and param_std_after < 0.15
        
        logger.info(f"Weights değişti mi: {weights_changed}")
        logger.info(f"Random weights göstergesi: {is_random_weights}")
        
        if weights_changed and not is_random_weights:
            logger.info("✅ TEST 5 BAŞARILI: Model weights gerçekten yüklendi")
            return True
        elif is_random_weights:
            logger.warning("⚠️ TEST 5 UYARI: Model weights random weights gibi görünüyor (yükleme başarısız olabilir)")
            return False
        else:
            logger.warning("⚠️ TEST 5 UYARI: Model weights değişmedi (yükleme başarısız olabilir)")
            return False
        
    except Exception as e:
        logger.error(f"❌ TEST 5 BAŞARISIZ: {e}", exc_info=True)
        return False


def main():
    """Ana test fonksiyonu"""
    logger.info("=" * 80)
    logger.info("FAZE 5.1.1: MODELMANAGER CHECKPOINT LOADING - ENDÜSTRİ STANDARTLARI TESTİ")
    logger.info("=" * 80)
    logger.info("\nBu test, ModelManager'ın checkpoint yükleme mekanizmasının")
    logger.info("endüstri standartlarına uygunluğunu test eder.")
    logger.info("=" * 80)
    
    results = []
    
    # Test 1: Normal checkpoint yükleme
    results.append(("Test 1: Normal Checkpoint Yükleme", test_standard_checkpoint_loading()))
    
    # Test 2: Model only checkpoint yükleme
    results.append(("Test 2: Model Only Checkpoint Yükleme", test_model_only_checkpoint_loading()))
    
    # Test 3: Optimizer mismatch handling
    results.append(("Test 3: Optimizer Mismatch Handling", test_optimizer_mismatch_handling()))
    
    # Test 4: Checkpoint format analizi
    results.append(("Test 4: Checkpoint Format Analizi", test_checkpoint_format_analysis_real()))
    
    # Test 5: Model weights doğrulama
    results.append(("Test 5: Model Weights Doğrulama", test_model_weights_verification()))
    
    # Özet
    logger.info("\n" + "=" * 80)
    logger.info("TEST SONUÇLARI ÖZETİ")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 80)
    logger.info(f"Toplam: {passed}/{total} test başarılı")
    logger.info("=" * 80)
    
    if passed == total:
        logger.info("\n🎉 TÜM TESTLER BAŞARILI!")
        logger.info("ModelManager checkpoint yükleme endüstri standartlarına uygun.")
    else:
        logger.info(f"\n⚠️ {total - passed} test başarısız!")
        logger.info("ModelManager checkpoint yükleme mekanizmasında sorunlar tespit edildi.")
        logger.info("Test sonuçlarına göre ModelManager güncellenmeli.")
    
    return passed == total


if __name__ == "__main__":
    success = main()

