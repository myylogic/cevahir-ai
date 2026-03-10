"""
KRİTİK TEST: TrainingManager Loss Function Doğrulama
=====================================================

Bu test, TrainingManager'ın loss hesaplamasında EOS weight'in gerçekten uygulanıp uygulanmadığını doğrular.

ENDÜSTRİ STANDARTLARI:
- ISO/IEC/IEEE 29119-3: Unit Test Coverage
- Test: Loss function'ın doğru kullanıldığını doğrula
- Test: EOS weight'in gradient'lere etkisini doğrula
- Test: Training loop'ta criterion kullanılıyor mu?

KRİTİK SORU:
- TrainingService'te EOS weight'li criterion oluşturuluyor
- AMA TrainingManager'da kullanılıyor mu?
- _compute_masked_loss_and_acc'de F.cross_entropy kullanılıyor - criterion kullanılmıyor mu?
"""

import os
import sys
import json
import logging
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training_management.training_manager import TrainingManager
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("TrainingManagerLossFunctionTest")


class TrainingManagerLossFunctionTest:
    """TrainingManager loss function doğrulama testi"""
    
    def __init__(self):
        self.vocab_size = 60000
        self.eos_id = 3
        self.pad_id = 0
        self.bos_id = 2
        self.eos_weight = 0.1
        self.label_smoothing = 0.1
        
    def create_mock_model(self):
        """Mock model oluştur"""
        class MockModel(torch.nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.vocab_size = vocab_size
                self.embedding = torch.nn.Embedding(1000, 512)
                self.linear = torch.nn.Linear(512, vocab_size)
                
            def forward(self, inputs):
                # Basit forward pass - [B, T] → [B, T, V]
                B, T = inputs.shape
                x = self.embedding(inputs)  # [B, T, 512]
                logits = self.linear(x)  # [B, T, V]
                return logits
        
        return MockModel(self.vocab_size)
    
    def create_criterion_with_eos_weight(self):
        """EOS weight'li criterion oluştur (TrainingService gibi)"""
        device = torch.device("cpu")
        loss_weights = torch.ones(self.vocab_size, device=device)
        
        if self.eos_id is not None and 0 <= self.eos_id < self.vocab_size:
            loss_weights[self.eos_id] = self.eos_weight
            logger.info(f"✅ EOS weight uygulandı: eos_id={self.eos_id}, weight={self.eos_weight}")
        
        criterion = CrossEntropyLoss(
            weight=loss_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=self.pad_id,
            reduction="mean"
        )
        
        return criterion
    
    def test_training_manager_loss_computation(self) -> Dict[str, Any]:
        """
        TrainingManager'ın loss hesaplamasını test et
        KRİTİK: _compute_masked_loss_and_acc'de F.cross_entropy kullanılıyor mu?
        Criterion kullanılıyor mu?
        """
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING MANAGER LOSS COMPUTATION TESTİ")
        logger.info("=" * 80)
        
        # Mock model
        model = self.create_mock_model()
        
        # EOS weight'li criterion (TrainingService gibi)
        criterion = self.create_criterion_with_eos_weight()
        
        # TrainingManager config
        config = {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_id,
            "device": "cpu",
            "use_amp": False,
            "grad_accum_steps": 1,
            "max_grad_norm": 1.0,
            "epochs": 1,
            "enable_file_logging": False
        }
        
        # Dummy data
        batch_size = 4
        seq_len = 10
        
        inputs = torch.randint(0, 1000, (batch_size, seq_len))
        targets = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        targets[:, -1] = self.eos_id  # Son pozisyona EOS koy
        
        dataset = TensorDataset(inputs, targets)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # TrainingManager oluştur
        training_manager = TrainingManager(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,  # EOS weight'li criterion veriyoruz
            config=config
        )
        
        # KRİTİK TEST: _compute_masked_loss_and_acc metodunu incele
        # Bu metod F.cross_entropy kullanıyor mu? Criterion kullanıyor mu?
        
        # Model'den logits al
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
        
        # TrainingManager'ın loss hesaplama metodunu çağır
        loss_tm, acc_tm, ppl_tm = training_manager._compute_masked_loss_and_acc(
            logits=logits,
            targets=targets,
            pad_id=self.pad_id
        )
        
        # MANUEL HESAPLAMA: Criterion ile (EOS weight'li)
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = targets.view(-1)
        loss_criterion = criterion(logits_flat, targets_flat)
        
        # MANUEL HESAPLAMA: F.cross_entropy ile (EOS weight YOK)
        loss_f_cross_entropy = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="mean"
        )
        
        # MANUEL HESAPLAMA: F.cross_entropy + weight ile (EOS weight VAR)
        loss_weights = torch.ones(self.vocab_size)
        loss_weights[self.eos_id] = self.eos_weight
        loss_f_cross_entropy_weighted = F.cross_entropy(
            logits_flat,
            targets_flat,
            weight=loss_weights,
            reduction="mean"
        )
        
        # EOS pozisyonlarındaki loss'u analiz et
        eos_positions = (targets_flat == self.eos_id).nonzero(as_tuple=True)[0]
        non_eos_positions = (targets_flat != self.eos_id).nonzero(as_tuple=True)[0]
        
        # EOS pozisyonlarındaki loss (criterion ile)
        if len(eos_positions) > 0:
            eos_loss_criterion = F.cross_entropy(
                logits_flat[eos_positions],
                targets_flat[eos_positions],
                weight=loss_weights,
                reduction="mean"
            )
        else:
            eos_loss_criterion = torch.tensor(0.0)
        
        # Non-EOS pozisyonlarındaki loss
        if len(non_eos_positions) > 0:
            non_eos_loss_criterion = F.cross_entropy(
                logits_flat[non_eos_positions],
                targets_flat[non_eos_positions],
                weight=loss_weights,
                reduction="mean"
            )
        else:
            non_eos_loss_criterion = torch.tensor(0.0)
        
        results = {
            "training_manager_loss": float(loss_tm.item()),
            "criterion_loss": float(loss_criterion.item()),
            "f_cross_entropy_loss": float(loss_f_cross_entropy.item()),
            "f_cross_entropy_weighted_loss": float(loss_f_cross_entropy_weighted.item()),
            "loss_difference_tm_vs_criterion": float(abs(loss_tm.item() - loss_criterion.item())),
            "loss_difference_tm_vs_f_cross_entropy": float(abs(loss_tm.item() - loss_f_cross_entropy.item())),
            "loss_difference_tm_vs_f_cross_entropy_weighted": float(abs(loss_tm.item() - loss_f_cross_entropy_weighted.item())),
            "eos_positions_count": len(eos_positions),
            "non_eos_positions_count": len(non_eos_positions),
            "eos_loss_criterion": float(eos_loss_criterion.item()),
            "non_eos_loss_criterion": float(non_eos_loss_criterion.item()),
            "eos_weight_applied": self.eos_weight,
            "label_smoothing_applied": self.label_smoothing,
            "kritik_bulgu": {
                "training_manager_uses_f_cross_entropy": abs(loss_tm.item() - loss_f_cross_entropy.item()) < 1e-5,
                "training_manager_uses_criterion": abs(loss_tm.item() - loss_criterion.item()) < 1e-5,
                "training_manager_uses_weighted_f_cross_entropy": abs(loss_tm.item() - loss_f_cross_entropy_weighted.item()) < 1e-5,
                "eos_weight_actually_applied": abs(loss_criterion.item() - loss_f_cross_entropy.item()) > 1e-5
            }
        }
        
        logger.info(f"\n📊 Loss Karşılaştırması:")
        logger.info(f"  TrainingManager loss: {results['training_manager_loss']:.6f}")
        logger.info(f"  Criterion loss (EOS weight'li): {results['criterion_loss']:.6f}")
        logger.info(f"  F.cross_entropy loss (EOS weight YOK): {results['f_cross_entropy_loss']:.6f}")
        logger.info(f"  F.cross_entropy weighted loss (EOS weight VAR): {results['f_cross_entropy_weighted_loss']:.6f}")
        logger.info(f"\n📊 Farklar:")
        logger.info(f"  TM vs Criterion: {results['loss_difference_tm_vs_criterion']:.6f}")
        logger.info(f"  TM vs F.cross_entropy: {results['loss_difference_tm_vs_f_cross_entropy']:.6f}")
        logger.info(f"  TM vs F.cross_entropy weighted: {results['loss_difference_tm_vs_f_cross_entropy_weighted']:.6f}")
        
        logger.info(f"\n🔍 KRİTİK BULGU:")
        logger.info(f"  TrainingManager F.cross_entropy kullanıyor mu? {results['kritik_bulgu']['training_manager_uses_f_cross_entropy']}")
        logger.info(f"  TrainingManager Criterion kullanıyor mu? {results['kritik_bulgu']['training_manager_uses_criterion']}")
        logger.info(f"  TrainingManager Weighted F.cross_entropy kullanıyor mu? {results['kritik_bulgu']['training_manager_uses_weighted_f_cross_entropy']}")
        logger.info(f"  EOS weight gerçekten uygulanıyor mu? {results['kritik_bulgu']['eos_weight_actually_applied']}")
        
        return results
    
    def test_gradient_impact(self) -> Dict[str, Any]:
        """
        EOS weight'in gradient'lere etkisini test et
        """
        logger.info("\n" + "=" * 80)
        logger.info("GRADIENT IMPACT TESTİ")
        logger.info("=" * 80)
        
        # Mock model
        model = self.create_mock_model()
        
        # EOS weight'li criterion
        criterion = self.create_criterion_with_eos_weight()
        
        # Dummy data
        batch_size = 4
        seq_len = 10
        inputs = torch.randint(0, 1000, (batch_size, seq_len))
        targets = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        targets[:, -1] = self.eos_id  # Son pozisyona EOS koy
        
        # Model forward
        model.train()
        logits = model(inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        # Loss hesapla (criterion ile - EOS weight'li)
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = targets.view(-1)
        loss = criterion(logits_flat, targets_flat)
        
        # Gradient hesapla
        model.zero_grad()
        loss.backward()
        
        # Gradient'leri topla
        total_grad_norm = 0.0
        param_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                param_grads[name] = grad_norm
        
        total_grad_norm = total_grad_norm ** 0.5
        
        # EOS weight olmadan loss hesapla (karşılaştırma için)
        criterion_no_weight = CrossEntropyLoss(
            reduction="mean",
            ignore_index=self.pad_id
        )
        
        model2 = self.create_mock_model()
        logits2 = model2(inputs)
        if isinstance(logits2, tuple):
            logits2 = logits2[0]
        
        loss2 = criterion_no_weight(logits2.view(-1, self.vocab_size), targets_flat)
        model2.zero_grad()
        loss2.backward()
        
        total_grad_norm2 = 0.0
        for name, param in model2.named_parameters():
            if param.grad is not None:
                total_grad_norm2 += param.grad.norm().item() ** 2
        total_grad_norm2 = total_grad_norm2 ** 0.5
        
        results = {
            "loss_with_eos_weight": float(loss.item()),
            "loss_without_eos_weight": float(loss2.item()),
            "grad_norm_with_eos_weight": float(total_grad_norm),
            "grad_norm_without_eos_weight": float(total_grad_norm2),
            "grad_norm_difference": float(abs(total_grad_norm - total_grad_norm2)),
            "param_grads": param_grads
        }
        
        logger.info(f"\n📊 Gradient Analizi:")
        logger.info(f"  Loss (EOS weight'li): {results['loss_with_eos_weight']:.6f}")
        logger.info(f"  Loss (EOS weight YOK): {results['loss_without_eos_weight']:.6f}")
        logger.info(f"  Grad norm (EOS weight'li): {results['grad_norm_with_eos_weight']:.6f}")
        logger.info(f"  Grad norm (EOS weight YOK): {results['grad_norm_without_eos_weight']:.6f}")
        logger.info(f"  Grad norm farkı: {results['grad_norm_difference']:.6f}")
        
        return results
    
    def run_full_test(self) -> Dict[str, Any]:
        """Tüm testleri çalıştır"""
        logger.info("=" * 80)
        logger.info("TRAINING MANAGER LOSS FUNCTION TESTİ BAŞLATIYOR")
        logger.info("=" * 80)
        
        results = {
            "test_date": datetime.now().isoformat(),
            "loss_computation_test": self.test_training_manager_loss_computation(),
            "gradient_impact_test": self.test_gradient_impact()
        }
        
        # Sonuçları kaydet
        output_file = "test_training_manager_loss_function_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ Test sonuçları kaydedildi: {output_file}")
        logger.info("=" * 80)
        
        return results


if __name__ == "__main__":
    tester = TrainingManagerLossFunctionTest()
    results = tester.run_full_test()
    
    print("\n" + "=" * 80)
    print("ÖZET")
    print("=" * 80)
    
    kritik = results["loss_computation_test"]["kritik_bulgu"]
    print(f"\nKRITIK BULGU:")
    print(f"  TrainingManager F.cross_entropy kullanıyor mu? {kritik['training_manager_uses_f_cross_entropy']}")
    print(f"  TrainingManager Criterion kullanıyor mu? {kritik['training_manager_uses_criterion']}")
    print(f"  EOS weight gerçekten uygulanıyor mu? {kritik['eos_weight_actually_applied']}")
    
    if kritik['training_manager_uses_f_cross_entropy'] and not kritik['training_manager_uses_criterion']:
        print("\nKRITIK HATA TESPIT EDILDI!")
        print("  TrainingManager criterion kullanmiyor, F.cross_entropy kullaniyor!")
        print("  Bu durumda EOS weight UYGULANMIYOR!")
    elif kritik['training_manager_uses_criterion']:
        print("\nTrainingManager criterion kullaniyor - EOS weight uygulaniyor")
    else:
        print("\nBelirsiz durum - daha fazla analiz gerekiyor")

