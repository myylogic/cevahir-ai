# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: warmup_calculator.py
Modül: training_system/v2/utils
Görev: Warmup Steps Calculator - Warmup steps'i dinamik olarak hesaplar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (warmup steps hesaplama)
- Endüstri Standartları: PyTorch warmup best practices

KULLANIM:
- TrainingService.train() içinde, train_loader hazır olduktan sonra çağrılır

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Dict, Any
from torch.utils.data import DataLoader


def calculate_warmup_steps(
    train_loader: DataLoader,
    config: Dict[str, Any]
) -> int:
    """
    Warmup steps'i dinamik olarak hesapla.
    
    Endüstri Standardı:
    - Warmup = İlk 1-2 epoch (önerilen)
    - Veya: Toplam step'lerin %5-10'u
    
    Args:
        train_loader: Training data loader
        config: Configuration dictionary
        
    Returns:
        Calculated warmup steps
    """
    num_batches_per_epoch = len(train_loader)
    grad_accum_steps = config.get("grad_accum_steps", 1)
    num_epochs = config.get("epochs", 50)
    
    # Steps per epoch (gradient accumulation dikkate alınır)
    steps_per_epoch = int(num_batches_per_epoch / grad_accum_steps)
    
    # Seçenek 1: İlk N epoch warmup (Önerilen - Endüstri Standardı)
    warmup_epochs = config.get("warmup_epochs", 1)
    warmup_steps = steps_per_epoch * warmup_epochs
    
    # Seçenek 2: Toplam step'lerin yüzdesi (Alternatif)
    # total_steps = steps_per_epoch * num_epochs
    # warmup_percentage = config.get("warmup_percentage", 0.1)  # %10
    # warmup_steps = int(total_steps * warmup_percentage)
    
    return max(1, warmup_steps)  # En az 1 step

