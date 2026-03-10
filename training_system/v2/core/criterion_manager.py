# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: criterion_manager.py
Modül: training_system/v2/core
Görev: Criterion Manager - Loss Function Yönetimi. EOS weight (varsayılan 1.0) ve label smoothing
       0.1 ile CrossEntropyLoss oluşturma. Endüstri standardı: GPT-4, LLaMA, Claude
       yaklaşımı. Loss function oluşturma ve yönetimi.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (loss function yönetimi)
- Design Patterns: Factory Pattern (loss function oluşturma)
- Endüstri Standartları: GPT-4/LLaMA/Claude loss function yaklaşımı

KULLANIM:
- Loss function oluşturmak için
- EOS weight ve label smoothing yönetimi için
- Criterion yönetimi için

BAĞIMLILIKLAR:
- torch.nn: Loss function modülleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Any


class CriterionManager:
    """Loss function (criterion) oluşturan ve yöneten manager"""
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Args:
            logger: Logger instance (opsiyonel)
        """
        self.logger = logger
    
    def create_criterion(
        self,
        vocab_size: int,
        eos_id: Optional[int],
        pad_id: int = 0,
        device: Optional[torch.device] = None,
        label_smoothing: float = 0.1,
        eos_weight: float = 1.0
    ) -> nn.CrossEntropyLoss:
        """
        EOS weight ve label smoothing ile CrossEntropyLoss oluştur.
        
        KRITIK: eos_weight=0.1 kullanıldığında EOS için gradient ~10x zayıf oluyor,
        model EOS öğrenemiyor (EOS prob 0 kalıyor). Bu yüzden varsayılan 1.0.
        
        Args:
            vocab_size: Vocab boyutu
            eos_id: EOS token ID (None ise weight uygulanmaz)
            pad_id: PAD token ID (ignore_index için)
            device: Cihaz (GPU/CPU)
            label_smoothing: Label smoothing değeri (default: 0.1)
            eos_weight: EOS token weight (default: 1.0; 0.1 EOS öğrenimini bozuyor)
            
        Returns:
            CrossEntropyLoss instance (EOS weight ve label smoothing ile)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Loss weight tensor oluştur (varsayılan: 1.0)
        loss_weights = torch.ones(vocab_size, device=device)
        
        # EOS weight uygula
        if eos_id is not None and 0 <= eos_id < vocab_size:
            loss_weights[eos_id] = eos_weight
            if self.logger:
                self.logger.info(
                    f"[OK] EOS Token Weight uygulandi: eos_id={eos_id}, weight={eos_weight}"
                )
        
        # CrossEntropyLoss oluştur
        criterion = nn.CrossEntropyLoss(
            weight=loss_weights,
            label_smoothing=label_smoothing,
            ignore_index=pad_id,
            reduction="mean"
        )
        
        if self.logger:
            self.logger.info(
                f"[OK] Loss function guncellendi: "
                f"Label Smoothing={label_smoothing}, EOS Weight={eos_weight}"
            )
        
        return criterion

