# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: metrics_calculator.py
Modül: training_management/v2/metrics
Görev: Metrics Calculator - Temel metrik hesaplama (accuracy, perplexity).
       Accuracy ve perplexity hesaplama, padding mask handling işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (temel metrik hesaplama)
- Design Patterns: Calculator Pattern (metrik hesaplama)
- Endüstri Standartları: Metrics calculation best practices

KULLANIM:
- Accuracy hesaplama için
- Perplexity hesaplama için
- Padding mask handling için

BAĞIMLILIKLAR:
- torch: Tensor işlemleri
- math: Matematik işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Optional, Any
import torch
import math


class MetricsCalculator:
    """
    Basic metrics calculator.
    
    Responsibilities:
    - Calculate accuracy
    - Calculate perplexity
    - Handle padding masks
    
    SOLID: Single Responsibility Principle
    """
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize MetricsCalculator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger
    
    def calculate_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pad_id: Optional[int] = None,
    ) -> float:
        """
        Calculate accuracy.
        
        Args:
            logits: Model output logits [B, T, V]
            targets: Target token IDs [B, T]
            pad_id: Padding token ID (None if no padding)
            
        Returns:
            Accuracy as float (0.0 to 1.0)
        """
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            
            if pad_id is None:
                mask = torch.ones_like(targets, dtype=torch.bool)
            else:
                mask = (targets != pad_id)
            
            correct = (preds == targets) & mask
            acc = correct.float().sum().item() / mask.float().sum().clamp_min(1.0).item()
            
            return float(acc)
    
    def calculate_perplexity(
        self,
        loss: float,
        max_exp: float = 20.0
    ) -> float:
        """
        Calculate perplexity from loss.
        
        Args:
            loss: Loss value
            max_exp: Maximum exponent (to prevent overflow)
            
        Returns:
            Perplexity as float
        """
        if not math.isfinite(loss):
            return float("inf")
        
        ppl = math.exp(min(max_exp, loss))
        return float(ppl)

