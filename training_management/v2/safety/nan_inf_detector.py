# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: nan_inf_detector.py
Modül: training_management/v2/safety
Görev: NaN/Inf Detector - NaN ve Inf detection utility. Tensor'larda NaN ve Inf
       değerlerini tespit eder, detection sonuçlarını raporlar. Eğitim sırasında
       sayısal sorunları erken tespit eder.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (NaN/Inf detection)
- Design Patterns: Detector Pattern (sorun tespiti)
- Endüstri Standartları: Numerical stability detection best practices

KULLANIM:
- Tensor'larda NaN değerleri tespit etmek için
- Tensor'larda Inf değerleri tespit etmek için
- Detection sonuçlarını raporlamak için

BAĞIMLILIKLAR:
- torch: Tensor işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from typing import Optional, Any, Dict
import torch


class NaNInfDetector:
    """
    NaN and Inf detection utility.
    
    Responsibilities:
    - Detect NaN values in tensors
    - Detect Inf values in tensors
    - Report detection results
    
    SOLID: Single Responsibility Principle
    """
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize NaNInfDetector.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger
    
    def detect(
        self,
        tensor: torch.Tensor,
        name: str = "tensor"
    ) -> Dict[str, bool]:
        """
        Detect NaN and Inf in tensor.
        
        Args:
            tensor: Tensor to check
            name: Name for logging
            
        Returns:
            Dictionary with detection results:
            {
                "has_nan": bool,
                "has_inf": bool,
                "is_finite": bool
            }
        """
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        is_finite = torch.isfinite(tensor).all().item()
        
        if has_nan or has_inf:
            if self.logger:
                self.logger.log_error(
                    f"NaN/Inf detected in '{name}': "
                    f"has_nan={has_nan}, has_inf={has_inf}"
                )
        
        return {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "is_finite": is_finite
        }
    
    def detect_loss(
        self,
        loss: torch.Tensor
    ) -> bool:
        """
        Detect NaN/Inf in loss tensor.
        
        Args:
            loss: Loss tensor
            
        Returns:
            True if loss is finite, False otherwise
        """
        result = self.detect(loss, "loss")
        return result["is_finite"]

